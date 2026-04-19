"""Offline RL pretrain on expert trajectories via AWAC-style advantage-weighted BC.

Losses:
  V(s)   ← MSE against observed MC return G_t (discounted shaped + terminal)
  Policy ← weighted BC, weight = exp(β · (G_t − V(s))) clipped to ≤ W_MAX

Clean pipeline: no separate Q head, advantage comes directly from MC returns
minus the learned value baseline. Sample-weighted to emphasize winners.

Usage:
  python training/offline_awac.py \
      --data-dir offline/2026-04-19 \
      --out training/checkpoints/offline_v1.pt \
      --epochs 15 --batch 64 --lr 3e-4 --beta 3.0
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from training.model import OrbitAgent


GAMMA = 0.997
W_MAX = 20.0


class OfflineStepDataset(torch.utils.data.Dataset):
    """Flat index of (shard_idx, step_idx) across all npz shards.

    Only step indices with at least one action are kept. Each sample returns
    features for state s_t, next state s_{t+1}, action (src, tgt, bkt), and
    the MC shape_return + terminal_reward.
    """

    def __init__(self, shards: list[pathlib.Path],
                 max_planets: int = 64, max_fleets: int = 64,
                 winners_only: bool = False):
        self.shards = shards
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.index: list[tuple[int, int, bool]] = []
        for si, p in enumerate(shards):
            d = np.load(p, allow_pickle=True)
            is_winner = bool(d["is_winner"])
            if winners_only and not is_winner:
                continue
            T = int(d["n_steps"])
            src_arr = d["src_planet_idx"]
            for t in range(T):
                if len(src_arr[t]) > 0:
                    self.index.append((si, t, is_winner))
        self._cache: dict = {}

    def __len__(self):
        return len(self.index)

    def _load(self, si: int):
        d = self._cache.get(si)
        if d is not None:
            return d
        d = dict(np.load(self.shards[si], allow_pickle=True))
        if len(self._cache) > 128:
            self._cache.clear()
        self._cache[si] = d
        return d

    def __getitem__(self, i: int):
        si, t, is_winner = self.index[i]
        d = self._load(si)
        T = int(d["n_steps"])
        planets = np.asarray(d["planets"][t])[: self.max_planets]
        planet_xy = np.asarray(d["planet_xy"][t])[: self.max_planets]
        fleets = np.asarray(d["fleets"][t])[: self.max_fleets]
        globals_ = np.asarray(d["globals"][t])
        owned = np.asarray(d["action_mask_owned"][t])[: self.max_planets]
        src = np.asarray(d["src_planet_idx"][t])
        tgt = np.asarray(d["target_planet_idx"][t])
        bkt = np.asarray(d["ships_bucket"][t])
        valid = (src < len(planets)) & (tgt < len(planets))
        src = src[valid].astype(np.int64)
        tgt = tgt[valid].astype(np.int64)
        bkt = bkt[valid].astype(np.int64)
        if len(src) == 0:
            return None
        shape_return = float(d["shape_return"][t])
        terminal_r = float(d["terminal_reward"])
        G = shape_return + terminal_r  # combined MC return
        return {
            "planets": planets.astype(np.float32),
            "planet_xy": planet_xy.astype(np.float32),
            "fleets": fleets.astype(np.float32) if len(fleets)
                      else np.zeros((0, 9), dtype=np.float32),
            "globals": globals_.astype(np.float32),
            "owned_mask": owned.astype(bool),
            "src": src, "tgt": tgt, "bkt": bkt,
            "G": np.float32(G),
            "is_winner": bool(is_winner),
        }


def collate_offline(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    B = len(batch)
    P = max(b["planets"].shape[0] for b in batch)
    Fmax = max(max(b["fleets"].shape[0] for b in batch), 1)
    planet_dim = batch[0]["planets"].shape[1]
    g_dim = batch[0]["globals"].shape[0]

    pl = np.zeros((B, P, planet_dim), dtype=np.float32)
    xy = np.zeros((B, P, 2), dtype=np.float32)
    pmask = np.zeros((B, P), dtype=bool)
    fl = np.zeros((B, Fmax, 9), dtype=np.float32)
    fmask = np.zeros((B, Fmax), dtype=bool)
    gl = np.zeros((B, g_dim), dtype=np.float32)
    owned_mask = np.zeros((B, P), dtype=bool)
    G_b = np.zeros((B,), dtype=np.float32)
    is_winner = np.zeros((B,), dtype=bool)

    flat_b, flat_src, flat_tgt, flat_bkt = [], [], [], []
    for i, b in enumerate(batch):
        np_ = b["planets"].shape[0]
        pl[i, :np_] = b["planets"]
        xy[i, :np_] = b["planet_xy"]
        pmask[i, :np_] = True
        owned_mask[i, :np_] = b["owned_mask"]
        nf = b["fleets"].shape[0]
        if nf > 0:
            fl[i, :nf] = b["fleets"]
            fmask[i, :nf] = True
        gl[i] = b["globals"]
        G_b[i] = b["G"]
        is_winner[i] = b["is_winner"]
        for s, t, k in zip(b["src"], b["tgt"], b["bkt"]):
            flat_b.append(i); flat_src.append(int(s))
            flat_tgt.append(int(t)); flat_bkt.append(int(k))

    return {
        "planets": torch.from_numpy(pl),
        "planet_xy": torch.from_numpy(xy),
        "planet_mask": torch.from_numpy(pmask),
        "fleets": torch.from_numpy(fl),
        "fleet_mask": torch.from_numpy(fmask),
        "globals": torch.from_numpy(gl),
        "owned_mask": torch.from_numpy(owned_mask),
        "G": torch.from_numpy(G_b),
        "is_winner": torch.from_numpy(is_winner),
        "flat_batch": torch.tensor(flat_b, dtype=torch.long),
        "flat_src": torch.tensor(flat_src, dtype=torch.long),
        "flat_tgt": torch.tensor(flat_tgt, dtype=torch.long),
        "flat_bkt": torch.tensor(flat_bkt, dtype=torch.long),
    }


def train(args):
    data_dir = pathlib.Path(args.data_dir)
    # Accept both flat layout (offline/*.npz) and nested (offline/<date>/*.npz)
    shards = sorted(set(data_dir.glob("*.npz")) | set(data_dir.glob("*/*.npz")))
    if not shards:
        print(f"no shards in {data_dir}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    ds = OfflineStepDataset(shards, max_planets=args.max_planets,
                            max_fleets=args.max_fleets,
                            winners_only=args.winners_only)
    print(f"device: {device}  shards: {len(shards)}  "
          f"samples: {len(ds)}  winners_only: {args.winners_only}",
          flush=True)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_offline,
        drop_last=True,
    )

    kwargs = dict(planet_dim=14, fleet_dim=9, global_dim=16,
                  d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, n_buckets=4)
    model = OrbitAgent(**kwargs).to(device)

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        if "kwargs" in ckpt and ckpt["kwargs"] != kwargs:
            print(f"warn: init-from kwargs differ — {ckpt['kwargs']} vs {kwargs}",
                  flush=True)
        model.load_state_dict(ckpt["model"])
        print(f"warm-started from {args.init_from}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-5)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: d={args.d_model} L={args.n_layers} H={args.n_heads}  "
          f"params={n_params:.2f}M",
          flush=True)

    t0 = time.time()
    for ep in range(args.epochs):
        tot = {"v": 0, "pi": 0, "adv_mean": 0, "adv_std": 0,
               "w_mean": 0, "acc_tgt": 0, "acc_bkt": 0, "n": 0}
        for step, batch in enumerate(loader):
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}

            tgt_logits, bkt_logits, value_pred = model(
                batch["planets"], batch["planet_mask"],
                batch["fleets"], batch["fleet_mask"], batch["globals"],
                target_mask=None,
            )
            fb = batch["flat_batch"]
            fs = batch["flat_src"]
            ft = batch["flat_tgt"]
            fk = batch["flat_bkt"]

            # V loss: regress to combined MC return G
            v_loss = F.mse_loss(value_pred, batch["G"])

            # Advantage weighting: A = G − V(s), per-batch, broadcast to actions
            with torch.no_grad():
                adv_per_state = batch["G"] - value_pred.detach()
                # Normalise across batch for stable beta
                if adv_per_state.numel() > 1:
                    adv_per_state = (adv_per_state - adv_per_state.mean()) / \
                                    adv_per_state.std().clamp(min=1e-6)
                adv_per_sample = adv_per_state[fb]
                w = torch.exp(args.beta * adv_per_sample).clamp(max=W_MAX)

            # Policy loss: weighted BC on (target, bucket)
            picked_tgt = tgt_logits[fb, fs]
            picked_bkt = bkt_logits[fb, fs]
            tgt_labels = ft + 1  # class 0 == pass
            logp_tgt = F.log_softmax(picked_tgt, dim=-1).gather(
                1, tgt_labels.unsqueeze(1)).squeeze(1)
            logp_bkt = F.log_softmax(picked_bkt, dim=-1).gather(
                1, fk.unsqueeze(1)).squeeze(1)
            pi_loss = -(w * (logp_tgt + logp_bkt)).mean()

            loss = args.value_coef * v_loss + pi_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                tot["v"] += v_loss.item()
                tot["pi"] += pi_loss.item()
                tot["adv_mean"] += float(adv_per_sample.mean())
                tot["adv_std"] += float(adv_per_sample.std())
                tot["w_mean"] += float(w.mean())
                tot["acc_tgt"] += (picked_tgt.argmax(-1) == tgt_labels
                                   ).float().mean().item()
                tot["acc_bkt"] += (picked_bkt.argmax(-1) == fk
                                   ).float().mean().item()
                tot["n"] += 1

        n = max(tot["n"], 1)
        dt = time.time() - t0
        print(f"ep {ep+1:02d}/{args.epochs}  "
              f"v={tot['v']/n:.4f}  pi={tot['pi']/n:.4f}  "
              f"adv={tot['adv_mean']/n:+.3f}±{tot['adv_std']/n:.3f}  "
              f"w={tot['w_mean']/n:.2f}  "
              f"tgt_acc={tot['acc_tgt']/n:.3f}  "
              f"bkt_acc={tot['acc_bkt']/n:.3f}  [{dt:.0f}s]",
              flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "kwargs": kwargs,
        "iter": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, out)
    print(f"saved {out}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--max-planets", type=int, default=64)
    ap.add_argument("--max-fleets", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--winners-only", action="store_true",
                    help="Train only on winning agent trajectories")
    ap.add_argument("--init-from", default=None,
                    help="Warm-start from existing checkpoint (for continual updates)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
