"""Supervised imitation learning on lb-1200 self-play trajectories.

Pure BC: cross-entropy on (target_class, bucket_class). No advantage weighting,
because lb-1200 is strong enough that every action is ~correct — AWAC would
just add noise.

Usage:
  python training/imitation_learn.py \
      --data-dir offline/lb1200_selfplay/2026-04-20 \
      --out training/checkpoints/imitation_v1.pt \
      --epochs 100 --batch 128 --lr 3e-4
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


class ILStepDataset(torch.utils.data.Dataset):
    def __init__(self, shards: list[pathlib.Path],
                 max_planets: int = 64, max_fleets: int = 64):
        self.shards = shards
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.index: list[tuple[int, int]] = []
        for si, p in enumerate(shards):
            d = np.load(p, allow_pickle=True)
            T = int(d["n_steps"])
            src_arr = d["src_planet_idx"]
            for t in range(T):
                if len(src_arr[t]) > 0:
                    self.index.append((si, t))
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
        si, t = self.index[i]
        d = self._load(si)
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
        return {
            "planets": planets.astype(np.float32),
            "planet_xy": planet_xy.astype(np.float32),
            "fleets": fleets.astype(np.float32) if len(fleets)
                      else np.zeros((0, 9), dtype=np.float32),
            "globals": globals_.astype(np.float32),
            "owned_mask": owned.astype(bool),
            "src": src, "tgt": tgt, "bkt": bkt,
        }


def collate(batch):
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
        for s, t, k in zip(b["src"], b["tgt"], b["bkt"]):
            flat_b.append(i); flat_src.append(int(s))
            flat_tgt.append(int(t)); flat_bkt.append(int(k))

    return {
        "planets": torch.from_numpy(pl),
        "planet_mask": torch.from_numpy(pmask),
        "fleets": torch.from_numpy(fl),
        "fleet_mask": torch.from_numpy(fmask),
        "globals": torch.from_numpy(gl),
        "owned_mask": torch.from_numpy(owned_mask),
        "flat_batch": torch.tensor(flat_b, dtype=torch.long),
        "flat_src": torch.tensor(flat_src, dtype=torch.long),
        "flat_tgt": torch.tensor(flat_tgt, dtype=torch.long),
        "flat_bkt": torch.tensor(flat_bkt, dtype=torch.long),
    }


def train(args):
    data_dir = pathlib.Path(args.data_dir)
    shards = sorted(set(data_dir.glob("*.npz")) | set(data_dir.glob("*/*.npz")))
    if not shards:
        print(f"no shards in {data_dir}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    ds = ILStepDataset(shards, max_planets=args.max_planets,
                       max_fleets=args.max_fleets)
    print(f"device={device}  shards={len(shards)}  samples={len(ds)}",
          flush=True)

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        drop_last=True,
    )

    from featurize import PLANET_DIM, FLEET_DIM, GLOBAL_DIM
    kwargs = dict(planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
                  d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, n_buckets=4)
    model = OrbitAgent(**kwargs).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"warm-started from {args.init_from}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: d={args.d_model} L={args.n_layers} H={args.n_heads}  "
          f"params={n_params:.2f}M  epochs={args.epochs}",
          flush=True)

    t0 = time.time()
    best_acc = 0.0
    for ep in range(args.epochs):
        tot = {"ce_tgt": 0, "ce_bkt": 0, "acc_tgt": 0, "acc_bkt": 0, "n": 0}
        for batch in loader:
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            tgt_logits, bkt_logits, _ = model(
                batch["planets"], batch["planet_mask"],
                batch["fleets"], batch["fleet_mask"],
                batch["globals"], target_mask=None,
            )
            fb = batch["flat_batch"]
            fs = batch["flat_src"]
            picked_tgt = tgt_logits[fb, fs]
            picked_bkt = bkt_logits[fb, fs]
            tgt_labels = batch["flat_tgt"] + 1  # class 0 = pass

            ce_tgt = F.cross_entropy(picked_tgt, tgt_labels)
            ce_bkt = F.cross_entropy(picked_bkt, batch["flat_bkt"])
            loss = ce_tgt + 0.5 * ce_bkt

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                tot["ce_tgt"] += ce_tgt.item()
                tot["ce_bkt"] += ce_bkt.item()
                tot["acc_tgt"] += (picked_tgt.argmax(-1) == tgt_labels).float().mean().item()
                tot["acc_bkt"] += (picked_bkt.argmax(-1) == batch["flat_bkt"]).float().mean().item()
                tot["n"] += 1

        sched.step()
        n = max(tot["n"], 1)
        dt = time.time() - t0
        acc_tgt = tot["acc_tgt"] / n
        acc_bkt = tot["acc_bkt"] / n
        print(f"ep {ep+1:03d}/{args.epochs}  "
              f"ce_tgt={tot['ce_tgt']/n:.4f}  ce_bkt={tot['ce_bkt']/n:.4f}  "
              f"tgt_acc={acc_tgt:.3f}  bkt_acc={acc_bkt:.3f}  "
              f"lr={sched.get_last_lr()[0]:.1e}  [{dt:.0f}s]",
              flush=True)

        # Save best + every 10 epochs
        if acc_tgt > best_acc or (ep + 1) % 10 == 0:
            best_acc = max(best_acc, acc_tgt)
            torch.save({
                "model": model.state_dict(), "kwargs": kwargs,
                "iter": ep + 1, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, args.out)

    print(f"done. final tgt_acc={acc_tgt:.3f} bkt_acc={acc_bkt:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--max-planets", type=int, default=64)
    ap.add_argument("--max-fleets", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
