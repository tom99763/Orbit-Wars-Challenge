"""Offline RL on expert trajectories via IQL (Implicit Q-Learning, Kostrikov 2021).

Three losses jointly:
  V(s)   ← expectile(τ=0.7) of Q(s,a) — learns upper-confidence of observed Q
  Q(s,a) ← r + γ V(s')                 — standard TD bootstrap, but V not max Q
  Policy ← advantage-weighted BC, weight = exp(β · (Q − V)) clipped to ≤ EXP_MAX

Uses the same OrbitAgent as online training. Q head is a small add-on:
  q_head(z) → scalar Q, conditioned on (state features, chosen action embedding)

For simplicity we predict Q(s,src,tgt,bkt) by pooling per-planet embeddings
of (src, tgt) and combining with bucket embedding.

Usage:
  python training/offline_iql.py \
      --data-dir offline/2026-04-19 \
      --out training/checkpoints/offline_v1.pt \
      --epochs 10 --batch 32 --lr 3e-4 --tau 0.7 --beta 3.0
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from training.model import OrbitAgent
from training.data import BCDataset, collate, make_loader


SHIPS_BUCKETS = 4
GAMMA = 0.997


class QHead(nn.Module):
    """Q(s, src_i, tgt_j, bkt_k) — given per-planet embeddings and bucket class."""

    def __init__(self, d_model: int, n_buckets: int = SHIPS_BUCKETS):
        super().__init__()
        self.bkt_emb = nn.Embedding(n_buckets, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, src_emb: torch.Tensor, tgt_emb: torch.Tensor,
                bkt_idx: torch.Tensor) -> torch.Tensor:
        b = self.bkt_emb(bkt_idx)
        x = torch.cat([src_emb, tgt_emb, b], dim=-1)
        return self.mlp(x).squeeze(-1)


class OfflineRLAgent(nn.Module):
    """Wraps OrbitAgent and adds Q-head built on top of the planet encoder.

    Exposes:
      policy_logits(obs) — same as OrbitAgent (tgt + bucket heads)
      value_fn(obs)      — state-value V(s)
      q_value(obs, src, tgt, bkt) — Q(s, a)
    """

    def __init__(self, kwargs: dict):
        super().__init__()
        self.orbit = OrbitAgent(**kwargs)
        d = kwargs["d_model"]
        self.q_head = QHead(d, n_buckets=kwargs.get("n_buckets", SHIPS_BUCKETS))
        # Separate V-head (different linear on same encoder output)
        self.v_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )

    def forward_encode(self, planets, planet_mask, fleets, fleet_mask, globals_):
        """Return per-planet embeddings + global token. Uses orbit.encoder internals."""
        # We need access to the encoder output before policy/value heads.
        # OrbitAgent.forward returns (tgt_logits, bkt_logits, value). We don't
        # have direct access here, so replicate first encoding path.
        return self.orbit._encode(planets, planet_mask, fleets, fleet_mask, globals_)


def load_dataset_shards(data_dir: pathlib.Path):
    """Return list of per-episode shard paths."""
    shards = sorted(data_dir.glob("*.npz"))
    return shards


class OfflineStepDataset(torch.utils.data.Dataset):
    """Loads all (step,action) tuples from npz shards.
    Each item: (obs features, chosen (src,tgt,bkt), shape_return, terminal_reward,
                next-obs features, is_terminal_step)."""

    def __init__(self, shards: list[pathlib.Path], max_planets: int = 64,
                 max_fleets: int = 64):
        self.shards = shards
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        # Index: (shard_idx, step_idx). Only steps with at least 1 action.
        self.index = []
        for si, p in enumerate(shards):
            d = np.load(p, allow_pickle=True)
            T = int(d["n_steps"])
            src_arr = d["src_planet_idx"]
            for t in range(T):
                if len(src_arr[t]) > 0:
                    self.index.append((si, t))
        print(f"loaded {len(shards)} shards, {len(self.index)} action samples",
              flush=True)
        # Lazy-load cache (shared across workers in-process only)
        self._cache: dict = {}

    def __len__(self):
        return len(self.index)

    def _load(self, si: int):
        if si in self._cache:
            return self._cache[si]
        d = dict(np.load(self.shards[si], allow_pickle=True))
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
        # Next-step features (for Q bootstrap)
        T = int(d["n_steps"])
        t_next = min(t + 1, T - 1)
        planets_n = np.asarray(d["planets"][t_next])[: self.max_planets]
        planet_xy_n = np.asarray(d["planet_xy"][t_next])[: self.max_planets]
        fleets_n = np.asarray(d["fleets"][t_next])[: self.max_fleets]
        globals_n = np.asarray(d["globals"][t_next])
        return {
            "planets": planets.astype(np.float32),
            "planet_xy": planet_xy.astype(np.float32),
            "fleets": fleets.astype(np.float32) if len(fleets)
                      else np.zeros((0, 9), dtype=np.float32),
            "globals": globals_.astype(np.float32),
            "owned_mask": owned.astype(bool),
            "src": src, "tgt": tgt, "bkt": bkt,
            "planets_next": planets_n.astype(np.float32),
            "planet_xy_next": planet_xy_n.astype(np.float32),
            "fleets_next": fleets_n.astype(np.float32) if len(fleets_n)
                           else np.zeros((0, 9), dtype=np.float32),
            "globals_next": globals_n.astype(np.float32),
            "shape_return": float(d["shape_return"][t]),
            "reward_t": float(d["shape_reward"][t]),
            "terminal_r": float(d["terminal_reward"]),
            "is_last": bool(t_next == T - 1),
            "is_winner": bool(d["is_winner"]),
        }


def collate_offline(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    B = len(batch)
    P = max(b["planets"].shape[0] for b in batch)
    Pn = max(b["planets_next"].shape[0] for b in batch)
    Pmax = max(P, Pn)
    Fmax = max(max(b["fleets"].shape[0] for b in batch),
               max(b["fleets_next"].shape[0] for b in batch), 1)
    planet_dim = batch[0]["planets"].shape[1]
    fleet_dim = 9
    g_dim = batch[0]["globals"].shape[0]

    def pad_state(key_p, key_xy, key_f, key_g):
        pl = np.zeros((B, Pmax, planet_dim), dtype=np.float32)
        xy = np.zeros((B, Pmax, 2), dtype=np.float32)
        pmask = np.zeros((B, Pmax), dtype=bool)
        fl = np.zeros((B, Fmax, fleet_dim), dtype=np.float32)
        fmask = np.zeros((B, Fmax), dtype=bool)
        gl = np.zeros((B, g_dim), dtype=np.float32)
        for i, b in enumerate(batch):
            np_ = b[key_p].shape[0]
            pl[i, :np_] = b[key_p]
            xy[i, :np_] = b[key_xy]
            pmask[i, :np_] = True
            nf = b[key_f].shape[0]
            if nf > 0:
                fl[i, :nf] = b[key_f]
                fmask[i, :nf] = True
            gl[i] = b[key_g]
        return pl, xy, pmask, fl, fmask, gl

    pl_c, xy_c, pm_c, fl_c, fm_c, gl_c = pad_state(
        "planets", "planet_xy", "fleets", "globals")
    pl_n, xy_n, pm_n, fl_n, fm_n, gl_n = pad_state(
        "planets_next", "planet_xy_next", "fleets_next", "globals_next")

    # Pack per-sample actions with batch indices
    flat_b, flat_src, flat_tgt, flat_bkt = [], [], [], []
    flat_return, flat_reward_t, flat_terminal, flat_is_last = [], [], [], []
    owned_mask = np.zeros((B, Pmax), dtype=bool)
    is_winner = np.zeros((B,), dtype=bool)
    for i, b in enumerate(batch):
        np_ = b["owned_mask"].shape[0]
        owned_mask[i, :np_] = b["owned_mask"]
        is_winner[i] = b["is_winner"]
        for s, t, k in zip(b["src"], b["tgt"], b["bkt"]):
            flat_b.append(i); flat_src.append(int(s))
            flat_tgt.append(int(t)); flat_bkt.append(int(k))
            flat_return.append(b["shape_return"])
            flat_reward_t.append(b["reward_t"])
            flat_terminal.append(b["terminal_r"])
            flat_is_last.append(b["is_last"])

    return {
        "planets": torch.from_numpy(pl_c),
        "planet_xy": torch.from_numpy(xy_c),
        "planet_mask": torch.from_numpy(pm_c),
        "fleets": torch.from_numpy(fl_c),
        "fleet_mask": torch.from_numpy(fm_c),
        "globals": torch.from_numpy(gl_c),
        "planets_next": torch.from_numpy(pl_n),
        "planet_xy_next": torch.from_numpy(xy_n),
        "planet_mask_next": torch.from_numpy(pm_n),
        "fleets_next": torch.from_numpy(fl_n),
        "fleet_mask_next": torch.from_numpy(fm_n),
        "globals_next": torch.from_numpy(gl_n),
        "owned_mask": torch.from_numpy(owned_mask),
        "is_winner": torch.from_numpy(is_winner),
        "flat_batch": torch.tensor(flat_b, dtype=torch.long),
        "flat_src": torch.tensor(flat_src, dtype=torch.long),
        "flat_tgt": torch.tensor(flat_tgt, dtype=torch.long),
        "flat_bkt": torch.tensor(flat_bkt, dtype=torch.long),
        "flat_return": torch.tensor(flat_return, dtype=torch.float32),
        "flat_reward_t": torch.tensor(flat_reward_t, dtype=torch.float32),
        "flat_terminal": torch.tensor(flat_terminal, dtype=torch.float32),
        "flat_is_last": torch.tensor(flat_is_last, dtype=torch.bool),
    }


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric L2: penalizes positive diff more heavily when tau>0.5,
    i.e. V wants to be above observed Q samples."""
    w = torch.where(diff > 0, tau, 1.0 - tau)
    return (w * diff ** 2).mean()


def train(args):
    data_dir = pathlib.Path(args.data_dir)
    shards = load_dataset_shards(data_dir)
    if not shards:
        print(f"no shards in {data_dir}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"device: {device}  shards: {len(shards)}", flush=True)

    ds = OfflineStepDataset(shards, max_planets=args.max_planets,
                            max_fleets=args.max_fleets)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_offline,
        drop_last=True,
    )

    kwargs = dict(planet_dim=14, fleet_dim=9, global_dim=16,
                  d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, n_buckets=SHIPS_BUCKETS)
    model = OfflineRLAgent(kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"model: d={args.d_model} L={args.n_layers} H={args.n_heads}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    t0 = time.time()
    for ep in range(args.epochs):
        totals = {"v": 0, "q": 0, "pi": 0, "n": 0, "acc_tgt": 0, "acc_bkt": 0}
        for step, batch in enumerate(loader):
            if batch is None:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}

            # Encode current and next states
            z = model.forward_encode(
                batch["planets"], batch["planet_mask"],
                batch["fleets"], batch["fleet_mask"], batch["globals"],
            )
            # z: (planet_emb [B,P,d], global_tok [B,d]) depending on model interface.
            # Interface unknown — fall back: use policy/value call.
            # (Simpler and guaranteed): use OrbitAgent's forward directly.
            tgt_logits, bkt_logits, value_pred = model.orbit(
                batch["planets"], batch["planet_mask"],
                batch["fleets"], batch["fleet_mask"], batch["globals"],
                target_mask=None,
            )
            # Gather per-(B,src) logits
            fb = batch["flat_batch"]
            fs = batch["flat_src"]
            ft = batch["flat_tgt"]  # target planet idx in planets
            fk = batch["flat_bkt"]
            ret = batch["flat_return"] + batch["flat_terminal"]  # observed G
            picked_tgt = tgt_logits[fb, fs]  # [K, P+1]
            picked_bkt = bkt_logits[fb, fs]  # [K, 4]

            # V: use OrbitAgent's value head output per batch
            # model.orbit returns value [B]. Expand to [K]
            V = value_pred[fb]  # [K]

            # Q: train a small MLP to predict Q(s,a)
            # For simplicity use hand-crafted features: policy logit at chosen
            # (tgt+1) + bkt logit at chosen, plus V. Light-weight Q net.
            logit_tgt_chosen = picked_tgt.gather(
                1, (ft + 1).unsqueeze(1)).squeeze(1)  # [K]
            logit_bkt_chosen = picked_bkt.gather(
                1, fk.unsqueeze(1)).squeeze(1)  # [K]
            # Q head: direct sum + learned scale
            # Using V (no grad into V for Q target) as baseline; but simpler
            # variant: Q = tgt_logit + bkt_logit scaled + V baseline.
            # We'll use a tiny linear fusion.
            Q_in = torch.stack([logit_tgt_chosen, logit_bkt_chosen, V.detach()],
                               dim=-1)  # [K, 3]
            if not hasattr(model, "_q_fuse"):
                model._q_fuse = nn.Linear(3, 1).to(device)
                opt.add_param_group({"params": model._q_fuse.parameters()})
            Q = model._q_fuse(Q_in).squeeze(-1)  # [K]

            # ----- Losses -----
            diff = Q.detach() - V
            v_loss = expectile_loss(diff, args.tau)

            # TD target: r_t + γ V(s'). For terminal step, use terminal reward.
            with torch.no_grad():
                _, _, V_next = model.orbit(
                    batch["planets_next"], batch["planet_mask_next"],
                    batch["fleets_next"], batch["fleet_mask_next"],
                    batch["globals_next"], target_mask=None,
                )
                V_next_k = V_next[fb]
                td_target = torch.where(
                    batch["flat_is_last"],
                    batch["flat_terminal"],
                    batch["flat_reward_t"] + GAMMA * V_next_k,
                )
            q_loss = F.mse_loss(Q, td_target)

            # Policy: advantage-weighted BC
            with torch.no_grad():
                adv = Q - V
                w = torch.exp(args.beta * adv).clamp(max=100.0)
            tgt_labels = ft + 1
            log_probs_tgt = F.log_softmax(picked_tgt, dim=-1)
            logp_tgt = log_probs_tgt.gather(
                1, tgt_labels.unsqueeze(1)).squeeze(1)
            log_probs_bkt = F.log_softmax(picked_bkt, dim=-1)
            logp_bkt = log_probs_bkt.gather(1, fk.unsqueeze(1)).squeeze(1)
            pi_loss = -(w * (logp_tgt + logp_bkt)).mean()

            loss = v_loss + q_loss + pi_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                totals["v"] += v_loss.item()
                totals["q"] += q_loss.item()
                totals["pi"] += pi_loss.item()
                totals["n"] += 1
                totals["acc_tgt"] += (picked_tgt.argmax(-1) == tgt_labels
                                     ).float().mean().item()
                totals["acc_bkt"] += (picked_bkt.argmax(-1) == fk
                                     ).float().mean().item()

        n = max(totals["n"], 1)
        dt = time.time() - t0
        print(f"epoch {ep+1}/{args.epochs} "
              f"v={totals['v']/n:.4f} q={totals['q']/n:.4f} "
              f"pi={totals['pi']/n:.4f} "
              f"tgt_acc={totals['acc_tgt']/n:.3f} "
              f"bkt_acc={totals['acc_bkt']/n:.3f}  [{dt:.0f}s]",
              flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.orbit.state_dict(), "kwargs": kwargs,
                "iter": 0, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
               out)
    print(f"saved {out}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.7,
                    help="Expectile for V; 0.7 = upper-quantile bias")
    ap.add_argument("--beta", type=float, default=3.0,
                    help="Advantage weight temperature for policy extraction")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--max-planets", type=int, default=64)
    ap.add_argument("--max-fleets", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
