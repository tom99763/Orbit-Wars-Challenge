"""Train V_φ reference value function for PBRS.

Reads offline/ npz shards (from training/build_offline_dataset.py), which include
per-step shape_return (MC-discounted sum of shaped rewards). Fits a value head
using the SAME state encoder architecture as OrbitAgent so the input schema
matches SAC's actor at runtime.

V_φ serves as the potential function for Potential-Based Reward Shaping (PBRS):
  r'_t = r_t + γ · V_φ(s_{t+1}) - V_φ(s_t)
Theoretical property (Ng et al. 1999): optimal policy is unchanged under this
transformation. So mis-fit V_φ only hurts sample efficiency, not optimality.

Prereq: after featurize.py changes (K=3 sliding window → PLANET=35/...), the
offline/ shards must be REBUILT by re-running build_offline_dataset.py so the
feature dims match. Otherwise loading will fail.

Usage:
  python training/train_v_phi.py \\
      --offline-dir offline/2026-04-19 \\
      --out training/checkpoints/v_phi.pt \\
      --epochs 5 --batch 256 --lr 3e-4
"""
from __future__ import annotations

import argparse
import glob
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from featurize import PLANET_DIM, FLEET_DIM, GLOBAL_DIM
from training.model import OrbitAgent


class ValueNet(nn.Module):
    """State encoder (same architecture as OrbitAgent) + scalar value head.

    Policy heads are dropped; we reuse OrbitAgent just for its encoder stack.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.enc = OrbitAgent(
            planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, planets, planet_mask, fleets, fleet_mask, globals_):
        B = planets.shape[0]
        enc = self.enc
        p_tok = enc.planet_embed(planets) + enc.type_embed.weight[0]
        f_tok = enc.fleet_embed(fleets) + enc.type_embed.weight[1]
        g_tok = enc.global_embed(globals_).unsqueeze(1) + enc.type_embed.weight[2]
        tokens = torch.cat([p_tok, f_tok, g_tok], dim=1)
        g_mask = torch.ones(B, 1, dtype=torch.bool, device=planets.device)
        valid = torch.cat([planet_mask, fleet_mask, g_mask], dim=1)
        for blk in enc.layers:
            tokens = blk(tokens, ~valid)
        global_tok = tokens[:, -1, :]
        return self.value_head(global_tok).squeeze(-1)


class OfflineValueDataset(Dataset):
    """Flattens offline npz shards into (state, shape_return) sample pairs.

    Loads lazily per shard — full dataset may be ~500k steps, too big to hold
    all tensors in memory. We index by (shard_idx, step_idx) and load on demand.
    """
    def __init__(self, offline_dir: str):
        self.shards = sorted(glob.glob(str(Path(offline_dir) / "*.npz")))
        if not self.shards:
            raise RuntimeError(f"no .npz found under {offline_dir}")
        # Build index: (shard_path, step_idx) per sample
        self.index: list[tuple[str, int]] = []
        print(f"[v_phi] indexing {len(self.shards)} shards...")
        for sh in self.shards:
            with np.load(sh, allow_pickle=True) as d:
                n = int(d["n_steps"])
                self.index.extend((sh, i) for i in range(n))
        print(f"[v_phi] total samples: {len(self.index)}")
        self._cache: dict[str, dict] = {}

    def __len__(self):
        return len(self.index)

    def _load_shard(self, path: str) -> dict:
        if path not in self._cache:
            if len(self._cache) > 16:
                self._cache.pop(next(iter(self._cache)))  # LRU-ish
            with np.load(path, allow_pickle=True) as d:
                self._cache[path] = {k: d[k] for k in d.files}
        return self._cache[path]

    def __getitem__(self, idx: int):
        sh_path, step = self.index[idx]
        s = self._load_shard(sh_path)
        planets = np.asarray(s["planets"][step], dtype=np.float32)
        fleets = np.asarray(s["fleets"][step], dtype=np.float32)
        globals_ = np.asarray(s["globals"][step], dtype=np.float32)
        G = float(s["shape_return"][step])  # MC discounted return
        return {
            "planets": planets,
            "fleets": fleets,
            "globals": globals_,
            "target": G,
        }


def collate_pad(batch: list[dict]) -> dict:
    B = len(batch)
    P = max(x["planets"].shape[0] for x in batch)
    F_ = max(x["fleets"].shape[0] for x in batch) if batch[0]["fleets"].ndim > 0 else 0
    F_ = max(F_, 1)

    planets = np.zeros((B, P, PLANET_DIM), dtype=np.float32)
    pmask = np.zeros((B, P), dtype=bool)
    fleets = np.zeros((B, F_, FLEET_DIM), dtype=np.float32)
    fmask = np.zeros((B, F_), dtype=bool)
    globals_ = np.zeros((B, GLOBAL_DIM), dtype=np.float32)
    target = np.zeros((B,), dtype=np.float32)

    for i, x in enumerate(batch):
        np_ = x["planets"]; nf = x["fleets"]
        planets[i, :np_.shape[0]] = np_
        pmask[i, :np_.shape[0]] = True
        if nf.ndim == 2 and nf.shape[0] > 0:
            fleets[i, :nf.shape[0]] = nf
            fmask[i, :nf.shape[0]] = True
        globals_[i] = x["globals"]
        target[i] = x["target"]

    return {
        "planets": torch.from_numpy(planets),
        "planet_mask": torch.from_numpy(pmask),
        "fleets": torch.from_numpy(fleets),
        "fleet_mask": torch.from_numpy(fmask),
        "globals": torch.from_numpy(globals_),
        "target": torch.from_numpy(target),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ValueNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ds = OfflineValueDataset(args.offline_dir)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate_pad,
                    pin_memory=True, drop_last=True)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total, n = 0.0, 0
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pred = model(batch["planets"], batch["planet_mask"],
                         batch["fleets"], batch["fleet_mask"], batch["globals"])
            loss = F.smooth_l1_loss(pred, batch["target"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * batch["target"].size(0)
            n += batch["target"].size(0)
        avg = total / max(1, n)
        print(f"[v_phi] epoch {epoch+1}/{args.epochs}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"[v_phi] saved {args.out}")

    print(f"[v_phi] done. best_loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
