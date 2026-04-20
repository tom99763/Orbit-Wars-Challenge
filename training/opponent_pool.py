"""Opponent pool for self-play non-stationarity mitigation.

Recipe (League-lite, simpler than AlphaStar league):
  - Save a snapshot of current actor every N iters to `pool_dir/`.
  - Cap pool at `max_size` (FIFO evict).
  - For each rollout task, sample opponent type by weighted distribution:
        self:    current actor π_θ (default 0.60)
        past:    uniform random from pool (default 0.20)
        lb1200:  static anchor (default 0.15)
        lb928:   static anchor (default 0.05)
  - Static anchors prevent the policy from drifting into "beats only past self"
    degenerate strategies that do not transfer to the real leaderboard.

Mix defaults informed by design_impala_v3_exploration.md + feedback_lb928_prob.md
(user-validated lb-928 at 0.1 historical; here split between lb-928 and lb-1200).
"""
from __future__ import annotations

import glob
import random
import shutil
import time
from pathlib import Path

import torch


class OpponentPool:
    def __init__(self, pool_dir: str, max_size: int = 10):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size

    def _snapshots(self) -> list[Path]:
        return sorted(self.pool_dir.glob("snap_*.pt"),
                      key=lambda p: p.stat().st_mtime)

    def add(self, state_dict: dict) -> Path:
        """Save a new snapshot, evicting oldest if over capacity."""
        ts = int(time.time())
        path = self.pool_dir / f"snap_{ts}.pt"
        torch.save(state_dict, path)
        snaps = self._snapshots()
        while len(snaps) > self.max_size:
            snaps[0].unlink(missing_ok=True)
            snaps = snaps[1:]
        return path

    def size(self) -> int:
        return len(self._snapshots())

    def sample_past(self) -> Path | None:
        snaps = self._snapshots()
        return random.choice(snaps) if snaps else None


def sample_opponent_mix(rng: random.Random, mix: dict[str, float]) -> str:
    """Return one of the opponent type keys by weighted sampling.

    `mix` maps type → probability. Keys must sum to 1.0 (not validated).
    Returns: "self" | "past" | "lb1200" | "lb928" | "starter" | ...
    """
    total = sum(mix.values())
    x = rng.random() * total
    cum = 0.0
    for k, w in mix.items():
        cum += w
        if x <= cum:
            return k
    # fallback to last
    return next(reversed(mix))


DEFAULT_MIX = {
    "self":   0.60,
    "past":   0.20,
    "lb1200": 0.15,
    "lb928":  0.05,
}
