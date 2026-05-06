"""PFSP opponent pool + frozen-anchor ELO bookkeeping.

Standalone simple_rl_v2 version of the league. Same mechanism as
training.physics_picker_k14_vec.OpponentPool (commit 9ccd914), but
self-contained so simple_rl_v2 has no cross-module dependency on the
old k14 trainer.

Design (per CLAUDE-2.md §5):
  - Pool stores frozen state_dicts. Each entry has a snapshot_elo set to
    the learner's ELO at the moment that snapshot was frozen.
  - Anchor ELOs NEVER update.
  - Learner has a single ELO (default 1500), updated only on games vs
    pool entries via standard Elo (K=16).
  - Per rollout: PFSP-sample one anchor, weight ∝ (1 − wr_vs_anchor)^p
    + floor, focusing on opponents we still lose to.
  - Optional latest-self mirror branch: with probability LATEST_PROB the
    rollout uses current weights as opponent (no ELO update).

The "purple→cyan" plot is a property of frozen ELOs + monotonic learner
improvement: as the learner gets stronger, beating older snapshots
gives diminishing-but-positive ELO gains; failing to beat newest
snapshots stays at ~50% WR with zero ELO change. Net trend rises iff
the learner is genuinely improving.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass, field

import torch


@dataclass
class PoolEntry:
    """One frozen snapshot in the pool."""
    state_dict_bytes: bytes   # serialized via torch.save / io.BytesIO
    tag: str                  # human-readable label (e.g. "iter0250")
    snapshot_elo: float       # learner's ELO at add() time — IMMUTABLE
    win_ema: float = 0.5      # EMA of P(learner beats this snapshot)
    games: int = 0            # raw count for confidence + eviction grace


class League:
    """PFSP pool + ELO bookkeeping.

    Args:
        max_size:    pool capacity (FIFO eviction with mature-protection)
        elo_init:    learner ELO at training start (default 1500)
        elo_k:       K factor in Elo update (default 16, matches CLAUDE-2.md)
        pfsp_p:      sampling weight exponent: w ∝ (1-wr)^p + floor
        pfsp_floor:  minimum sampling weight per entry
        ema_alpha:   EMA mixing for win_ema; smaller = smoother
        min_games_for_eviction: don't evict snapshots with fewer games
                     than this (so fresh additions get a fair chance)
        latest_prob: per-rollout probability of using "latest self" as
                     opponent instead of pool (no ELO update). The
                     trainer is responsible for honouring this — League
                     just exposes the constant.
    """
    def __init__(
        self,
        max_size: int = 16,
        elo_init: float = 1500.0,
        elo_k: float = 16.0,
        pfsp_p: float = 2.0,
        pfsp_floor: float = 0.05,
        ema_alpha: float = 0.15,
        min_games_for_eviction: int = 3,
        latest_prob: float = 0.3,
    ):
        self.max_size = max_size
        self.elo_init = float(elo_init)
        self.elo_k    = float(elo_k)
        self.pfsp_p   = float(pfsp_p)
        self.pfsp_floor = float(pfsp_floor)
        self.ema_alpha = float(ema_alpha)
        self.min_games_for_eviction = int(min_games_for_eviction)
        self.latest_prob = float(latest_prob)

        self.entries: list[PoolEntry] = []
        self.learner_elo: float = float(elo_init)
        self.elo_games: int = 0

    # ── pool management ────────────────────────────────────────────────────
    def add(self, model: torch.nn.Module, tag: str) -> None:
        """Snapshot current learner with its current ELO (frozen)."""
        buf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
        entry = PoolEntry(
            state_dict_bytes=buf.getvalue(),
            tag=tag,
            snapshot_elo=self.learner_elo,
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            self._evict()

    def _evict(self) -> None:
        """Drop the snapshot we beat MOST (highest win_ema), among entries
        that have enough games to be confident. Falls back to FIFO."""
        mature = [
            i for i, e in enumerate(self.entries)
            if e.games >= self.min_games_for_eviction
        ]
        if mature:
            drop = max(mature, key=lambda i: self.entries[i].win_ema)
        else:
            drop = 0
        self.entries.pop(drop)

    def __len__(self) -> int:
        return len(self.entries)

    def load_into(self, model: torch.nn.Module, idx: int) -> None:
        """Load the indexed snapshot's weights into `model` (in-place)."""
        if not (0 <= idx < len(self.entries)):
            raise IndexError(f"pool idx {idx} out of range (size {len(self)})")
        buf = io.BytesIO(self.entries[idx].state_dict_bytes)
        sd = torch.load(buf, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)

    # ── sampling ───────────────────────────────────────────────────────────
    def sample_pfsp(self) -> int:
        """Return an opponent index, weighted toward harder opponents.

        Returns -1 if pool is empty.
        """
        n = len(self.entries)
        if n == 0:
            return -1
        if n == 1:
            return 0
        weights = [
            (1.0 - min(0.99, e.win_ema)) ** self.pfsp_p + self.pfsp_floor
            for e in self.entries
        ]
        return random.choices(range(n), weights=weights, k=1)[0]

    # ── result accounting ──────────────────────────────────────────────────
    def record_result(self, idx: int, won: bool) -> None:
        """Record one game outcome against snapshot `idx`.

        Updates win_ema (PFSP / eviction tracking) and learner_elo
        (Elo update vs the snapshot's frozen ELO). Snapshot ELO is NEVER
        modified.
        """
        if not (0 <= idx < len(self.entries)):
            return
        e = self.entries[idx]
        x = 1.0 if won else 0.0
        # PFSP / eviction stats
        e.win_ema = (1.0 - self.ema_alpha) * e.win_ema + self.ema_alpha * x
        e.games += 1
        # Frozen-anchor Elo: learner only
        expected = 1.0 / (1.0 + 10.0 ** ((e.snapshot_elo - self.learner_elo) / 400.0))
        self.learner_elo += self.elo_k * (x - expected)
        self.elo_games += 1

    # ── diagnostics ────────────────────────────────────────────────────────
    def hardest_summary(self, top_k: int = 3) -> str:
        """Tag-keyed summary of the K opponents we struggle most against."""
        if not self.entries:
            return ""
        ranked = sorted(self.entries, key=lambda e: e.win_ema)[:top_k]
        return " ".join(
            f"{e.tag}={e.win_ema:.2f}({e.games})" for e in ranked
        )

    def elo_summary(self) -> str:
        """Compact ELO line for trainer logs."""
        if not self.entries:
            return f"learner={self.learner_elo:.0f}"
        elos = [e.snapshot_elo for e in self.entries]
        return (f"learner={self.learner_elo:.0f} "
                f"pool_elo=[{min(elos):.0f}..{max(elos):.0f}] "
                f"elo_games={self.elo_games}")


__all__ = ["League", "PoolEntry"]
