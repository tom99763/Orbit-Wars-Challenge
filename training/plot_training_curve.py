"""Plot training curves from physics_picker_k14_vec.py logs.

Parses the per-iter log lines emitted by the trainer and produces a
multi-panel matplotlib figure showing the most useful curves over time:

  - learner_elo (frozen-anchor ELO) — the AlphaStar-style "purple→cyan"
    monotonic-rise plot if training is healthy
  - per-iter wins/total — raw win-rate vs the rollout opponent mix
  - pi_loss / v_loss / entropy — PPO health indicators
  - bc_mode_ce / bc_frac_ce — BC aux loss (only plotted if --bc-data-dir
    was used in the training run)

Log line format (anchor: see physics_picker_k14_vec.py main loop):
    [iter 00042]  wins=12/16  (lb1200=2/4 pool=10/12)  T=2034
    pi=0.043  v=0.872  ent=2.110  r=1.02/1.5  ent_c=0.000  pool=8
    hardest=[iter0040=0.45(8) ...]  lb_noise=0.40  mc=[...] fc=[...]
    bc_m=1.21/0.55  bc_f=1.85/0.32  learner_elo=1542.3  elo_games=84  [12s]

Usage:
    python training/plot_training_curve.py path/to/train.log -o curves.png
    python training/plot_training_curve.py path/to/train.log --show
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass, field


# ────────────────────────────────────────────────────────────────────────────
# Log line parser
# ────────────────────────────────────────────────────────────────────────────

_FIELD_PATTERNS = {
    "iter":         r"\[iter (\d+)\]",
    "wins":         r"wins=(\d+)/(\d+)",
    "T":            r"T=(\d+)",
    "pi_loss":      r"pi=([-\d.]+)",
    "v_loss":       r"v=([-\d.]+)",
    "ent":          r"ent=([-\d.]+)",
    "ratio_mean":   r"r=([-\d.]+)/[-\d.]+",
    "ratio_max":    r"r=[-\d.]+/([-\d.]+)",
    "ent_c":        r"ent_c=([-\d.]+)",
    "pool_size":    r"pool=(\d+)",
    "lb_noise":     r"lb_noise=([-\d.]+)",
    "bc_mode_ce":   r"bc_m=([-\d.]+)/[-\d.]+",
    "bc_mode_acc":  r"bc_m=[-\d.]+/([-\d.]+)",
    "bc_frac_ce":   r"bc_f=([-\d.]+)/[-\d.]+",
    "bc_frac_acc":  r"bc_f=[-\d.]+/([-\d.]+)",
    "learner_elo":  r"learner_elo=([-\d.]+)",
    "elo_games":    r"elo_games=(\d+)",
    "wall":         r"\[(\d+)s\]\s*$",
}


@dataclass
class IterRow:
    iter: int
    wins: int = 0
    total: int = 0
    T: float = 0.0
    pi_loss: float = float("nan")
    v_loss: float = float("nan")
    ent: float = float("nan")
    ratio_mean: float = float("nan")
    ratio_max: float = float("nan")
    ent_c: float = float("nan")
    pool_size: int = 0
    lb_noise: float = float("nan")
    bc_mode_ce: float = float("nan")
    bc_mode_acc: float = float("nan")
    bc_frac_ce: float = float("nan")
    bc_frac_acc: float = float("nan")
    learner_elo: float = float("nan")
    elo_games: int = 0
    wall: float = float("nan")

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else float("nan")


def _grab(line: str, key: str) -> str | None:
    m = re.search(_FIELD_PATTERNS[key], line)
    if not m:
        return None
    return m.group(1)


def parse_log(path: pathlib.Path) -> list[IterRow]:
    rows: list[IterRow] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[iter " not in line:
                continue
            # Skip "[iter NNNNN] saved …" lines emitted at checkpoint boundaries
            # — they reuse the iter prefix but lack the metric fields.
            if "saved" in line and "wins=" not in line:
                continue
            it_str = _grab(line, "iter")
            if it_str is None:
                continue
            row = IterRow(iter=int(it_str))
            wins_match = re.search(_FIELD_PATTERNS["wins"], line)
            if wins_match:
                row.wins = int(wins_match.group(1))
                row.total = int(wins_match.group(2))
            for key in ("T", "pool_size", "elo_games"):
                v = _grab(line, key)
                if v is not None:
                    setattr(row, key, int(float(v)))
            for key in ("pi_loss", "v_loss", "ent", "ratio_mean", "ratio_max",
                        "ent_c", "lb_noise", "bc_mode_ce", "bc_mode_acc",
                        "bc_frac_ce", "bc_frac_acc", "learner_elo", "wall"):
                v = _grab(line, key)
                if v is not None:
                    setattr(row, key, float(v))
            rows.append(row)
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

def _has_data(rows: list[IterRow], attr: str) -> bool:
    """Return True if at least one row has a finite, non-default value."""
    import math
    for r in rows:
        v = getattr(r, attr)
        if isinstance(v, float):
            if not math.isnan(v):
                return True
        elif v:
            return True
    return False


def plot_curves(rows: list[IterRow], out_path: pathlib.Path | None,
                show: bool = False) -> None:
    if not rows:
        print("no rows parsed — nothing to plot", file=sys.stderr)
        return
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; install with `pip install matplotlib`",
              file=sys.stderr)
        sys.exit(1)

    iters = [r.iter for r in rows]
    has_bc = _has_data(rows, "bc_mode_ce")

    n_panels = 4 + (1 if has_bc else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.6 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: ELO
    ax = axes[0]
    elo_vals = [r.learner_elo for r in rows]
    ax.plot(iters, elo_vals, color="#3a86ff", lw=1.6)
    ax.set_ylabel("learner_elo")
    ax.set_title("Frozen-anchor learner ELO  (rises monotonically iff learner is improving)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(1500.0, color="#888", ls="--", lw=0.7, alpha=0.6)

    # Panel 2: rollout WR
    ax = axes[1]
    wr_vals = [r.win_rate for r in rows]
    ax.plot(iters, wr_vals, color="#06d6a0", lw=1.4)
    ax.set_ylabel("win_rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Rollout win rate (mixed opp distribution per iter)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color="#888", ls="--", lw=0.7, alpha=0.6)

    # Panel 3: PPO losses
    ax = axes[2]
    ax.plot(iters, [r.pi_loss for r in rows], color="#ef476f", lw=1.2, label="pi_loss")
    ax.plot(iters, [r.v_loss for r in rows], color="#ffd166", lw=1.2, label="v_loss")
    ax.set_ylabel("loss")
    ax.set_title("PPO losses", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4: entropy + ent_coef
    ax = axes[3]
    ax.plot(iters, [r.ent for r in rows], color="#8338ec", lw=1.2, label="ent")
    ax.set_ylabel("entropy")
    ax2 = ax.twinx()
    ax2.plot(iters, [r.ent_c for r in rows], color="#aaa", lw=0.8, ls=":",
             label="ent_coef")
    ax2.set_ylabel("ent_c", color="#888")
    ax.set_title("Policy entropy + scheduled entropy coef", fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 5 (optional): BC losses
    if has_bc:
        ax = axes[4]
        ax.plot(iters, [r.bc_mode_ce for r in rows], color="#118ab2", lw=1.2,
                label="bc_mode_ce")
        ax.plot(iters, [r.bc_frac_ce for r in rows], color="#073b4c", lw=1.2,
                label="bc_frac_ce")
        ax.set_ylabel("BC CE")
        ax.set_title("Behavioural-cloning auxiliary loss (when --bc-data-dir set)",
                     fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("iter")
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        print(f"wrote {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=pathlib.Path,
                    help="Trainer log file (output of physics_picker_k14_vec.py)")
    ap.add_argument("-o", "--out", type=pathlib.Path, default=None,
                    help="Output PNG path. Default: <log>.png next to the log.")
    ap.add_argument("--show", action="store_true",
                    help="Open an interactive matplotlib window (uses TkAgg).")
    ap.add_argument("--csv", type=pathlib.Path, default=None,
                    help="Optional: also emit per-iter rows as CSV for ad-hoc analysis.")
    args = ap.parse_args()

    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 1

    rows = parse_log(args.log)
    print(f"parsed {len(rows)} iter rows from {args.log}")
    if not rows:
        return 1

    last = rows[-1]
    print(f"  last iter: {last.iter}  learner_elo={last.learner_elo:.1f}  "
          f"pool={last.pool_size}  win_rate={last.win_rate:.2f}  "
          f"pi={last.pi_loss:.3f}  v={last.v_loss:.3f}  ent={last.ent:.3f}")

    if args.csv:
        import csv
        cols = list(IterRow.__dataclass_fields__.keys()) + ["win_rate"]
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([getattr(r, c) for c in cols[:-1]] + [r.win_rate])
        print(f"wrote {args.csv}")

    out = args.out or args.log.with_suffix(args.log.suffix + ".png")
    plot_curves(rows, out, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
