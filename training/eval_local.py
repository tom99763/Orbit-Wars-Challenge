"""Round-robin local evaluation of our agent vs baselines.

Usage:
  python training/eval_local.py --ckpt training/checkpoints/bc_v1.pt \
                                --n-games 40

Runs `kaggle_environments.make('orbit_wars').run([A, B])` many times and
prints win rates. Supports comparing against `random`, `starter`, and
optionally a submission.py path (imports its `agent` callable).
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch   # noqa: E402
from kaggle_environments import make   # noqa: E402
from training.agent import load_agent   # noqa: E402


def cap_gpu_memory(fraction: float, device_index: int = 0) -> None:
    """Hard-cap our process to `fraction` of the visible GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction, device_index)


def load_external_agent(path: str):
    """Import `agent` from a Python file path (for e.g. lb-928 planner)."""
    spec = importlib.util.spec_from_file_location("ext_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def play_matches(a, b, n: int, label_a: str, label_b: str, seats="both"):
    """Play n games with agents a and b. seats='both' rotates starting seat."""
    env = make("orbit_wars", debug=False)
    wins_a, wins_b, draws = 0, 0, 0
    for i in range(n):
        # Alternate seats
        if seats == "both":
            first, second = (a, b) if i % 2 == 0 else (b, a)
            swapped = i % 2 == 1
        else:
            first, second = a, b
            swapped = False
        env.reset()
        env.run([first, second])
        final = env.steps[-1]
        r0, r1 = final[0].reward, final[1].reward
        if swapped:
            r0, r1 = r1, r0
        if r0 > r1:
            wins_a += 1
        elif r1 > r0:
            wins_b += 1
        else:
            draws += 1
    total = wins_a + wins_b + draws
    wr = wins_a / total if total else 0.0
    print(f"{label_a} vs {label_b}: {wins_a}W/{wins_b}L/{draws}D  "
          f"win_rate={wr:.1%}", flush=True)
    return wr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-games", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-fraction", type=float, default=0.15,
                    help="Hard cap on GPU memory fraction when device=cuda*")
    ap.add_argument("--extra-agent", default=None,
                    help="Optional path to another submission.py for comparison")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        cap_gpu_memory(args.mem_fraction, 0)
    me = load_agent(args.ckpt, device=args.device)
    label_me = f"ckpt({pathlib.Path(args.ckpt).stem})"

    t0 = time.time()
    results = {}
    results["random"] = play_matches(me, "random", args.n_games, label_me, "random")
    results["starter"] = play_matches(me, "starter", args.n_games, label_me, "starter")

    if args.extra_agent:
        ext = load_external_agent(args.extra_agent)
        label = pathlib.Path(args.extra_agent).stem
        results[label] = play_matches(me, ext, args.n_games, label_me, label)

    print(f"\ntotal={time.time()-t0:.1f}s  winrates={results}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
