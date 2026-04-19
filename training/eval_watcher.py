"""Poll a checkpoint file; whenever it changes, run small eval vs starter
and lb-928, append row to CSV. Non-blocking relative to the training run.

Tolerates mid-write races (torch.load failures → sleep + retry).

Usage:
  python training/eval_watcher.py --ckpt training/checkpoints/a2c_v1.pt \
      --interval 600 --n-games 10 --out .a2c_eval_watch.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kaggle_environments import make
from training.agent import load_agent
from training.lb928_agent import agent as lb928_agent
from kaggle_environments.envs.orbit_wars.orbit_wars import starter_agent


def play(me, opp, n: int, seats_both: bool = True) -> dict:
    env = make("orbit_wars", debug=False)
    wins = 0; losses = 0; draws = 0
    for i in range(n):
        if seats_both and i % 2 == 1:
            first, second = opp, me
            swapped = True
        else:
            first, second = me, opp
            swapped = False
        env.reset()
        env.run([first, second])
        final = env.steps[-1]
        r0, r1 = final[0].reward or 0, final[1].reward or 0
        if swapped:
            r0, r1 = r1, r0
        if r0 > r1: wins += 1
        elif r1 > r0: losses += 1
        else: draws += 1
    total = wins + losses + draws
    return {"wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / total if total else 0.0}


def safe_load_agent(ckpt_path: str, retries: int = 5, delay: float = 2.0):
    """Load a checkpoint, retrying if the file is being written."""
    for i in range(retries):
        try:
            return load_agent(ckpt_path, device="cpu", temperature=0.0)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(delay)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--interval", type=int, default=600,
                    help="Seconds between checks (default 600)")
    ap.add_argument("--n-games", type=int, default=10)
    ap.add_argument("--out", default=".a2c_eval_watch.csv")
    ap.add_argument("--starter", action="store_true", default=True)
    ap.add_argument("--lb928", action="store_true", default=True)
    args = ap.parse_args()

    ckpt_path = pathlib.Path(args.ckpt).resolve()
    out_path = pathlib.Path(args.out).resolve()

    fields = ["ts", "ckpt_mtime", "vs_starter_wins", "vs_starter_losses",
              "vs_starter_wr", "vs_lb928_wins", "vs_lb928_losses",
              "vs_lb928_wr", "wall_s"]
    if not out_path.exists():
        with out_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    last_mtime = 0.0
    print(f"watching {ckpt_path}  interval={args.interval}s  N={args.n_games}",
          flush=True)
    while True:
        try:
            mtime = os.path.getmtime(ckpt_path) if ckpt_path.exists() else 0
        except OSError:
            mtime = 0
        if mtime == 0 or mtime == last_mtime:
            time.sleep(args.interval)
            continue
        t0 = time.time()
        print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] "
              f"checkpoint updated ({mtime - last_mtime:.0f}s since last) — "
              f"evaluating …", flush=True)
        try:
            me = safe_load_agent(str(ckpt_path))
        except Exception as e:
            print(f"  failed to load: {e}", flush=True)
            time.sleep(60)
            continue

        res_starter = play(me, starter_agent, args.n_games)
        res_lb = play(me, lb928_agent, args.n_games) if args.lb928 else {
            "wins": 0, "losses": 0, "draws": 0, "win_rate": 0}

        row = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "ckpt_mtime": int(mtime),
            "vs_starter_wins": res_starter["wins"],
            "vs_starter_losses": res_starter["losses"],
            "vs_starter_wr": round(res_starter["win_rate"], 3),
            "vs_lb928_wins": res_lb["wins"],
            "vs_lb928_losses": res_lb["losses"],
            "vs_lb928_wr": round(res_lb["win_rate"], 3),
            "wall_s": round(time.time() - t0, 1),
        }
        with out_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"  vs starter: {res_starter['wins']}W/{res_starter['losses']}L "
              f"({res_starter['win_rate']:.0%})  "
              f"vs lb928: {res_lb['wins']}W/{res_lb['losses']}L "
              f"({res_lb['win_rate']:.0%})  [{row['wall_s']}s]",
              flush=True)
        last_mtime = mtime
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
