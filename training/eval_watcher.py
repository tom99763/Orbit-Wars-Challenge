"""Poll a checkpoint file; whenever it changes, run an eval that MIRRORS the
training matchup distribution (80% 2P / 20% 4P, non-learner seats 50% lb928 /
50% starter). Append one row per eval pass to CSV.

The `me` seat is always seat 0. Other seats are independently rolled.

Usage:
  python training/eval_watcher.py --ckpt training/checkpoints/impala_v4.pt \
      --interval 120 --n-games 10 \
      --four-player-prob 0.2 --lb928-prob 0.5 \
      --out .impala_v4_eval_watch.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kaggle_environments import make
from training.agent_v4 import load_agent
from training.lb928_agent import agent as lb928_agent
from kaggle_environments.envs.orbit_wars.orbit_wars import starter_agent


def pick_matchup(four_player_prob: float, lb928_prob: float) -> tuple[int, list[str]]:
    n_players = 4 if random.random() < four_player_prob else 2
    types = ["me"]
    for _ in range(n_players - 1):
        types.append("lb928" if random.random() < lb928_prob else "starter")
    return n_players, types


def play_one(me, n_players: int, seat_types: list[str]) -> dict:
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)
    agents = []
    for t in seat_types:
        if t == "me":
            agents.append(me)
        elif t == "lb928":
            agents.append(lb928_agent)
        else:
            agents.append(starter_agent)
    env.run(agents)
    final = env.steps[-1]
    rewards = [final[s].reward or 0 for s in range(n_players)]
    max_r = max(rewards)
    my_r = rewards[0]
    won = (my_r == max_r) and (rewards.count(max_r) == 1 or my_r > 0)
    has_lb928 = any(t == "lb928" for t in seat_types)
    return {"won": bool(won), "has_lb928": has_lb928,
            "my_reward": my_r, "max_reward": max_r}


def safe_load_agent(ckpt_path: str, retries: int = 5, delay: float = 2.0):
    for i in range(retries):
        try:
            return load_agent(ckpt_path, device="cpu", temperature=0.0)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(delay)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--n-games", type=int, default=10)
    ap.add_argument("--four-player-prob", type=float, default=0.2)
    ap.add_argument("--lb928-prob", type=float, default=0.5)
    ap.add_argument("--out", default=".eval_watch.csv")
    args = ap.parse_args()

    ckpt_path = pathlib.Path(args.ckpt).resolve()
    out_path = pathlib.Path(args.out).resolve()

    fields = ["ts", "ckpt_mtime", "n_games", "overall_wins", "overall_wr",
              "vs_lb928_games", "vs_lb928_wins", "vs_lb928_share",
              "vs_starter_games", "vs_starter_wins", "vs_starter_share",
              "wall_s"]
    if not out_path.exists():
        with out_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    last_mtime = 0.0
    print(f"watching {ckpt_path}  interval={args.interval}s  N={args.n_games}  "
          f"fpp={args.four_player_prob} lb={args.lb928_prob}",
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
              f"checkpoint updated — evaluating …", flush=True)
        try:
            me = safe_load_agent(str(ckpt_path))
        except Exception as e:
            print(f"  failed to load: {e}", flush=True)
            time.sleep(60)
            continue

        overall_wins = 0
        vs_lb_games = vs_lb_wins = 0
        vs_st_games = vs_st_wins = 0
        for _ in range(args.n_games):
            n, types = pick_matchup(args.four_player_prob, args.lb928_prob)
            r = play_one(me, n, types)
            if r["won"]:
                overall_wins += 1
            if r["has_lb928"]:
                vs_lb_games += 1
                if r["won"]:
                    vs_lb_wins += 1
            else:
                vs_st_games += 1
                if r["won"]:
                    vs_st_wins += 1

        overall_wr = overall_wins / args.n_games
        vs_lb_share = vs_lb_wins / vs_lb_games if vs_lb_games else 0.0
        vs_st_share = vs_st_wins / vs_st_games if vs_st_games else 0.0
        wall_s = round(time.time() - t0, 1)

        row = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "ckpt_mtime": int(mtime),
            "n_games": args.n_games,
            "overall_wins": overall_wins,
            "overall_wr": round(overall_wr, 3),
            "vs_lb928_games": vs_lb_games,
            "vs_lb928_wins": vs_lb_wins,
            "vs_lb928_share": round(vs_lb_share, 3),
            "vs_starter_games": vs_st_games,
            "vs_starter_wins": vs_st_wins,
            "vs_starter_share": round(vs_st_share, 3),
            "wall_s": wall_s,
        }
        with out_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"  overall: {overall_wins}/{args.n_games} ({overall_wr:.0%})  "
              f"vs_lb928: {vs_lb_wins}/{vs_lb_games} ({vs_lb_share:.0%})  "
              f"vs_starter: {vs_st_wins}/{vs_st_games} ({vs_st_share:.0%})  "
              f"[{wall_s}s]",
              flush=True)
        last_mtime = mtime
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
