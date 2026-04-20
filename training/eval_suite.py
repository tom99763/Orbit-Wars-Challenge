"""Eval a trained agent against starter, lb-928, and lb-1200 (50 games each).

Greedy (temperature=0). Reports win rate per opponent + overall, plus a
distribution over starting-seat positions to catch any seat bias.

Usage:
  python training/eval_suite.py \
      --ckpt training/checkpoints/imitation_v1.pt \
      --n-games 50 --four-player-prob 0.2
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import starter_agent
from training.agent_v4 import load_agent
from training.lb928_agent import agent as lb928_agent
from training.lb1200_agent import agent as lb1200_agent


def play_one(me, opp, n_players: int, me_seat: int = 0) -> int:
    """Return +1 if me won, 0 if tied/lost. me_seat chooses which slot we take."""
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)
    agents = []
    for s in range(n_players):
        agents.append(me if s == me_seat else opp)
    env.run(agents)
    final = env.steps[-1]
    my_r = final[me_seat].reward or 0
    max_r = max((final[s].reward or 0) for s in range(n_players))
    return 1 if (my_r == max_r and my_r > 0) else 0


def eval_vs(me, opp, name: str, n_games: int, four_player_prob: float) -> dict:
    wins = 0
    per_seat_wins = {}
    t0 = time.time()
    for i in range(n_games):
        n_players = 4 if random.random() < four_player_prob else 2
        me_seat = random.randint(0, n_players - 1)
        w = play_one(me, opp, n_players, me_seat)
        wins += w
        per_seat_wins.setdefault((n_players, me_seat), [0, 0])
        per_seat_wins[(n_players, me_seat)][0] += w
        per_seat_wins[(n_players, me_seat)][1] += 1
    dt = time.time() - t0
    print(f"[{name}] {wins}/{n_games} ({wins/n_games:.1%})  [{dt:.0f}s]")
    for k in sorted(per_seat_wins.keys()):
        w, n = per_seat_wins[k]
        print(f"  {k[0]}P seat {k[1]}: {w}/{n}")
    return {"name": name, "wins": wins, "games": n_games, "wr": wins / n_games}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-games", type=int, default=50)
    ap.add_argument("--four-player-prob", type=float, default=0.2)
    ap.add_argument("--opponents", default="starter,lb928,lb1200",
                    help="Comma-separated subset")
    args = ap.parse_args()

    print(f"loading {args.ckpt}", flush=True)
    me = load_agent(args.ckpt, device="cpu", temperature=0.0)

    OPP = {
        "starter": starter_agent,
        "lb928": lb928_agent,
        "lb1200": lb1200_agent,
    }
    requested = [x.strip() for x in args.opponents.split(",")]

    print(f"eval ckpt={args.ckpt} n={args.n_games} 4P_prob={args.four_player_prob}",
          flush=True)
    results = []
    for name in requested:
        if name not in OPP:
            print(f"  skipping unknown opp {name}")
            continue
        r = eval_vs(me, OPP[name], name, args.n_games, args.four_player_prob)
        results.append(r)

    print("\n=== summary ===")
    for r in results:
        print(f"  vs {r['name']:>8s}: {r['wins']}/{r['games']} ({r['wr']:.1%})")


if __name__ == "__main__":
    sys.exit(main() or 0)
