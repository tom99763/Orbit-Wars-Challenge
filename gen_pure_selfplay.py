"""Pure self-play: same checkpoint plays all seats, 2p or 4p randomly.

Each game logs a per-game reward summary so we can monitor how the
policy splits reward across seats and game modes. Diversity comes from
temperature sampling (T>0) inside the agent.

Output:
  simulation/<date>/_synth_selfplay_<tag>/<ep_id>/replay.json + episode.json
  .selfplay_monitor.csv (appended): game_id, worker_id, n_players, steps,
                                    seat_rewards, winner_seat, timestamp

Usage:
  python gen_pure_selfplay.py --ckpt training/checkpoints/bc_v2.pt \
                              --temperature 1.0 \
                              --n-games 250 \
                              --four-player-prob 0.5 \
                              --worker-id 0
"""

import argparse
import csv
import datetime
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from kaggle_environments import make
from training.agent import load_agent


MONITOR_CSV = pathlib.Path(__file__).parent / ".selfplay_monitor.csv"
MONITOR_HEADERS = ["ts", "worker", "game_id", "n_players", "steps",
                   "rewards", "winner_seat", "final_ships_by_seat"]


def append_monitor(row: dict):
    new_file = not MONITOR_CSV.exists()
    with MONITOR_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MONITOR_HEADERS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def run_one_game(agent_fn, n_players: int) -> dict:
    agents = [agent_fn for _ in range(n_players)]
    env = make("orbit_wars", debug=False)
    env.run(agents)
    return env


def save_replay_from_env(env, out_dir: pathlib.Path, team_slug: str,
                         ep_id: int, scrape_time: str):
    replay = env.toJSON()
    replay["id"] = ep_id
    replay.setdefault("info", {})
    replay["info"]["EpisodeId"] = ep_id
    n = len(replay.get("rewards") or [])
    replay["info"]["TeamNames"] = [f"{team_slug}_p{i}" for i in range(n)]
    replay["info"]["Agents"] = [{"Name": replay["info"]["TeamNames"][i],
                                 "ThumbnailUrl": None} for i in range(n)]
    ep_dir = out_dir / team_slug / str(ep_id)
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    (ep_dir / "episode.json").write_text(json.dumps({
        "episode_id": ep_id, "team_seen_by": team_slug,
        "rank_at_scrape": -1, "scrape_time": scrape_time,
        "TeamNames": replay["info"]["TeamNames"],
        "Agents": replay["info"]["Agents"],
        "rewards": replay.get("rewards"),
        "statuses": replay.get("statuses"),
        "configuration": replay.get("configuration"),
        "synthetic": True, "selfplay": True,
    }, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-games", type=int, default=250)
    ap.add_argument("--four-player-prob", type=float, default=0.5)
    ap.add_argument("--out-date", default=None)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    date = args.out_date or datetime.datetime.now().date().isoformat()
    root = pathlib.Path(__file__).parent / "simulation" / date
    root.mkdir(parents=True, exist_ok=True)

    agent_fn = load_agent(args.ckpt, device=args.device,
                          temperature=args.temperature)
    tag = pathlib.Path(args.ckpt).stem

    # ID range unused by other generators (real=75M, lb928=800M,
    # starter=900M). Use 600M for selfplay, worker-spaced.
    id_base = 600_000_000 + args.worker_id * 300_000

    team_slug_2p = f"_synth_selfplay_{tag}_T{str(args.temperature).replace('.','p')}_2p"
    team_slug_4p = f"_synth_selfplay_{tag}_T{str(args.temperature).replace('.','p')}_4p"

    print(f"worker {args.worker_id}: {args.n_games} games  "
          f"temp={args.temperature}  ckpt={tag}", flush=True)

    for i in range(args.n_games):
        n_players = 4 if random.random() < args.four_player_prob else 2
        ep_id = id_base + i
        ts = datetime.datetime.now().isoformat(timespec="seconds")

        env = run_one_game(agent_fn, n_players)
        slug = team_slug_4p if n_players == 4 else team_slug_2p
        save_replay_from_env(env, root, slug, ep_id, ts)

        # Extract per-seat rewards + ship counts
        final_state = env.steps[-1]
        rewards = [s.reward if s.reward is not None else 0 for s in final_state]
        winner_seat = int(max(range(len(rewards)), key=lambda k: rewards[k]))
        # Ship counts from obs[0] (shared state)
        planets = final_state[0].observation.get("planets") or []
        fleets = final_state[0].observation.get("fleets") or []
        ships_by_seat = []
        for seat in range(n_players):
            s = sum(p[5] for p in planets if p[1] == seat) + sum(
                f[6] for f in fleets if f[1] == seat)
            ships_by_seat.append(s)

        append_monitor({
            "ts": ts,
            "worker": args.worker_id,
            "game_id": ep_id,
            "n_players": n_players,
            "steps": len(env.steps),
            "rewards": ";".join(str(r) for r in rewards),
            "winner_seat": winner_seat,
            "final_ships_by_seat": ";".join(str(s) for s in ships_by_seat),
        })

        if (i + 1) % 10 == 0:
            print(f"  w{args.worker_id} {i+1}/{args.n_games}  "
                  f"last: {n_players}p steps={len(env.steps)} "
                  f"winner={winner_seat} rewards={rewards}", flush=True)

    print(f"worker {args.worker_id} done: {args.n_games} games", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
