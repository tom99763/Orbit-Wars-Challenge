"""Generate synthetic lb-928 vs {lb-928, random} replays for BC distillation.

Mirrors gen_starter_dataset.py but uses the rules-based lb-928 planner
extracted into training/lb928_agent.py.
"""

import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from kaggle_environments import make
from training.lb928_agent import agent as lb928_agent


def run_game(agents: list) -> dict:
    env = make("orbit_wars", debug=False)
    env.run(agents)
    return env.toJSON()


def save_replay(replay: dict, out_dir: pathlib.Path, team_slug: str,
                game_id: int, id_base: int = 800_000_000):
    synthetic_id = id_base + game_id
    replay["id"] = synthetic_id
    if "info" not in replay:
        replay["info"] = {}
    replay["info"]["EpisodeId"] = synthetic_id
    n_players = len(replay.get("rewards") or [])
    replay["info"]["TeamNames"] = [f"{team_slug}_p{i}" for i in range(n_players)]
    replay["info"]["Agents"] = [{"Name": replay["info"]["TeamNames"][i],
                                 "ThumbnailUrl": None} for i in range(n_players)]

    ep_dir = out_dir / team_slug / str(synthetic_id)
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    meta = {
        "episode_id": synthetic_id, "team_seen_by": team_slug,
        "rank_at_scrape": -1,
        "scrape_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "TeamNames": replay["info"]["TeamNames"],
        "Agents": replay["info"]["Agents"],
        "rewards": replay.get("rewards"),
        "statuses": replay.get("statuses"),
        "configuration": replay.get("configuration"),
        "synthetic": True,
    }
    (ep_dir / "episode.json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lb928-v-lb928", type=int, default=400)
    ap.add_argument("--n-lb928-v-random", type=int, default=150)
    ap.add_argument("--n-lb928-v-starter", type=int, default=150)
    ap.add_argument("--out-date", default=None)
    ap.add_argument("--worker-id", type=int, default=0,
                    help="Parallel worker index (0..N-1); separates id ranges")
    args = ap.parse_args()
    id_base = 800_000_000 + args.worker_id * 100_000

    date = args.out_date or datetime.datetime.now().date().isoformat()
    root = pathlib.Path(__file__).parent / "simulation" / date
    root.mkdir(parents=True, exist_ok=True)
    print(f"out: {root}", flush=True)

    game_id = 0
    for i in range(args.n_lb928_v_lb928):
        r = run_game([lb928_agent, lb928_agent])
        save_replay(r, root, "_synth_lb928_v_lb928", game_id, id_base=id_base)
        game_id += 1
        if (i + 1) % 20 == 0:
            print(f"  lb928 v lb928: {i+1}/{args.n_lb928_v_lb928}", flush=True)

    for i in range(args.n_lb928_v_random):
        r = run_game([lb928_agent, "random"])
        save_replay(r, root, "_synth_lb928_v_random", game_id, id_base=id_base)
        game_id += 1
        if (i + 1) % 20 == 0:
            print(f"  lb928 v random: {i+1}/{args.n_lb928_v_random}", flush=True)

    for i in range(args.n_lb928_v_starter):
        r = run_game([lb928_agent, "starter"])
        save_replay(r, root, "_synth_lb928_v_starter", game_id, id_base=id_base)
        game_id += 1
        if (i + 1) % 20 == 0:
            print(f"  lb928 v starter: {i+1}/{args.n_lb928_v_starter}", flush=True)

    print(f"wrote {game_id} lb-928 synthetic games under {root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
