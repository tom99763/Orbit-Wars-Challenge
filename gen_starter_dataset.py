"""Generate synthetic replays by running `starter` vs {`starter`, `random`}
to augment the BC training set. Used for Phase-1 curriculum — ensures
the BC model learns the basic aggressive-expansion behavior that beats
the starter agent baseline.

Output lands in `simulation/<date>/_synth_starter_<game>/<episode_id>/`
so it's picked up by the existing `*/*/replay.json` glob in
parse_replays.py and featurize.py. Team name is recorded as
`synthetic_starter` so analytics can distinguish it.

Usage:
  python gen_starter_dataset.py --n-starter-v-starter 300 \
                                --n-starter-v-random 200 \
                                --out-date 2026-04-19
"""

import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from kaggle_environments import make


def run_game(agents: list[str]) -> dict:
    """Run one game, return a replay-dict in the same schema as the
    Kaggle /api/v1/competitions/episodes/<id>/replay endpoint."""
    env = make("orbit_wars", debug=False)
    env.run(agents)
    replay = env.toJSON()
    return replay


def save_replay(replay: dict, out_dir: pathlib.Path, team_slug: str, game_id: int):
    """Save under simulation/<date>/<team_slug>/<episode_id>/replay.json
    + episode.json + play.html. Uses our synthetic game id so it doesn't
    collide with real Kaggle ids."""
    # Use a synthetic episode_id that won't collide with Kaggle's
    # (Kaggle ids are currently 75M range; we'll use 900M+).
    synthetic_id = 900_000_000 + game_id
    replay["id"] = synthetic_id
    if "info" not in replay:
        replay["info"] = {}
    replay["info"]["EpisodeId"] = synthetic_id
    # Fake team names so parse_replays can produce trajectories
    n_players = len(replay.get("rewards") or [])
    replay["info"]["TeamNames"] = [f"{team_slug}_p{i}" for i in range(n_players)]
    replay["info"]["Agents"] = [{"Name": replay["info"]["TeamNames"][i],
                                 "ThumbnailUrl": None} for i in range(n_players)]

    ep_dir = out_dir / team_slug / str(synthetic_id)
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "replay.json").write_text(json.dumps(replay), encoding="utf-8")

    meta = {
        "episode_id": synthetic_id,
        "team_seen_by": team_slug,
        "rank_at_scrape": -1,
        "scrape_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "TeamNames": replay["info"]["TeamNames"],
        "Agents": replay["info"]["Agents"],
        "rewards": replay.get("rewards"),
        "statuses": replay.get("statuses"),
        "configuration": replay.get("configuration"),
        "list_episodes_entry": None,
        "synthetic": True,
    }
    (ep_dir / "episode.json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-starter-v-starter", type=int, default=300)
    ap.add_argument("--n-starter-v-random", type=int, default=200)
    ap.add_argument("--out-date", default=None,
                    help="Date folder under simulation/ (default: today)")
    args = ap.parse_args()

    date = args.out_date or datetime.datetime.now().date().isoformat()
    root = pathlib.Path(__file__).parent / "simulation" / date
    root.mkdir(parents=True, exist_ok=True)
    print(f"out: {root}", flush=True)

    game_id = 0
    for i in range(args.n_starter_v_starter):
        replay = run_game(["starter", "starter"])
        save_replay(replay, root, "_synth_starter_v_starter", game_id)
        game_id += 1
        if (i + 1) % 20 == 0:
            print(f"  starter v starter: {i+1}/{args.n_starter_v_starter}",
                  flush=True)

    for i in range(args.n_starter_v_random):
        replay = run_game(["starter", "random"])
        save_replay(replay, root, "_synth_starter_v_random", game_id)
        game_id += 1
        if (i + 1) % 20 == 0:
            print(f"  starter v random: {i+1}/{args.n_starter_v_random}",
                  flush=True)

    print(f"wrote {game_id} synthetic games under {root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
