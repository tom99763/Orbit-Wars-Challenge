"""Generic self-play dataset generator.

Runs many games of (agent_a, agent_b) for arbitrary agent pairs.
Used to scale up Phase-1 BC data beyond the starter / lb-928 baselines.

Supports:
  - "starter", "random"  — builtin kaggle agents
  - "lb928"              — training/lb928_agent.py
  - "ckpt:<path>:T=<f>"  — load an OrbitAgent checkpoint at temp T

Example:
  python gen_selfplay_dataset.py \
      --matchup lb928:lb928:200 \
      --matchup "ckpt:training/checkpoints/bc_v2.pt:T=1.0":"ckpt:training/checkpoints/bc_v2.pt:T=1.0":200 \
      --matchup "ckpt:training/checkpoints/bc_v2.pt:T=1.0":lb928:100 \
      --worker-id 0
"""

import argparse
import datetime
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from kaggle_environments import make


def resolve_agent(spec: str):
    """Return (callable, slug) from a string spec."""
    if spec == "random":
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent
        return random_agent, "random"
    if spec == "starter":
        from kaggle_environments.envs.orbit_wars.orbit_wars import starter_agent
        return starter_agent, "starter"
    if spec == "lb928":
        from training.lb928_agent import agent as lb928
        return lb928, "lb928"
    if spec.startswith("ckpt:"):
        # "ckpt:<path>[:T=<float>]"
        parts = spec.split(":")
        path = parts[1]
        temp = 0.0
        for p in parts[2:]:
            if p.startswith("T="):
                temp = float(p[2:])
        from training.agent import load_agent
        fn = load_agent(path, device="cpu", temperature=temp)
        stem = pathlib.Path(path).stem
        slug = f"{stem}_T{str(temp).replace('.','p')}"
        return fn, slug
    raise ValueError(f"unknown agent spec: {spec}")


def run_game(agents):
    env = make("orbit_wars", debug=False)
    env.run([agents[0][0], agents[1][0]])
    return env.toJSON()


def save_replay(replay, out_dir, team_slug, game_id, id_base):
    ep_id = id_base + game_id
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
        "rank_at_scrape": -1,
        "scrape_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "TeamNames": replay["info"]["TeamNames"],
        "Agents": replay["info"]["Agents"],
        "rewards": replay.get("rewards"), "statuses": replay.get("statuses"),
        "configuration": replay.get("configuration"), "synthetic": True,
    }, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matchup", action="append", required=True,
                    help='Format "<spec_a>:<spec_b>:<N>" — repeatable')
    ap.add_argument("--out-date", default=None)
    ap.add_argument("--worker-id", type=int, default=0)
    args = ap.parse_args()

    date = args.out_date or datetime.datetime.now().date().isoformat()
    root = pathlib.Path(__file__).parent / "simulation" / date
    root.mkdir(parents=True, exist_ok=True)
    print(f"out: {root}  worker={args.worker_id}", flush=True)

    # ID base: pick a range unlikely to collide with real episodes (75M)
    # or earlier synth runs (800M, 900M).
    id_base = 700_000_000 + args.worker_id * 200_000

    game_id = 0
    for mspec in args.matchup:
        # "a:b:N" but specs may themselves contain colons → right-split
        parts = mspec.rsplit(":", 1)
        ab = parts[0]; N = int(parts[1])
        # a:b split — but ckpt specs contain colons. Find the split between
        # two agents by recognising the boundary at the middle colon:
        # we accept that ckpt specs won't contain embedded bare "lb928" etc.
        # Simple rule: agent_b is the last subspec; split on the colon that
        # precedes a known agent prefix or "ckpt:".
        # Safer: require `;` as the separator in multi-colon case.
        if ";" in ab:
            a_spec, b_spec = ab.split(";", 1)
        else:
            # Single-colon case (no ckpt specs)
            a_spec, b_spec = ab.split(":", 1)

        a = resolve_agent(a_spec)
        b = resolve_agent(b_spec)
        team_slug = f"_synth_{a[1]}_v_{b[1]}"
        print(f"  matchup: {a[1]} v {b[1]} × {N}", flush=True)

        for i in range(N):
            r = run_game([a, b])
            save_replay(r, root, team_slug, game_id, id_base)
            game_id += 1
            if (i + 1) % 20 == 0:
                print(f"    {a[1]} v {b[1]}: {i+1}/{N}", flush=True)

    print(f"wrote {game_id} games under {root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
