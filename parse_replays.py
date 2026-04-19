"""Parse scraped Orbit Wars replay JSONs into clean per-(episode, agent) trajectories.

Input layout (current, one folder per episode, accumulated per day):
  simulation/<date>/<team_slug>/<episode_id>/replay.json
  simulation/<date>/<team_slug>/<episode_id>/episode.json

Output layout (one file per (episode, agent); all episodes from a day
pooled together):
  trajectories/<date>/<episode_id>__<team_slug>.pkl
  trajectories/<date>/index.csv   (summary across all trajectories)

Legacy layout `simulation/<date>/<HH-MM>/rank*/replay_<episode_id>.json`
is also recognised for back-compat.

Trajectory schema (pickle file):

    {
        "episode_id": int,
        "team_name": str,                 # this agent's team
        "agent_idx": int,                 # 0-based player slot in the game
        "opponents": [str, ...],          # other teams in the game
        "n_players": int,
        "final_reward": int,              # scalar terminal reward from replay.rewards
        "final_status": str,              # "DONE" / "ERROR" / ...
        "winner": bool,                   # this agent had argmax final_reward
        "config": {..., "seed": int},     # full game configuration
        "angular_velocity": float,
        "initial_planets": [[id, owner, x, y, r, ships, prod], ...],
        "n_steps": int,
        "steps": [
            {
                "step": int,
                "planets": [[id, owner, x, y, r, ships, prod], ...],
                "fleets":  [[id, owner, x, y, angle, from_id, ships], ...],
                "action":  [[from_id, angle, ships], ...],   # what THIS agent submitted
                "reward":  float,                             # per-step reward
                "status":  str,
                "done":    bool,                              # terminal flag (last step OR non-ACTIVE status)
                # convenience scalars for quick analytics / reward shaping:
                "my_ships_on_planets":     int,
                "my_ships_in_fleets":      int,
                "my_total_ships":          int,
                "my_planet_count":         int,
                "enemy_ships_on_planets":  int,
                "enemy_ships_in_fleets":   int,
                "enemy_total_ships":       int,
                "enemy_planet_count":      int,
                "neutral_planet_count":    int,
                "num_fleets_on_board":     int,
                "num_actions":             int,
            },
            ...
        ],
    }
"""

import argparse
import csv
import json
import pathlib
import pickle
import re
import sys


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_") or "unknown"


def parse_replay(replay_path: pathlib.Path) -> list[dict]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))

    info = replay.get("info") or {}
    team_names = info.get("TeamNames") or []
    ep_id = info.get("EpisodeId") or replay.get("id")

    config = replay.get("configuration") or {}
    rewards = replay.get("rewards") or []
    statuses = replay.get("statuses") or []
    steps = replay.get("steps") or []
    n_steps = len(steps)
    if n_steps == 0:
        return []
    n_players = len(steps[0])

    # initial_planets / angular_velocity are read from step 0's observation
    first_obs = (steps[0][0] or {}).get("observation") or {}
    initial_planets = first_obs.get("initial_planets") or []
    angular_velocity = first_obs.get("angular_velocity")

    max_reward = max(rewards) if rewards else None

    trajectories = []
    for agent_idx in range(n_players):
        team_name = team_names[agent_idx] if agent_idx < len(team_names) else f"agent{agent_idx}"
        traj_steps = []

        for t, step in enumerate(steps):
            a = step[agent_idx] if agent_idx < len(step) else {}
            obs = a.get("observation") or {}
            planets = obs.get("planets") or []
            fleets = obs.get("fleets") or []
            action = a.get("action") or []
            status = a.get("status")
            reward = a.get("reward", 0) or 0

            my_planets = [p for p in planets if p[1] == agent_idx]
            my_fleets = [f for f in fleets if f[1] == agent_idx]
            enemy_planets = [
                p for p in planets if p[1] not in (agent_idx, -1)
            ]
            enemy_fleets = [f for f in fleets if f[1] != agent_idx]
            neutral_planets = [p for p in planets if p[1] == -1]

            my_ships_planets = sum(p[5] for p in my_planets)
            my_ships_fleets = sum(f[6] for f in my_fleets)
            enemy_ships_planets = sum(p[5] for p in enemy_planets)
            enemy_ships_fleets = sum(f[6] for f in enemy_fleets)

            traj_steps.append(
                {
                    "step": t,
                    "planets": planets,
                    "fleets": fleets,
                    "action": action,
                    "reward": reward,
                    "status": status,
                    "done": (t == n_steps - 1) or (status not in (None, "ACTIVE")),
                    "my_ships_on_planets": my_ships_planets,
                    "my_ships_in_fleets": my_ships_fleets,
                    "my_total_ships": my_ships_planets + my_ships_fleets,
                    "my_planet_count": len(my_planets),
                    "enemy_ships_on_planets": enemy_ships_planets,
                    "enemy_ships_in_fleets": enemy_ships_fleets,
                    "enemy_total_ships": enemy_ships_planets + enemy_ships_fleets,
                    "enemy_planet_count": len(enemy_planets),
                    "neutral_planet_count": len(neutral_planets),
                    "num_fleets_on_board": len(fleets),
                    "num_actions": len(action),
                }
            )

        final_reward = rewards[agent_idx] if agent_idx < len(rewards) else None
        trajectories.append(
            {
                "episode_id": ep_id,
                "team_name": team_name,
                "agent_idx": agent_idx,
                "opponents": [
                    team_names[i] for i in range(len(team_names)) if i != agent_idx
                ],
                "n_players": n_players,
                "final_reward": final_reward,
                "final_status": statuses[agent_idx] if agent_idx < len(statuses) else None,
                "winner": (max_reward is not None and final_reward == max_reward),
                "config": config,
                "angular_velocity": angular_velocity,
                "initial_planets": initial_planets,
                "n_steps": n_steps,
                "steps": traj_steps,
            }
        )

    return trajectories


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-dir", required=True,
                    help="Scraped simulation folder containing rank*/replay_*.json")
    ap.add_argument("--out-dir", required=True,
                    help="Trajectory output folder")
    args = ap.parse_args()

    sim_dir = pathlib.Path(args.sim_dir)
    out_dir = pathlib.Path(args.out_dir)

    if not sim_dir.exists():
        print(f"ERROR: sim-dir does not exist: {sim_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique replays (dedup by episode id; 4-player games appear under
    # multiple team folders but contain identical replay JSON).
    unique: dict[int, pathlib.Path] = {}
    rank_map: dict[int, list[str]] = {}

    # Current layout: <team_slug>/<episode_id>/replay.json
    for p in sorted(sim_dir.glob("*/*/replay.json")):
        try:
            ep_id = int(p.parent.name)
        except ValueError:
            continue
        unique.setdefault(ep_id, p)
        rank_map.setdefault(ep_id, []).append(p.parent.parent.name)

    # Legacy layout: rank*/replay_<ep>.json
    for p in sorted(sim_dir.glob("rank*/replay_*.json")):
        m = re.search(r"replay_(\d+)\.json$", p.name)
        if not m:
            continue
        ep_id = int(m.group(1))
        unique.setdefault(ep_id, p)
        rank_map.setdefault(ep_id, []).append(p.parent.name)

    print(f"found {len(unique)} unique episodes across {sim_dir}", flush=True)

    index_rows = []
    for ep_id, replay_path in unique.items():
        try:
            trajs = parse_replay(replay_path)
        except Exception as e:
            print(f"  failed to parse {replay_path.name}: {e}", flush=True)
            continue

        for traj in trajs:
            slug = slugify(traj["team_name"])
            out_name = f"{traj['episode_id']}__{slug}.pkl"
            out_path = out_dir / out_name
            with open(out_path, "wb") as f:
                pickle.dump(traj, f, protocol=pickle.HIGHEST_PROTOCOL)

            index_rows.append(
                {
                    "episode_id": traj["episode_id"],
                    "team_name": traj["team_name"],
                    "agent_idx": traj["agent_idx"],
                    "n_players": traj["n_players"],
                    "n_steps": traj["n_steps"],
                    "final_reward": traj["final_reward"],
                    "final_status": traj["final_status"],
                    "winner": traj["winner"],
                    "seed": (traj["config"] or {}).get("seed"),
                    "source_rank_folders": ";".join(rank_map.get(ep_id, [])),
                    "file": out_name,
                }
            )

        print(f"  episode {ep_id}: {len(trajs)} trajectories "
              f"(seen in {rank_map.get(ep_id, [])})", flush=True)

    if index_rows:
        idx_path = out_dir / "index.csv"
        with open(idx_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader()
            w.writerows(index_rows)
        print(f"index → {idx_path} ({len(index_rows)} rows)", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
