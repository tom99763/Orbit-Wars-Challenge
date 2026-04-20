"""Generate lb-1200 self-play trajectories for imitation learning.

Runs lb-1200 vs lb-1200 in 2P and 4P matches across multiple workers, and
saves one npz per (episode, seat) in the same schema as build_offline_dataset
so existing loaders just work.

Usage:
  python training/generate_lb1200_trajectories.py \
      --games 10000 --workers 8 --four-player-prob 0.5 \
      --out-dir offline/lb1200_selfplay/2026-04-19
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from featurize import featurize_step
from training.lb928_agent import agent as lb1200_agent


GAMMA = 0.997
TOTAL_STEPS = 500
SUN_CENTER = 50.0


def compute_shaped_rewards(steps: list[dict], seat: int) -> list[float]:
    """Per-step shaped reward (same formula as current online_impala)."""
    shaped = [0.0]
    for t in range(1, len(steps)):
        pp = steps[t - 1].get("planets") or []
        cp = steps[t].get("planets") or []
        pf = steps[t - 1].get("fleets") or []
        cf = steps[t].get("fleets") or []

        p_n = sum(1 for p in pp if p[1] == seat)
        c_n = sum(1 for p in cp if p[1] == seat)
        p_prod = sum(p[6] for p in pp if p[1] == seat)
        c_prod = sum(p[6] for p in cp if p[1] == seat)
        p_enemy = sum(p[5] for p in pp if p[1] != seat and p[1] != -1) + sum(
            f[6] for f in pf if f[1] != seat)
        c_enemy = sum(p[5] for p in cp if p[1] != seat and p[1] != -1) + sum(
            f[6] for f in cf if f[1] != seat)

        pf_ids = {f[0]: f for f in pf if f[1] == seat}
        cf_ids = {f[0] for f in cf if f[1] == seat}
        sun_lost = 0
        for fid, f in pf_ids.items():
            if fid not in cf_ids:
                d = math.hypot(f[2] - SUN_CENTER, f[3] - SUN_CENTER)
                if d < 15.0:
                    sun_lost += f[6]

        dr = (
            0.10 * (c_n - p_n)
            + 0.02 * (c_prod - p_prod)
            - 0.10 * sun_lost / 100.0
            + 0.10 * max(0, p_enemy - c_enemy) / 50.0
        )
        shaped.append(max(-1.0, min(1.0, dr)))
    return shaped


def run_episode(args) -> list[dict]:
    """One worker call: play one game, return list of per-(episode,seat) dicts
    (each is the full npz-ready payload)."""
    ep_idx, four_player_prob = args
    from kaggle_environments import make

    n_players = 4 if random.random() < four_player_prob else 2
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)
    env.run([lb1200_agent] * n_players)

    steps = [
        {
            "step": t,
            "planets": list(s[0].observation.get("planets") or []),
            "fleets": list(s[0].observation.get("fleets") or []),
            "action": s[seat].action or [],
        }
        for t, s in enumerate(env.steps)
        for seat in [0]  # placeholder; replaced below
    ]
    # Re-index properly — need per-seat `action`
    steps = []
    for t, s in enumerate(env.steps):
        shared_obs = s[0].observation
        steps.append({
            "step": t,
            "planets": list(shared_obs.get("planets") or []),
            "fleets": list(shared_obs.get("fleets") or []),
            "_per_seat_action": [s[seat].action or [] for seat in range(n_players)],
        })

    # Terminal reward (env gives ±1)
    rewards = [s.reward if s.reward is not None else 0 for s in env.state]

    # Init planets + comets
    obs0 = env.state[0].observation
    ang_vel = float(obs0.get("angular_velocity") or 0.0)
    init_planets = list(obs0.get("initial_planets") or [])
    init_ids = {int(p[0]) for p in init_planets}
    comet_ids: set = set()
    for s in steps:
        for p in s["planets"]:
            if int(p[0]) not in init_ids:
                comet_ids.add(int(p[0]))

    results = []
    for seat in range(n_players):
        shape_r = compute_shaped_rewards(
            [{"planets": s["planets"], "fleets": s["fleets"]} for s in steps],
            seat,
        )
        T = len(shape_r)
        G = [0.0] * (T + 1)
        for t in range(T - 1, -1, -1):
            G[t] = shape_r[t] + GAMMA * G[t + 1]

        planets_arr = np.empty((T,), dtype=object)
        planet_ids_arr = np.empty((T,), dtype=object)
        planet_xy_arr = np.empty((T,), dtype=object)
        fleets_arr = np.empty((T,), dtype=object)
        globals_arr = np.zeros((T, 16), dtype=np.float32)
        owned_arr = np.empty((T,), dtype=object)
        src_arr = np.empty((T,), dtype=object)
        tgt_arr = np.empty((T,), dtype=object)
        bkt_arr = np.empty((T,), dtype=object)
        for t, s in enumerate(steps):
            step_dict = {
                "step": s["step"],
                "planets": s["planets"],
                "fleets": s["fleets"],
                "action": s["_per_seat_action"][seat],
                "my_total_ships": sum(p[5] for p in s["planets"] if p[1] == seat)
                                 + sum(f[6] for f in s["fleets"] if f[1] == seat),
                "enemy_total_ships": sum(p[5] for p in s["planets"]
                                         if p[1] != seat and p[1] != -1)
                                    + sum(f[6] for f in s["fleets"] if f[1] != seat),
                "my_planet_count": sum(1 for p in s["planets"] if p[1] == seat),
                "enemy_planet_count": sum(1 for p in s["planets"]
                                          if p[1] != seat and p[1] != -1),
                "neutral_planet_count": sum(1 for p in s["planets"] if p[1] == -1),
            }
            feat = featurize_step(step_dict, seat, ang_vel, n_players,
                                  init_planets, comet_ids)
            planets_arr[t] = feat["planets"]
            planet_ids_arr[t] = feat["planet_ids"]
            planet_xy_arr[t] = feat["planet_xy"]
            fleets_arr[t] = feat["fleets"]
            globals_arr[t] = feat["globals"]
            owned_arr[t] = feat["action_mask_owned"]
            src_arr[t] = feat["src_planet_idx"]
            tgt_arr[t] = feat["target_planet_idx"]
            bkt_arr[t] = feat["ships_bucket"]

        results.append({
            "ep_idx": ep_idx,
            "seat": seat,
            "n_players": n_players,
            "payload": {
                "planets": planets_arr,
                "planet_ids": planet_ids_arr,
                "planet_xy": planet_xy_arr,
                "fleets": fleets_arr,
                "globals": globals_arr,
                "action_mask_owned": owned_arr,
                "src_planet_idx": src_arr,
                "target_planet_idx": tgt_arr,
                "ships_bucket": bkt_arr,
                "shape_reward": np.array(
                    [shape_r[i] for i in range(T)], dtype=np.float32),
                "shape_return": np.array(G[:T], dtype=np.float32),
                "terminal_reward": np.float32(rewards[seat]),
                "is_winner": np.bool_(rewards[seat] == max(rewards) and max(rewards) > 0),
                "n_steps": np.int32(T),
            },
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--four-player-prob", type=float, default=0.5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start-idx", type=int, default=0,
                    help="Episode index offset (for resuming)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp.set_start_method("spawn", force=True)

    tasks = [(args.start_idx + i, args.four_player_prob)
             for i in range(args.games)]

    print(f"Generating {args.games} games with {args.workers} workers "
          f"→ {out_dir}", flush=True)

    t0 = time.time()
    done = 0
    seat_count = 0
    with mp.Pool(args.workers) as pool:
        for ep_results in pool.imap_unordered(run_episode, tasks):
            for item in ep_results:
                ep = item["ep_idx"]
                seat = item["seat"]
                np.savez(out_dir / f"lb1200_ep{ep:06d}_seat{seat}.npz",
                         **item["payload"])
                seat_count += 1
            done += 1
            if done % 20 == 0:
                dt = time.time() - t0
                rate = done / dt
                eta = (args.games - done) / max(rate, 1e-6)
                print(f"  {done}/{args.games} games done  "
                      f"seats={seat_count}  {rate:.2f} games/s  "
                      f"eta {eta/60:.1f} min", flush=True)

    print(f"done: {done} games, {seat_count} seat-trajectories in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
