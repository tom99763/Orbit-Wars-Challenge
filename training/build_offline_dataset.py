"""Build offline RL dataset from trajectory pickles.

Reads trajectories/{date}/*.pkl, computes per-step shaped reward + MC discounted
return, and writes per-episode npz shards containing the same feature schema
as the BC npz plus reward/return fields.

Output schema (per episode, per agent):
    planets           object array [T] of [N_t, 14] float32
    planet_ids        object array [T] of [N_t]    int32
    planet_xy         object array [T] of [N_t, 2] float32
    fleets            object array [T] of [F_t, 9] float32
    globals           [T, 16] float32
    action_mask_owned object array [T] of [N_t]    bool
    src_planet_idx    object array [T] of [K_t]    int32
    target_planet_idx object array [T] of [K_t]    int32
    ships_bucket      object array [T] of [K_t]    int32
    shape_reward      [T] float32     — per-step shaped reward (pre-discount)
    shape_return      [T] float32     — MC discounted sum G_t = r_t + γ G_{t+1}
    terminal_reward   scalar float32  — env terminal reward for this seat
    is_winner         scalar bool
    n_steps           scalar int32

Usage:
    python training/build_offline_dataset.py \
        --traj-dir trajectories/2026-04-19 \
        --out-dir offline/2026-04-19 \
        [--include-synth]   # include synth_selfplay_* trajectories too
"""
from __future__ import annotations

import argparse
import math
import pathlib
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from featurize import featurize_step


SUN_R = 10.0
CENTER = 50.0
GAMMA = 0.997


def compute_shaped_rewards(steps: list[dict], seat: int) -> list[float]:
    """Same shaped reward as online_impala.compute_shaped_rewards but operates
    on our trajectory `steps` list format (list of dicts with planets/fleets)."""
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
                d = math.hypot(f[2] - CENTER, f[3] - CENTER)
                if d < 15.0:
                    sun_lost += f[6]

        dr = (
            0.10 * (c_n - p_n)
            + 0.02 * (c_prod - p_prod)
            - 0.10 * sun_lost / 100.0
            + 0.05 * max(0, p_enemy - c_enemy) / 100.0
        )
        shaped.append(max(-1.0, min(1.0, dr)))
    return shaped


def process_trajectory(traj: dict) -> dict | None:
    seat = traj.get("agent_idx")
    if seat is None:
        return None
    steps = traj.get("steps") or []
    T = len(steps)
    if T < 2:
        return None

    ang_vel = float(traj.get("angular_velocity") or 0.0)
    init_planets = traj.get("initial_planets") or []
    init_ids = {int(p[0]) for p in init_planets}
    n_players = int(traj.get("n_players") or 2)

    # Precompute comet IDs (planets that appear but are not in initial)
    comet_ids: set = set()
    for s in steps:
        for p in s.get("planets") or []:
            if int(p[0]) not in init_ids:
                comet_ids.add(int(p[0]))

    shape_r = compute_shaped_rewards(steps, seat)
    # MC discounted return
    G = [0.0] * (T + 1)
    for t in range(T - 1, -1, -1):
        G[t] = shape_r[t] + GAMMA * G[t + 1]

    # Per-step features (skip steps with no action — pure observation padding)
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
        feat = featurize_step(s, seat, ang_vel, n_players, init_planets, comet_ids)
        planets_arr[t] = feat["planets"]
        planet_ids_arr[t] = feat["planet_ids"]
        planet_xy_arr[t] = feat["planet_xy"]
        fleets_arr[t] = feat["fleets"]
        globals_arr[t] = feat["globals"]
        owned_arr[t] = feat["action_mask_owned"]
        src_arr[t] = feat["src_planet_idx"]
        tgt_arr[t] = feat["target_planet_idx"]
        bkt_arr[t] = feat["ships_bucket"]

    return {
        "planets": planets_arr,
        "planet_ids": planet_ids_arr,
        "planet_xy": planet_xy_arr,
        "fleets": fleets_arr,
        "globals": globals_arr,
        "action_mask_owned": owned_arr,
        "src_planet_idx": src_arr,
        "target_planet_idx": tgt_arr,
        "ships_bucket": bkt_arr,
        "shape_reward": np.array(shape_r, dtype=np.float32),
        "shape_return": np.array(G[:T], dtype=np.float32),
        "terminal_reward": np.float32(traj.get("final_reward", 0)),
        "is_winner": np.bool_(traj.get("winner", False)),
        "n_steps": np.int32(T),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-synth", action="store_true",
                    help="Include synth_selfplay_* trajectories")
    args = ap.parse_args()

    traj_dir = pathlib.Path(args.traj_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(traj_dir.glob("*.pkl"))
    if not args.include_synth:
        pkls = [p for p in pkls if "synth" not in p.name]
    print(f"processing {len(pkls)} trajectories from {traj_dir}", flush=True)

    t0 = time.time()
    ok = skipped = 0
    winners = 0
    for i, pkl in enumerate(pkls):
        try:
            with pkl.open("rb") as f:
                traj = pickle.load(f)
        except Exception as e:
            skipped += 1
            continue
        out = process_trajectory(traj)
        if out is None:
            skipped += 1
            continue
        np.savez(out_dir / pkl.with_suffix(".npz").name, **out)
        ok += 1
        if bool(out["is_winner"]):
            winners += 1
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{len(pkls)}  ok={ok}  skip={skipped}  "
                  f"winners={winners}  [{dt:.0f}s]", flush=True)

    dt = time.time() - t0
    print(f"done: {ok}/{len(pkls)} ok, {skipped} skipped, "
          f"{winners} winners ({winners/max(ok,1):.0%}) in {dt:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
