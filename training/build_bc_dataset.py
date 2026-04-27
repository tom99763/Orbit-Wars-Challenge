"""Convert parsed expert trajectories into k14 BC-aux-loss dataset.

Reads `trajectories/<date>/<time>/*.pkl` (output of parse_replays.py),
runs the per-seat featurization that matches the trainer's online path,
and emits one npz per (episode, seat) under
`training/bc_data/<date>/<time>/`.

Each shard contains all steps from one expert seat in one game:
  planets        (T,)  object  → (P_t, PLANET_DIM)  float32 per step
  fleets         (T,)  object  → (F_t, FLEET_DIM)   float32 per step
  globals        (T, GLOBAL_DIM)                    float32
  spatial        (T, N_SPATIAL_CHANNELS, GRID, GRID) float32
  planet_ids     (T,)  object  → (P_t,)             int64
  mode_label     (T,)  object  → (P_t,)             int8
                   -1 = ignore (not owned at this step)
                    0 = pass   (owned, no action)
                  1-4 = expand/attack/reinforce/denial
  frac_label     (T,)  object  → (P_t,)             int8
                   -1 = no frac (mode 0, unowned, or unlabelable)
                  0-7 = quantized fraction bucket
  ep_id, seat, is_winner, n_steps  scalars

The trainer pads at batch time. mode_label = -1 / frac_label = -1 are
masked out of the BC CE loss (no penalty for "we couldn't recover the
intent of this expert action").

Usage:
  python training/build_bc_dataset.py \
      --traj-dir trajectories/2026-04-27/08-52 \
      --out-dir  training/bc_data/2026-04-27/08-52
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from featurize import (
    FLEET_DIM,
    GLOBAL_DIM,
    HISTORY_K,
    PLANET_DIM,
    featurize_step,
    nearest_target_index,
    ship_bucket_idx,
)
from training.dual_stream_model import N_SPATIAL_CHANNELS, rasterize_obs
from training.expert_action_labeler import label_turn, summarize_labels
from training.physics_picker_k13_ppo import GRID


def _per_seat_init() -> dict:
    return {
        "obs_history": collections.deque(maxlen=HISTORY_K),
        "action_history": collections.deque(maxlen=HISTORY_K),
        "last_actions_by_planet": {},
        "cumulative_stats": {"total_ships_sent": 0, "total_actions": 0},
    }


def process_trajectory(traj: dict) -> dict | None:
    """Featurize + label one trajectory pickle. Returns npz-ready payload
    or None on irrecoverable error."""
    seat = int(traj["agent_idx"])
    n_players = int(traj["n_players"])
    init_planets = traj.get("initial_planets") or []
    ang_vel = float(traj.get("angular_velocity") or 0.0)
    init_ids = {int(p[0]) for p in init_planets}

    steps = traj.get("steps") or []
    if not steps:
        return None

    T = len(steps)
    planets_arr = np.empty((T,), dtype=object)
    fleets_arr = np.empty((T,), dtype=object)
    planet_ids_arr = np.empty((T,), dtype=object)
    mode_arr = np.empty((T,), dtype=object)
    frac_arr = np.empty((T,), dtype=object)
    globals_arr = np.zeros((T, GLOBAL_DIM), dtype=np.float32)
    spatial_arr = np.zeros((T, N_SPATIAL_CHANNELS, GRID, GRID), dtype=np.float32)

    sess = _per_seat_init()
    comet_ids: set = set()

    n_active_actions = 0
    n_active_labeled = 0
    n_passes = 0
    mode_hist = [0] * 5
    frac_hist = [0] * 8

    for t, s in enumerate(steps):
        raw_planets = list(s.get("planets") or [])
        raw_fleets = list(s.get("fleets") or [])
        raw_action = list(s.get("action") or [])

        for p in raw_planets:
            pid = int(p[0])
            if pid not in init_ids:
                comet_ids.add(pid)

        # Featurize identically to the trainer's online path
        step_dict = {
            "step": int(s.get("step", t)),
            "planets": raw_planets,
            "fleets": raw_fleets,
            "action": raw_action,
            "my_total_ships": int(s.get("my_total_ships", 0)),
            "enemy_total_ships": int(s.get("enemy_total_ships", 0)),
            "my_planet_count": int(s.get("my_planet_count", 0)),
            "enemy_planet_count": int(s.get("enemy_planet_count", 0)),
            "neutral_planet_count": int(s.get("neutral_planet_count", 0)),
        }
        feat = featurize_step(
            step_dict, seat, ang_vel, n_players, init_planets, comet_ids,
            last_actions_by_planet=sess["last_actions_by_planet"],
            cumulative_stats=sess["cumulative_stats"],
            obs_history=list(sess["obs_history"]),
            action_history=list(sess["action_history"]),
        )

        obs_for_spatial = {
            "planets": raw_planets, "fleets": raw_fleets,
            "player": seat,
            "step": int(s.get("step", t)),
            "angular_velocity": ang_vel,
            "initial_planets": init_planets,
            "comet_planet_ids": list(comet_ids),
        }
        spatial = rasterize_obs(obs_for_spatial, seat, grid=GRID)

        # Label expert actions in k14 factored space
        obs_for_label = {
            "planets": raw_planets, "fleets": raw_fleets,
            "angular_velocity": ang_vel,
            "initial_planets": init_planets,
            "comet_planet_ids": list(comet_ids),
            "player": seat,
        }
        labeled = label_turn(raw_action, obs_for_label, my_player=seat)
        labeled_by_pid = {lp.pid: lp for lp in labeled}

        planet_ids = np.asarray(feat.get("planet_ids", []), dtype=np.int64)
        P = planet_ids.shape[0]
        mode_label = np.full((P,), -1, dtype=np.int8)
        frac_label = np.full((P,), -1, dtype=np.int8)

        # Quick lookup: which planets are owned by `seat` at this step
        owners_by_id = {int(p[0]): int(p[1]) for p in raw_planets}
        # Set of source planets whose action we COULDN'T label (drop those)
        unlabeled_active_srcs: set[int] = set()
        for mv in raw_action:
            if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
                continue
            try:
                src_id = int(mv[0])
            except (TypeError, ValueError):
                continue
            if src_id not in labeled_by_pid:
                unlabeled_active_srcs.add(src_id)

        for i, pid in enumerate(planet_ids.tolist()):
            if owners_by_id.get(int(pid), -1) != seat:
                continue   # not owned → leave -1
            if int(pid) in labeled_by_pid:
                lp = labeled_by_pid[int(pid)]
                mode_label[i] = lp.mode_idx
                frac_label[i] = lp.frac_idx
                mode_hist[lp.mode_idx] += 1
                frac_hist[lp.frac_idx] += 1
                n_active_labeled += 1
            elif int(pid) in unlabeled_active_srcs:
                # Owned, took an action, but we couldn't recover the
                # factored label → ignore (leave -1) instead of mislabeling
                pass
            else:
                mode_label[i] = 0   # owned + no action = pass
                mode_hist[0] += 1
                n_passes += 1

        for mv in raw_action:
            if isinstance(mv, (list, tuple)) and len(mv) == 3:
                n_active_actions += 1

        planets_arr[t] = feat["planets"].astype(np.float32, copy=False)
        fleets_arr[t] = feat["fleets"].astype(np.float32, copy=False)
        planet_ids_arr[t] = planet_ids
        mode_arr[t] = mode_label
        frac_arr[t] = frac_label
        globals_arr[t] = feat["globals"].astype(np.float32, copy=False)
        spatial_arr[t] = spatial.astype(np.float32, copy=False)

        # Update history with whatever the expert actually did (raw),
        # so featurize sees the same context the policy would have seen
        # had it perfectly imitated. This is the standard BC convention.
        for mv in raw_action:
            if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
                continue
            try:
                from_id = int(mv[0])
                angle = float(mv[1])
                ships = int(mv[2])
            except (TypeError, ValueError):
                continue
            src_p = next((p for p in raw_planets if int(p[0]) == from_id), None)
            if src_p is None:
                continue
            ti = nearest_target_index(src_p, angle, raw_planets)
            tpid = int(raw_planets[ti][0]) if ti is not None else -1
            garrison = int(src_p[5]) + ships
            bi = ship_bucket_idx(ships, max(1, garrison))
            prev = sess["last_actions_by_planet"].get(from_id, (-1, 0, -1, 0))
            sess["last_actions_by_planet"][from_id] = (tpid, bi, t, prev[3] + 1)
            sess["cumulative_stats"]["total_ships_sent"] += ships
            sess["cumulative_stats"]["total_actions"] += 1
            sess["action_history"].append((from_id, tpid, bi, t))
        sess["obs_history"].append({"planets": raw_planets, "step": t})

    payload = {
        "planets": planets_arr,
        "fleets": fleets_arr,
        "planet_ids": planet_ids_arr,
        "mode_label": mode_arr,
        "frac_label": frac_arr,
        "globals": globals_arr,
        "spatial": spatial_arr,
        "ep_id": np.int64(traj.get("episode_id", -1)),
        "seat": np.int8(seat),
        "is_winner": np.bool_(traj.get("winner", False)),
        "n_steps": np.int32(T),
    }

    label_yield = (n_active_labeled / max(1, n_active_actions))
    return {
        "payload": payload,
        "stats": {
            "T": T,
            "active_actions": n_active_actions,
            "active_labeled": n_active_labeled,
            "passes": n_passes,
            "label_yield": label_yield,
            "mode_hist": mode_hist,
            "frac_hist": frac_hist,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True,
                    help="Directory of trajectory pickles "
                         "(e.g. trajectories/2026-04-27/08-52).")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for BC dataset shards.")
    ap.add_argument("--winners-only", action="store_true",
                    help="Skip non-winning trajectories (standard BC practice).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N trajectories (for smoke).")
    args = ap.parse_args()

    traj_dir = pathlib.Path(args.traj_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(traj_dir.glob("*.pkl"))
    if args.limit is not None:
        pkls = pkls[: args.limit]
    print(f"found {len(pkls)} trajectory pickles in {traj_dir}", flush=True)

    t0 = time.time()
    n_done = 0
    n_skipped = 0
    total_active = 0
    total_labeled = 0
    total_passes = 0
    agg_mode = [0] * 5
    agg_frac = [0] * 8

    for pkl_path in pkls:
        try:
            with pkl_path.open("rb") as f:
                traj = pickle.load(f)
        except Exception as e:
            print(f"  skip {pkl_path.name}: load error {e}", flush=True)
            n_skipped += 1
            continue

        if args.winners_only and not traj.get("winner", False):
            n_skipped += 1
            continue

        result = process_trajectory(traj)
        if result is None:
            n_skipped += 1
            continue

        ep = traj.get("episode_id", -1)
        seat = traj.get("agent_idx", 0)
        team = (traj.get("team_name") or "unknown").replace(" ", "_")[:40]
        out = out_dir / f"bc_{ep}__{team}__seat{seat}.npz"
        np.savez(out, **result["payload"])

        st = result["stats"]
        total_active += st["active_actions"]
        total_labeled += st["active_labeled"]
        total_passes += st["passes"]
        for i in range(5):
            agg_mode[i] += st["mode_hist"][i]
        for i in range(8):
            agg_frac[i] += st["frac_hist"][i]
        n_done += 1
        if n_done % 5 == 0 or n_done == len(pkls):
            print(f"  processed {n_done}/{len(pkls)}  "
                  f"yield={total_labeled}/{total_active}="
                  f"{total_labeled/max(1,total_active):.1%}", flush=True)

    print(f"\nDone in {time.time()-t0:.1f}s  written={n_done}  skipped={n_skipped}",
          flush=True)
    print(f"  active actions:  {total_active}")
    print(f"  labeled:         {total_labeled} ({total_labeled/max(1,total_active):.1%})")
    print(f"  pass labels:     {total_passes}")
    print(f"  mode hist (pass/expand/attack/reinforce/denial): {agg_mode}")
    print(f"  frac hist (5/15/30/50/65/80/95/100%):            {agg_frac}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
