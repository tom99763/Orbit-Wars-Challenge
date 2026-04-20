"""Convert trajectory pickles into training-ready feature arrays.

Input:  trajectories/<date>/*.pkl   (output of parse_replays.py)
Output: processed/<date>/<episode_id>__<agent_slug>.npz
        processed/<date>/index.csv  (summary of all processed files)

One .npz per (episode, agent). Each file holds ragged arrays — the number
of planets / fleets / owned-source-planets varies per step, so we keep
lists-of-arrays at the Python level via np.object and let the DataLoader
pad at batch time. For BC we only emit steps where the agent submitted
an action (skip "no-op" turns so the policy isn't taught to idle).

By default, only WINNER trajectories are featurised (controllable via
--keep-losers). This gives us a clean supervised signal: the actions we
train on all led to a +1 terminal reward.

================================================================
Feature schema (per step, unpadded)
================================================================

Arrays saved in the .npz (all dtypes numpy; N = # planets at that step,
F = # fleets, K = # my owned planets with garrison > 0):

Per-planet (N-rows):
  planets                float32 [N, 14]
      [owner_is_me, owner_is_enemy, owner_is_neutral,
       x_norm, y_norm,                          # (x-50)/50 etc
       radius,                                  # raw
       log_ships,                               # log1p(ships)/8
       prod_1, prod_2, prod_3, prod_4, prod_5,  # one-hot prod 1..5
       is_static,                               # orbital_r + r >= 50
       is_comet]                                # in comet_planet_ids
  planet_ids             int32   [N]            # env ids (for masking)
  planet_xy              float32 [N, 2]         # absolute [x, y]

Per-fleet (F-rows; F may be 0):
  fleets                 float32 [F, 9]
      [owner_is_me, owner_is_enemy, x_norm, y_norm,
       sin_a, cos_a, log_ships,
       from_planet_idx_norm,    # from_planet_id / max_planet_id
       eta_norm]                # rough time-to-impact / 50

Global scalars:
  globals                float32 [G]
      [step_norm,                                # step / 500
       angular_velocity,
       remaining_turns_norm,                     # (500-step)/500
       player_slot_0, player_slot_1, player_slot_2, player_slot_3,
       is_4p,
       n_my_planets / 20, n_enemy_planets / 20, n_neutral_planets / 20,
       log_my_total_ships,  log_enemy_total_ships,
       comet_window_active,                      # step in [45..55, 145..155, ...]
       rotation_phase_sin, rotation_phase_cos]   # angular_velocity * step

Action targets (per source planet the agent acted from — K rows):
  src_planet_idx         int32   [K]    # index into planets[] above
  target_planet_idx      int32   [K]    # index into planets[] above (-1 == pass, but we skip pass rows)
  ships_bucket           int32   [K]    # 0..3 → {25,50,75,100}%

Per-planet action mask (for PPO masking, emitted even for non-action
steps so the caller can train a "pass" head):
  action_mask_owned      bool    [N]    # planet is mine AND ships > 0
"""

import argparse
import csv
import math
import pathlib
import pickle
import re
import sys
import numpy as np

BOARD = 100.0
CENTER = 50.0
ROT_LIMIT = 50.0
PROD_MAX = 5
COMET_STEPS = {45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
               145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
               245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
               345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355,
               445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455}
SHIPS_BUCKETS = (0.25, 0.50, 0.75, 1.00)


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "unknown"


def ship_bucket_idx(num_ships: int, garrison: int) -> int:
    """Map an observed (num_ships, garrison) → one of 4 buckets."""
    if garrison <= 0:
        return 3
    frac = num_ships / garrison
    # Snap to nearest bucket
    return int(min(range(4), key=lambda i: abs(frac - SHIPS_BUCKETS[i])))


def nearest_target_index(src_planet, angle, planets, sun_safe=True):
    """Given a source planet and shot angle, find the planet the fleet is
    aimed at. We approximate by projecting each other planet onto the
    shot vector and keeping whoever has smallest perpendicular distance
    while being forward-of-source. Returns index into `planets` list, or
    None if no reasonable hit."""
    sx, sy = src_planet[2], src_planet[3]
    dx, dy = math.cos(angle), math.sin(angle)

    best_i = None
    best_score = float("inf")
    for i, p in enumerate(planets):
        if p[0] == src_planet[0]:
            continue
        vx, vy = p[2] - sx, p[3] - sy
        forward = vx * dx + vy * dy
        if forward <= 0:
            continue  # behind source
        perp = abs(vx * (-dy) + vy * dx)  # perpendicular distance
        # Require the perp distance to be within a few radii (fleet hit zone)
        if perp > p[4] + 2.0:
            continue
        score = forward + 5.0 * perp  # prefer close, aligned targets
        if score < best_score:
            best_score = score
            best_i = i
    return best_i


def fleet_eta_norm(fleet):
    """Rough time-to-impact / 50. Fleet speed scales with ships; use the
    env formula. Just a heuristic feature, not exact."""
    ships = max(1, fleet[6])
    speed = 1.0 + 5.0 * (math.log(ships) / math.log(1000)) ** 1.5
    speed = min(speed, 6.0)
    # Distance to board edge along angle as a ceiling
    x, y, ang = fleet[2], fleet[3], fleet[4]
    dx, dy = math.cos(ang), math.sin(ang)
    tx = (BOARD - x) / dx if dx > 1e-6 else (-x / dx if dx < -1e-6 else 1e6)
    ty = (BOARD - y) / dy if dy > 1e-6 else (-y / dy if dy < -1e-6 else 1e6)
    # Forward-positive
    t = min(abs(tx), abs(ty))
    return min(t / speed, 200.0) / 50.0


HISTORY_K = 3  # sliding window size for obs and action history
PLANET_DIM = 23 + HISTORY_K * 4   # +4 context + 4×K past-obs dims = 35
FLEET_DIM = 10
GLOBAL_DIM = 26 + HISTORY_K * 4   # +2 cumulative + 4×K past-action dims = 38


def featurize_step(step_dict, agent_idx, angular_velocity, n_players, initial_planets,
                   comet_ids=None, last_actions_by_planet=None,
                   cumulative_stats=None,
                   obs_history=None, action_history=None):
    planets = step_dict["planets"]
    fleets = step_dict["fleets"]
    action = step_dict["action"] or []
    step = step_dict["step"]
    comet_ids = set(comet_ids or [])
    last_actions_by_planet = last_actions_by_planet or {}
    cumulative_stats = cumulative_stats or {"total_ships_sent": 0,
                                             "total_actions": 0}
    # obs_history: list of up to K previous raw obs dicts (oldest→newest), each
    # must have {"planets": list, "step": int}. If len<K, older slots pad zeros.
    obs_history = obs_history or []
    # action_history: list of up to K recent (src_pid, tgt_pid, bkt_idx, turn)
    # tuples for this seat (oldest→newest).
    action_history = action_history or []

    # Initial-position lookup for orbit prediction
    init_by_id = {int(p[0]): p for p in (initial_planets or [])}

    def _predict_future_xy(pid, x, y, r, dt):
        """Predict (x, y) at time step+dt using lb-1200's formula: rotate around
        center at angular_velocity · dt, preserving orbit radius. Static planets
        (outer) don't move."""
        init = init_by_id.get(int(pid))
        if init is None:
            return x, y
        _, _, ix, iy, iradius, _, _ = init
        orb_r = math.hypot(ix - CENTER, iy - CENTER)
        if orb_r + iradius >= ROT_LIMIT:
            return x, y  # static
        cur_ang = math.atan2(y - CENTER, x - CENTER)
        new_ang = cur_ang + angular_velocity * dt
        return CENTER + orb_r * math.cos(new_ang), CENTER + orb_r * math.sin(new_ang)

    N = len(planets)
    planet_feat = np.zeros((N, PLANET_DIM), dtype=np.float32)
    planet_ids = np.zeros((N,), dtype=np.int32)
    planet_xy = np.zeros((N, 2), dtype=np.float32)
    action_mask_owned = np.zeros((N,), dtype=bool)
    for i, p in enumerate(planets):
        pid, owner, x, y, r, ships, prod = p
        planet_ids[i] = pid
        planet_xy[i] = (x, y)
        planet_feat[i, 0] = 1.0 if owner == agent_idx else 0.0
        planet_feat[i, 1] = 1.0 if (owner != agent_idx and owner != -1) else 0.0
        planet_feat[i, 2] = 1.0 if owner == -1 else 0.0
        planet_feat[i, 3] = (x - CENTER) / CENTER
        planet_feat[i, 4] = (y - CENTER) / CENTER
        planet_feat[i, 5] = r
        planet_feat[i, 6] = math.log1p(max(0, ships)) / 8.0
        if 1 <= prod <= PROD_MAX:
            planet_feat[i, 6 + prod] = 1.0
        orb_r = math.hypot(x - CENTER, y - CENTER)
        is_static = (orb_r + r >= ROT_LIMIT)
        planet_feat[i, 12] = 1.0 if is_static else 0.0
        planet_feat[i, 13] = 1.0 if pid in comet_ids else 0.0
        # Physics additions (lb-1200 borrow):
        # [14,15] future (x,y) at t+10 normalized; [16,17] at t+20.
        fx10, fy10 = _predict_future_xy(pid, x, y, r, 10)
        fx20, fy20 = _predict_future_xy(pid, x, y, r, 20)
        planet_feat[i, 14] = (fx10 - CENTER) / CENTER
        planet_feat[i, 15] = (fy10 - CENTER) / CENTER
        planet_feat[i, 16] = (fx20 - CENTER) / CENTER
        planet_feat[i, 17] = (fy20 - CENTER) / CENTER
        # [18] is-rotating (complement of static; makes it explicit for model)
        planet_feat[i, 18] = 0.0 if is_static else 1.0
        # [19-22] Context history: what did this planet do previously?
        last = last_actions_by_planet.get(int(pid))
        if last is not None:
            last_tgt_pid, last_bkt_idx, last_turn, n_actions = last
            turns_ago = max(0, step - last_turn)
            # Decay: 0 = just acted, 1 ≈ ≥20 turns ago
            planet_feat[i, 19] = math.exp(-turns_ago / 10.0)
            # Last target as normalized idx (use max_pid computed later; fallback)
            planet_feat[i, 20] = float(last_tgt_pid) / 40.0 if last_tgt_pid >= 0 else 0.0
            planet_feat[i, 21] = last_bkt_idx / 3.0
            planet_feat[i, 22] = math.log1p(n_actions) / 5.0
        # else all 0 — no history yet
        # [23-34] Sliding window: this planet's state at t-1, t-2, t-3
        # 4 dims per timestep: (was_present, was_mine, was_enemy, ships_log)
        for k in range(HISTORY_K):
            # obs_history stored oldest→newest; we want newest at k=0
            idx = len(obs_history) - 1 - k
            base = 23 + k * 4
            if idx < 0:
                continue  # keep zeros — no history yet
            prev_obs = obs_history[idx]
            prev_planets = prev_obs.get("planets") or []
            # Find this planet by pid
            match = None
            for pp in prev_planets:
                if int(pp[0]) == int(pid):
                    match = pp
                    break
            if match is None:
                continue  # planet didn't exist then — stay zero
            planet_feat[i, base + 0] = 1.0  # was_present
            planet_feat[i, base + 1] = 1.0 if match[1] == agent_idx else 0.0
            planet_feat[i, base + 2] = 1.0 if (match[1] != agent_idx and match[1] != -1) else 0.0
            planet_feat[i, base + 3] = math.log1p(max(0, match[5])) / 8.0
        if owner == agent_idx and ships > 0:
            action_mask_owned[i] = True

    # Fleets (lb-1200 addition: inferred target planet id)
    F = len(fleets)
    fleet_feat = np.zeros((F, FLEET_DIM), dtype=np.float32)
    max_pid = max(planet_ids.max(initial=1), 1)
    for i, f in enumerate(fleets):
        fid, owner, x, y, ang, from_id, ships = f
        fleet_feat[i, 0] = 1.0 if owner == agent_idx else 0.0
        fleet_feat[i, 1] = 1.0 if owner != agent_idx else 0.0
        fleet_feat[i, 2] = (x - CENTER) / CENTER
        fleet_feat[i, 3] = (y - CENTER) / CENTER
        fleet_feat[i, 4] = math.sin(ang)
        fleet_feat[i, 5] = math.cos(ang)
        fleet_feat[i, 6] = math.log1p(max(0, ships)) / 8.0
        fleet_feat[i, 7] = from_id / max_pid
        fleet_feat[i, 8] = fleet_eta_norm(f)
        # [9] inferred target planet (from angle + position)
        src_proxy = (fid, owner, x, y, 0.0, ships, 0)  # pseudo-planet struct
        tgt_i = nearest_target_index(src_proxy, ang, planets)
        fleet_feat[i, 9] = (planet_ids[tgt_i] / max_pid) if tgt_i is not None else 0.0

    # Globals (lb-1200 addition: phase one-hot, per-player strength)
    G = GLOBAL_DIM
    g = np.zeros((G,), dtype=np.float32)
    g[0] = step / 500.0
    g[1] = angular_velocity
    g[2] = max(0.0, (500 - step) / 500.0)
    g[3 + agent_idx] = 1.0  # slots 3..6
    g[7] = 1.0 if n_players == 4 else 0.0
    my_planets = int(planet_feat[:, 0].sum())
    en_planets = int(planet_feat[:, 1].sum())
    nu_planets = int(planet_feat[:, 2].sum())
    g[8] = my_planets / 20.0
    g[9] = en_planets / 20.0
    g[10] = nu_planets / 20.0
    my_total = step_dict.get("my_total_ships", 0)
    en_total = step_dict.get("enemy_total_ships", 0)
    g[11] = math.log1p(my_total) / 8.0
    g[12] = math.log1p(en_total) / 8.0
    g[13] = 1.0 if step in COMET_STEPS else 0.0
    phase = angular_velocity * step
    g[14] = math.sin(phase)
    g[15] = math.cos(phase)
    # Phase one-hot (lb-1200 uses turn-count phases): early/opening/mid/late/very_late
    remaining = 500 - step
    if step < 40: phase_idx = 0        # early
    elif step < 80: phase_idx = 1      # opening
    elif remaining > 60: phase_idx = 2 # mid
    elif remaining > 25: phase_idx = 3 # late
    else: phase_idx = 4                # very_late
    g[16 + phase_idx] = 1.0            # slots 16..20
    # Per-player strength stats: weakest enemy / strongest enemy / my rank
    strengths = [0.0] * max(n_players, 1)
    for p in planets:
        if p[1] != -1 and 0 <= p[1] < len(strengths):
            strengths[p[1]] += p[5]
    for f in fleets:
        if 0 <= f[1] < len(strengths):
            strengths[f[1]] += f[6]
    others = [strengths[o] for o in range(n_players) if o != agent_idx]
    if others:
        weak = min(others); strong = max(others)
        # normalize via log to keep in a bounded range
        g[21] = math.log1p(weak) / 8.0
        g[22] = math.log1p(strong) / 8.0
        # my rank among all players (0 = worst, 1 = best)
        sorted_s = sorted(strengths)
        rank = sorted_s.index(strengths[agent_idx]) if strengths[agent_idx] in sorted_s else 0
        g[23] = rank / max(1, n_players - 1)
    else:
        g[21] = g[22] = g[23] = 0.0
    # Cumulative context globals: total ships sent so far + action count
    g[24] = math.log1p(cumulative_stats.get("total_ships_sent", 0)) / 10.0
    g[25] = math.log1p(cumulative_stats.get("total_actions", 0)) / 6.0
    # [26-37] Sliding window of last K actions (globally): 4 dims each
    # (src_pid_norm, tgt_pid_norm, bucket/3, turns_ago_decay)
    for k in range(HISTORY_K):
        idx = len(action_history) - 1 - k  # newest at k=0
        base = 26 + k * 4
        if idx < 0:
            continue  # zeros = no action yet
        src_pid, tgt_pid, bkt_idx, act_turn = action_history[idx]
        g[base + 0] = float(src_pid) / 40.0 if src_pid >= 0 else 0.0
        g[base + 1] = float(tgt_pid) / 40.0 if tgt_pid >= 0 else 0.0
        g[base + 2] = bkt_idx / 3.0
        g[base + 3] = math.exp(-max(0, step - act_turn) / 10.0)

    # Action targets (only non-pass actions)
    src_idx, tgt_idx, bucket = [], [], []
    pid_to_idx = {int(pid): i for i, pid in enumerate(planet_ids)}
    for move in action:
        if len(move) != 3:
            continue
        from_id, angle, ships = move
        src_i = pid_to_idx.get(int(from_id))
        if src_i is None:
            continue
        src_planet = planets[src_i]
        # Garrison is the count BEFORE this move was executed, but in our
        # stored step we already hold planets pre-move (observation at
        # step t is what the agent saw before submitting action). Good.
        garrison = src_planet[5] + ships  # undo the env's pre-subtract
        tgt_i = nearest_target_index(src_planet, angle, planets)
        if tgt_i is None:
            continue
        src_idx.append(src_i)
        tgt_idx.append(tgt_i)
        bucket.append(ship_bucket_idx(ships, max(1, garrison)))

    return {
        "planets": planet_feat,
        "planet_ids": planet_ids,
        "planet_xy": planet_xy,
        "fleets": fleet_feat,
        "globals": g,
        "action_mask_owned": action_mask_owned,
        "src_planet_idx": np.array(src_idx, dtype=np.int32),
        "target_planet_idx": np.array(tgt_idx, dtype=np.int32),
        "ships_bucket": np.array(bucket, dtype=np.int32),
    }


def featurize_trajectory(traj: dict, min_step: int = 1, skip_noop_steps: bool = True):
    """Yield per-step feature dicts for one trajectory."""
    ang_vel = traj.get("angular_velocity") or 0.0
    n_players = traj["n_players"]
    agent_idx = traj["agent_idx"]
    init_planets = traj.get("initial_planets") or []

    # Comet ids accumulate as the episode progresses. We re-derive by
    # diffing planet IDs against the initial set — any planet not in
    # `initial_planets` that appears in a later step is a comet.
    init_ids = {int(p[0]) for p in init_planets}
    comet_ids: set[int] = set()

    for step in traj["steps"]:
        if step["step"] < min_step:
            continue
        # Update comet set
        for p in step["planets"]:
            if int(p[0]) not in init_ids:
                comet_ids.add(int(p[0]))

        if skip_noop_steps and not (step["action"] or []):
            continue

        yield featurize_step(step, agent_idx, ang_vel, n_players,
                             init_planets, comet_ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", required=True,
                    help="Input: trajectories/<date>/ directory containing *.pkl + index.csv")
    ap.add_argument("--out-dir", required=True,
                    help="Output: processed/<date>/")
    ap.add_argument("--keep-losers", action="store_true",
                    help="Include non-winning trajectories (default: winners only)")
    ap.add_argument("--keep-noops", action="store_true",
                    help="Keep steps where the agent passed (default: drop)")
    args = ap.parse_args()

    traj_dir = pathlib.Path(args.traj_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_path = traj_dir / "index.csv"
    if not idx_path.exists():
        print(f"ERROR: missing {idx_path}", file=sys.stderr)
        return 1

    # Filter the index to winners-only unless --keep-losers
    rows = list(csv.DictReader(idx_path.open()))
    if not args.keep_losers:
        rows = [r for r in rows if r["winner"] == "True"]
    print(f"{len(rows)} trajectories to featurise", flush=True)

    summary_rows = []
    total_examples = 0
    for row in rows:
        pkl_path = traj_dir / row["file"]
        if not pkl_path.exists():
            print(f"  (missing) {pkl_path}", flush=True)
            continue
        with open(pkl_path, "rb") as f:
            traj = pickle.load(f)

        examples = list(featurize_trajectory(
            traj, skip_noop_steps=not args.keep_noops))
        if not examples:
            continue

        # Stack ragged arrays via object dtype (each step may have N varying)
        keys = examples[0].keys()
        bundle = {k: np.array([ex[k] for ex in examples], dtype=object)
                  for k in keys}
        # globals and scalar-count ones are uniform — stack proper
        bundle["globals"] = np.stack([ex["globals"] for ex in examples])

        out_name = f"{traj['episode_id']}__{slugify(traj['team_name'])}__a{traj['agent_idx']}.npz"
        out_path = out_dir / out_name
        np.savez_compressed(out_path, **bundle)

        summary_rows.append({
            "episode_id": traj["episode_id"],
            "team_name": traj["team_name"],
            "agent_idx": traj["agent_idx"],
            "winner": traj["winner"],
            "n_players": traj["n_players"],
            "n_examples": len(examples),
            "n_steps": traj["n_steps"],
            "file": out_name,
        })
        total_examples += len(examples)

    # Index CSV
    if summary_rows:
        with (out_dir / "index.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print(f"wrote {len(summary_rows)} files, {total_examples} examples "
          f"→ {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
