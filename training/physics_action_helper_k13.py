"""k13 factored action helper — mode × fraction action space.

Action = (mode_idx, frac_idx):
  mode:  {0:pass, 1:expand, 2:attack, 3:reinforce, 4:denial}
  frac:  ship fraction bins {5%, 15%, 30%, 50%, 65%, 80%, 95%, 100%}

Physics helper picks target planet based on mode, safely aims with prediction,
and scales ships by fraction bin.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from typing import Any, Optional

import numpy as np

from training.lb1200_agent import (
    aim_with_prediction, segment_hits_sun, fleet_speed, Planet,
)


def _build_fleet_arrivals(fleets_raw, planets, my_player):
    """Scan raw fleet list and return {planet_id: [enemy_eta_min, friendly_ships]}.

    fleets_raw: list of [id, owner, x, y, angle, from_id, ships]
    Uses ray-sphere intersection (same as lb1200 fleet_target_planet).
    """
    arrivals = {}
    if not fleets_raw:
        return arrivals
    for f in fleets_raw:
        owner, fx, fy, fangle, fships = int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[6])
        spd = max(fleet_speed(fships), 0.1)
        dir_x = math.cos(fangle)
        dir_y = math.sin(fangle)
        for p in planets:
            dx = p.x - fx; dy = p.y - fy
            proj = dx * dir_x + dy * dir_y
            if proj <= 0:
                continue
            perp_sq = dx * dx + dy * dy - proj * proj
            r = getattr(p, 'radius', 1.5)
            if perp_sq >= r * r:
                continue
            eta = proj / spd
            pid = p.id
            if pid not in arrivals:
                arrivals[pid] = [float('inf'), 0]
            if owner != my_player:
                if eta < arrivals[pid][0]:
                    arrivals[pid][0] = eta
            else:
                arrivals[pid][1] += int(fships)
            break  # fleet targets only one planet
    return arrivals

N_MODES = 5
N_FRACS = 8
MODE_NAMES = ["pass", "expand", "attack", "reinforce", "denial"]
FRACTIONS = [0.05, 0.15, 0.30, 0.50, 0.65, 0.80, 0.95, 1.00]

MAX_DISTANCE = 50.0
MIN_SHIPS_TO_COMMIT = 2
MIN_DEFENSE_KEEP = 1

# Target-selection top-K constants
TOP_K_TARGETS = 4   # max candidates exposed to the model
CAND_FEAT_DIM = 10  # per-candidate feature vector dimension
# Features: [prod/5, dist/50, garrison/200, is_static, prod_over_dist/2,
#             eta_norm, projected_garrison/200,
#             friendly_en_route/200, enemy_eta_norm, race_margin[-1,1]]
_FLEET_MAX_SPEED = 6.0  # approximate max fleet speed (ships ~1000)
_MAX_ETA = MAX_DISTANCE / _FLEET_MAX_SPEED  # normalizer for ETA features


def _reachable(src: Planet, tgt: Planet) -> bool:
    if src.id == tgt.id:
        return False
    if segment_hits_sun(src.x, src.y, tgt.x, tgt.y):
        return False
    return math.hypot(tgt.x - src.x, tgt.y - src.y) <= MAX_DISTANCE


def _find_target(src: Planet, world, my_player: int, mode: int) -> Optional[Planet]:
    """Return the chosen target planet for this mode, or None if no valid target."""
    planets = world.planets
    if mode == 0:
        return None

    if mode == 1:  # expand — nearest neutral
        candidates = [p for p in planets if p.owner == -1 and _reachable(src, p)]
    elif mode == 2:  # attack — nearest enemy
        candidates = [p for p in planets
                      if p.owner != my_player and p.owner != -1 and _reachable(src, p)]
    elif mode == 3:  # reinforce — friendly with lowest ships (under threat)
        candidates = [p for p in planets
                      if p.owner == my_player and p.id != src.id and _reachable(src, p)]
    elif mode == 4:  # denial — planet between me and enemy
        enemies = [p for p in planets if p.owner != my_player and p.owner != -1]
        if not enemies:
            return None
        candidates = []
        for p in planets:
            if p.owner == my_player or not _reachable(src, p):
                continue
            # score by "between-ness": close to src and close to an enemy
            for e in enemies:
                me_to_p = math.hypot(p.x - src.x, p.y - src.y)
                p_to_e  = math.hypot(e.x - p.x,   e.y - p.y)
                me_to_e = math.hypot(e.x - src.x, e.y - src.y) + 1e-6
                detour  = (me_to_p + p_to_e) - me_to_e
                if detour < 15.0:
                    candidates.append(p)
                    break
    else:
        return None

    if not candidates:
        return None

    if mode == 3:
        # reinforce: lowest garrison first (most in need)
        candidates.sort(key=lambda p: (p.ships, math.hypot(p.x - src.x, p.y - src.y)))
    else:
        # expand/attack/denial: nearest first
        candidates.sort(key=lambda p: math.hypot(p.x - src.x, p.y - src.y))
    return candidates[0]


def _build_one_action(src: Planet, tgt: Planet, ships: int,
                      world, my_player: int) -> Optional[list]:
    if ships < MIN_SHIPS_TO_COMMIT or ships > src.ships - MIN_DEFENSE_KEEP:
        return None
    aim = aim_with_prediction(
        src, tgt, ships, world.initial_by_id, world.ang_vel,
        comets=world.comets, comet_ids=world.comet_ids,
    )
    if aim is None:
        return None
    angle = aim[0] if isinstance(aim, tuple) else aim
    return [int(src.id), float(angle), int(ships)]


def compute_mode_mask(src: Planet, world, my_player: int) -> list:
    """Return [N_MODES] boolean list: True = mode is valid for this src planet.

    Pass is always valid. Other modes valid iff _find_target would return non-None.
    Computing all 5 at once is cheap (same pool of planets).
    """
    mask = [False] * N_MODES
    mask[0] = True  # pass always valid

    planets = world.planets
    have_neutral = any(p.owner == -1 and _reachable(src, p) for p in planets)
    have_enemy   = any(p.owner != my_player and p.owner != -1 and _reachable(src, p)
                       for p in planets)
    have_friend  = any(p.owner == my_player and p.id != src.id and _reachable(src, p)
                       for p in planets)

    mask[1] = have_neutral
    mask[2] = have_enemy
    mask[3] = have_friend
    # denial: any enemy exists (not necessarily reachable) + any reachable non-friendly
    any_enemy = any(p.owner != my_player and p.owner != -1 for p in planets)
    mask[4] = any_enemy and (have_neutral or have_enemy)
    return mask


def get_top_k_candidates(src: Planet, world, my_player: int, mode: int,
                         k: int = TOP_K_TARGETS,
                         fleets_raw=None, committed=None):
    """Return (cands, feats, n_valid) for top-K candidates sorted by prod/(dist+1).

    feats: np.ndarray (k, CAND_FEAT_DIM=10) — zero-padded beyond n_valid rows
    fleets_raw: raw obs["fleets"] list for fleet race features
    committed:  dict {planet_id -> ships_already_dispatched} for coordination
    Features: [prod/5, dist/50, garrison/200, is_static, prod_over_dist/2,
               eta_norm, projected_garrison/200,
               friendly_en_route/200, enemy_eta_norm, race_margin[-1,1]]
    """
    planets = world.planets
    if mode == 0 or mode > 4:
        return [], np.zeros((k, CAND_FEAT_DIM), dtype=np.float32), 0

    if mode == 1:
        cands = [p for p in planets if p.owner == -1 and _reachable(src, p)]
    elif mode == 2:
        cands = [p for p in planets
                 if p.owner != my_player and p.owner != -1 and _reachable(src, p)]
    elif mode == 3:
        cands = [p for p in planets
                 if p.owner == my_player and p.id != src.id and _reachable(src, p)]
    elif mode == 4:
        enemies = [p for p in planets if p.owner != my_player and p.owner != -1]
        if not enemies:
            return [], np.zeros((k, CAND_FEAT_DIM), dtype=np.float32), 0
        cands = []
        for p in planets:
            if p.owner == my_player or not _reachable(src, p):
                continue
            for e in enemies:
                me_to_p = math.hypot(p.x - src.x, p.y - src.y)
                p_to_e  = math.hypot(e.x - p.x,   e.y - p.y)
                me_to_e = math.hypot(e.x - src.x, e.y - src.y) + 1e-6
                if (me_to_p + p_to_e) - me_to_e < 15.0:
                    cands.append(p); break
    else:
        return [], np.zeros((k, CAND_FEAT_DIM), dtype=np.float32), 0

    if not cands:
        return [], np.zeros((k, CAND_FEAT_DIM), dtype=np.float32), 0

    # Sort by production / (dist+1) descending — prefer high-value, nearby targets
    cands.sort(key=lambda p: p.production / (math.hypot(p.x - src.x, p.y - src.y) + 1.0),
               reverse=True)
    cands = cands[:k]
    n_valid = len(cands)

    # Precompute fleet arrival info once for all candidates
    arrivals = _build_fleet_arrivals(fleets_raw, world.planets, my_player) if fleets_raw else {}

    feats = np.zeros((k, CAND_FEAT_DIM), dtype=np.float32)
    for ci, p in enumerate(cands):
        dist = math.hypot(p.x - src.x, p.y - src.y)
        r_from_center = math.hypot(p.x - 50.0, p.y - 50.0)
        is_static = 1.0 if r_from_center >= 45.0 else 0.0
        my_eta = dist / _FLEET_MAX_SPEED
        projected = min((p.ships + p.production * my_eta) / 200.0, 2.0)

        arr = arrivals.get(p.id, [float('inf'), 0])
        enemy_eta = arr[0] if arr[0] < float('inf') else _MAX_ETA
        friendly_ships = arr[1] + (committed.get(p.id, 0) if committed else 0)
        race_margin = max(-1.0, min(1.0, (enemy_eta - my_eta) / _MAX_ETA))

        feats[ci] = [
            p.production / 5.0,
            dist / MAX_DISTANCE,
            p.ships / 200.0,
            is_static,
            (p.production / (dist + 1.0)) / 2.0,
            my_eta / _MAX_ETA,                      # normalized ETA
            projected,                              # projected garrison at ETA
            min(friendly_ships / 200.0, 1.0),      # friendly ships en route (incl. committed)
            min(enemy_eta / _MAX_ETA, 1.0),         # enemy nearest fleet ETA norm
            race_margin,                            # >0 I arrive first, <0 enemy wins
        ]
    return cands, feats, n_valid


def materialize_with_targets(picks, world, my_player: int) -> list:
    """picks = [(planet_id, mode_idx, frac_idx, target_pid), ...] — target pre-selected."""
    planet_by_id = {p.id: p for p in world.planets}
    actions = []
    for pid, mode, frac_idx, target_pid in picks:
        src = planet_by_id.get(int(pid))
        if src is None or src.owner != my_player or mode == 0:
            continue
        tgt = planet_by_id.get(int(target_pid))
        if tgt is None:
            continue
        frac = FRACTIONS[int(frac_idx)]
        ships = max(1, int(src.ships * frac))
        ships = min(ships, src.ships - MIN_DEFENSE_KEEP)
        if ships < MIN_SHIPS_TO_COMMIT:
            continue
        act = _build_one_action(src, tgt, ships, world, my_player)
        if act is not None:
            actions.append(act)
    return actions


def materialize_joint_action(picks, world, my_player: int) -> list:
    """Given picks = [(planet_id, mode_idx, frac_idx), ...], return env action list."""
    planet_by_id = {p.id: p for p in world.planets}
    actions = []
    for pid, mode, frac_idx in picks:
        src = planet_by_id.get(int(pid))
        if src is None or src.owner != my_player:
            continue
        if mode == 0:  # pass
            continue
        tgt = _find_target(src, world, my_player, int(mode))
        if tgt is None:
            continue
        frac = FRACTIONS[int(frac_idx)]
        ships = max(1, int(src.ships * frac))
        ships = min(ships, src.ships - MIN_DEFENSE_KEEP)
        if ships < MIN_SHIPS_TO_COMMIT:
            continue
        act = _build_one_action(src, tgt, ships, world, my_player)
        if act is not None:
            actions.append(act)
    return actions
