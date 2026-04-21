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

from training.lb1200_agent import (
    aim_with_prediction, segment_hits_sun, fleet_speed, Planet,
)

N_MODES = 5
N_FRACS = 8
MODE_NAMES = ["pass", "expand", "attack", "reinforce", "denial"]
FRACTIONS = [0.05, 0.15, 0.30, 0.50, 0.65, 0.80, 0.95, 1.00]

MAX_DISTANCE = 50.0
MIN_SHIPS_TO_COMMIT = 2
MIN_DEFENSE_KEEP = 1


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
