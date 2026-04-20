"""Physics-informed action candidate generator.

Instead of full action space (per-planet tgt × bkt ≈ 160^P joint actions),
use physics to produce K_PER_SOURCE=6 candidates per owned planet:

  C0: pass (do nothing)
  C1: minimum-ships capture of nearest capturable planet
  C2: half-commit capture of nearest capturable
  C3: all-in capture of nearest capturable (keep 1 for defense)
  C4: minimum-ships capture of highest-production capturable
  C5: reinforce nearest friendly in danger (if any)

Each candidate (except C0) goes through physics:
  - segment_hits_sun check
  - aim_with_prediction for orbital-aware intercept angle
  - sufficient-ships check (ships >= target defense + margin)

Joint action = for each owned planet, NN picks one of C0-C5.
Joint action space = 6^P (e.g., 6^10 = 60M), much smaller than full (160^10).

Benefits:
  - Bounded per-planet branching (6)
  - Physics primitives encoded, not learned
  - Still expressive: covers pass / cheap capture / heavy commit / reinforce
  - NN learns state-dependent selection

Limitations:
  - Can't do "send X ships at an obscure angle" (pruned by physics)
  - Bound by what physics generator considers viable
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from typing import Any, Optional

from training.lb1200_agent import (
    aim_with_prediction, segment_hits_sun,
    simulate_planet_timeline, fleet_target_planet, Fleet, Planet,
)


K_PER_SOURCE = 6  # candidates per owned planet (C0-C5)

# Thresholds
MAX_DISTANCE = 50.0           # cap for "reachable" target
MIN_SHIPS_TO_COMMIT = 3        # source needs at least this to attack
SAFETY_MARGIN = 2              # extra ships over needed for reliable capture
MIN_DEFENSE_KEEP = 1           # always keep at least 1 on source


CANDIDATE_NAMES = [
    "pass",
    "min_nearest",
    "half_nearest",
    "allin_nearest",
    "min_high_prod",
    "reinforce",
]


def _is_reachable(src: Planet, target: Planet) -> bool:
    """Sun-safe + within range."""
    if src.id == target.id:
        return False
    if segment_hits_sun(src.x, src.y, target.x, target.y):
        return False
    dist = math.hypot(target.x - src.x, target.y - src.y)
    return dist <= MAX_DISTANCE


def _estimate_capture_cost(target: Planet, world: Any, travel_turns: int) -> int:
    """Minimum ships needed to capture target, accounting for production during travel."""
    if target.owner == -1:
        # Neutral — ships grow 0/turn (neutral planets don't produce)
        base = max(1, int(target.ships))
    else:
        # Enemy planet grows during travel
        base = max(1, int(target.ships) + int(target.production) * travel_turns)
    return base + SAFETY_MARGIN


def _find_nearest_capturable(src: Planet, world: Any, my_player: int,
                              high_prod_only: bool = False) -> Optional[Planet]:
    """Find nearest planet that isn't ours, reachable, and within capture cost."""
    best = None
    best_score = -1e9
    for p in world.planets:
        if p.owner == my_player:
            continue
        if not _is_reachable(src, p):
            continue
        # Score: closer + higher production
        dist = math.hypot(p.x - src.x, p.y - src.y)
        prod_score = int(p.production) * (3 if high_prod_only else 1)
        score = prod_score - dist * 0.2
        if score > best_score:
            best_score = score
            best = p
    return best


def _find_friendly_in_danger(src: Planet, world: Any, my_player: int) -> Optional[Planet]:
    """Find a nearby friendly planet predicted to fall — candidate for reinforcement."""
    best = None; best_dist = 1e9
    for p in world.planets:
        if p.owner != my_player or p.id == src.id:
            continue
        if not _is_reachable(src, p):
            continue
        # Check if this planet holds (use lb-1200's simulate_planet_timeline)
        # For simplicity, skip expensive simulation — use heuristic: low-ships friendly near enemies
        nearby_enemies = sum(1 for q in world.planets
                              if q.owner not in (my_player, -1)
                              and math.hypot(q.x - p.x, q.y - p.y) <= 30)
        if nearby_enemies == 0 or p.ships > 15:
            continue
        # candidate
        dist = math.hypot(p.x - src.x, p.y - src.y)
        if dist < best_dist:
            best_dist = dist
            best = p
    return best


def _compute_travel_turns(src: Planet, target: Planet, ships: int) -> int:
    """Approx turns for fleet of `ships` from src to target (straight line)."""
    from training.lb1200_agent import fleet_speed
    dist = math.hypot(target.x - src.x, target.y - src.y)
    speed = fleet_speed(max(1, int(ships)))
    return max(1, int(math.ceil(dist / max(speed, 1e-6))))


def generate_per_source_candidates(
    src: Planet, world: Any, my_player: int,
    comets: Optional[list] = None,
) -> list[Optional[list]]:
    """Return list of K_PER_SOURCE candidate env-actions (or None for pass).

    Each non-None candidate is [src_id, angle, ships] — ready for env.step.
    Returns list of length K_PER_SOURCE.
    """
    candidates: list[Optional[list]] = [None] * K_PER_SOURCE
    candidates[0] = None  # C0 pass

    # Can we launch from this planet at all?
    if src.ships < MIN_SHIPS_TO_COMMIT:
        return candidates   # all passes (insufficient ships)

    comets = comets or []
    comet_ids = set(getattr(world, "comet_ids", set()) or set())
    initial_by_id = getattr(world, "initial_by_id", {}) or {}
    ang_vel = getattr(world, "ang_vel", 0.0) or 0.0

    # Find nearest capturable
    nearest = _find_nearest_capturable(src, world, my_player, high_prod_only=False)
    if nearest is not None:
        # Approximate ships needed (will refine via actual target position)
        rough_ships = min(int(src.ships) - MIN_DEFENSE_KEEP, 20)
        travel = _compute_travel_turns(src, nearest, rough_ships)
        needed = _estimate_capture_cost(nearest, world, travel)

        if src.ships >= needed + MIN_DEFENSE_KEEP:
            # C1: min_nearest — minimum viable capture
            try:
                aim = aim_with_prediction(src, nearest, needed, initial_by_id,
                                           ang_vel, comets, comet_ids)
                if aim is not None:
                    angle = aim[0] if isinstance(aim, (tuple, list)) else aim
                    candidates[1] = [src.id, float(angle), int(needed)]
            except Exception:
                pass
            # C2: half_nearest — commit 50% of available garrison
            half_ships = max(needed, min(int(src.ships) // 2,
                                          int(src.ships) - MIN_DEFENSE_KEEP))
            try:
                aim = aim_with_prediction(src, nearest, half_ships, initial_by_id,
                                           ang_vel, comets, comet_ids)
                if aim is not None:
                    angle = aim[0] if isinstance(aim, (tuple, list)) else aim
                    candidates[2] = [src.id, float(angle), int(half_ships)]
            except Exception:
                pass
            # C3: all-in (keep 1 defense)
            allin_ships = max(needed, int(src.ships) - MIN_DEFENSE_KEEP)
            try:
                aim = aim_with_prediction(src, nearest, allin_ships, initial_by_id,
                                           ang_vel, comets, comet_ids)
                if aim is not None:
                    angle = aim[0] if isinstance(aim, (tuple, list)) else aim
                    candidates[3] = [src.id, float(angle), int(allin_ships)]
            except Exception:
                pass

    # C4: high-production target
    high_prod = _find_nearest_capturable(src, world, my_player, high_prod_only=True)
    if high_prod is not None and high_prod.id != (nearest.id if nearest else -1):
        travel = _compute_travel_turns(src, high_prod, 10)
        needed = _estimate_capture_cost(high_prod, world, travel)
        if src.ships >= needed + MIN_DEFENSE_KEEP:
            try:
                aim = aim_with_prediction(src, high_prod, needed, initial_by_id,
                                           ang_vel, comets, comet_ids)
                if aim is not None:
                    angle = aim[0] if isinstance(aim, (tuple, list)) else aim
                    candidates[4] = [src.id, float(angle), int(needed)]
            except Exception:
                pass

    # C5: reinforce friendly in danger
    friendly = _find_friendly_in_danger(src, world, my_player)
    if friendly is not None:
        reinf_ships = min(int(src.ships) // 3, int(src.ships) - MIN_DEFENSE_KEEP)
        if reinf_ships >= MIN_SHIPS_TO_COMMIT:
            try:
                aim = aim_with_prediction(src, friendly, reinf_ships, initial_by_id,
                                           ang_vel, comets, comet_ids)
                if aim is not None:
                    angle = aim[0] if isinstance(aim, (tuple, list)) else aim
                    candidates[5] = [src.id, float(angle), int(reinf_ships)]
            except Exception:
                pass

    return candidates


def materialize_joint_action(per_source_picks: list[tuple[int, int]],
                              world: Any, my_player: int) -> list:
    """Given list of (source_id, candidate_idx), build env action list.

    Filters out None (pass) candidates. Caller gets clean action list.
    """
    planet_by_id = {p.id: p for p in world.planets}
    comets = getattr(world, "comets", []) or []
    action_list = []
    for src_id, cand_idx in per_source_picks:
        src = planet_by_id.get(src_id)
        if src is None or src.owner != my_player:
            continue
        candidates = generate_per_source_candidates(src, world, my_player, comets)
        if 0 <= cand_idx < len(candidates) and candidates[cand_idx] is not None:
            action_list.append(candidates[cand_idx])
    return action_list


__all__ = [
    "K_PER_SOURCE", "CANDIDATE_NAMES",
    "generate_per_source_candidates", "materialize_joint_action",
]


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from kaggle_environments import make
    from training.lb1200_agent import build_world
    env = make("orbit_wars", debug=False); env.reset(num_agents=2)
    for _ in range(50):
        env.step([[], []])
    obs = env.state[0].observation
    world = build_world(obs)
    print(f"my_planets: {len(world.my_planets)}")
    for src in world.my_planets[:3]:
        cands = generate_per_source_candidates(src, world, 0)
        print(f"  planet {src.id} (ships={src.ships}): K=6 candidates")
        for i, c in enumerate(cands):
            lbl = CANDIDATE_NAMES[i]
            if c is None:
                print(f"    C{i} {lbl:>15s}: pass")
            else:
                print(f"    C{i} {lbl:>15s}: {c}")
