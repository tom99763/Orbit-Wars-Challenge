"""lb-1200 + shallow action-set lookahead.

Design:
  1. Get lb-1200's primary action list A_0 (its normal decision).
  2. Generate N variants: A_1..A_k (perturbed versions of A_0).
  3. For each variant, score it by projected "my total ships + owned planets
     value" at horizon H, using lb-1200's own `simulate_planet_timeline` as
     the per-planet forward model.
  4. Pick the variant with the highest score.

This is NOT full MCTS:
  - No state tree, no UCB, no rollouts.
  - Scoring assumes no new opponent actions in the horizon (worst-case approx).
  - Variants are crude perturbations, not systematic action-space exploration.

But it does add value over raw lb-1200:
  - Catches cases where lb-1200's single-planet heuristic leads to joint
    suboptimality (over-committing to an offensive that drains defense).
  - Provides a safety check: "pass" is always considered as a variant.

Runtime cost per turn:
  - lb-1200 primary: ~0.3-0.7s
  - 3 variants × 0.05s scoring (per-planet timeline): ~0.15s
  - Total: ~0.5-0.9s (well within Kaggle's 1-5s budget per turn)

Usage (local):
  from training.lb1200_lookahead_agent import agent
  # plug into kaggle_environments.make('orbit_wars').run([agent, starter_agent])

Kaggle submission:
  See submission_lb1200_lookahead/main.py (self-contained bundle).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Sequence

# Import lb-1200's core helpers. At Kaggle submission time this will be
# inlined into a self-contained main.py (see prep script).
from training.lb1200_agent import (
    agent as lb1200_base_agent,
    build_world, simulate_planet_timeline, resolve_arrival_event,
    estimate_arrival, travel_time, fleet_target_planet,
    aim_with_prediction, fleet_speed, actual_path_geometry,
    SIM_HORIZON, TOTAL_STEPS,
    Fleet,
)


# Default scoring horizon — balances lookahead depth vs compute.
# Bumped from 30 → 50: 30 steps was 1/16 of a 500-step game and missed the
# capture-then-defend cascades that take 40+ turns to play out. Time budget
# guard already truncates per-variant scoring if we run long.
LOOKAHEAD_HORIZON = 50

# Weights for scoring function:
W_MY_SHIPS = 1.0       # my total ships at horizon
W_MY_PLANETS = 30.0    # my planet count at horizon (each worth ~30 ships)
W_MY_PRODUCTION = 50.0 # NEW: my owned production at horizon — high-prod
                       # planets are more valuable than low-prod ones, the
                       # original W_MY_PLANETS=30 treated them all equal.
W_ENEMY_SHIPS = -0.5   # penalize enemy ships
W_ENEMY_PRODUCTION = -25.0  # NEW: penalize enemy production (mirror)
W_RISK = -20.0         # penalize each of our planets predicted to fall

# Comet windows: ±5 turns around each spawn step. Capturing a comet during
# its window grants free production for the rest of the game, so my_ships
# and my_planets at horizon are worth more if we're inside a window now.
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
COMET_WIN = {s + d for s in COMET_SPAWN_STEPS for d in range(-5, 6)}
COMET_BONUS_MULT = 1.5  # multiply my-side scoring while in window


def _infer_target_planet(src, angle: float, planets):
    """Given a source planet and launch angle, find the target planet by
    picking the one whose trajectory the fleet will hit. Reuses lb-1200's
    `fleet_target_planet` logic via a pseudo-fleet.
    """
    # Build a pseudo-fleet at the source
    pseudo = Fleet(-1, 0, src.x, src.y, angle, src.id, 1)
    return fleet_target_planet(pseudo, planets)


def _compute_arrivals(world, action_list, include_existing=True):
    """Build a {planet_id: [(arrival_turn, owner, ships), ...]} ledger.

    `action_list`: list of [src_id, angle, ships] triples (lb-1200 format).
    Fleets from `world.fleets` (existing in-flight) are always included
    if include_existing=True.
    """
    arrivals = defaultdict(list)
    # Existing in-flight fleets
    if include_existing:
        for f in world.fleets:
            result = fleet_target_planet(f, world.planets)
            if result is None:
                continue
            target, travel_turns = result
            if target is None:
                continue
            arrivals[target.id].append(
                (int(travel_turns), f.owner, int(f.ships)))

    # New launches from candidate action set
    planet_by_id = {p.id: p for p in world.planets}
    for mv in action_list:
        if len(mv) != 3:
            continue
        src_id, angle, ships = mv
        src = planet_by_id.get(int(src_id))
        if src is None or ships <= 0:
            continue
        result = _infer_target_planet(src, angle, world.planets)
        if result is None:
            continue
        target, travel_turns = result
        if target is None:
            continue
        arrivals[target.id].append(
            (int(travel_turns), world.player, int(ships)))
    return arrivals


def _score_action_set(world, action_list, horizon=LOOKAHEAD_HORIZON, step=0):
    """Score an action set by predicted ship/planet outcome at horizon.

    Higher is better. Considers:
      - My total ships on owned planets at horizon (W_MY_SHIPS)
      - My planet count at horizon (W_MY_PLANETS)
      - My total production at horizon (W_MY_PRODUCTION)  — NEW
      - Enemy ships on owned planets (W_ENEMY_SHIPS)
      - Enemy total production (W_ENEMY_PRODUCTION)  — NEW
      - Risk: planets expected to fall to enemies (W_RISK)
      - Comet bonus: my-side terms scaled by COMET_BONUS_MULT during
        comet spawn windows (capturing comets = free long-term production).
    """
    arrivals = _compute_arrivals(world, action_list, include_existing=True)
    me = world.player

    my_ships = 0.0
    my_planets = 0
    my_production = 0.0
    enemy_ships = 0.0
    enemy_production = 0.0
    fallen_planets = 0

    for planet in world.planets:
        plan_arrivals = arrivals.get(planet.id, [])
        timeline_info = simulate_planet_timeline(planet, plan_arrivals, me, horizon)
        owner_at = timeline_info.get("owner_at", {})
        ships_at = timeline_info.get("ships_at", {})
        final_owner = owner_at.get(horizon, planet.owner)
        final_ships = ships_at.get(horizon, 0.0)

        if final_owner == me:
            my_ships += final_ships
            my_planets += 1
            my_production += planet.production
        elif final_owner != -1:
            enemy_ships += final_ships
            enemy_production += planet.production

        # Risk: our planet this turn, lost at horizon
        if planet.owner == me and final_owner != me:
            fallen_planets += 1

    in_comet = (step in COMET_WIN)
    my_mult = COMET_BONUS_MULT if in_comet else 1.0

    score = (my_mult * (W_MY_SHIPS * my_ships
                        + W_MY_PLANETS * my_planets
                        + W_MY_PRODUCTION * my_production)
             + W_ENEMY_SHIPS * enemy_ships
             + W_ENEMY_PRODUCTION * enemy_production
             + W_RISK * fallen_planets)
    return score, {
        "my_ships": my_ships, "my_planets": my_planets,
        "my_prod": my_production,
        "enemy_ships": enemy_ships, "enemy_prod": enemy_production,
        "fallen": fallen_planets, "comet_bonus": in_comet,
    }


def _generate_variants(primary: list, world) -> list[list]:
    """Generate candidate action variants around the primary action list.

    Variants:
      V0: original primary
      V1: same actions, each reduced to 75% of ships
      V2: same actions, each reduced to 50% of ships
      V3: drop the lowest-priority single action (one with smallest ships)
      V4: no actions (pass this turn)
      V5 (NEW): drop the highest-priority single action — sometimes the
                "obvious" big move is over-committing; let the simulation
                check whether holding back the largest fleet is actually
                better. Mirror to V3.
      V6 (NEW): drop two weakest actions if available — a "minimal
                action" variant. Catches cases where lb1200 wants to do
                3 things but only 1 is critical.
    """
    variants = [list(primary)]  # V0
    if not primary:
        # Nothing to vary if lb-1200 chose to pass
        return variants

    # V1: 75%
    v1 = [[m[0], m[1], max(1, int(m[2] * 0.75))] for m in primary]
    variants.append(v1)
    # V2: 50%
    v2 = [[m[0], m[1], max(1, int(m[2] * 0.5))] for m in primary]
    variants.append(v2)
    # V3: drop weakest (smallest ships count — likely least impactful)
    if len(primary) >= 2:
        weakest_idx = min(range(len(primary)), key=lambda i: primary[i][2])
        v3 = [m for i, m in enumerate(primary) if i != weakest_idx]
        variants.append(v3)
    # V4: pass
    variants.append([])
    # V5: drop strongest — counter-intuitive but the biggest move can be
    # the most over-committal one
    if len(primary) >= 2:
        strongest_idx = max(range(len(primary)), key=lambda i: primary[i][2])
        v5 = [m for i, m in enumerate(primary) if i != strongest_idx]
        variants.append(v5)
    # V6: drop two weakest (only if we have ≥3 actions to start with)
    if len(primary) >= 3:
        sorted_idx = sorted(range(len(primary)), key=lambda i: primary[i][2])
        drop = set(sorted_idx[:2])
        v6 = [m for i, m in enumerate(primary) if i not in drop]
        variants.append(v6)

    return variants


def agent(obs, config=None):
    """Main entry — lb-1200 + shallow lookahead."""
    start_time = time.perf_counter()

    # 1. Get lb-1200's primary action
    primary = lb1200_base_agent(obs, config) or []

    # Early exit if pass (nothing to vary)
    if not primary:
        return []

    # Budget guard: if lb-1200 already took >0.7s, skip lookahead
    elapsed = time.perf_counter() - start_time
    act_timeout = 1.0
    if config is not None:
        try:
            act_timeout = float(config.get("actTimeout", 1.0))
        except Exception:
            pass
    budget = max(0.3, act_timeout * 0.3)  # 30% of turn for lookahead
    if elapsed > act_timeout * 0.6:
        return primary

    # 2. Build world model (reuse lb-1200's builder)
    try:
        world = build_world(obs)
    except Exception:
        return primary

    # 3. Generate variants
    variants = _generate_variants(primary, world)

    # 4. Score each within budget — pass current step so comet bonus
    #    triggers when we're inside a window
    step = int(obs.get("step", 0) or 0) if isinstance(obs, dict) \
           else int(getattr(obs, "step", 0) or 0)
    deadline = start_time + act_timeout * 0.9
    best_score = -float("inf")
    best_action = primary
    for v in variants:
        if time.perf_counter() > deadline:
            break
        try:
            score, _ = _score_action_set(world, v, horizon=LOOKAHEAD_HORIZON, step=step)
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_action = v

    return best_action


# Convenience: a "debug" variant that returns scoring metadata for analysis.
def agent_debug(obs, config=None):
    """Same as agent() but also returns the scoring details (for offline analysis)."""
    start_time = time.perf_counter()
    primary = lb1200_base_agent(obs, config) or []
    if not primary:
        return [], {"variants": [], "chosen": -1}
    try:
        world = build_world(obs)
    except Exception:
        return primary, {"variants": [], "chosen": 0}
    variants = _generate_variants(primary, world)
    scores = []
    for v in variants:
        try:
            score, info = _score_action_set(world, v)
        except Exception:
            score, info = -float("inf"), {}
        scores.append({"variant": v, "score": score, **info})
    chosen = max(range(len(scores)), key=lambda i: scores[i]["score"])
    return variants[chosen], {"variants": scores, "chosen": chosen}


__all__ = ["agent", "agent_debug"]
