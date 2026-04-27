"""Translate raw expert actions into k14 factored-action space.

Expert trajectories (both scraped top-N replays and lb1200 self-play)
record actions as `[from_id, angle_rad, ships]` — the raw env format.
k14 policy is `(mode × target_from_topK × frac)` per owned planet:

    mode_idx  ∈ [0..N_MODES-1]       (pass, expand, attack, reinforce, denial)
    frac_idx  ∈ [0..N_FRACS-1]       quantized from ships/src.ships
    tgt_idx   ∈ [0..TOP_K_TARGETS-1] index into get_top_k_candidates output

This module provides the inverse mapping so those raw actions can become
supervision targets for BC-aux-loss on the k14 policy heads.

Design choices:
  - Mode is inferred from target ownership
    (own → reinforce; neutral → expand; enemy → attack). Denial is NOT
    labeled here because it's not deterministically recoverable from a
    single (src, angle, ships) tuple — the model can still learn it from
    self-play rollouts.
  - If the expert's angle doesn't resolve to a planet in src's top-K for
    the inferred mode, the action is dropped (return None). Typically
    happens when the expert aims through empty space or at a planet
    reinforce/expand considers unreachable. We don't force-label these
    because a wrong mode/tgt label would pull the policy toward the
    wrong mode — silence is safer.
  - Expert moves within the same turn are labeled in sequence; each
    subsequent move sees the `committed` ships dict updated with prior
    dispatches, so top-K features reflect intra-turn coordination as
    they would at inference time.

Public API:
  label_turn(raw_actions, obs, my_player)
      → list[LabeledPick]  (one entry per successfully labeled raw move)

Each `LabeledPick` is a named tuple:
    (pid, mode_idx, frac_idx, tgt_idx)
"""
from __future__ import annotations

import math
from collections import namedtuple
from typing import Optional

import numpy as np

from featurize import nearest_target_index
from training.lb1200_agent import build_world
from training.physics_action_helper_k13 import (
    FRACTIONS,
    N_FRACS,
    TOP_K_TARGETS,
    compute_mode_mask,
    get_top_k_candidates,
)


LabeledPick = namedtuple("LabeledPick", ["pid", "mode_idx", "frac_idx", "tgt_idx"])


def _mode_from_target(target_owner: int, my_player: int) -> Optional[int]:
    """Infer k14 mode from target planet ownership. Returns None if we
    can't classify (e.g., malformed owner field)."""
    if target_owner == my_player:
        return 3   # reinforce
    if target_owner == -1:
        return 1   # expand
    if target_owner >= 0:
        return 2   # attack
    return None


def _frac_idx_from_ships(ships: int, src_ships: int) -> int:
    """Quantize ships/src.ships to nearest FRACTIONS bucket.

    Ties go to the larger bucket (argmin breaks ties on first; we flip via
    reverse-order search to prefer higher fractions when distance equal).
    """
    if src_ships <= 0:
        return 0
    frac = min(1.0, max(0.0, ships / src_ships))
    diffs = [abs(frac - f) for f in FRACTIONS]
    best = 0
    best_d = diffs[0]
    for i, d in enumerate(diffs):
        if d < best_d:
            best = i
            best_d = d
    return best


def label_turn(
    raw_actions: list,
    obs: dict,
    my_player: int,
) -> list[LabeledPick]:
    """Label one turn's worth of raw expert actions.

    raw_actions: the seat's action list for this step — list of
                 [from_id, angle_rad, ships] (the env action format).
    obs:         that seat's observation dict for the same step. Must
                 include planets, fleets, angular_velocity, initial_planets,
                 and ideally comet_planet_ids — build_world needs these.
    my_player:   seat index.

    Returns list of LabeledPick for the actions we could recover. Drops
    actions where:
      - src planet isn't owned by my_player (shouldn't happen but
        defensively handled)
      - angle doesn't resolve to a nearby planet
      - target planet's mode isn't valid per compute_mode_mask
      - target planet isn't in get_top_k_candidates for that mode
    """
    if not raw_actions:
        return []

    # Need a "player" key for build_world to seat-localize
    obs_copy = {**obs, "player": my_player}
    try:
        world = build_world(obs_copy)
    except Exception:
        return []

    planet_by_id = {p.id: p for p in world.planets}
    raw_planets = obs.get("planets", []) or []
    raw_fleets = obs.get("fleets", []) or []

    committed: dict[int, int] = {}
    labeled: list[LabeledPick] = []

    for mv in raw_actions:
        if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
            continue
        try:
            from_id = int(mv[0])
            angle = float(mv[1])
            ships = int(mv[2])
        except (TypeError, ValueError):
            continue

        src = planet_by_id.get(from_id)
        if src is None or src.owner != my_player or src.ships <= 0:
            continue

        # Resolve target from angle (same logic used at inference-history time)
        src_raw = next((p for p in raw_planets if int(p[0]) == from_id), None)
        if src_raw is None:
            continue
        ti = nearest_target_index(src_raw, angle, raw_planets)
        if ti is None:
            continue
        target_pid = int(raw_planets[ti][0])
        target = planet_by_id.get(target_pid)
        if target is None or target.id == src.id:
            continue

        mode_idx = _mode_from_target(target.owner, my_player)
        if mode_idx is None:
            continue

        mask = compute_mode_mask(src, world, my_player)
        if not mask[mode_idx]:
            # The expert's choice isn't valid under our mode definitions —
            # usually means the target is outside _reachable() for this mode.
            continue

        cands, _feats, n_valid = get_top_k_candidates(
            src, world, my_player, mode_idx,
            fleets_raw=raw_fleets, committed=committed,
        )
        if n_valid == 0:
            continue

        tgt_idx = None
        for ci, cand in enumerate(cands[:n_valid]):
            if cand.id == target_pid:
                tgt_idx = ci
                break
        if tgt_idx is None:
            # Expert picked a target outside our top-K ranking. Skip
            # rather than force an incorrect label.
            continue

        frac_idx = _frac_idx_from_ships(ships, src.ships)

        labeled.append(LabeledPick(
            pid=int(from_id),
            mode_idx=int(mode_idx),
            frac_idx=int(frac_idx),
            tgt_idx=int(tgt_idx),
        ))

        # Update committed for subsequent moves in the same turn
        ships_est = max(1, int(src.ships * FRACTIONS[frac_idx]))
        committed[target_pid] = committed.get(target_pid, 0) + ships_est

    return labeled


def summarize_labels(labeled_batches: list[list[LabeledPick]]) -> dict:
    """Lightweight diagnostic: mode/frac distribution + label yield rate.

    Expects a list of per-turn label lists (e.g. one per step or one per
    trajectory). Returns histograms + fraction of turns that produced any
    labels (useful to catch "yield is too low, BC signal is sparse").
    """
    from training.physics_action_helper_k13 import N_MODES

    mode_counts = [0] * N_MODES
    frac_counts = [0] * N_FRACS
    tgt_counts = [0] * TOP_K_TARGETS
    total_labels = 0
    turns_with_labels = 0
    turns_total = 0
    for turn in labeled_batches:
        turns_total += 1
        if turn:
            turns_with_labels += 1
        for pick in turn:
            mode_counts[pick.mode_idx] += 1
            frac_counts[pick.frac_idx] += 1
            tgt_counts[pick.tgt_idx] += 1
            total_labels += 1
    return {
        "total_labels": total_labels,
        "turns_total": turns_total,
        "turns_with_labels": turns_with_labels,
        "mode_counts": mode_counts,
        "frac_counts": frac_counts,
        "tgt_counts": tgt_counts,
    }
