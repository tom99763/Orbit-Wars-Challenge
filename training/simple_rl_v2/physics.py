"""Lead-target aim + sun-cross masking.

Numpy-vectorized so all per-action physics happens in one batched call,
matching the simple_rl_v2 design (no Python-loop per env per planet).

CLAUDE-2.md notes that lead aim is the single biggest fix in the action
ablation — without it, fleets aimed via plain atan2 miss orbiting
targets that have moved by arrival time. Net: ~95% loss vs lead-aim
agents in their head-to-head.

Public functions:
    fleet_speed(ships)
        ships: array → speed array. Matches lb1200_agent.fleet_speed.
    lead_target_angles(src_xy, tgt_xy, target_static, omega, ships,
                       center, max_iter)
        Predict where the target will be at fleet arrival time and aim
        there. Iterative because travel_time depends on dist which
        depends on lead which depends on travel_time.
    sun_crosses(src_xy, tgt_xy, sun_radius, safety)
        Boolean mask: which (src→tgt) lines pass within (radius+safety)
        of the origin? Used to skip wasted fleet launches.
"""
from __future__ import annotations

import numpy as np


# Game constants (must match orbit_wars_vec_env)
SUN_RADIUS = 10.0
SUN_SAFETY = 1.5
BOARD_CENTER = 50.0
MAX_SPEED = 6.0


def fleet_speed(ships: np.ndarray) -> np.ndarray:
    """Per-fleet speed: 1 + 5 · (log(ships) / log(1000))^1.5, capped at 6.

    Mirrors lb1200_agent.fleet_speed. Inputs/outputs are numpy arrays of
    any shape; the formula broadcasts elementwise.
    """
    ships = np.maximum(np.asarray(ships, dtype=np.float32), 1.0)
    s = np.log(ships) / np.log(1000.0)
    speed = 1.0 + 5.0 * np.power(np.maximum(s, 0.0), 1.5)
    return np.minimum(speed, MAX_SPEED).astype(np.float32)


def _orbit_position_at(
    target_static: np.ndarray,
    init_x: np.ndarray,
    init_y: np.ndarray,
    init_angle: np.ndarray,
    omega: np.ndarray,
    t: np.ndarray,
    center: float = BOARD_CENTER,
) -> tuple[np.ndarray, np.ndarray]:
    """Position of each planet at travel_time t.

    Static planets: stay at (init_x, init_y).
    Orbiting planets: rotate around (center, center) at angular velocity
    omega rad/turn. Initial angle is `init_angle` from center.

    All inputs broadcast to a common shape; t can broadcast against a
    different shape than the rest if needed (e.g. one t per src→tgt pair
    while init_x/init_y are per planet).
    """
    # Orbit radius (distance from center to init position)
    dx0 = init_x - center
    dy0 = init_y - center
    r = np.sqrt(dx0 * dx0 + dy0 * dy0)
    new_angle = init_angle + omega * t
    new_x = center + r * np.cos(new_angle)
    new_y = center + r * np.sin(new_angle)
    # Static targets: ignore orbit math
    out_x = np.where(target_static, init_x, new_x)
    out_y = np.where(target_static, init_y, new_y)
    return out_x.astype(np.float32), out_y.astype(np.float32)


def lead_target_angles(
    src_x: np.ndarray, src_y: np.ndarray,
    tgt_init_x: np.ndarray, tgt_init_y: np.ndarray,
    tgt_init_angle: np.ndarray, tgt_static: np.ndarray,
    omega: np.ndarray,
    ships: np.ndarray,
    center: float = BOARD_CENTER,
    max_iter: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively solve lead-target launch angles.

    For each (src, tgt, ships) triple, predict where the target will be
    when our fleet arrives, and aim there. We iterate because:
        travel_time  ≈ dist(src, predicted_tgt(t)) / fleet_speed(ships)
        predicted_tgt(t) = orbit position at time t (depends on t)

    Static targets converge in 1 iter; orbiting targets need 2-4.

    All array args broadcast to a common shape.

    Returns:
        angle      — atan2(dy, dx) in (-π, π]
        travel_time — predicted arrival time (used downstream for sun-mask
                      and arrival ledgers)
    """
    src_x = np.asarray(src_x, dtype=np.float32)
    src_y = np.asarray(src_y, dtype=np.float32)
    tgt_init_x = np.asarray(tgt_init_x, dtype=np.float32)
    tgt_init_y = np.asarray(tgt_init_y, dtype=np.float32)
    tgt_init_angle = np.asarray(tgt_init_angle, dtype=np.float32)
    tgt_static = np.asarray(tgt_static, dtype=bool)
    omega = np.asarray(omega, dtype=np.float32)

    speed = fleet_speed(ships)

    # Iter 0: aim at current position (no lead)
    pred_x, pred_y = tgt_init_x, tgt_init_y
    travel = np.zeros_like(src_x)
    for _ in range(max_iter):
        dx = pred_x - src_x
        dy = pred_y - src_y
        dist = np.sqrt(dx * dx + dy * dy)
        travel = np.where(speed > 0, dist / np.maximum(speed, 1e-6), 0.0)
        pred_x, pred_y = _orbit_position_at(
            tgt_static, tgt_init_x, tgt_init_y, tgt_init_angle, omega,
            travel, center=center,
        )

    angle = np.arctan2(pred_y - src_y, pred_x - src_x).astype(np.float32)
    return angle, travel.astype(np.float32)


def sun_crosses(
    src_x: np.ndarray, src_y: np.ndarray,
    tgt_x: np.ndarray, tgt_y: np.ndarray,
    sun_radius: float = SUN_RADIUS,
    safety: float = SUN_SAFETY,
    center: float = BOARD_CENTER,
) -> np.ndarray:
    """Bool mask: True for src→tgt lines that pass within (sun_radius+safety) of center.

    Standard point-to-segment distance: project center onto the segment,
    clamp to [0,1], compute distance to projection.

    Vectorized: src_*/tgt_* broadcast.
    """
    src_x = np.asarray(src_x, dtype=np.float32)
    src_y = np.asarray(src_y, dtype=np.float32)
    tgt_x = np.asarray(tgt_x, dtype=np.float32)
    tgt_y = np.asarray(tgt_y, dtype=np.float32)

    cx = float(center)
    cy = float(center)
    dx = tgt_x - src_x
    dy = tgt_y - src_y
    seg_len_sq = dx * dx + dy * dy
    # Param t = projection of (center − src) onto (tgt − src), clamped
    t = ((cx - src_x) * dx + (cy - src_y) * dy) / np.maximum(seg_len_sq, 1e-9)
    t = np.clip(t, 0.0, 1.0)
    proj_x = src_x + t * dx
    proj_y = src_y + t * dy
    d_sq = (proj_x - cx) ** 2 + (proj_y - cy) ** 2
    threshold = (sun_radius + safety) ** 2
    return d_sq < threshold


__all__ = ["fleet_speed", "lead_target_angles", "sun_crosses"]
