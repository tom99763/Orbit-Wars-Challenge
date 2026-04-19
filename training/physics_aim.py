"""Physics-accurate launch angle.

Wraps lb928's `aim_with_prediction` so fleets are aimed at where the target
planet WILL BE at arrival time (accounting for orbital motion) while avoiding
the sun. Falls back to straight-line atan2 if no safe intercept is found.
"""
from __future__ import annotations
import math
from typing import Sequence

from lb928_agent import Planet, aim_with_prediction


def compute_aim_angle(
    src_row: Sequence,
    tgt_row: Sequence,
    ships: int,
    ang_vel: float,
    initial_planets_raw: Sequence,
    comets: list | None,
    comet_ids,
) -> float:
    src = Planet(*src_row)
    tgt = Planet(*tgt_row)
    initial_by_id = {int(p[0]): Planet(*p) for p in initial_planets_raw}
    cids = set(int(i) for i in (comet_ids or []))
    try:
        result = aim_with_prediction(
            src, tgt, max(1, int(ships)),
            initial_by_id, float(ang_vel),
            comets or [], cids,
        )
    except Exception:
        result = None
    if result is None:
        return math.atan2(tgt.y - src.y, tgt.x - src.x)
    return float(result[0])
