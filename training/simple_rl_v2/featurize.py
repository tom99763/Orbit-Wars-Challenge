"""Feature extraction from OrbitWarsVecEnv → model input tensors.

CLAUDE-2.md describes a `get_batched_features(player_idx)` method on
the vec env, but the actual training/orbit_wars_vec_env.py we have
doesn't expose that — only the raw `pl_*` / `fl_*` arrays. This module
fills that gap with a `build_features` helper that reads the env's
public arrays and produces:

    pf         [N, NP, PLANET_FEAT_DIM=8]    per-planet features
    gf         [N, GLOBAL_FEAT_DIM=6]         per-env globals
    src_mask   [N, NP] bool                   owned planets (= valid src)
    cand_pidx  [N, NP, K=8] int               which planet idx fills each slot
                                              (-1 = NOOP for slot 0, -1 also
                                              for invalid slots when there
                                              are fewer than 3 enemies, etc.)
    cand_feat  [N, NP, K, CAND_FEAT_DIM=6]    per-candidate features
    cand_valid [N, NP, K] bool                True = candidate is real

Operates fully in numpy then converts to torch at the end. No Python
loop over env-or-planet — everything broadcasts across [N, NP].
"""
from __future__ import annotations

import numpy as np
import torch

from training.simple_rl_v2.model import (
    PLANET_FEAT_DIM, GLOBAL_FEAT_DIM, CAND_FEAT_DIM, K_CAND,
    SLOT_NOOP, SLOT_ENEMY_LO, SLOT_ENEMY_HI,
    SLOT_NEUTRAL_LO, SLOT_NEUTRAL_HI,
    SLOT_FRIEND_LO, SLOT_FRIEND_HI,
)


def _topk_nearest(
    src_x: np.ndarray, src_y: np.ndarray,         # [N, NP]
    cand_mask: np.ndarray,                         # [N, NP] — bool, candidate eligibility
    k: int,
) -> np.ndarray:
    """For each (env, src_planet), return indices of the k nearest planets
    among `cand_mask` (excluding self). Padded with -1 when there are
    fewer than k eligible planets.

    Returns [N, NP, k] int64.
    """
    N, NP = src_x.shape
    # Pairwise distance: [N, NP, NP]
    dx = src_x[:, :, None] - src_x[:, None, :]
    dy = src_y[:, :, None] - src_y[:, None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    # Mask self-distance + non-candidate slots with +inf
    self_mask = np.eye(NP, dtype=bool)[None, :, :]    # [1, NP, NP]
    invalid = self_mask | (~cand_mask[:, None, :])     # broadcast cand_mask over src dim
    dist = np.where(invalid, np.inf, dist)
    # Argsort + take first k
    idx = np.argsort(dist, axis=-1)[..., :k]           # [N, NP, k]
    # Mark slots whose distance was inf as invalid (-1)
    chosen_dist = np.take_along_axis(dist, idx, axis=-1)
    idx = np.where(np.isfinite(chosen_dist), idx, -1).astype(np.int64)
    return idx


def build_features(
    env,
    player: int,
    max_eta: float = 50.0,
) -> dict:
    """Build a tensors dict for `player`'s view of the current env state.

    Args:
        env: OrbitWarsVecEnv instance (must have pl_*, ang_vel, step_num)
        player: int 0..env.P-1
        max_eta: normalization constant for ETA features (50 ≈ board diag / max speed)

    Returns dict with numpy arrays (caller converts to torch):
        pf, gf, src_mask, cand_pidx, cand_feat, cand_valid
        (plus tgt_ships, tgt_static for downstream action conversion)
    """
    N, NP = env.N, env.NP
    pl_owner = env.pl_owner.copy()       # [N, NP] int8
    pl_ships = env.pl_ships              # [N, NP] float
    pl_x     = env.pl_x                  # [N, NP] float
    pl_y     = env.pl_y                  # [N, NP] float
    pl_prod  = env.pl_prod               # [N, NP] float
    pl_static = env.pl_is_static         # [N, NP] bool
    step_num = env.step_num              # [N] int

    is_me      = (pl_owner == player)
    is_neutral = (pl_owner == -1)
    is_enemy   = ~(is_me | is_neutral)

    # Per-planet features [N, NP, PLANET_FEAT_DIM=8]
    pf = np.stack([
        (pl_x - 50.0) / 50.0,
        (pl_y - 50.0) / 50.0,
        np.log1p(pl_ships) / np.log1p(1000.0),
        pl_prod / 5.0,
        is_me.astype(np.float32),
        is_enemy.astype(np.float32),
        is_neutral.astype(np.float32),
        pl_static.astype(np.float32),
    ], axis=-1).astype(np.float32)
    assert pf.shape[-1] == PLANET_FEAT_DIM

    # Global features [N, GLOBAL_FEAT_DIM=6]
    n_my  = is_me.sum(axis=1).astype(np.float32)        # [N]
    n_en  = is_enemy.sum(axis=1).astype(np.float32)
    n_nu  = is_neutral.sum(axis=1).astype(np.float32)
    my_ships    = (pl_ships * is_me).sum(axis=1)
    enemy_ships = (pl_ships * is_enemy).sum(axis=1)
    total_ships = my_ships + enemy_ships + 1e-6

    gf = np.stack([
        step_num.astype(np.float32) / 500.0,
        my_ships / total_ships,
        enemy_ships / total_ships,
        n_my / NP,
        n_en / NP,
        n_nu / NP,
    ], axis=-1).astype(np.float32)
    assert gf.shape[-1] == GLOBAL_FEAT_DIM

    # Source mask: only owned planets can launch
    src_mask = is_me

    # Candidate slot table per (env, src):
    #   slot 0: NOOP (always valid)
    #   slots 1-3: 3 nearest enemies
    #   slots 4-6: 3 nearest neutrals
    #   slot 7:    nearest friendly excluding self
    enemy_topk    = _topk_nearest(pl_x, pl_y, is_enemy,   k=3)   # [N, NP, 3]
    neutral_topk  = _topk_nearest(pl_x, pl_y, is_neutral, k=3)
    friend_topk   = _topk_nearest(pl_x, pl_y, is_me,      k=1)

    cand_pidx = np.full((N, NP, K_CAND), -1, dtype=np.int64)
    cand_pidx[..., SLOT_ENEMY_LO:SLOT_ENEMY_HI]      = enemy_topk
    cand_pidx[..., SLOT_NEUTRAL_LO:SLOT_NEUTRAL_HI]  = neutral_topk
    cand_pidx[..., SLOT_FRIEND_LO:SLOT_FRIEND_HI]    = friend_topk
    # slot 0 is NOOP — leave at -1; we mark cand_valid[..., 0] = True separately

    cand_valid = (cand_pidx >= 0)
    cand_valid[..., SLOT_NOOP] = True   # NOOP always available

    # Per-candidate features. We need ships, distance, prod, static, eta_norm,
    # is_noop. For invalid slots, keep zeros (still need real shapes).
    safe_pidx = np.maximum(cand_pidx, 0)   # avoid -1 indexing; we'll mask with cand_valid

    # Gather along last dim of pl_*
    def gather(arr, pidx):  # pidx: [N, NP, K], arr: [N, NP] → [N, NP, K]
        return np.take_along_axis(arr[:, None, :], pidx.transpose(0, 2, 1)[:, None, :, :], axis=-1)
    # Actually simpler: take_along with broadcast
    def gather_simple(arr, pidx):
        # arr[N, NP] → arr_e[N, 1, NP] then take_along axis 2 with pidx[N, NP, K]
        # Easier: arr[ env_index, pidx ] gather
        nidx = np.arange(N)[:, None, None]   # [N, 1, 1]
        return arr[nidx, pidx]              # [N, NP, K]

    tgt_ships  = gather_simple(pl_ships,  safe_pidx)   # [N, NP, K]
    tgt_x      = gather_simple(pl_x,      safe_pidx)
    tgt_y      = gather_simple(pl_y,      safe_pidx)
    tgt_prod   = gather_simple(pl_prod,   safe_pidx)
    tgt_static = gather_simple(pl_static, safe_pidx)

    # Distance from src to candidate (src is the planet at axis=1 dim of cand_pidx)
    dx = tgt_x - pl_x[..., None]
    dy = tgt_y - pl_y[..., None]
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)

    # ETA: dist / fleet_speed (use ships sent ≈ src ships at full commit)
    # Approximate v14 ship rule preview: send ~max(tgt+1, 20) bounded by src.
    src_ships_b = pl_ships[..., None]   # [N, NP, 1]
    approx_ships = np.minimum(src_ships_b, np.maximum(tgt_ships + 1.0, 20.0))
    speed = 1.0 + 5.0 * np.power(np.maximum(np.log(np.maximum(approx_ships, 1.0)) / np.log(1000.0), 0.0), 1.5)
    speed = np.minimum(speed, 6.0)
    eta = dist / np.maximum(speed, 1e-6)

    is_noop = np.zeros((N, NP, K_CAND), dtype=np.float32)
    is_noop[..., SLOT_NOOP] = 1.0

    cand_feat = np.stack([
        np.minimum(tgt_ships / 200.0, 5.0).astype(np.float32),
        np.minimum(dist / 100.0, 2.0).astype(np.float32),
        (tgt_prod / 5.0).astype(np.float32),
        tgt_static.astype(np.float32),
        np.minimum(eta / max_eta, 2.0).astype(np.float32),
        is_noop,
    ], axis=-1)
    # Zero out invalid candidates (otherwise garbage features)
    cand_feat = cand_feat * cand_valid[..., None].astype(np.float32)

    return {
        "pf":         pf,
        "gf":         gf,
        "src_mask":   src_mask,
        "cand_pidx":  cand_pidx,
        "cand_valid": cand_valid,
        "cand_feat":  cand_feat,
        "tgt_ships":  tgt_ships,
        "tgt_static": tgt_static,
        "tgt_x":      tgt_x,
        "tgt_y":      tgt_y,
    }


def to_torch(features: dict, device: str = "cpu") -> dict:
    """Convert numpy feature dict to torch tensors on device."""
    out = {}
    for k, v in features.items():
        if isinstance(v, np.ndarray):
            if v.dtype == bool:
                out[k] = torch.from_numpy(v).to(device)
            elif v.dtype.kind == "i":
                out[k] = torch.from_numpy(v.astype(np.int64)).to(device)
            else:
                out[k] = torch.from_numpy(v.astype(np.float32)).to(device)
        else:
            out[k] = v
    return out


__all__ = ["build_features", "to_torch"]
