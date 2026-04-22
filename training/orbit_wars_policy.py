"""Orbit Wars policy — factored (mode × target × frac) action space.

Provides TWO entry points with IDENTICAL log_prob formulas:

  rollout_step(net, feat, spatial, world, seat, raw_fleets, device)
    → (picks, sample_dict)
    Used during episode collection.  No grad.

  eval_batch(net, batch_td, device)
    → (log_prob [B], entropy [B], value [B])   all tensors with grad
    Used during PPO mini-batch updates.

Log_prob convention (must be IDENTICAL in both paths):
  For each owned planet i:
    lp += log_p_mode(mode_idx_i)
    if mode_idx_i != pass:
      lp += log_p_target(tgt_idx_i)
      lp += log_p_frac(frac_idx_i)
  Final: log_prob = total_lp / n_owned_planets   (normalise by planet count)

batch_td keys (all float32 / bool / int64, shape [B, ...]):
  obs:   planets [B, MAX_P, PD], planet_mask [B, MAX_P],
         fleets  [B, MAX_F, FD], fleet_mask  [B, MAX_F],
         globals_ [B, GD], spatial [B, C, G, G]
  acts:  planet_mode_idx [B, MAX_P], planet_frac_idx [B, MAX_P],
         planet_tgt_idx  [B, MAX_P],
         planet_cand_feats [B, MAX_P, K, CAND_FEAT_DIM],
         planet_n_valid [B, MAX_P], planet_mode_mask [B, MAX_P, N_MODES],
         planet_is_owned [B, MAX_P]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from featurize import PLANET_DIM, FLEET_DIM, GLOBAL_DIM
from training.dual_stream_model import N_SPATIAL_CHANNELS
from training.physics_picker_k13_ppo import GRID, GAMMA
from training.physics_action_helper_k13 import (
    N_MODES, N_FRACS, TOP_K_TARGETS, CAND_FEAT_DIM, FRACTIONS,
    compute_mode_mask, get_top_k_candidates, materialize_with_targets,
)

MAX_PLANETS = 40
MAX_FLEETS  = 100


# ──────────────────────────────────────────────────────────────────────────────
# Rollout (no grad)
# ──────────────────────────────────────────────────────────────────────────────

def rollout_step(net, feat: dict, spatial, world, seat: int,
                 raw_fleets: list, device: str) -> tuple:
    """Sample factored actions for all owned planets.

    Returns
    -------
    picks : list of 8-tuples
        (pid, mode_idx, frac_idx, cand_feats, tgt_idx, n_valid, target_pid, mask)
    sample : dict | None
        {'log_prob', 'entropy', 'value', 'feat', 'spatial', 'picks',
         'planet_ids', 'reward', 'my_player'}
        None if agent has no owned planets.
    """
    pl = feat["planets"]
    fl = feat.get("fleets", None)
    if pl.shape[0] == 0:
        pl    = np.zeros((1, PLANET_DIM), dtype=np.float32)
        pmask = np.zeros(1, dtype=bool)
    else:
        pmask = np.ones(pl.shape[0], dtype=bool)
    if fl is None or fl.ndim < 2 or fl.shape[0] == 0:
        fl    = np.zeros((1, FLEET_DIM), dtype=np.float32)
        fmask = np.zeros(1, dtype=bool)
    else:
        fmask = np.ones(fl.shape[0], dtype=bool)

    with torch.no_grad():
        fused_tokens, mode_logits, v = net(
            torch.from_numpy(pl).unsqueeze(0).to(device),
            torch.from_numpy(pmask).unsqueeze(0).to(device),
            torch.from_numpy(fl).unsqueeze(0).to(device),
            torch.from_numpy(fmask).unsqueeze(0).to(device),
            torch.from_numpy(feat["globals"]).unsqueeze(0).to(device),
            torch.from_numpy(spatial).unsqueeze(0).to(device),
        )
    ml_np = mode_logits[0].cpu().numpy()   # [P, N_MODES]
    planet_ids = feat.get("planet_ids", [])

    picks = []
    committed: dict = {}
    log_prob_sum = 0.0
    ent_sum = 0.0
    n_owned = 0

    for i, pid in enumerate(planet_ids):
        src = next((p for p in world.planets if p.id == int(pid)), None)
        if src is None or src.owner != seat:
            continue
        n_owned += 1
        mask = compute_mode_mask(src, world, seat)

        m_log = ml_np[i].copy()
        for k in range(N_MODES):
            if not mask[k]:
                m_log[k] = -1e9
        m_p = np.exp(m_log - m_log.max())
        m_p /= m_p.sum()
        mode_idx = int(np.random.choice(N_MODES, p=m_p))

        log_prob_sum += float(np.log(m_p[mode_idx] + 1e-9))
        ent_sum -= float((m_p * np.log(m_p + 1e-9)).sum())

        if mode_idx == 0:  # pass
            picks.append((
                int(pid), 0, 0,
                np.zeros((TOP_K_TARGETS, CAND_FEAT_DIM), dtype=np.float32),
                0, 1, -1, tuple(mask),
            ))
            continue

        # Target selection
        cands, cand_feats, n_valid = get_top_k_candidates(
            src, world, seat, mode_idx,
            fleets_raw=raw_fleets, committed=committed,
        )
        if n_valid == 0:
            picks.append((
                int(pid), 0, 0,
                np.zeros((TOP_K_TARGETS, CAND_FEAT_DIM), dtype=np.float32),
                0, 1, -1, tuple(mask),
            ))
            n_owned -= 1  # forced pass doesn't count
            continue
        cf_t = torch.from_numpy(cand_feats).unsqueeze(0).to(device)  # (1,K,D)
        with torch.no_grad():
            t_scores = net.target_logits_for(
                fused_tokens[0, i].unsqueeze(0), cf_t
            )[0].cpu().numpy()   # (K,)
        t_scores[n_valid:] = -1e9
        t_p = np.exp(t_scores - t_scores[:n_valid].max())
        t_p[n_valid:] = 0.0
        t_p /= t_p.sum()
        tgt_idx = int(np.random.choice(TOP_K_TARGETS, p=t_p))
        target_pid = cands[tgt_idx].id

        log_prob_sum += float(np.log(t_p[tgt_idx] + 1e-9))
        ent_sum -= float((t_p[:n_valid] * np.log(t_p[:n_valid] + 1e-9)).sum())

        # Fraction selection (conditioned on mode)
        with torch.no_grad():
            f_logits = net.frac_logits_for(
                fused_tokens[0, i],
                torch.tensor(mode_idx, dtype=torch.long, device=device),
            ).cpu().numpy()
        f_p = np.exp(f_logits - f_logits.max())
        f_p /= f_p.sum()
        frac_idx = int(np.random.choice(N_FRACS, p=f_p))

        log_prob_sum += float(np.log(f_p[frac_idx] + 1e-9))
        ent_sum -= float((f_p * np.log(f_p + 1e-9)).sum())

        ships_sent = max(1, int(src.ships * FRACTIONS[frac_idx]))
        committed[target_pid] = committed.get(target_pid, 0) + ships_sent
        picks.append((
            int(pid), mode_idx, frac_idx, cand_feats,
            tgt_idx, n_valid, target_pid, tuple(mask),
        ))

    if not picks:
        return picks, None

    n_owned = max(1, n_owned)
    log_prob = log_prob_sum / n_owned
    # entropy: mean over all per-decision entropies (mode+tgt+frac per planet)
    n_decisions = sum(3 if p[1] != 0 else 1 for p in picks)
    entropy = ent_sum / max(1, n_decisions)

    sample = {
        "feat": feat,
        "spatial": spatial,
        "planet_ids": planet_ids,
        "picks": picks,
        "log_prob": log_prob,
        "entropy": entropy,
        "value": float(v.item()),
        "reward": 0.0,
        "my_player": seat,
        "n_owned": n_owned,
    }
    return picks, sample


# ──────────────────────────────────────────────────────────────────────────────
# Eval batch (with grad — for PPO)
# ──────────────────────────────────────────────────────────────────────────────

def eval_batch(net, batch_td: dict, device: str):
    """Recompute log_prob / entropy / value for stored actions.

    Parameters
    ----------
    batch_td : dict of tensors, each [B, ...]
    device   : torch device string

    Returns
    -------
    log_prob : Tensor [B]   — normalised by n_owned_planets, with grad
    entropy  : Tensor [B]   — mean per-decision entropy, with grad
    value    : Tensor [B]   — state value, with grad
    """
    def t(key, dtype=None):
        x = batch_td[key]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.to(device)
        if dtype is not None:
            x = x.to(dtype)
        return x

    planets     = t("planets",     torch.float32)   # [B, MAX_P, PD]
    planet_mask = t("planet_mask", torch.bool)
    fleets      = t("fleets",      torch.float32)   # [B, MAX_F, FD]
    fleet_mask  = t("fleet_mask",  torch.bool)
    globals_    = t("globals_",    torch.float32)   # [B, GD]
    spatial     = t("spatial",     torch.float32)   # [B, C, G, G]

    mode_idx    = t("planet_mode_idx",   torch.long)    # [B, MAX_P]
    frac_idx    = t("planet_frac_idx",   torch.long)
    tgt_idx     = t("planet_tgt_idx",    torch.long)
    cand_feats  = t("planet_cand_feats", torch.float32) # [B, MAX_P, K, D]
    n_valid     = t("planet_n_valid",    torch.long)    # [B, MAX_P]
    mode_mask   = t("planet_mode_mask",  torch.float32) # [B, MAX_P, N_MODES]
    is_owned    = t("planet_is_owned",   torch.float32) # [B, MAX_P]

    B, MAX_P, _ = planets.shape

    # Forward
    fused_tokens, mode_logits, value = net(
        planets, planet_mask, fleets, fleet_mask, globals_, spatial,
    )
    # fused_tokens: [B, MAX_P, d_entity]
    # mode_logits:  [B, MAX_P, N_MODES]
    # value:        [B]

    # ── Mode log_prob ─────────────────────────────────────────────────────────
    masked_mode = mode_logits + (1.0 - mode_mask) * (-1e9)
    log_p_mode  = F.log_softmax(masked_mode, dim=-1)       # [B, MAX_P, N_MODES]
    log_p_mode_sel = log_p_mode.gather(
        2, mode_idx.unsqueeze(-1)
    ).squeeze(-1)                                           # [B, MAX_P]

    # ── Target log_prob ───────────────────────────────────────────────────────
    K = cand_feats.shape[2]
    ft_flat = fused_tokens.view(B * MAX_P, -1)             # [B*MAX_P, d]
    cf_flat = cand_feats.view(B * MAX_P, K, CAND_FEAT_DIM)
    t_logits = net.target_logits_for(ft_flat, cf_flat).view(B, MAX_P, K)  # [B,P,K]
    # Mask positions beyond n_valid
    k_idx = torch.arange(K, device=device).view(1, 1, K)  # [1,1,K]
    t_logits = t_logits.masked_fill(k_idx >= n_valid.unsqueeze(-1), -1e9)
    log_p_tgt = F.log_softmax(t_logits, dim=-1)
    log_p_tgt_sel = log_p_tgt.gather(
        2, tgt_idx.unsqueeze(-1)
    ).squeeze(-1)                                           # [B, MAX_P]

    # ── Frac log_prob ─────────────────────────────────────────────────────────
    f_logits    = net.frac_logits_for(fused_tokens, mode_idx)  # [B, MAX_P, N_FRACS]
    log_p_frac  = F.log_softmax(f_logits, dim=-1)
    log_p_frac_sel = log_p_frac.gather(
        2, frac_idx.unsqueeze(-1)
    ).squeeze(-1)                                           # [B, MAX_P]

    # ── Combine ───────────────────────────────────────────────────────────────
    not_pass  = ((mode_idx != 0).float() * is_owned)       # [B, MAX_P]
    lp_sum    = (is_owned * log_p_mode_sel
                 + not_pass * (log_p_tgt_sel + log_p_frac_sel))   # [B, MAX_P]
    n_owned   = is_owned.sum(dim=1).clamp(min=1.0)         # [B]
    log_prob  = lp_sum.sum(dim=1) / n_owned                # [B]

    # ── Entropy ───────────────────────────────────────────────────────────────
    p_mode = log_p_mode.exp()
    ent_mode = -(p_mode * log_p_mode).sum(-1) * is_owned   # [B, MAX_P]
    p_tgt  = log_p_tgt.exp()
    ent_tgt  = -(p_tgt  * log_p_tgt ).sum(-1) * not_pass
    p_frac = log_p_frac.exp()
    ent_frac = -(p_frac * log_p_frac).sum(-1) * not_pass

    n_decs   = (is_owned + 2.0 * not_pass).sum(dim=1).clamp(min=1.0)  # [B]
    entropy  = (ent_mode + ent_tgt + ent_frac).sum(dim=1) / n_decs    # [B]

    return log_prob, entropy, value


# ──────────────────────────────────────────────────────────────────────────────
# Build padded TensorDict sample from a rollout step
# ──────────────────────────────────────────────────────────────────────────────

def build_sample_td(sample: dict) -> dict:
    """Convert one rollout sample dict to padded fixed-size tensors.

    Returns plain dict of numpy arrays — converted to TensorDict by caller.
    """
    feat    = sample["feat"]
    spatial = sample["spatial"]
    picks   = sample["picks"]
    pid_list = [p[0] for p in picks]
    pid_set  = set(int(x) for x in pid_list)

    planet_ids_feat = [int(x) for x in feat.get("planet_ids", [])]
    pid_to_feat_idx = {pid: j for j, pid in enumerate(planet_ids_feat)}

    pl_feat = feat["planets"]  # [P, PLANET_DIM]
    fl_feat = feat.get("fleets", np.zeros((0, FLEET_DIM), dtype=np.float32))
    if fl_feat.ndim < 2:
        fl_feat = np.zeros((0, FLEET_DIM), dtype=np.float32)
    P = pl_feat.shape[0]
    F = fl_feat.shape[0]

    # Padded obs
    planets_pad    = np.zeros((MAX_PLANETS, PLANET_DIM),             dtype=np.float32)
    planet_mask_pad= np.zeros((MAX_PLANETS,),                        dtype=bool)
    fleets_pad     = np.zeros((MAX_FLEETS,  FLEET_DIM),              dtype=np.float32)
    fleet_mask_pad = np.zeros((MAX_FLEETS,),                         dtype=bool)
    planet_ids_pad = np.full((MAX_PLANETS,), -1,                     dtype=np.int64)

    if P > 0:
        n = min(P, MAX_PLANETS)
        planets_pad[:n]     = pl_feat[:n]
        planet_mask_pad[:n] = True
        for j, pid in enumerate(planet_ids_feat[:MAX_PLANETS]):
            planet_ids_pad[j] = pid
    if F > 0:
        n = min(F, MAX_FLEETS)
        fleets_pad[:n]     = fl_feat[:n]
        fleet_mask_pad[:n] = True

    # Padded action tensors
    mode_idx_pad   = np.zeros((MAX_PLANETS,),                        dtype=np.int64)
    frac_idx_pad   = np.zeros((MAX_PLANETS,),                        dtype=np.int64)
    tgt_idx_pad    = np.zeros((MAX_PLANETS,),                        dtype=np.int64)
    cand_feats_pad = np.zeros((MAX_PLANETS, TOP_K_TARGETS, CAND_FEAT_DIM), dtype=np.float32)
    n_valid_pad    = np.ones( (MAX_PLANETS,),                        dtype=np.int64)
    mode_mask_pad  = np.zeros((MAX_PLANETS, N_MODES),                dtype=bool)
    is_owned_pad   = np.zeros((MAX_PLANETS,),                        dtype=bool)

    for p in picks:
        pid, mi, fi, cf, ti, nv, _, mask = p
        j = pid_to_feat_idx.get(int(pid), -1)
        if j < 0 or j >= MAX_PLANETS:
            continue
        mode_idx_pad[j]    = mi
        frac_idx_pad[j]    = fi
        tgt_idx_pad[j]     = ti
        cand_feats_pad[j]  = cf
        n_valid_pad[j]     = max(1, nv)
        mode_mask_pad[j]   = mask
        is_owned_pad[j]    = True

    return {
        "planets":          planets_pad,
        "planet_mask":      planet_mask_pad,
        "fleets":           fleets_pad,
        "fleet_mask":       fleet_mask_pad,
        "globals_":         feat["globals"].astype(np.float32),
        "spatial":          spatial.astype(np.float32),
        "planet_ids":       planet_ids_pad,
        "planet_mode_idx":  mode_idx_pad,
        "planet_frac_idx":  frac_idx_pad,
        "planet_tgt_idx":   tgt_idx_pad,
        "planet_cand_feats": cand_feats_pad,
        "planet_n_valid":   n_valid_pad,
        "planet_mode_mask": mode_mask_pad,
        "planet_is_owned":  is_owned_pad,
        "log_prob_old":     np.float32(sample["log_prob"]),
        "value_old":        np.float32(sample["value"]),
        "reward":           np.float32(sample.get("reward", 0.0)),
    }
