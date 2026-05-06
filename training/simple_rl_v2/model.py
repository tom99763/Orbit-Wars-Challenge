"""SimpleRLAgentV2 — minimal RL policy for Orbit Wars (CLAUDE-2.md spec).

This MVP implements the BEST setting per the doc's TL;DR:
    --backbone mlp --hidden 128
    --target-head hier
    --ship-head v14
    --lead-aim 1

Other variants (transformer backbones, pointer/k8 heads, bucket5 ship)
are TODO — they exist as an ablation surface, not as the recommended
default.

Architecture
============

Input features per (env, planet):
    pf [B, NP, 8] = [x/100, y/100, log1p(ships)/log1p(1000), prod/5,
                     is_me, is_enemy, is_neutral, is_static]
Global features per env:
    gf [B, 6]    = [step/500, my_ships/total, enemy_ships/total,
                    n_my_planets/NP, n_enemy_planets/NP, n_neutral_planets/NP]

Backbone (MLP):
    pf  → Linear(8 → 128) → GELU → Linear(128 → 128)        : enc_p [B, NP, 128]
    gf  → Linear(6 → 128) → GELU → Linear(128 → 128)        : enc_g [B, 128]
    fused = enc_p + enc_g.unsqueeze(1)                       : [B, NP, 128]

Per-planet candidate slots (K=8):
    Slot 0: NOOP
    Slot 1-3: 3 nearest enemy planets
    Slot 4-6: 3 nearest neutral planets
    Slot 7:   1 nearest friendly (excl self)
    Built externally (build_candidates_vec, called by trainer).

Hier action head:
    type_head:  fused → 4 logits (NOOP / SNIPE / EXPAND / REINFORCE)
    sub_head:   for each planet+candidate, score (fused, cand_feat) → logit
    Joint log-prob:
        log p_type[type_for_k] + log p_sub[k] over the candidates of that type
        — type_for_k is fixed by slot index
        — sub-head softmax is restricted to candidates of the chosen type

Ship rule (v14, deterministic):
    send = min(src_ships, max(tgt_ships + 1, 20))
    No learned ship head — ship_head MLP is unused but reserved for future
    bucket5 mode.

Value head:
    mean-pool(fused) → Linear → scalar V(s)
    NOTE: input to value_head is detached so v_loss does NOT backprop into
    the backbone (Tom rl10 Fix #2). Toggle off via ctor flag if you need
    backbone to learn from value gradient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Action types (hier head)
TYPE_NOOP      = 0
TYPE_SNIPE     = 1   # attack enemy
TYPE_EXPAND    = 2   # capture neutral
TYPE_REINFORCE = 3   # send to own
N_TYPES = 4

# Candidate slot layout (K=8)
K_CAND          = 8
SLOT_NOOP       = 0
SLOT_ENEMY_LO   = 1   # enemies fill slots 1-3
SLOT_ENEMY_HI   = 4   # exclusive end
SLOT_NEUTRAL_LO = 4
SLOT_NEUTRAL_HI = 7
SLOT_FRIEND_LO  = 7
SLOT_FRIEND_HI  = 8

# Map each candidate slot → action type for the hier head
SLOT_TO_TYPE = np.array([
    TYPE_NOOP,        # 0
    TYPE_SNIPE,       # 1
    TYPE_SNIPE,       # 2
    TYPE_SNIPE,       # 3
    TYPE_EXPAND,      # 4
    TYPE_EXPAND,      # 5
    TYPE_EXPAND,      # 6
    TYPE_REINFORCE,   # 7
], dtype=np.int64)

# Per type: which slots belong to it
TYPE_TO_SLOTS = {
    TYPE_NOOP:      [0],
    TYPE_SNIPE:     [1, 2, 3],
    TYPE_EXPAND:    [4, 5, 6],
    TYPE_REINFORCE: [7],
}

# Candidate feature dim (used by sub-head). Built externally.
#   [tgt_ships/200, dist/100, prod/5, is_static, eta_norm, is_noop]
CAND_FEAT_DIM = 6

# Per-planet feature dim (input to backbone)
PLANET_FEAT_DIM = 8
GLOBAL_FEAT_DIM = 6


# ----------------------------------------------------------------------------
# Backbone
# ----------------------------------------------------------------------------

class _MLPBackbone(nn.Module):
    """Per-planet 2-layer MLP. Operates on the last dim, broadcasting over
    [B, NP] to produce [B, NP, hidden]."""
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, pf: torch.Tensor) -> torch.Tensor:
        return self.net(pf)


class _GlobalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, gf: torch.Tensor) -> torch.Tensor:
        return self.net(gf)


# ----------------------------------------------------------------------------
# Action head: hierarchical (type × sub-target)
# ----------------------------------------------------------------------------

class _HierHead(nn.Module):
    """Hierarchical action head.

    For each (env, src_planet):
      - type_head produces logits over 4 action types from the planet's
        fused embedding
      - sub_head produces logits over K=8 candidate slots, scoring each
        slot via its (fused, cand_feat) pair
      - The joint distribution is constructed by MASKING sub-logits to
        only the slots of the chosen type, then renormalising.

    Output of forward():
      type_logits [B, NP, N_TYPES]
      sub_logits  [B, NP, K_CAND]
    The trainer combines these into joint log-probs via
    `log_prob_action()` for loss computation.
    """
    def __init__(self, hidden: int, cand_feat_dim: int = CAND_FEAT_DIM):
        super().__init__()
        self.type_head = nn.Linear(hidden, N_TYPES)
        # Sub head scores each (fused, cand_feat) pair
        self.sub_head = nn.Sequential(
            nn.Linear(hidden + cand_feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        fused: torch.Tensor,        # [B, NP, hidden]
        cand_feat: torch.Tensor,    # [B, NP, K, cand_feat_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        type_logits = self.type_head(fused)   # [B, NP, N_TYPES]

        B, NP, K, _ = cand_feat.shape
        H = fused.shape[-1]
        # Broadcast fused [B, NP, H] → [B, NP, K, H]
        fused_b = fused.unsqueeze(2).expand(B, NP, K, H)
        sub_input = torch.cat([fused_b, cand_feat], dim=-1)
        sub_logits = self.sub_head(sub_input).squeeze(-1)   # [B, NP, K]
        return type_logits, sub_logits


# ----------------------------------------------------------------------------
# Value head (with detached input by default — Fix #2)
# ----------------------------------------------------------------------------

class _ValueHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, fused_pooled: torch.Tensor) -> torch.Tensor:
        return self.net(fused_pooled).squeeze(-1)


# ----------------------------------------------------------------------------
# v14 ship rule (deterministic)
# ----------------------------------------------------------------------------

def v14_ship_count(src_ships: np.ndarray, tgt_ships: np.ndarray) -> np.ndarray:
    """min(src, max(tgt + 1, 20)).

    src_ships, tgt_ships: arrays broadcasting to a common shape.
    Returns int array of ships to send. Slot-0 (NOOP) callers should
    avoid using this entirely.
    """
    src = np.asarray(src_ships, dtype=np.int32)
    tgt = np.asarray(tgt_ships, dtype=np.int32)
    minimum = np.maximum(tgt + 1, 20)
    return np.minimum(src, minimum).astype(np.int32)


# ----------------------------------------------------------------------------
# Top-level agent
# ----------------------------------------------------------------------------

@dataclass
class SimpleRLAgentV2Config:
    hidden: int = 128
    planet_feat_dim: int = PLANET_FEAT_DIM
    global_feat_dim: int = GLOBAL_FEAT_DIM
    cand_feat_dim: int = CAND_FEAT_DIM
    detach_value: bool = True   # Tom Fix #2 default ON


class SimpleRLAgentV2(nn.Module):
    """MLP backbone + hier action head + v14 ship rule + value head.

    Forward returns:
        type_logits  [B, NP, N_TYPES]
        sub_logits   [B, NP, K_CAND]
        value        [B]   (gradient-detached by default)
    Plus the fused planet embedding for downstream introspection.

    Action sampling and log-prob computation live in helper functions
    (act_distribution / log_prob_action) so the trainer can call them
    without re-materialising the joint log-prob distribution.
    """
    def __init__(self, cfg: Optional[SimpleRLAgentV2Config] = None):
        super().__init__()
        self.cfg = cfg or SimpleRLAgentV2Config()
        c = self.cfg
        self.planet_enc = _MLPBackbone(c.planet_feat_dim, c.hidden)
        self.global_enc = _GlobalEncoder(c.global_feat_dim, c.hidden)
        self.hier_head  = _HierHead(c.hidden, c.cand_feat_dim)
        self.value_head = _ValueHead(c.hidden)

    def forward(
        self,
        pf: torch.Tensor,           # [B, NP, planet_feat_dim]
        gf: torch.Tensor,           # [B, global_feat_dim]
        cand_feat: torch.Tensor,    # [B, NP, K, cand_feat_dim]
        src_mask: Optional[torch.Tensor] = None,   # [B, NP] bool — owned planets
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_p = self.planet_enc(pf)                    # [B, NP, H]
        enc_g = self.global_enc(gf).unsqueeze(1)       # [B, 1, H]
        fused = enc_p + enc_g                          # [B, NP, H]

        type_logits, sub_logits = self.hier_head(fused, cand_feat)

        # Pool over OWNED planets only for value (if mask provided)
        if src_mask is not None:
            mask_f = src_mask.float().unsqueeze(-1)    # [B, NP, 1]
            denom = mask_f.sum(dim=1).clamp(min=1.0)   # [B, 1]
            pooled = (fused * mask_f).sum(dim=1) / denom
        else:
            pooled = fused.mean(dim=1)
        if self.cfg.detach_value:
            pooled = pooled.detach()
        value = self.value_head(pooled)                 # [B]
        return type_logits, sub_logits, value, fused


# ----------------------------------------------------------------------------
# Joint distribution helpers
# ----------------------------------------------------------------------------

def _slot_type_table(device: torch.device) -> torch.Tensor:
    """Fixed [K_CAND] tensor mapping slot → type. On device, for indexing."""
    return torch.from_numpy(SLOT_TO_TYPE).to(device)


def joint_log_probs(
    type_logits: torch.Tensor,   # [B, NP, N_TYPES]
    sub_logits: torch.Tensor,    # [B, NP, K_CAND]
    cand_valid: torch.Tensor,    # [B, NP, K_CAND] bool — True = valid candidate
) -> torch.Tensor:
    """Compute joint log-probability for every (env, planet, slot).

    Joint π(type, slot | s) = π(type | s) · π(slot | type, s)

    The slot's type is determined by SLOT_TO_TYPE. Sub-distribution is
    formed by masking sub_logits to slots of the chosen type, then
    softmax-ing over the surviving slots. Invalid candidates (e.g. no
    enemy planets exist → enemy slots invalid) get -inf logits.

    Returns:
        joint_log_p [B, NP, K_CAND]
    """
    B, NP, K = sub_logits.shape
    device = sub_logits.device
    slot_type = _slot_type_table(device)       # [K]

    # Mask invalid candidates with -inf
    masked_sub = sub_logits.masked_fill(~cand_valid, -1e9)

    # For each TYPE, build the per-type softmax over its slots and zero
    # out other-type slots.
    joint = torch.full_like(masked_sub, -1e9)
    log_p_type = F.log_softmax(type_logits, dim=-1)   # [B, NP, N_TYPES]
    for t in range(N_TYPES):
        slots_of_t = (slot_type == t).nonzero(as_tuple=True)[0]   # [n_slots]
        if slots_of_t.numel() == 0:
            continue
        # gather sub_logits for these slots
        sub_t = masked_sub.index_select(2, slots_of_t)    # [B, NP, n_slots]
        log_p_sub = F.log_softmax(sub_t, dim=-1)          # [B, NP, n_slots]
        # log p_type for type t
        log_p_t = log_p_type[..., t].unsqueeze(-1)        # [B, NP, 1]
        # Joint contribution: log p_type[t] + log p_sub[k_within_t]
        joint_t = log_p_t + log_p_sub                     # [B, NP, n_slots]
        # Scatter back to full K positions
        for j, k in enumerate(slots_of_t.tolist()):
            joint[..., k] = joint_t[..., j]

    # Re-mask invalid candidates (a NOOP "type" with no valid slot is OK
    # because slot 0 is always valid by construction)
    joint = joint.masked_fill(~cand_valid, -1e9)
    return joint


def sample_action(
    joint_log_p: torch.Tensor,   # [B, NP, K_CAND]
    src_mask: torch.Tensor,      # [B, NP] bool — only sample from owned planets
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one slot per owned planet.

    Returns:
        chosen_slot  [B, NP] int — index in [0, K_CAND); 0 = NOOP for
                      unowned planets too.
        chosen_logp  [B, NP] float — log π(chosen_slot | …); 0 for unowned
                      slots.
    """
    B, NP, K = joint_log_p.shape
    flat = joint_log_p.view(B * NP, K)
    if deterministic:
        chosen = flat.argmax(dim=-1)
    else:
        # Convert log-prob to probabilities for Categorical sampling
        probs = flat.exp()
        # Replace any NaN / all-zero rows with uniform NOOP
        row_sum = probs.sum(dim=-1, keepdim=True)
        bad = row_sum < 1e-8
        if bad.any():
            uniform = torch.zeros_like(probs)
            uniform[..., 0] = 1.0
            probs = torch.where(bad, uniform, probs)
        chosen = torch.multinomial(probs, num_samples=1).squeeze(-1)
    chosen = chosen.view(B, NP)
    chosen_logp = joint_log_p.gather(2, chosen.unsqueeze(-1)).squeeze(-1)
    # Zero out non-owned slots
    chosen_logp = chosen_logp * src_mask.float()
    chosen = torch.where(src_mask, chosen, torch.zeros_like(chosen))
    return chosen, chosen_logp


def entropy(
    joint_log_p: torch.Tensor,   # [B, NP, K]
    src_mask: torch.Tensor,      # [B, NP] bool
) -> torch.Tensor:
    """Mean per-owned-planet entropy across the joint distribution. [B]"""
    p = joint_log_p.exp()
    H_per_planet = -(p * joint_log_p).sum(dim=-1)        # [B, NP]
    H_per_planet = H_per_planet.masked_fill(~src_mask, 0.0)
    n_owned = src_mask.sum(dim=1).clamp(min=1).float()   # [B]
    return H_per_planet.sum(dim=1) / n_owned


__all__ = [
    "SimpleRLAgentV2",
    "SimpleRLAgentV2Config",
    "joint_log_probs",
    "sample_action",
    "entropy",
    "v14_ship_count",
    "K_CAND", "N_TYPES", "SLOT_TO_TYPE",
    "PLANET_FEAT_DIM", "GLOBAL_FEAT_DIM", "CAND_FEAT_DIM",
    "TYPE_NOOP", "TYPE_SNIPE", "TYPE_EXPAND", "TYPE_REINFORCE",
]
