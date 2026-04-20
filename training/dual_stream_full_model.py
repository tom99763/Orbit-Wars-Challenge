"""DualStreamFullAgent — Entity+Spatial+Scalar encoder with OrbitAgent-style
per-planet action heads (not K=8 variant picker).

Rationale (user request 2026-04-20):
  - V2 picker is limited to K=8 mechanical variants of lb-1200's primary
  - To escape that ceiling, neural net must GENERATE actions itself
  - But keep physics primitives (aim_with_prediction) for angle computation
    + sun avoidance — we don't try to learn physics from scratch

Architecture:
  Input: (planets, fleets, globals_, spatial)
  Encoder streams:
    - Entity: Set Transformer → planet tokens [P, d] + fleet tokens + global token
    - Spatial: CNN over 12-channel grid → flat vector
    - Scalar: MLP over globals → flat vector
  Fusion:
    fused_global = concat(entity_global, spatial, scalar) → MLP → [d]
    fused_planet_tokens = planet_tokens + fused_global.unsqueeze(1)  (broadcast)
  Action heads (per planet):
    tgt_logits [P, P+1] — softmax over destinations (P planets + pass)
    bkt_logits [P, n_buckets] — softmax over ship fractions
  Value head:
    value [scalar] — learned state value

At inference / MCTS:
  For each owned planet (source):
    sample target ~ softmax(tgt_logits[src])
    sample bucket ~ softmax(bkt_logits[src])
    if target == pass: skip
    ships = planet.ships * bucket_fraction
    angle = aim_with_prediction(source, target_planet, ships)  # physics
    add [src_id, angle, ships] to action list

MCTS uses `sample_k_distinct_actions` (from training/mcts.py) to branch the tree.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model import OrbitAgent, SetAttentionBlock
from training.dual_stream_model import (
    SpatialCNN, ScalarMLP, N_SPATIAL_CHANNELS, rasterize_obs,
)


class DualStreamFullAgent(nn.Module):
    """Full-action-space agent: per-planet tgt + bkt logits (no K=8 variant limit).

    Action heads follow OrbitAgent structure (pointer-attention target + MLP bucket).
    """
    def __init__(
        self,
        planet_dim: int,
        fleet_dim: int,
        global_dim: int,
        d_entity: int = 128,
        d_spatial: int = 128,
        d_scalar: int = 64,
        n_buckets: int = 4,
        n_heads: int = 4,
        n_layers: int = 3,
    ):
        super().__init__()
        self.d = d_entity
        # Entity encoder — same pattern as OrbitAgent's encode path
        self.planet_embed = nn.Sequential(
            nn.Linear(planet_dim, d_entity), nn.GELU(), nn.Linear(d_entity, d_entity))
        self.fleet_embed = nn.Sequential(
            nn.Linear(fleet_dim, d_entity), nn.GELU(), nn.Linear(d_entity, d_entity))
        self.global_embed = nn.Sequential(
            nn.Linear(global_dim, d_entity), nn.GELU(), nn.Linear(d_entity, d_entity))
        self.type_embed = nn.Embedding(3, d_entity)
        self.attn_layers = nn.ModuleList(
            [SetAttentionBlock(d_entity, n_heads) for _ in range(n_layers)]
        )

        # Spatial + Scalar streams
        self.spatial_enc = SpatialCNN(
            in_channels=N_SPATIAL_CHANNELS, d_model=d_spatial, grid=32
        )
        self.scalar_enc = ScalarMLP(global_dim=global_dim, d_model=d_scalar)

        # Fusion of the 3 global representations → per-planet context
        self.fuse_global = nn.Sequential(
            nn.Linear(d_entity + d_spatial + d_scalar, d_entity), nn.GELU(),
            nn.Linear(d_entity, d_entity),
        )

        # Action heads (per-planet, pointer-attention style)
        self.pass_token = nn.Parameter(torch.zeros(1, 1, d_entity))
        nn.init.normal_(self.pass_token, std=0.02)
        self.target_q = nn.Linear(d_entity, d_entity)
        self.target_k = nn.Linear(d_entity, d_entity)
        self.bucket_head = nn.Linear(d_entity, n_buckets)

        # Value head (from fused global)
        self.value_head = nn.Sequential(
            nn.Linear(d_entity, d_entity), nn.GELU(),
            nn.Linear(d_entity, 1),
        )

    def forward(
        self,
        planets: torch.Tensor,          # [B, P, planet_dim]
        planet_mask: torch.Tensor,      # [B, P] bool
        fleets: torch.Tensor,           # [B, F, fleet_dim]
        fleet_mask: torch.Tensor,       # [B, F] bool
        globals_: torch.Tensor,         # [B, global_dim]
        spatial: torch.Tensor,          # [B, C, H, W]
        target_mask: Optional[torch.Tensor] = None,   # [B, P, P] — sun/self masking
    ):
        B, P, _ = planets.shape

        # --- Entity encoder (inline for global token + planet tokens) ---
        p_tok = self.planet_embed(planets) + self.type_embed.weight[0]       # [B,P,d]
        f_tok = self.fleet_embed(fleets) + self.type_embed.weight[1]         # [B,F,d]
        g_tok = self.global_embed(globals_).unsqueeze(1) + self.type_embed.weight[2]  # [B,1,d]

        tokens = torch.cat([p_tok, f_tok, g_tok], dim=1)
        g_mask_tf = torch.ones(B, 1, dtype=torch.bool, device=planets.device)
        valid = torch.cat([planet_mask, fleet_mask, g_mask_tf], dim=1)
        for blk in self.attn_layers:
            tokens = blk(tokens, ~valid)

        planet_tokens = tokens[:, :P, :]                                      # [B,P,d]
        entity_global = tokens[:, -1, :]                                      # [B,d]

        # --- Spatial + Scalar streams ---
        spatial_feat = self.spatial_enc(spatial)                              # [B,d_spatial]
        scalar_feat = self.scalar_enc(globals_)                               # [B,d_scalar]

        # --- Fusion + per-planet context injection ---
        fused_g = self.fuse_global(
            torch.cat([entity_global, spatial_feat, scalar_feat], dim=-1)
        )                                                                      # [B,d]
        fused_planet_tokens = planet_tokens + fused_g.unsqueeze(1)           # [B,P,d]

        # --- Target head (pointer-attention) ---
        pass_tok = self.pass_token.expand(B, 1, self.d)
        all_dests = torch.cat([pass_tok, planet_tokens], dim=1)               # [B,P+1,d]
        q = self.target_q(fused_planet_tokens)                                # [B,P,d]
        k = self.target_k(all_dests)                                          # [B,P+1,d]
        tgt_logits = torch.einsum("bpd,bqd->bpq", q, k) / math.sqrt(self.d)   # [B,P,P+1]

        if target_mask is not None:
            pass_col = torch.ones(B, P, 1, dtype=torch.bool,
                                  device=target_mask.device)
            full_mask = torch.cat([pass_col, target_mask], dim=2)
            tgt_logits = tgt_logits.masked_fill(~full_mask, -1e9)

        # --- Bucket head (per-planet) ---
        bkt_logits = self.bucket_head(fused_planet_tokens)                    # [B,P,n_buckets]

        # --- Value head ---
        value = self.value_head(fused_g).squeeze(-1)                          # [B]

        return tgt_logits, bkt_logits, value


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from featurize import PLANET_DIM, FLEET_DIM, GLOBAL_DIM
    from kaggle_environments import make

    m = DualStreamFullAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    )
    print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    env = make("orbit_wars", debug=False); env.reset(num_agents=2)
    for _ in range(10):
        env.step([[], []])
    obs = env.state[0].observation

    B = 1; P = 24; F_ = 8
    planets = torch.randn(B, P, PLANET_DIM)
    pmask = torch.ones(B, P, dtype=torch.bool)
    fleets = torch.randn(B, F_, FLEET_DIM)
    fmask = torch.ones(B, F_, dtype=torch.bool)
    globals_ = torch.randn(B, GLOBAL_DIM)
    spatial = torch.from_numpy(rasterize_obs(obs, 0, grid=32)).unsqueeze(0)

    with torch.no_grad():
        tgt, bkt, v = m(planets, pmask, fleets, fmask, globals_, spatial)
    print(f"tgt_logits: {tgt.shape}  bkt_logits: {bkt.shape}  value: {v.shape}")
    print("✓ smoke test passes")
