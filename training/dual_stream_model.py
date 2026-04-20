"""Dual-stream encoder for Orbit Wars — Entity + Spatial (CNN) + Scalar.

Rationale (see memory: design_spatial_vector_field.md):
  - Orbit Wars is a continuous 2D RTS with moving entities (planets orbit,
    fleets traverse, sun blocks). Spatial patterns like "enemy concentration"
    or "territory control" are natural in an image-based representation.
  - Current entity-only Set Transformer (training/model.py) requires many
    attention layers to infer spatial relationships.
  - AlphaStar precedent: dual-stream (entity + spatial + scalar) outperforms
    single-stream encoders for 2D real-time games.

Components:
  1. `rasterize_obs(obs_dict, grid=32) -> np.array [C, H, W]`
     Projects planets/fleets/comets onto a 2D grid with multiple channels.
  2. `SpatialCNN(in_channels, d_model)`
     Small 3-conv-block encoder with global average pooling.
  3. `DualStreamAgent(planet_dim, fleet_dim, global_dim, spatial_channels,
                      d_model, n_heads, n_layers)`
     - Entity stream: reuses OrbitAgent encoder for (planets, fleets, globals)
     - Spatial stream: CNN over rasterized grid
     - Concat global tokens → policy + value heads

Use cases:
  - Phase 4 AlphaZero value net (primary target)
  - Variant-picker policy (as upgrade from entity-only)
  - Imitation BC (if clean retrain is acceptable)
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model import OrbitAgent, SetAttentionBlock
from training.lb1200_agent import fleet_speed  # for velocity field


# -----------------------------------------------------------------------------
# Rasterization
# -----------------------------------------------------------------------------

# Grid channels (indexed below):
#   0-3: owner one-hot (me / enemy_1 / enemy_2 / neutral)  -- 4 channels
#   4:   log(ships_on_planet) density
#   5:   log(production) density
#   6:   fleet velocity x component (signed)
#   7:   fleet velocity y component (signed)
#   8:   fleet ship density (log)
#   9:   sun mask (1 inside sun radius, else 0)
#  10:   is_comet density (comets on this cell)
#  11:   is_own_planet mask (1 where own planet center sits)
N_SPATIAL_CHANNELS = 12

BOARD = 100.0
SUN_CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0


def _cell_coord(x: float, y: float, grid: int) -> tuple[int, int]:
    gx = int(max(0, min(grid - 1, int(x / BOARD * grid))))
    gy = int(max(0, min(grid - 1, int(y / BOARD * grid))))
    return gx, gy


def rasterize_obs(obs: Any, my_player: int, grid: int = 32) -> np.ndarray:
    """Rasterize obs into [N_SPATIAL_CHANNELS, grid, grid] float32 tensor.

    my_player: caller's player index so we can build me/enemy channels.
    """
    def _read(key, default):
        if isinstance(obs, dict):
            return obs.get(key, default) if obs.get(key) is not None else default
        v = getattr(obs, key, None)
        return v if v is not None else default

    planets = _read("planets", []) or []
    fleets = _read("fleets", []) or []
    comet_ids = set(_read("comet_planet_ids", []) or [])

    canvas = np.zeros((N_SPATIAL_CHANNELS, grid, grid), dtype=np.float32)

    # ---- Planets ----
    for p in planets:
        pid, owner, x, y, radius, ships, prod = p
        gx, gy = _cell_coord(float(x), float(y), grid)
        if int(owner) == int(my_player):
            canvas[0, gy, gx] += 1.0           # me
        elif int(owner) == -1:
            canvas[3, gy, gx] += 1.0           # neutral
        else:
            # enemy — bucket by slot parity to distinguish up to 2 enemies
            ch = 1 if int(owner) % 2 == 0 else 2
            canvas[ch, gy, gx] += 1.0
        canvas[4, gy, gx] += math.log1p(max(0, int(ships))) / 8.0
        canvas[5, gy, gx] += math.log1p(max(1, int(prod))) / 3.0
        if int(pid) in comet_ids:
            canvas[10, gy, gx] += 1.0
        if int(owner) == int(my_player):
            canvas[11, gy, gx] = 1.0

    # ---- Fleets (velocity field) ----
    for f in fleets:
        fid, owner, x, y, angle, from_id, ships = f
        gx, gy = _cell_coord(float(x), float(y), grid)
        sp = fleet_speed(max(1, int(ships)))
        # Signed velocity: positive if friendly, negative if enemy — preserves
        # directional info while encoding ownership as sign.
        sign = 1.0 if int(owner) == int(my_player) else -1.0
        canvas[6, gy, gx] += sign * math.cos(float(angle)) * sp / 6.0
        canvas[7, gy, gx] += sign * math.sin(float(angle)) * sp / 6.0
        canvas[8, gy, gx] += math.log1p(max(0, int(ships))) / 8.0

    # ---- Sun mask (static, could precompute) ----
    cx, cy = SUN_CENTER
    for gy in range(grid):
        wy = (gy + 0.5) / grid * BOARD
        for gx in range(grid):
            wx = (gx + 0.5) / grid * BOARD
            if math.hypot(wx - cx, wy - cy) <= SUN_RADIUS:
                canvas[9, gy, gx] = 1.0

    return canvas


# -----------------------------------------------------------------------------
# Spatial CNN
# -----------------------------------------------------------------------------

class SpatialCNN(nn.Module):
    """Small conv-block encoder: [C, H, W] → global pool → d_model vector."""
    def __init__(self, in_channels: int = N_SPATIAL_CHANNELS,
                 d_model: int = 128, grid: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), nn.GELU(),
            nn.BatchNorm2d(128),
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        h = self.conv(x)
        # Global average pool
        h = h.mean(dim=(2, 3))          # [B, 128]
        return self.proj(h)              # [B, d_model]


# -----------------------------------------------------------------------------
# Scalar MLP (for pure-scalar features; globals_ tensor repeated)
# -----------------------------------------------------------------------------

class ScalarMLP(nn.Module):
    def __init__(self, global_dim: int, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, globals_: torch.Tensor) -> torch.Tensor:
        return self.net(globals_)


# -----------------------------------------------------------------------------
# Dual-stream agent
# -----------------------------------------------------------------------------

class DualStreamAgent(nn.Module):
    """Entity (Set Transformer) + Spatial (CNN) + Scalar (MLP).

    Heads: variable based on `heads` arg. Default: policy (K-way softmax) + value (scalar).
    """
    def __init__(
        self,
        planet_dim: int,
        fleet_dim: int,
        global_dim: int,
        n_variants: int = 8,
        d_entity: int = 128,
        d_spatial: int = 128,
        d_scalar: int = 64,
        d_fuse: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        grid: int = 32,
    ):
        super().__init__()
        # Entity stream (reuse OrbitAgent encoder — only the encoder, not its action heads)
        self.entity_enc = OrbitAgent(
            planet_dim=planet_dim, fleet_dim=fleet_dim, global_dim=global_dim,
            d_model=d_entity, n_heads=n_heads, n_layers=n_layers,
        )
        # Spatial stream
        self.spatial_enc = SpatialCNN(in_channels=N_SPATIAL_CHANNELS,
                                      d_model=d_spatial, grid=grid)
        # Scalar stream (redundant with globals inside entity but useful for shortcut)
        self.scalar_enc = ScalarMLP(global_dim=global_dim, d_model=d_scalar)

        # Fuse
        self.fuse = nn.Sequential(
            nn.Linear(d_entity + d_spatial + d_scalar, d_fuse), nn.GELU(),
            nn.Linear(d_fuse, d_fuse), nn.GELU(),
        )

        # Heads
        self.policy_head = nn.Linear(d_fuse, n_variants)
        self.value_head = nn.Linear(d_fuse, 1)

    def _entity_global_tok(self, planets, pmask, fleets, fmask, globals_):
        """Re-implement the OrbitAgent encoder forward to extract global token only."""
        enc = self.entity_enc
        B = planets.shape[0]
        p_tok = enc.planet_embed(planets) + enc.type_embed.weight[0]
        f_tok = enc.fleet_embed(fleets) + enc.type_embed.weight[1]
        g_tok = enc.global_embed(globals_).unsqueeze(1) + enc.type_embed.weight[2]
        tokens = torch.cat([p_tok, f_tok, g_tok], dim=1)
        g_mask = torch.ones(B, 1, dtype=torch.bool, device=planets.device)
        valid = torch.cat([pmask, fmask, g_mask], dim=1)
        for blk in enc.layers:
            tokens = blk(tokens, ~valid)
        return tokens[:, -1, :]     # [B, d_entity]

    def forward(
        self,
        planets: torch.Tensor,
        planet_mask: torch.Tensor,
        fleets: torch.Tensor,
        fleet_mask: torch.Tensor,
        globals_: torch.Tensor,
        spatial: torch.Tensor,      # [B, C, H, W]
    ):
        e = self._entity_global_tok(planets, planet_mask, fleets, fleet_mask, globals_)
        s = self.spatial_enc(spatial)
        sc = self.scalar_enc(globals_)
        fused = self.fuse(torch.cat([e, s, sc], dim=-1))
        return self.policy_head(fused), self.value_head(fused).squeeze(-1)


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from featurize import PLANET_DIM, FLEET_DIM, GLOBAL_DIM
    from kaggle_environments import make

    model = DualStreamAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
        n_variants=8,
    )
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Make a dummy obs
    env = make("orbit_wars", debug=False); env.reset(num_agents=2)
    for _ in range(10):
        env.step([[], []])
    obs = env.state[0].observation
    my_player = 0

    # Rasterize
    canvas = rasterize_obs(obs, my_player, grid=32)
    print(f"spatial shape: {canvas.shape}, channel sums: {canvas.sum(axis=(1,2))[:6]}")

    # Dummy tensors for entity stream (use feature-zero as placeholder for shape only)
    B = 1; P = 24; F_ = 8
    planets = torch.randn(B, P, PLANET_DIM)
    pmask = torch.ones(B, P, dtype=torch.bool)
    fleets = torch.randn(B, F_, FLEET_DIM)
    fmask = torch.ones(B, F_, dtype=torch.bool)
    globals_ = torch.randn(B, GLOBAL_DIM)
    spatial = torch.from_numpy(canvas).unsqueeze(0)

    with torch.no_grad():
        policy_logits, value = model(planets, pmask, fleets, fmask, globals_, spatial)
    print(f"policy shape: {policy_logits.shape}, value shape: {value.shape}")
    print("✓ smoke test passes")
