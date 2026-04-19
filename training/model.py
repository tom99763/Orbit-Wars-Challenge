"""Set-Transformer encoder + per-owned-planet action head.

Input tensors (see featurize.py and wiki/training-design.md):
  planets:        [B, P, 14] — planet features per step
  planet_mask:    [B, P]     — 1 for valid planet rows, 0 for padding
  fleets:         [B, F, 9]  — fleet features; F may be 0
  fleet_mask:     [B, F]
  globals:        [B, 16]
  owned_mask:     [B, P]     — 1 iff this planet is a valid action source
  target_mask:    [B, P, P]  — 1 iff planet j is a valid destination for source i
                                (drops self-target and sun-crossing shots)

Output:
  target_logits:  [B, P, P+1]  — per source, a softmax over P destinations + "pass" class
  bucket_logits:  [B, P, 4]    — 25/50/75/100% of garrison
  value:          [B]          — scalar critic estimate (for PPO phase)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SetAttentionBlock(nn.Module):
    """Pre-LN Transformer block operating on a padded set."""

    def __init__(self, d: int, n_heads: int = 4, ff: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(),
                                nn.Linear(ff * d, d))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        # key_padding_mask: 1 where PAD (MHA convention: True = ignore)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class OrbitAgent(nn.Module):
    def __init__(
        self,
        planet_dim: int = 14,
        fleet_dim: int = 9,
        global_dim: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_buckets: int = 4,
    ):
        super().__init__()
        self.d = d_model
        self.planet_embed = nn.Sequential(nn.Linear(planet_dim, d_model),
                                          nn.GELU(), nn.Linear(d_model, d_model))
        self.fleet_embed = nn.Sequential(nn.Linear(fleet_dim, d_model),
                                         nn.GELU(), nn.Linear(d_model, d_model))
        self.global_embed = nn.Sequential(nn.Linear(global_dim, d_model),
                                          nn.GELU(), nn.Linear(d_model, d_model))
        # Learned type embeddings so attention can tell planets from fleets
        # from the global summary token.
        self.type_embed = nn.Embedding(3, d_model)

        self.layers = nn.ModuleList(
            [SetAttentionBlock(d_model, n_heads) for _ in range(n_layers)]
        )

        # Action heads (applied on each owned planet's token).
        # Destination head: cross-attention to all planet tokens, plus a
        # learned "pass" embedding prepended.
        self.pass_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pass_token, std=0.02)
        self.target_q = nn.Linear(d_model, d_model)
        self.target_k = nn.Linear(d_model, d_model)
        self.bucket_head = nn.Linear(d_model, n_buckets)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                        nn.Linear(d_model, 1))

    def forward(
        self,
        planets: torch.Tensor,          # [B, P, 14]
        planet_mask: torch.Tensor,      # [B, P] bool
        fleets: torch.Tensor,           # [B, F, 9]
        fleet_mask: torch.Tensor,       # [B, F] bool
        globals_: torch.Tensor,         # [B, 16]
        target_mask: torch.Tensor = None,   # [B, P, P]  — forbids sun, self, etc.
    ):
        B, P, _ = planets.shape
        F_ = fleets.shape[1]

        p_tok = self.planet_embed(planets) + self.type_embed.weight[0]  # [B,P,d]
        f_tok = self.fleet_embed(fleets) + self.type_embed.weight[1]    # [B,F,d]
        g_tok = self.global_embed(globals_).unsqueeze(1) + self.type_embed.weight[2]
        # [B,1,d]

        tokens = torch.cat([p_tok, f_tok, g_tok], dim=1)
        g_mask = torch.ones(B, 1, dtype=torch.bool, device=planets.device)
        valid = torch.cat([planet_mask, fleet_mask, g_mask], dim=1)  # [B, P+F+1]

        # MHA expects: True where to IGNORE
        kpm = ~valid
        for blk in self.layers:
            tokens = blk(tokens, kpm)

        planet_tokens = tokens[:, :P, :]                # [B,P,d]
        global_token = tokens[:, -1, :]                 # [B,d]

        # ---- Policy: per-source target scores via scaled dot-product
        pass_tok = self.pass_token.expand(B, 1, self.d)  # [B,1,d]
        all_dests = torch.cat([pass_tok, planet_tokens], dim=1)  # [B,P+1,d]
        q = self.target_q(planet_tokens)                         # [B,P,d]
        k = self.target_k(all_dests)                             # [B,P+1,d]
        target_logits = torch.einsum("bpd,bqd->bpq", q, k) / math.sqrt(self.d)

        # Mask invalid dests. pass (idx 0) always valid; rest by target_mask.
        # Also mask fake/padded planet rows and sources.
        if target_mask is not None:
            # prepend a "pass always allowed" column of ones
            pass_col = torch.ones(B, P, 1, dtype=torch.bool,
                                  device=target_mask.device)
            full_mask = torch.cat([pass_col, target_mask], dim=2)  # [B,P,P+1]
            # -1e9 (not -inf) so CE stays finite even if the expert label
            # happens to fall on a class we mask as unsafe. Masked classes
            # still get ~0 softmax probability.
            target_logits = target_logits.masked_fill(~full_mask, -1e9)
        # Rows for padded-source planets will be ignored by the loss anyway.

        bucket_logits = self.bucket_head(planet_tokens)           # [B,P,4]
        value = self.value_head(global_token).squeeze(-1)         # [B]

        return target_logits, bucket_logits, value


def sun_blocker_mask(planet_xy: torch.Tensor, planet_mask: torch.Tensor,
                     center=(50.0, 50.0), sun_radius: float = 10.0):
    """Return a [B, P, P] bool mask: True iff the line segment from planet i
    to planet j does NOT pass within sun_radius of the sun centre AND i != j.
    Used to pre-reject moves that would immolate the fleet."""
    B, P, _ = planet_xy.shape
    cx, cy = center
    a = planet_xy                                             # [B,P,2]
    # segment from i to j: use point-to-segment distance from (cx, cy)
    ax = a[..., 0:1]                                          # [B,P,1]
    ay = a[..., 1:2]
    bx = a[..., 0:1].transpose(1, 2)                          # [B,1,P]
    by = a[..., 1:2].transpose(1, 2)
    dx = bx - ax                                              # [B,P,P]
    dy = by - ay
    l2 = dx * dx + dy * dy + 1e-9
    tx = cx - ax
    ty = cy - ay
    t = ((tx * dx + ty * dy) / l2).clamp(0.0, 1.0)
    px = ax + t * dx
    py = ay + t * dy
    d = torch.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    ok = d >= sun_radius                                      # [B,P,P]
    # Reject self-target
    eye = torch.eye(P, dtype=torch.bool, device=ok.device).unsqueeze(0)
    ok = ok & ~eye
    # Reject padded-destination columns
    col_valid = planet_mask.unsqueeze(1).expand(-1, P, -1)
    return ok & col_valid
