"""Kaggle submission entry point for bc_v2.

A Set-Transformer policy trained via behaviour cloning on 830 trajectories
(500 starter-curriculum + 41 top-10 winners + earlier scrapes). Loads
model.pt once at import time, returns per-owned-planet (target, bucket)
actions at inference.
"""

import math
import pathlib
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------
# Model (copied from training/model.py to keep submission self-contained)
# ---------------------------------------------------------------

class SetAttentionBlock(nn.Module):
    def __init__(self, d: int, n_heads: int = 4, ff: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(),
                                nn.Linear(ff * d, d))

    def forward(self, x, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class OrbitAgent(nn.Module):
    def __init__(self, planet_dim=14, fleet_dim=9, global_dim=16,
                 d_model=128, n_heads=4, n_layers=4, n_buckets=4):
        super().__init__()
        self.d = d_model
        self.planet_embed = nn.Sequential(nn.Linear(planet_dim, d_model),
                                          nn.GELU(), nn.Linear(d_model, d_model))
        self.fleet_embed = nn.Sequential(nn.Linear(fleet_dim, d_model),
                                         nn.GELU(), nn.Linear(d_model, d_model))
        self.global_embed = nn.Sequential(nn.Linear(global_dim, d_model),
                                          nn.GELU(), nn.Linear(d_model, d_model))
        self.type_embed = nn.Embedding(3, d_model)
        self.layers = nn.ModuleList(
            [SetAttentionBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.pass_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.target_q = nn.Linear(d_model, d_model)
        self.target_k = nn.Linear(d_model, d_model)
        self.bucket_head = nn.Linear(d_model, n_buckets)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                        nn.Linear(d_model, 1))

    def forward(self, planets, planet_mask, fleets, fleet_mask, globals_,
                target_mask=None):
        B, P, _ = planets.shape
        F_ = fleets.shape[1]
        p_tok = self.planet_embed(planets) + self.type_embed.weight[0]
        f_tok = self.fleet_embed(fleets) + self.type_embed.weight[1]
        g_tok = self.global_embed(globals_).unsqueeze(1) + self.type_embed.weight[2]
        tokens = torch.cat([p_tok, f_tok, g_tok], dim=1)
        g_mask = torch.ones(B, 1, dtype=torch.bool, device=planets.device)
        valid = torch.cat([planet_mask, fleet_mask, g_mask], dim=1)
        kpm = ~valid
        for blk in self.layers:
            tokens = blk(tokens, kpm)
        planet_tokens = tokens[:, :P, :]
        global_token = tokens[:, -1, :]
        pass_tok = self.pass_token.expand(B, 1, self.d)
        all_dests = torch.cat([pass_tok, planet_tokens], dim=1)
        q = self.target_q(planet_tokens)
        k = self.target_k(all_dests)
        target_logits = torch.einsum("bpd,bqd->bpq", q, k) / math.sqrt(self.d)
        if target_mask is not None:
            pass_col = torch.ones(B, P, 1, dtype=torch.bool,
                                  device=target_mask.device)
            full_mask = torch.cat([pass_col, target_mask], dim=2)
            target_logits = target_logits.masked_fill(~full_mask, -1e9)
        bucket_logits = self.bucket_head(planet_tokens)
        value = self.value_head(global_token).squeeze(-1)
        return target_logits, bucket_logits, value


def sun_blocker_mask(planet_xy, planet_mask, center=(50.0, 50.0), sun_radius=10.0):
    B, P, _ = planet_xy.shape
    cx, cy = center
    a = planet_xy
    ax, ay = a[..., 0:1], a[..., 1:2]
    bx, by = a[..., 0:1].transpose(1, 2), a[..., 1:2].transpose(1, 2)
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy + 1e-9
    tx, ty = cx - ax, cy - ay
    t = ((tx * dx + ty * dy) / l2).clamp(0.0, 1.0)
    px, py = ax + t * dx, ay + t * dy
    d = torch.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    ok = d >= sun_radius
    eye = torch.eye(P, dtype=torch.bool, device=ok.device).unsqueeze(0)
    ok = ok & ~eye
    col_valid = planet_mask.unsqueeze(1).expand(-1, P, -1)
    return ok & col_valid


# ---------------------------------------------------------------
# Obs encoder (copied from training/agent.py)
# ---------------------------------------------------------------

BOARD = 100.0
CENTER = 50.0
ROT_LIMIT = 50.0
SHIPS_BUCKETS = (0.25, 0.50, 0.75, 1.00)
COMET_SPAWN = {50, 150, 250, 350, 450}
COMET_WIN = set()
for s in COMET_SPAWN:
    for d_ in range(-5, 6):
        COMET_WIN.add(s + d_)


def _encode_obs(obs):
    planets_raw = obs.get("planets") or []
    fleets_raw = obs.get("fleets") or []
    player = int(obs.get("player", 0))
    ang_vel = float(obs.get("angular_velocity") or 0.0)
    step = int(obs.get("step") or 0)
    initial_planets = obs.get("initial_planets") or []
    init_ids = {int(p[0]) for p in initial_planets}

    N = len(planets_raw)
    F_ = len(fleets_raw)
    planet_feat = np.zeros((N, 14), dtype=np.float32)
    planet_xy = np.zeros((N, 2), dtype=np.float32)
    planet_ids = np.zeros((N,), dtype=np.int64)
    action_mask = np.zeros((N,), dtype=bool)
    for i, p in enumerate(planets_raw):
        pid, owner, x, y, r, ships, prod = p
        planet_ids[i] = pid
        planet_xy[i] = (x, y)
        planet_feat[i, 0] = 1.0 if owner == player else 0.0
        planet_feat[i, 1] = 1.0 if (owner != player and owner != -1) else 0.0
        planet_feat[i, 2] = 1.0 if owner == -1 else 0.0
        planet_feat[i, 3] = (x - CENTER) / CENTER
        planet_feat[i, 4] = (y - CENTER) / CENTER
        planet_feat[i, 5] = r
        planet_feat[i, 6] = math.log1p(max(0, ships)) / 8.0
        if 1 <= prod <= 5:
            planet_feat[i, 6 + prod] = 1.0
        orb_r = math.hypot(x - CENTER, y - CENTER)
        planet_feat[i, 12] = 1.0 if (orb_r + r >= ROT_LIMIT) else 0.0
        planet_feat[i, 13] = 1.0 if int(pid) not in init_ids else 0.0
        if owner == player and ships > 0:
            action_mask[i] = True

    fleet_feat = np.zeros((F_, 9), dtype=np.float32)
    max_pid = max(int(planet_ids.max() if N else 1), 1)
    for i, f in enumerate(fleets_raw):
        fid, owner, x, y, ang, from_id, ships = f
        fleet_feat[i, 0] = 1.0 if owner == player else 0.0
        fleet_feat[i, 1] = 1.0 if owner != player else 0.0
        fleet_feat[i, 2] = (x - CENTER) / CENTER
        fleet_feat[i, 3] = (y - CENTER) / CENTER
        fleet_feat[i, 4] = math.sin(ang)
        fleet_feat[i, 5] = math.cos(ang)
        fleet_feat[i, 6] = math.log1p(max(0, ships)) / 8.0
        fleet_feat[i, 7] = from_id / max_pid
        speed = 1.0 + 5.0 * (math.log(max(1, ships)) / math.log(1000)) ** 1.5
        fleet_feat[i, 8] = 50.0 / (min(speed, 6.0) + 0.1) / 50.0

    my_total = sum(p[5] for p in planets_raw if p[1] == player) + sum(
        f[6] for f in fleets_raw if f[1] == player)
    enemy_total = sum(p[5] for p in planets_raw if p[1] != player and p[1] != -1) + sum(
        f[6] for f in fleets_raw if f[1] != player)
    n_my_pl = sum(1 for p in planets_raw if p[1] == player)
    n_en_pl = sum(1 for p in planets_raw if p[1] != player and p[1] != -1)
    n_nu_pl = sum(1 for p in planets_raw if p[1] == -1)
    owners = {p[1] for p in planets_raw if p[1] != -1} | {f[1] for f in fleets_raw}
    n_players = 4 if len(owners) > 2 else 2

    g = np.zeros((16,), dtype=np.float32)
    g[0] = step / 500.0
    g[1] = ang_vel
    g[2] = max(0.0, (500 - step) / 500.0)
    if 0 <= player <= 3:
        g[3 + player] = 1.0
    g[7] = 1.0 if n_players == 4 else 0.0
    g[8] = n_my_pl / 20.0
    g[9] = n_en_pl / 20.0
    g[10] = n_nu_pl / 20.0
    g[11] = math.log1p(my_total) / 8.0
    g[12] = math.log1p(enemy_total) / 8.0
    g[13] = 1.0 if step in COMET_WIN else 0.0
    phase = ang_vel * step
    g[14] = math.sin(phase)
    g[15] = math.cos(phase)

    return (planet_feat, planet_xy, planet_ids, action_mask,
            fleet_feat, g, planets_raw, player)


# ---------------------------------------------------------------
# Load the model once at import time
# ---------------------------------------------------------------

_MODEL_PATH = pathlib.Path(__file__).parent / "model.pt"
_model = None


def _get_model():
    """Lazy-load: defers ~1-1.5s of torch.load off the import path.
    Kaggle grader gives 2s of bankable overage; first agent() call will
    consume some of it. Subsequent calls are fast (~5ms)."""
    global _model
    if _model is None:
        ckpt = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
        m = OrbitAgent(**ckpt["kwargs"])
        m.load_state_dict(ckpt["model"])
        m.eval()
        _model = m
    return _model


def agent(obs):
    obs = obs if isinstance(obs, dict) else dict(obs)
    pf, pxy, pids, omask, ff, g, planets_raw, player = _encode_obs(obs)
    if not omask.any() or len(planets_raw) == 0:
        return []
    with torch.no_grad():
        planets = torch.from_numpy(pf).unsqueeze(0)
        planet_xy = torch.from_numpy(pxy).unsqueeze(0)
        planet_mask = torch.ones((1, pf.shape[0]), dtype=torch.bool)
        if ff.shape[0] > 0:
            fleets = torch.from_numpy(ff).unsqueeze(0)
            fleet_mask = torch.ones((1, ff.shape[0]), dtype=torch.bool)
        else:
            fleets = torch.zeros((1, 1, 9), dtype=torch.float32)
            fleet_mask = torch.zeros((1, 1), dtype=torch.bool)
        globals_ = torch.from_numpy(g).unsqueeze(0)
        tgt_mask = sun_blocker_mask(planet_xy, planet_mask)
        tgt_logits, bkt_logits, _ = _get_model()(
            planets, planet_mask, fleets, fleet_mask, globals_, tgt_mask
        )
    moves = []
    owned_indices = np.where(omask)[0]
    tgt_l = tgt_logits[0].cpu().numpy()
    bkt_l = bkt_logits[0].cpu().numpy()
    for si in owned_indices:
        row = tgt_l[si]
        tgt_class = int(row.argmax())
        if tgt_class == 0:
            continue
        tgt_idx = tgt_class - 1
        if tgt_idx >= len(planets_raw):
            continue
        bkt_idx = int(bkt_l[si].argmax())
        frac = SHIPS_BUCKETS[bkt_idx]
        src = planets_raw[si]
        tgt = planets_raw[tgt_idx]
        garrison = int(src[5])
        num_ships = max(1, int(round(frac * garrison)))
        if num_ships <= 0 or num_ships > garrison:
            continue
        angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
        moves.append([int(src[0]), float(angle), int(num_ships)])
    return moves
