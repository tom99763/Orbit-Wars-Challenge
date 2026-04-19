"""Wrap a trained OrbitAgent checkpoint into a Kaggle-grader-compatible
`agent(obs)` function for local evaluation via `kaggle_environments.make`.

Usage in code:
    from training.agent import load_agent
    agent_fn = load_agent("training/checkpoints/bc_v1.pt")
    env.run([agent_fn, "random"])

The function translates observation → feature tensors (mirroring
featurize.py), runs the model, then decodes `(target_planet, ships_bucket)`
back into the env's `[src, angle, ships]` action format.
"""

from __future__ import annotations

import math
import pathlib
from typing import Callable

import numpy as np
import torch

from training.model import OrbitAgent, sun_blocker_mask


BOARD = 100.0
CENTER = 50.0
ROT_LIMIT = 50.0
SHIPS_BUCKETS = (0.25, 0.50, 0.75, 1.00)
COMET_SPAWN = {50, 150, 250, 350, 450}
COMET_WIN = set()
for s in COMET_SPAWN:
    for d in range(-5, 6):
        COMET_WIN.add(s + d)


def _encode_obs(obs: dict, max_planets: int = 64, max_fleets: int = 64):
    planets_raw = obs.get("planets") or []
    fleets_raw = obs.get("fleets") or []
    player = int(obs.get("player", 0))
    ang_vel = float(obs.get("angular_velocity") or 0.0)
    step = int(obs.get("step") or 0)
    initial_planets = obs.get("initial_planets") or []
    init_ids = {int(p[0]) for p in initial_planets}

    N = min(len(planets_raw), max_planets)
    F = min(len(fleets_raw), max_fleets)

    planet_feat = np.zeros((N, 14), dtype=np.float32)
    planet_xy = np.zeros((N, 2), dtype=np.float32)
    planet_ids = np.zeros((N,), dtype=np.int64)
    action_mask = np.zeros((N,), dtype=bool)
    for i, p in enumerate(planets_raw[:N]):
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

    fleet_feat = np.zeros((F, 9), dtype=np.float32)
    max_pid = max(int(planet_ids.max() if N else 1), 1)
    for i, f in enumerate(fleets_raw[:F]):
        fid, owner, x, y, ang, from_id, ships = f
        fleet_feat[i, 0] = 1.0 if owner == player else 0.0
        fleet_feat[i, 1] = 1.0 if owner != player else 0.0
        fleet_feat[i, 2] = (x - CENTER) / CENTER
        fleet_feat[i, 3] = (y - CENTER) / CENTER
        fleet_feat[i, 4] = math.sin(ang)
        fleet_feat[i, 5] = math.cos(ang)
        fleet_feat[i, 6] = math.log1p(max(0, ships)) / 8.0
        fleet_feat[i, 7] = from_id / max_pid
        # eta_norm rough
        speed = 1.0 + 5.0 * (math.log(max(1, ships)) / math.log(1000)) ** 1.5
        speed = min(speed, 6.0)
        fleet_feat[i, 8] = 50.0 / (speed + 0.1) / 50.0

    # Count ships quickly
    my_total = sum(p[5] for p in planets_raw if p[1] == player) + sum(
        f[6] for f in fleets_raw if f[1] == player
    )
    enemy_total = sum(p[5] for p in planets_raw if p[1] != player and p[1] != -1) + sum(
        f[6] for f in fleets_raw if f[1] != player
    )
    n_my_pl = sum(1 for p in planets_raw if p[1] == player)
    n_en_pl = sum(1 for p in planets_raw if p[1] != player and p[1] != -1)
    n_nu_pl = sum(1 for p in planets_raw if p[1] == -1)
    # n_players we can't know exactly from obs — infer from distinct owners
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


def load_agent(ckpt_path: str, device: str = "cpu",
               temperature: float = 0.0) -> Callable[[dict], list]:
    """Load a trained OrbitAgent checkpoint as a kaggle `agent(obs)` fn.
    temperature > 0 enables stochastic sampling (useful for self-play
    diversity); temperature == 0 means argmax (for evaluation)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = OrbitAgent(**ckpt["kwargs"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from physics_aim import compute_aim_angle

    def agent(obs: dict) -> list:
        obs = obs if isinstance(obs, dict) else dict(obs)
        (pf, pxy, pids, omask, ff, g, planets_raw, player) = _encode_obs(obs)
        if not omask.any() or len(planets_raw) == 0:
            return []
        ang_vel = float(obs.get("angular_velocity") or 0.0)
        init_planets = list(obs.get("initial_planets") or [])
        comets = list(obs.get("comets") or [])
        comet_ids = set(obs.get("comet_planet_ids") or [])
        with torch.no_grad():
            planets = torch.from_numpy(pf).unsqueeze(0).to(device)
            planet_xy = torch.from_numpy(pxy).unsqueeze(0).to(device)
            planet_mask = torch.ones((1, pf.shape[0]), dtype=torch.bool, device=device)
            if ff.shape[0] > 0:
                fleets = torch.from_numpy(ff).unsqueeze(0).to(device)
                fleet_mask = torch.ones((1, ff.shape[0]), dtype=torch.bool, device=device)
            else:
                fleets = torch.zeros((1, 1, 9), device=device)
                fleet_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)
            globals_ = torch.from_numpy(g).unsqueeze(0).to(device)
            tgt_mask = sun_blocker_mask(planet_xy, planet_mask)
            tgt_logits, bkt_logits, _ = model(
                planets, planet_mask, fleets, fleet_mask, globals_, tgt_mask
            )
        moves = []
        owned_indices = np.where(omask)[0]
        tgt_l = tgt_logits[0].cpu().numpy()      # [P, P+1]
        bkt_l = bkt_logits[0].cpu().numpy()      # [P, 4]
        for si in owned_indices:
            row = tgt_l[si]
            if temperature > 0:
                # Stable softmax + sample
                z = row / max(temperature, 1e-6)
                z = z - z.max()
                p = np.exp(z); p = p / p.sum()
                tgt_class = int(np.random.choice(len(p), p=p))
            else:
                tgt_class = int(row.argmax())
            if tgt_class == 0:
                continue  # pass
            tgt_idx = tgt_class - 1
            if temperature > 0:
                z = bkt_l[si] / max(temperature, 1e-6)
                z = z - z.max()
                p = np.exp(z); p = p / p.sum()
                bkt_idx = int(np.random.choice(len(p), p=p))
            else:
                bkt_idx = int(bkt_l[si].argmax())
            frac = SHIPS_BUCKETS[bkt_idx]
            src = planets_raw[si]
            tgt = planets_raw[tgt_idx]
            garrison = int(src[5])
            num_ships = max(1, int(round(frac * garrison)))
            if num_ships <= 0 or num_ships > garrison:
                continue
            angle = compute_aim_angle(
                src, tgt, num_ships,
                ang_vel, init_planets, comets, comet_ids,
            )
            moves.append([int(src[0]), float(angle), int(num_ships)])
        return moves

    return agent
