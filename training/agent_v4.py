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
from featurize import featurize_step


BOARD = 100.0
CENTER = 50.0
ROT_LIMIT = 50.0
SHIPS_BUCKETS = (0.25, 0.50, 0.75, 1.00)
COMET_SPAWN = {50, 150, 250, 350, 450}
COMET_WIN = set()
for s in COMET_SPAWN:
    for d in range(-5, 6):
        COMET_WIN.add(s + d)


def _encode_obs(obs: dict, session=None,
                max_planets: int = 64, max_fleets: int = 64):
    """Inference-time obs encoder. Delegates to featurize.featurize_step so
    the feature schema always matches training. The optional `session` dict
    carries sliding-window history across successive calls within one game:
        session["obs_history"]           deque[dict]
        session["action_history"]        deque[(src_pid, tgt_pid, bkt, turn)]
        session["last_actions_by_planet"] dict[pid → (tgt_pid, bkt, turn, n)]
        session["cumulative_stats"]      dict
        session["last_step"]             int   # reset detector
    """
    planets_raw = obs.get("planets") or []
    fleets_raw = obs.get("fleets") or []
    player = int(obs.get("player", 0))
    ang_vel = float(obs.get("angular_velocity") or 0.0)
    step = int(obs.get("step") or 0)
    initial_planets = obs.get("initial_planets") or []
    init_ids = {int(p[0]) for p in initial_planets}

    owners = {p[1] for p in planets_raw if p[1] != -1} | {f[1] for f in fleets_raw}
    n_players = 4 if len(owners) > 2 else 2

    comet_ids = {int(p[0]) for p in planets_raw if int(p[0]) not in init_ids}

    my_total = sum(p[5] for p in planets_raw if p[1] == player) + sum(
        f[6] for f in fleets_raw if f[1] == player)
    enemy_total = sum(p[5] for p in planets_raw if p[1] != player and p[1] != -1) + sum(
        f[6] for f in fleets_raw if f[1] != player)
    n_my = sum(1 for p in planets_raw if p[1] == player)
    n_en = sum(1 for p in planets_raw if p[1] != player and p[1] != -1)
    n_nu = sum(1 for p in planets_raw if p[1] == -1)

    step_dict = {
        "step": step,
        "planets": list(planets_raw[:max_planets]),
        "fleets": list(fleets_raw[:max_fleets]),
        "action": [],
        "my_total_ships": my_total,
        "enemy_total_ships": enemy_total,
        "my_planet_count": n_my,
        "enemy_planet_count": n_en,
        "neutral_planet_count": n_nu,
    }
    obs_hist = list(session["obs_history"]) if session else []
    act_hist = list(session["action_history"]) if session else []
    last_actions = session["last_actions_by_planet"] if session else {}
    cum_stats = session["cumulative_stats"] if session else None
    feat = featurize_step(
        step_dict, player, ang_vel, n_players, initial_planets, comet_ids,
        last_actions_by_planet=last_actions, cumulative_stats=cum_stats,
        obs_history=obs_hist, action_history=act_hist,
    )

    return (feat["planets"], feat["planet_xy"], feat["planet_ids"].astype(np.int64),
            feat["action_mask_owned"], feat["fleets"], feat["globals"],
            list(planets_raw[:max_planets]), player)


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
    from featurize import ship_bucket_idx, HISTORY_K, nearest_target_index
    import collections

    session = {
        "obs_history": collections.deque(maxlen=HISTORY_K),
        "action_history": collections.deque(maxlen=HISTORY_K),
        "last_actions_by_planet": {},
        "cumulative_stats": {"total_ships_sent": 0, "total_actions": 0},
        "last_step": -1,
    }

    def agent(obs: dict) -> list:
        obs = obs if isinstance(obs, dict) else dict(obs)
        step = int(obs.get("step") or 0)
        # Reset session if we detect game restart (step went backwards)
        if step < session["last_step"]:
            session["obs_history"].clear()
            session["action_history"].clear()
            session["last_actions_by_planet"].clear()
            session["cumulative_stats"] = {"total_ships_sent": 0, "total_actions": 0}
        session["last_step"] = step

        (pf, pxy, pids, omask, ff, g, planets_raw, player) = _encode_obs(obs, session)
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
                from featurize import FLEET_DIM
                fleets = torch.zeros((1, 1, FLEET_DIM), device=device)
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
        # Update session history with this turn's obs + actions
        session["obs_history"].append({"planets": list(planets_raw), "step": step})
        for m in moves:
            src_pid, ang, ships = m
            # find planet in planets_raw for target inference
            src_p = None
            for p in planets_raw:
                if int(p[0]) == int(src_pid):
                    src_p = p
                    break
            if src_p is None:
                continue
            tgt_i = nearest_target_index(src_p, ang, planets_raw)
            tgt_pid = int(planets_raw[tgt_i][0]) if tgt_i is not None else -1
            garrison = int(src_p[5]) + int(ships)
            bkt_idx = ship_bucket_idx(int(ships), max(1, garrison))
            prev = session["last_actions_by_planet"].get(int(src_pid),
                                                          (-1, 0, -1, 0))
            session["last_actions_by_planet"][int(src_pid)] = (
                tgt_pid, bkt_idx, step, prev[3] + 1
            )
            session["cumulative_stats"]["total_ships_sent"] += int(ships)
            session["cumulative_stats"]["total_actions"] += 1
            session["action_history"].append(
                (int(src_pid), tgt_pid, bkt_idx, step))
        return moves

    return agent
