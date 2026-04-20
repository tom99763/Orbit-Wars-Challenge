"""Variant Picker PPO v2 — Dual-Stream encoder (Entity + Spatial + Scalar).

Differences from v1 (training/variant_picker_ppo.py):
  1. Uses `DualStreamAgent` (Entity transformer + Spatial CNN + Scalar MLP).
  2. Adds `rasterize_obs()` call in rollout worker to produce spatial tensor.
  3. **Fixes V6 (add_idle) and V7 (retarget_weakest)** to be actually distinct
     — v1 had them as silent duplicates of V0 primary.
  4. Higher entropy coefficient (0.03 vs 0.01) to discourage early collapse.

All else mirrors v1 PPO: 4 workers, 4 games/iter, categorical softmax over K=8.

Usage:
  python training/variant_picker_dual_stream_ppo.py \\
      --workers 4 --target-iters 2000 --games-per-iter 4 \\
      --four-player-prob 0.3 --lr 3e-4 --ent-coef 0.03 \\
      --out training/checkpoints/variant_picker_v2.pt
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import collections
import io
import math
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from featurize import (featurize_step, PLANET_DIM, FLEET_DIM, GLOBAL_DIM,
                       HISTORY_K, nearest_target_index, ship_bucket_idx)
from training.lb1200_agent import (
    aim_with_prediction, segment_hits_sun, fleet_target_planet, Fleet,
)
from training.dual_stream_model import (
    DualStreamAgent, rasterize_obs, N_SPATIAL_CHANNELS,
)


K_VARIANTS = 8
VARIANT_NAMES = ["primary", "75%", "50%", "125%", "drop_weakest",
                 "pass", "add_idle", "retarget_weakest"]
GRID = 32


# -----------------------------------------------------------------------------
# Variant generation — FULL implementation (V6, V7 no longer stubs)
# -----------------------------------------------------------------------------

MIN_SHIPS_TO_VARY = 3
MAX_SECONDARY_DISTANCE = 50.0
IDLE_MIN_SHIPS = 5


def _infer_target(src, angle, planets):
    pseudo = Fleet(-1, 0, src.x, src.y, angle, src.id, 1)
    result = fleet_target_planet(pseudo, planets)
    if result is None:
        return None
    target, _ = result
    return target


def generate_variants(primary: list, obs_planets: list, obs_fleets: list,
                      my_player: int, ang_vel: float = 0.0,
                      initial_by_id: dict | None = None) -> list[list]:
    """Produce K=8 variant action lists from primary.

    Args:
        primary: lb-1200's action list (list of [src_id, angle, ships] triples)
        obs_planets: raw planets from obs (list of [id, owner, x, y, r, ships, prod])
        obs_fleets: raw fleets from obs (for aim_with_prediction)
        my_player: our player index
        ang_vel, initial_by_id: for aim_with_prediction (optional)
    """
    variants: list[list] = [list(primary) for _ in range(K_VARIANTS)]
    if not primary:
        return [[] for _ in range(K_VARIANTS)]

    planet_ships = {int(p[0]): int(p[5]) for p in obs_planets}

    # Raw tuples → Planet-like namedtuple expected by fleet_target_planet
    # Use lb-1200's Fleet/Planet via a minimal adapter
    from training.lb1200_agent import Planet as _Planet
    planets_nt = [_Planet(int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                          float(p[4]), int(p[5]), int(p[6])) for p in obs_planets]
    planet_by_id = {p.id: p for p in planets_nt}

    # V0 primary (already set)
    # V1 75%
    variants[1] = [[m[0], m[1], max(1, int(m[2] * 0.75))] for m in primary]
    # V2 50%
    variants[2] = [[m[0], m[1], max(1, int(m[2] * 0.5))] for m in primary]
    # V3 125% (cap available)
    v3 = []
    for m in primary:
        pid = int(m[0])
        max_avail = max(1, planet_ships.get(pid, 1) - 1)
        v3.append([pid, m[1], min(max_avail, max(1, int(m[2] * 1.25)))])
    variants[3] = v3
    # V4 drop weakest
    if len(primary) >= 2:
        weakest = min(range(len(primary)), key=lambda i: primary[i][2])
        variants[4] = [m for i, m in enumerate(primary) if i != weakest]
    # V5 pass
    variants[5] = []

    # V6 add_idle — add launch from a currently-idle (unused by primary) planet
    used_sources = {int(m[0]) for m in primary}
    idle_planets = [p for p in planets_nt
                    if p.owner == my_player
                    and p.id not in used_sources
                    and int(p.ships) >= IDLE_MIN_SHIPS]
    v6 = list(primary)
    if idle_planets:
        # Pick idle with most ships
        src = max(idle_planets, key=lambda p: int(p.ships))
        # Nearest non-friendly target
        best_dist = 1e9; best_target = None
        for p in planets_nt:
            if p.owner == my_player or p.id == src.id:
                continue
            d = math.hypot(p.x - src.x, p.y - src.y)
            if d < best_dist and d <= MAX_SECONDARY_DISTANCE:
                if not segment_hits_sun(src.x, src.y, p.x, p.y):
                    best_dist = d
                    best_target = p
        if best_target is not None:
            try:
                send = max(MIN_SHIPS_TO_VARY, int(src.ships) // 2)
                aim = aim_with_prediction(
                    src, best_target, send,
                    initial_by_id or {}, ang_vel, [], set(),
                )
                if aim is not None:
                    angle = aim[0] if isinstance(aim, (tuple, list)) else aim
                    v6 = list(primary) + [[src.id, float(angle), send]]
            except Exception:
                pass
    variants[6] = v6

    # V7 retarget_weakest — swap weakest primary action's target to nearest non-friendly
    v7 = list(primary)
    if primary:
        weakest_i = min(range(len(primary)), key=lambda i: primary[i][2])
        src_id = int(primary[weakest_i][0])
        src = planet_by_id.get(src_id)
        if src is not None:
            # Current target
            curr_target = _infer_target(src, float(primary[weakest_i][1]), planets_nt)
            # Find DIFFERENT nearest non-friendly
            best_dist = 1e9; best_target = None
            for p in planets_nt:
                if p.owner == my_player or p.id == src_id:
                    continue
                if curr_target is not None and p.id == curr_target.id:
                    continue
                d = math.hypot(p.x - src.x, p.y - src.y)
                if d < best_dist and d <= MAX_SECONDARY_DISTANCE:
                    if not segment_hits_sun(src.x, src.y, p.x, p.y):
                        best_dist = d
                        best_target = p
            if best_target is not None:
                try:
                    aim = aim_with_prediction(
                        src, best_target, int(primary[weakest_i][2]),
                        initial_by_id or {}, ang_vel, [], set(),
                    )
                    if aim is not None:
                        angle = aim[0] if isinstance(aim, (tuple, list)) else aim
                        v7 = [list(m) for m in primary]
                        v7[weakest_i] = [src_id, float(angle), int(primary[weakest_i][2])]
                except Exception:
                    pass
    variants[7] = v7

    return variants


# -----------------------------------------------------------------------------
# Rollout worker
# -----------------------------------------------------------------------------

def _rollout_worker_init(state_dict_bytes: bytes):
    global _worker_net, _worker_lb1200
    from kaggle_environments import make   # noqa: F401
    _worker_net = DualStreamAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
        n_variants=K_VARIANTS,
    )
    buf = io.BytesIO(state_dict_bytes)
    sd = torch.load(buf, map_location="cpu", weights_only=False)
    _worker_net.load_state_dict(sd)
    _worker_net.eval()
    from training.lb1200_agent import agent as lb1200_agent
    _worker_lb1200 = lb1200_agent


def _rollout_game(task: dict) -> dict:
    import collections
    from kaggle_environments import make
    from training.lb1200_agent import agent as lb1200_agent

    n_players = task["n_players"]
    picker_seat = task["picker_seat"]
    ang_vel_init = None
    init_planets = None

    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)

    obs_history = collections.deque(maxlen=HISTORY_K)
    action_history = collections.deque(maxlen=HISTORY_K)
    last_actions_by_planet: dict = {}
    cum_stats = {"total_ships_sent": 0, "total_actions": 0}

    samples = []
    step = 0
    prev_my_ships = 0
    prev_my_planets = 1

    while not env.done and step < 500:
        actions_all = []
        for s in range(n_players):
            obs = env.state[s].observation
            if s == picker_seat:
                primary = lb1200_agent(obs, env.configuration) or []
                if primary:
                    obs_planets = obs.get("planets", []) if isinstance(obs, dict) \
                                  else getattr(obs, "planets", [])
                    obs_fleets = obs.get("fleets", []) if isinstance(obs, dict) \
                                 else getattr(obs, "fleets", [])
                    if ang_vel_init is None:
                        ang_vel_init = float(obs.get("angular_velocity", 0.0) if isinstance(obs, dict)
                                             else getattr(obs, "angular_velocity", 0.0))
                        init_planets = obs.get("initial_planets", []) if isinstance(obs, dict) \
                                       else getattr(obs, "initial_planets", [])
                    init_by_id_nt = None
                    if init_planets:
                        from training.lb1200_agent import Planet as _Planet
                        init_by_id_nt = {int(p[0]): _Planet(int(p[0]), int(p[1]), float(p[2]),
                                                             float(p[3]), float(p[4]), int(p[5]),
                                                             int(p[6]))
                                          for p in init_planets}

                    variants = generate_variants(
                        primary, obs_planets, obs_fleets, s,
                        ang_vel=ang_vel_init, initial_by_id=init_by_id_nt,
                    )

                    step_dict = {
                        "step": step,
                        "planets": obs_planets, "fleets": obs_fleets,
                        "action": primary,
                        "my_total_ships": sum(p[5] for p in obs_planets if p[1] == s),
                        "enemy_total_ships": 0, "my_planet_count": 0,
                        "enemy_planet_count": 0, "neutral_planet_count": 0,
                    }
                    feat = featurize_step(
                        step_dict, s, ang_vel_init, n_players, init_planets,
                        last_actions_by_planet=last_actions_by_planet,
                        cumulative_stats=cum_stats,
                        obs_history=list(obs_history),
                        action_history=list(action_history),
                    )
                    spatial = rasterize_obs(obs, s, grid=GRID)

                    pl = feat["planets"]; fl = feat["fleets"]
                    pmask = np.ones(pl.shape[0], dtype=bool) if pl.shape[0] > 0 \
                            else np.zeros(0, dtype=bool)
                    if pl.shape[0] == 0:
                        pl = np.zeros((1, PLANET_DIM), dtype=np.float32)
                        pmask = np.zeros(1, dtype=bool)
                    if fl.ndim < 2 or fl.shape[0] == 0:
                        fl = np.zeros((1, FLEET_DIM), dtype=np.float32)
                        fmask = np.zeros(1, dtype=bool)
                    else:
                        fmask = np.ones(fl.shape[0], dtype=bool)

                    with torch.no_grad():
                        logits, v = _worker_net(
                            torch.from_numpy(pl).unsqueeze(0),
                            torch.from_numpy(pmask).unsqueeze(0),
                            torch.from_numpy(fl).unsqueeze(0),
                            torch.from_numpy(fmask).unsqueeze(0),
                            torch.from_numpy(feat["globals"]).unsqueeze(0),
                            torch.from_numpy(spatial).unsqueeze(0),
                        )
                        probs = F.softmax(logits[0], dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        vi = int(dist.sample().item())
                        log_prob = float(dist.log_prob(torch.tensor(vi)).item())
                    chosen = variants[vi]
                    actions_all.append(chosen)
                    samples.append({
                        "feat": feat, "spatial": spatial,
                        "variant_idx": vi, "log_prob": log_prob,
                        "value": float(v.item()),
                        "reward": 0.0,
                    })
                else:
                    actions_all.append([])
            else:
                actions_all.append(lb1200_agent(obs, env.configuration) or [])
        env.step(actions_all)
        step += 1

        if samples:
            cur_planets = env.state[picker_seat].observation.get("planets", []) or []
            my_ships = sum(p[5] for p in cur_planets if p[1] == picker_seat)
            my_planets_ct = sum(1 for p in cur_planets if p[1] == picker_seat)
            r = 0.001 * (my_ships - prev_my_ships) + 0.02 * (my_planets_ct - prev_my_planets)
            samples[-1]["reward"] = r
            prev_my_ships, prev_my_planets = my_ships, my_planets_ct

    if samples:
        terminal = float(env.state[picker_seat].reward or 0)
        samples[-1]["reward"] += terminal

    # Compute Monte-Carlo returns-to-go (discounted). This is the proper
    # advantage target — per-step shaped rewards alone give near-zero gradient.
    GAMMA = 0.99
    G = 0.0
    for s_ in reversed(samples):
        G = s_["reward"] + GAMMA * G
        s_["mc_return"] = G

    return {"samples": samples, "n_players": n_players, "picker_seat": picker_seat,
            "win": float(env.state[picker_seat].reward or 0) > 0}


# -----------------------------------------------------------------------------
# Collate + PPO update
# -----------------------------------------------------------------------------

def _collate_batch(samples: list[dict], device: str) -> dict:
    B = len(samples)
    P = max(s["feat"]["planets"].shape[0] for s in samples) or 1
    F_ = max((s["feat"]["fleets"].shape[0] if s["feat"]["fleets"].ndim == 2 else 1)
             for s in samples) or 1
    planets = np.zeros((B, P, PLANET_DIM), dtype=np.float32)
    pmask = np.zeros((B, P), dtype=bool)
    fleets = np.zeros((B, F_, FLEET_DIM), dtype=np.float32)
    fmask = np.zeros((B, F_), dtype=bool)
    globals_ = np.zeros((B, GLOBAL_DIM), dtype=np.float32)
    spatial = np.zeros((B, N_SPATIAL_CHANNELS, GRID, GRID), dtype=np.float32)
    variant_idx = np.zeros((B,), dtype=np.int64)
    log_prob = np.zeros((B,), dtype=np.float32)
    returns = np.zeros((B,), dtype=np.float32)
    values = np.zeros((B,), dtype=np.float32)
    for i, s in enumerate(samples):
        f = s["feat"]
        np_ = f["planets"].shape[0]
        nf = f["fleets"].shape[0] if f["fleets"].ndim == 2 else 0
        if np_ > 0:
            planets[i, :np_] = f["planets"]; pmask[i, :np_] = True
        if nf > 0:
            fleets[i, :nf] = f["fleets"]; fmask[i, :nf] = True
        globals_[i] = f["globals"]
        spatial[i] = s["spatial"]
        variant_idx[i] = s["variant_idx"]
        log_prob[i] = s["log_prob"]
        # Use MC return-to-go (discounted) not per-step reward. This is the
        # proper PPO advantage target.
        returns[i] = s.get("mc_return", s["reward"])
        values[i] = s["value"]
    adv = returns - values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return {
        "planets": torch.from_numpy(planets).to(device),
        "pmask": torch.from_numpy(pmask).to(device),
        "fleets": torch.from_numpy(fleets).to(device),
        "fmask": torch.from_numpy(fmask).to(device),
        "globals": torch.from_numpy(globals_).to(device),
        "spatial": torch.from_numpy(spatial).to(device),
        "variant_idx": torch.from_numpy(variant_idx).to(device),
        "log_prob": torch.from_numpy(log_prob).to(device),
        "returns": torch.from_numpy(returns).to(device),
        "adv": torch.from_numpy(adv).to(device),
    }


def ppo_update(net: DualStreamAgent, opt: torch.optim.Optimizer,
               samples: list[dict], device: str,
               epochs: int = 4, clip: float = 0.2,
               ent_coef: float = 0.03, val_coef: float = 0.5) -> dict:
    if not samples:
        return {}
    batch = _collate_batch(samples, device)
    old_log_probs = batch["log_prob"]
    advantages = batch["adv"]
    returns = batch["returns"]

    info_sum = {"pi_loss": 0.0, "v_loss": 0.0, "ent": 0.0, "n": 0}
    for _ in range(epochs):
        logits, values = net(
            batch["planets"], batch["pmask"],
            batch["fleets"], batch["fmask"], batch["globals"],
            batch["spatial"],
        )
        dist = torch.distributions.Categorical(logits=logits)
        new_log_prob = dist.log_prob(batch["variant_idx"])
        ratio = (new_log_prob - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
        pi_loss = -torch.minimum(surr1, surr2).mean()
        v_loss = F.mse_loss(values, returns)
        ent = dist.entropy().mean()
        loss = pi_loss + val_coef * v_loss - ent_coef * ent
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        info_sum["pi_loss"] += pi_loss.item()
        info_sum["v_loss"] += v_loss.item()
        info_sum["ent"] += ent.item()
        info_sum["n"] += 1
    for k in ("pi_loss", "v_loss", "ent"):
        info_sum[k] /= max(1, info_sum["n"])
    return info_sum


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-iters", type=int, default=2000)
    ap.add_argument("--games-per-iter", type=int, default=4)
    ap.add_argument("--four-player-prob", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.03)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-every", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DualStreamAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
        n_variants=K_VARIANTS,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"[picker v2] device={device}  params={sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"ent_coef={args.ent_coef}", flush=True)

    def state_dict_bytes():
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
        return buf.getvalue()

    pool = mp.get_context("spawn").Pool(
        processes=args.workers,
        initializer=_rollout_worker_init,
        initargs=(state_dict_bytes(),),
    )

    t0 = time.time()
    for iter_ in range(1, args.target_iters + 1):
        if iter_ % 5 == 1:
            pool.close()
            pool = mp.get_context("spawn").Pool(
                processes=args.workers,
                initializer=_rollout_worker_init,
                initargs=(state_dict_bytes(),),
            )

        tasks = []
        for _ in range(args.games_per_iter):
            n_players = 4 if random.random() < args.four_player_prob else 2
            tasks.append({
                "n_players": n_players,
                "picker_seat": random.randint(0, n_players - 1),
            })
        rollouts = pool.map(_rollout_game, tasks)

        all_samples = []
        wins = 0
        variant_counts = [0] * K_VARIANTS
        for r in rollouts:
            all_samples.extend(r["samples"])
            if r["win"]: wins += 1
        for s in all_samples:
            variant_counts[s["variant_idx"]] += 1

        info = ppo_update(net, opt, all_samples, device, ent_coef=args.ent_coef) \
               if all_samples else {}

        elapsed = time.time() - t0
        vc_str = " ".join(f"{c}" for c in variant_counts)
        print(f"[iter {iter_:04d}] games={args.games_per_iter}  "
              f"samples={len(all_samples)}  wins={wins}/{args.games_per_iter}  "
              f"pi={info.get('pi_loss', 0):.3f}  v={info.get('v_loss', 0):.3f}  "
              f"ent={info.get('ent', 0):.3f}  "
              f"vc=[{vc_str}]  [{elapsed:.0f}s]", flush=True)

        if iter_ % args.snapshot_every == 0:
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                       args.out)
            print(f"[iter {iter_:04d}] saved {args.out}", flush=True)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
