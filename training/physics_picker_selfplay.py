"""Physics Picker Self-Play PPO — next stage after physics_picker_ppo.py.

Activates when V3 reaches dominance vs lb-1200 (e.g., 3 consecutive 4/4 iters).

Key changes vs physics_picker_ppo.py:
  1. **Opponent is NOT fixed lb-1200 anymore** — drawn from mixture:
       - self: current policy net (our own weights)                 60%
       - past: random past snapshot from opponent pool              20%
       - lb1200: rule-based agent (anchor to prevent policy drift)   15%
       - random: uniform K=6 sampler (anti-overfit, diversity)        5%
  2. **Opponent pool maintained** via training/opponent_pool.py
     - Snapshot every `--pool-every-iter` iters (default 20)
     - Max pool size 10 (FIFO)
  3. **Same loss + logging as V3** (pi, v, ent, cc)
  4. **Versioned ckpt snapshots** (per memory: todo_versioned_checkpoints.md)
  5. **Warm-start required** — must load from physics_picker_v3.pt

Usage (post V3 domination):
  python training/physics_picker_selfplay.py \\
      --warm-start training/checkpoints/physics_picker_v3.pt \\
      --pool-dir training/checkpoints/physics_pool \\
      --workers 4 --target-iters 1000 --games-per-iter 4 \\
      --pool-every-iter 20 \\
      --lr 1e-4 --ent-coef 0.04 \\
      --out training/checkpoints/physics_picker_sp.pt

Rationale for hyperparam changes vs V3:
  - lr 3e-4 → 1e-4: self-play dynamics are non-stationary, gentler updates
  - ent_coef 0.03 → 0.04: self-play needs more exploration (opp adapts)
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
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from featurize import (featurize_step, PLANET_DIM, FLEET_DIM, GLOBAL_DIM, HISTORY_K)
from training.dual_stream_model import (SpatialCNN, ScalarMLP, rasterize_obs,
                                         N_SPATIAL_CHANNELS)
from training.lb1200_agent import build_world
from training.physics_action_helper import (
    K_PER_SOURCE, CANDIDATE_NAMES, generate_per_source_candidates,
    materialize_joint_action,
)
from training.physics_picker_ppo import (
    DualStreamCandidateAgent,
    ppo_update as ppo_update_base,   # reuse same PPO math
    _collate_batch as _collate_batch_base,
    GRID, GAMMA,
)
from training.opponent_pool import OpponentPool


DEFAULT_MIX = {"self": 0.60, "past": 0.20, "lb1200": 0.15, "random": 0.05}


def sample_opponent_kind(rng: random.Random, mix: dict) -> str:
    total = sum(mix.values())
    x = rng.random() * total
    acc = 0.0
    for k, v in mix.items():
        acc += v
        if x <= acc:
            return k
    return next(reversed(mix))


# -----------------------------------------------------------------------------
# Rollout worker — multi-opponent self-play
# -----------------------------------------------------------------------------

# Worker state: current net + dict of past_net (opponent pool)
_worker_nets: dict = {}    # key: "self" | "past_{idx}" → net
_worker_rng = None


def _worker_init(current_state_dict_bytes: bytes, past_sds: list[bytes]):
    global _worker_nets, _worker_rng
    from kaggle_environments import make   # noqa
    _worker_rng = random.Random()

    # Load current (self) net
    cur = DualStreamCandidateAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    )
    buf = io.BytesIO(current_state_dict_bytes)
    sd = torch.load(buf, map_location="cpu", weights_only=False)
    cur.load_state_dict(sd)
    cur.eval()
    _worker_nets["self"] = cur

    # Load past snapshots (limited to avoid bloat; take first 3 latest)
    _worker_nets["past"] = []
    for sd_bytes in past_sds[-3:]:
        p_net = DualStreamCandidateAgent(
            planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
        )
        pbuf = io.BytesIO(sd_bytes)
        psd = torch.load(pbuf, map_location="cpu", weights_only=False)
        p_net.load_state_dict(psd)
        p_net.eval()
        _worker_nets["past"].append(p_net)


def _net_pick_action(net, obs, env_config, my_player: int, world, feat, spatial,
                      planet_ids) -> tuple[list, list, list, float]:
    """Run net forward and sample per-planet candidate. Returns
    (action_list, picks, log_probs, entropy)."""
    pl = feat["planets"]; fl = feat["fleets"]
    if pl.shape[0] == 0:
        pl = np.zeros((1, PLANET_DIM), dtype=np.float32)
        pmask = np.zeros(1, dtype=bool)
    else:
        pmask = np.ones(pl.shape[0], dtype=bool)
    if fl.ndim < 2 or fl.shape[0] == 0:
        fl = np.zeros((1, FLEET_DIM), dtype=np.float32)
        fmask = np.zeros(1, dtype=bool)
    else:
        fmask = np.ones(fl.shape[0], dtype=bool)

    with torch.no_grad():
        logits, v = net(
            torch.from_numpy(pl).unsqueeze(0),
            torch.from_numpy(pmask).unsqueeze(0),
            torch.from_numpy(fl).unsqueeze(0),
            torch.from_numpy(fmask).unsqueeze(0),
            torch.from_numpy(feat["globals"]).unsqueeze(0),
            torch.from_numpy(spatial).unsqueeze(0),
        )
    logits_np = logits[0].cpu().numpy()
    picks, log_probs, entropies = [], [], []
    for i, pid in enumerate(planet_ids):
        src = next((pl_ for pl_ in world.planets if pl_.id == int(pid)), None)
        if src is None or src.owner != my_player:
            continue
        pl_logits = logits_np[i]
        probs = np.exp(pl_logits - pl_logits.max())
        probs = probs / probs.sum()
        ci = int(np.random.choice(K_PER_SOURCE, p=probs))
        picks.append((int(pid), ci))
        log_probs.append(float(np.log(probs[ci] + 1e-9)))
        entropies.append(-float((probs * np.log(probs + 1e-9)).sum()))
    action_list = materialize_joint_action(picks, world, my_player)
    return action_list, picks, log_probs, (sum(entropies) / max(1, len(entropies))), float(v.item())


def _random_opp_action(world, my_player: int) -> list:
    """Uniform random over K=6 candidates per owned planet."""
    picks = []
    for src in world.planets:
        if src.owner != my_player:
            continue
        ci = random.randint(0, K_PER_SOURCE - 1)
        picks.append((src.id, ci))
    return materialize_joint_action(picks, world, my_player)


def _rollout_game(task: dict) -> dict:
    from kaggle_environments import make
    from training.lb1200_agent import agent as lb1200_agent

    n_players = task["n_players"]
    picker_seat = task["picker_seat"]
    mix = task["mix"]
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)

    obs_history = collections.deque(maxlen=HISTORY_K)
    action_history = collections.deque(maxlen=HISTORY_K)
    last_actions_by_planet: dict = {}
    cum_stats = {"total_ships_sent": 0, "total_actions": 0}

    # Decide opponent types per seat (fixed for this game)
    seat_opps: dict[int, str] = {}
    for s in range(n_players):
        if s == picker_seat:
            continue
        seat_opps[s] = sample_opponent_kind(_worker_rng, mix)

    samples = []
    ang_vel_init = None; init_planets = None
    step = 0
    prev_my_ships = 0; prev_my_planets = 1
    cand_choice_counts = [0] * K_PER_SOURCE

    while not env.done and step < 500:
        actions_all = []
        for s in range(n_players):
            obs = env.state[s].observation
            # Build world (may fail for some seats)
            try:
                world = build_world(obs)
            except Exception:
                actions_all.append([])
                continue

            raw_planets = obs.get("planets", []) or []
            my_planets_list = [p for p in world.planets if p.owner == s]

            if not my_planets_list:
                actions_all.append([])
                continue

            if ang_vel_init is None:
                ang_vel_init = float(obs.get("angular_velocity", 0.0) or 0.0)
                init_planets = obs.get("initial_planets", []) or []

            # Featurize (only learner seat records training data)
            if s == picker_seat:
                step_dict = {
                    "step": step, "planets": raw_planets,
                    "fleets": obs.get("fleets", []) or [],
                    "action": [],
                    "my_total_ships": sum(p[5] for p in raw_planets if p[1] == s),
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
                action_list, picks, log_probs, ent, v = _net_pick_action(
                    _worker_nets["self"], obs, env.configuration, s, world,
                    feat, spatial, feat.get("planet_ids", [])
                )
                for pid_ci in picks:
                    cand_choice_counts[pid_ci[1]] += 1
                actions_all.append(action_list)
                if picks:
                    samples.append({
                        "feat": feat, "spatial": spatial,
                        "planet_ids": feat["planet_ids"],
                        "picks": picks,
                        "log_prob_sum": sum(log_probs),
                        "entropy": ent,
                        "value": v,
                        "reward": 0.0,
                        "my_player": s,
                    })
            else:
                # Opponent seat: dispatch by type
                opp_kind = seat_opps.get(s, "self")
                if opp_kind == "lb1200":
                    actions_all.append(lb1200_agent(obs, env.configuration) or [])
                elif opp_kind == "random":
                    actions_all.append(_random_opp_action(world, s))
                elif opp_kind == "past" and _worker_nets.get("past"):
                    past_net = random.choice(_worker_nets["past"])
                    # Minimal featurize for forward (no history tracking)
                    step_dict = {
                        "step": step, "planets": raw_planets,
                        "fleets": obs.get("fleets", []) or [],
                        "action": [],
                        "my_total_ships": 0, "enemy_total_ships": 0,
                        "my_planet_count": 0, "enemy_planet_count": 0,
                        "neutral_planet_count": 0,
                    }
                    feat = featurize_step(step_dict, s, ang_vel_init, n_players, init_planets,
                                           last_actions_by_planet={},
                                           cumulative_stats={"total_ships_sent": 0, "total_actions": 0},
                                           obs_history=[], action_history=[])
                    spatial = rasterize_obs(obs, s, grid=GRID)
                    act_list, _p, _lp, _e, _v = _net_pick_action(
                        past_net, obs, env.configuration, s, world,
                        feat, spatial, feat.get("planet_ids", [])
                    )
                    actions_all.append(act_list)
                else:
                    # "self" — current net plays opponent seat
                    step_dict = {
                        "step": step, "planets": raw_planets,
                        "fleets": obs.get("fleets", []) or [],
                        "action": [],
                        "my_total_ships": 0, "enemy_total_ships": 0,
                        "my_planet_count": 0, "enemy_planet_count": 0,
                        "neutral_planet_count": 0,
                    }
                    feat = featurize_step(step_dict, s, ang_vel_init, n_players, init_planets,
                                           last_actions_by_planet={},
                                           cumulative_stats={"total_ships_sent": 0, "total_actions": 0},
                                           obs_history=[], action_history=[])
                    spatial = rasterize_obs(obs, s, grid=GRID)
                    act_list, _p, _lp, _e, _v = _net_pick_action(
                        _worker_nets["self"], obs, env.configuration, s, world,
                        feat, spatial, feat.get("planet_ids", [])
                    )
                    actions_all.append(act_list)

        # Update learner's history
        if samples:
            for mv in actions_all[picker_seat]:
                if len(mv) != 3:
                    continue
                from featurize import nearest_target_index, ship_bucket_idx
                raw_planets = env.state[picker_seat].observation.get("planets", []) or []
                src_id, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
                src_planet = next((p for p in raw_planets if int(p[0]) == src_id), None)
                if src_planet is None:
                    continue
                tgt_i = nearest_target_index(src_planet, ang, raw_planets)
                tgt_pid = int(raw_planets[tgt_i][0]) if tgt_i is not None else -1
                garrison = int(src_planet[5]) + ships
                bkt_idx = ship_bucket_idx(ships, max(1, garrison))
                prev = last_actions_by_planet.get(src_id, (-1, 0, -1, 0))
                last_actions_by_planet[src_id] = (tgt_pid, bkt_idx, step, prev[3] + 1)
                cum_stats["total_ships_sent"] += ships
                cum_stats["total_actions"] += 1
                action_history.append((src_id, tgt_pid, bkt_idx, step))
            obs_history.append({"planets": env.state[picker_seat].observation.get("planets", []) or [],
                                "step": step})

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

    G = 0.0
    for s_ in reversed(samples):
        G = s_["reward"] + GAMMA * G
        s_["mc_return"] = G

    return {"samples": samples, "picker_seat": picker_seat,
            "win": float(env.state[picker_seat].reward or 0) > 0,
            "cand_choice_counts": cand_choice_counts,
            "seat_opps": seat_opps}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", required=True,
                    help="path to physics_picker_v3.pt ckpt (required)")
    ap.add_argument("--pool-dir", default="training/checkpoints/physics_pool")
    ap.add_argument("--pool-max-size", type=int, default=10)
    ap.add_argument("--pool-every-iter", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-iters", type=int, default=1000)
    ap.add_argument("--games-per-iter", type=int, default=4)
    ap.add_argument("--four-player-prob", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent-coef", type=float, default=0.04)
    ap.add_argument("--mix-self", type=float, default=0.60)
    ap.add_argument("--mix-past", type=float, default=0.20)
    ap.add_argument("--mix-lb1200", type=float, default=0.15)
    ap.add_argument("--mix-random", type=float, default=0.05)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DualStreamCandidateAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    ).to(device)

    # Warm-start required
    assert Path(args.warm_start).exists(), f"warm-start {args.warm_start} missing"
    ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(sd)
    print(f"[selfplay] warm-started from {args.warm_start}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"[selfplay] device={device}  params={sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"mix={{self:{args.mix_self},past:{args.mix_past},lb1200:{args.mix_lb1200},random:{args.mix_random}}}",
          flush=True)

    pool = OpponentPool(args.pool_dir, max_size=args.pool_max_size)
    # Seed pool with warm-start ckpt
    if pool.size() == 0:
        pool.add(net.state_dict())
        print(f"[selfplay] seeded opponent pool with warm-start", flush=True)

    mix = {"self": args.mix_self, "past": args.mix_past,
           "lb1200": args.mix_lb1200, "random": args.mix_random}

    def cur_sd_bytes():
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
        return buf.getvalue()

    def past_sds_bytes():
        paths = pool._snapshots()
        results = []
        for p in paths[-3:]:    # last 3
            results.append(p.read_bytes())
        return results

    mp_ctx = mp.get_context("spawn")
    pool_p = mp_ctx.Pool(processes=args.workers,
                          initializer=_worker_init,
                          initargs=(cur_sd_bytes(), past_sds_bytes()))

    t0 = time.time()
    for iter_ in range(1, args.target_iters + 1):
        # Rebuild pool every 5 iters to sync latest weights + new opp pool entries
        if iter_ % 5 == 1:
            pool_p.close()
            pool_p = mp_ctx.Pool(processes=args.workers,
                                  initializer=_worker_init,
                                  initargs=(cur_sd_bytes(), past_sds_bytes()))

        tasks = []
        for _ in range(args.games_per_iter):
            n_players = 4 if random.random() < args.four_player_prob else 2
            tasks.append({
                "n_players": n_players,
                "picker_seat": random.randint(0, n_players - 1),
                "mix": mix,
            })
        rollouts = pool_p.map(_rollout_game, tasks)

        all_samples = []
        wins = 0
        cc_total = [0] * K_PER_SOURCE
        opp_summary: dict[str, int] = {}
        for r in rollouts:
            all_samples.extend(r["samples"])
            if r["win"]: wins += 1
            for i, c in enumerate(r["cand_choice_counts"]):
                cc_total[i] += c
            for kind in r["seat_opps"].values():
                opp_summary[kind] = opp_summary.get(kind, 0) + 1

        info = ppo_update_base(net, opt, all_samples, device, ent_coef=args.ent_coef)

        elapsed = time.time() - t0
        cc_str = " ".join(f"{c}" for c in cc_total)
        opp_str = " ".join(f"{k}:{v}" for k, v in sorted(opp_summary.items()))
        print(f"[iter {iter_:04d}] games={args.games_per_iter}  "
              f"samples={len(all_samples)}  wins={wins}/{args.games_per_iter}  "
              f"pi={info.get('pi_loss', 0):.3f}  "
              f"v={info.get('v_loss', 0):.3f}  "
              f"ent={info.get('entropy', 0):.3f}  "
              f"cc=[{cc_str}]  opps=[{opp_str}]  [{elapsed:.0f}s]",
              flush=True)

        # Pool update
        if iter_ % args.pool_every_iter == 0:
            pool.add(net.state_dict())
            print(f"[iter {iter_:04d}] added snapshot to opponent pool (size={pool.size()})",
                  flush=True)

        if iter_ % args.snapshot_every == 0:
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, args.out)
            stem = Path(args.out).stem
            versioned = Path(args.out).parent / f"{stem}_iter{iter_:04d}.pt"
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, versioned)
            print(f"[iter {iter_:04d}] saved {args.out} + {versioned.name}",
                  flush=True)

    pool_p.close()
    pool_p.join()


if __name__ == "__main__":
    main()
