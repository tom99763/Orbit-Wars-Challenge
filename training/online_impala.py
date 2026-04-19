"""Lux-AI-S1-flavoured self-play trainer, adapted for Orbit Wars.

Differences from our earlier PPO/A2C attempts:
  1. Reward shaping tailored to Orbit Wars (planet count, production lead,
     sun-kill penalty) — decays over time so terminal ±1 eventually dominates.
  2. Teacher KL to a frozen bc_v2 — prevents policy collapse into local
     optima while still allowing improvement.
  3. All seats contribute to value loss; learner action log-probs weighted
     by shaped-return advantages.
  4. Advantage clipping + entropy bonus for stability.
  5. Synchronous multiprocessing actors (4 workers) — simpler than IMPALA
     async; on-policy so V-trace is unnecessary.

Skipped (would improve more, larger eng cost):
  - UPGO (upper-bound policy gradient) — needs Q-values we don't have
  - V-trace (only matters for off-policy / async)
  - TTA (180° rotation augmentation)
  - AlphaStar-style league exploiters

Usage:
  python training/online_impala.py \
      --bc-ckpt training/checkpoints/bc_v2.pt \
      --out training/checkpoints/impala_v1.pt \
      --iters 125 --workers 4 --four-player-prob 0.5 \
      --shape-decay-iters 50 --teacher-kl 0.3
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import math
import multiprocessing as mp
import pathlib
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from training.model import OrbitAgent
from training.agent import _encode_obs, SHIPS_BUCKETS
from training.mcts import (
    sample_joint_action, _forward_policy, _shared_obs,
    disable_env_validation,
)
from featurize import featurize_step


# ---------------------------------------------------------------
# Orbit-Wars-specific reward shaping
# ---------------------------------------------------------------

def compute_shaped_rewards(env_steps, seat: int) -> list[float]:
    """Dense shaped reward per step for a given seat.

    Components:
      +0.10 · Δmy_planet_count          (territory gain)
      +0.02 · Δmy_production_rate       (economic lead)
      -0.10 · my_ships_vanishing_near_sun (sun-kill penalty)
      +0.05 · Δenemy_ships_destroyed_in_combat (offensive kills)
    """
    shaped = [0.0]
    prev = {}
    for t in range(1, len(env_steps)):
        prev_obs = env_steps[t-1][0].observation
        cur_obs = env_steps[t][0].observation
        pp = list(prev_obs.get("planets") or [])
        cp = list(cur_obs.get("planets") or [])
        pf = list(prev_obs.get("fleets") or [])
        cf = list(cur_obs.get("fleets") or [])

        p_n = sum(1 for p in pp if p[1] == seat)
        c_n = sum(1 for p in cp if p[1] == seat)
        p_prod = sum(p[6] for p in pp if p[1] == seat)
        c_prod = sum(p[6] for p in cp if p[1] == seat)
        # Enemy total ships (planets + fleets, non-me non-neutral)
        p_enemy = sum(p[5] for p in pp if p[1] != seat and p[1] != -1) + sum(
            f[6] for f in pf if f[1] != seat)
        c_enemy = sum(p[5] for p in cp if p[1] != seat and p[1] != -1) + sum(
            f[6] for f in cf if f[1] != seat)

        # Sun-kill: my fleet present at t-1, gone at t, and was within 15 of sun
        pf_ids = {f[0]: f for f in pf if f[1] == seat}
        cf_ids = {f[0] for f in cf if f[1] == seat}
        sun_lost = 0
        for fid, f in pf_ids.items():
            if fid not in cf_ids:
                d = math.hypot(f[2] - 50.0, f[3] - 50.0)
                if d < 15.0:
                    sun_lost += f[6]

        dr = (
            0.10 * (c_n - p_n)
            + 0.02 * (c_prod - p_prod)
            - 0.10 * sun_lost / 100.0
            + 0.05 * max(0, p_enemy - c_enemy) / 100.0
        )
        shaped.append(max(-1.0, min(1.0, dr)))
    return shaped


# ---------------------------------------------------------------
# Worker
# ---------------------------------------------------------------

_worker_state: dict = {}


def _worker_init(model_kwargs: dict, teacher_state):
    torch.set_num_threads(1)
    disable_env_validation()
    _worker_state["model"] = OrbitAgent(**model_kwargs)
    _worker_state["model"].eval()
    _worker_state["teacher"] = OrbitAgent(**model_kwargs)
    _worker_state["teacher"].load_state_dict(teacher_state)
    _worker_state["teacher"].eval()
    for p in _worker_state["teacher"].parameters():
        p.requires_grad_(False)


def _worker_play(task: dict) -> dict:
    from kaggle_environments import make
    buf = io.BytesIO(task["weights"])
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    _worker_state["model"].load_state_dict(sd)
    model = _worker_state["model"]
    n_players = task["n_players"]
    opp_temp = task["opp_temp"]

    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)
    obs0 = env.state[0].observation
    ang_vel = float(obs0.get("angular_velocity") or 0.0)
    init_planets = list(obs0.get("initial_planets") or [])
    init_ids = {int(p[0]) for p in init_planets}
    comet_ids: set = set()

    # Play full game; each seat samples from current policy
    while not env.done:
        actions: list = [None] * n_players
        for seat in range(n_players):
            obs = _shared_obs(env, seat)
            tl, bl, _, planets_raw, pids, omask = _forward_policy(model, obs)
            _sub, m, _ = sample_joint_action(
                tl, bl, planets_raw, omask, temperature=opp_temp
            )
            actions[seat] = m
        env.step(actions)

    # Featurize + per-step records for every seat
    records: dict[int, list] = {s: [] for s in range(n_players)}
    for t, step in enumerate(env.steps):
        shared_obs = step[0].observation
        planets = list(shared_obs.get("planets") or [])
        fleets = list(shared_obs.get("fleets") or [])
        for p in planets:
            if int(p[0]) not in init_ids:
                comet_ids.add(int(p[0]))
        for seat in range(n_players):
            action = step[seat].action or []
            if not action:
                continue
            step_dict = {
                "step": t, "planets": planets, "fleets": fleets, "action": action,
                "my_total_ships": sum(p[5] for p in planets if p[1] == seat) + sum(
                    f[6] for f in fleets if f[1] == seat),
                "enemy_total_ships": sum(p[5] for p in planets
                                         if p[1] != seat and p[1] != -1) + sum(
                    f[6] for f in fleets if f[1] != seat),
                "my_planet_count": sum(1 for p in planets if p[1] == seat),
                "enemy_planet_count": sum(1 for p in planets
                                          if p[1] != seat and p[1] != -1),
                "neutral_planet_count": sum(1 for p in planets if p[1] == -1),
            }
            feat = featurize_step(step_dict, seat, ang_vel, n_players,
                                  init_planets, comet_ids)
            feat["_seat"] = seat
            feat["_t"] = t
            records[seat].append(feat)

    rewards = [s.reward if s.reward is not None else 0 for s in env.state]
    # Per-seat shaped reward trajectories + terminal
    shaped_per_seat = {s: compute_shaped_rewards(env.steps, s)
                       for s in range(n_players)}
    # Attach cumulative returns (Monte Carlo) with decay γ=0.997
    gamma = 0.997
    for seat in range(n_players):
        shaped = shaped_per_seat[seat]
        T = len(shaped)
        # Shape-only return
        G = [0.0] * (T + 1)
        for t in reversed(range(T)):
            G[t] = shaped[t] + gamma * G[t+1]
        # Add terminal ±1 at the end of shape-return (so Monte Carlo return sees it)
        # We attach terminal_r to each record so master can combine
        for feat in records[seat]:
            t = feat["_t"]
            if t >= T:
                t = T - 1
            feat["_shape_return"] = G[t]
            feat["_terminal_reward"] = float(rewards[seat])

    return {
        "n_players": n_players,
        "steps": len(env.steps),
        "rewards": rewards,
        "records": records,
    }


# ---------------------------------------------------------------
# Master: policy + value + teacher-KL loss
# ---------------------------------------------------------------

def pad_stack(samples, max_planets=64, max_fleets=64):
    if not samples:
        return None
    P = min(max_planets, max(s["planets"].shape[0] for s in samples))
    FN = min(max_fleets, max(max(1, s["fleets"].shape[0]) for s in samples))
    B = len(samples)
    planets = np.zeros((B, P, 14), dtype=np.float32)
    planet_mask = np.zeros((B, P), dtype=bool)
    fleets = np.zeros((B, FN, 9), dtype=np.float32)
    fleet_mask = np.zeros((B, FN), dtype=bool)
    globals_ = np.zeros((B, 16), dtype=np.float32)
    for i, s in enumerate(samples):
        N = min(s["planets"].shape[0], P)
        planets[i, :N] = s["planets"][:N]
        planet_mask[i, :N] = True
        nf = min(s["fleets"].shape[0], FN)
        if nf > 0:
            fleets[i, :nf] = s["fleets"][:nf]
            fleet_mask[i, :nf] = True
        globals_[i] = s["globals"]
    return {
        "planets": torch.from_numpy(planets),
        "planet_mask": torch.from_numpy(planet_mask),
        "fleets": torch.from_numpy(fleets),
        "fleet_mask": torch.from_numpy(fleet_mask),
        "globals": torch.from_numpy(globals_),
        "P": P,
    }


def impala_train_step(model, teacher, optimizer, samples,
                      shape_weight: float, teacher_kl_coef: float,
                      vf_coef: float = 0.5, ent_coef: float = 0.01):
    """Policy + value + teacher-KL update, Orbit-Wars shaping weighted."""
    if not samples:
        return None
    tensors = pad_stack(samples)
    if tensors is None:
        return None
    P = tensors["P"]

    tgt_logits, bkt_logits, value = model(
        tensors["planets"], tensors["planet_mask"],
        tensors["fleets"], tensors["fleet_mask"],
        tensors["globals"], None
    )
    with torch.no_grad():
        t_tgt, t_bkt, _ = teacher(
            tensors["planets"], tensors["planet_mask"],
            tensors["fleets"], tensors["fleet_mask"],
            tensors["globals"], None
        )

    # Returns: combined shaped + terminal
    shape_r = torch.tensor([s["_shape_return"] for s in samples],
                           dtype=torch.float32)
    term_r = torch.tensor([s["_terminal_reward"] for s in samples],
                          dtype=torch.float32)
    returns = shape_weight * shape_r + term_r
    advantage = (returns - value.detach())
    # Normalize + clip
    advantage = (advantage - advantage.mean()) / (advantage.std().clamp(min=1e-6))
    advantage = torch.clamp(advantage, -2.0, 2.0)

    # Per-label log-prob of the actual sub-actions taken
    flat_batch, flat_src, flat_tgt, flat_bkt = [], [], [], []
    for i, s in enumerate(samples):
        srcs = s["src_planet_idx"]; tgts = s["target_planet_idx"]; bkts = s["ships_bucket"]
        valid = (srcs < P) & (tgts < P)
        for j in range(len(srcs)):
            if valid[j]:
                flat_batch.append(i)
                flat_src.append(int(srcs[j]))
                flat_tgt.append(int(tgts[j]))
                flat_bkt.append(int(bkts[j]))
    if not flat_batch:
        return None
    fb = torch.tensor(flat_batch, dtype=torch.long)
    fs = torch.tensor(flat_src, dtype=torch.long)
    ft = torch.tensor(flat_tgt, dtype=torch.long)
    fk = torch.tensor(flat_bkt, dtype=torch.long)

    picked_tgt = tgt_logits[fb, fs]
    picked_bkt = bkt_logits[fb, fs]
    tgt_labels = ft + 1
    logp_tgt = F.log_softmax(picked_tgt, dim=-1).gather(1, tgt_labels.unsqueeze(1)).squeeze(1)
    logp_bkt = F.log_softmax(picked_bkt, dim=-1).gather(1, fk.unsqueeze(1)).squeeze(1)
    logp = logp_tgt + 0.5 * logp_bkt
    adv_per_label = advantage[fb]
    policy_loss = -(adv_per_label * logp).mean()

    value_loss = 0.5 * (value - returns).pow(2).mean()

    # Teacher KL: KL(π_nn || π_teacher) at every acting step (src planet)
    t_rows = tgt_logits[fb, fs]
    t_rows_teacher = t_tgt[fb, fs]
    b_rows = bkt_logits[fb, fs]
    b_rows_teacher = t_bkt[fb, fs]
    kl_tgt = F.kl_div(F.log_softmax(t_rows, dim=-1),
                      F.softmax(t_rows_teacher, dim=-1),
                      reduction="batchmean")
    kl_bkt = F.kl_div(F.log_softmax(b_rows, dim=-1),
                      F.softmax(b_rows_teacher, dim=-1),
                      reduction="batchmean")
    teacher_kl = 0.5 * (kl_tgt + kl_bkt)

    # Entropy over tgt distribution (regularisation)
    p_tgt = F.softmax(tgt_logits, dim=-1)
    entropy = -(p_tgt * F.log_softmax(tgt_logits, dim=-1)).sum(-1).mean()

    loss = (policy_loss + vf_coef * value_loss
            + teacher_kl_coef * teacher_kl - ent_coef * entropy)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "teacher_kl": teacher_kl.item(),
        "entropy": entropy.item(),
        "advantage_mean": advantage.mean().item(),
        "advantage_std": advantage.std().item(),
        "n_labels": len(flat_batch),
        "n_samples": len(samples),
    }


def serialize_state(model) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-ckpt", required=True,
                    help="Starting + teacher checkpoint (frozen for KL)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=125)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--four-player-prob", type=float, default=0.5)
    ap.add_argument("--opp-temp", type=float, default=1.0,
                    help="Policy temperature during game play")
    ap.add_argument("--shape-decay-iters", type=int, default=50,
                    help="Linear decay of shaped-reward weight 1→0 over N iters")
    ap.add_argument("--teacher-kl", type=float, default=0.3,
                    help="Coefficient on KL(π_nn || π_teacher)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-steps-per-iter", type=int, default=2)
    ap.add_argument("--monitor", default=".impala_monitor.csv")
    args = ap.parse_args()

    ckpt = torch.load(args.bc_ckpt, map_location="cpu", weights_only=False)
    model = OrbitAgent(**ckpt["kwargs"])
    model.load_state_dict(ckpt["model"])
    teacher_state = {k: v.clone() for k, v in ckpt["model"].items()}
    teacher = OrbitAgent(**ckpt["kwargs"])
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mon = pathlib.Path(args.monitor)
    fields = ["ts", "iter", "games", "total_steps", "seat_win_counts",
              "shape_weight", "n_samples", "n_labels",
              "loss", "policy_loss", "value_loss", "teacher_kl",
              "entropy", "wall_seconds"]
    if not mon.exists():
        with mon.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(args.workers, initializer=_worker_init,
                   initargs=(ckpt["kwargs"], teacher_state))

    print(f"Online IMPALA (Lux-S1-flavoured) iters={args.iters} "
          f"workers={args.workers} teacher_kl={args.teacher_kl} "
          f"shape_decay={args.shape_decay_iters}", flush=True)

    try:
        for it in range(args.iters):
            t0 = time.time()
            # Shape weight: 1.0 → 0.0 linearly over first N iters
            shape_weight = max(0.0, 1.0 - it / max(1, args.shape_decay_iters))
            state_bytes = serialize_state(model)
            tasks = []
            for _ in range(args.workers):
                n_players = 4 if random.random() < args.four_player_prob else 2
                tasks.append({"weights": state_bytes, "n_players": n_players,
                              "opp_temp": args.opp_temp})
            results = pool.map(_worker_play, tasks)

            # Gather all-seat records
            all_samples = []
            total_steps = 0
            seat_win_counts: dict[int, int] = defaultdict(int)
            for r in results:
                total_steps += r["steps"]
                mx = max(r["rewards"])
                for s, rr in enumerate(r["rewards"]):
                    if rr == mx:
                        seat_win_counts[s] += 1
                for seat, recs in r["records"].items():
                    all_samples.extend(recs)

            stats = None
            if all_samples:
                for _ in range(args.grad_steps_per_iter):
                    stats = impala_train_step(
                        model, teacher, optimizer, all_samples,
                        shape_weight=shape_weight,
                        teacher_kl_coef=args.teacher_kl,
                    )

            dt = time.time() - t0
            st = stats or {}
            row = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "iter": it, "games": args.workers,
                "total_steps": total_steps,
                "seat_win_counts": ";".join(f"{k}:{v}" for k, v in
                                            sorted(seat_win_counts.items())),
                "shape_weight": round(shape_weight, 3),
                "n_samples": st.get("n_samples", 0),
                "n_labels": st.get("n_labels", 0),
                "loss": round(st.get("loss", 0), 4),
                "policy_loss": round(st.get("policy_loss", 0), 4),
                "value_loss": round(st.get("value_loss", 0), 4),
                "teacher_kl": round(st.get("teacher_kl", 0), 4),
                "entropy": round(st.get("entropy", 0), 4),
                "wall_seconds": round(dt, 1),
            }
            with mon.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            print(f"[iter {it:03d}/{args.iters}] {dt:.1f}s  "
                  f"games={args.workers}  wins={dict(seat_win_counts)}  "
                  f"shape_w={shape_weight:.2f}  "
                  f"loss={row['loss']}  pol={row['policy_loss']}  "
                  f"val={row['value_loss']}  kl_T={row['teacher_kl']}  "
                  f"ent={row['entropy']}", flush=True)

            if (it + 1) % 10 == 0 or it == args.iters - 1:
                torch.save({
                    "model": model.state_dict(),
                    "kwargs": ckpt["kwargs"],
                    "iter": it + 1,
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                }, out_path)
                print(f"  → saved {out_path}", flush=True)
    finally:
        pool.close()
        pool.join()

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
