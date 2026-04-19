"""impala_v2: Self-play with 50/50 lb-928 opponent mix.

Same framework as online_impala.py, but each non-learner seat has a
`--lb928-prob` chance of being replaced by the rules-based lb-928
planner. Training records only include self-seats (lb-928 seats are
present as opposition only).

Guarantee: at least one seat is self-played per game (forced fallback
if all die rolls favour lb-928).

Launch:
  python training/online_impala_v2.py \
      --bc-ckpt training/checkpoints/bc_v2.pt \
      --out training/checkpoints/impala_v2.pt \
      --iters 125 --workers 4 --four-player-prob 0.5 \
      --lb928-prob 0.5 --shape-decay-iters 50 --lr 1e-4
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
from training.lb928_agent import agent as lb928_agent_fn
from training.online_impala import (
    compute_shaped_rewards, pad_stack, impala_train_step,
    serialize_state,
)
from featurize import featurize_step


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


def _pick_opponent_config(n_players: int, lb928_prob: float) -> list[bool]:
    """Return per-seat 'is_self' flags. Force at least 1 self seat."""
    seat_is_self = [random.random() >= lb928_prob for _ in range(n_players)]
    if not any(seat_is_self):
        seat_is_self[random.randint(0, n_players - 1)] = True
    return seat_is_self


def _worker_play(task: dict) -> dict:
    from kaggle_environments import make

    buf = io.BytesIO(task["weights"])
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    _worker_state["model"].load_state_dict(sd)
    model = _worker_state["model"]

    n_players = task["n_players"]
    opp_temp = task["opp_temp"]
    lb928_prob = task["lb928_prob"]

    seat_is_self = _pick_opponent_config(n_players, lb928_prob)

    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)
    obs0 = env.state[0].observation
    ang_vel = float(obs0.get("angular_velocity") or 0.0)
    init_planets = list(obs0.get("initial_planets") or [])
    init_ids = {int(p[0]) for p in init_planets}
    comet_ids: set = set()

    # Play game
    while not env.done:
        actions: list = [None] * n_players
        for seat in range(n_players):
            obs = _shared_obs(env, seat)
            if seat_is_self[seat]:
                tl, bl, _, planets_raw, pids, omask = _forward_policy(model, obs)
                _sub, m, _ = sample_joint_action(
                    tl, bl, planets_raw, omask, temperature=opp_temp
                )
                actions[seat] = m
            else:
                try:
                    actions[seat] = lb928_agent_fn(obs) or []
                except Exception:
                    actions[seat] = []
        env.step(actions)

    # Featurize + per-step records for SELF seats only
    records: dict[int, list] = {s: [] for s in range(n_players) if seat_is_self[s]}
    for t, step in enumerate(env.steps):
        shared_obs = step[0].observation
        planets = list(shared_obs.get("planets") or [])
        fleets = list(shared_obs.get("fleets") or [])
        for p in planets:
            if int(p[0]) not in init_ids:
                comet_ids.add(int(p[0]))
        for seat in range(n_players):
            if not seat_is_self[seat]:
                continue
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
    gamma = 0.997
    for seat in records.keys():
        shaped = compute_shaped_rewards(env.steps, seat)
        T = len(shaped)
        G = [0.0] * (T + 1)
        for t in reversed(range(T)):
            G[t] = shaped[t] + gamma * G[t+1]
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
        "seat_is_self": seat_is_self,
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=125)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--four-player-prob", type=float, default=0.5)
    ap.add_argument("--lb928-prob", type=float, default=0.5,
                    help="Probability any given seat is replaced by lb-928")
    ap.add_argument("--opp-temp", type=float, default=1.0)
    ap.add_argument("--shape-decay-iters", type=int, default=50)
    ap.add_argument("--teacher-kl", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-steps-per-iter", type=int, default=2)
    ap.add_argument("--monitor", default=".impala_v2_monitor.csv")
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
              "lb928_opp_seats", "self_opp_seats",
              "shape_weight", "n_samples", "n_labels",
              "loss", "policy_loss", "value_loss", "teacher_kl",
              "entropy", "wall_seconds"]
    if not mon.exists():
        with mon.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(args.workers, initializer=_worker_init,
                   initargs=(ckpt["kwargs"], teacher_state))

    print(f"impala_v2 self-play: iters={args.iters} workers={args.workers} "
          f"lb928_prob={args.lb928_prob} teacher_kl={args.teacher_kl} "
          f"shape_decay={args.shape_decay_iters}", flush=True)

    try:
        for it in range(args.iters):
            t0 = time.time()
            shape_weight = max(0.0, 1.0 - it / max(1, args.shape_decay_iters))
            state_bytes = serialize_state(model)
            tasks = []
            for _ in range(args.workers):
                n_players = 4 if random.random() < args.four_player_prob else 2
                tasks.append({
                    "weights": state_bytes,
                    "n_players": n_players,
                    "opp_temp": args.opp_temp,
                    "lb928_prob": args.lb928_prob,
                })
            results = pool.map(_worker_play, tasks)

            all_samples = []
            total_steps = 0
            seat_win_counts: dict[int, int] = defaultdict(int)
            lb928_opp_seats = 0
            self_opp_seats = 0
            for r in results:
                total_steps += r["steps"]
                mx = max(r["rewards"])
                for s, rr in enumerate(r["rewards"]):
                    if rr == mx:
                        seat_win_counts[s] += 1
                for s, is_self in enumerate(r["seat_is_self"]):
                    if is_self:
                        self_opp_seats += 1
                    else:
                        lb928_opp_seats += 1
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
                "lb928_opp_seats": lb928_opp_seats,
                "self_opp_seats": self_opp_seats,
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
                  f"self/lb928={self_opp_seats}/{lb928_opp_seats}  "
                  f"shape_w={shape_weight:.2f}  "
                  f"loss={row['loss']}  pol={row['policy_loss']}  "
                  f"val={row['value_loss']}  ent={row['entropy']}",
                  flush=True)

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
