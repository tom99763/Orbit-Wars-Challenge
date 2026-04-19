"""impala_v4: Physics-accurate aiming on top of v3.

Everything from v3 (exploration schedules, versioned ckpts, vs928 tracking,
reward / adv logging) plus:

 * Worker passes `physics_aim_ctx` to sample_joint_action. Launch angles are
   computed via lb928's `aim_with_prediction` so fleets are aimed at the
   target planet's PREDICTED position at arrival time (avoids sun).
   Fallback to atan2 only if physics solver finds no safe intercept.

This fixes the dominant geometry gap where the learned policy was sending
fleets to where orbital targets ARE NOW (they move during flight).

Launch:
  python training/online_impala_v4.py \
      --bc-ckpt training/checkpoints/bc_v2.pt \
      --out training/checkpoints/impala_v4.pt \
      --iters 125 --workers 4 --four-player-prob 0.2 \
      --lb928-prob 0.5 --random-opp-prob 0.0 --random-opp-end 0.0 \
      --planet-action-noise 0.10 --shape-decay-iters 50 --lr 1e-4 --random-init
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

from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

from training.model import OrbitAgent
from training.agent import _encode_obs, SHIPS_BUCKETS
from training.mcts import (
    sample_joint_action, _forward_policy, _shared_obs,
    disable_env_validation,
)
from training.lb928_agent import agent as lb928_agent_fn
from training.online_impala import (
    compute_shaped_rewards, pad_stack, impala_train_step, serialize_state,
)
from featurize import featurize_step


# ---------------------------------------------------------------
# Schedules (picklable)
# ---------------------------------------------------------------

class LinearSchedule:
    __slots__ = ("start", "end", "total_iters")
    def __init__(self, start, end, total_iters):
        self.start = float(start); self.end = float(end)
        self.total_iters = int(total_iters)
    def __call__(self, it):
        if self.total_iters <= 0: return self.end
        t = min(1.0, it / self.total_iters)
        return self.start + t * (self.end - self.start)


class StepSchedule:
    __slots__ = ("high", "low", "switch_iter")
    def __init__(self, high, low, switch_iter):
        self.high = float(high); self.low = float(low); self.switch_iter = int(switch_iter)
    def __call__(self, it):
        return self.high if it < self.switch_iter else self.low


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


def _pick_opponent_types(n_players: int, lb928_prob: float,
                         random_opp_prob: float) -> list[str]:
    """Per-seat: 'self', 'lb928', or 'random'. At least 1 self guaranteed."""
    types = []
    for _ in range(n_players):
        r = random.random()
        if r < lb928_prob:
            types.append("lb928")
        elif r < lb928_prob + random_opp_prob:
            types.append("random")
        else:
            types.append("self")
    if not any(t == "self" for t in types):
        types[random.randint(0, n_players - 1)] = "self"
    return types


def _worker_play(task: dict) -> dict:
    from kaggle_environments import make

    buf = io.BytesIO(task["weights"])
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    _worker_state["model"].load_state_dict(sd)
    model = _worker_state["model"]

    n_players = task["n_players"]
    opp_temp = task["opp_temp"]
    lb928_prob = task["lb928_prob"]
    random_opp_prob = task["random_opp_prob"]
    planet_action_noise = task["planet_action_noise"]

    seat_types = _pick_opponent_types(n_players, lb928_prob, random_opp_prob)

    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)
    obs0 = env.state[0].observation
    ang_vel = float(obs0.get("angular_velocity") or 0.0)
    init_planets = list(obs0.get("initial_planets") or [])
    init_ids = {int(p[0]) for p in init_planets}
    comet_ids: set = set()

    while not env.done:
        actions: list = [None] * n_players
        step_obs0 = env.state[0].observation
        step_comets = list(step_obs0.get("comets") or [])
        for p in list(step_obs0.get("planets") or []):
            if int(p[0]) not in init_ids:
                comet_ids.add(int(p[0]))
        physics_ctx = {
            "ang_vel": ang_vel,
            "initial_planets": init_planets,
            "comets": step_comets,
            "comet_ids": comet_ids,
        }
        for seat in range(n_players):
            obs = _shared_obs(env, seat)
            if seat_types[seat] == "self":
                tl, bl, _, planets_raw, pids, omask = _forward_policy(model, obs)
                _sub, m, _ = sample_joint_action(
                    tl, bl, planets_raw, omask,
                    temperature=opp_temp,
                    planet_action_noise=planet_action_noise,
                    physics_aim_ctx=physics_ctx,
                )
                actions[seat] = m
            elif seat_types[seat] == "lb928":
                try:
                    actions[seat] = lb928_agent_fn(obs) or []
                except Exception:
                    actions[seat] = []
            else:  # random
                try:
                    actions[seat] = random_agent(obs) or []
                except Exception:
                    actions[seat] = []
        env.step(actions)

    records: dict[int, list] = {s: [] for s, t in enumerate(seat_types) if t == "self"}
    for t, step in enumerate(env.steps):
        shared_obs = step[0].observation
        planets = list(shared_obs.get("planets") or [])
        fleets = list(shared_obs.get("fleets") or [])
        for p in planets:
            if int(p[0]) not in init_ids:
                comet_ids.add(int(p[0]))
        for seat in range(n_players):
            if seat_types[seat] != "self":
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

    seat_type_counts = {k: seat_types.count(k) for k in ("self", "lb928", "random")}

    # Per-type winners (a game may have ties at max reward → multiple winners)
    max_r = max(rewards)
    type_wins = {"self": 0, "lb928": 0, "random": 0}
    for s, rr in enumerate(rewards):
        if rr == max_r:
            type_wins[seat_types[s]] += 1

    return {
        "n_players": n_players,
        "steps": len(env.steps),
        "rewards": rewards,
        "seat_types": seat_types,
        "seat_type_counts": seat_type_counts,
        "type_wins": type_wins,
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-ckpt", required=True)
    ap.add_argument("--out", required=True,
                    help="Base name; saves <base>.pt (latest) + <base>_iter<N>.pt (per-N)")
    ap.add_argument("--iters", type=int, default=125)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--four-player-prob", type=float, default=0.5)
    ap.add_argument("--lb928-prob", type=float, default=0.1,
                    help="Probability any non-learner seat is lb-928")
    ap.add_argument("--random-opp-prob", type=float, default=0.06,
                    help="Probability a non-learner seat is random (starting value)")
    ap.add_argument("--random-opp-end", type=float, default=0.0,
                    help="Random opponent probability at end of --random-opp-decay-iters")
    ap.add_argument("--random-opp-decay-iters", type=int, default=80,
                    help="Iters over which random-opp-prob linearly decays start→end")
    ap.add_argument("--planet-action-noise", type=float, default=0.10,
                    help="Per-planet prob to replace sampled action with random legal one")
    ap.add_argument("--temp-start", type=float, default=1.2)
    ap.add_argument("--temp-end", type=float, default=0.7)
    ap.add_argument("--ent-high", type=float, default=0.03)
    ap.add_argument("--ent-low", type=float, default=0.005)
    ap.add_argument("--ent-switch-iter", type=int, default=30)
    ap.add_argument("--shape-decay-iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-steps-per-iter", type=int, default=2)
    ap.add_argument("--snapshot-every", type=int, default=10,
                    help="Save versioned ckpt every N iters")
    ap.add_argument("--monitor", default=".impala_v4_monitor.csv")
    # Architecture overrides — if any set, start from random init (skip warm-start)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--random-init", action="store_true",
                    help="Skip bc ckpt weight loading even if arch matches")
    args = ap.parse_args()

    ckpt = torch.load(args.bc_ckpt, map_location="cpu", weights_only=False)
    kwargs = dict(ckpt["kwargs"])
    arch_overridden = False
    if args.d_model is not None:
        kwargs["d_model"] = args.d_model; arch_overridden = True
    if args.n_layers is not None:
        kwargs["n_layers"] = args.n_layers; arch_overridden = True
    if args.n_heads is not None:
        kwargs["n_heads"] = args.n_heads; arch_overridden = True
    model = OrbitAgent(**kwargs)
    if arch_overridden or args.random_init:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"RANDOM INIT — {n_params/1e6:.2f}M params, kwargs={kwargs}"
              + (" (arch overridden)" if arch_overridden else ""), flush=True)
        ckpt["kwargs"] = kwargs
    else:
        model.load_state_dict(ckpt["model"])
    # Teacher (kept around for compatibility; unused when teacher_kl=0)
    # When arch overridden, teacher is just current model's copy
    teacher_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    teacher = OrbitAgent(**kwargs)
    teacher.load_state_dict(teacher_state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    torch.set_num_threads(2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)

    out_base = pathlib.Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    versioned_dir = out_base.parent
    base_stem = out_base.stem   # e.g. "impala_v3"

    mon = pathlib.Path(args.monitor)
    fields = ["ts", "iter", "games", "total_steps", "seat_win_counts",
              "self_seats", "lb928_seats", "random_seats",
              "self_wins", "lb928_wins", "random_wins",
              "vs928_games", "vs928_self_wins", "vs928_lb928_wins",
              "shape_weight", "entropy_coef", "temperature",
              "n_samples", "n_labels",
              "mean_term_r", "mean_shape_r",
              "loss", "policy_loss", "value_loss", "entropy",
              "adv_mean", "adv_std",
              "wall_seconds"]
    if not mon.exists():
        with mon.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    temp_sched = LinearSchedule(args.temp_start, args.temp_end, args.iters)
    ent_sched = StepSchedule(args.ent_high, args.ent_low, args.ent_switch_iter)
    random_opp_sched = LinearSchedule(args.random_opp_prob, args.random_opp_end,
                                      args.random_opp_decay_iters)

    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(args.workers, initializer=_worker_init,
                   initargs=(kwargs, teacher_state))

    print(f"impala_v4 self-play (physics aim): iters={args.iters} workers={args.workers}", flush=True)
    print(f"  lb928_prob={args.lb928_prob}  "
          f"random_opp {args.random_opp_prob}→{args.random_opp_end} "
          f"over {args.random_opp_decay_iters} iters", flush=True)
    print(f"  planet_action_noise={args.planet_action_noise}", flush=True)
    print(f"  temp {args.temp_start}→{args.temp_end},  "
          f"ent {args.ent_high}→{args.ent_low} at iter {args.ent_switch_iter}",
          flush=True)
    print(f"  shape_decay_iters={args.shape_decay_iters}  "
          f"snapshot_every={args.snapshot_every}", flush=True)

    try:
        for it in range(args.iters):
            t0 = time.time()
            shape_w = max(0.0, 1.0 - it / max(1, args.shape_decay_iters))
            temp_t = temp_sched(it)
            ent_coef_t = ent_sched(it)

            state_bytes = serialize_state(model)
            random_opp_t = random_opp_sched(it)
            tasks = []
            for _ in range(args.workers):
                n_players = 4 if random.random() < args.four_player_prob else 2
                tasks.append({
                    "weights": state_bytes,
                    "n_players": n_players,
                    "opp_temp": temp_t,
                    "lb928_prob": args.lb928_prob,
                    "random_opp_prob": random_opp_t,
                    "planet_action_noise": args.planet_action_noise,
                })
            results = pool.map(_worker_play, tasks)

            all_samples = []
            total_steps = 0
            seat_win_counts: dict[int, int] = defaultdict(int)
            self_seats = lb928_seats = random_seats = 0
            self_wins = lb928_wins = random_wins = 0
            vs928_games = vs928_self_wins = vs928_lb928_wins = 0
            for r in results:
                total_steps += r["steps"]
                mx = max(r["rewards"])
                for s, rr in enumerate(r["rewards"]):
                    if rr == mx:
                        seat_win_counts[s] += 1
                c = r["seat_type_counts"]
                self_seats += c["self"]
                lb928_seats += c["lb928"]
                random_seats += c["random"]
                tw = r["type_wins"]
                self_wins += tw["self"]
                lb928_wins += tw["lb928"]
                random_wins += tw["random"]
                if "lb928" in r["seat_types"]:
                    vs928_games += 1
                    winner_types = {r["seat_types"][s]
                                    for s, rr in enumerate(r["rewards"])
                                    if rr == mx}
                    if "self" in winner_types:
                        vs928_self_wins += 1
                    if "lb928" in winner_types:
                        vs928_lb928_wins += 1
                for seat, recs in r["records"].items():
                    all_samples.extend(recs)

            mean_term_r = 0.0
            mean_shape_r = 0.0
            if all_samples:
                mean_term_r = sum(s["_terminal_reward"] for s in all_samples) / len(all_samples)
                mean_shape_r = sum(s["_shape_return"] for s in all_samples) / len(all_samples)

            stats = None
            if all_samples:
                for _ in range(args.grad_steps_per_iter):
                    stats = impala_train_step(
                        model, teacher, optimizer, all_samples,
                        shape_weight=shape_w,
                        teacher_kl_coef=0.0,
                        ent_coef=ent_coef_t,
                    )

            dt = time.time() - t0
            st = stats or {}
            row = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "iter": it, "games": args.workers,
                "total_steps": total_steps,
                "seat_win_counts": ";".join(f"{k}:{v}" for k, v in
                                            sorted(seat_win_counts.items())),
                "self_seats": self_seats,
                "lb928_seats": lb928_seats,
                "random_seats": random_seats,
                "self_wins": self_wins,
                "lb928_wins": lb928_wins,
                "random_wins": random_wins,
                "vs928_games": vs928_games,
                "vs928_self_wins": vs928_self_wins,
                "vs928_lb928_wins": vs928_lb928_wins,
                "shape_weight": round(shape_w, 3),
                "entropy_coef": round(ent_coef_t, 4),
                "temperature": round(temp_t, 3),
                "n_samples": st.get("n_samples", 0),
                "n_labels": st.get("n_labels", 0),
                "mean_term_r": round(mean_term_r, 4),
                "mean_shape_r": round(mean_shape_r, 4),
                "loss": round(st.get("loss", 0), 4),
                "policy_loss": round(st.get("policy_loss", 0), 4),
                "value_loss": round(st.get("value_loss", 0), 4),
                "entropy": round(st.get("entropy", 0), 4),
                "adv_mean": round(st.get("advantage_mean", 0), 4),
                "adv_std": round(st.get("advantage_std", 0), 4),
                "wall_seconds": round(dt, 1),
            }
            with mon.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            print(f"[iter {it:03d}/{args.iters}] {dt:.1f}s  "
                  f"wins[s/l/r]={self_wins}/{lb928_wins}/{random_wins}  "
                  f"seats[s/l/r]={self_seats}/{lb928_seats}/{random_seats}  "
                  f"vs928={vs928_self_wins}/{vs928_games}(lb={vs928_lb928_wins})  "
                  f"shape_w={shape_w:.2f} T={temp_t:.2f} "
                  f"ent_c={ent_coef_t:.3f} rnd={random_opp_t:.3f}  "
                  f"term_r={mean_term_r:+.3f} shape_r={mean_shape_r:+.3f}  "
                  f"loss={row['loss']}  pol={row['policy_loss']}  "
                  f"val={row['value_loss']}  ent={row['entropy']}  "
                  f"adv={row['adv_mean']:+.2f}±{row['adv_std']:.2f}",
                  flush=True)

            if (it + 1) % args.snapshot_every == 0 or it == args.iters - 1:
                payload = {
                    "model": model.state_dict(),
                    "kwargs": kwargs,
                    "iter": it + 1,
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                }
                # Always overwrite "latest" for eval watcher
                torch.save(payload, out_base)
                # Versioned snapshot — preserved forever
                versioned = versioned_dir / f"{base_stem}_iter{it+1:03d}.pt"
                torch.save(payload, versioned)
                print(f"  → saved {out_base}  +  {versioned.name}", flush=True)
    finally:
        pool.close()
        pool.join()

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
