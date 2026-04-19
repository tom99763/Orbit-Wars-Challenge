"""AlphaZero-style self-play + training for Orbit Wars.

Core design (matches user's requirement "all seats feedback to model"):
- Each game picks ONE seat as the learner; MCTS runs only from that seat's
  perspective. Other seats' actions come from the plain policy.
- **EVERY seat's experience trains the value network** — we record
  (obs, terminal_reward_for_seat) for all seats, not only the learner.
- Policy loss uses only the learner's MCTS visit targets.
- learner seat rotates randomly across games; over many games every seat
  gets MCTS-quality data.

Training signal:
  policy_loss = KL(π_nn, π_mcts)              # learner seat only
  value_loss  = MSE(v_nn, terminal_reward)    # ALL seats (winners + losers)
  total       = policy_loss + vf * value_loss - ent_coef * entropy(π_nn)

Flow (per iter):
  1. Master broadcasts weights to W workers
  2. Each worker plays 1 game (random 2p or 4p, random learner_seat),
     runs MCTS only on learner turns, records all seats' data
  3. Master aggregates:
        learner_records: list[(obs, π_mcts, terminal_reward)]
        other_records:   list[(obs, terminal_reward)]  (value-only)
  4. Gradient step(s) on combined loss
  5. Save checkpoint periodically; eval_watcher.py can poll separately

Notes:
- n_sims=10, k_samples=4 for smoke tests; bump for real runs
- Deepcopying kaggle_environments.Environment is the bottleneck; smoke
  first to measure actual wall time.
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

from training.model import OrbitAgent, sun_blocker_mask
from training.agent import _encode_obs, SHIPS_BUCKETS
from training.mcts import (
    MCTS, mcts_action_from_search, sample_joint_action, _forward_policy,
    _shared_obs,
)
from featurize import featurize_step


# ---------------------------------------------------------------
# Worker
# ---------------------------------------------------------------

_worker_state: dict = {}


def _worker_init(model_kwargs: dict):
    torch.set_num_threads(1)
    _worker_state["model"] = OrbitAgent(**model_kwargs)
    _worker_state["model"].eval()
    _worker_state["kwargs"] = model_kwargs


def _policy_action(model, env, seat: int, n_players: int, temperature: float = 1.0):
    """Sample a single joint action for `seat` from the current policy.
    Returns (env_move, sub_action)."""
    obs = _shared_obs(env, seat)
    tl, bl, _, planets_raw, pids, omask = _forward_policy(model, obs)
    sub, env_move, _ = sample_joint_action(
        tl, bl, planets_raw, omask, temperature=temperature
    )
    return env_move, sub


def _collect_featurized(env, seat: int, n_players: int,
                        ang_vel: float, init_planets: list, comet_ids: set,
                        action: list, t: int):
    """Reuse featurize.featurize_step on a reconstructed step-dict."""
    shared_obs = env.state[0].observation
    planets = list(shared_obs.get("planets") or [])
    fleets = list(shared_obs.get("fleets") or [])
    for p in planets:
        if int(p[0]) not in init_ids_set(init_planets):
            comet_ids.add(int(p[0]))
    step_dict = {
        "step": t,
        "planets": planets,
        "fleets": fleets,
        "action": action,
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
    return featurize_step(step_dict, seat, ang_vel, n_players,
                          init_planets, comet_ids)


def init_ids_set(init_planets: list) -> set:
    return {int(p[0]) for p in init_planets}


def _worker_play(task: dict) -> dict:
    from kaggle_environments import make
    # Load weights
    buf = io.BytesIO(task["weights"])
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    _worker_state["model"].load_state_dict(sd)
    model = _worker_state["model"]

    n_players = task["n_players"]
    learner_seat = task["learner_seat"]
    n_sims = task["n_sims"]
    k_samples = task["k_samples"]
    mcts_temp_schedule = task["mcts_temp"]  # temperature for visit sampling
    opp_temp = task["opp_temp"]

    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)

    # Capture ang_vel + init_planets once
    obs0 = env.state[0].observation
    ang_vel = float(obs0.get("angular_velocity") or 0.0)
    init_planets = list(obs0.get("initial_planets") or [])
    comet_ids: set = set()

    mcts = MCTS(model, n_sims=n_sims, k_samples=k_samples,
                opp_temperature=opp_temp, my_seat=learner_seat,
                n_players=n_players)

    # Records per seat: each a list of dicts
    records: dict[int, list] = {s: [] for s in range(n_players)}
    # Per learner turn we additionally store π_mcts
    learner_mcts: list[dict] = []   # aligned with records[learner_seat]

    t = 0
    mcts_calls = 0
    while not env.done:
        # Decide each seat's action
        actions: list = [None] * n_players
        # Learner: MCTS
        keys, stats = mcts.search(env, add_noise=True)
        # Decreasing temperature over time (more deterministic late game)
        temp = mcts_temp_schedule(t)
        sub, env_move, pi_target = mcts_action_from_search(
            keys, stats, temperature=temp
        )
        actions[learner_seat] = env_move
        mcts_calls += 1
        # Store learner record
        feat = _collect_featurized(
            env, learner_seat, n_players, ang_vel, init_planets, comet_ids,
            env_move, t
        )
        records[learner_seat].append(feat)
        # π_target is a dict over root children keys. Flatten to ordered list
        # aligned with stats['sub_actions'].
        learner_mcts.append({
            "keys": keys,                   # list of action keys
            "visits": stats["visits"],      # np.array [K]
            "sub_actions": stats["sub_actions"],  # list of per-K sub_action
            "chosen_sub": sub,              # the sub_action we actually picked
        })

        # Others: plain policy
        for s in range(n_players):
            if s == learner_seat:
                continue
            obs = _shared_obs(env, s)
            tl, bl, _, planets_raw, pids, omask = _forward_policy(model, obs)
            _sub, m, _ = sample_joint_action(
                tl, bl, planets_raw, omask, temperature=opp_temp
            )
            actions[s] = m
            feat = _collect_featurized(env, s, n_players, ang_vel,
                                       init_planets, comet_ids, m, t)
            records[s].append(feat)

        env.step(actions)
        t += 1

    # Terminal rewards
    rewards = [s.reward if s.reward is not None else 0 for s in env.state]

    # Pack
    return {
        "learner_seat": learner_seat,
        "n_players": n_players,
        "steps": t,
        "rewards": rewards,
        "learner_records": records[learner_seat],
        "learner_mcts": learner_mcts,
        "other_records": {s: records[s] for s in range(n_players)
                          if s != learner_seat},
        "mcts_calls": mcts_calls,
    }


# ---------------------------------------------------------------
# Loss
# ---------------------------------------------------------------

def pad_stack(samples: list, max_planets=64, max_fleets=64):
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
    }


def az_train_step(model, optimizer, all_records, vf_coef=1.0, ent_coef=0.01,
                  pol_coef=0.1):  # low pol_coef: composite-action log-prob is naturally large
    """One AlphaZero-style gradient update.

    all_records: list of dicts with keys:
      'obs'           : featurized dict (planets, fleets, globals, ...)
      'terminal_r'    : float terminal reward for this seat
      'mcts_pi'       : optional np.array [K] — None for non-learner seats
      'mcts_subs'     : optional list[list[tuple]] — per-K sub_actions
                        (to pick corresponding action log-probs)
    """
    # Split into (learner, other) for loss computation
    learner_idx = [i for i, r in enumerate(all_records) if r.get("mcts_pi") is not None]
    if not all_records:
        return None

    tensors = pad_stack([r["obs"] for r in all_records])
    if tensors is None:
        return None
    tgt_logits, bkt_logits, value = model(
        tensors["planets"], tensors["planet_mask"],
        tensors["fleets"], tensors["fleet_mask"],
        tensors["globals"], None
    )

    # Value loss: ALL seats (the whole point)
    term = torch.tensor([r["terminal_r"] for r in all_records], dtype=torch.float32)
    value_loss = 0.5 * (value - term).pow(2).mean()

    # Policy loss: only learner records. Compute -Σ π_mcts(a) · log π_nn(a)
    # Where log π_nn(a) for a composite action = Σ_i log π_nn(sub_i).
    policy_losses = []
    for i in learner_idx:
        r = all_records[i]
        pi_mcts = torch.tensor(r["mcts_pi"], dtype=torch.float32)  # [K]
        subs_list = r["mcts_subs"]  # list of length K, each a list[(src, tgt_cls, bkt)]
        # Compute log π_nn for each sampled composite action
        logp_composite = torch.zeros(len(subs_list), dtype=torch.float32)
        t_row = F.log_softmax(tgt_logits[i], dim=-1)   # [P, P+1]
        b_row = F.log_softmax(bkt_logits[i], dim=-1)   # [P, 4]
        for ai, sub in enumerate(subs_list):
            if not sub:
                # "pass" action — no log-prob contribution (or treat as log(1)=0)
                continue
            lp = 0.0
            valid = True
            for (src_i, tgt_cls, bkt) in sub:
                if src_i >= t_row.shape[0]:
                    valid = False; break
                lp = lp + t_row[src_i, tgt_cls] + b_row[src_i, bkt]
            if not valid:
                logp_composite[ai] = torch.tensor(-20.0)
            else:
                logp_composite[ai] = lp
        # Cross-entropy against MCTS target
        pl = -(pi_mcts * logp_composite).sum()
        policy_losses.append(pl)
    if policy_losses:
        policy_loss = torch.stack(policy_losses).mean()
    else:
        policy_loss = torch.tensor(0.0)

    # Simple policy entropy over the src-level distribution (regularisation)
    p_src = F.softmax(tgt_logits, dim=-1)
    entropy = -(p_src * F.log_softmax(tgt_logits, dim=-1)).sum(-1).mean()

    loss = pol_coef * policy_loss + vf_coef * value_loss - ent_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "n_total": len(all_records),
        "n_learner": len(learner_idx),
    }


# ---------------------------------------------------------------
# Master
# ---------------------------------------------------------------

class TempSchedule:
    """Picklable temperature schedule so multiprocessing.Pool can ship it."""
    __slots__ = ("high", "low", "switch_t")

    def __init__(self, high=1.0, low=0.1, switch_t=30):
        self.high = high; self.low = low; self.switch_t = switch_t

    def __call__(self, t):
        return self.high if t < self.switch_t else self.low


def serialize_state(model) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-sims", type=int, default=10,
                    help="MCTS sims per move (for learner seat only)")
    ap.add_argument("--k-samples", type=int, default=4,
                    help="Sampled actions per MCTS node (Sampled MuZero style)")
    ap.add_argument("--four-player-prob", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-steps-per-iter", type=int, default=2)
    ap.add_argument("--monitor", default=".alphazero_monitor.csv")
    args = ap.parse_args()

    ckpt = torch.load(args.bc_ckpt, map_location="cpu", weights_only=False)
    model = OrbitAgent(**ckpt["kwargs"])
    model.load_state_dict(ckpt["model"])
    torch.set_num_threads(2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mon = pathlib.Path(args.monitor)
    fields = ["ts", "iter", "games", "total_steps", "mcts_calls",
              "seat_win_counts", "n_total", "n_learner",
              "loss", "policy_loss", "value_loss", "entropy",
              "wall_seconds"]
    if not mon.exists():
        with mon.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(args.workers, initializer=_worker_init,
                   initargs=(ckpt["kwargs"],))

    temp_sched = TempSchedule(high=1.0, low=0.1, switch_t=30)
    print(f"AlphaZero self-play: iters={args.iters} workers={args.workers} "
          f"n_sims={args.n_sims} k_samples={args.k_samples}", flush=True)

    try:
        for it in range(args.iters):
            t0 = time.time()
            state_bytes = serialize_state(model)
            tasks = []
            for _ in range(args.workers):
                n_players = 4 if random.random() < args.four_player_prob else 2
                tasks.append({
                    "weights": state_bytes,
                    "n_players": n_players,
                    "learner_seat": random.randint(0, n_players - 1),
                    "n_sims": args.n_sims,
                    "k_samples": args.k_samples,
                    "mcts_temp": temp_sched,
                    "opp_temp": 1.0,
                })
            results = pool.map(_worker_play, tasks)

            # Aggregate all records across all 4 games
            all_records = []
            total_steps = 0
            total_mcts = 0
            seat_win_counts: dict[int, int] = defaultdict(int)
            for r in results:
                total_steps += r["steps"]
                total_mcts += r["mcts_calls"]
                rewards = r["rewards"]
                max_r = max(rewards)
                for ws, rr in enumerate(rewards):
                    if rr == max_r:
                        seat_win_counts[ws] += 1
                # Learner records with MCTS targets
                for feat, mcts_info in zip(r["learner_records"], r["learner_mcts"]):
                    visits = np.array(mcts_info["visits"], dtype=np.float32)
                    total_v = visits.sum()
                    pi = visits / total_v if total_v > 0 else np.ones_like(visits) / len(visits)
                    all_records.append({
                        "obs": feat,
                        "terminal_r": float(rewards[r["learner_seat"]]),
                        "mcts_pi": pi,
                        "mcts_subs": mcts_info["sub_actions"],
                    })
                # Other seats: value only
                for seat, recs in r["other_records"].items():
                    for feat in recs:
                        all_records.append({
                            "obs": feat,
                            "terminal_r": float(rewards[seat]),
                            "mcts_pi": None,
                            "mcts_subs": None,
                        })

            stats = None
            if all_records:
                for _ in range(args.grad_steps_per_iter):
                    stats = az_train_step(model, optimizer, all_records)

            dt = time.time() - t0
            st = stats or {}
            row = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "iter": it, "games": args.workers,
                "total_steps": total_steps, "mcts_calls": total_mcts,
                "seat_win_counts": ";".join(f"{k}:{v}" for k, v in
                                            sorted(seat_win_counts.items())),
                "n_total": st.get("n_total", 0),
                "n_learner": st.get("n_learner", 0),
                "loss": round(st.get("loss", 0), 4),
                "policy_loss": round(st.get("policy_loss", 0), 4),
                "value_loss": round(st.get("value_loss", 0), 4),
                "entropy": round(st.get("entropy", 0), 4),
                "wall_seconds": round(dt, 1),
            }
            with mon.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"[iter {it:03d}/{args.iters}] {dt:.1f}s  "
                  f"games={args.workers}  wins={dict(seat_win_counts)}  "
                  f"mcts_calls={total_mcts}  "
                  f"n_lrn/tot={row['n_learner']}/{row['n_total']}  "
                  f"loss={row['loss']}  "
                  f"pol={row['policy_loss']}  val={row['value_loss']}  "
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
