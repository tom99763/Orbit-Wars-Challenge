"""Online A2C-style self-play with BC-from-winners training signal.

Per iteration:
  1. Master broadcasts current weights to 4 persistent workers (via state_dict bytes)
  2. Each worker plays 1 full game (randomly 2p or 4p) with the current policy
     at temperature T, returns per-step winner features + labels
  3. Master concatenates all 4 games' training data
  4. Master runs N gradient steps (BC on winner's actions) and saves a new
     checkpoint; the next iter's workers will load it
  5. Per-iter reward / loss stats go to .a2c_monitor.csv

Notes:
 - "BC-from-winners" avoids the pathologies we saw in PPO (teacher-KL drag,
   Adv normalisation noise with tiny batches, shaped-reward contamination).
 - Workers stay alive via multiprocessing.Pool; they re-load state_dict per
   task but don't re-import torch.
 - CPU only (both master and workers). Small 0.9M model, fast enough.

Usage:
  python training/online_a2c.py --bc-ckpt training/checkpoints/bc_v2.pt \
      --out training/checkpoints/a2c_v1.pt \
      --iters 250 --workers 4 --temperature 1.0 --four-player-prob 0.5
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
from featurize import featurize_step


# ---------------------------------------------------------------
# Worker
# ---------------------------------------------------------------

_worker_state: dict = {}


def _worker_init(model_kwargs: dict):
    """Called once per worker. Builds a local CPU model (empty weights)."""
    # Avoid thread oversubscription; each worker uses 1 thread
    torch.set_num_threads(1)
    _worker_state["model"] = OrbitAgent(**model_kwargs)
    _worker_state["model"].eval()
    _worker_state["kwargs"] = model_kwargs


def _agent_fn_from_model(model: OrbitAgent, temperature: float):
    """Build a callable agent(obs) that uses `model` in-process."""
    from training.model import sun_blocker_mask  # local import ok

    def agent(obs):
        obs_d = obs if isinstance(obs, dict) else dict(obs)
        pf, pxy, pids, omask, ff, g, planets_raw, player = _encode_obs(obs_d)
        if not omask.any() or len(planets_raw) == 0:
            return []
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
        with torch.no_grad():
            tgt_logits, bkt_logits, _ = model(
                planets, planet_mask, fleets, fleet_mask, globals_, tgt_mask
            )
        moves = []
        tgt_l = tgt_logits[0].cpu().numpy()
        bkt_l = bkt_logits[0].cpu().numpy()
        for si in np.where(omask)[0]:
            row = tgt_l[si]
            if temperature > 0:
                z = row / max(temperature, 1e-6); z = z - z.max()
                p = np.exp(z); p = p / p.sum()
                tgt_class = int(np.random.choice(len(p), p=p))
            else:
                tgt_class = int(row.argmax())
            if tgt_class == 0:
                continue
            tgt_idx = tgt_class - 1
            if tgt_idx >= len(planets_raw):
                continue
            if temperature > 0:
                z = bkt_l[si] / max(temperature, 1e-6); z = z - z.max()
                p = np.exp(z); p = p / p.sum()
                bkt_idx = int(np.random.choice(len(p), p=p))
            else:
                bkt_idx = int(bkt_l[si].argmax())
            frac = SHIPS_BUCKETS[bkt_idx]
            src = planets_raw[si]
            tgt = planets_raw[tgt_idx]
            garrison = int(src[5])
            ships = max(1, int(round(frac * garrison)))
            if ships <= 0 or ships > garrison:
                continue
            ang = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
            moves.append([int(src[0]), float(ang), int(ships)])
        return moves
    return agent


def _worker_play(task: dict) -> dict:
    """Run one game and return features for the winner(s)."""
    from kaggle_environments import make
    state_bytes = task["weights"]
    temperature = task["temperature"]
    n_players = task["n_players"]
    # Load weights into persistent local model
    buf = io.BytesIO(state_bytes)
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    _worker_state["model"].load_state_dict(sd)
    agent_fn = _agent_fn_from_model(_worker_state["model"], temperature)

    env = make("orbit_wars", debug=False)
    env.run([agent_fn] * n_players)

    # Terminal reward per seat (+1 / −1 in orbit_wars)
    rewards = [s.reward if s.reward is not None else 0 for s in env.steps[-1]]
    max_r = max(rewards)
    winner_seats = [i for i, r in enumerate(rewards) if r == max_r]

    # Collect training data from EVERY seat, tagged with terminal reward.
    # This lets the master weight gradients by advantage (reward − value).
    samples = []  # list of dicts: {feat_dict, seat, terminal_reward}
    ang_vel = 0.0
    for step in env.steps:
        obs0 = step[0].observation
        av = obs0.get("angular_velocity") if hasattr(obs0, "get") else None
        if av:
            ang_vel = float(av)
            break
    init_planets = []
    try:
        ip = env.steps[0][0].observation.get("initial_planets")
        if ip:
            init_planets = list(ip)
    except Exception:
        pass
    init_ids = {int(p[0]) for p in init_planets}
    comet_ids: set = set()

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
                continue  # only train on steps where THIS seat acted
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
            feat = featurize_step(step_dict, seat, ang_vel, n_players,
                                  init_planets, comet_ids)
            feat["_terminal_reward"] = float(rewards[seat])
            feat["_seat"] = seat
            samples.append(feat)

    return {
        "samples": samples,
        "n_players": n_players,
        "steps": len(env.steps),
        "rewards": rewards,
        "winner_seats": winner_seats,
    }


# ---------------------------------------------------------------
# Master
# ---------------------------------------------------------------

def policy_gradient_step(model, optimizer, samples,
                         max_planets=64, max_fleets=64,
                         vf_coef: float = 0.5, ent_coef: float = 0.01):
    """REINFORCE with learned baseline. Every seat's (s, a, r) tuple is used:
       loss = −(advantage · log π(a|s))  +  vf · (V(s) − r)²  −  ent · H(π)
    where advantage = terminal_reward − V(s).detach().

    Winners' samples have advantage > 0 → policy pushes UP log π(a|s)
    Losers' samples have advantage < 0 → policy pushes DOWN log π(a|s)
    """
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
    terminal_r = np.zeros((B,), dtype=np.float32)

    flat_batch, flat_src, flat_tgt, flat_bkt = [], [], [], []

    for i, s in enumerate(samples):
        N = min(s["planets"].shape[0], P)
        planets[i, :N] = s["planets"][:N]
        planet_mask[i, :N] = True
        nf = min(s["fleets"].shape[0], FN)
        if nf > 0:
            fleets[i, :nf] = s["fleets"][:nf]
            fleet_mask[i, :nf] = True
        globals_[i] = s["globals"]
        terminal_r[i] = s["_terminal_reward"]
        src = s["src_planet_idx"]
        tgt = s["target_planet_idx"]
        bkt = s["ships_bucket"]
        valid = (src < N) & (tgt < N)
        for j in range(len(src)):
            if valid[j]:
                flat_batch.append(i)
                flat_src.append(int(src[j]))
                flat_tgt.append(int(tgt[j]))
                flat_bkt.append(int(bkt[j]))

    if not flat_batch:
        return None

    planets_t = torch.from_numpy(planets)
    planet_mask_t = torch.from_numpy(planet_mask)
    fleets_t = torch.from_numpy(fleets)
    fleet_mask_t = torch.from_numpy(fleet_mask)
    globals_t = torch.from_numpy(globals_)
    term_t = torch.from_numpy(terminal_r)
    fb = torch.tensor(flat_batch, dtype=torch.long)
    fs = torch.tensor(flat_src, dtype=torch.long)
    ft = torch.tensor(flat_tgt, dtype=torch.long)
    fk = torch.tensor(flat_bkt, dtype=torch.long)

    tgt_logits, bkt_logits, value = model(
        planets_t, planet_mask_t, fleets_t, fleet_mask_t, globals_t, None
    )
    # value: [B]; each sample i has target terminal_r[i]
    advantage = (term_t - value.detach())  # per-sample
    # Normalize to stabilize scale — like PPO
    adv_mean, adv_std = advantage.mean(), advantage.std().clamp(min=1e-6)
    advantage = (advantage - adv_mean) / adv_std

    picked_tgt = tgt_logits[fb, fs]          # [K, P+1]
    picked_bkt = bkt_logits[fb, fs]          # [K, 4]
    tgt_labels = ft + 1

    logp_tgt = F.log_softmax(picked_tgt, dim=-1).gather(
        1, tgt_labels.unsqueeze(1)).squeeze(1)
    logp_bkt = F.log_softmax(picked_bkt, dim=-1).gather(
        1, fk.unsqueeze(1)).squeeze(1)
    logp = logp_tgt + 0.5 * logp_bkt          # joint sub-action log-prob

    # Per-label advantage: map batch indices → per-sample advantage
    adv_per_label = advantage[fb]
    # Clamp to avoid exploding gradient on extreme advantages
    adv_per_label = torch.clamp(adv_per_label, -2.0, 2.0)

    policy_loss = -(adv_per_label * logp).mean()
    value_loss = 0.5 * (value - term_t).pow(2).mean()
    # Policy entropy (approx, over src actions only — cheap but informative)
    p_tgt = F.softmax(picked_tgt, dim=-1)
    entropy = -(p_tgt * F.log_softmax(picked_tgt, dim=-1)).sum(-1).mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        acc_tgt = (picked_tgt.argmax(-1) == tgt_labels).float().mean().item()
        acc_bkt = (picked_bkt.argmax(-1) == fk).float().mean().item()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "acc_tgt": acc_tgt,
        "acc_bkt": acc_bkt,
        "n_labels": len(flat_batch),
        "n_samples": len(samples),
        "adv_mean": adv_mean.item(),
        "adv_std": (advantage.std().item()),
    }


def serialize_state(model) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--four-player-prob", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-steps-per-iter", type=int, default=3,
                    help="Gradient steps per batch of 4 games")
    ap.add_argument("--monitor", default=".a2c_monitor.csv")
    args = ap.parse_args()

    # Load initial model
    ckpt = torch.load(args.bc_ckpt, map_location="cpu", weights_only=False)
    model = OrbitAgent(**ckpt["kwargs"])
    model.load_state_dict(ckpt["model"])
    torch.set_num_threads(2)  # master gets 2 threads for backward pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_path = pathlib.Path(args.monitor)
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_fields = ["ts", "iter", "games", "total_steps",
                      "seat_win_counts", "rewards", "n_samples", "n_labels",
                      "loss", "policy_loss", "value_loss", "entropy",
                      "acc_tgt", "acc_bkt", "iter_wall_seconds"]
    if not monitor_path.exists():
        with monitor_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=monitor_fields).writeheader()

    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(args.workers, initializer=_worker_init,
                   initargs=(ckpt["kwargs"],))

    print(f"A2C self-play: iters={args.iters} workers={args.workers} "
          f"temp={args.temperature} 4p_prob={args.four_player_prob} "
          f"lr={args.lr}", flush=True)

    try:
        for it in range(args.iters):
            t0 = time.time()
            # Build tasks
            state_bytes = serialize_state(model)
            tasks = []
            for _ in range(args.workers):
                tasks.append({
                    "weights": state_bytes,
                    "temperature": args.temperature,
                    "n_players": 4 if random.random() < args.four_player_prob else 2,
                })
            results = pool.map(_worker_play, tasks)

            # Gather
            all_samples = []
            all_rewards = []
            total_steps = 0
            seat_win_counts: dict[int, int] = defaultdict(int)
            for r in results:
                all_samples.extend(r["samples"])
                all_rewards.append((r["n_players"], r["rewards"]))
                total_steps += r["steps"]
                for w in r["winner_seats"]:
                    seat_win_counts[w] += 1

            # Policy-gradient steps on the fresh batch (all seats)
            stats = None
            if all_samples:
                for _ in range(args.grad_steps_per_iter):
                    stats = policy_gradient_step(model, optimizer, all_samples)

            dt = time.time() - t0
            st = stats or {}
            row = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "iter": it,
                "games": args.workers,
                "total_steps": total_steps,
                "seat_win_counts": ";".join(f"{k}:{v}" for k, v in sorted(seat_win_counts.items())),
                "rewards": " | ".join(f"{n}p:{','.join(map(str,r))}"
                                      for n, r in all_rewards),
                "n_samples": st.get("n_samples", 0),
                "n_labels": st.get("n_labels", 0),
                "loss": round(st.get("loss", 0), 4) if stats else "",
                "policy_loss": round(st.get("policy_loss", 0), 4) if stats else "",
                "value_loss": round(st.get("value_loss", 0), 4) if stats else "",
                "entropy": round(st.get("entropy", 0), 4) if stats else "",
                "acc_tgt": round(st.get("acc_tgt", 0), 4) if stats else "",
                "acc_bkt": round(st.get("acc_bkt", 0), 4) if stats else "",
                "iter_wall_seconds": round(dt, 1),
            }
            with monitor_path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=monitor_fields).writerow(row)

            print(f"[iter {it:03d}/{args.iters}] {dt:.1f}s  "
                  f"games={args.workers}  "
                  f"winners={dict(seat_win_counts)}  "
                  f"n_samp={row['n_samples']}  n_lbl={row['n_labels']}  "
                  f"pol_loss={row['policy_loss']}  "
                  f"val_loss={row['value_loss']}  "
                  f"ent={row['entropy']}  "
                  f"tgt_acc={row['acc_tgt']}",
                  flush=True)

            # Save checkpoint periodically
            if (it + 1) % 20 == 0 or it == args.iters - 1:
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
