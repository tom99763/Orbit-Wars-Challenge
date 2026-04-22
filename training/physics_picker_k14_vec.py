"""k14 vec-env training — 8 workers × vec-env per worker.

Replaces kaggle_environments (slow Python sim) with OrbitWarsVecEnv
(numpy-batched, 100-300× faster per step).  Each worker plays N_ENVS
games in parallel via vec env; 8 workers × N_ENVS_PER_WORKER games
per iter total.

Architecture
------------
  Main process:  policy on GPU, PPO training, opponent pool, eval.
  8 Workers:     policy on CPU, vec-env rollouts, return experience.

Usage
-----
  python training/physics_picker_k14_vec.py \\
      --warm-start training/checkpoints/physics_picker_k14.pt \\
      --workers 8 --n-envs-per-worker 8 --target-iters 200000 \\
      --eval-every 10 --eval-games 20 \\
      --out training/checkpoints/physics_picker_k14_vec.pt
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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tensordict import TensorDict

from featurize import featurize_step, PLANET_DIM, FLEET_DIM, GLOBAL_DIM, HISTORY_K
from training.dual_stream_model import rasterize_obs, N_SPATIAL_CHANNELS
from training.lb1200_agent import build_world
from training.physics_action_helper_k13 import (
    N_MODES, N_FRACS, MODE_NAMES, FRACTIONS, materialize_with_targets,
)
from training.physics_picker_k13_ppo import (
    DualStreamK13Agent, load_k12_into_k13, init_head_biases, GRID, GAMMA,
)
from training.orbit_wars_policy import (
    rollout_step, eval_batch, build_sample_td,
    MAX_PLANETS, MAX_FLEETS,
)
from training.orbit_wars_vec_env import OrbitWarsVecEnv

GAE_LAMBDA = 0.95


# ──────────────────────────────────────────────────────────────────────────────
# Worker globals
# ──────────────────────────────────────────────────────────────────────────────
_w_main: Optional[DualStreamK13Agent] = None
_w_opp:  Optional[DualStreamK13Agent] = None


def _worker_init(main_sd_bytes: bytes):
    global _w_main, _w_opp
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    _w_main = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM)
    buf = io.BytesIO(main_sd_bytes)
    sd = torch.load(buf, map_location="cpu", weights_only=False)
    _w_main.load_state_dict(sd)
    _w_main.eval()
    _w_opp = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM)
    _w_opp.eval()


# ──────────────────────────────────────────────────────────────────────────────
# Scripted baselines
# ──────────────────────────────────────────────────────────────────────────────
def _noop_action(_obs, _player): return []


def _random_action(obs, my_player):
    planets = obs.get("planets", []) or []
    mine = [p for p in planets if p[1] == my_player and p[5] > 2]
    if not mine: return []
    src = random.choice(mine)
    others = [p for p in planets if p[0] != src[0]]
    if not others: return []
    tgt = random.choice(others)
    ang = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
    return [[int(src[0]), float(ang), max(1, int(src[5] * random.uniform(0.3, 0.8)))]]


# ──────────────────────────────────────────────────────────────────────────────
# Vec-env rollout worker — plays N_ENVS games in parallel
# ──────────────────────────────────────────────────────────────────────────────
def _vec_rollout_worker(task: dict) -> dict:
    """Play `n_envs` games using vec env; collect experience for all train seats.

    task:
      n_envs, n_players, opp_type,
      opp_weights_bytes (for 'pool'), noise_prob (for 'noisy_lb*'),
      seed
    """
    n_envs      = task["n_envs"]
    n_players   = task["n_players"]
    opp_type    = task["opp_type"]
    seed        = task.get("seed", random.randint(0, 1 << 30))

    if opp_type == "pool" and task.get("opp_weights_bytes"):
        buf = io.BytesIO(task["opp_weights_bytes"])
        sd  = torch.load(buf, map_location="cpu", weights_only=False)
        _w_opp.load_state_dict(sd)
        _w_opp.eval()

    env = OrbitWarsVecEnv(n_envs=n_envs, n_players=n_players, seed=seed)

    # train_seats: in stochastic_self all seats train; else picker_seat only
    pick_seats = [random.randrange(n_players) for _ in range(n_envs)]
    train_seats: list[set] = []
    for eid in range(n_envs):
        if opp_type == "stochastic_self":
            train_seats.append(set(range(n_players)))
        else:
            train_seats.append({pick_seats[eid]})

    # Per-(env, seat) history for featurize_step
    histories = {
        (eid, s): {
            "obs_history":            collections.deque(maxlen=HISTORY_K),
            "action_history":         collections.deque(maxlen=HISTORY_K),
            "last_actions_by_planet": {},
            "cum_stats":              {"total_ships_sent": 0, "total_actions": 0},
        }
        for eid in range(n_envs) for s in range(n_players)
    }
    seat_samples = {(eid, s): [] for eid in range(n_envs) for s in range(n_players)}
    prev_phi     = {
        (eid, s): {"prod_phi": 0.0, "ship_phi": 0.0, "enemy_ship_phi": 0.0}
        for eid in range(n_envs) for s in range(n_players)
    }

    mode_counts = [0] * N_MODES
    frac_counts = [0] * N_FRACS
    steps_taken = 0

    while not env.done_mask.all() and steps_taken < 500:
        actions_batch: list[dict] = []
        for eid in range(n_envs):
            if env.done_mask[eid]:
                actions_batch.append({})
                continue
            obs = env.get_obs_dict(eid)
            env_actions = {}
            for seat in range(n_players):
                hist = histories[(eid, seat)]
                is_train = seat in train_seats[eid]

                if is_train or opp_type == "pool":
                    try:
                        world = build_world(obs)
                    except Exception:
                        env_actions[seat] = []
                        continue
                    raw_planets = obs.get("planets", []) or []
                    raw_fleets  = obs.get("fleets",  []) or []
                    step_dict = {
                        "step": int(obs.get("step", 0) or 0),
                        "planets": raw_planets, "fleets": raw_fleets, "action": [],
                        "my_total_ships":    sum(p[5] for p in raw_planets if p[1] == seat),
                        "enemy_total_ships": 0, "my_planet_count": 0,
                        "enemy_planet_count": 0, "neutral_planet_count": 0,
                    }
                    feat = featurize_step(
                        step_dict, seat, float(obs.get("angular_velocity", 0.02)),
                        n_players, obs.get("initial_planets", []) or [],
                        last_actions_by_planet=hist["last_actions_by_planet"],
                        cumulative_stats=hist["cum_stats"],
                        obs_history=list(hist["obs_history"]),
                        action_history=list(hist["action_history"]),
                    )
                    spatial = rasterize_obs(obs, seat, grid=GRID)

                    net = _w_main if is_train else _w_opp
                    picks, smpl = rollout_step(net, feat, spatial, world, seat,
                                               raw_fleets, "cpu")
                    action_list = materialize_with_targets(
                        [(p[0], p[1], p[2], p[6]) for p in picks], world, seat,
                    ) if picks else []
                    env_actions[seat] = action_list

                    if is_train and smpl is not None:
                        seat_samples[(eid, seat)].append(smpl)
                        if seat == pick_seats[eid]:
                            for p in picks:
                                mode_counts[p[1]] += 1
                                if p[1] != 0:
                                    frac_counts[p[2]] += 1
                else:
                    # scripted opponent
                    if opp_type == "noop":
                        action_list = _noop_action(obs, seat)
                    elif opp_type == "random":
                        action_list = _random_action(obs, seat)
                    elif opp_type in ("noisy_lb928", "noisy_lb1200"):
                        name = "lb928" if "928" in opp_type else "lb1200"
                        noise = task.get("noise_prob", 0.0)
                        if random.random() < noise:
                            action_list = _random_action(obs, seat)
                        else:
                            try:
                                if name == "lb928":
                                    from training.lb928_agent import agent as _a
                                else:
                                    from training.lb1200_agent import agent as _a
                                action_list = _a(obs) or []
                            except Exception:
                                action_list = []
                    else:
                        action_list = []
                    env_actions[seat] = action_list

                # update seat history
                raw_planets = obs.get("planets", []) or []
                hist["obs_history"].append({"planets": raw_planets, "step": steps_taken})
                for mv in action_list:
                    if len(mv) != 3:
                        continue
                    from featurize import nearest_target_index, ship_bucket_idx
                    src_id, ang_rad, ships = int(mv[0]), float(mv[1]), int(mv[2])
                    src_p = next((p for p in raw_planets if int(p[0]) == src_id), None)
                    if src_p is None:
                        continue
                    ti   = nearest_target_index(src_p, ang_rad, raw_planets)
                    tpid = int(raw_planets[ti][0]) if ti is not None else -1
                    garrison = int(src_p[5]) + ships
                    bi   = ship_bucket_idx(ships, max(1, garrison))
                    prev_ = hist["last_actions_by_planet"].get(src_id, (-1, 0, -1, 0))
                    hist["last_actions_by_planet"][src_id] = (tpid, bi, steps_taken, prev_[3] + 1)
                    hist["cum_stats"]["total_ships_sent"] += ships
                    hist["cum_stats"]["total_actions"]    += 1
                    hist["action_history"].append((src_id, tpid, bi, steps_taken))

            actions_batch.append(env_actions)

        _, rewards, dones = env.step(actions_batch)
        steps_taken += 1

        # PBRS per (eid, train_seat)
        for eid in range(n_envs):
            if env.done_mask[eid]:
                continue
            for seat in train_seats[eid]:
                if not seat_samples[(eid, seat)]:
                    continue
                cur_obs = env.get_obs_dict(eid)
                cur_pl  = cur_obs["planets"]
                cur_fl  = cur_obs["fleets"]
                tot_sh  = sum(p[5] for p in cur_pl) + sum(f[6] for f in cur_fl)
                tot_pr  = max(1, sum(p[6] for p in cur_pl))
                my_pr   = sum(p[6] for p in cur_pl if p[1] == seat)
                my_sh   = (sum(p[5] for p in cur_pl if p[1] == seat)
                           + sum(f[6] for f in cur_fl if f[1] == seat))
                en_sh   = (sum(p[5] for p in cur_pl if p[1] != seat and p[1] != -1)
                           + sum(f[6] for f in cur_fl  if f[1] != seat))
                prod_phi = my_pr / tot_pr
                ship_phi = my_sh / max(1, tot_sh)
                en_phi   = en_sh / max(1, tot_sh)
                pp = prev_phi[(eid, seat)]
                r = ((prod_phi - GAMMA * pp["prod_phi"])
                     + 0.5 * (ship_phi - GAMMA * pp["ship_phi"])
                     - 0.3 * (en_phi   - GAMMA * pp["enemy_ship_phi"]))
                seat_samples[(eid, seat)][-1]["reward"] = r
                prev_phi[(eid, seat)] = {"prod_phi": prod_phi, "ship_phi": ship_phi,
                                          "enemy_ship_phi": en_phi}

    # Terminal reward + GAE per (eid, seat)
    wins = {}
    for eid in range(n_envs):
        for seat in train_seats[eid]:
            steps = seat_samples[(eid, seat)]
            if not steps:
                continue
            # winner signal from env
            if env.done_mask[eid]:
                w = int(env.winner[eid])
            else:
                w = -1
            terminal = (1.0 if w == seat else (-1.0 if w >= 0 else 0.0))
            steps[-1]["reward"] += terminal
            # GAE
            adv = 0.0
            for t in range(len(steps) - 1, -1, -1):
                next_v = steps[t + 1]["value"] if t + 1 < len(steps) else 0.0
                delta  = steps[t]["reward"] + GAMMA * next_v - steps[t]["value"]
                adv    = delta + GAMMA * GAE_LAMBDA * adv
                steps[t]["advantage"]    = adv
                steps[t]["value_target"] = adv + steps[t]["value"]
        wins[eid] = int(env.winner[eid]) if env.done_mask[eid] else -1

    return {
        "seat_samples": seat_samples,
        "train_seats":  {eid: list(ts) for eid, ts in enumerate(train_seats)},
        "pick_seats":   pick_seats,
        "wins":         wins,
        "opp":          opp_type,
        "mode_counts":  mode_counts,
        "frac_counts":  frac_counts,
        "n_envs":       n_envs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Opponent pool
# ──────────────────────────────────────────────────────────────────────────────
class OpponentPool:
    def __init__(self, max_size=20):
        self.snapshots, self.tags = [], []
        self.max_size = max_size

    def add(self, sd_bytes, tag):
        self.snapshots.append(sd_bytes); self.tags.append(tag)
        if len(self.snapshots) > self.max_size:
            drop = len(self.snapshots) // 2
            self.snapshots.pop(drop); self.tags.pop(drop)

    def __len__(self): return len(self.snapshots)

    def sample_index(self) -> int:
        n = len(self.snapshots); r = random.random()
        if r < 0.5 and n >= 2:
            return n - 1 - random.randrange(min(5, n))
        if r < 0.8 and n >= 3:
            lo = n // 3; hi = max(lo + 1, 2 * n // 3)
            return random.randrange(lo, hi)
        return random.randrange(n)


def _opp_task(pool, iter_, lb_prob, lb_start, noise_end, noise_start):
    n = len(pool)
    noise = max(0.0, noise_start * (1.0 - (iter_ - lb_start) / max(1, noise_end - lb_start))) \
            if iter_ >= lb_start else 1.0
    if n < 2:
        return {"opp_type": "stochastic_self" if random.random() < 0.5 else "noop"}
    r = random.random()
    if iter_ >= lb_start and r < lb_prob:
        return {"opp_type": random.choice(["noisy_lb928", "noisy_lb1200"]),
                "noise_prob": noise}
    rr = (r - lb_prob) / max(1e-6, 1.0 - lb_prob) if iter_ >= lb_start else r
    if rr < 0.15:
        return {"opp_type": "noop"}
    if rr < 0.30:
        return {"opp_type": "stochastic_self"}
    return {"opp_type": "pool", "opp_weights_bytes": pool.snapshots[pool.sample_index()]}


# ──────────────────────────────────────────────────────────────────────────────
# Flatten rollouts into a training list
# ──────────────────────────────────────────────────────────────────────────────
def _flatten_rollouts(rollouts: list[dict]) -> tuple:
    all_samples = []
    wins = 0; total_games = 0
    opp_wins: dict = collections.defaultdict(lambda: [0, 0])
    mc = [0] * N_MODES; fc = [0] * N_FRACS

    for r in rollouts:
        n = r["n_envs"]
        for eid in range(n):
            total_games += 1
            pick_s = r["pick_seats"][eid]
            w = r["wins"].get(eid, -1)
            if w == pick_s: wins += 1
            opp_wins[r["opp"]][1] += 1
            if w == pick_s: opp_wins[r["opp"]][0] += 1
            for seat in r["train_seats"].get(eid, []):
                all_samples.extend(r["seat_samples"].get((eid, seat), []))
        for i, c in enumerate(r["mode_counts"]): mc[i] += c
        for i, c in enumerate(r["frac_counts"]): fc[i] += c

    return all_samples, wins, total_games, opp_wins, mc, fc


# ──────────────────────────────────────────────────────────────────────────────
# TensorDict batch build + PPO
# ──────────────────────────────────────────────────────────────────────────────
def _samples_to_td(samples):
    rows = []
    for s in samples:
        d = build_sample_td(s)
        d["advantage"]    = np.float32(s.get("advantage",    0.0))
        d["value_target"] = np.float32(s.get("value_target", s["value"]))
        rows.append(d)
    keys = list(rows[0].keys())
    def stack(k):
        return torch.from_numpy(np.stack([r[k] for r in rows], axis=0))
    return TensorDict({k: stack(k) for k in keys}, batch_size=[len(rows)])


def ppo_update(net, opt, all_samples, device,
               ppo_epochs=4, mini_batch=256, clip=0.2,
               val_clip=0.2, val_coef=0.5, ent_coef=0.01):
    if not all_samples:
        return {}
    full = _samples_to_td(all_samples)
    T = len(full)
    adv = full["advantage"].float()
    if adv.std() > 1e-6:
        full["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

    buf = ReplayBuffer(
        storage=LazyTensorStorage(max_size=max(T + 1, 2)),
        sampler=SamplerWithoutReplacement(drop_last=False),
        batch_size=min(mini_batch, T),
    )
    buf.extend(full)

    n_per_ep = max(1, math.ceil(T / min(mini_batch, T)))
    total = ppo_epochs * n_per_ep
    info = {"pi_loss": 0.0, "v_loss": 0.0, "ent": 0.0,
            "ratio_mean": 0.0, "ratio_max": 0.0, "n": 0}

    for _ in range(total):
        mini = buf.sample().to(device)
        lp_old = mini["log_prob_old"].float()
        v_old  = mini["value_old"].float()
        A      = mini["advantage"].float()
        vt     = mini["value_target"].float()

        lp_new, ent, v_new = eval_batch(net, mini, device)
        ratio = (lp_new - lp_old).exp()
        s1 = ratio * A
        s2 = ratio.clamp(1 - clip, 1 + clip) * A
        pi_loss = -torch.min(s1, s2).mean()
        v_clip_ = v_old + (v_new - v_old).clamp(-val_clip, val_clip)
        v_loss  = torch.max((v_new - vt) ** 2, (v_clip_ - vt) ** 2).mean()
        loss = pi_loss + val_coef * v_loss - ent_coef * ent.mean()

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        info["pi_loss"]    += pi_loss.item()
        info["v_loss"]     += v_loss.item()
        info["ent"]        += ent.mean().item()
        info["ratio_mean"] += ratio.mean().item()
        info["ratio_max"]   = max(info["ratio_max"], ratio.max().item())
        info["n"]          += 1

    n = max(1, info["n"])
    for k in ("pi_loss", "v_loss", "ent", "ratio_mean"):
        info[k] /= n
    info["T"] = T
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start",         default=None)
    ap.add_argument("--workers",            type=int, default=8)
    ap.add_argument("--n-envs-per-worker",  type=int, default=8)
    ap.add_argument("--target-iters",       type=int, default=200000)
    ap.add_argument("--ppo-epochs",         type=int, default=4)
    ap.add_argument("--mini-batch",         type=int, default=256)
    ap.add_argument("--lr",                 type=float, default=3e-4)
    ap.add_argument("--ent-coef-start",     type=float, default=0.05)
    ap.add_argument("--ent-coef-end",       type=float, default=0.01)
    ap.add_argument("--ent-decay-iters",    type=int, default=100)
    ap.add_argument("--clip",               type=float, default=0.2)
    ap.add_argument("--val-coef",           type=float, default=0.5)
    ap.add_argument("--snapshot-every",     type=int, default=5)
    ap.add_argument("--pool-size",          type=int, default=20)
    ap.add_argument("--eval-every",         type=int, default=10)
    ap.add_argument("--eval-games",         type=int, default=20)
    ap.add_argument("--lb-prob",            type=float, default=0.2)
    ap.add_argument("--lb-start-iter",      type=int, default=40)
    ap.add_argument("--noise-start-prob",   type=float, default=0.60)
    ap.add_argument("--noise-end-iter",     type=int, default=500)
    ap.add_argument("--four-player-prob",   type=float, default=0.2)
    ap.add_argument("--start-iter",         type=int, default=1)
    ap.add_argument("--out",                required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    ).to(device)
    if args.warm_start and Path(args.warm_start).exists():
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        net.load_state_dict(sd, strict=False)
        resume_iter = ckpt.get("iter", args.start_iter - 1) if isinstance(ckpt, dict) else args.start_iter - 1
        if args.start_iter == 1:
            args.start_iter = resume_iter + 1
        print(f"[k14-vec] loaded {args.warm_start} (iter {resume_iter})", flush=True)
    init_head_biases(net)

    opt  = torch.optim.Adam(net.parameters(), lr=args.lr)
    pool = OpponentPool(max_size=args.pool_size)

    def sdb():
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
        return buf.getvalue()

    ctx = mp.get_context("spawn")
    wp  = ctx.Pool(processes=args.workers, initializer=_worker_init, initargs=(sdb(),))
    print(f"[k14-vec] device={device}  "
          f"params={sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"workers={args.workers}  n_envs_per_worker={args.n_envs_per_worker}  "
          f"games/iter={args.workers * args.n_envs_per_worker}", flush=True)

    t0 = time.time()

    for iter_ in range(args.start_iter, args.start_iter + args.target_iters):
        is_eval = (iter_ % args.eval_every == 0)

        # Refresh worker weights every 5 iters (skip if eval follows)
        if iter_ % 5 == 0 and not is_eval:
            wp.close(); wp.join()
            wp = ctx.Pool(processes=args.workers, initializer=_worker_init, initargs=(sdb(),))

        # Build tasks — 8 workers × n_envs each
        tasks = []
        for w in range(args.workers):
            opp = _opp_task(pool, iter_, args.lb_prob, args.lb_start_iter,
                            args.noise_end_iter, args.noise_start_prob)
            n_p = 4 if random.random() < args.four_player_prob else 2
            tasks.append({
                "n_envs": args.n_envs_per_worker,
                "n_players": n_p,
                "seed": random.randint(0, 1 << 30),
                **opp,
            })
        rollouts = wp.map(_vec_rollout_worker, tasks)

        all_samples, wins, tg, ow, mc, fc = _flatten_rollouts(rollouts)

        alpha = min(1.0, (iter_ - args.start_iter) / max(1, args.ent_decay_iters))
        ent_c = args.ent_coef_start + alpha * (args.ent_coef_end - args.ent_coef_start)
        info = ppo_update(net, opt, all_samples, device,
                          ppo_epochs=args.ppo_epochs, mini_batch=args.mini_batch,
                          clip=args.clip, val_coef=args.val_coef, ent_coef=ent_c)

        if iter_ % args.snapshot_every == 0:
            pool.add(sdb(), f"iter{iter_:04d}")

        elapsed = time.time() - t0
        opp_str = "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in ow.items() if v[1])
        mc_str  = " ".join(str(c) for c in mc)
        fc_str  = " ".join(str(c) for c in fc)
        print(
            f"[iter {iter_:05d}]  "
            f"wins={wins}/{tg}  ({opp_str})  "
            f"T={info.get('T',0)}  "
            f"pi={info.get('pi_loss',0):.3f}  v={info.get('v_loss',0):.3f}  "
            f"ent={info.get('ent',0):.3f}  "
            f"r={info.get('ratio_mean',1):.2f}/{info.get('ratio_max',1):.1f}  "
            f"ent_c={ent_c:.3f}  pool={len(pool)}  "
            f"mc=[{mc_str}]  fc=[{fc_str}]  [{elapsed:.0f}s]",
            flush=True,
        )

        if iter_ % (args.snapshot_every * 2) == 0:
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, args.out)
            versioned = Path(args.out).parent / f"{Path(args.out).stem}_iter{iter_:05d}.pt"
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, versioned)
            print(f"[iter {iter_:05d}] saved {args.out} + {versioned.name}", flush=True)

        # Eval vs clean lb928 + lb1200
        if is_eval:
            wp.close(); wp.join()
            wp = ctx.Pool(processes=args.workers, initializer=_worker_init, initargs=(sdb(),))
            eval_tasks = []
            for opp_name in ("lb928", "lb1200"):
                # split args.eval_games into worker chunks
                per_worker = max(1, args.eval_games // args.workers)
                for _ in range(args.workers):
                    eval_tasks.append({
                        "n_envs": per_worker,
                        "n_players": 2,
                        "opp_type": f"noisy_{opp_name}",
                        "noise_prob": 0.0,
                        "seed": random.randint(0, 1 << 30),
                    })
            ev = wp.map(_vec_rollout_worker, eval_tasks)
            ew: dict = collections.defaultdict(lambda: [0, 0])
            for r in ev:
                n = r["n_envs"]
                for eid in range(n):
                    ps = r["pick_seats"][eid]
                    opp_name = "lb928" if "928" in r.get("opp", "") else "lb1200"
                    ew[opp_name][1] += 1
                    if r["wins"].get(eid, -1) == ps:
                        ew[opp_name][0] += 1
            parts = [f"{k}={w}/{g}" for k, (w, g) in sorted(ew.items())]
            print(f"[iter {iter_:05d}] EVAL  {' '.join(parts)}", flush=True)

    wp.close(); wp.join()
    torch.save({"model": net.state_dict(),
                "iter": args.start_iter + args.target_iters - 1,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, args.out)
    print(f"[k14-vec] final checkpoint saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
