"""k14 v2 — Multi-agent IPPO with TorchRL infrastructure.

Key improvements over k14_selfplay.py
--------------------------------------
1. ALL agents in self-play contribute experience  (2× data per game)
2. Pure on-policy  (no cross-iter replay mixing)
3. Mini-batch PPO epochs via TorchRL ReplayBuffer + SamplerWithoutReplacement
4. Consistent log_prob formula in collection and update (orbit_wars_policy.py)
5. GAE advantage stored at collection time; value not re-subtracted at update

TorchRL components used
-----------------------
  torchrl.objectives.value.functional.generalized_advantage_estimate  — GAE
  torchrl.data.ReplayBuffer + LazyTensorStorage                       — buffer
  torchrl.data.replay_buffers.samplers.SamplerWithoutReplacement      — PPO epochs

Action / reward design from k14_selfplay.py (unchanged):
  mode×target×frac factored actions, PBRS reward, opponent pool curriculum.

Usage
-----
  python training/physics_picker_k14_torchrl.py \\
      --warm-start training/checkpoints/physics_picker_k14.pt \\
      --workers 4 --games-per-iter 16 --ppo-epochs 4 --mini-batch 256 \\
      --out training/checkpoints/physics_picker_k14v2.pt
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import collections
import copy
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

# TorchRL components
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from tensordict import TensorDict

from featurize import featurize_step, PLANET_DIM, FLEET_DIM, GLOBAL_DIM, HISTORY_K
from training.dual_stream_model import rasterize_obs, N_SPATIAL_CHANNELS
from training.lb1200_agent import build_world
from training.physics_action_helper_k13 import (
    N_MODES, N_FRACS, MODE_NAMES, FRACTIONS,
    materialize_with_targets,
)
from training.physics_picker_k13_ppo import (
    DualStreamK13Agent, load_k12_into_k13, init_head_biases, GRID, GAMMA,
)
from training.orbit_wars_policy import (
    rollout_step, eval_batch, build_sample_td,
    MAX_PLANETS, MAX_FLEETS,
)

GAE_LAMBDA  = 0.95


# ──────────────────────────────────────────────────────────────────────────────
# Worker globals
# ──────────────────────────────────────────────────────────────────────────────

_worker_main: Optional[DualStreamK13Agent] = None
_worker_opp:  Optional[DualStreamK13Agent] = None


def _worker_init(main_sd_bytes: bytes):
    global _worker_main, _worker_opp
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    from kaggle_environments import make  # noqa: pre-warm
    _worker_main = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM)
    buf = io.BytesIO(main_sd_bytes)
    sd  = torch.load(buf, map_location="cpu", weights_only=False)
    _worker_main.load_state_dict(sd)
    _worker_main.eval()
    _worker_opp = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM)
    _worker_opp.eval()


# ──────────────────────────────────────────────────────────────────────────────
# Opponent helpers
# ──────────────────────────────────────────────────────────────────────────────

def _random_action(obs: dict, my_player: int) -> list:
    planets = obs.get("planets", []) or []
    mine = [p for p in planets if p[1] == my_player and p[5] > 2]
    if not mine:
        return []
    src = random.choice(mine)
    others = [p for p in planets if p[0] != src[0]]
    if not others:
        return []
    tgt = random.choice(others)
    ang = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
    return [[int(src[0]), float(ang), max(1, int(src[5] * random.uniform(0.2, 0.8)))]]


def _noisy_lb(obs: dict, my_player: int, name: str, noise: float) -> list:
    if random.random() < noise:
        return _random_action(obs, my_player)
    if name == "lb928":
        from training.lb928_agent import agent as _a
        return _a(obs) or []
    from training.lb1200_agent import agent as _a
    return _a(obs) or []


# ──────────────────────────────────────────────────────────────────────────────
# Rollout worker — ALL trainable agents collect experience
# ──────────────────────────────────────────────────────────────────────────────

def _rollout_game(task: dict) -> dict:
    """Play one game; collect per-step samples for every trainable agent.

    task keys:
      n_players, opp_type, picker_seat,
      opp_weights_bytes (pool), noise_prob
    """
    from kaggle_environments import make

    global _worker_opp
    n_players  = task["n_players"]
    pick_seat  = task["picker_seat"]
    opp_type   = task["opp_type"]

    # Determine which seats train (contribute grad)
    # stochastic_self: all seats train (use same policy)
    # others: only pick_seat trains
    train_seats = (set(range(n_players))
                   if opp_type == "stochastic_self"
                   else {pick_seat})

    if opp_type == "pool" and task.get("opp_weights_bytes"):
        buf = io.BytesIO(task["opp_weights_bytes"])
        sd  = torch.load(buf, map_location="cpu", weights_only=False)
        _worker_opp.load_state_dict(sd)
        _worker_opp.eval()

    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)

    histories = {
        s: {
            "obs_history":            collections.deque(maxlen=HISTORY_K),
            "action_history":         collections.deque(maxlen=HISTORY_K),
            "last_actions_by_planet": {},
            "cum_stats":              {"total_ships_sent": 0, "total_actions": 0},
        }
        for s in range(n_players)
    }
    ang_vel   = None
    init_plan = None

    # Per-seat: accumulated steps, PBRS state
    seat_samples = {s: [] for s in range(n_players)}
    prev_phi = {
        s: {"prod_phi": 0.0, "ship_phi": 0.0, "enemy_ship_phi": 0.0}
        for s in range(n_players)
    }

    mode_counts = [0] * N_MODES
    frac_counts = [0] * N_FRACS
    step = 0

    while not env.done and step < 500:
        actions_all = []

        for s in range(n_players):
            obs = env.state[s].observation
            if obs is None:
                actions_all.append([])
                continue

            if ang_vel is None:
                ang_vel   = float(obs.get("angular_velocity", 0.0) or 0.0)
                init_plan = obs.get("initial_planets", []) or []

            hist = histories[s]
            raw_planets = obs.get("planets", []) or []
            raw_fleets  = obs.get("fleets",  []) or []

            if s in train_seats:
                try:
                    world = build_world(obs)
                except Exception:
                    actions_all.append([])
                    continue

                step_dict = {
                    "step": step, "planets": raw_planets, "fleets": raw_fleets,
                    "action": [],
                    "my_total_ships": sum(p[5] for p in raw_planets if p[1] == s),
                    "enemy_total_ships": 0, "my_planet_count": 0,
                    "enemy_planet_count": 0, "neutral_planet_count": 0,
                }
                feat    = featurize_step(
                    step_dict, s, ang_vel, n_players, init_plan,
                    last_actions_by_planet=hist["last_actions_by_planet"],
                    cumulative_stats=hist["cum_stats"],
                    obs_history=list(hist["obs_history"]),
                    action_history=list(hist["action_history"]),
                )
                spatial = rasterize_obs(obs, s, grid=GRID)

                picks, smpl = rollout_step(
                    _worker_main, feat, spatial, world, s, raw_fleets, device="cpu",
                )
                action_list = materialize_with_targets(
                    [(p[0], p[1], p[2], p[6]) for p in picks], world, s,
                ) if picks else []
                actions_all.append(action_list)

                if smpl is not None:
                    seat_samples[s].append(smpl)
                    if s == pick_seat:
                        for p in picks:
                            mode_counts[p[1]] += 1
                            if p[1] != 0:
                                frac_counts[p[2]] += 1
            else:
                # Frozen / scripted opponent
                if opp_type == "noop":
                    action_list = []
                elif opp_type == "random":
                    action_list = _random_action(obs, s)
                elif opp_type in ("noisy_lb928", "noisy_lb1200"):
                    name = "lb928" if "928" in opp_type else "lb1200"
                    action_list = _noisy_lb(obs, s, name, task.get("noise_prob", 0.3))
                elif opp_type == "pool":
                    try:
                        world_opp = build_world(obs)
                        feat_opp  = _featurize_quick(obs, s, ang_vel, init_plan,
                                                     hist, n_players)
                        spat_opp  = rasterize_obs(obs, s, grid=GRID)
                        picks_opp, _ = rollout_step(
                            _worker_opp, feat_opp, spat_opp, world_opp,
                            s, raw_fleets, "cpu",
                        )
                        action_list = materialize_with_targets(
                            [(p[0], p[1], p[2], p[6]) for p in picks_opp],
                            world_opp, s,
                        ) if picks_opp else []
                    except Exception:
                        action_list = []
                else:
                    action_list = []
                actions_all.append(action_list)

            # update histories
            hist["obs_history"].append({"planets": raw_planets, "step": step})
            for mv in actions_all[-1]:
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
                hist["last_actions_by_planet"][src_id] = (tpid, bi, step, prev_[3] + 1)
                hist["cum_stats"]["total_ships_sent"] += ships
                hist["cum_stats"]["total_actions"]    += 1
                hist["action_history"].append((src_id, tpid, bi, step))

        env.step(actions_all)
        step += 1

        # PBRS reward per trainable seat
        for s in train_seats:
            if not seat_samples[s]:
                continue
            cur_obs = env.state[s].observation or {}
            cur_pl  = cur_obs.get("planets", []) or []
            cur_fl  = cur_obs.get("fleets",  []) or []
            tot_ships = (sum(p[5] for p in cur_pl) + sum(f[6] for f in cur_fl))
            my_prod   = sum(p[6] for p in cur_pl if p[1] == s)
            tot_prod  = max(1, sum(p[6] for p in cur_pl))
            prod_phi  = my_prod / tot_prod
            my_ships  = (sum(p[5] for p in cur_pl if p[1] == s)
                         + sum(f[6] for f in cur_fl if f[1] == s))
            ship_phi  = my_ships / max(1, tot_ships)
            en_ships  = (sum(p[5] for p in cur_pl if p[1] != s and p[1] != -1)
                         + sum(f[6] for f in cur_fl  if f[1] != s))
            en_phi    = en_ships / max(1, tot_ships)
            pp = prev_phi[s]
            r  = ((prod_phi - GAMMA * pp["prod_phi"])
                  + 0.5 * (ship_phi - GAMMA * pp["ship_phi"])
                  - 0.3 * (en_phi   - GAMMA * pp["enemy_ship_phi"]))
            seat_samples[s][-1]["reward"] = r
            prev_phi[s] = {"prod_phi": prod_phi, "ship_phi": ship_phi,
                           "enemy_ship_phi": en_phi}

    # Terminal reward
    for s in train_seats:
        if seat_samples[s]:
            seat_samples[s][-1]["reward"] += float(env.state[s].reward or 0)

    # GAE per trainable seat
    for s in train_seats:
        steps = seat_samples[s]
        if not steps:
            continue
        adv = 0.0
        for t in range(len(steps) - 1, -1, -1):
            next_v = steps[t + 1]["value"] if t + 1 < len(steps) else 0.0
            delta  = steps[t]["reward"] + GAMMA * next_v - steps[t]["value"]
            adv    = delta + GAMMA * GAE_LAMBDA * adv
            steps[t]["advantage"]    = adv
            steps[t]["value_target"] = adv + steps[t]["value"]

    wins = {s: float(env.state[s].reward or 0) > 0 for s in range(n_players)}
    return {
        "seat_samples": seat_samples,
        "train_seats":  list(train_seats),
        "pick_seat":    pick_seat,
        "wins":         wins,
        "opp":          opp_type,
        "mode_counts":  mode_counts,
        "frac_counts":  frac_counts,
    }


def _featurize_quick(obs, seat, ang_vel, init_plan, hist, n_players):
    """Minimal featurize for opponent pool agents."""
    raw_pl = obs.get("planets", []) or []
    raw_fl = obs.get("fleets",  []) or []
    step_dict = {
        "step": int(obs.get("step", 0) or 0),
        "planets": raw_pl, "fleets": raw_fl, "action": [],
        "my_total_ships": sum(p[5] for p in raw_pl if p[1] == seat),
        "enemy_total_ships": 0, "my_planet_count": 0,
        "enemy_planet_count": 0, "neutral_planet_count": 0,
    }
    return featurize_step(
        step_dict, seat, ang_vel, n_players, init_plan,
        last_actions_by_planet=hist["last_actions_by_planet"],
        cumulative_stats=hist["cum_stats"],
        obs_history=list(hist["obs_history"]),
        action_history=list(hist["action_history"]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Opponent pool
# ──────────────────────────────────────────────────────────────────────────────

class OpponentPool:
    def __init__(self, max_size: int = 20):
        self.snapshots: list[bytes] = []
        self.tags: list[str]        = []
        self.max_size = max_size

    def add(self, sd_bytes: bytes, tag: str):
        self.snapshots.append(sd_bytes)
        self.tags.append(tag)
        if len(self.snapshots) > self.max_size:
            drop = len(self.snapshots) // 2
            self.snapshots.pop(drop)
            self.tags.pop(drop)

    def __len__(self):
        return len(self.snapshots)

    def sample_index(self) -> int:
        n = len(self.snapshots)
        r = random.random()
        if r < 0.5 and n >= 2:
            return n - 1 - random.randrange(min(5, n))
        if r < 0.8 and n >= 3:
            lo = n // 3; hi = max(lo + 1, 2 * n // 3)
            return random.randrange(lo, hi)
        return random.randrange(n)


def _opponent_task(pool: OpponentPool, iter_: int,
                   lb_prob: float, lb_start: int,
                   noise_end: int, noise_start: float) -> dict:
    n = len(pool)
    noise = max(0.0, noise_start * (1.0 - (iter_ - lb_start) / max(1, noise_end - lb_start))) \
            if iter_ >= lb_start else 1.0

    if n < 2:
        return {"opp_type": "stochastic_self" if random.random() < 0.5 else "noop"}

    r = random.random()
    if iter_ >= lb_start and r < lb_prob:
        name = random.choice(["noisy_lb928", "noisy_lb1200"])
        return {"opp_type": name, "noise_prob": noise}

    rr = (r - lb_prob) / max(1e-6, 1.0 - lb_prob) if iter_ >= lb_start else r
    if rr < 0.15:
        return {"opp_type": "noop"}
    if rr < 0.30:
        return {"opp_type": "stochastic_self"}
    idx = pool.sample_index()
    return {"opp_type": "pool", "opp_weights_bytes": pool.snapshots[idx]}


# ──────────────────────────────────────────────────────────────────────────────
# Build TorchRL-compatible TensorDict from rollout samples
# ──────────────────────────────────────────────────────────────────────────────

def _samples_to_tensordict(samples: list[dict]) -> TensorDict:
    """Convert a list of per-step samples to a batch TensorDict.

    Each sample must already have 'advantage' and 'value_target' set.
    Returns TensorDict with batch_size=[T].
    """
    rows = []
    for s in samples:
        d = build_sample_td(s)
        d["advantage"]    = np.float32(s.get("advantage",    0.0))
        d["value_target"] = np.float32(s.get("value_target", s["value"]))
        rows.append(d)

    def stack_key(key):
        arrs = [r[key] for r in rows]
        arr  = np.stack(arrs, axis=0)
        return torch.from_numpy(arr)

    keys = list(rows[0].keys())
    td   = TensorDict(
        {k: stack_key(k) for k in keys},
        batch_size=[len(rows)],
    )
    return td


# ──────────────────────────────────────────────────────────────────────────────
# PPO update — TorchRL ReplayBuffer + SamplerWithoutReplacement
# ──────────────────────────────────────────────────────────────────────────────

def ppo_update(net, opt, all_samples: list[dict], device: str,
               ppo_epochs: int = 4, mini_batch: int = 256,
               clip: float = 0.2, val_clip: float = 0.2,
               val_coef: float = 0.5, ent_coef: float = 0.01):
    """Run PPO using TorchRL ReplayBuffer + SamplerWithoutReplacement.

    all_samples: flat list of per-step dicts (one game-step × one agent each).
    """
    if not all_samples:
        return {}

    # Build TensorDict batch
    full_td = _samples_to_tensordict(all_samples)
    T       = len(full_td)

    # Normalize advantages across the full collected batch
    adv_all = full_td["advantage"].float()
    adv_std = adv_all.std()
    if adv_std > 1e-6:
        full_td["advantage"] = (adv_all - adv_all.mean()) / (adv_std + 1e-8)

    # Build ReplayBuffer (pure on-policy — filled fresh each call)
    buf = ReplayBuffer(
        storage=LazyTensorStorage(max_size=T + 1),
        sampler=SamplerWithoutReplacement(drop_last=False),
        batch_size=min(mini_batch, T),
    )
    buf.extend(full_td)

    # SamplerWithoutReplacement cycles automatically after one full pass.
    # K epochs = K * ceil(T / batch_size) total .sample() calls.
    actual_bs   = min(mini_batch, T)
    n_per_epoch = max(1, math.ceil(T / actual_bs))
    total_steps = ppo_epochs * n_per_epoch

    info = {"pi_loss": 0.0, "v_loss": 0.0, "ent": 0.0,
            "ratio_mean": 0.0, "ratio_max": 0.0, "n": 0}

    for _ in range(total_steps):
        mini_td = buf.sample().to(device)

        log_prob_old = mini_td["log_prob_old"].float()
        value_old    = mini_td["value_old"].float()
        advantage    = mini_td["advantage"].float()
        value_target = mini_td["value_target"].float()

        log_prob_new, entropy, value_new = eval_batch(net, mini_td, device)

        ratio  = (log_prob_new - log_prob_old).exp()
        surr1  = ratio * advantage
        surr2  = ratio.clamp(1 - clip, 1 + clip) * advantage
        pi_loss = -torch.min(surr1, surr2).mean()

        v_clip  = value_old + (value_new - value_old).clamp(-val_clip, val_clip)
        v_loss  = torch.max(
            (value_new - value_target) ** 2,
            (v_clip    - value_target) ** 2,
        ).mean()

        loss = pi_loss + val_coef * v_loss - ent_coef * entropy.mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        info["pi_loss"]    += pi_loss.item()
        info["v_loss"]     += v_loss.item()
        info["ent"]        += entropy.mean().item()
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
    ap.add_argument("--warm-start", default=None,
                    help="k12/k13/k14 checkpoint to warm-start from")
    ap.add_argument("--workers",          type=int,   default=4)
    ap.add_argument("--target-iters",     type=int,   default=2000)
    ap.add_argument("--games-per-iter",   type=int,   default=16)
    ap.add_argument("--ppo-epochs",       type=int,   default=4)
    ap.add_argument("--mini-batch",       type=int,   default=256)
    ap.add_argument("--four-player-prob", type=float, default=0.2)
    ap.add_argument("--lr",               type=float, default=3e-4)
    ap.add_argument("--ent-coef-start",   type=float, default=0.05)
    ap.add_argument("--ent-coef-end",     type=float, default=0.01)
    ap.add_argument("--ent-decay-iters",  type=int,   default=100)
    ap.add_argument("--clip",             type=float, default=0.2)
    ap.add_argument("--val-coef",         type=float, default=0.5)
    ap.add_argument("--snapshot-every",   type=int,   default=5)
    ap.add_argument("--pool-size",        type=int,   default=20)
    ap.add_argument("--eval-every",       type=int,   default=20)
    ap.add_argument("--eval-games",       type=int,   default=20)
    ap.add_argument("--lb-prob",          type=float, default=0.2)
    ap.add_argument("--lb-start-iter",    type=int,   default=40)
    ap.add_argument("--noise-start-prob", type=float, default=0.60)
    ap.add_argument("--noise-end-iter",   type=int,   default=500)
    ap.add_argument("--start-iter",       type=int,   default=1)
    ap.add_argument("--out",              required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    ).to(device)

    if args.warm_start and Path(args.warm_start).exists():
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        net.load_state_dict(sd, strict=False)
        resume_iter = (ckpt.get("iter", args.start_iter - 1)
                       if isinstance(ckpt, dict) else args.start_iter - 1)
        if args.start_iter == 1:
            args.start_iter = resume_iter + 1
        print(f"[k14v2] loaded {args.warm_start} (iter {resume_iter})", flush=True)
    else:
        print(f"[k14v2] no warm-start", flush=True)
    init_head_biases(net)

    opt  = torch.optim.Adam(net.parameters(), lr=args.lr)
    pool = OpponentPool(max_size=args.pool_size)

    def sdb():
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
        return buf.getvalue()

    ctx         = mp.get_context("spawn")
    worker_pool = ctx.Pool(processes=args.workers, initializer=_worker_init,
                           initargs=(sdb(),))
    t0 = time.time()

    print(f"[k14v2] device={device}  "
          f"params={sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"games/iter={args.games_per_iter}  ppo_epochs={args.ppo_epochs}  "
          f"mini_batch={args.mini_batch}  lr={args.lr}  "
          f"ent={args.ent_coef_start}→{args.ent_coef_end}", flush=True)

    for iter_ in range(args.start_iter, args.start_iter + args.target_iters):
        is_eval_iter = (iter_ % args.eval_every == 0)

        # Refresh worker weights every 5 iters (skip if eval does it)
        if iter_ % 5 == 0 and not is_eval_iter:
            worker_pool.close(); worker_pool.join()
            worker_pool = ctx.Pool(processes=args.workers, initializer=_worker_init,
                                   initargs=(sdb(),))

        # Build game tasks
        tasks = []
        for _ in range(args.games_per_iter):
            n_p    = 4 if random.random() < args.four_player_prob else 2
            opp    = _opponent_task(pool, iter_, args.lb_prob, args.lb_start_iter,
                                    args.noise_end_iter, args.noise_start_prob)
            tasks.append({"n_players": n_p,
                          "picker_seat": random.randint(0, n_p - 1),
                          **opp})

        rollouts = worker_pool.map(_rollout_game, tasks)

        # Collect all training samples from ALL trainable agents
        all_samples: list[dict] = []
        wins = 0; total_games = 0
        opp_wins: dict = collections.defaultdict(lambda: [0, 0])
        mc_iter = [0] * N_MODES
        fc_iter = [0] * N_FRACS

        for r in rollouts:
            total_games += 1
            pick_seat = r["pick_seat"]
            if r["wins"].get(pick_seat, False):
                wins += 1
            opp_t = r.get("opp", "pool")
            opp_wins[opp_t][1] += 1
            if r["wins"].get(pick_seat, False):
                opp_wins[opp_t][0] += 1
            for ts in r["train_seats"]:
                all_samples.extend(r["seat_samples"][ts])
            for i, c in enumerate(r["mode_counts"]): mc_iter[i] += c
            for i, c in enumerate(r["frac_counts"]): fc_iter[i] += c

        # Entropy schedule
        alpha   = min(1.0, (iter_ - args.start_iter) / max(1, args.ent_decay_iters))
        ent_c   = args.ent_coef_start + alpha * (args.ent_coef_end - args.ent_coef_start)

        info = ppo_update(net, opt, all_samples, device,
                          ppo_epochs=args.ppo_epochs,
                          mini_batch=args.mini_batch,
                          clip=args.clip, val_coef=args.val_coef,
                          ent_coef=ent_c)

        # Pool snapshot
        if iter_ % args.snapshot_every == 0:
            pool.add(sdb(), f"iter{iter_:04d}")

        elapsed = time.time() - t0
        mc_str  = " ".join(str(c) for c in mc_iter)
        fc_str  = " ".join(str(c) for c in fc_iter)
        opp_parts = []
        for k, (w, g) in opp_wins.items():
            if g: opp_parts.append(f"{k}={w}/{g}")
        print(
            f"[iter {iter_:04d}]  "
            f"wins={wins}/{total_games}  ({' '.join(opp_parts)})  "
            f"T={info.get('T',0)}  "
            f"pi={info.get('pi_loss',0):.3f}  v={info.get('v_loss',0):.3f}  "
            f"ent={info.get('ent',0):.3f}  "
            f"r={info.get('ratio_mean',1):.2f}/{info.get('ratio_max',1):.1f}  "
            f"ent_c={ent_c:.3f}  pool={len(pool)}  "
            f"mc=[{mc_str}]  fc=[{fc_str}]  [{elapsed:.0f}s]",
            flush=True,
        )

        # Save checkpoint
        if iter_ % (args.snapshot_every * 2) == 0:
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, args.out)
            stem     = Path(args.out).stem
            versioned = Path(args.out).parent / f"{stem}_iter{iter_:04d}.pt"
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, versioned)
            print(f"[iter {iter_:04d}] saved {args.out} + {versioned.name}", flush=True)

        # Eval vs clean lb agents
        if is_eval_iter:
            worker_pool.close(); worker_pool.join()
            worker_pool = ctx.Pool(processes=args.workers, initializer=_worker_init,
                                   initargs=(sdb(),))
            eval_tasks = []
            for opp_name in ("lb928", "lb1200"):
                for _ in range(args.eval_games):
                    n_p = 2
                    eval_tasks.append({
                        "n_players": n_p,
                        "picker_seat": random.randint(0, n_p - 1),
                        "opp_type": f"noisy_{opp_name}",
                        "noise_prob": 0.0,
                    })
            eval_r = worker_pool.map(_rollout_game, eval_tasks)
            ew: dict = collections.defaultdict(lambda: [0, 0])
            for r in eval_r:
                ps  = r["pick_seat"]
                opp = r.get("opp", "")
                key = "lb928" if "928" in opp else "lb1200"
                ew[key][1] += 1
                if r["wins"].get(ps, False):
                    ew[key][0] += 1
            parts = [f"{k}={w}/{g}" for k, (w, g) in sorted(ew.items())]
            print(f"[iter {iter_:04d}] EVAL  {' '.join(parts)}", flush=True)

    worker_pool.close(); worker_pool.join()
    torch.save({"model": net.state_dict(), "iter": args.start_iter + args.target_iters - 1,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, args.out)
    print(f"[k14v2] final checkpoint saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
