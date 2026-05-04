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
# Reward-shaping constants (audit add-ons)
SUN_CENTER = 50.0
SUN_DEATH_RADIUS = 15.0          # fleets disappearing within this radius are sun-killed
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
COMET_WIN = {s + d for s in COMET_SPAWN_STEPS for d in range(-5, 6)}
COMET_PROD_MULT = 1.5            # multiplier on prod_phi delta during comet windows
SUN_PENALTY = 0.10               # weight on sun_lost / 100 term
PBRS_CLIP = 1.0                  # |r_step| cap, mirrors offline shaper

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
    prev_fleets = {eid: [] for eid in range(n_envs)}   # for sun-loss diff

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
            # Capture pre-step fleet state per eid for sun-loss diff later
            prev_fleets[eid] = list(obs.get("fleets") or [])
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
                                # vec_env's obs lacks `player` and `comet_planet_ids`;
                                # without `player` lb defaults to seat 0 and either
                                # crashes (wrong planet ownership) or no-ops, masking
                                # the real opponent. Patch the obs per-seat each call.
                                lb_obs = {
                                    **obs,
                                    "player": seat,
                                    "comet_planet_ids": obs.get("comet_planet_ids", []),
                                    "remainingOverageTime": obs.get("remainingOverageTime", 60.0),
                                }
                                action_list = _a(lb_obs) or []
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

        # PBRS per (eid, train_seat) — production/ship potentials + sun-loss
        # penalty + comet-window multiplier + per-step clip.
        in_comet = (steps_taken in COMET_WIN)
        for eid in range(n_envs):
            if env.done_mask[eid]:
                continue
            cur_obs = env.get_obs_dict(eid)
            cur_pl  = cur_obs["planets"]
            cur_fl  = cur_obs["fleets"]
            tot_sh  = sum(p[5] for p in cur_pl) + sum(f[6] for f in cur_fl)
            tot_pr  = max(1, sum(p[6] for p in cur_pl))
            cur_fids = {int(f[0]) for f in cur_fl}
            for seat in train_seats[eid]:
                if not seat_samples[(eid, seat)]:
                    continue
                my_pr   = sum(p[6] for p in cur_pl if p[1] == seat)
                my_sh   = (sum(p[5] for p in cur_pl if p[1] == seat)
                           + sum(f[6] for f in cur_fl if f[1] == seat))
                en_sh   = (sum(p[5] for p in cur_pl if p[1] != seat and p[1] != -1)
                           + sum(f[6] for f in cur_fl  if f[1] != seat))
                prod_phi = my_pr / tot_pr
                ship_phi = my_sh / max(1, tot_sh)
                en_phi   = en_sh / max(1, tot_sh)
                pp = prev_phi[(eid, seat)]

                # Sun-loss: this seat's fleets present last step, gone now,
                # and last seen near the sun → counted as sun-killed.
                sun_lost = 0
                for f in prev_fleets[eid]:
                    if int(f[1]) != seat or int(f[0]) in cur_fids:
                        continue
                    if math.hypot(f[2] - SUN_CENTER, f[3] - SUN_CENTER) < SUN_DEATH_RADIUS:
                        sun_lost += int(f[6])

                comet_mult = COMET_PROD_MULT if in_comet else 1.0
                r = (comet_mult * (prod_phi - GAMMA * pp["prod_phi"])
                     + 0.5 * (ship_phi - GAMMA * pp["ship_phi"])
                     - 0.3 * (en_phi   - GAMMA * pp["enemy_ship_phi"])
                     - SUN_PENALTY * sun_lost / 100.0)
                r = max(-PBRS_CLIP, min(PBRS_CLIP, r))
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
        "pool_idx":     task.get("pool_idx", -1),
        "mode_counts":  mode_counts,
        "frac_counts":  frac_counts,
        "n_envs":       n_envs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Opponent pool
# ──────────────────────────────────────────────────────────────────────────────
class OpponentPool:
    """League pool with PFSP sampling, adaptive eviction, and frozen-anchor
    ELO bookkeeping (per AlphaStar / CLAUDE-2.md `simple_rl_v2`).

    Per-snapshot win-rate (EMA) drives two policies:
      sampling — pick HARD opponents more often: w_i ∝ (1 − win_rate_i)^p + floor
      eviction — drop the EASIEST mature snapshot when pool is full

    Frozen-anchor ELO: each pool entry stores `snapshot_elo[i]` = the
    learner's ELO at the moment that snapshot was frozen. Anchor ELOs
    NEVER update. The learner has a single `learner_elo` (default 1500)
    that updates via standard Elo (K factor `elo_k`) on every game played
    against a pool entry. Because snapshot ELOs are immutable and new
    snapshots are added at the learner's current ELO, this produces the
    monotonically-rising "purple→cyan" curve described in the AlphaStar
    league/elo_learner plot — provided the learner is genuinely improving.

    `record_result(idx, won)` must be called for every game played against
    a pool snapshot; idx is the value returned by the matching
    `sample_index()`. It updates BOTH the EMA win-rate (for PFSP +
    eviction) AND the learner's ELO (against the snapshot's frozen ELO).

    Indices are stable within an iteration (no add() runs while rollouts
    are in flight), so the round-trip task → result is safe.
    """

    def __init__(self, max_size: int = 20, pfsp_p: float = 2.0,
                 pfsp_floor: float = 0.05, ema_alpha: float = 0.15,
                 elo_init: float = 1500.0, elo_k: float = 16.0):
        self.snapshots: list[bytes] = []
        self.tags:      list[str]   = []
        self.win_ema:   list[float] = []   # EMA of P(we beat snapshot i), in [0, 1]
        self.games:     list[int]   = []   # raw count for confidence / eviction grace
        self.snapshot_elo: list[float] = []  # FROZEN at add() time, never updates
        self.max_size  = max_size
        self.pfsp_p    = pfsp_p
        self.pfsp_floor = pfsp_floor
        self.ema_alpha = ema_alpha
        # Learner state — single mutable ELO updated only by record_result()
        self.learner_elo = float(elo_init)
        self.elo_k       = float(elo_k)
        self.elo_games   = 0   # total games used for ELO updates (any anchor)

    def add(self, sd_bytes: bytes, tag: str) -> None:
        """Snapshot current learner with its current ELO (frozen forever)."""
        self.snapshots.append(sd_bytes)
        self.tags.append(tag)
        self.win_ema.append(0.5)        # neutral prior — sampled often until proven easy
        self.games.append(0)
        self.snapshot_elo.append(self.learner_elo)
        if len(self.snapshots) > self.max_size:
            # Drop the snapshot we've beaten most (least informative training signal).
            # Protect entries with < min_games games so fresh additions get a chance.
            min_games = 3
            mature = [i for i, g in enumerate(self.games) if g >= min_games]
            drop = (max(mature, key=lambda i: self.win_ema[i])
                    if mature else 0)   # FIFO fallback when nothing mature
            for arr in (self.snapshots, self.tags, self.win_ema, self.games,
                        self.snapshot_elo):
                arr.pop(drop)

    def __len__(self) -> int:
        return len(self.snapshots)

    def sample_index(self) -> int:
        n = len(self.snapshots)
        if n == 0:
            return -1
        if n == 1:
            return 0
        weights = [
            (1.0 - min(0.99, self.win_ema[i])) ** self.pfsp_p + self.pfsp_floor
            for i in range(n)
        ]
        return random.choices(range(n), weights=weights, k=1)[0]

    def record_result(self, idx: int, won: bool) -> None:
        """Record one game outcome against snapshot `idx`.

        Updates:
          - PFSP/eviction state: win_ema, games count
          - ELO: learner_elo only (snapshot's elo is frozen)
        """
        if not (0 <= idx < len(self.snapshots)):
            return
        x = 1.0 if won else 0.0
        # PFSP + eviction tracking
        self.win_ema[idx] = (1.0 - self.ema_alpha) * self.win_ema[idx] + self.ema_alpha * x
        self.games[idx] += 1
        # Frozen-anchor Elo update — only learner_elo moves
        opp_elo = self.snapshot_elo[idx]
        expected = 1.0 / (1.0 + 10.0 ** ((opp_elo - self.learner_elo) / 400.0))
        self.learner_elo += self.elo_k * (x - expected)
        self.elo_games += 1

    def stats(self, top_k: int = 3) -> str:
        """Short summary of the K hardest snapshots (lowest win-rate against them)."""
        if not self.snapshots:
            return ""
        order = sorted(range(len(self.snapshots)), key=lambda i: self.win_ema[i])
        return " ".join(
            f"{self.tags[i]}={self.win_ema[i]:.2f}({self.games[i]})"
            for i in order[:top_k]
        )

    def elo_summary(self) -> str:
        """Compact ELO line: learner ELO + range across pool snapshots."""
        if not self.snapshot_elo:
            return f"learner={self.learner_elo:.0f}"
        lo = min(self.snapshot_elo)
        hi = max(self.snapshot_elo)
        return (f"learner={self.learner_elo:.0f} "
                f"pool_elo=[{lo:.0f}..{hi:.0f}] elo_games={self.elo_games}")


def _current_noise(iter_, lb_start, noise_end, noise_start) -> float:
    """Same schedule as _opp_task — exposed for logging."""
    if iter_ < lb_start:
        return 1.0
    return max(0.0, noise_start * (1.0 - (iter_ - lb_start) / max(1, noise_end - lb_start)))


def _opp_task(pool, iter_, lb_prob, lb_start, noise_end, noise_start):
    n = len(pool)
    noise = _current_noise(iter_, lb_start, noise_end, noise_start)
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
    pool_idx = pool.sample_index()
    return {"opp_type": "pool",
            "opp_weights_bytes": pool.snapshots[pool_idx],
            "pool_idx": pool_idx}


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
               val_clip=0.2, val_coef=0.5, ent_coef=0.01,
               bc_loader=None, bc_weight=0.0):
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
            "ratio_mean": 0.0, "ratio_max": 0.0, "n": 0,
            "bc_mode_ce": 0.0, "bc_frac_ce": 0.0,
            "bc_mode_acc": 0.0, "bc_frac_acc": 0.0, "bc_n": 0}

    use_bc = bc_loader is not None and bc_weight > 0.0
    if use_bc:
        from training.bc_aux_loss import compute_bc_loss

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

        if use_bc:
            bc_batch = bc_loader.sample()
            bc_loss, bc_info = compute_bc_loss(net, bc_batch)
            loss = loss + bc_weight * bc_loss
            info["bc_mode_ce"]  += bc_info["mode_ce"]
            info["bc_frac_ce"]  += bc_info["frac_ce"]
            info["bc_mode_acc"] += bc_info["mode_acc"]
            info["bc_frac_acc"] += bc_info["frac_acc"]
            info["bc_n"]        += 1

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
    if info["bc_n"] > 0:
        nbc = info["bc_n"]
        for k in ("bc_mode_ce", "bc_frac_ce", "bc_mode_acc", "bc_frac_acc"):
            info[k] /= nbc
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
    # Tom rl10 Fix #8: positive entropy bonus + WR<50% causes entropy creep
    # (bad decisions → more losses → more entropy → worse decisions). Default
    # is neutral start, small penalty end. Pass --ent-coef-start 0.05 to
    # restore the old "encourage exploration" behaviour.
    ap.add_argument("--ent-coef-start",     type=float, default=0.0)
    ap.add_argument("--ent-coef-end",       type=float, default=-0.005)
    ap.add_argument("--ent-decay-iters",    type=int, default=100)
    ap.add_argument("--clip",               type=float, default=0.2)
    ap.add_argument("--val-coef",           type=float, default=0.5)
    ap.add_argument("--snapshot-every",     type=int, default=5)
    ap.add_argument("--pool-size",          type=int, default=20)
    ap.add_argument("--pfsp-p",             type=float, default=2.0,
                    help="PFSP exponent: w_i ∝ (1 - winrate_i)^p. Higher = harder bias.")
    ap.add_argument("--pfsp-floor",         type=float, default=0.05,
                    help="Min sampling weight per snapshot (prevents starvation).")
    ap.add_argument("--pfsp-ema-alpha",     type=float, default=0.15,
                    help="EMA mixing for win-rate updates (~ window of 1/alpha games).")
    ap.add_argument("--eval-every",         type=int, default=10)
    ap.add_argument("--eval-games",         type=int, default=20)
    ap.add_argument("--lb-prob",            type=float, default=0.2)
    ap.add_argument("--lb-start-iter",      type=int, default=40)
    ap.add_argument("--noise-start-prob",   type=float, default=0.60)
    ap.add_argument("--noise-end-iter",     type=int, default=500)
    ap.add_argument("--four-player-prob",   type=float, default=0.2)
    ap.add_argument("--start-iter",         type=int, default=1)
    ap.add_argument("--out",                required=True)
    ap.add_argument("--bc-data-dir",        default=None,
                    help="Directory of bc_*.npz shards from build_bc_dataset.py. "
                         "If set, BC aux loss runs alongside PPO each minibatch.")
    ap.add_argument("--bc-weight",          type=float, default=0.1,
                    help="Multiplier on BC loss in total = pi + val*v - ent + bc_weight*bc.")
    ap.add_argument("--bc-batch-size",      type=int, default=64)
    ap.add_argument("--bc-winners-only",    action="store_true",
                    help="Restrict BC supervision to winning expert seats.")
    # Frozen-anchor ELO (CLAUDE-2.md / simple_rl_v2 league mechanism)
    ap.add_argument("--elo-init",           type=float, default=1500.0)
    ap.add_argument("--elo-k",              type=float, default=16.0)
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
    pool = OpponentPool(max_size=args.pool_size,
                        pfsp_p=args.pfsp_p,
                        pfsp_floor=args.pfsp_floor,
                        ema_alpha=args.pfsp_ema_alpha,
                        elo_init=args.elo_init,
                        elo_k=args.elo_k)

    bc_loader = None
    if args.bc_data_dir:
        from training.bc_aux_loss import BCLoader
        bc_loader = BCLoader(
            args.bc_data_dir,
            batch_size=args.bc_batch_size,
            device=device,
            winners_only=args.bc_winners_only,
            seed=42,
        )
        print(f"[k14-vec] BC aux: data={args.bc_data_dir}  "
              f"weight={args.bc_weight}  batch={args.bc_batch_size}  "
              f"winners_only={args.bc_winners_only}", flush=True)

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

        # PFSP bookkeeping: record win/loss against the sampled pool snapshot.
        # Indices are stable here — pool.add() runs later in the iter, so
        # pool_idx values returned by workers still point at the same entry.
        for r in rollouts:
            if r.get("opp") != "pool":
                continue
            pidx = r.get("pool_idx", -1)
            if pidx < 0:
                continue
            for eid in range(r["n_envs"]):
                w = r["wins"].get(eid, -1)
                if w < 0:
                    continue                       # game didn't finish in step budget
                pool.record_result(pidx, w == r["pick_seats"][eid])

        alpha = min(1.0, (iter_ - args.start_iter) / max(1, args.ent_decay_iters))
        ent_c = args.ent_coef_start + alpha * (args.ent_coef_end - args.ent_coef_start)
        info = ppo_update(net, opt, all_samples, device,
                          ppo_epochs=args.ppo_epochs, mini_batch=args.mini_batch,
                          clip=args.clip, val_coef=args.val_coef, ent_coef=ent_c,
                          bc_loader=bc_loader, bc_weight=args.bc_weight)

        if iter_ % args.snapshot_every == 0:
            pool.add(sdb(), f"iter{iter_:04d}")

        elapsed = time.time() - t0
        opp_str = "  ".join(f"{k}={v[0]}/{v[1]}" for k, v in ow.items() if v[1])
        mc_str  = " ".join(str(c) for c in mc)
        fc_str  = " ".join(str(c) for c in fc)
        noise_now = _current_noise(iter_, args.lb_start_iter,
                                   args.noise_end_iter, args.noise_start_prob)
        pfsp_str = pool.stats(top_k=3)
        bc_str = ""
        if info.get("bc_n", 0) > 0:
            bc_str = (f"  bc_m={info['bc_mode_ce']:.2f}/{info['bc_mode_acc']:.2f}"
                      f"  bc_f={info['bc_frac_ce']:.2f}/{info['bc_frac_acc']:.2f}")
        # Frozen-anchor ELO: learner ELO + games used for updates this run
        elo_str = (f"  learner_elo={pool.learner_elo:.1f}"
                   f"  elo_games={pool.elo_games}")
        print(
            f"[iter {iter_:05d}]  "
            f"wins={wins}/{tg}  ({opp_str})  "
            f"T={info.get('T',0)}  "
            f"pi={info.get('pi_loss',0):.3f}  v={info.get('v_loss',0):.3f}  "
            f"ent={info.get('ent',0):.3f}  "
            f"r={info.get('ratio_mean',1):.2f}/{info.get('ratio_max',1):.1f}  "
            f"ent_c={ent_c:.3f}  pool={len(pool)}  "
            f"hardest=[{pfsp_str}]  "
            f"lb_noise={noise_now:.2f}  "
            f"mc=[{mc_str}]  fc=[{fc_str}]{bc_str}{elo_str}  [{elapsed:.0f}s]",
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
            print(f"[iter {iter_:05d}] EVAL  {' '.join(parts)}  "
                  f"(eval uses clean lb agents, noise=0.00)", flush=True)

    wp.close(); wp.join()
    torch.save({"model": net.state_dict(),
                "iter": args.start_iter + args.target_iters - 1,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, args.out)
    print(f"[k14-vec] final checkpoint saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
