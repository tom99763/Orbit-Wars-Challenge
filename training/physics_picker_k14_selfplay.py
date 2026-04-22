"""k14 — pure self-play league training on top of k13 architecture.

Design shift vs k13:
  - NO more fixed lb928 / lb1200 opponents (strong-opponent trap).
  - Build opponent pool from self-snapshots + noop + random baselines.
  - Curriculum emerges automatically: early snapshots are weak, grow with main.
  - 50% recent / 30% mid / 20% old+weak sampling.
  - Entropy annealing 0.05 → 0.02 → 0.01 over training.
  - Games/iter bumped to 16.

Reuses k13 building blocks:
  - DualStreamK13Agent (model with factored mode + conditional frac)
  - physics_action_helper_k13 (mode mask + fraction-based action)
  - PBRS reward (production fraction primary + ship fraction auxiliary)

Usage:
  python training/physics_picker_k14_selfplay.py \\
      --warm-start-k12 training/checkpoints/physics_picker_k12_v3.pt \\
      --workers 4 --games-per-iter 16 --snapshot-every 5 \\
      --out training/checkpoints/physics_picker_k14.pt
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
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from featurize import (featurize_step, PLANET_DIM, FLEET_DIM, GLOBAL_DIM, HISTORY_K)
from training.dual_stream_model import rasterize_obs, N_SPATIAL_CHANNELS
from training.lb1200_agent import build_world
from training.physics_action_helper_k13 import (
    N_MODES, N_FRACS, MODE_NAMES, FRACTIONS,
    materialize_joint_action, materialize_with_targets, compute_mode_mask,
    get_top_k_candidates, TOP_K_TARGETS, CAND_FEAT_DIM,
)
from training.physics_picker_k13_ppo import (
    DualStreamK13Agent, _collate_batch, ppo_update,
    load_k12_into_k13, init_head_biases,
    GRID, GAMMA,
)


# -----------------------------------------------------------------------------
# Worker — supports main policy + NN opponent policy + scripted baselines
# -----------------------------------------------------------------------------

_worker_main: Optional[DualStreamK13Agent] = None
_worker_opp: Optional[DualStreamK13Agent] = None


def _worker_init(main_sd_bytes: bytes):
    global _worker_main, _worker_opp
    import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""  # workers always on CPU
    torch.set_num_threads(2)   # limit per-worker threads to reduce CPU contention
    from kaggle_environments import make   # noqa
    _worker_main = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    )
    buf = io.BytesIO(main_sd_bytes)
    sd = torch.load(buf, map_location="cpu", weights_only=False)
    _worker_main.load_state_dict(sd)
    _worker_main.eval()
    _worker_opp = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    )
    _worker_opp.eval()


def _random_action(obs: dict, my_player: int) -> list:
    """Baseline: send a random fraction from a random owned planet to a random other."""
    planets = obs.get("planets", []) or []
    mine = [p for p in planets if p[1] == my_player]
    if not mine:
        return []
    src = random.choice(mine)
    if src[5] < 3:
        return []
    others = [p for p in planets if p[0] != src[0]]
    if not others:
        return []
    tgt = random.choice(others)
    dx, dy = tgt[2] - src[2], tgt[3] - src[3]
    ang = math.atan2(dy, dx)
    ships = max(1, int(src[5] * random.uniform(0.2, 0.8)))
    return [[int(src[0]), float(ang), int(ships)]]


def _noisy_lb_action(obs: dict, my_player: int, agent_name: str, noise_prob: float) -> list:
    """lb1200 / lb928 with random action injection — handicap for late-stage training."""
    if random.random() < noise_prob:
        # crippled step: noise_prob fraction of turns, replace with random action
        return _random_action(obs, my_player)
    if agent_name == "lb928":
        from training.lb928_agent import agent as _a
        return _a(obs) or []
    else:
        from training.lb1200_agent import agent as _a
        return _a(obs) or []


def _nn_sample_action(net: DualStreamK13Agent, obs: dict, my_player: int,
                      n_players: int, ang_vel_init: float, init_planets: list,
                      last_actions_by_planet: dict, cum_stats: dict,
                      obs_history: list, action_history: list,
                      want_logprob: bool = False) -> tuple:
    """Sample action from an NN policy. Returns (action_list, maybe sample_dict)."""
    try:
        world = build_world(obs)
    except Exception:
        return [], None
    raw_planets = obs.get("planets", []) or []
    raw_fleets  = obs.get("fleets",  []) or []
    my_list = [p for p in world.planets if p.owner == my_player]
    if not my_list:
        return [], None

    step_dict = {
        "step": int(obs.get("step", 0) or 0),
        "planets": raw_planets, "fleets": raw_fleets,
        "action": [],
        "my_total_ships": sum(p[5] for p in raw_planets if p[1] == my_player),
        "enemy_total_ships": 0, "my_planet_count": 0,
        "enemy_planet_count": 0, "neutral_planet_count": 0,
    }
    feat = featurize_step(step_dict, my_player, ang_vel_init, n_players, init_planets,
                          last_actions_by_planet=last_actions_by_planet,
                          cumulative_stats=cum_stats,
                          obs_history=obs_history, action_history=action_history)
    spatial = rasterize_obs(obs, my_player, grid=GRID)

    pl = feat["planets"]; fl = feat["fleets"]
    if pl.shape[0] == 0:
        pl = np.zeros((1, PLANET_DIM), dtype=np.float32); pmask = np.zeros(1, dtype=bool)
    else:
        pmask = np.ones(pl.shape[0], dtype=bool)
    if fl.ndim < 2 or fl.shape[0] == 0:
        fl = np.zeros((1, FLEET_DIM), dtype=np.float32); fmask = np.zeros(1, dtype=bool)
    else:
        fmask = np.ones(fl.shape[0], dtype=bool)

    with torch.no_grad():
        fused_tokens, mode_logits, v = net(
            torch.from_numpy(pl).unsqueeze(0),
            torch.from_numpy(pmask).unsqueeze(0),
            torch.from_numpy(fl).unsqueeze(0),
            torch.from_numpy(fmask).unsqueeze(0),
            torch.from_numpy(feat["globals"]).unsqueeze(0),
            torch.from_numpy(spatial).unsqueeze(0),
        )
    ml_np = mode_logits[0].cpu().numpy()
    planet_ids_in_feat = feat.get("planet_ids", [])

    picks = []
    committed = {}  # target_pid -> ships already dispatched this step (coordination)
    log_prob_sum = 0.0
    ent_sum = 0.0; ent_ct = 0
    for i, pid in enumerate(planet_ids_in_feat):
        src = next((pl_ for pl_ in world.planets if pl_.id == int(pid)), None)
        if src is None or src.owner != my_player: continue
        mask = compute_mode_mask(src, world, my_player)
        m_log = ml_np[i].copy()
        for k in range(N_MODES):
            if not mask[k]: m_log[k] = -1e9
        m_p = np.exp(m_log - m_log.max()); m_p = m_p / m_p.sum()
        mode_idx = int(np.random.choice(N_MODES, p=m_p))
        if want_logprob:
            log_prob_sum += float(np.log(m_p[mode_idx] + 1e-9))
            ent_sum += -float((m_p * np.log(m_p + 1e-9)).sum()); ent_ct += 1
        if mode_idx == 0:
            picks.append((int(pid), 0, 0, np.zeros((TOP_K_TARGETS, CAND_FEAT_DIM), dtype=np.float32), 0, 1, -1, tuple(mask)))
            continue

        # Top-K target selection (with fleet race + coordination features)
        cands, cand_feats, n_valid = get_top_k_candidates(
            src, world, my_player, mode_idx,
            fleets_raw=raw_fleets, committed=committed,
        )
        if n_valid == 0:
            picks.append((int(pid), 0, 0, np.zeros((TOP_K_TARGETS, CAND_FEAT_DIM), dtype=np.float32), 0, 1, -1, tuple(mask)))
            continue
        cand_feats_t = torch.from_numpy(cand_feats).unsqueeze(0).to(fused_tokens.device)  # (1,K,d)
        with torch.no_grad():
            t_scores = net.target_logits_for(fused_tokens[0, i].unsqueeze(0), cand_feats_t)[0].cpu().numpy()  # (K,)
        t_scores[n_valid:] = -1e9
        t_p = np.exp(t_scores - t_scores[:n_valid].max()); t_p[n_valid:] = 0.0; t_p = t_p / t_p.sum()
        tgt_idx = int(np.random.choice(TOP_K_TARGETS, p=t_p))
        target_pid = cands[tgt_idx].id
        if want_logprob:
            log_prob_sum += float(np.log(t_p[tgt_idx] + 1e-9))
            ent_sum += -float((t_p[:n_valid] * np.log(t_p[:n_valid] + 1e-9)).sum()); ent_ct += 1

        with torch.no_grad():
            fl_cond = net.frac_logits_for(
                fused_tokens[0, i],
                torch.tensor(mode_idx, dtype=torch.long),
            ).cpu().numpy()
        f_p = np.exp(fl_cond - fl_cond.max()); f_p = f_p / f_p.sum()
        frac_idx = int(np.random.choice(N_FRACS, p=f_p))
        if want_logprob:
            log_prob_sum += float(np.log(f_p[frac_idx] + 1e-9))
            ent_sum += -float((f_p * np.log(f_p + 1e-9)).sum()); ent_ct += 1

        # Update committed fleet budget for subsequent planet decisions
        ships_sent = max(1, int(src.ships * FRACTIONS[frac_idx]))
        committed[target_pid] = committed.get(target_pid, 0) + ships_sent

        # (pid, mode_idx, frac_idx, cand_feats, tgt_idx, n_valid, target_pid, mask)
        picks.append((int(pid), mode_idx, frac_idx, cand_feats, tgt_idx, n_valid, target_pid, tuple(mask)))

    action_list = materialize_with_targets(
        [(p[0], p[1], p[2], p[6]) for p in picks], world, my_player)

    sample_dict = None
    if want_logprob and picks:
        # Normalize log_prob by number of decisions (1 per pass pick, 2 per non-pass pick)
        # to prevent trajectory-level ratio from exploding with many planets
        n_decisions = sum(3 if p[1] != 0 else 1 for p in picks)
        sample_dict = {
            "feat": feat, "spatial": spatial,
            "planet_ids": planet_ids_in_feat,
            "picks": picks,
            "log_prob_sum": log_prob_sum / max(1, n_decisions),
            "entropy": ent_sum / max(1, ent_ct),
            "value": float(v.item()),
            "reward": 0.0,
            "my_player": my_player,
        }
    return action_list, sample_dict


def _rollout_game(task: dict) -> dict:
    """Play one game. task:
      - n_players, picker_seat
      - opp_type: "noop" / "random" / "pool"
      - opp_weights_bytes: state_dict bytes (only when opp_type == "pool")
    """
    from kaggle_environments import make

    global _worker_opp
    opp_type = task["opp_type"]
    if opp_type == "pool":
        buf = io.BytesIO(task["opp_weights_bytes"])
        sd = torch.load(buf, map_location="cpu", weights_only=False)
        _worker_opp.load_state_dict(sd)
        _worker_opp.eval()

    n_players = task["n_players"]
    picker_seat = task["picker_seat"]
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)

    # per-seat histories (main gets full histories; opponents get minimal)
    histories = {
        seat: {
            "obs_history": collections.deque(maxlen=HISTORY_K),
            "action_history": collections.deque(maxlen=HISTORY_K),
            "last_actions_by_planet": {},
            "cum_stats": {"total_ships_sent": 0, "total_actions": 0},
        }
        for seat in range(n_players)
    }

    samples = []
    ang_vel_init = None; init_planets = None
    step = 0
    prev_prod_phi = 0.0; prev_ship_phi = 0.0; prev_enemy_ship_phi = 0.0
    mode_counts = [0] * N_MODES
    frac_counts = [0] * N_FRACS

    while not env.done and step < 500:
        actions_all = []
        for s in range(n_players):
            obs = env.state[s].observation
            if obs is None:
                actions_all.append([]); continue
            if ang_vel_init is None:
                ang_vel_init = float(obs.get("angular_velocity", 0.0) or 0.0)
                init_planets = obs.get("initial_planets", []) or []

            hist = histories[s]
            if s == picker_seat:
                action_list, smpl = _nn_sample_action(
                    _worker_main, obs, s, n_players, ang_vel_init, init_planets,
                    hist["last_actions_by_planet"], hist["cum_stats"],
                    list(hist["obs_history"]), list(hist["action_history"]),
                    want_logprob=True,
                )
                actions_all.append(action_list)
                if smpl is not None:
                    samples.append(smpl)
                    for p in smpl["picks"]:
                        mi, fi = p[1], p[2]
                        mode_counts[mi] += 1
                        if mi != 0: frac_counts[fi] += 1
            else:
                if opp_type == "noop":
                    action_list = []
                elif opp_type == "random":
                    action_list = _random_action(obs, s)
                elif opp_type in ("noisy_lb928", "noisy_lb1200"):
                    name = "lb928" if opp_type == "noisy_lb928" else "lb1200"
                    action_list = _noisy_lb_action(obs, s, name,
                                                    task.get("noise_prob", 0.3))
                elif opp_type == "stochastic_self":
                    action_list, _ = _nn_sample_action(
                        _worker_main, obs, s, n_players, ang_vel_init, init_planets,
                        hist["last_actions_by_planet"], hist["cum_stats"],
                        list(hist["obs_history"]), list(hist["action_history"]),
                        want_logprob=False,
                    )
                else:
                    action_list, _ = _nn_sample_action(
                        _worker_opp, obs, s, n_players, ang_vel_init, init_planets,
                        hist["last_actions_by_planet"], hist["cum_stats"],
                        list(hist["obs_history"]), list(hist["action_history"]),
                        want_logprob=False,
                    )
                actions_all.append(action_list)

            # update histories
            raw_planets = obs.get("planets", []) or []
            hist["obs_history"].append({"planets": raw_planets, "step": step})
            for mv in actions_all[-1]:
                if len(mv) != 3: continue
                from featurize import nearest_target_index, ship_bucket_idx
                src_id, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
                src_p = next((p for p in raw_planets if int(p[0]) == src_id), None)
                if src_p is None: continue
                ti = nearest_target_index(src_p, ang, raw_planets)
                tpid = int(raw_planets[ti][0]) if ti is not None else -1
                garrison = int(src_p[5]) + ships
                bi = ship_bucket_idx(ships, max(1, garrison))
                prev_ = hist["last_actions_by_planet"].get(src_id, (-1, 0, -1, 0))
                hist["last_actions_by_planet"][src_id] = (tpid, bi, step, prev_[3] + 1)
                hist["cum_stats"]["total_ships_sent"] += ships
                hist["cum_stats"]["total_actions"] += 1
                hist["action_history"].append((src_id, tpid, bi, step))

        env.step(actions_all)
        step += 1

        # PBRS reward for picker
        if samples:
            cur_obs = env.state[picker_seat].observation or {}
            cur_planets = cur_obs.get("planets", []) or []
            cur_fleets  = cur_obs.get("fleets",  []) or []
            my_prod  = sum(p[6] for p in cur_planets if p[1] == picker_seat)
            tot_prod = sum(p[6] for p in cur_planets)
            prod_phi = my_prod / max(1, tot_prod)
            my_ships   = sum(p[5] for p in cur_planets if p[1] == picker_seat)
            my_ships  += sum(f[6] for f in cur_fleets   if f[1] == picker_seat)
            tot_ships  = sum(p[5] for p in cur_planets) + sum(f[6] for f in cur_fleets)
            ship_phi   = my_ships / max(1, tot_ships)
            enemy_ships  = sum(p[5] for p in cur_planets if p[1] != picker_seat and p[1] != -1)
            enemy_ships += sum(f[6] for f in cur_fleets   if f[1] != picker_seat)
            enemy_ship_phi = enemy_ships / max(1, tot_ships)
            r = ((prod_phi - GAMMA * prev_prod_phi)
                 + 0.5 * (ship_phi       - GAMMA * prev_ship_phi)
                 - 0.3 * (enemy_ship_phi - GAMMA * prev_enemy_ship_phi))
            samples[-1]["reward"] = r
            prev_prod_phi      = prod_phi
            prev_ship_phi      = ship_phi
            prev_enemy_ship_phi = enemy_ship_phi

    if samples:
        terminal = float(env.state[picker_seat].reward or 0)
        samples[-1]["reward"] += terminal

    # GAE(λ=0.95) — lower variance than MC, better credit assignment
    GAE_LAMBDA = 0.95
    adv = 0.0
    for t in range(len(samples) - 1, -1, -1):
        next_v = samples[t+1]["value"] if t + 1 < len(samples) else 0.0
        delta = samples[t]["reward"] + GAMMA * next_v - samples[t]["value"]
        adv = delta + GAMMA * GAE_LAMBDA * adv
        samples[t]["gae_adv"] = adv
        samples[t]["mc_return"] = adv + samples[t]["value"]   # V-target

    return {"samples": samples, "picker_seat": picker_seat,
            "win": float(env.state[picker_seat].reward or 0) > 0,
            "opp": opp_type,
            "mode_counts": mode_counts,
            "frac_counts": frac_counts}


# -----------------------------------------------------------------------------
# Opponent pool
# -----------------------------------------------------------------------------

class OpponentPool:
    def __init__(self, max_size: int = 20):
        self.snapshots: list[bytes] = []
        self.tags: list[str] = []       # e.g. "iter0005"
        self.max_size = max_size

    def add(self, sd_bytes: bytes, tag: str):
        self.snapshots.append(sd_bytes)
        self.tags.append(tag)
        if len(self.snapshots) > self.max_size:
            # drop middle (keep oldest for diversity + recent for difficulty)
            drop = len(self.snapshots) // 2
            self.snapshots.pop(drop)
            self.tags.pop(drop)

    def __len__(self):
        return len(self.snapshots)

    def sample_index(self) -> int:
        n = len(self.snapshots)
        r = random.random()
        if r < 0.5 and n >= 2:
            # recent: last min(5, n)
            k = min(5, n)
            return n - 1 - random.randrange(k)
        elif r < 0.8 and n >= 3:
            # mid
            lo = n // 3; hi = max(lo + 1, 2 * n // 3)
            return random.randrange(lo, hi)
        else:
            return random.randrange(n)


def _lb_noise_prob(iter_: int, lb_start: int = 40, noise_end: int = 500,
                   noise_start: float = 0.60) -> float:
    """Linear decay: noise_start at lb_start → 0.0 at noise_end."""
    if iter_ < lb_start:
        return 1.0  # not active yet
    if iter_ >= noise_end:
        return 0.0
    return noise_start * (1.0 - (iter_ - lb_start) / (noise_end - lb_start))


def _opponent_task(pool: OpponentPool, iter_: int,
                   lb_prob: float, lb_start: int = 40,
                   noise_end: int = 500, noise_start: float = 0.60) -> dict:
    """Return opp spec for one game.

    Schedule:
      pool < 2 (bootstrap): 50% stochastic_self, 50% noop
      pool >= 2, iter < lb_start: 15% noop, 15% stochastic_self, 70% self-pool
      pool >= 2, iter >= lb_start: lb_prob noisy-lb (noise decays 60%→0% by iter noise_end)
                                   + remaining: 15% noop, 15% stochastic_self, 70% pool
    """
    n = len(pool)
    noise_prob = _lb_noise_prob(iter_, lb_start, noise_end, noise_start)

    if n < 2:
        if random.random() < 0.5:
            return {"opp_type": "stochastic_self", "opp_weights_bytes": None}
        return {"opp_type": "noop", "opp_weights_bytes": None}

    r = random.random()
    if iter_ >= lb_start and r < lb_prob:
        opp_type = random.choice(["noisy_lb928", "noisy_lb1200"])
        return {"opp_type": opp_type, "opp_weights_bytes": None, "noise_prob": noise_prob}

    remaining_r = (r - lb_prob) / max(1e-6, 1.0 - lb_prob) if iter_ >= lb_start else r
    if remaining_r < 0.15:
        return {"opp_type": "noop", "opp_weights_bytes": None}
    if remaining_r < 0.30:
        return {"opp_type": "stochastic_self", "opp_weights_bytes": None}
    idx = pool.sample_index()
    return {"opp_type": "pool", "opp_weights_bytes": pool.snapshots[idx],
            "opp_tag": pool.tags[idx]}


def _eval_task(opp_name: str) -> dict:
    """Eval task: play vs error-free lb928 or lb1200 (noise_prob=0)."""
    return {"opp_type": f"noisy_{opp_name}",   # reuses rollout path
            "opp_weights_bytes": None,
            "noise_prob": 0.0,
            "eval_mode": True}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start-k12", default="training/checkpoints/physics_picker_k12_v3.pt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-iters", type=int, default=1000)
    ap.add_argument("--games-per-iter", type=int, default=16)
    ap.add_argument("--replay-iters", type=int, default=3)
    ap.add_argument("--four-player-prob", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef-start", type=float, default=0.05)
    ap.add_argument("--ent-coef-end", type=float, default=0.01)
    ap.add_argument("--ent-decay-iters", type=int, default=100)
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--pool-size", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=20,
                    help="run eval vs error-free strong bots every N iters")
    ap.add_argument("--eval-games", type=int, default=20,
                    help="games per strong bot in eval")
    ap.add_argument("--lb-prob", type=float, default=0.2,
                    help="fraction of training games vs noisy lb (once lb_start_iter reached)")
    ap.add_argument("--lb-start-iter", type=int, default=40,
                    help="iter at which noisy lb opponents first appear")
    ap.add_argument("--noise-start-prob", type=float, default=0.60,
                    help="lb mistake rate at lb_start_iter (decays to 0 by noise_end_iter)")
    ap.add_argument("--noise-end-iter", type=int, default=500,
                    help="iter at which lb mistake rate reaches 0 (pure lb)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None,
                    help="resume from a k14 checkpoint (overrides --warm-start-k12)")
    ap.add_argument("--start-iter", type=int, default=1,
                    help="starting iteration number (use with --resume)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    ).to(device)

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        net.load_state_dict(sd, strict=False)
        resume_iter = ckpt.get("iter", args.start_iter - 1) if isinstance(ckpt, dict) else args.start_iter - 1
        print(f"[k14] resumed from {args.resume} (iter {resume_iter})", flush=True)
        if args.start_iter == 1:
            args.start_iter = resume_iter + 1
    elif args.warm_start_k12 and Path(args.warm_start_k12).exists():
        loaded = load_k12_into_k13(net, args.warm_start_k12)
        print(f"[k14] warm-started {loaded} tensors from {args.warm_start_k12}", flush=True)
    init_head_biases(net)
    print(f"[k14] mode_bias={net.mode_head.bias.detach().cpu().numpy().round(2).tolist()}",
          flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"[k14] device={device}  params={sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"games/iter={args.games_per_iter}  snapshot_every={args.snapshot_every}  "
          f"pool_size={args.pool_size}  ent_coef={args.ent_coef_start}→{args.ent_coef_end}",
          flush=True)

    def sdb():
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
        return buf.getvalue()

    pool = OpponentPool(max_size=args.pool_size)
    last_eval = {"vs928": 0, "vs1200": 0, "wr": 0.0, "iter": 0}

    ctx = mp.get_context("spawn")
    worker_pool = ctx.Pool(processes=args.workers, initializer=_worker_init,
                           initargs=(sdb(),))
    t0 = time.time()
    replay_buf: collections.deque = collections.deque(maxlen=args.replay_iters)

    for iter_ in range(args.start_iter, args.start_iter + args.target_iters):
        # Refresh worker pool with latest weights every 5 iters.
        # Skip if this iter will also trigger eval (eval does its own refresh with updated weights).
        is_eval_iter = (iter_ % args.eval_every == 0)
        if iter_ % 5 == 0 and not is_eval_iter:
            worker_pool.close(); worker_pool.join()
            worker_pool = ctx.Pool(processes=args.workers, initializer=_worker_init,
                                   initargs=(sdb(),))

        # Build tasks: sample opponent per game from pool
        tasks = []
        for _ in range(args.games_per_iter):
            n_players = 4 if random.random() < args.four_player_prob else 2
            opp_info = _opponent_task(pool, iter_,
                                      args.lb_prob, args.lb_start_iter,
                                      args.noise_end_iter, args.noise_start_prob)
            tasks.append({"n_players": n_players,
                          "picker_seat": random.randint(0, n_players - 1),
                          **opp_info})

        rollouts = worker_pool.map(_rollout_game, tasks)

        iter_samples = []
        wins = 0
        opp_wins = {"noop": [0, 0], "random": [0, 0], "stochastic_self": [0, 0],
                    "pool": [0, 0], "noisy_lb928": [0, 0], "noisy_lb1200": [0, 0]}
        mc_iter = [0] * N_MODES
        fc_iter = [0] * N_FRACS
        for r in rollouts:
            iter_samples.extend(r["samples"])
            if r["win"]: wins += 1
            opp = r.get("opp", "pool")
            if opp in opp_wins:
                opp_wins[opp][1] += 1
                if r["win"]: opp_wins[opp][0] += 1
            for i, c in enumerate(r["mode_counts"]): mc_iter[i] += c
            for i, c in enumerate(r["frac_counts"]): fc_iter[i] += c
        replay_buf.append(iter_samples)
        all_training = [s for chunk in replay_buf for s in chunk]

        # Entropy annealing
        alpha = min(1.0, iter_ / max(1, args.ent_decay_iters))
        ent_coef = args.ent_coef_start + alpha * (args.ent_coef_end - args.ent_coef_start)

        info = ppo_update(net, opt, all_training, device, ent_coef=ent_coef)

        # Snapshot to pool
        if iter_ % args.snapshot_every == 0:
            pool.add(sdb(), f"iter{iter_:04d}")

        elapsed = time.time() - t0
        mc_str = " ".join(f"{c}" for c in mc_iter)
        fc_str = " ".join(f"{c}" for c in fc_iter)
        noise_now = _lb_noise_prob(iter_, args.lb_start_iter, args.noise_end_iter, args.noise_start_prob)
        opp_parts = [f"noop={opp_wins['noop'][0]}/{opp_wins['noop'][1]}",
                     f"self={opp_wins['stochastic_self'][0]}/{opp_wins['stochastic_self'][1]}",
                     f"pool={opp_wins['pool'][0]}/{opp_wins['pool'][1]}"]
        if opp_wins["noisy_lb928"][1] > 0:
            opp_parts.append(f"lb928={opp_wins['noisy_lb928'][0]}/{opp_wins['noisy_lb928'][1]}")
        if opp_wins["noisy_lb1200"][1] > 0:
            opp_parts.append(f"lb1200={opp_wins['noisy_lb1200'][0]}/{opp_wins['noisy_lb1200'][1]}")
        opp_str = "  ".join(opp_parts)
        noise_str = f"  noise={noise_now:.2f}" if iter_ >= args.lb_start_iter else ""
        print(f"[iter {iter_:04d}] games={args.games_per_iter}  "
              f"new={len(iter_samples)} train={len(all_training)}  "
              f"wins={wins}/{args.games_per_iter} ({opp_str})  "
              f"pi={info.get('pi_loss',0):.3f}  v={info.get('v_loss',0):.3f}  "
              f"Hm={info.get('ent_mode',0):.3f}  Hf={info.get('ent_frac',0):.3f}  "
              f"ratio={info.get('ratio_mean',1.0):.2f}/{info.get('ratio_max',1.0):.1f}  "
              f"ent_coef={ent_coef:.3f}  pool={len(pool)}{noise_str}  "
              f"mc=[{mc_str}]  fc=[{fc_str}]  [{elapsed:.0f}s]", flush=True)

        if iter_ % (args.snapshot_every * 2) == 0:
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, args.out)
            stem = Path(args.out).stem
            versioned = Path(args.out).parent / f"{stem}_iter{iter_:04d}.pt"
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, versioned)
            print(f"[iter {iter_:04d}] saved {args.out} + {versioned.name}", flush=True)

        # Evaluation: every N iters, play vs error-free lb928 & lb1200
        if iter_ % args.eval_every == 0:
            # Refresh workers so they have the current policy before eval
            worker_pool.close(); worker_pool.join()
            worker_pool = ctx.Pool(processes=args.workers, initializer=_worker_init,
                                   initargs=(sdb(),))
            eval_tasks = []
            for _ in range(args.eval_games):
                eval_tasks.append({"n_players": 2, "picker_seat": random.randint(0, 1),
                                    **_eval_task("lb928")})
            for _ in range(args.eval_games):
                eval_tasks.append({"n_players": 2, "picker_seat": random.randint(0, 1),
                                    **_eval_task("lb1200")})
            eval_rollouts = worker_pool.map(_rollout_game, eval_tasks)
            w928 = sum(1 for r in eval_rollouts[:args.eval_games] if r["win"])
            w1200 = sum(1 for r in eval_rollouts[args.eval_games:] if r["win"])
            wr = (w928 + w1200) / (2 * args.eval_games)
            last_eval = {"vs928": w928, "vs1200": w1200, "wr": wr, "iter": iter_}
            print(f"[eval {iter_:04d}] vs928={w928}/{args.eval_games}  "
                  f"vs1200={w1200}/{args.eval_games}  wr={wr:.1%}  "
                  f"lb_active={'YES' if iter_ >= args.lb_start_iter else 'NO'}", flush=True)
            # lb schedule is now time-based (iter >= lb_start_iter), not eval-gated

    worker_pool.close(); worker_pool.join()


if __name__ == "__main__":
    main()
