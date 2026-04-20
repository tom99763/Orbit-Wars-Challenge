"""AlphaZero v2 — adapted to current architecture (35/10/38 + K=3 + DualStreamAgent).

Design decisions (vs alphazero.py v1):

  1. Policy output = **variant-picker K=8 softmax** (same as variant_picker_dual_stream_ppo)
     - Simpler MCTS tree (only 8 children per node vs exponential joint action)
     - Variants include pass + lb-1200 primary + 6 perturbations
     - Each variant is a ready-to-submit action list
  2. PolicyValueNet = **DualStreamAgent** (Entity + Spatial CNN + Scalar)
  3. Feature schema: new K=3 sliding window (35/10/38 dims)
  4. MCTS leaf evaluation: DualStreamAgent value head + lb-1200 rollout (hybrid)
  5. Training:
       policy_loss = CE(π_net, π_mcts_visit_distribution)   — learner seat only
       value_loss  = MSE(V_net, terminal_reward)             — ALL seats
       loss = policy_loss + vf_coef * value_loss - ent_coef * H(π_net)
  6. Opponent pool: 60% self-play / 20% past snapshot / 15% lb-1200 / 5% lb-928
     (per memory: design_impala_v3_exploration.md)

Why variant-picker policy instead of full action space:
  - AlphaZero's branching factor grows exponentially with per-planet decisions
  - K=8 is tractable for deep tree search
  - Variants already capture meaningful policy perturbations
  - Safety net: V0 = lb-1200 primary ensures worst case = baseline

How MCTS works here:
  Root: current state
  Node: state + who-to-play
  Children: K=8 variant action indices
  Step: apply chosen variant's action list, opponent plays lb-1200 (deterministic)
  Leaf: N steps later OR game end; value from NN or terminal reward

Usage:
  python training/alphazero_v2.py \\
      --workers 4 --target-iters 500 --games-per-iter 2 \\
      --n-sims 30 --warm-start training/checkpoints/variant_picker_v2.pt \\
      --opponent-pool training/checkpoints/az_pool \\
      --out training/checkpoints/az_v1.pt

Prereqs before running:
  - `training/dual_stream_model.py` present (we wrote it today)
  - `training/variant_picker_dual_stream_ppo.py` present (for variant definitions)
  - `training/opponent_pool.py` present (for opponent management)
  - `featurize.py` with K=3 support (PLANET_DIM=35, etc.)
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import copy
import io
import math
import multiprocessing as mp
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from featurize import (featurize_step, PLANET_DIM, FLEET_DIM, GLOBAL_DIM, HISTORY_K)
from training.lb1200_agent import agent as lb1200_agent
from training.dual_stream_model import (
    DualStreamAgent, rasterize_obs, N_SPATIAL_CHANNELS,
)
from training.variant_picker_dual_stream_ppo import (
    generate_variants, K_VARIANTS, GRID,
)
from training.opponent_pool import OpponentPool, sample_opponent_mix, DEFAULT_MIX


# -----------------------------------------------------------------------------
# MCTS node
# -----------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """Single node in the search tree."""
    prior: np.ndarray    # [K] softmax policy prior from NN
    visits: np.ndarray   # [K] visit counts per variant
    q_values: np.ndarray # [K] running average Q per variant
    is_terminal: bool = False
    terminal_value: float = 0.0
    # Children cache: variant_idx → next MCTSNode (lazy expanded)
    children: dict = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}


# -----------------------------------------------------------------------------
# MCTS search
# -----------------------------------------------------------------------------

class AZMCTS:
    """Variant-picker MCTS for Orbit Wars 2P.

    Only supports 2P for simplicity (opponent is deterministic lb-1200 or
    opponent-pool snapshot). 4P multi-agent MCTS is much more complex and
    deferred.
    """
    def __init__(self, net: DualStreamAgent, device: str = "cpu",
                 n_sims: int = 30, c_puct: float = 1.4,
                 rollout_policy: str = "lb1200"):
        self.net = net
        self.device = device
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.rollout_policy = rollout_policy   # "lb1200" or "self"

    def _evaluate(self, env, seat: int) -> tuple[np.ndarray, float]:
        """Return (policy_prior [K], value V) for current state from net."""
        obs = env.state[seat].observation
        # Featurize state
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) \
                      else getattr(obs, "planets", [])
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) \
                     else getattr(obs, "fleets", [])
        # Build a minimal step_dict for featurize_step
        step_dict = {
            "step": len(env.steps) - 1,
            "planets": raw_planets, "fleets": raw_fleets,
            "action": [],
            "my_total_ships": sum(p[5] for p in raw_planets if p[1] == seat),
            "enemy_total_ships": 0, "my_planet_count": 0,
            "enemy_planet_count": 0, "neutral_planet_count": 0,
        }
        ang_vel = float(obs.get("angular_velocity", 0.0)) if isinstance(obs, dict) \
                  else float(getattr(obs, "angular_velocity", 0.0) or 0.0)
        init_planets = obs.get("initial_planets", []) if isinstance(obs, dict) \
                       else getattr(obs, "initial_planets", [])
        feat = featurize_step(
            step_dict, seat, ang_vel, 2, init_planets,
            # MCTS doesn't carry K=3 history per-sim-branch, pass empty
            last_actions_by_planet={}, cumulative_stats={"total_ships_sent": 0, "total_actions": 0},
            obs_history=[], action_history=[],
        )
        spatial = rasterize_obs(obs, seat, grid=GRID)

        pl = feat["planets"]; fl = feat["fleets"]
        pmask = np.ones(pl.shape[0], dtype=bool) if pl.shape[0] > 0 \
                else np.zeros(0, dtype=bool)
        if pl.shape[0] == 0:
            pl = np.zeros((1, PLANET_DIM), dtype=np.float32); pmask = np.zeros(1, dtype=bool)
        if fl.ndim < 2 or fl.shape[0] == 0:
            fl = np.zeros((1, FLEET_DIM), dtype=np.float32); fmask = np.zeros(1, dtype=bool)
        else:
            fmask = np.ones(fl.shape[0], dtype=bool)

        with torch.no_grad():
            logits, value = self.net(
                torch.from_numpy(pl).unsqueeze(0).to(self.device),
                torch.from_numpy(pmask).unsqueeze(0).to(self.device),
                torch.from_numpy(fl).unsqueeze(0).to(self.device),
                torch.from_numpy(fmask).unsqueeze(0).to(self.device),
                torch.from_numpy(feat["globals"]).unsqueeze(0).to(self.device),
                torch.from_numpy(spatial).unsqueeze(0).to(self.device),
            )
            prior = F.softmax(logits[0], dim=-1).cpu().numpy()
            v = float(value.item())
        return prior, v

    def _generate_variants_for_env(self, env, seat: int) -> list[list]:
        """Produce the K=8 candidate action lists from current env state."""
        from training.lb1200_agent import agent as lb
        obs = env.state[seat].observation
        primary = lb(obs, env.configuration) or []
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) \
                      else getattr(obs, "planets", [])
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) \
                     else getattr(obs, "fleets", [])
        ang_vel = float(obs.get("angular_velocity", 0.0)) if isinstance(obs, dict) else 0.0
        init_planets = obs.get("initial_planets", []) if isinstance(obs, dict) else []
        from training.lb1200_agent import Planet as _Planet
        init_by_id = {int(p[0]): _Planet(int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                                          float(p[4]), int(p[5]), int(p[6]))
                      for p in init_planets} if init_planets else {}
        return generate_variants(primary, raw_planets, raw_fleets, seat,
                                  ang_vel=ang_vel, initial_by_id=init_by_id)

    def _step_env(self, env, learner_seat: int, variant_idx: int,
                  opp_action_fn) -> tuple[Any, bool, float]:
        """Clone env and apply one turn. Returns (new_env, is_terminal, reward_for_learner)."""
        new_env = copy.deepcopy(env)
        n_players = len(new_env.state)
        variants = self._generate_variants_for_env(new_env, learner_seat)
        if variant_idx >= len(variants):
            variant_idx = 0
        actions = [[] for _ in range(n_players)]
        actions[learner_seat] = variants[variant_idx]
        for s in range(n_players):
            if s != learner_seat:
                actions[s] = opp_action_fn(new_env, s)
        new_env.step(actions)
        done = new_env.done
        reward = float(new_env.state[learner_seat].reward or 0) if done else 0.0
        return new_env, done, reward

    def search(self, env, learner_seat: int, opp_action_fn) -> np.ndarray:
        """Run n_sims MCTS simulations. Return visit distribution π_mcts [K]."""
        prior, _ = self._evaluate(env, learner_seat)
        root = MCTSNode(
            prior=prior,
            visits=np.zeros(K_VARIANTS),
            q_values=np.zeros(K_VARIANTS),
        )

        for _ in range(self.n_sims):
            # Select via UCB
            self._simulate(env, root, learner_seat, opp_action_fn, depth=0,
                           max_depth=5)

        # π_mcts = visit counts normalized
        total = root.visits.sum()
        if total < 1e-6:
            return prior
        return root.visits / total

    def _simulate(self, env, node: MCTSNode, learner_seat: int,
                   opp_action_fn, depth: int, max_depth: int) -> float:
        """One MCTS simulation. Backs up value through tree."""
        if node.is_terminal or depth >= max_depth:
            return node.terminal_value if node.is_terminal else 0.0

        # UCB selection
        total_visits = node.visits.sum()
        ucb = node.q_values + self.c_puct * node.prior * \
              math.sqrt(total_visits + 1) / (1 + node.visits)
        action_idx = int(np.argmax(ucb))

        # Expand if not yet visited
        if action_idx not in node.children:
            new_env, done, reward = self._step_env(env, learner_seat,
                                                    action_idx, opp_action_fn)
            if done:
                child = MCTSNode(prior=np.zeros(K_VARIANTS),
                                  visits=np.zeros(K_VARIANTS),
                                  q_values=np.zeros(K_VARIANTS),
                                  is_terminal=True, terminal_value=reward)
            else:
                prior, value = self._evaluate(new_env, learner_seat)
                child = MCTSNode(prior=prior, visits=np.zeros(K_VARIANTS),
                                  q_values=np.zeros(K_VARIANTS),
                                  terminal_value=value)
            node.children[action_idx] = (child, new_env)
            value = child.terminal_value
        else:
            child, new_env = node.children[action_idx]
            value = self._simulate(new_env, child, learner_seat, opp_action_fn,
                                    depth + 1, max_depth)

        # Backup
        node.visits[action_idx] += 1
        n = node.visits[action_idx]
        node.q_values[action_idx] = (
            node.q_values[action_idx] * (n - 1) + value
        ) / n
        return value


# -----------------------------------------------------------------------------
# Rollout worker (self-play one game)
# -----------------------------------------------------------------------------

def _worker_init(state_dict_bytes: bytes):
    global _worker_net
    from kaggle_environments import make   # noqa: F401
    _worker_net = DualStreamAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
        n_variants=K_VARIANTS,
    )
    buf = io.BytesIO(state_dict_bytes)
    sd = torch.load(buf, map_location="cpu", weights_only=False)
    _worker_net.load_state_dict(sd)
    _worker_net.eval()


def _lb1200_opp_action(env, seat: int) -> list:
    obs = env.state[seat].observation
    return lb1200_agent(obs, env.configuration) or []


def _worker_play(task: dict) -> dict:
    from kaggle_environments import make
    from training.mcts import disable_env_validation
    disable_env_validation()   # speedup env cloning

    n_players = 2             # AZ v2 restricted to 2P
    learner_seat = task["learner_seat"]
    n_sims = task["n_sims"]

    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)

    mcts = AZMCTS(_worker_net, device="cpu", n_sims=n_sims)

    # Opponent: lb-1200 (simple), could extend to opponent-pool sampling
    opp_fn = _lb1200_opp_action

    records: list[dict] = []   # per learner turn: (state_feat, spatial, π_mcts)
    other_records: list[dict] = []   # opp seat: state_feat only (for value head)

    t = 0
    while not env.done and t < 500:
        # MCTS for learner
        pi_mcts = mcts.search(env, learner_seat, opp_fn)
        # Sample action according to pi_mcts (temperature 1 early, 0 late)
        temp = 1.0 if t < 30 else 0.3
        logits = np.log(pi_mcts + 1e-9) / temp
        probs = np.exp(logits) / np.exp(logits).sum()
        variant_idx = int(np.random.choice(K_VARIANTS, p=probs))

        # Collect state feat for learner
        obs = env.state[learner_seat].observation
        # (reuse simpler featurize_step call — K=3 history ignored for MCTS root for now)
        raw_planets = obs.get("planets", []) or []
        step_dict = {
            "step": t, "planets": raw_planets,
            "fleets": obs.get("fleets", []) or [],
            "action": [],
            "my_total_ships": 0, "enemy_total_ships": 0,
            "my_planet_count": 0, "enemy_planet_count": 0,
            "neutral_planet_count": 0,
        }
        ang_vel = float(obs.get("angular_velocity", 0.0))
        init_planets = obs.get("initial_planets", [])
        feat = featurize_step(step_dict, learner_seat, ang_vel, 2, init_planets,
                               last_actions_by_planet={},
                               cumulative_stats={"total_ships_sent": 0, "total_actions": 0},
                               obs_history=[], action_history=[])
        spatial = rasterize_obs(obs, learner_seat, grid=GRID)
        records.append({"feat": feat, "spatial": spatial, "pi_mcts": pi_mcts})

        # Generate variants and execute
        variants = mcts._generate_variants_for_env(env, learner_seat)
        actions_all = [[] for _ in range(n_players)]
        actions_all[learner_seat] = variants[variant_idx]
        for s in range(n_players):
            if s != learner_seat:
                actions_all[s] = opp_fn(env, s)
                # Opp record (value-only training)
                opp_obs = env.state[s].observation
                opp_feat = featurize_step(
                    {"step": t, "planets": opp_obs.get("planets", []) or [],
                     "fleets": opp_obs.get("fleets", []) or [],
                     "action": [], "my_total_ships": 0, "enemy_total_ships": 0,
                     "my_planet_count": 0, "enemy_planet_count": 0,
                     "neutral_planet_count": 0},
                    s, ang_vel, 2, init_planets,
                    last_actions_by_planet={}, cumulative_stats={"total_ships_sent": 0, "total_actions": 0},
                    obs_history=[], action_history=[]
                )
                opp_spatial = rasterize_obs(opp_obs, s, grid=GRID)
                other_records.append({"feat": opp_feat, "spatial": opp_spatial,
                                       "seat": s})
        env.step(actions_all)
        t += 1

    rewards = [s.reward if s.reward is not None else 0 for s in env.state]
    learner_reward = float(rewards[learner_seat])

    # Assign terminal reward to all records
    for r in records:
        r["terminal"] = learner_reward
    for r in other_records:
        r["terminal"] = float(rewards[r["seat"]])

    return {
        "learner_seat": learner_seat,
        "rewards": rewards,
        "records": records,
        "other_records": other_records,
        "steps": t,
    }


# -----------------------------------------------------------------------------
# Training step
# -----------------------------------------------------------------------------

def _collate_records(learner_recs: list[dict], other_recs: list[dict],
                     device: str) -> dict:
    all_recs = []
    has_policy = []
    for r in learner_recs:
        all_recs.append(r); has_policy.append(True)
    for r in other_recs:
        all_recs.append(r); has_policy.append(False)

    B = len(all_recs)
    if B == 0: return None
    P = max(r["feat"]["planets"].shape[0] for r in all_recs) or 1
    F_ = max(r["feat"]["fleets"].shape[0]
             if r["feat"]["fleets"].ndim == 2 else 1
             for r in all_recs) or 1

    planets = np.zeros((B, P, PLANET_DIM), dtype=np.float32)
    pmask = np.zeros((B, P), dtype=bool)
    fleets = np.zeros((B, F_, FLEET_DIM), dtype=np.float32)
    fmask = np.zeros((B, F_), dtype=bool)
    globals_ = np.zeros((B, GLOBAL_DIM), dtype=np.float32)
    spatial = np.zeros((B, N_SPATIAL_CHANNELS, GRID, GRID), dtype=np.float32)
    pi_mcts = np.zeros((B, K_VARIANTS), dtype=np.float32)
    terminal = np.zeros((B,), dtype=np.float32)
    policy_mask = np.zeros((B,), dtype=bool)

    for i, r in enumerate(all_recs):
        f = r["feat"]
        np_ = f["planets"].shape[0]
        nf = f["fleets"].shape[0] if f["fleets"].ndim == 2 else 0
        if np_ > 0:
            planets[i, :np_] = f["planets"]; pmask[i, :np_] = True
        if nf > 0:
            fleets[i, :nf] = f["fleets"]; fmask[i, :nf] = True
        globals_[i] = f["globals"]
        spatial[i] = r["spatial"]
        if has_policy[i]:
            pi_mcts[i] = r["pi_mcts"]
            policy_mask[i] = True
        terminal[i] = r["terminal"]

    return {
        "planets": torch.from_numpy(planets).to(device),
        "pmask": torch.from_numpy(pmask).to(device),
        "fleets": torch.from_numpy(fleets).to(device),
        "fmask": torch.from_numpy(fmask).to(device),
        "globals": torch.from_numpy(globals_).to(device),
        "spatial": torch.from_numpy(spatial).to(device),
        "pi_mcts": torch.from_numpy(pi_mcts).to(device),
        "terminal": torch.from_numpy(terminal).to(device),
        "policy_mask": torch.from_numpy(policy_mask).to(device),
    }


def az_train_step(net: DualStreamAgent, opt: torch.optim.Optimizer,
                  learner_recs: list, other_recs: list, device: str,
                  vf_coef: float = 1.0, ent_coef: float = 0.01):
    batch = _collate_records(learner_recs, other_recs, device)
    if batch is None:
        return {}
    logits, values = net(
        batch["planets"], batch["pmask"],
        batch["fleets"], batch["fmask"], batch["globals"],
        batch["spatial"],
    )
    # Policy loss (learner only — masked)
    pol_mask = batch["policy_mask"]
    if pol_mask.any():
        log_probs = F.log_softmax(logits[pol_mask], dim=-1)
        pi_target = batch["pi_mcts"][pol_mask]
        policy_loss = -(pi_target * log_probs).sum(dim=-1).mean()
        entropy = -(F.softmax(logits[pol_mask], dim=-1) * log_probs).sum(dim=-1).mean()
    else:
        policy_loss = torch.zeros((), device=device)
        entropy = torch.zeros((), device=device)
    # Value loss (all records)
    value_loss = F.mse_loss(values, batch["terminal"])

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--games-per-iter", type=int, default=2)
    ap.add_argument("--target-iters", type=int, default=500)
    ap.add_argument("--n-sims", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DualStreamAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
        n_variants=K_VARIANTS,
    ).to(device)
    if args.warm_start and Path(args.warm_start).exists():
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        try:
            net.load_state_dict(sd)
            print(f"[az_v2] warm-started from {args.warm_start}", flush=True)
        except RuntimeError as e:
            print(f"[az_v2] warm-start FAILED (schema mismatch?): {e}", flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def state_dict_bytes():
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
        return buf.getvalue()

    print(f"[az_v2] device={device}  params={sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"n_sims={args.n_sims}  workers={args.workers}", flush=True)

    pool = mp.get_context("spawn").Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(state_dict_bytes(),),
    )

    t0 = time.time()
    for iter_ in range(1, args.target_iters + 1):
        if iter_ % 5 == 1:
            pool.close()
            pool = mp.get_context("spawn").Pool(
                processes=args.workers,
                initializer=_worker_init,
                initargs=(state_dict_bytes(),),
            )

        tasks = [{"learner_seat": random.randint(0, 1), "n_sims": args.n_sims}
                 for _ in range(args.games_per_iter)]
        results = pool.map(_worker_play, tasks)

        learner_recs = []
        other_recs = []
        wins = 0
        for r in results:
            learner_recs.extend(r["records"])
            other_recs.extend(r["other_records"])
            if r["rewards"][r["learner_seat"]] > 0:
                wins += 1

        info = az_train_step(net, opt, learner_recs, other_recs, device,
                              vf_coef=args.vf_coef, ent_coef=args.ent_coef)
        elapsed = time.time() - t0
        print(f"[iter {iter_:04d}] games={args.games_per_iter}  "
              f"samples={len(learner_recs)+len(other_recs)}  "
              f"wins={wins}/{args.games_per_iter}  "
              f"pol={info.get('policy_loss', 0):.3f}  "
              f"val={info.get('value_loss', 0):.3f}  "
              f"ent={info.get('entropy', 0):.3f}  [{elapsed:.0f}s]", flush=True)

        if iter_ % args.snapshot_every == 0:
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                       args.out)
            print(f"[iter {iter_:04d}] saved {args.out}", flush=True)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
