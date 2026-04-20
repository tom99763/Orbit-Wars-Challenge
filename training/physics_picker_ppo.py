"""Physics Picker PPO v3 — DualStream + per-planet × K=6 physics candidates.

Escape the K=8 variant ceiling by having NN output per-planet action selection
over physics-generated candidates. Still keeps physics primitives (aim_with_prediction,
sun_blocker) hardcoded — NN only learns which physical option is best per state.

Action space:
  For each owned planet (source):
    NN outputs softmax over K_PER_SOURCE=6 candidates:
      C0: pass
      C1: min-ships capture of nearest capturable
      C2: half-ships capture of nearest
      C3: all-in capture of nearest
      C4: min-ships capture of highest-production nearby
      C5: reinforce friendly in danger
  Joint action = compose per-source picks into env action list

Branching factor per source: 6 (vs 160 in full per-planet tgt×bkt)
Joint space: 6^P where P = owned planets (typical 5-10) → 8k to 60M
Physics helper: materializes (source, candidate_idx) → [src, angle, ships]

Logs: pi_loss, v_loss, entropy, wins, variant_counts, iter_time (per spec).

Usage:
  python training/physics_picker_ppo.py \\
      --workers 4 --target-iters 2000 --games-per-iter 4 \\
      --four-player-prob 0.3 --lr 3e-4 --ent-coef 0.03 \\
      --out training/checkpoints/physics_picker_v3.pt
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
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from featurize import (featurize_step, PLANET_DIM, FLEET_DIM, GLOBAL_DIM, HISTORY_K)
from training.model import OrbitAgent, SetAttentionBlock
from training.dual_stream_model import (SpatialCNN, ScalarMLP, rasterize_obs,
                                         N_SPATIAL_CHANNELS)
from training.lb1200_agent import build_world, Planet as _Planet
from training.physics_action_helper import (
    K_PER_SOURCE, CANDIDATE_NAMES, generate_per_source_candidates,
    materialize_joint_action,
)

GRID = 32
GAMMA = 0.99


# -----------------------------------------------------------------------------
# DualStreamCandidateAgent — per-planet × K=6 candidate softmax
# -----------------------------------------------------------------------------

class DualStreamCandidateAgent(nn.Module):
    """Dual-stream encoder → per-planet candidate logits + scalar value.

    Output:
      cand_logits [B, P, K_PER_SOURCE]  — per planet, softmax over 6 candidates
      value       [B]                    — scalar V(s)
    """
    def __init__(
        self,
        planet_dim: int,
        fleet_dim: int,
        global_dim: int,
        k_per_source: int = K_PER_SOURCE,
        d_entity: int = 128,
        d_spatial: int = 128,
        d_scalar: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
    ):
        super().__init__()
        self.d = d_entity
        self.k = k_per_source

        # Entity encoder
        self.planet_embed = nn.Sequential(
            nn.Linear(planet_dim, d_entity), nn.GELU(), nn.Linear(d_entity, d_entity))
        self.fleet_embed = nn.Sequential(
            nn.Linear(fleet_dim, d_entity), nn.GELU(), nn.Linear(d_entity, d_entity))
        self.global_embed = nn.Sequential(
            nn.Linear(global_dim, d_entity), nn.GELU(), nn.Linear(d_entity, d_entity))
        self.type_embed = nn.Embedding(3, d_entity)
        self.attn_layers = nn.ModuleList(
            [SetAttentionBlock(d_entity, n_heads) for _ in range(n_layers)]
        )

        # Spatial + Scalar
        self.spatial_enc = SpatialCNN(N_SPATIAL_CHANNELS, d_spatial, grid=GRID)
        self.scalar_enc = ScalarMLP(global_dim=global_dim, d_model=d_scalar)

        # Fusion
        self.fuse_global = nn.Sequential(
            nn.Linear(d_entity + d_spatial + d_scalar, d_entity), nn.GELU(),
            nn.Linear(d_entity, d_entity),
        )
        # Per-planet candidate head
        self.cand_head = nn.Linear(d_entity, k_per_source)
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_entity, d_entity), nn.GELU(),
            nn.Linear(d_entity, 1),
        )

    def forward(self, planets, planet_mask, fleets, fleet_mask, globals_, spatial):
        B, P, _ = planets.shape
        p_tok = self.planet_embed(planets) + self.type_embed.weight[0]
        f_tok = self.fleet_embed(fleets) + self.type_embed.weight[1]
        g_tok = self.global_embed(globals_).unsqueeze(1) + self.type_embed.weight[2]
        tokens = torch.cat([p_tok, f_tok, g_tok], dim=1)
        g_mask = torch.ones(B, 1, dtype=torch.bool, device=planets.device)
        valid = torch.cat([planet_mask, fleet_mask, g_mask], dim=1)
        for blk in self.attn_layers:
            tokens = blk(tokens, ~valid)

        planet_tokens = tokens[:, :P, :]                # [B,P,d]
        entity_global = tokens[:, -1, :]                 # [B,d]
        spatial_feat = self.spatial_enc(spatial)         # [B,d_spatial]
        scalar_feat = self.scalar_enc(globals_)          # [B,d_scalar]
        fused_g = self.fuse_global(
            torch.cat([entity_global, spatial_feat, scalar_feat], dim=-1)
        )                                                 # [B,d]
        fused_planet_tokens = planet_tokens + fused_g.unsqueeze(1)
        cand_logits = self.cand_head(fused_planet_tokens)   # [B,P,K]
        value = self.value_head(fused_g).squeeze(-1)         # [B]
        return cand_logits, value


# -----------------------------------------------------------------------------
# Rollout worker
# -----------------------------------------------------------------------------

def _worker_init(state_dict_bytes: bytes):
    global _worker_net
    from kaggle_environments import make   # noqa
    _worker_net = DualStreamCandidateAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    )
    buf = io.BytesIO(state_dict_bytes)
    sd = torch.load(buf, map_location="cpu", weights_only=False)
    _worker_net.load_state_dict(sd)
    _worker_net.eval()


def _rollout_game(task: dict) -> dict:
    from kaggle_environments import make
    from training.lb1200_agent import agent as lb1200_agent

    n_players = task["n_players"]
    picker_seat = task["picker_seat"]
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)

    obs_history = collections.deque(maxlen=HISTORY_K)
    action_history = collections.deque(maxlen=HISTORY_K)
    last_actions_by_planet: dict = {}
    cum_stats = {"total_ships_sent": 0, "total_actions": 0}

    samples = []
    ang_vel_init = None
    init_planets = None
    step = 0
    prev_my_ships = 0
    prev_my_planets = 1
    cand_choice_counts = [0] * K_PER_SOURCE

    while not env.done and step < 500:
        actions_all = []
        for s in range(n_players):
            obs = env.state[s].observation
            if s == picker_seat:
                try:
                    world = build_world(obs)
                except Exception:
                    actions_all.append([])
                    continue
                raw_planets = obs.get("planets", []) or []
                raw_fleets = obs.get("fleets", []) or []

                my_planets_list = [p for p in world.planets if p.owner == s]
                if not my_planets_list:
                    actions_all.append([])
                    # Still advance histories
                    continue

                if ang_vel_init is None:
                    ang_vel_init = float(obs.get("angular_velocity", 0.0) or 0.0)
                    init_planets = obs.get("initial_planets", []) or []

                step_dict = {
                    "step": step, "planets": raw_planets, "fleets": raw_fleets,
                    "action": [],
                    "my_total_ships": sum(p[5] for p in raw_planets if p[1] == s),
                    "enemy_total_ships": 0, "my_planet_count": 0,
                    "enemy_planet_count": 0, "neutral_planet_count": 0,
                }
                feat = featurize_step(
                    step_dict, s, ang_vel_init, n_players, init_planets,
                    last_actions_by_planet=last_actions_by_planet,
                    cumulative_stats=cum_stats,
                    obs_history=list(obs_history),
                    action_history=list(action_history),
                )
                spatial = rasterize_obs(obs, s, grid=GRID)

                pl = feat["planets"]; fl = feat["fleets"]
                if pl.shape[0] == 0:
                    pl = np.zeros((1, PLANET_DIM), dtype=np.float32)
                    pmask = np.zeros(1, dtype=bool)
                else:
                    pmask = np.ones(pl.shape[0], dtype=bool)
                if fl.ndim < 2 or fl.shape[0] == 0:
                    fl = np.zeros((1, FLEET_DIM), dtype=np.float32)
                    fmask = np.zeros(1, dtype=bool)
                else:
                    fmask = np.ones(fl.shape[0], dtype=bool)

                with torch.no_grad():
                    logits, v = _worker_net(
                        torch.from_numpy(pl).unsqueeze(0),
                        torch.from_numpy(pmask).unsqueeze(0),
                        torch.from_numpy(fl).unsqueeze(0),
                        torch.from_numpy(fmask).unsqueeze(0),
                        torch.from_numpy(feat["globals"]).unsqueeze(0),
                        torch.from_numpy(spatial).unsqueeze(0),
                    )
                    # logits: [1, P_actual, K]
                    logits_np = logits[0].cpu().numpy()

                # Map featurize planet idx → planet id
                planet_ids_in_feat = feat.get("planet_ids")
                if planet_ids_in_feat is None:
                    actions_all.append([]); continue

                # Sample per-source candidate only for owned planets
                picks: list[tuple[int, int]] = []
                log_probs: list[float] = []
                total_entropy = 0.0
                n_owned = 0
                for i, pid in enumerate(planet_ids_in_feat):
                    # Find this planet in world to check owner
                    src = next((pl for pl in world.planets if pl.id == int(pid)), None)
                    if src is None or src.owner != s:
                        continue
                    # Sample candidate
                    planet_logits = logits_np[i]
                    probs = np.exp(planet_logits - planet_logits.max())
                    probs = probs / probs.sum()
                    ci = int(np.random.choice(K_PER_SOURCE, p=probs))
                    picks.append((int(pid), ci))
                    log_probs.append(float(np.log(probs[ci] + 1e-9)))
                    total_entropy += -float((probs * np.log(probs + 1e-9)).sum())
                    cand_choice_counts[ci] += 1
                    n_owned += 1

                # Materialize joint action via physics
                action_list = materialize_joint_action(picks, world, s)
                actions_all.append(action_list)

                if n_owned > 0:
                    samples.append({
                        "feat": feat, "spatial": spatial,
                        "planet_ids": planet_ids_in_feat,
                        "picks": picks,           # list of (planet_id, cand_idx)
                        "log_prob_sum": sum(log_probs),
                        "entropy": total_entropy / max(1, n_owned),
                        "value": float(v.item()),
                        "reward": 0.0,
                        "my_player": s,
                    })

                # Update history with executed action
                for mv in action_list:
                    from training.lb1200_agent import fleet_target_planet
                    if len(mv) != 3:
                        continue
                    src_id, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
                    pseudo_f = None
                    # Use nearest_target from featurize helpers
                    from featurize import nearest_target_index, ship_bucket_idx
                    src_planet = next((p for p in raw_planets if int(p[0]) == src_id), None)
                    if src_planet is None:
                        continue
                    tgt_i = nearest_target_index(src_planet, ang, raw_planets)
                    tgt_pid = int(raw_planets[tgt_i][0]) if tgt_i is not None else -1
                    garrison = int(src_planet[5]) + ships
                    bkt_idx = ship_bucket_idx(ships, max(1, garrison))
                    prev = last_actions_by_planet.get(src_id, (-1, 0, -1, 0))
                    last_actions_by_planet[src_id] = (tgt_pid, bkt_idx, step, prev[3] + 1)
                    cum_stats["total_ships_sent"] += ships
                    cum_stats["total_actions"] += 1
                    action_history.append((src_id, tgt_pid, bkt_idx, step))
                obs_history.append({"planets": raw_planets, "step": step})
            else:
                actions_all.append(lb1200_agent(obs, env.configuration) or [])

        env.step(actions_all)
        step += 1

        if samples:
            cur_planets = env.state[picker_seat].observation.get("planets", []) or []
            my_ships = sum(p[5] for p in cur_planets if p[1] == picker_seat)
            my_planets_ct = sum(1 for p in cur_planets if p[1] == picker_seat)
            r = 0.001 * (my_ships - prev_my_ships) + 0.02 * (my_planets_ct - prev_my_planets)
            samples[-1]["reward"] = r
            prev_my_ships, prev_my_planets = my_ships, my_planets_ct

    if samples:
        terminal = float(env.state[picker_seat].reward or 0)
        samples[-1]["reward"] += terminal

    # MC returns
    G = 0.0
    for s_ in reversed(samples):
        G = s_["reward"] + GAMMA * G
        s_["mc_return"] = G

    return {"samples": samples, "n_players": n_players, "picker_seat": picker_seat,
            "win": float(env.state[picker_seat].reward or 0) > 0,
            "cand_choice_counts": cand_choice_counts}


# -----------------------------------------------------------------------------
# PPO update — per-planet summed log-prob
# -----------------------------------------------------------------------------

def _collate_batch(samples: list[dict], device: str) -> dict:
    B = len(samples)
    P = max(s["feat"]["planets"].shape[0] for s in samples) or 1
    F_ = max((s["feat"]["fleets"].shape[0] if s["feat"]["fleets"].ndim == 2 else 1)
             for s in samples) or 1

    planets = np.zeros((B, P, PLANET_DIM), dtype=np.float32)
    pmask = np.zeros((B, P), dtype=bool)
    fleets = np.zeros((B, F_, FLEET_DIM), dtype=np.float32)
    fmask = np.zeros((B, F_), dtype=bool)
    globals_ = np.zeros((B, GLOBAL_DIM), dtype=np.float32)
    spatial = np.zeros((B, N_SPATIAL_CHANNELS, GRID, GRID), dtype=np.float32)
    returns = np.zeros((B,), dtype=np.float32)
    values = np.zeros((B,), dtype=np.float32)
    log_probs_old = np.zeros((B,), dtype=np.float32)
    # Per-planet pick indices: record planet-idx-in-feat → cand_idx
    pick_lists = []   # list of (planet_idx_in_feat_np, cand_idx_np)

    for i, s in enumerate(samples):
        f = s["feat"]
        np_ = f["planets"].shape[0]
        nf = f["fleets"].shape[0] if f["fleets"].ndim == 2 else 0
        if np_ > 0:
            planets[i, :np_] = f["planets"]; pmask[i, :np_] = True
        if nf > 0:
            fleets[i, :nf] = f["fleets"]; fmask[i, :nf] = True
        globals_[i] = f["globals"]
        spatial[i] = s["spatial"]
        returns[i] = s.get("mc_return", s["reward"])
        values[i] = s["value"]
        log_probs_old[i] = s["log_prob_sum"]

        # Map planet_id → index in feat
        pid_to_idx = {int(pid): j for j, pid in enumerate(s["planet_ids"])}
        picks_planet_idx = []; picks_cand_idx = []
        for pid, ci in s["picks"]:
            j = pid_to_idx.get(int(pid), -1)
            if j >= 0:
                picks_planet_idx.append(j)
                picks_cand_idx.append(ci)
        pick_lists.append((picks_planet_idx, picks_cand_idx))

    adv = returns - values
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    else:
        adv = adv - adv.mean()

    return {
        "planets": torch.from_numpy(planets).to(device),
        "pmask": torch.from_numpy(pmask).to(device),
        "fleets": torch.from_numpy(fleets).to(device),
        "fmask": torch.from_numpy(fmask).to(device),
        "globals": torch.from_numpy(globals_).to(device),
        "spatial": torch.from_numpy(spatial).to(device),
        "returns": torch.from_numpy(returns).to(device),
        "adv": torch.from_numpy(adv.astype(np.float32)).to(device),
        "log_probs_old": torch.from_numpy(log_probs_old).to(device),
        "pick_lists": pick_lists,
    }


def ppo_update(net: DualStreamCandidateAgent, opt: torch.optim.Optimizer,
               samples: list[dict], device: str,
               epochs: int = 4, clip: float = 0.2,
               ent_coef: float = 0.03, val_coef: float = 0.5) -> dict:
    if not samples:
        return {}
    batch = _collate_batch(samples, device)
    log_probs_old = batch["log_probs_old"]
    advantages = batch["adv"]
    returns = batch["returns"]
    pick_lists = batch["pick_lists"]
    B = len(samples)

    info = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "n": 0}
    for _ in range(epochs):
        logits, values = net(
            batch["planets"], batch["pmask"],
            batch["fleets"], batch["fmask"], batch["globals"],
            batch["spatial"],
        )
        log_probs_new_list = []
        ent_list = []
        for i in range(B):
            planet_idx_list, cand_idx_list = pick_lists[i]
            if not planet_idx_list:
                log_probs_new_list.append(torch.zeros((), device=device))
                ent_list.append(torch.zeros((), device=device))
                continue
            planet_idx_t = torch.tensor(planet_idx_list, device=device, dtype=torch.long)
            cand_idx_t = torch.tensor(cand_idx_list, device=device, dtype=torch.long)
            per_planet_logits = logits[i, planet_idx_t]        # [n_owned, K]
            log_probs_k = F.log_softmax(per_planet_logits, dim=-1)
            selected_lp = log_probs_k.gather(1, cand_idx_t.unsqueeze(1)).squeeze(1)
            log_probs_new_list.append(selected_lp.sum())
            probs_k = log_probs_k.exp()
            ent = -(probs_k * log_probs_k).sum(dim=-1).mean()
            ent_list.append(ent)

        log_probs_new = torch.stack(log_probs_new_list)
        ent_mean = torch.stack(ent_list).mean()
        ratio = (log_probs_new - log_probs_old).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
        pi_loss = -torch.minimum(surr1, surr2).mean()
        v_loss = F.mse_loss(values, returns)
        loss = pi_loss + val_coef * v_loss - ent_coef * ent_mean

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        info["pi_loss"] += pi_loss.item()
        info["v_loss"] += v_loss.item()
        info["entropy"] += ent_mean.item()
        info["n"] += 1

    for k in ("pi_loss", "v_loss", "entropy"):
        info[k] /= max(1, info["n"])
    return info


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-iters", type=int, default=2000)
    ap.add_argument("--games-per-iter", type=int, default=4)
    ap.add_argument("--four-player-prob", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.03)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-every", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DualStreamCandidateAgent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"[physics_picker] device={device}  "
          f"params={sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"K_PER_SOURCE={K_PER_SOURCE}  ent_coef={args.ent_coef}",
          flush=True)

    def state_dict_bytes():
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
        return buf.getvalue()

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
        tasks = []
        for _ in range(args.games_per_iter):
            n_players = 4 if random.random() < args.four_player_prob else 2
            tasks.append({"n_players": n_players,
                          "picker_seat": random.randint(0, n_players - 1)})
        rollouts = pool.map(_rollout_game, tasks)

        all_samples = []
        wins = 0
        cc_total = [0] * K_PER_SOURCE
        for r in rollouts:
            all_samples.extend(r["samples"])
            if r["win"]: wins += 1
            for i, c in enumerate(r["cand_choice_counts"]):
                cc_total[i] += c
        info = ppo_update(net, opt, all_samples, device, ent_coef=args.ent_coef)

        elapsed = time.time() - t0
        cc_str = " ".join(f"{c}" for c in cc_total)
        print(f"[iter {iter_:04d}] games={args.games_per_iter}  "
              f"samples={len(all_samples)}  wins={wins}/{args.games_per_iter}  "
              f"pi={info.get('pi_loss', 0):.3f}  "
              f"v={info.get('v_loss', 0):.3f}  "
              f"ent={info.get('entropy', 0):.3f}  "
              f"cc=[{cc_str}]  [{elapsed:.0f}s]", flush=True)

        if iter_ % args.snapshot_every == 0:
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, args.out)
            # Also save versioned snapshot (per todo_versioned_checkpoints memory)
            stem = Path(args.out).stem
            versioned = Path(args.out).parent / f"{stem}_iter{iter_:04d}.pt"
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, versioned)
            print(f"[iter {iter_:04d}] saved {args.out} + {versioned.name}",
                  flush=True)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
