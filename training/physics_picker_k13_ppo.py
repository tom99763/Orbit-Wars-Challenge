"""k13 Physics Picker PPO — factored mode × fraction action space.

Replaces k12's monolithic K=12 cand_head with TWO factored heads per planet:
  mode_head:     [d, 5]  → {pass, expand, attack, reinforce, denial}
  fraction_head: [d, 8]  → {5%, 15%, 30%, 50%, 65%, 80%, 95%, 100%}

Per-planet joint action = (mode_idx, frac_idx). Independent softmaxes, so
log π(a) = log π_mode(m) + log π_frac(f) when m != pass, else just log π_mode(pass).

Physics helper in training/physics_action_helper_k13.py picks target by mode,
scales ships by fraction, applies safe-intercept aim via lb1200_agent helpers.

PBRS reward: φ(s) = my_production / total_production (primary)
           + 0.1 × my_ships_fraction (auxiliary, hoarding-safe due to PBRS form).

Usage:
  python training/physics_picker_k13_ppo.py \\
      --warm-start-k12 training/checkpoints/physics_picker_k12_v3.pt \\
      --workers 4 --games-per-iter 8 --replay-iters 3 \\
      --lb928-prob 0.4 --ent-coef 0.01 \\
      --out training/checkpoints/physics_picker_k13.pt
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
from training.model import SetAttentionBlock
from training.dual_stream_model import (SpatialCNN, ScalarMLP, rasterize_obs,
                                         N_SPATIAL_CHANNELS)
from training.lb1200_agent import build_world
from training.physics_action_helper_k13 import (
    N_MODES, N_FRACS, MODE_NAMES, FRACTIONS,
    materialize_joint_action, compute_mode_mask,
)

GRID = 32
GAMMA = 0.99


# -----------------------------------------------------------------------------
# Model — Dual-stream encoder + factored mode + fraction heads
# -----------------------------------------------------------------------------

class DualStreamK13Agent(nn.Module):
    def __init__(
        self,
        planet_dim: int,
        fleet_dim: int,
        global_dim: int,
        n_modes: int = N_MODES,
        n_fracs: int = N_FRACS,
        d_entity: int = 128,
        d_spatial: int = 128,
        d_scalar: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
    ):
        super().__init__()
        self.d = d_entity
        self.n_modes = n_modes
        self.n_fracs = n_fracs

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

        self.spatial_enc = SpatialCNN(N_SPATIAL_CHANNELS, d_spatial, grid=GRID)
        self.scalar_enc = ScalarMLP(global_dim=global_dim, d_model=d_scalar)

        self.fuse_global = nn.Sequential(
            nn.Linear(d_entity + d_spatial + d_scalar, d_entity), nn.GELU(),
            nn.Linear(d_entity, d_entity),
        )
        # Factored heads with mode-conditional fraction
        self.mode_head = nn.Linear(d_entity, n_modes)
        self.mode_embed = nn.Embedding(n_modes, d_entity)   # condition frac on sampled mode
        self.frac_head = nn.Linear(d_entity, n_fracs)
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

        planet_tokens = tokens[:, :P, :]
        entity_global = tokens[:, -1, :]
        spatial_feat = self.spatial_enc(spatial)
        scalar_feat = self.scalar_enc(globals_)
        fused_g = self.fuse_global(
            torch.cat([entity_global, spatial_feat, scalar_feat], dim=-1)
        )
        fused_planet_tokens = planet_tokens + fused_g.unsqueeze(1)
        mode_logits = self.mode_head(fused_planet_tokens)   # [B, P, 5]
        value = self.value_head(fused_g).squeeze(-1)
        # NOTE: frac_logits NOT computed here — call frac_logits_for(fused, mode_idx)
        # to get [*, N_FRACS] conditioned on chosen mode.
        return fused_planet_tokens, mode_logits, value

    def frac_logits_for(self, fused_tokens, mode_idx):
        """Compute frac logits conditional on mode_idx.

        fused_tokens: [..., d_entity]  (may be [B, P, d] or [N_picks, d])
        mode_idx:     [...]            matching leading dims, long
        Returns [..., N_FRACS]
        """
        mode_emb = self.mode_embed(mode_idx)   # broadcasts into last dim
        return self.frac_head(fused_tokens + mode_emb)


# -----------------------------------------------------------------------------
# Warm-start: k12 ckpt → k13 (skip cand_head, random-init mode/frac heads)
# -----------------------------------------------------------------------------

def load_k12_into_k13(net_k13, k12_ckpt_path: str) -> int:
    ckpt = torch.load(k12_ckpt_path, map_location="cpu", weights_only=False)
    k12_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    k13_sd = net_k13.state_dict()
    loaded = 0
    for key, v in k12_sd.items():
        if key in k13_sd and k13_sd[key].shape == v.shape:
            k13_sd[key] = v.clone()
            loaded += 1
    net_k13.load_state_dict(k13_sd, strict=False)
    return loaded


def init_head_biases(net_k13):
    """Strong prior to break reinforce-collapse.

    mode prior (as logit bias):
      pass     : -1.5  (strongly discourage idleness)
      expand   : +1.5  (strongly prefer neutral grab = territory control)
      attack   : +1.0  (encourage — ships-in-combat = PBRS planet gain)
      reinforce: -2.0  (strongly suppress — was collapsing to 78%)
      denial   : +0.5  (mildly encourage)

    fraction prior (prefer 50-80% — Shun_PI sweet spot):
      5%  : -0.5
      15% : -0.2
      30% :  0.0
      50% : +0.4
      65% : +0.4
      80% : +0.3
      95% :  0.0
      100%: -0.3
    """
    with torch.no_grad():
        mode_bias = torch.tensor([-1.5, 1.5, 1.0, -2.0, 0.5])
        frac_bias = torch.tensor([-0.5, -0.2, 0.0, 0.4, 0.4, 0.3, 0.0, -0.3])
        net_k13.mode_head.bias.copy_(mode_bias)
        net_k13.frac_head.bias.copy_(frac_bias)
        # keep Kaiming weights — bias provides the prior, weights let policy be state-dependent


# -----------------------------------------------------------------------------
# Rollout worker — samples (mode, frac) per owned planet, records both idx
# -----------------------------------------------------------------------------

_worker_net: Optional[DualStreamK13Agent] = None


def _worker_init(state_dict_bytes: bytes):
    global _worker_net
    from kaggle_environments import make   # noqa
    _worker_net = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    )
    buf = io.BytesIO(state_dict_bytes)
    sd = torch.load(buf, map_location="cpu", weights_only=False)
    _worker_net.load_state_dict(sd)
    _worker_net.eval()


def _rollout_game(task: dict) -> dict:
    from kaggle_environments import make
    from training.lb1200_agent import agent as lb1200_agent
    try:
        from training.lb928_agent import agent as lb928_agent
    except Exception:
        lb928_agent = lb1200_agent

    _lb928_wrapped = lambda obs, cfg=None: lb928_agent(obs)
    opp_agent = _lb928_wrapped if task.get("use_lb928") else lb1200_agent

    n_players = task["n_players"]
    picker_seat = task["picker_seat"]
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)

    obs_history = collections.deque(maxlen=HISTORY_K)
    action_history = collections.deque(maxlen=HISTORY_K)
    last_actions_by_planet: dict = {}
    cum_stats = {"total_ships_sent": 0, "total_actions": 0}

    samples = []
    ang_vel_init = None; init_planets = None
    step = 0
    prev_prod_phi = 0.0
    prev_ship_phi = 0.0
    mode_counts = [0] * N_MODES
    frac_counts = [0] * N_FRACS

    while not env.done and step < 500:
        actions_all = []
        for s in range(n_players):
            obs = env.state[s].observation
            if s == picker_seat:
                try:
                    world = build_world(obs)
                except Exception:
                    actions_all.append([]); continue
                raw_planets = obs.get("planets", []) or []
                raw_fleets = obs.get("fleets", []) or []
                my_planets_list = [p for p in world.planets if p.owner == s]
                if not my_planets_list:
                    actions_all.append([]); continue

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
                    pl = np.zeros((1, PLANET_DIM), dtype=np.float32); pmask = np.zeros(1, dtype=bool)
                else:
                    pmask = np.ones(pl.shape[0], dtype=bool)
                if fl.ndim < 2 or fl.shape[0] == 0:
                    fl = np.zeros((1, FLEET_DIM), dtype=np.float32); fmask = np.zeros(1, dtype=bool)
                else:
                    fmask = np.ones(fl.shape[0], dtype=bool)

                with torch.no_grad():
                    fused_tokens, mode_logits, v = _worker_net(
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
                log_prob_sum = 0.0
                ent_sum = 0.0
                ent_ct = 0
                for i, pid in enumerate(planet_ids_in_feat):
                    src = next((pl_ for pl_ in world.planets if pl_.id == int(pid)), None)
                    if src is None or src.owner != s: continue
                    # Compute action mask — invalid modes get -inf before softmax
                    mask = compute_mode_mask(src, world, s)
                    m_log = ml_np[i].copy()
                    for k in range(N_MODES):
                        if not mask[k]: m_log[k] = -1e9
                    m_p = np.exp(m_log - m_log.max()); m_p = m_p / m_p.sum()
                    mode_idx = int(np.random.choice(N_MODES, p=m_p))
                    mode_counts[mode_idx] += 1
                    log_prob_sum += float(np.log(m_p[mode_idx] + 1e-9))
                    ent_sum += -float((m_p * np.log(m_p + 1e-9)).sum()); ent_ct += 1
                    if mode_idx == 0:
                        picks.append((int(pid), 0, 0, tuple(mask)))  # pass
                        continue
                    # Compute frac logits conditional on sampled mode
                    with torch.no_grad():
                        fl_cond = _worker_net.frac_logits_for(
                            fused_tokens[0, i],
                            torch.tensor(mode_idx, dtype=torch.long),
                        ).cpu().numpy()
                    f_p = np.exp(fl_cond - fl_cond.max()); f_p = f_p / f_p.sum()
                    frac_idx = int(np.random.choice(N_FRACS, p=f_p))
                    frac_counts[frac_idx] += 1
                    log_prob_sum += float(np.log(f_p[frac_idx] + 1e-9))
                    ent_sum += -float((f_p * np.log(f_p + 1e-9)).sum()); ent_ct += 1
                    picks.append((int(pid), mode_idx, frac_idx, tuple(mask)))

                # materialize takes 3-tuples; strip mask
                action_list = materialize_joint_action(
                    [(p[0], p[1], p[2]) for p in picks], world, s)
                actions_all.append(action_list)

                if picks:
                    samples.append({
                        "feat": feat, "spatial": spatial,
                        "planet_ids": planet_ids_in_feat,
                        "picks": picks,
                        "log_prob_sum": log_prob_sum,
                        "entropy": ent_sum / max(1, ent_ct),
                        "value": float(v.item()),
                        "reward": 0.0,
                        "my_player": s,
                    })

                for mv in action_list:
                    if len(mv) != 3: continue
                    from featurize import nearest_target_index, ship_bucket_idx
                    src_id, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
                    src_planet = next((p for p in raw_planets if int(p[0]) == src_id), None)
                    if src_planet is None: continue
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
                actions_all.append(opp_agent(obs, env.configuration) or [])

        env.step(actions_all)
        step += 1

        # PBRS reward on production fraction + ship fraction
        if samples:
            cur_obs = env.state[picker_seat].observation or {}
            cur_planets = cur_obs.get("planets", []) or []
            cur_fleets  = cur_obs.get("fleets",  []) or []
            my_prod  = sum(p[6] for p in cur_planets if p[1] == picker_seat)
            tot_prod = sum(p[6] for p in cur_planets)
            prod_phi = my_prod / max(1, tot_prod)
            my_ships  = sum(p[5] for p in cur_planets if p[1] == picker_seat)
            my_ships += sum(f[6] for f in cur_fleets   if f[1] == picker_seat)
            tot_ships  = sum(p[5] for p in cur_planets) + sum(f[6] for f in cur_fleets)
            ship_phi = my_ships / max(1, tot_ships)
            r = (prod_phi - GAMMA * prev_prod_phi) + 0.1 * (ship_phi - GAMMA * prev_ship_phi)
            samples[-1]["reward"] = r
            prev_prod_phi = prod_phi
            prev_ship_phi = ship_phi

    if samples:
        terminal = float(env.state[picker_seat].reward or 0)
        samples[-1]["reward"] += terminal

    G = 0.0
    for s_ in reversed(samples):
        G = s_["reward"] + GAMMA * G
        s_["mc_return"] = G

    return {"samples": samples, "picker_seat": picker_seat,
            "win": float(env.state[picker_seat].reward or 0) > 0,
            "opp": "lb928" if task.get("use_lb928") else "lb1200",
            "mode_counts": mode_counts,
            "frac_counts": frac_counts}


# -----------------------------------------------------------------------------
# PPO (factored log_prob: log π_mode + log π_frac when mode != pass)
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
    pick_lists = []  # each: list of (planet_j, mode_idx, frac_idx_or_-1_if_pass)

    for i, s in enumerate(samples):
        f = s["feat"]
        np_ = f["planets"].shape[0]
        nf = f["fleets"].shape[0] if f["fleets"].ndim == 2 else 0
        if np_ > 0: planets[i, :np_] = f["planets"]; pmask[i, :np_] = True
        if nf > 0: fleets[i, :nf] = f["fleets"]; fmask[i, :nf] = True
        globals_[i] = f["globals"]
        spatial[i] = s["spatial"]
        returns[i] = s.get("mc_return", s["reward"])
        values[i] = s["value"]
        log_probs_old[i] = s["log_prob_sum"]
        pid_to_idx = {int(pid): j for j, pid in enumerate(s["planet_ids"])}
        picks_j = []
        for p in s["picks"]:
            pid, mi, fi = p[0], p[1], p[2]
            mask = p[3] if len(p) > 3 else (True,) * N_MODES
            j = pid_to_idx.get(int(pid), -1)
            if j >= 0:
                picks_j.append((j, int(mi), int(fi), mask))
        pick_lists.append(picks_j)

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
        "values_old": torch.from_numpy(values).to(device),   # for value clipping
        "adv": torch.from_numpy(adv.astype(np.float32)).to(device),
        "log_probs_old": torch.from_numpy(log_probs_old).to(device),
        "pick_lists": pick_lists,
    }


def ppo_update(net, opt, samples, device, epochs=4, clip=0.2,
               ent_coef=0.01, val_coef=0.5, val_clip=0.2, frac_ent_w=1.5):
    if not samples: return {}
    batch = _collate_batch(samples, device)
    log_probs_old = batch["log_probs_old"]
    advantages = batch["adv"]
    returns = batch["returns"]
    values_old = batch["values_old"]
    pick_lists = batch["pick_lists"]
    B = len(samples)

    info = {"pi_loss": 0.0, "v_loss": 0.0, "ent_mode": 0.0, "ent_frac": 0.0,
            "ratio_mean": 0.0, "ratio_max": 0.0, "n": 0}
    for _ in range(epochs):
        fused_tokens, mode_logits, values = net(
            batch["planets"], batch["pmask"],
            batch["fleets"], batch["fmask"], batch["globals"],
            batch["spatial"],
        )
        lp_list = []; em_list = []; ef_list = []
        for i in range(B):
            picks_j = pick_lists[i]
            if not picks_j:
                lp_list.append(torch.zeros((), device=device))
                em_list.append(torch.zeros((), device=device))
                ef_list.append(torch.zeros((), device=device)); continue
            p_idx = torch.tensor([p[0] for p in picks_j], device=device, dtype=torch.long)
            m_idx = torch.tensor([p[1] for p in picks_j], device=device, dtype=torch.long)
            f_idx = torch.tensor([p[2] for p in picks_j], device=device, dtype=torch.long)
            # Per-pick mode mask for consistency with rollout sampling
            mode_mask = torch.tensor([[float(x) for x in p[3]] for p in picks_j],
                                     device=device, dtype=torch.float32)  # [n_picks, N_MODES]
            m_logits_pl = mode_logits[i, p_idx]   # [n_picks, N_MODES]
            m_logits_pl = m_logits_pl + (1.0 - mode_mask) * (-1e9)  # mask invalid modes
            # Conditional frac logits — same mode used at rollout time
            fused_pl = fused_tokens[i, p_idx]     # [n_picks, d]
            f_logits_pl = net.frac_logits_for(fused_pl, m_idx)  # [n_picks, N_FRACS]
            m_lp = F.log_softmax(m_logits_pl, dim=-1)
            f_lp = F.log_softmax(f_logits_pl, dim=-1)
            sel_m = m_lp.gather(1, m_idx.unsqueeze(1)).squeeze(1)
            sel_f = f_lp.gather(1, f_idx.unsqueeze(1)).squeeze(1)
            # Only add frac log-prob when mode != pass (mode_idx != 0)
            mask_not_pass = (m_idx != 0).float()
            # Normalize by number of decisions to prevent trajectory ratio explosion
            n_decs = float(mask_not_pass.sum().item() * 2 + (1 - mask_not_pass).sum().item())
            n_decs = max(1.0, n_decs)
            lp_i = (sel_m + mask_not_pass * sel_f).sum() / n_decs
            lp_list.append(lp_i)
            m_probs = m_lp.exp()
            f_probs = f_lp.exp()
            em_list.append(-(m_probs * m_lp).sum(dim=-1).mean())
            ef_list.append(-(f_probs * f_lp).sum(dim=-1).mean())

        lp_new = torch.stack(lp_list)
        em_mean = torch.stack(em_list).mean()
        ef_mean = torch.stack(ef_list).mean()
        ent_total = em_mean + frac_ent_w * ef_mean   # balance (frac has higher max H)
        ratio = (lp_new - log_probs_old).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
        pi_loss = -torch.minimum(surr1, surr2).mean()

        # Clipped value loss (PPO style) — curbs value drift under PBRS
        v_pred_clipped = values_old + (values - values_old).clamp(-val_clip, val_clip)
        v_loss_unclipped = (values - returns) ** 2
        v_loss_clipped   = (v_pred_clipped - returns) ** 2
        v_loss = torch.maximum(v_loss_unclipped, v_loss_clipped).mean()

        loss = pi_loss + val_coef * v_loss - ent_coef * ent_total
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        info["pi_loss"] += pi_loss.item()
        info["v_loss"] += v_loss.item()
        info["ent_mode"] += em_mean.item()
        info["ent_frac"] += ef_mean.item()
        info["ratio_mean"] += ratio.mean().item()
        info["ratio_max"]  = max(info["ratio_max"], ratio.max().item())
        info["n"] += 1
    for k in ("pi_loss", "v_loss", "ent_mode", "ent_frac", "ratio_mean"):
        info[k] /= max(1, info["n"])
    return info


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-start-k12", default="training/checkpoints/physics_picker_k12_v3.pt",
                    help="k12 ckpt to warm-start encoder/value (skips cand_head)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-iters", type=int, default=1000)
    ap.add_argument("--games-per-iter", type=int, default=8)
    ap.add_argument("--replay-iters", type=int, default=3)
    ap.add_argument("--four-player-prob", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--lb928-prob", type=float, default=0.4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-every", type=int, default=10)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    ).to(device)

    if args.warm_start_k12 and Path(args.warm_start_k12).exists():
        loaded = load_k12_into_k13(net, args.warm_start_k12)
        print(f"[k13] warm-started {loaded} tensors from {args.warm_start_k12}"
              f"  (mode/frac heads will be re-biased)", flush=True)
    else:
        print(f"[k13] no warm-start (path missing: {args.warm_start_k12})", flush=True)

    # Override mode/frac head biases to break random-init reinforce-trap
    init_head_biases(net)
    print(f"[k13] mode_bias={net.mode_head.bias.detach().cpu().numpy().round(2).tolist()}"
          f"  (pass, expand, attack, reinforce, denial)", flush=True)
    print(f"[k13] frac_bias={net.frac_head.bias.detach().cpu().numpy().round(2).tolist()}",
          flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"[k13] device={device}  "
          f"params={sum(p.numel() for p in net.parameters())/1e6:.2f}M  "
          f"N_MODES={N_MODES}  N_FRACS={N_FRACS}  "
          f"ent_coef={args.ent_coef}  games/iter={args.games_per_iter}  "
          f"replay_iters={args.replay_iters}  lb928_prob={args.lb928_prob}",
          flush=True)
    print(f"[k13] modes={MODE_NAMES}  fracs={FRACTIONS}", flush=True)

    def sdb():
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in net.state_dict().items()}, buf)
        return buf.getvalue()

    pool = mp.get_context("spawn").Pool(processes=args.workers,
                                         initializer=_worker_init,
                                         initargs=(sdb(),))
    t0 = time.time()
    replay_buf: collections.deque = collections.deque(maxlen=args.replay_iters)

    for iter_ in range(1, args.target_iters + 1):
        if iter_ % 5 == 1:
            pool.close()
            pool = mp.get_context("spawn").Pool(processes=args.workers,
                                                 initializer=_worker_init,
                                                 initargs=(sdb(),))
        tasks = []
        for _ in range(args.games_per_iter):
            n_players = 4 if random.random() < args.four_player_prob else 2
            tasks.append({"n_players": n_players,
                          "picker_seat": random.randint(0, n_players - 1),
                          "use_lb928": random.random() < args.lb928_prob})
        rollouts = pool.map(_rollout_game, tasks)

        iter_samples = []
        wins = 0
        wins_928 = 0; games_928 = 0
        wins_1200 = 0; games_1200 = 0
        mc_iter = [0] * N_MODES
        fc_iter = [0] * N_FRACS
        for r in rollouts:
            iter_samples.extend(r["samples"])
            if r["win"]: wins += 1
            if r.get("opp") == "lb928":
                games_928 += 1
                if r["win"]: wins_928 += 1
            else:
                games_1200 += 1
                if r["win"]: wins_1200 += 1
            for i, c in enumerate(r["mode_counts"]): mc_iter[i] += c
            for i, c in enumerate(r["frac_counts"]): fc_iter[i] += c
        replay_buf.append(iter_samples)
        all_training = [s for chunk in replay_buf for s in chunk]

        info = ppo_update(net, opt, all_training, device, ent_coef=args.ent_coef)

        elapsed = time.time() - t0
        mc_str = " ".join(f"{c}" for c in mc_iter)
        fc_str = " ".join(f"{c}" for c in fc_iter)
        opp_str = ""
        if games_928: opp_str += f"  vs928={wins_928}/{games_928}"
        if games_1200: opp_str += f"  vs1200={wins_1200}/{games_1200}"
        print(f"[iter {iter_:04d}] games={args.games_per_iter}  "
              f"new={len(iter_samples)} train={len(all_training)} "
              f"wins={wins}/{args.games_per_iter}{opp_str}  "
              f"pi={info.get('pi_loss',0):.3f}  v={info.get('v_loss',0):.3f}  "
              f"Hm={info.get('ent_mode',0):.3f}  Hf={info.get('ent_frac',0):.3f}  "
              f"mc=[{mc_str}]  fc=[{fc_str}]  [{elapsed:.0f}s]", flush=True)

        if iter_ % args.snapshot_every == 0:
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, args.out)
            stem = Path(args.out).stem
            versioned = Path(args.out).parent / f"{stem}_iter{iter_:04d}.pt"
            torch.save({"model": net.state_dict(), "iter": iter_,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, versioned)
            print(f"[iter {iter_:04d}] saved {args.out} + {versioned.name}", flush=True)

    pool.close(); pool.join()


if __name__ == "__main__":
    main()
