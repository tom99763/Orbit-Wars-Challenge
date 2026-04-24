"""Wrap a k14/k13 DualStreamK13Agent checkpoint as a kaggle agent(obs) fn.

Used by training/benchmark.py so checkpoints produced by physics_picker_k14_*
(saved as {"model": state_dict, "iter": int, ...}) can be plugged into the
benchmark harness like any other opponent.

Inference strategy: argmax over mode → argmax over target → argmax over
fraction. Deterministic given the checkpoint; sampling determinism would
also require seeding np.random inside each game, which argmax sidesteps.

History tracking mirrors the per-seat `hist` dict maintained by the
vec-env rollout worker: obs_history / action_history deques, per-planet
last-action tuples, cumulative ships-sent / actions counters. Resets when
obs.step regresses (new game in the same agent instance).
"""
from __future__ import annotations

import collections
import pathlib
from typing import Callable

import numpy as np
import torch

from featurize import (
    FLEET_DIM, GLOBAL_DIM, HISTORY_K, PLANET_DIM,
    featurize_step, nearest_target_index, ship_bucket_idx,
)
from training.dual_stream_model import rasterize_obs
from training.lb1200_agent import build_world
from training.physics_action_helper_k13 import (
    CAND_FEAT_DIM, FRACTIONS, N_FRACS, N_MODES, TOP_K_TARGETS,
    compute_mode_mask, get_top_k_candidates, materialize_with_targets,
)
from training.physics_picker_k13_ppo import GRID, DualStreamK13Agent


def _new_session() -> dict:
    return {
        "obs_history": collections.deque(maxlen=HISTORY_K),
        "action_history": collections.deque(maxlen=HISTORY_K),
        "last_actions_by_planet": {},
        "cum_stats": {"total_ships_sent": 0, "total_actions": 0},
        "last_step": -1,
    }


def load_k14_agent(
    ckpt_path: str,
    device: str = "cpu",
    temperature: float = 0.0,
) -> Callable[[dict], list]:
    """Load a DualStreamK13Agent checkpoint as `agent(obs) → list[moves]`.

    temperature is accepted for API parity with training.agent_v4.load_agent
    but is currently ignored — inference is argmax. Wire it through if you
    need stochastic eval later.
    """
    net = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(sd, strict=True)
    net.to(device).eval()

    session = _new_session()

    def _pick_one_game_step(obs: dict) -> list:
        raw_planets = obs.get("planets", []) or []
        raw_fleets = obs.get("fleets", []) or []
        seat = int(obs.get("player", 0) or 0)
        step = int(obs.get("step", 0) or 0)

        if step < session["last_step"]:
            # Game reset → clear history for this agent instance
            session.update(_new_session())
        session["last_step"] = step

        if not raw_planets:
            return []

        try:
            world = build_world(obs)
        except Exception:
            return []

        my_planets = [p for p in world.planets if p.owner == seat]
        if not my_planets:
            return []

        ang_vel = float(obs.get("angular_velocity", 0.02) or 0.02)
        initial_planets = obs.get("initial_planets", []) or []
        owners = {int(p[1]) for p in raw_planets if int(p[1]) >= 0}
        owners |= {int(f[1]) for f in raw_fleets if int(f[1]) >= 0}
        n_players = 4 if len(owners) > 2 else 2

        step_dict = {
            "step": step,
            "planets": list(raw_planets),
            "fleets": list(raw_fleets),
            "action": [],
            "my_total_ships": sum(p[5] for p in raw_planets if p[1] == seat),
            "enemy_total_ships": 0,
            "my_planet_count": 0,
            "enemy_planet_count": 0,
            "neutral_planet_count": 0,
        }
        feat = featurize_step(
            step_dict, seat, ang_vel, n_players, initial_planets,
            last_actions_by_planet=session["last_actions_by_planet"],
            cumulative_stats=session["cum_stats"],
            obs_history=list(session["obs_history"]),
            action_history=list(session["action_history"]),
        )
        spatial = rasterize_obs(obs, seat, grid=GRID)

        pl = feat["planets"]
        fl = feat.get("fleets", None)
        if pl.shape[0] == 0:
            pl = np.zeros((1, PLANET_DIM), dtype=np.float32)
            pmask = np.zeros(1, dtype=bool)
        else:
            pmask = np.ones(pl.shape[0], dtype=bool)
        if fl is None or fl.ndim < 2 or fl.shape[0] == 0:
            fl = np.zeros((1, FLEET_DIM), dtype=np.float32)
            fmask = np.zeros(1, dtype=bool)
        else:
            fmask = np.ones(fl.shape[0], dtype=bool)

        with torch.no_grad():
            fused_tokens, mode_logits, _v = net(
                torch.from_numpy(pl).unsqueeze(0).to(device),
                torch.from_numpy(pmask).unsqueeze(0).to(device),
                torch.from_numpy(fl).unsqueeze(0).to(device),
                torch.from_numpy(fmask).unsqueeze(0).to(device),
                torch.from_numpy(feat["globals"]).unsqueeze(0).to(device),
                torch.from_numpy(spatial).unsqueeze(0).to(device),
            )
        ml_np = mode_logits[0].cpu().numpy()   # (P, N_MODES)
        planet_ids = feat.get("planet_ids", [])

        picks_tuples = []   # (pid, mode, frac, tgt_idx)
        committed: dict = {}

        for i, pid in enumerate(planet_ids):
            src = next((p for p in world.planets if p.id == int(pid)), None)
            if src is None or src.owner != seat:
                continue
            mask = compute_mode_mask(src, world, seat)
            m_log = ml_np[i].copy()
            for k in range(N_MODES):
                if not mask[k]:
                    m_log[k] = -1e9
            mode_idx = int(np.argmax(m_log))
            if mode_idx == 0:   # pass
                continue

            cands, cand_feats, n_valid = get_top_k_candidates(
                src, world, seat, mode_idx,
                fleets_raw=raw_fleets, committed=committed,
            )
            if n_valid == 0:
                continue

            cf_t = torch.from_numpy(cand_feats).unsqueeze(0).to(device)
            with torch.no_grad():
                t_scores = net.target_logits_for(
                    fused_tokens[0, i].unsqueeze(0), cf_t
                )[0].cpu().numpy()
            t_scores[n_valid:] = -1e9
            tgt_idx = int(np.argmax(t_scores))
            target_pid = cands[tgt_idx].id

            with torch.no_grad():
                f_logits = net.frac_logits_for(
                    fused_tokens[0, i],
                    torch.tensor(mode_idx, dtype=torch.long, device=device),
                ).cpu().numpy()
            frac_idx = int(np.argmax(f_logits))

            ships_sent = max(1, int(src.ships * FRACTIONS[frac_idx]))
            committed[target_pid] = committed.get(target_pid, 0) + ships_sent
            picks_tuples.append((int(pid), mode_idx, frac_idx, tgt_idx))

        if not picks_tuples:
            return []

        action_list = materialize_with_targets(picks_tuples, world, seat) or []

        # Update history (mirrors physics_picker_k14_vec history bookkeeping)
        session["obs_history"].append({"planets": raw_planets, "step": step})
        for mv in action_list:
            if len(mv) != 3:
                continue
            src_id, ang_rad, ships = int(mv[0]), float(mv[1]), int(mv[2])
            src_p = next((p for p in raw_planets if int(p[0]) == src_id), None)
            if src_p is None:
                continue
            ti = nearest_target_index(src_p, ang_rad, raw_planets)
            tpid = int(raw_planets[ti][0]) if ti is not None else -1
            garrison = int(src_p[5]) + ships
            bi = ship_bucket_idx(ships, max(1, garrison))
            prev_ = session["last_actions_by_planet"].get(src_id, (-1, 0, -1, 0))
            session["last_actions_by_planet"][src_id] = (tpid, bi, step, prev_[3] + 1)
            session["cum_stats"]["total_ships_sent"] += ships
            session["cum_stats"]["total_actions"] += 1
            session["action_history"].append((src_id, tpid, bi, step))

        return action_list

    def agent(obs, config=None):
        try:
            return _pick_one_game_step(obs)
        except Exception:
            return []

    return agent


def is_k14_checkpoint(ckpt_path: str | pathlib.Path) -> bool:
    """Cheap heuristic: file is a k14/k13 ckpt if it has `mode_head.weight`
    but no `kwargs` key (the agent_v4 loader requires kwargs)."""
    try:
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not isinstance(ck, dict):
        return False
    if "kwargs" in ck:
        return False
    sd = ck.get("model", ck)
    if not isinstance(sd, dict):
        return False
    return "mode_head.weight" in sd
