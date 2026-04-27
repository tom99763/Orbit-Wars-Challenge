"""RL + rule hybrid agent for Orbit Wars.

Builds on the observation that lb1200 has very strong physics (lead-
prediction aim, sun-avoidance, target valuation) but its strategic
"when to act, when to hold" decision is rigid — while k14's neural net
may have better strategic intuition but weaker low-level execution.

The hybrid lets each side play to its strength:

  Strategy "lb1200_with_k14_veto" (default):
      lb1200 proposes actions; for each proposed action, query the k14
      mode head for the source planet. If k14's argmax mode is 0 (pass),
      drop the action — i.e. k14 is a binary act/pass gate over
      lb1200's full action set. Tactical execution stays with lb1200.

  Strategy "k14_with_lb1200_fill":
      k14 plays normally. For any owned planet where k14 chose pass but
      lb1200 had an action ready, fall back to lb1200's action.
      (Inverse polarity: k14 leads, lb1200 fills gaps.)

Public API:
  load_hybrid_agent(ckpt_path, strategy=..., device="cpu") → agent(obs)

The returned callable matches Kaggle's `agent(obs, config=None)` shape
and is plug-compatible with training/benchmark.py and a Kaggle
submission wrapper.
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
from training.lb1200_agent import agent as lb1200_agent, build_world
from training.physics_action_helper_k13 import (
    FRACTIONS, N_FRACS, N_MODES, TOP_K_TARGETS,
    compute_mode_mask, get_top_k_candidates, materialize_with_targets,
)
from training.physics_picker_k13_ppo import GRID, DualStreamK13Agent


# Tag sentinels for clarity in logs / diagnostics
STRATEGIES = ("lb1200_with_k14_veto", "k14_with_lb1200_fill")


def _new_session() -> dict:
    return {
        "obs_history": collections.deque(maxlen=HISTORY_K),
        "action_history": collections.deque(maxlen=HISTORY_K),
        "last_actions_by_planet": {},
        "cum_stats": {"total_ships_sent": 0, "total_actions": 0},
        "last_step": -1,
        # Lightweight diagnostic counters — useful for inspecting how
        # often each branch fires during a benchmark run
        "n_lb_proposed": 0,
        "n_lb_vetoed": 0,
        "n_k14_acted": 0,
        "n_lb_filled": 0,
    }


def _featurize_and_forward(
    net: DualStreamK13Agent,
    obs: dict,
    seat: int,
    session: dict,
    device: str,
) -> tuple | None:
    """Run k14's full forward pass once. Returns
        (fused_planet_tokens, mode_logits[0], world, planet_ids)
    or None if obs is malformed / agent has no owned planets.
    """
    raw_planets = obs.get("planets", []) or []
    raw_fleets = obs.get("fleets", []) or []
    if not raw_planets:
        return None

    try:
        world = build_world({**obs, "player": seat})
    except Exception:
        return None
    my_planets = [p for p in world.planets if p.owner == seat]
    if not my_planets:
        return None

    ang_vel = float(obs.get("angular_velocity", 0.02) or 0.02)
    initial_planets = obs.get("initial_planets", []) or []
    owners = {int(p[1]) for p in raw_planets if int(p[1]) >= 0}
    owners |= {int(f[1]) for f in raw_fleets if int(f[1]) >= 0}
    n_players = 4 if len(owners) > 2 else 2
    step = int(obs.get("step", 0) or 0)

    step_dict = {
        "step": step, "planets": list(raw_planets), "fleets": list(raw_fleets),
        "action": [],
        "my_total_ships": sum(p[5] for p in raw_planets if p[1] == seat),
        "enemy_total_ships": 0, "my_planet_count": 0,
        "enemy_planet_count": 0, "neutral_planet_count": 0,
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

    planet_ids = list(feat.get("planet_ids", []))
    return fused_tokens, mode_logits[0].cpu().numpy(), world, planet_ids


def _k14_argmax_mode_for_planet(
    src_planet, world, my_player: int,
    mode_logits_np: np.ndarray, planet_ids: list[int], src_id: int,
) -> int:
    """Return the k14 mode_argmax for one source planet, applying the
    standard mode mask so invalid modes can't be picked."""
    try:
        idx = planet_ids.index(int(src_id))
    except ValueError:
        return 0   # not in the policy's view → safest default is pass
    mask = compute_mode_mask(src_planet, world, my_player)
    m_log = mode_logits_np[idx].copy()
    for k in range(N_MODES):
        if not mask[k]:
            m_log[k] = -1e9
    return int(np.argmax(m_log))


def _build_k14_actions(
    fused_tokens, mode_logits_np: np.ndarray, world, planet_ids: list[int],
    seat: int, raw_fleets: list, net: DualStreamK13Agent, device: str,
) -> list:
    """Mirror training/k14_agent_wrapper.py argmax inference: produce
    k14's full action list for the current state."""
    picks_tuples = []
    committed: dict = {}
    planet_by_id = {p.id: p for p in world.planets}

    for i, pid in enumerate(planet_ids):
        src = planet_by_id.get(int(pid))
        if src is None or src.owner != seat:
            continue
        mask = compute_mode_mask(src, world, seat)
        m_log = mode_logits_np[i].copy()
        for k in range(N_MODES):
            if not mask[k]:
                m_log[k] = -1e9
        mode_idx = int(np.argmax(m_log))
        if mode_idx == 0:
            continue
        cands, _feats, n_valid = get_top_k_candidates(
            src, world, seat, mode_idx,
            fleets_raw=raw_fleets, committed=committed,
        )
        if n_valid == 0:
            continue
        cf_t = torch.from_numpy(_feats).unsqueeze(0).to(device)
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

    # Convert tgt_idx to target_pid (materialize_with_targets needs target_pid)
    full_picks = []
    committed = {}
    for (pid, mode_idx, frac_idx, tgt_idx) in picks_tuples:
        src = planet_by_id[pid]
        cands, _, n_valid = get_top_k_candidates(
            src, world, seat, mode_idx,
            fleets_raw=raw_fleets, committed=committed,
        )
        if tgt_idx >= n_valid:
            continue
        target_pid = cands[tgt_idx].id
        full_picks.append((pid, mode_idx, frac_idx, target_pid))
        ships_sent = max(1, int(src.ships * FRACTIONS[frac_idx]))
        committed[target_pid] = committed.get(target_pid, 0) + ships_sent

    return materialize_with_targets(full_picks, world, seat) or []


def load_hybrid_agent(
    ckpt_path: str | pathlib.Path,
    strategy: str = "lb1200_with_k14_veto",
    device: str = "cpu",
) -> Callable:
    """Load a k14 ckpt and return a hybrid `agent(obs, config=None)` fn.

    `strategy` selects how k14 and lb1200 combine — see module docstring.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; "
                         f"choose from {STRATEGIES}")

    net = DualStreamK13Agent(
        planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
    )
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(sd, strict=True)
    net.to(device).eval()

    session = _new_session()

    def _update_history(action_list, raw_planets, step):
        for mv in action_list:
            if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
                continue
            try:
                src_id = int(mv[0]); ang = float(mv[1]); ships = int(mv[2])
            except (TypeError, ValueError):
                continue
            src_p = next((p for p in raw_planets if int(p[0]) == src_id), None)
            if src_p is None:
                continue
            ti = nearest_target_index(src_p, ang, raw_planets)
            tpid = int(raw_planets[ti][0]) if ti is not None else -1
            garrison = int(src_p[5]) + ships
            bi = ship_bucket_idx(ships, max(1, garrison))
            prev = session["last_actions_by_planet"].get(src_id, (-1, 0, -1, 0))
            session["last_actions_by_planet"][src_id] = (tpid, bi, step, prev[3] + 1)
            session["cum_stats"]["total_ships_sent"] += ships
            session["cum_stats"]["total_actions"] += 1
            session["action_history"].append((src_id, tpid, bi, step))
        session["obs_history"].append({"planets": raw_planets, "step": step})

    def agent(obs, config=None):
        try:
            seat = int(obs.get("player", 0) or 0)
            step = int(obs.get("step", 0) or 0)
            if step < session["last_step"]:
                # Game reset — clear all per-game state including counters
                session.update(_new_session())
            session["last_step"] = step

            raw_planets = obs.get("planets", []) or []
            fwd = _featurize_and_forward(net, obs, seat, session, device)
            if fwd is None:
                # No owned planets / malformed obs → just trust lb1200
                action_list = lb1200_agent(obs, config) or []
                _update_history(action_list, raw_planets, step)
                return action_list
            fused_tokens, mode_logits_np, world, planet_ids = fwd
            planet_by_id = {p.id: p for p in world.planets}

            if strategy == "lb1200_with_k14_veto":
                proposed = lb1200_agent(obs, config) or []
                kept = []
                for mv in proposed:
                    if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
                        continue
                    try:
                        src_id = int(mv[0])
                    except (TypeError, ValueError):
                        continue
                    src = planet_by_id.get(src_id)
                    if src is None or src.owner != seat:
                        continue
                    session["n_lb_proposed"] += 1
                    mode = _k14_argmax_mode_for_planet(
                        src, world, seat, mode_logits_np, planet_ids, src_id,
                    )
                    if mode == 0:
                        session["n_lb_vetoed"] += 1
                        continue
                    kept.append(mv)
                _update_history(kept, raw_planets, step)
                return kept

            # k14_with_lb1200_fill
            k14_actions = _build_k14_actions(
                fused_tokens, mode_logits_np, world, planet_ids,
                seat, obs.get("fleets", []) or [], net, device,
            )
            session["n_k14_acted"] += len(k14_actions)
            k14_srcs = {int(mv[0]) for mv in k14_actions
                        if isinstance(mv, (list, tuple)) and len(mv) == 3}
            lb_actions = lb1200_agent(obs, config) or []
            fillers = []
            for mv in lb_actions:
                if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
                    continue
                try:
                    src_id = int(mv[0])
                except (TypeError, ValueError):
                    continue
                if src_id in k14_srcs:
                    continue   # k14 already acts on this planet
                src = planet_by_id.get(src_id)
                if src is None or src.owner != seat:
                    continue
                fillers.append(mv)
                session["n_lb_filled"] += 1

            combined = list(k14_actions) + fillers
            _update_history(combined, raw_planets, step)
            return combined
        except Exception:
            # Last-resort fallback: don't crash the game over a wrapper bug
            return lb1200_agent(obs, config) or []

    # Expose the session dict for inspection (benchmark harness can read it)
    agent._session = session   # type: ignore[attr-defined]
    return agent
