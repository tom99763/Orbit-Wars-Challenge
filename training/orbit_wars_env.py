"""TorchRL EnvBase wrapper for Orbit Wars — multi-agent, padded fixed-shape obs.

Key design decisions:
  - All N agents return obs/action at every step → multi-agent IPPO
  - In self-play:  both agents use trainable policy, both contribute gradients
  - In pool games: agent[picker_seat] trains; others run frozen policy internally
  - Featurization (featurize_step + rasterize_obs) happens inside the env
  - PBRS reward computed per-agent inside _step
  - Obs padded to MAX_PLANETS / MAX_FLEETS for fixed-shape TensorDict specs
"""
from __future__ import annotations

import collections
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import math
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
)

from featurize import (
    featurize_step, PLANET_DIM, FLEET_DIM, GLOBAL_DIM, HISTORY_K,
)
from training.dual_stream_model import rasterize_obs, N_SPATIAL_CHANNELS
from training.physics_action_helper_k13 import (
    N_MODES, N_FRACS, TOP_K_TARGETS, CAND_FEAT_DIM,
    materialize_with_targets,
)

GRID = 32
GAMMA = 0.99
GAE_LAMBDA = 0.95

MAX_PLANETS = 40
MAX_FLEETS  = 100

# ──────────────────────────────────────────────────────────────────────────────
# Observation helper: featurize one agent's raw obs into fixed-size tensors
# ──────────────────────────────────────────────────────────────────────────────

def _featurize_agent(obs: dict, seat: int, n_players: int,
                     ang_vel: float, init_planets: list,
                     history: dict) -> dict:
    """Return dict of numpy arrays with fixed shapes (padded)."""
    raw_planets = obs.get("planets", []) or []
    raw_fleets  = obs.get("fleets",  []) or []
    step_num    = int(obs.get("step", 0) or 0)

    step_dict = {
        "step": step_num,
        "planets": raw_planets,
        "fleets":  raw_fleets,
        "action": [],
        "my_total_ships":    sum(p[5] for p in raw_planets if p[1] == seat),
        "enemy_total_ships": sum(p[5] for p in raw_planets if p[1] != seat and p[1] != -1),
        "my_planet_count":    sum(1  for p in raw_planets if p[1] == seat),
        "enemy_planet_count": sum(1  for p in raw_planets if p[1] != seat and p[1] != -1),
        "neutral_planet_count": sum(1 for p in raw_planets if p[1] == -1),
    }
    feat = featurize_step(
        step_dict, seat, ang_vel, n_players, init_planets,
        last_actions_by_planet=history["last_actions_by_planet"],
        cumulative_stats=history["cum_stats"],
        obs_history=list(history["obs_history"]),
        action_history=list(history["action_history"]),
    )
    spatial = rasterize_obs(obs, seat, grid=GRID)

    P = feat["planets"].shape[0]
    F = feat["fleets"].shape[0] if feat["fleets"].ndim == 2 else 0

    # Pad planets → [MAX_PLANETS, PLANET_DIM]
    pl_pad  = np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32)
    pm_pad  = np.zeros(MAX_PLANETS, dtype=bool)
    id_pad  = np.full(MAX_PLANETS, -1, dtype=np.int64)
    if P > 0:
        n = min(P, MAX_PLANETS)
        pl_pad[:n]  = feat["planets"][:n]
        pm_pad[:n]  = True
        ids = feat.get("planet_ids", [])
        if ids:
            id_pad[:len(ids)] = [int(x) for x in ids[:MAX_PLANETS]]

    # Pad fleets → [MAX_FLEETS, FLEET_DIM]
    fl_pad = np.zeros((MAX_FLEETS, FLEET_DIM), dtype=np.float32)
    fm_pad = np.zeros(MAX_FLEETS, dtype=bool)
    if F > 0:
        n = min(F, MAX_FLEETS)
        fl_pad[:n] = feat["fleets"][:n]
        fm_pad[:n] = True

    return {
        "planets":      pl_pad,
        "planet_mask":  pm_pad,
        "fleets":       fl_pad,
        "fleet_mask":   fm_pad,
        "globals":      feat["globals"].astype(np.float32),
        "spatial":      spatial.astype(np.float32),
        "planet_ids":   id_pad,
        "raw_fleets":   raw_fleets,  # kept for fleet-race features (not in spec)
    }


def _pbrs(obs: dict, seat: int, n_players: int,
          prev: dict, gamma: float = GAMMA) -> tuple[float, dict]:
    """Compute PBRS reward for one agent. Returns (r, new_prev)."""
    planets = obs.get("planets", []) or []
    fleets  = obs.get("fleets",  []) or []
    my_prod  = sum(p[6] for p in planets if p[1] == seat)
    tot_prod = sum(p[6] for p in planets)
    prod_phi = my_prod / max(1, tot_prod)

    my_ships   = sum(p[5] for p in planets if p[1] == seat)
    my_ships  += sum(f[6] for f in fleets   if f[1] == seat)
    tot_ships  = sum(p[5] for p in planets) + sum(f[6] for f in fleets)
    ship_phi   = my_ships / max(1, tot_ships)

    enemy_ships  = sum(p[5] for p in planets if p[1] != seat and p[1] != -1)
    enemy_ships += sum(f[6] for f in fleets   if f[1] != seat)
    enemy_ship_phi = enemy_ships / max(1, tot_ships)

    r = ((prod_phi           - gamma * prev["prod_phi"])
         + 0.5 * (ship_phi       - gamma * prev["ship_phi"])
         - 0.3 * (enemy_ship_phi - gamma * prev["enemy_ship_phi"]))

    new_prev = {"prod_phi": prod_phi, "ship_phi": ship_phi,
                "enemy_ship_phi": enemy_ship_phi}
    return float(r), new_prev


# ──────────────────────────────────────────────────────────────────────────────
# OrbitWarsEnv
# ──────────────────────────────────────────────────────────────────────────────

class OrbitWarsEnv(EnvBase):
    """Single Orbit Wars game, multi-agent (2P default).

    Observation structure (per agent, shape [n_agents, ...]):
        planets      [MAX_PLANETS, PLANET_DIM]
        planet_mask  [MAX_PLANETS]
        fleets       [MAX_FLEETS, FLEET_DIM]
        fleet_mask   [MAX_FLEETS]
        globals      [GLOBAL_DIM]
        spatial      [N_SPATIAL_CHANNELS, GRID, GRID]
        planet_ids   [MAX_PLANETS]  int64 (-1 = padding)

    Action structure (per agent):
        planet_mode_idx   [MAX_PLANETS]  int64
        planet_frac_idx   [MAX_PLANETS]  int64
        planet_tgt_idx    [MAX_PLANETS]  int64
        planet_cand_feats [MAX_PLANETS, TOP_K_TARGETS, CAND_FEAT_DIM]
        planet_n_valid    [MAX_PLANETS]  int64
        planet_mode_mask  [MAX_PLANETS, N_MODES]  bool
        planet_is_owned   [MAX_PLANETS]  bool

    Extras stored in step TensorDict:
        log_prob   [n_agents]   scalar per agent (set by policy)
        entropy    [n_agents]   scalar per agent (set by policy)
        value      [n_agents]   scalar per agent (set by policy)
        training_mask [n_agents] float (1 = trains, 0 = frozen opponent)

    The policy fills action + log_prob + entropy + value.
    Frozen opponent actions are computed by env internally (in _step) based on
    `opponent_type` passed via reset TensorDict.
    """

    def __init__(self, n_players: int = 2, device: str = "cpu",
                 seed: int | None = None):
        super().__init__(device=device, batch_size=[])
        self.n_players  = n_players
        self._make_spec()
        if seed is not None:
            self.set_seed(seed)
        self._reset_state()

    # ── specs ─────────────────────────────────────────────────────────────────

    def _make_spec(self):
        n = self.n_players
        obs_spec = CompositeSpec({
            "planets":      UnboundedContinuousTensorSpec([n, MAX_PLANETS, PLANET_DIM]),
            "planet_mask":  BinaryDiscreteTensorSpec(n=[n, MAX_PLANETS], dtype=torch.bool),
            "fleets":       UnboundedContinuousTensorSpec([n, MAX_FLEETS, FLEET_DIM]),
            "fleet_mask":   BinaryDiscreteTensorSpec(n=[n, MAX_FLEETS], dtype=torch.bool),
            "globals":      UnboundedContinuousTensorSpec([n, GLOBAL_DIM]),
            "spatial":      UnboundedContinuousTensorSpec([n, N_SPATIAL_CHANNELS, GRID, GRID]),
            "planet_ids":   UnboundedDiscreteTensorSpec([n, MAX_PLANETS], dtype=torch.int64),
        }, device=self.device)

        act_spec = CompositeSpec({
            "planet_mode_idx":   UnboundedDiscreteTensorSpec([n, MAX_PLANETS], dtype=torch.int64),
            "planet_frac_idx":   UnboundedDiscreteTensorSpec([n, MAX_PLANETS], dtype=torch.int64),
            "planet_tgt_idx":    UnboundedDiscreteTensorSpec([n, MAX_PLANETS], dtype=torch.int64),
            "planet_cand_feats": UnboundedContinuousTensorSpec(
                                     [n, MAX_PLANETS, TOP_K_TARGETS, CAND_FEAT_DIM]),
            "planet_n_valid":    UnboundedDiscreteTensorSpec([n, MAX_PLANETS], dtype=torch.int64),
            "planet_mode_mask":  BinaryDiscreteTensorSpec(n=[n, MAX_PLANETS, N_MODES],
                                                          dtype=torch.bool),
            "planet_is_owned":   BinaryDiscreteTensorSpec(n=[n, MAX_PLANETS], dtype=torch.bool),
        }, device=self.device)

        rew_spec = UnboundedContinuousTensorSpec([n], device=self.device)
        done_spec = BinaryDiscreteTensorSpec(n=1, dtype=torch.bool, device=self.device)

        self.observation_spec = obs_spec
        self.action_spec      = act_spec
        self.reward_spec      = rew_spec
        self.done_spec        = done_spec

    # ── internal state ────────────────────────────────────────────────────────

    def _reset_state(self):
        self._env          = None
        self._step_num     = 0
        self._ang_vel      = 0.0
        self._init_planets = []
        self._histories    = [
            {
                "obs_history":           collections.deque(maxlen=HISTORY_K),
                "action_history":        collections.deque(maxlen=HISTORY_K),
                "last_actions_by_planet": {},
                "cum_stats":             {"total_ships_sent": 0, "total_actions": 0},
            }
            for _ in range(self.n_players)
        ]
        self._prev_phi = [
            {"prod_phi": 0.0, "ship_phi": 0.0, "enemy_ship_phi": 0.0}
            for _ in range(self.n_players)
        ]
        # Opponent config — set by _reset via reset TensorDict
        self._opponent_type   = "stochastic_self"  # default: all agents train
        self._opponent_fn     = None   # callable(obs, seat) -> action_list
        self._picker_seat     = 0      # main trainable agent seat
        self._training_mask   = torch.ones(self.n_players, dtype=torch.float32)

    # ── reset ─────────────────────────────────────────────────────────────────

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        from kaggle_environments import make

        self._reset_state()

        # Extract opponent config from reset tensordict if provided
        if tensordict is not None:
            if "opponent_type" in tensordict.keys():
                self._opponent_type = tensordict["opponent_type"]
            if "picker_seat" in tensordict.keys():
                self._picker_seat = int(tensordict["picker_seat"].item())
            if "opponent_fn" in tensordict.keys():
                self._opponent_fn = tensordict["opponent_fn"]
            if "training_mask" in tensordict.keys():
                self._training_mask = tensordict["training_mask"].float()

        # Build training mask: 1.0 for all seats in self-play, only picker_seat otherwise
        if self._opponent_type == "stochastic_self":
            self._training_mask = torch.ones(self.n_players, dtype=torch.float32)
        else:
            mask = torch.zeros(self.n_players, dtype=torch.float32)
            mask[self._picker_seat] = 1.0
            self._training_mask = mask

        self._env = make("orbit_wars", debug=False)
        self._env.reset(num_agents=self.n_players)

        obs_list = []
        for seat in range(self.n_players):
            obs = self._env.state[seat].observation or {}
            if self._ang_vel == 0.0:
                self._ang_vel      = float(obs.get("angular_velocity", 0.0) or 0.0)
                self._init_planets = obs.get("initial_planets", []) or []
            obs_list.append(self._featurize(obs, seat))

        td = self._obs_list_to_td(obs_list)
        td["training_mask"] = self._training_mask.to(self.device)
        return td

    # ── step ──────────────────────────────────────────────────────────────────

    def _step(self, tensordict: TensorDict) -> TensorDict:
        actions_all = []
        for seat in range(self.n_players):
            if (self._opponent_type == "stochastic_self"
                    or seat == self._picker_seat
                    or self._training_mask[seat] > 0):
                # Action comes from the policy (already in tensordict)
                action_list = self._td_action_to_list(tensordict, seat)
            else:
                # Frozen opponent — compute action internally
                obs = self._env.state[seat].observation or {}
                action_list = self._frozen_action(obs, seat)
            actions_all.append(action_list)
            self._update_history(seat, actions_all[-1])

        self._env.step(actions_all)
        self._step_num += 1

        done = bool(self._env.done)

        obs_list = []
        rewards  = []
        for seat in range(self.n_players):
            obs = self._env.state[seat].observation or {}
            obs_list.append(self._featurize(obs, seat))

            # PBRS reward
            r, new_prev = _pbrs(obs, seat, self.n_players, self._prev_phi[seat])
            self._prev_phi[seat] = new_prev

            # Terminal reward
            if done:
                terminal = float(self._env.state[seat].reward or 0)
                r += terminal
            rewards.append(r)

        td = self._obs_list_to_td(obs_list)
        td["reward"]        = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        td["done"]          = torch.tensor([done], dtype=torch.bool, device=self.device)
        td["terminated"]    = torch.tensor([done], dtype=torch.bool, device=self.device)
        td["training_mask"] = self._training_mask.to(self.device)
        return td

    # ── seed ──────────────────────────────────────────────────────────────────

    def _set_seed(self, seed: int | None):
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _featurize(self, obs: dict, seat: int) -> dict:
        if not obs:
            # Empty obs (game over for this seat) — return zeros
            return {
                "planets":     np.zeros((MAX_PLANETS, PLANET_DIM), dtype=np.float32),
                "planet_mask": np.zeros(MAX_PLANETS, dtype=bool),
                "fleets":      np.zeros((MAX_FLEETS,  FLEET_DIM),  dtype=np.float32),
                "fleet_mask":  np.zeros(MAX_FLEETS,  dtype=bool),
                "globals":     np.zeros(GLOBAL_DIM,  dtype=np.float32),
                "spatial":     np.zeros((N_SPATIAL_CHANNELS, GRID, GRID), dtype=np.float32),
                "planet_ids":  np.full(MAX_PLANETS, -1, dtype=np.int64),
                "raw_fleets":  [],
            }
        return _featurize_agent(
            obs, seat, self.n_players,
            self._ang_vel, self._init_planets,
            self._histories[seat],
        )

    def _obs_list_to_td(self, obs_list: list[dict]) -> TensorDict:
        """Stack per-agent obs dicts into a TensorDict with shape []."""
        def stack(key, dtype):
            return torch.from_numpy(
                np.stack([o[key] for o in obs_list], axis=0)
            ).to(dtype=dtype, device=self.device)

        return TensorDict({
            "planets":     stack("planets",     torch.float32),
            "planet_mask": stack("planet_mask", torch.bool),
            "fleets":      stack("fleets",      torch.float32),
            "fleet_mask":  stack("fleet_mask",  torch.bool),
            "globals":     stack("globals",     torch.float32),
            "spatial":     stack("spatial",     torch.float32),
            "planet_ids":  stack("planet_ids",  torch.int64),
        }, batch_size=[], device=self.device)

    def _td_action_to_list(self, td: TensorDict, seat: int) -> list:
        """Convert TensorDict action for one seat to raw game action list."""
        obs = self._env.state[seat].observation or {}
        if not obs:
            return []

        from training.lb1200_agent import build_world

        try:
            world = build_world(obs)
        except Exception:
            return []

        mode_idx  = td["planet_mode_idx"][seat].cpu().numpy()   # [MAX_PLANETS]
        frac_idx  = td["planet_frac_idx"][seat].cpu().numpy()
        tgt_idx   = td["planet_tgt_idx"][seat].cpu().numpy()
        n_valid   = td["planet_n_valid"][seat].cpu().numpy()
        cf        = td["planet_cand_feats"][seat].cpu().numpy()  # [MAX_PLANETS, K, D]
        ids       = td["planet_ids"][seat].cpu().numpy()         # [MAX_PLANETS]
        is_owned  = td["planet_is_owned"][seat].cpu().numpy()    # [MAX_PLANETS]
        raw_fleets = obs.get("fleets", []) or []

        picks = []
        for i in range(MAX_PLANETS):
            if not is_owned[i] or ids[i] < 0:
                continue
            pid = int(ids[i])
            mi  = int(mode_idx[i])
            if mi == 0:
                continue
            # Recover target planet id from cand_feats + tgt_idx
            from training.physics_action_helper_k13 import get_top_k_candidates
            from training.lb1200_agent import build_world as _bw
            src = next((p for p in world.planets if p.id == pid), None)
            if src is None:
                continue
            committed = {picks[j][3]: 1 for j in range(len(picks))}
            cands, _, nv = get_top_k_candidates(
                src, world, seat, mi,
                fleets_raw=raw_fleets, committed=committed,
            )
            if nv == 0:
                continue
            ti = min(int(tgt_idx[i]), nv - 1)
            target_pid = cands[ti].id
            picks.append((pid, mi, int(frac_idx[i]), target_pid))

        return materialize_with_targets(picks, world, seat)

    def _frozen_action(self, obs: dict, seat: int) -> list:
        """Compute action for a frozen (non-trainable) opponent."""
        if self._opponent_fn is not None:
            return self._opponent_fn(obs, seat) or []
        opp_type = self._opponent_type
        if opp_type == "noop":
            return []
        if opp_type == "random":
            return _random_action(obs, seat)
        return []

    def _update_history(self, seat: int, action_list: list):
        """Update sliding-window history for one seat after a step."""
        from featurize import nearest_target_index, ship_bucket_idx
        hist = self._histories[seat]
        obs  = self._env.state[seat].observation or {}
        raw_planets = obs.get("planets", []) or []
        hist["obs_history"].append({"planets": raw_planets, "step": self._step_num})
        for mv in action_list:
            if len(mv) != 3:
                continue
            src_id, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
            src_p = next((p for p in raw_planets if int(p[0]) == src_id), None)
            if src_p is None:
                continue
            ti   = nearest_target_index(src_p, ang, raw_planets)
            tpid = int(raw_planets[ti][0]) if ti is not None else -1
            garrison = int(src_p[5]) + ships
            bi   = ship_bucket_idx(ships, max(1, garrison))
            prev = hist["last_actions_by_planet"].get(src_id, (-1, 0, -1, 0))
            hist["last_actions_by_planet"][src_id] = (tpid, bi, self._step_num, prev[3] + 1)
            hist["cum_stats"]["total_ships_sent"] += ships
            hist["cum_stats"]["total_actions"]    += 1
            hist["action_history"].append((src_id, tpid, bi, self._step_num))


# ──────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ──────────────────────────────────────────────────────────────────────────────

def _random_action(obs: dict, seat: int) -> list:
    import random
    planets = obs.get("planets", []) or []
    my_planets = [p for p in planets if p[1] == seat and p[5] > 2]
    if not my_planets:
        return []
    src = random.choice(my_planets)
    others = [p for p in planets if p[0] != src[0]]
    if not others:
        return []
    tgt = random.choice(others)
    ships = max(1, int(src[5] * 0.5))
    angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
    return [[int(src[0]), float(angle), int(ships)]]
