"""Sampled MCTS for Orbit Wars.

Design:
- Simultaneous-move env: we explore OUR action tree; opponents' moves
  are drawn from the same NN policy at each sim (treated as env stochasticity).
- Complex action space (multiple owned planets × targets × buckets):
  at each state we sample K joint actions from the policy prior; MCTS
  only considers those K (Sampled MuZero style).
- Env cloning: copy.deepcopy(env) per sim. Slower than
  learned dynamics model but simpler + exact.

Each MCTS.search(env, root_player) returns:
  visits_action_keys: list[tuple]   — one per sampled root action
  visit_counts:       list[int]     — visits aligned with actions
  env_moves:          list[list]    — [[src_id, angle, ships], ...] per action
    (what you'd pass to env.step)
  sub_action_tensors: list[dict]    — (src, tgt, bkt) per sampled action,
                                      needed later to reconstruct loss labels
"""

from __future__ import annotations

import copy
import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from training.model import OrbitAgent, sun_blocker_mask
from training.agent import _encode_obs, SHIPS_BUCKETS


# ---------------------------------------------------------------
# Speed hack: skip kaggle_environments' schema validation during MCTS.
# Every env.step() validates agent actions against jsonschema, which
# cProfile'd at ~5ms per step (≈ 50% of MCTS time). We feed well-formed
# actions internally so validation is redundant.
# ---------------------------------------------------------------

def disable_env_validation():
    """Skip per-step ACTION schema validation only. State / spec / config
    schemas still go through the original validator so defaults get filled.
    The per-step action check was profiled at ~22 % of MCTS time."""
    from kaggle_environments import utils, core
    if getattr(utils, "_orig_process_schema", None) is not None:
        return
    _orig = utils.process_schema
    utils._orig_process_schema = _orig

    def _fast(schema, data):
        # Per-step action is passed as a list of moves. State/spec/config
        # are dicts/Structs — those keep original validation (cheap here,
        # only called at init/reset).
        if isinstance(data, list):
            return (None, data)
        return _orig(schema, data)

    utils.process_schema = _fast
    core.process_schema = _fast


# ---------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------

class Node:
    __slots__ = ("prior", "N", "W", "children", "terminal", "value",
                 "sub_action", "env_move")

    def __init__(self, prior: float = 0.0):
        self.prior = prior       # P(a|s) from NN
        self.N = 0               # visit count
        self.W = 0.0             # total value from children's backups
        self.children: dict = {}  # action_key -> Node
        self.terminal = False
        self.value = 0.0         # NN value estimate at this node
        # Cached per-sampled-action info so we can reconstruct env.step & loss
        self.sub_action: Optional[list[tuple]] = None  # list of (src_i, tgt_class, bkt) per owned planet
        self.env_move: Optional[list[list]] = None     # [[src_id, angle, ships], ...]

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


# ---------------------------------------------------------------
# Joint action sampling
# ---------------------------------------------------------------

def _forward_policy(model: OrbitAgent, obs_dict: dict, device: str = "cpu"):
    """Run one forward pass; return (tgt_logits, bkt_logits, value, planets_raw, owned_indices).
    All tensors are on CPU. Masks sun-crossing moves."""
    pf, pxy, pids, omask, ff, g, planets_raw, player = _encode_obs(obs_dict)
    planets = torch.from_numpy(pf).unsqueeze(0)
    planet_xy = torch.from_numpy(pxy).unsqueeze(0)
    planet_mask = torch.ones((1, pf.shape[0]), dtype=torch.bool)
    if ff.shape[0] > 0:
        fleets = torch.from_numpy(ff).unsqueeze(0)
        fleet_mask = torch.ones((1, ff.shape[0]), dtype=torch.bool)
    else:
        fleets = torch.zeros((1, 1, 9), dtype=torch.float32)
        fleet_mask = torch.zeros((1, 1), dtype=torch.bool)
    globals_ = torch.from_numpy(g).unsqueeze(0)
    tgt_mask = sun_blocker_mask(planet_xy, planet_mask)
    with torch.no_grad():
        tgt_logits, bkt_logits, value = model(
            planets, planet_mask, fleets, fleet_mask, globals_, tgt_mask
        )
    return tgt_logits[0], bkt_logits[0], value[0].item(), planets_raw, pids, omask


def sample_joint_action(tgt_logits: torch.Tensor, bkt_logits: torch.Tensor,
                        planets_raw: list, owned_mask: np.ndarray,
                        temperature: float = 1.0,
                        deterministic: bool = False,
                        planet_action_noise: float = 0.0):
    """Sample ONE joint action (one tuple per owned planet) and compute its
    joint log-prob. Returns (sub_action, env_move, log_prob).

    sub_action[i] = (src_i, tgt_class, bkt) where tgt_class==0 means pass.
    env_move is [[src_id, angle, ships], ...] for planets that actually act.

    planet_action_noise: per-planet probability to override the sampled
    (tgt, bkt) with a uniform-random one (for exploration during self-play).
    """
    sub: list[tuple] = []
    env_move: list[list] = []
    log_prob = 0.0
    owned_indices = np.where(owned_mask)[0]
    if len(owned_indices) == 0:
        return sub, env_move, 0.0

    t_rows = tgt_logits[owned_indices]    # [K, P+1]
    b_rows = bkt_logits[owned_indices]    # [K, 4]
    t_logp = F.log_softmax(t_rows / max(temperature, 1e-6), dim=-1)
    b_logp = F.log_softmax(b_rows / max(temperature, 1e-6), dim=-1)
    t_prob = t_logp.exp().cpu().numpy()
    b_prob = b_logp.exp().cpu().numpy()

    for i, si in enumerate(owned_indices):
        use_noise = (planet_action_noise > 0.0
                     and random.random() < planet_action_noise)
        if use_noise:
            tgt_class = random.randint(0, len(t_prob[i]) - 1)
            bkt = random.randint(0, len(b_prob[i]) - 1)
        elif deterministic:
            tgt_class = int(t_prob[i].argmax())
            bkt = int(b_prob[i].argmax())
        else:
            tgt_class = int(np.random.choice(len(t_prob[i]), p=t_prob[i]))
            bkt = int(np.random.choice(len(b_prob[i]), p=b_prob[i]))
        log_prob += float(t_logp[i, tgt_class]) + float(b_logp[i, bkt])
        sub.append((int(si), tgt_class, bkt))
        if tgt_class == 0:
            continue
        tgt_idx = tgt_class - 1
        if tgt_idx >= len(planets_raw):
            continue
        src = planets_raw[si]
        tgt = planets_raw[tgt_idx]
        garrison = int(src[5])
        ships = max(1, int(round(SHIPS_BUCKETS[bkt] * garrison)))
        if ships <= 0 or ships > garrison:
            continue
        ang = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
        env_move.append([int(src[0]), float(ang), int(ships)])
    return sub, env_move, log_prob


def sample_k_distinct_actions(model: OrbitAgent, obs_dict: dict,
                              k: int = 8, temperature: float = 1.0) \
        -> tuple[list[tuple[list, list, float]], float]:
    """Return up to k unique joint action samples, each (sub, env_move, log_prob),
    plus the value estimate at this state."""
    tgt_logits, bkt_logits, value, planets_raw, pids, omask = \
        _forward_policy(model, obs_dict)
    seen = set()
    out = []
    attempts = 0
    while len(out) < k and attempts < k * 8:
        attempts += 1
        sub, env_move, lp = sample_joint_action(
            tgt_logits, bkt_logits, planets_raw, omask, temperature
        )
        key = tuple(sub) if sub else ("pass",)
        if key in seen:
            continue
        seen.add(key)
        out.append((sub, env_move, lp))
    return out, value, planets_raw, omask


# ---------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------

class MCTS:
    def __init__(self, model: OrbitAgent, n_sims: int = 50,
                 c_puct: float = 1.5, k_samples: int = 8,
                 dirichlet_alpha: float = 0.3, root_noise_frac: float = 0.25,
                 opp_temperature: float = 1.0, my_seat: int = 0,
                 n_players: int = 2):
        self.model = model
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.k_samples = k_samples
        self.dir_alpha = dirichlet_alpha
        self.noise_frac = root_noise_frac
        self.opp_temp = opp_temperature
        self.my_seat = my_seat
        self.n_players = n_players

    # ----- Selection -----
    def _puct(self, parent: Node, child: Node) -> float:
        u = self.c_puct * child.prior * math.sqrt(parent.N) / (1 + child.N)
        return child.Q + u

    def _select_child(self, node: Node) -> tuple[object, Node]:
        best_key, best_child, best_score = None, None, -1e18
        for k, ch in node.children.items():
            s = self._puct(node, ch)
            if s > best_score:
                best_score = s
                best_key = k
                best_child = ch
        return best_key, best_child

    # ----- Action sampling at my seat -----
    def _expand(self, node: Node, env):
        """Expand node by sampling K actions from policy at env's current state
        (for my_seat)."""
        if node.children:
            return  # already expanded
        obs = _shared_obs(env, self.my_seat)
        samples, value, planets_raw, omask = sample_k_distinct_actions(
            self.model, obs, k=self.k_samples, temperature=1.0
        )
        node.value = value
        if not samples:
            # No-op: only the "pass" action
            child = Node(prior=1.0)
            child.sub_action, child.env_move = [], []
            node.children[("pass",)] = child
            return
        # Convert per-sample log_prob to priors via softmax
        lps = np.array([s[2] for s in samples], dtype=np.float32)
        lps = lps - lps.max()
        priors = np.exp(lps); priors = priors / priors.sum()
        for (sub, env_move, _lp), p in zip(samples, priors):
            key = tuple(sub) if sub else ("pass",)
            ch = Node(prior=float(p))
            ch.sub_action = sub
            ch.env_move = env_move
            node.children[key] = ch

    def _add_dirichlet_noise(self, node: Node):
        if not node.children:
            return
        ks = list(node.children.keys())
        noise = np.random.dirichlet([self.dir_alpha] * len(ks))
        for k, n in zip(ks, noise):
            node.children[k].prior = (
                (1 - self.noise_frac) * node.children[k].prior
                + self.noise_frac * float(n)
            )

    # ----- Opponent action at a state (sampled from same NN) -----
    def _opponent_moves(self, env) -> dict[int, list]:
        """For each seat != my_seat, sample an action from the policy."""
        out: dict[int, list] = {}
        for seat in range(self.n_players):
            if seat == self.my_seat:
                continue
            obs = _shared_obs(env, seat)
            tl, bl, _, planets_raw, pids, omask = \
                _forward_policy(self.model, obs)
            _sub, env_move, _ = sample_joint_action(
                tl, bl, planets_raw, omask, temperature=self.opp_temp
            )
            out[seat] = env_move
        return out

    def _step_env(self, env, my_move: list) -> float:
        """Advance env by one turn with my_seat using my_move, others via NN.
        Return terminal reward (for my_seat) if done, else None."""
        opp = self._opponent_moves(env)
        actions = [None] * self.n_players
        actions[self.my_seat] = my_move
        for s, m in opp.items():
            actions[s] = m
        env.step(actions)
        if env.done:
            r = env.state[self.my_seat].reward
            return float(r or 0)
        return None

    # ----- Run one simulation from root -----
    def _simulate(self, root: Node, root_env):
        # Temporarily drop env.steps history so deepcopy is O(1) per entry.
        # env.step() only reads env.state, so truncating is safe.
        saved_steps = root_env.steps
        root_env.steps = root_env.steps[-1:] if root_env.steps else []
        env = copy.deepcopy(root_env)
        root_env.steps = saved_steps  # restore original history
        path = [root]
        node = root
        # Walk down existing tree
        while node.children and not node.terminal:
            key, child = self._select_child(node)
            path.append(child)
            reward = self._step_env(env, child.env_move)
            if reward is not None:
                child.terminal = True
                node = child
                value = reward
                break
            node = child
            if not node.children:
                # New leaf: expand
                self._expand(node, env)
                value = node.value  # NN value estimate at leaf
                break
        else:
            # Loop exited because node.terminal (already handled) — use its value
            value = node.value if not node.terminal else 0.0

        # Backup
        for n in path:
            n.N += 1
            n.W += value

    # ----- Main entry -----
    def search(self, env, add_noise: bool = True) -> tuple[list, dict]:
        """Run n_sims simulations; return (action_keys_sorted_by_visit,
        stats_dict). stats_dict contains visit_counts, priors, q_values, and
        the chosen root children so caller can replay + compute loss labels.
        """
        root = Node(prior=1.0)
        self._expand(root, env)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_sims):
            self._simulate(root, env)

        keys = list(root.children.keys())
        children = [root.children[k] for k in keys]
        visits = np.array([c.N for c in children], dtype=np.float32)
        priors = np.array([c.prior for c in children], dtype=np.float32)
        qs = np.array([c.Q for c in children], dtype=np.float32)
        return keys, {
            "visits": visits,
            "priors": priors,
            "q_values": qs,
            "sub_actions": [c.sub_action for c in children],
            "env_moves": [c.env_move for c in children],
            "value_at_root": root.value,
        }


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _shared_obs(env, seat: int) -> dict:
    """Grab the shared world state from seat 0 and relabel player id."""
    raw = env.state[0].observation
    try:
        d = {k: raw[k] for k in raw}
    except Exception:
        d = dict(raw) if isinstance(raw, dict) else raw
    d["player"] = seat
    return d


def mcts_action_from_search(keys: list, stats: dict, temperature: float = 1.0):
    """Sample an action from MCTS visit counts, return (sub, env_move, π_target_dict).

    π_target_dict maps each of the K root children's keys -> probability."""
    visits = stats["visits"]
    if visits.sum() == 0:
        # Fallback to prior
        probs = stats["priors"]
    else:
        if temperature == 0:
            probs = np.zeros_like(visits)
            probs[visits.argmax()] = 1.0
        else:
            v = visits ** (1.0 / max(temperature, 1e-3))
            probs = v / v.sum()
    idx = int(np.random.choice(len(probs), p=probs))
    sub = stats["sub_actions"][idx] or []
    env_move = stats["env_moves"][idx] or []
    pi_target = {keys[i]: float(probs[i]) for i in range(len(keys))}
    return sub, env_move, pi_target
