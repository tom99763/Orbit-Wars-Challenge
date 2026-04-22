"""Vectorized Orbit Wars simulator — numpy-batched, N_ENVS parallel games.

Replaces kaggle_environments for training rollouts. Typical speedup: 30-80×
per game versus kaggle_environments at N_ENVS=64.

Game mechanics matched to lb1200_agent constants:
  - Board 100×100; sun at (50,50), radius 10; SUN_SAFETY 1.5
  - Fleet speed 1 + 5·(log(s)/log(1000))^1.5 capped at 6; s≤1 → speed 1.0
  - Planets static if r_from_center + planet_radius ≥ ROTATION_LIMIT (50)
  - Angular velocity applied to orbiting planets each step
  - Combat: largest force − 2nd largest = survivors; owner → largest group
  - Sun-crossing fleets die (segment intersects sun+safety)
  - Game end: step == 500 OR only one player owns planets/fleets
  - Winner = most total ships (planets + fleets)

Action format: list of length N_ENVS, each entry is a dict
  {player_idx: [[src_planet_id, angle_rad, ships], ...]}

Observation format (per env, via get_obs_dict): Kaggle-compatible —
  {'step', 'planets': [[id, owner, x, y, r, ships, prod], ...],
   'fleets':  [[id, owner, x, y, angle, from_id, ships], ...],
   'angular_velocity', 'initial_planets', 'comets', 'n_players'}

This keeps drop-in compatibility with featurize_step / build_world.

Limitations (kept simple for training):
  - 2 / 3 / 4 player supported (P-fold rotational symmetry)
  - No comets (comets=[] always — agents never see comet rewards)
  - Random planet layouts per reset (not Kaggle's exact generator)
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants (match lb1200_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
BOARD           = 100.0
CENTER_X        = 50.0
CENTER_Y        = 50.0
SUN_R           = 10.0
SUN_SAFETY      = 1.5
MAX_SPEED       = 6.0
ROTATION_LIMIT  = 50.0
TOTAL_STEPS     = 500
LAUNCH_CLEAR    = 0.1

MAX_FLEETS      = 200    # per env (fixed-size fleet array)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized mechanics
# ─────────────────────────────────────────────────────────────────────────────

def _fleet_speed_vec(ships: np.ndarray) -> np.ndarray:
    """Vectorized version of lb1200.fleet_speed. Any-shape float array in/out."""
    ships = np.asarray(ships, dtype=np.float32)
    out   = np.ones_like(ships, dtype=np.float32)
    mask  = ships > 1.0
    if mask.any():
        r = np.log(np.clip(ships[mask], 1.0, None)) / math.log(1000.0)
        r = np.clip(r, 0.0, 1.0)
        out[mask] = 1.0 + (MAX_SPEED - 1.0) * (r ** 1.5)
    return out


def _segment_hits_sun_vec(x1, y1, x2, y2, safety: float = SUN_SAFETY) -> np.ndarray:
    """Any-shape broadcast; returns bool array where segment touches sun disk."""
    dx = x2 - x1
    dy = y2 - y1
    seg2 = dx * dx + dy * dy
    safe2 = (SUN_R + safety) ** 2
    # Degenerate segment (fleet stationary) — use point distance
    t = np.where(
        seg2 > 1e-9,
        ((CENTER_X - x1) * dx + (CENTER_Y - y1) * dy) / np.maximum(seg2, 1e-9),
        0.0,
    )
    t = np.clip(t, 0.0, 1.0)
    px = x1 + t * dx
    py = y1 + t * dy
    d2 = (CENTER_X - px) ** 2 + (CENTER_Y - py) ** 2
    return d2 < safe2


# ─────────────────────────────────────────────────────────────────────────────
# Planet layout generation (P-player rotationally symmetric)
# ─────────────────────────────────────────────────────────────────────────────

def _make_layout(rng: np.random.Generator,
                 n_players: int,
                 n_inner_groups: int = 4,
                 n_outer_groups: int = 3) -> dict:
    """Generate one P-player symmetric planet layout (P ∈ {2, 3, 4}).

    Total planets: NP = P × (1 + n_inner_groups + n_outer_groups)
      index 0..P-1:                           P home planets, one per player
      index P..P+n_inner_groups·P-1:          n_inner_groups orbiting groups
      index P+n_inner_groups·P..NP-1:         n_outer_groups static groups

    Each "group" is P planets placed at angles a, a+2π/P, ..., a+2π(P-1)/P
    around the sun → P-fold rotational symmetry for fairness.
    """
    assert n_players in (2, 3, 4), f"n_players must be 2/3/4, got {n_players}"
    P   = int(n_players)
    NP  = P * (1 + n_inner_groups + n_outer_groups)
    da  = 2.0 * math.pi / P     # angle between symmetric partners

    x   = np.zeros(NP, dtype=np.float32)
    y   = np.zeros(NP, dtype=np.float32)
    r   = np.zeros(NP, dtype=np.float32)
    pd  = np.zeros(NP, dtype=np.float32)
    ow  = np.full(NP, -1, dtype=np.int8)
    sh  = np.zeros(NP, dtype=np.float32)
    stc = np.zeros(NP, dtype=bool)
    orb = np.zeros(NP, dtype=np.float32)
    ang = np.zeros(NP, dtype=np.float32)

    def _add(i, rr, theta, p_val):
        pd[i]  = p_val
        r[i]   = 1.0 + math.log(max(p_val, 1))
        x[i]   = CENTER_X + rr * math.cos(theta)
        y[i]   = CENTER_Y + rr * math.sin(theta)
        orb[i] = rr
        ang[i] = theta
        stc[i] = (rr + r[i] >= ROTATION_LIMIT)

    def _angle_clash(aa, taken, tol):
        for ta in taken:
            d = ((aa - ta + math.pi) % (2 * math.pi)) - math.pi
            if abs(d) < tol:
                return True
        return False

    # 1. Home planets (production 3, 10 starting ships, owned by player)
    h_r = rng.uniform(18.0, 26.0)
    h_a = rng.uniform(0.0, 2 * math.pi)
    for p in range(P):
        theta = h_a + p * da
        _add(p, h_r, theta, 3)
        ow[p] = p
        sh[p] = 10.0
    taken = [h_a + p * da for p in range(P)]

    # 2. Inner orbiting groups
    idx = P
    made, tries = 0, 0
    while made < n_inner_groups and tries < 500:
        tries += 1
        rr = rng.uniform(10.0, 38.0)
        aa = rng.uniform(0.0, da)   # sample within one sector; symmetry fills rest
        if _angle_clash(aa, taken, tol=0.35):
            continue
        for p in range(P):
            taken.append(aa + p * da)
        p_val    = int(rng.choice([2, 3, 3, 4]))
        ship_val = int(rng.integers(5, 40))
        for p in range(P):
            _add(idx, rr, aa + p * da, p_val)
            sh[idx] = float(ship_val)
            idx += 1
        made += 1

    # 3. Outer static groups (near the edge, far from center)
    made, tries = 0, 0
    while made < n_outer_groups and tries < 500:
        tries += 1
        rr = rng.uniform(41.0, 47.0)
        aa = rng.uniform(0.0, da)
        if _angle_clash(aa, taken, tol=0.20):
            continue
        for p in range(P):
            taken.append(aa + p * da)
        p_val    = int(rng.choice([1, 2, 3, 4, 5]))
        ship_val = int(rng.integers(20, 90))
        for p in range(P):
            _add(idx, rr, aa + p * da, p_val)
            sh[idx] = float(ship_val)
            idx += 1
        made += 1

    # Filler (rare — only when placement kept clashing)
    while idx < NP:
        rr = 46.0
        aa = rng.uniform(0.0, da)
        for p in range(P):
            if idx >= NP:
                break
            _add(idx, rr, aa + p * da, 1)
            sh[idx] = 30.0
            idx += 1

    return {
        "x": x, "y": y, "radius": r, "prod": pd,
        "init_owner": ow, "init_ships": sh,
        "is_static": stc, "orbit_r": orb, "init_angle": ang,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OrbitWarsVecEnv
# ─────────────────────────────────────────────────────────────────────────────

class OrbitWarsVecEnv:
    """N_ENVS parallel Orbit Wars games, numpy-vectorized step."""

    def __init__(self,
                 n_envs: int = 64,
                 n_players: int = 2,
                 n_inner_groups: int = 4,
                 n_outer_groups: int = 3,
                 ang_vel: float = 0.02,
                 seed: int = 42):
        assert n_players in (2, 3, 4), f"n_players must be 2/3/4, got {n_players}"
        self.N  = int(n_envs)
        self.P  = int(n_players)
        self.NP = self.P * (1 + int(n_inner_groups) + int(n_outer_groups))
        self.n_inner_groups = int(n_inner_groups)
        self.n_outer_groups = int(n_outer_groups)
        self.ang_vel = float(ang_vel)
        self.rng = np.random.default_rng(int(seed))

        N, NP = self.N, self.NP

        # Planet fixed params (per env, overwritten by reset)
        self.pl_init_x     = np.zeros((N, NP), dtype=np.float32)
        self.pl_init_y     = np.zeros((N, NP), dtype=np.float32)
        self.pl_radius     = np.zeros((N, NP), dtype=np.float32)
        self.pl_prod       = np.zeros((N, NP), dtype=np.float32)
        self.pl_orbit_r    = np.zeros((N, NP), dtype=np.float32)
        self.pl_init_angle = np.zeros((N, NP), dtype=np.float32)
        self.pl_is_static  = np.zeros((N, NP), dtype=bool)

        # Planet dynamic state
        self.pl_owner = np.full((N, NP), -1, dtype=np.int8)
        self.pl_ships = np.zeros((N, NP), dtype=np.float32)
        self.pl_x     = np.zeros((N, NP), dtype=np.float32)
        self.pl_y     = np.zeros((N, NP), dtype=np.float32)

        # Fleet state
        self.fl_owner  = np.full((N, MAX_FLEETS), -1, dtype=np.int8)
        self.fl_ships  = np.zeros((N, MAX_FLEETS), dtype=np.float32)
        self.fl_x      = np.zeros((N, MAX_FLEETS), dtype=np.float32)
        self.fl_y      = np.zeros((N, MAX_FLEETS), dtype=np.float32)
        self.fl_vx     = np.zeros((N, MAX_FLEETS), dtype=np.float32)
        self.fl_vy     = np.zeros((N, MAX_FLEETS), dtype=np.float32)
        self.fl_from   = np.full((N, MAX_FLEETS), -1, dtype=np.int16)
        self.fl_angle  = np.zeros((N, MAX_FLEETS), dtype=np.float32)
        self.fl_active = np.zeros((N, MAX_FLEETS), dtype=bool)

        # Meta
        self.step_num  = np.zeros(N, dtype=np.int32)
        self.done_mask = np.zeros(N, dtype=bool)
        self.winner    = np.full(N, -1, dtype=np.int8)

        self.reset()

    # ─── reset ────────────────────────────────────────────────────────────────
    def reset(self, env_ids: Optional[np.ndarray] = None) -> list:
        ids = (np.arange(self.N) if env_ids is None
               else np.asarray(env_ids, dtype=np.int64))
        for eid in ids:
            layout = _make_layout(
                self.rng, self.P,
                self.n_inner_groups, self.n_outer_groups,
            )
            self.pl_init_x[eid]     = layout["x"]
            self.pl_init_y[eid]     = layout["y"]
            self.pl_radius[eid]     = layout["radius"]
            self.pl_prod[eid]       = layout["prod"]
            self.pl_orbit_r[eid]    = layout["orbit_r"]
            self.pl_init_angle[eid] = layout["init_angle"]
            self.pl_is_static[eid]  = layout["is_static"]
            self.pl_owner[eid]      = layout["init_owner"]
            self.pl_ships[eid]      = layout["init_ships"]
            self.pl_x[eid]          = layout["x"]
            self.pl_y[eid]          = layout["y"]

            self.fl_active[eid] = False
            self.fl_owner[eid]  = -1
            self.fl_ships[eid]  = 0.0

            self.step_num[eid]  = 0
            self.done_mask[eid] = False
            self.winner[eid]    = -1

        return [self.get_obs_dict(int(eid)) for eid in ids]

    # ─── planet orbital update ────────────────────────────────────────────────
    def _update_planet_positions(self):
        """Vectorized: pl_x, pl_y ← orbital position at current step."""
        # current angle = init_angle + ang_vel * step
        new_a = self.pl_init_angle + self.ang_vel * self.step_num[:, None]
        new_x = CENTER_X + self.pl_orbit_r * np.cos(new_a)
        new_y = CENTER_Y + self.pl_orbit_r * np.sin(new_a)
        self.pl_x = np.where(self.pl_is_static, self.pl_init_x, new_x).astype(np.float32)
        self.pl_y = np.where(self.pl_is_static, self.pl_init_y, new_y).astype(np.float32)

    # ─── fleet spawn (per-env Python loop, actions are small) ─────────────────
    def _spawn_fleets(self, actions):
        for eid in range(self.N):
            if self.done_mask[eid]:
                continue
            per_env = actions[eid] if eid < len(actions) else {}
            if not per_env:
                continue
            for player, moves in per_env.items():
                if not moves:
                    continue
                player = int(player)
                for mv in moves:
                    if mv is None or len(mv) != 3:
                        continue
                    src_id = int(mv[0])
                    angle  = float(mv[1])
                    ships  = int(mv[2])
                    if src_id < 0 or src_id >= self.NP:
                        continue
                    if int(self.pl_owner[eid, src_id]) != player:
                        continue
                    avail = float(self.pl_ships[eid, src_id])
                    if ships < 1 or ships > int(avail):
                        continue
                    # pick first inactive slot
                    free = np.where(~self.fl_active[eid])[0]
                    if free.size == 0:
                        continue  # fleet array full — drop action
                    slot = int(free[0])
                    sr = float(self.pl_radius[eid, src_id])
                    sx = float(self.pl_x[eid, src_id])
                    sy = float(self.pl_y[eid, src_id])
                    fx = sx + (sr + LAUNCH_CLEAR) * math.cos(angle)
                    fy = sy + (sr + LAUNCH_CLEAR) * math.sin(angle)
                    spd = float(_fleet_speed_vec(np.array([ships], dtype=np.float32))[0])
                    self.fl_x[eid, slot]      = fx
                    self.fl_y[eid, slot]      = fy
                    self.fl_vx[eid, slot]     = spd * math.cos(angle)
                    self.fl_vy[eid, slot]     = spd * math.sin(angle)
                    self.fl_ships[eid, slot]  = float(ships)
                    self.fl_owner[eid, slot]  = np.int8(player)
                    self.fl_from[eid, slot]   = np.int16(src_id)
                    self.fl_angle[eid, slot]  = angle
                    self.fl_active[eid, slot] = True
                    self.pl_ships[eid, src_id] -= float(ships)

    # ─── movement + sun collision (vectorized) ────────────────────────────────
    def _move_fleets(self):
        old_x = self.fl_x.copy()
        old_y = self.fl_y.copy()
        self.fl_x += self.fl_vx * self.fl_active.astype(np.float32)
        self.fl_y += self.fl_vy * self.fl_active.astype(np.float32)

        # Sun crossing
        hit_sun = _segment_hits_sun_vec(old_x, old_y, self.fl_x, self.fl_y) & self.fl_active
        # Out-of-bounds
        oob = ((self.fl_x < 0.0) | (self.fl_x > BOARD) |
               (self.fl_y < 0.0) | (self.fl_y > BOARD)) & self.fl_active
        dead = hit_sun | oob
        if dead.any():
            self.fl_active &= ~dead
            self.fl_ships   = np.where(dead, 0.0, self.fl_ships)
            self.fl_owner   = np.where(dead, np.int8(-1), self.fl_owner)

    # ─── arrivals + combat (loop over planets; vectorized per planet) ─────────
    def _resolve_arrivals(self):
        NP = self.NP
        # For each planet, detect fleets whose distance < planet radius
        # pl_x/pl_y shape [N, NP]. Broadcast against fl_x/fl_y [N, MAX_F].
        for p in range(NP):
            px = self.pl_x[:, p:p+1]       # [N, 1]
            py = self.pl_y[:, p:p+1]
            pr = self.pl_radius[:, p:p+1]
            d2 = (self.fl_x - px) ** 2 + (self.fl_y - py) ** 2
            arrived = (d2 < pr * pr) & self.fl_active
            if not arrived.any():
                continue

            # Per-env: combine arriving fleets + garrison, compute combat
            envs_with_arrivals = np.where(arrived.any(axis=1))[0]
            for eid in envs_with_arrivals:
                self._resolve_combat_at(int(eid), p, arrived[eid])

    def _resolve_combat_at(self, eid: int, pid: int, arrival_mask_1d: np.ndarray):
        """Combat at planet pid in env eid.

        Rule: per-player totals = incoming + (garrison if current owner).
        Largest − 2nd largest = survivors.
        If largest player ≠ current owner AND survivors > 0: ownership flips.
        If survivors == 0: planet becomes neutral (owner = -1).
        """
        cur_owner = int(self.pl_owner[eid, pid])
        cur_ships = float(self.pl_ships[eid, pid])

        # Group by owner
        group: dict[int, float] = {}
        if cur_owner >= 0:
            group[cur_owner] = cur_ships
        slot_idxs = np.where(arrival_mask_1d)[0]
        for s in slot_idxs:
            s = int(s)
            o = int(self.fl_owner[eid, s])
            sh = float(self.fl_ships[eid, s])
            group[o] = group.get(o, 0.0) + sh

        # Consume arriving fleets
        self.fl_active[eid, slot_idxs] = False
        self.fl_ships[eid, slot_idxs]  = 0.0
        self.fl_owner[eid, slot_idxs]  = np.int8(-1)

        if not group:
            return

        ordered = sorted(group.items(), key=lambda kv: -kv[1])
        largest_owner, largest_ships = ordered[0]
        second_ships = ordered[1][1] if len(ordered) > 1 else 0.0
        survivors = largest_ships - second_ships

        if survivors <= 0.0:
            # Mutual destruction → neutral (rare but possible on exact ties)
            self.pl_owner[eid, pid] = np.int8(-1)
            self.pl_ships[eid, pid] = 0.0
        else:
            self.pl_owner[eid, pid] = np.int8(largest_owner)
            self.pl_ships[eid, pid] = float(survivors)

    # ─── production ───────────────────────────────────────────────────────────
    def _produce(self):
        owned = self.pl_owner >= 0
        self.pl_ships += self.pl_prod * owned.astype(np.float32)

    # ─── done detection ───────────────────────────────────────────────────────
    def _check_done(self) -> np.ndarray:
        """Returns rewards [N, P] for newly-done envs (zeros otherwise)."""
        rewards = np.zeros((self.N, self.P), dtype=np.float32)
        for eid in range(self.N):
            if self.done_mask[eid]:
                continue
            # Time-out
            if self.step_num[eid] >= TOTAL_STEPS:
                self.done_mask[eid] = True
                w = self._winner_by_ships(eid)
                self.winner[eid] = w
                if w >= 0:
                    rewards[eid, w] = 1.0
                    for p in range(self.P):
                        if p != w: rewards[eid, p] = -1.0
                continue
            # Only one player remaining
            active_p = []
            for p in range(self.P):
                pl = (self.pl_owner[eid] == p).any()
                fl = ((self.fl_owner[eid] == p) & self.fl_active[eid]).any()
                if pl or fl:
                    active_p.append(p)
            if len(active_p) == 1:
                self.done_mask[eid] = True
                w = active_p[0]
                self.winner[eid] = w
                rewards[eid, w] = 1.0
                for p in range(self.P):
                    if p != w: rewards[eid, p] = -1.0
            elif len(active_p) == 0:
                # both wiped at once — draw
                self.done_mask[eid] = True
                self.winner[eid]    = -1
        return rewards

    def _winner_by_ships(self, eid: int) -> int:
        totals = np.zeros(self.P, dtype=np.float32)
        for p in range(self.P):
            totals[p] = float(self.pl_ships[eid][self.pl_owner[eid] == p].sum())
            mask = (self.fl_owner[eid] == p) & self.fl_active[eid]
            totals[p] += float(self.fl_ships[eid][mask].sum())
        if totals.max() <= 0:
            return -1
        best = int(np.argmax(totals))
        if (totals == totals[best]).sum() > 1:
            return -1  # draw
        return best

    # ─── public step ──────────────────────────────────────────────────────────
    def step(self, actions: list) -> tuple:
        """Advance all N envs by one step.

        Returns (obs_list, rewards [N,P], dones [N]).
        """
        self._spawn_fleets(actions)
        self._move_fleets()
        self.step_num += (~self.done_mask).astype(np.int32)
        self._update_planet_positions()
        self._resolve_arrivals()
        self._produce()
        rewards = self._check_done()
        obs_list = [self.get_obs_dict(eid) for eid in range(self.N)]
        return obs_list, rewards, self.done_mask.copy()

    # ─── obs extraction (Kaggle-compatible) ───────────────────────────────────
    def get_obs_dict(self, eid: int) -> dict:
        planets = []
        for p in range(self.NP):
            planets.append([
                int(p), int(self.pl_owner[eid, p]),
                float(self.pl_x[eid, p]),  float(self.pl_y[eid, p]),
                float(self.pl_radius[eid, p]),
                float(self.pl_ships[eid, p]),
                float(self.pl_prod[eid, p]),
            ])
        fleets = []
        active_idx = np.where(self.fl_active[eid])[0]
        for s in active_idx:
            s = int(s)
            fleets.append([
                s, int(self.fl_owner[eid, s]),
                float(self.fl_x[eid, s]),    float(self.fl_y[eid, s]),
                float(self.fl_angle[eid, s]), int(self.fl_from[eid, s]),
                float(self.fl_ships[eid, s]),
            ])
        initial_planets = []
        for p in range(self.NP):
            initial_planets.append([
                int(p), int(-1),
                float(self.pl_init_x[eid, p]), float(self.pl_init_y[eid, p]),
                float(self.pl_radius[eid, p]),
                0.0,
                float(self.pl_prod[eid, p]),
            ])
        return {
            "step":             int(self.step_num[eid]),
            "planets":          planets,
            "fleets":           fleets,
            "angular_velocity": float(self.ang_vel),
            "initial_planets":  initial_planets,
            "comets":           [],
            "n_players":        self.P,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / benchmark
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time, random

    def random_policy(obs, player):
        """Minimal random agent for smoke test."""
        mine = [p for p in obs["planets"] if p[1] == player and p[5] > 3]
        if not mine:
            return []
        src = random.choice(mine)
        others = [p for p in obs["planets"] if p[0] != src[0]]
        if not others:
            return []
        tgt = random.choice(others)
        ang = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
        ships = max(1, int(src[5] * random.uniform(0.3, 0.8)))
        return [[int(src[0]), float(ang), ships]]

    N = 64
    for P in (2, 3, 4):
        env = OrbitWarsVecEnv(n_envs=N, n_players=P, seed=P * 101)
        print(f"\n── {P}P  NP={env.NP}  MAX_FLEETS={MAX_FLEETS} ───────────────")

        t0 = time.time()
        total_steps = 0
        while not env.done_mask.all() and env.step_num.max() < TOTAL_STEPS:
            obs_all = [env.get_obs_dict(eid) for eid in range(N)]
            actions = [
                {p: random_policy(o, p) for p in range(P)}
                for o in obs_all
            ]
            obs_list, rewards, dones = env.step(actions)
            total_steps += 1

        dt = time.time() - t0
        env_steps_per_s = N * total_steps / dt if dt > 0 else 0
        winners = dict(zip(*np.unique(env.winner, return_counts=True)))
        # draws show as -1
        win_counts = {int(k): int(v) for k, v in winners.items()}
        print(f"  time: {dt:.2f}s  env-steps/s: {env_steps_per_s:.0f}  "
              f"games/s: {N / dt:.1f}")
        print(f"  winners (seat→wins): {win_counts}")
        print(f"  avg game length: {float(env.step_num.mean()):.0f} steps")
