"""Full JAX rollout + JAX PPO training loop.

Target: 10K+ SPS end-to-end.
"""
import os, sys, math, time, argparse
sys.path.insert(0, "/home/lab/orbit-war")
from pathlib import Path
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax

from training.v92 import env_v3
from training.v92.env_jax import (
    reset_from_np, step_batch, _planet_pos_at_step,
    MAX_PLANETS as JAX_MAX_PLANETS, MAX_FLEETS as JAX_MAX_FLEETS,
    BOARD_SIZE, CENTER, N_PLAYERS, EPISODE_STEPS,
)
from training.v92.rollout_jax import featurize_jax_2seats
from training.v92.policy_jax import V92PolicyJAX, TargetHead, ShipHead, init_policy
from training.v92.ppo_jax import make_ppo_update, compute_gae
import optax
from training.v92.features import PLANET_FEAT_DIM, FLEET_FEAT_DIM, GLOBAL_FEAT_DIM, N_SHIP_BUCKETS, SHIP_FRACS
from training.v92.bias_config import BIAS_PBRS, active_biases
# Tutorial-aligned fix for gradient-flatness without sacrificing too much SPS:
# - PBRS disabled (force-on Φ=ships made policy learn DON'T FIRE because
#   firing temporarily drops my_ships before fleet lands → negative shaped
#   reward → NOOP convergence in 60 upd). Reverts to sparse ±1.
# Tutorial: 4 epochs × minibatch=256 over a 128-sample rollout ≈ 14
# gradient updates per rollout. Ours (16384 samples): 4 epochs × 8
# minibatches (size 2048) = 32 gradient updates per rollout. Each
# minibatch step is ~4× cheaper than the full-batch step → ~2× total
# PPO time vs the previous 2-epoch-full-batch path.
PPO_EPOCHS = 4
MINIBATCH_SIZE = 2048


def make_rollout_step(body, target_head, ship_head, ship_fracs):
    """Returns jit'd rollout_step that does featurize → policy → sample → materialize → step."""

    @jax.jit
    def rollout_step(state_batch, body_params, th_params, sh_params, key):
        n_envs = state_batch.planet_active.shape[0]
        feats = featurize_jax_2seats(state_batch)
        pf, pm = feats["planet_feat"], feats["planet_mask"]
        ff, fm = feats["fleet_feat"], feats["fleet_mask"]
        gf = feats["global_feat"]
        own = feats["own_mask"]
        sun = feats["sun_block"]

        p_h, value = body.apply(body_params, pf, pm, ff, fm, gf)

        twoN = 2 * n_envs
        # R20 audit #1: vectorized TargetHead (B, MAX_P, H) — no broadcast.
        flat_src_h = p_h.reshape(-1, p_h.shape[-1])  # (2N*MAX_P, hidden) — for ship_head
        from training.v92.bias_config import USE_SHIP_HEAD as _USE_SHIP_HEAD
        tgt_logits_3d = target_head.apply(th_params, p_h, p_h, pm, sun)  # (2N, S, T+1)
        tgt_logits = tgt_logits_3d.reshape(-1, tgt_logits_3d.shape[-1])  # (2N*S, T+1)
        # P0 #B fix: advance key — was returning the original key, so same Gumbel noise reused every step
        k1, k2, key_next = jr.split(key, 3)
        t_idx = jr.categorical(k1, tgt_logits)
        lp_t = jnp.take_along_axis(jax.nn.log_softmax(tgt_logits), t_idx[:, None], axis=-1).squeeze(-1)
        if _USE_SHIP_HEAD:
            ship_logits = ship_head.apply(sh_params, flat_src_h)
            s_idx = jr.categorical(k2, ship_logits)
            lp_s = jnp.take_along_axis(jax.nn.log_softmax(ship_logits), s_idx[:, None], axis=-1).squeeze(-1)
        else:
            s_idx = jnp.zeros_like(t_idx)
            lp_s = jnp.zeros_like(lp_t)

        # Mask: if not own_mask, set NOOP
        own_flat = own.reshape(-1)
        t_idx = jnp.where(own_flat, t_idx, JAX_MAX_PLANETS)
        s_idx = jnp.where(own_flat, s_idx, 0)
        lp_t = jnp.where(own_flat, lp_t, 0.0)
        lp_s = jnp.where(own_flat, lp_s, 0.0)

        t_idx = t_idx.reshape(twoN, JAX_MAX_PLANETS)
        s_idx = s_idx.reshape(twoN, JAX_MAX_PLANETS)
        lp = (lp_t + lp_s).reshape(twoN, JAX_MAX_PLANETS)

        # Sum log probs per seat per env
        seat_lp = lp.sum(axis=-1)  # (2N,)

        # Materialize actions. Tutorial-style baseline: ships = max(tgt+1, 20).
        # USE_SHIP_HEAD → 7-bucket learn ratio.  BIAS_REQ_SHIPS → ow_proto's
        # ships = base + tgt.prod × tick_arrival (capture during transit growth).
        from training.v92.bias_config import (
            USE_SHIP_HEAD as _USE_SHIP_HEAD,
            BIAS_REQ_SHIPS as _BIAS_REQ_SHIPS,
            BIAS_INTERCEPT as _BIAS_INTERCEPT,
        )
        from training.v92.env_jax import CENTER as _CENTER
        MIN_SHIPS = 10  # relaxed from 20 — match rollout_jax to keep home planets able to fire from start
        MAX_FLEET_SPEED = 6.0
        LN1000 = math.log(1000.0)
        def build_seat_actions(t_seat, s_seat, state_batch_local):
            n_envs_local = state_batch_local.planet_active.shape[0]
            valid_t = t_seat < JAX_MAX_PLANETS
            if _USE_SHIP_HEAD:
                valid = valid_t & (s_seat != 0)
            else:
                valid = valid_t
            src = jnp.arange(JAX_MAX_PLANETS, dtype=jnp.float32)[None, :].repeat(n_envs_local, axis=0)
            src = jnp.where(valid, src, -1)
            cur_pos = jax.vmap(_planet_pos_at_step)(state_batch_local, jnp.maximum(state_batch_local.step - 1, 0))
            tgt_safe = jnp.clip(t_seat, 0, JAX_MAX_PLANETS - 1)
            tgt_pos_x = jnp.take_along_axis(cur_pos[:, :, 0], tgt_safe, axis=1)
            tgt_pos_y = jnp.take_along_axis(cur_pos[:, :, 1], tgt_safe, axis=1)
            if _BIAS_INTERCEPT:
                # AS4: 3-iter solve for tick T where fleet of speed_base lands
                # where target WILL be (vs naive direct atan2 which misses
                # rotating inner planets). Only orbiting targets need this;
                # static planets just use atan2.
                src_x = cur_pos[:, :, 0]; src_y = cur_pos[:, :, 1]
                # Speed estimate using base ship count (min sniper count)
                tgt_ships_for_speed = jnp.take_along_axis(state_batch_local.planet_ships, tgt_safe, axis=1)
                base_speed_ships = jnp.maximum(tgt_ships_for_speed + 1, MIN_SHIPS).astype(jnp.float32)
                speed = jnp.minimum(1.0 + 5.0 * (jnp.log(jnp.maximum(base_speed_ships, 1.0)) / LN1000) ** 1.5, MAX_FLEET_SPEED)
                # Initial pos (current step)
                fx, fy = tgt_pos_x, tgt_pos_y
                cur_step = state_batch_local.step.astype(jnp.float32)
                init_ang = jnp.take_along_axis(state_batch_local.planet_init_angle, tgt_safe, axis=1)
                orb_r = jnp.take_along_axis(state_batch_local.planet_orb_r, tgt_safe, axis=1)
                is_orb = jnp.take_along_axis(state_batch_local.planet_is_orbiting, tgt_safe, axis=1)
                omega = state_batch_local.angular_velocity[:, None]
                for _ in range(3):
                    dx = fx - src_x; dy = fy - src_y
                    T = jnp.sqrt(dx * dx + dy * dy) / jnp.maximum(speed, 0.1)
                    future_ang = init_ang + omega * (cur_step[:, None] + T)
                    fx_orb = _CENTER + orb_r * jnp.cos(future_ang)
                    fy_orb = _CENTER + orb_r * jnp.sin(future_ang)
                    fx = jnp.where(is_orb, fx_orb, tgt_pos_x)
                    fy = jnp.where(is_orb, fy_orb, tgt_pos_y)
                angle = jnp.arctan2(fy - src_y, fx - src_x)
            else:
                angle = jnp.arctan2(tgt_pos_y - cur_pos[:, :, 1], tgt_pos_x - cur_pos[:, :, 0])
            if _USE_SHIP_HEAD:
                frac = ship_fracs[s_seat]
                ships = jnp.maximum(1, (state_batch_local.planet_ships.astype(jnp.float32) * frac).astype(jnp.int32))
                ships = jnp.minimum(ships, state_batch_local.planet_ships)
            else:
                tgt_ships = jnp.take_along_axis(state_batch_local.planet_ships, tgt_safe, axis=1)
                base = jnp.maximum(tgt_ships + 1, MIN_SHIPS).astype(jnp.float32)
                if _BIAS_REQ_SHIPS:
                    # ow_proto calculate_req_ships (single-attacker simplification)
                    dx = tgt_pos_x - cur_pos[:, :, 0]; dy = tgt_pos_y - cur_pos[:, :, 1]
                    dist = jnp.sqrt(dx * dx + dy * dy)
                    speed = jnp.minimum(1.0 + 5.0 * (jnp.log(jnp.maximum(base, 1.0)) / LN1000) ** 1.5, MAX_FLEET_SPEED)
                    tick_arrival = jnp.floor(dist / jnp.maximum(speed, 0.1))
                    tgt_prod = jnp.take_along_axis(state_batch_local.planet_prod, tgt_safe, axis=1).astype(jnp.float32)
                    needed = base + tick_arrival * tgt_prod
                else:
                    needed = base
                needed_int = jnp.maximum(needed, MIN_SHIPS).astype(jnp.int32)
                ships = jnp.minimum(state_batch_local.planet_ships, needed_int)
            ships = jnp.where(valid, ships, 0).astype(jnp.float32)
            return jnp.stack([src, angle, ships], axis=-1)

        t0_seat = t_idx[:n_envs]
        s0_seat = s_idx[:n_envs]
        t1_seat = t_idx[n_envs:]
        s1_seat = s_idx[n_envs:]
        a0 = build_seat_actions(t0_seat, s0_seat, state_batch)
        a1 = build_seat_actions(t1_seat, s1_seat, state_batch)
        actions = jnp.concatenate([a0, a1], axis=1)

        new_state = step_batch(state_batch, actions)
        # R20 audit #2: return the features we already computed so the outer
        # loop doesn't have to recompute featurize_jax_2seats on the same
        # state (was being done twice — once outside, once inside JIT).
        return new_state, value, seat_lp, new_state.rewards, new_state.done, t_idx, s_idx, key_next, feats

    return rollout_step


def make_rollout_step_snapshot(body, target_head, ship_head, ship_fracs):
    """Snapshot-opponent rollout: seat 0 uses learner (cur), seat 1 uses
    snapshot (snap_p) params. Two body forwards per step. PPO loss should
    later mask P1 own_mask to skip training on snapshot's actions.

    Triggered when BIAS_SNAPSHOT_OPP=1. Adds ~2× rollout cost.
    """
    @jax.jit
    def rollout_step_snap(state_batch,
                           body_p, th_p, sh_p,
                           body_p_snap, th_p_snap, sh_p_snap,
                           key):
        from training.v92.bias_config import USE_SHIP_HEAD as _USE_SHIP_HEAD
        n_envs = state_batch.planet_active.shape[0]
        feats = featurize_jax_2seats(state_batch)
        # Slice each (2N, ...) into seat-0 and seat-1 halves
        def split(arr): return arr[:n_envs], arr[n_envs:]
        pf0, pf1 = split(feats["planet_feat"])
        pm0, pm1 = split(feats["planet_mask"])
        ff0, ff1 = split(feats["fleet_feat"])
        fm0, fm1 = split(feats["fleet_mask"])
        gf0, gf1 = split(feats["global_feat"])
        own0, own1 = split(feats["own_mask"])
        sun0, sun1 = split(feats["sun_block"])

        def forward_one(body_p_x, th_p_x, sh_p_x, pf, pm, ff, fm, gf, own, sun, key):
            n = pf.shape[0]
            p_h, value = body.apply(body_p_x, pf, pm, ff, fm, gf)
            flat_src_h = p_h.reshape(-1, p_h.shape[-1])  # for ship_head
            # R20 audit #1: vectorized TargetHead.
            tgt_logits_3d = target_head.apply(th_p_x, p_h, p_h, pm, sun)  # (n, S, T+1)
            tgt_logits = tgt_logits_3d.reshape(-1, tgt_logits_3d.shape[-1])
            k1, k2 = jr.split(key)
            t_idx = jr.categorical(k1, tgt_logits)
            lp_t = jnp.take_along_axis(jax.nn.log_softmax(tgt_logits), t_idx[:, None], axis=-1).squeeze(-1)
            if _USE_SHIP_HEAD:
                ship_logits = ship_head.apply(sh_p_x, flat_src_h)
                s_idx = jr.categorical(k2, ship_logits)
                lp_s = jnp.take_along_axis(jax.nn.log_softmax(ship_logits), s_idx[:, None], axis=-1).squeeze(-1)
            else:
                s_idx = jnp.zeros_like(t_idx)
                lp_s = jnp.zeros_like(lp_t)
            own_flat = own.reshape(-1)
            t_idx = jnp.where(own_flat, t_idx, JAX_MAX_PLANETS)
            s_idx = jnp.where(own_flat, s_idx, 0)
            lp_t = jnp.where(own_flat, lp_t, 0.0)
            lp_s = jnp.where(own_flat, lp_s, 0.0)
            t_idx = t_idx.reshape(n, JAX_MAX_PLANETS)
            s_idx = s_idx.reshape(n, JAX_MAX_PLANETS)
            lp = (lp_t + lp_s).reshape(n, JAX_MAX_PLANETS)
            return t_idx, s_idx, lp.sum(axis=-1), value

        k0, k1, k_next = jr.split(key, 3)
        t0, s0, slp0, v0 = forward_one(body_p, th_p, sh_p, pf0, pm0, ff0, fm0, gf0, own0, sun0, k0)
        t1, s1, slp1, v1 = forward_one(body_p_snap, th_p_snap, sh_p_snap, pf1, pm1, ff1, fm1, gf1, own1, sun1, k1)

        t_idx = jnp.concatenate([t0, t1], axis=0)
        s_idx = jnp.concatenate([s0, s1], axis=0)
        seat_lp = jnp.concatenate([slp0, slp1], axis=0)
        value = jnp.concatenate([v0, v1], axis=0)

        # Materialize: same as single-pass rollout
        from training.v92.bias_config import (
            USE_SHIP_HEAD as _USE_SHIP_HEAD_b,
            BIAS_REQ_SHIPS as _BIAS_REQ_SHIPS_b,
            BIAS_INTERCEPT as _BIAS_INTERCEPT_b,
        )
        from training.v92.env_jax import CENTER as _CENTER_b
        MIN_SHIPS_b = 10
        MAX_FLEET_SPEED_b = 6.0
        LN1000_b = math.log(1000.0)
        def build_act(t_seat, s_seat, state_local):
            n = state_local.planet_active.shape[0]
            valid_t = t_seat < JAX_MAX_PLANETS
            valid = (valid_t & (s_seat != 0)) if _USE_SHIP_HEAD_b else valid_t
            src = jnp.arange(JAX_MAX_PLANETS, dtype=jnp.float32)[None, :].repeat(n, axis=0)
            src = jnp.where(valid, src, -1)
            cur_pos = jax.vmap(_planet_pos_at_step)(state_local, jnp.maximum(state_local.step - 1, 0))
            tgt_safe = jnp.clip(t_seat, 0, JAX_MAX_PLANETS - 1)
            tgt_x = jnp.take_along_axis(cur_pos[:, :, 0], tgt_safe, axis=1)
            tgt_y = jnp.take_along_axis(cur_pos[:, :, 1], tgt_safe, axis=1)
            angle = jnp.arctan2(tgt_y - cur_pos[:, :, 1], tgt_x - cur_pos[:, :, 0])
            if _USE_SHIP_HEAD_b:
                frac = ship_fracs[s_seat]
                ships = jnp.maximum(1, (state_local.planet_ships.astype(jnp.float32) * frac).astype(jnp.int32))
                ships = jnp.minimum(ships, state_local.planet_ships)
            else:
                tgt_ships = jnp.take_along_axis(state_local.planet_ships, tgt_safe, axis=1)
                base = jnp.maximum(tgt_ships + 1, MIN_SHIPS_b).astype(jnp.float32)
                if _BIAS_REQ_SHIPS_b:
                    dx = tgt_x - cur_pos[:, :, 0]; dy = tgt_y - cur_pos[:, :, 1]
                    dist = jnp.sqrt(dx * dx + dy * dy)
                    speed = jnp.minimum(1.0 + 5.0 * (jnp.log(jnp.maximum(base, 1.0)) / LN1000_b) ** 1.5, MAX_FLEET_SPEED_b)
                    tick = jnp.floor(dist / jnp.maximum(speed, 0.1))
                    tgt_prod = jnp.take_along_axis(state_local.planet_prod, tgt_safe, axis=1).astype(jnp.float32)
                    needed = base + tick * tgt_prod
                else:
                    needed = base
                ships = jnp.minimum(state_local.planet_ships, jnp.maximum(needed, MIN_SHIPS_b).astype(jnp.int32))
            ships = jnp.where(valid, ships, 0).astype(jnp.float32)
            return jnp.stack([src, angle, ships], axis=-1)

        a0 = build_act(t_idx[:n_envs], s_idx[:n_envs], state_batch)
        a1 = build_act(t_idx[n_envs:], s_idx[n_envs:], state_batch)
        actions = jnp.concatenate([a0, a1], axis=1)

        new_state = step_batch(state_batch, actions)
        # R20 audit #2: also return feats (computed at top of snap path)
        return new_state, value, seat_lp, new_state.rewards, new_state.done, t_idx, s_idx, k_next, feats

    return rollout_step_snap


def collect_rollout_jax(state_batch, body_params, th_params, sh_params, key,
                         rollout_step, t_rollout: int,
                         snap_params=None):
    """Collect T rollout via Python loop (could be lax.scan'd for speed).

    snap_params: if provided, (body_p_snap, th_p_snap, sh_p_snap) for snapshot
    opponent self-play (rollout_step must be the snapshot variant). Else
    rollout_step is the single-param variant.
    """
    buf_value = []
    buf_lp = []
    buf_reward = []
    buf_done = []
    buf_t_idx = []
    buf_s_idx = []
    buf_pf = []
    buf_pm = []
    buf_ff = []
    buf_fm = []
    buf_gf = []
    buf_own = []
    buf_sun = []

    for t in range(t_rollout):
        # R20 audit #2: rollout_step now returns the features it used
        # (computed on pre-action state). Skip the outer featurize call
        # that previously duplicated work.
        if snap_params is None:
            state_batch, value, seat_lp, rewards, dones, t_idx, s_idx, key, feats = rollout_step(
                state_batch, body_params, th_params, sh_params, key
            )
        else:
            body_p_snap, th_p_snap, sh_p_snap = snap_params
            state_batch, value, seat_lp, rewards, dones, t_idx, s_idx, key, feats = rollout_step(
                state_batch, body_params, th_params, sh_params,
                body_p_snap, th_p_snap, sh_p_snap, key
            )
        buf_pf.append(feats["planet_feat"])
        buf_pm.append(feats["planet_mask"])
        buf_ff.append(feats["fleet_feat"])
        buf_fm.append(feats["fleet_mask"])
        buf_gf.append(feats["global_feat"])
        buf_own.append(feats["own_mask"])
        buf_sun.append(feats["sun_block"])  # combined invalid-action mask
        buf_value.append(value)
        buf_lp.append(seat_lp)
        buf_reward.append(rewards)
        buf_done.append(dones)
        buf_t_idx.append(t_idx)
        buf_s_idx.append(s_idx)

    return state_batch, key, {
        "value": jnp.stack(buf_value),
        "lp": jnp.stack(buf_lp),
        "reward": jnp.stack(buf_reward),
        "done": jnp.stack(buf_done),
        "t_idx": jnp.stack(buf_t_idx),
        "s_idx": jnp.stack(buf_s_idx),
        "pf": jnp.stack(buf_pf),
        "pm": jnp.stack(buf_pm),
        "ff": jnp.stack(buf_ff),
        "fm": jnp.stack(buf_fm),
        "gf": jnp.stack(buf_gf),
        "own": jnp.stack(buf_own),
        "sun": jnp.stack(buf_sun),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=128)  # 10K SPS GPU throughput target
    parser.add_argument("--t-rollout", type=int, default=64)  # was 32; doubled to amortize ppo_step cost across 2× env-steps
    parser.add_argument("--total-updates", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default="save/v92_jax")
    parser.add_argument("--resume", default=None,
                        help="Path to a full checkpoint (.ckpt.pkl) to resume from. "
                             "Restores params + opt_state + upd counter + PRNG key.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    key = jr.PRNGKey(args.seed)
    body, target_head, ship_head, body_p, th_p, sh_p = init_policy(key)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(body_p)) + \
               sum(p.size for p in jax.tree_util.tree_leaves(th_p)) + \
               sum(p.size for p in jax.tree_util.tree_leaves(sh_p))
    print(f"[init] V92PolicyJAX {n_params:,} params", flush=True)
    print(f"[init] active biases: {active_biases()}", flush=True)

    ship_fracs = jnp.array(SHIP_FRACS, dtype=jnp.float32)
    # Always build BOTH rollout fns — pool activation chooses at runtime.
    rollout_step_continuous = make_rollout_step(body, target_head, ship_head, ship_fracs)
    rollout_step_snap = make_rollout_step_snapshot(body, target_head, ship_head, ship_fracs)

    # ── Snapshot pool (PFSP) ──
    # Pool starts EMPTY. After upd POOL_START_UPD, every POOL_SNAP_INTERVAL
    # updates we push a copy of current params into the pool. Pool caps at
    # POOL_MAX_SIZE (drop oldest). Per rollout: PFSP-sample one snap from
    # pool, weight by (1 - learner_wr_against_snap) ** PFSP_P, plus epsilon.
    # If pool empty (upd < POOL_START_UPD), fall back to continuous self-play.
    # PFSP pool DISABLED 2026-05-13: pool start at upd 600 was triggering
    # instability (cf 0.07 → 0.42, ent_t 0.7 → 0.32 within 30 upd) right
    # as the first snapshot landed. Reverting to pure continuous self-play
    # (both seats = current learner) for the full run. Re-enable by setting
    # POOL_START_UPD to a finite value.
    POOL_START_UPD = 10**12   # effectively never
    POOL_SNAP_INTERVAL = 50
    POOL_MAX_SIZE = 20
    PFSP_P = 2.0
    PFSP_EPS = 0.05
    snap_pool = []        # list of param-tuples (body_p, th_p, sh_p)
    snap_pool_wr = []     # list of learner-WR EMA (one per pool entry)
    snap_pool_n = []      # # of games used to update EMA (for cold-start damping)
    print(f"[init] PFSP pool: start_upd={POOL_START_UPD}, snap_interval={POOL_SNAP_INTERVAL}, max_size={POOL_MAX_SIZE}, p={PFSP_P}", flush=True)

    # Init envs
    states = [reset_from_np(env_v3.reset(seed=1000+i+args.seed*1000), key) for i in range(args.n_envs)]
    state_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    print(f"[init] {args.n_envs} envs ready", flush=True)

    # Compile (continuous path — snap path compiles on first pool sample).
    # R20 audit #9: throw away the post-compile state so update 0 starts
    # from the true initial reset, not state advanced by 2 ticks.
    print("[init] compiling...", flush=True)
    t0 = time.time()
    pristine_state = state_batch
    _scratch_state, key, _ = collect_rollout_jax(
        state_batch, body_p, th_p, sh_p, key, rollout_step_continuous, t_rollout=2, snap_params=None,
    )
    _scratch_state.planet_ships.block_until_ready()
    state_batch = pristine_state  # restore — only the compile cache should persist
    print(f"[init] compile took {time.time()-t0:.1f}s", flush=True)

    # LR warmup + cosine decay (opt-step granularity).
    # PPO_EPOCHS × n_mb opt steps per upd; total_updates known here.
    _n_mb_per_upd = max(1, (args.t_rollout * 2 * args.n_envs) // MINIBATCH_SIZE)
    _opt_steps_per_upd = PPO_EPOCHS * _n_mb_per_upd
    _total_opt_steps = args.total_updates * _opt_steps_per_upd
    _warmup_opt_steps = int(0.1 * _total_opt_steps)  # 10% warmup
    _peak_lr = 3e-4
    _end_lr = _peak_lr * 0.1  # cosine end = 10% of peak
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=_peak_lr,
        warmup_steps=_warmup_opt_steps,
        decay_steps=_total_opt_steps,
        end_value=_end_lr,
    )
    print(f"[init] LR schedule: warmup {_warmup_opt_steps}/{_total_opt_steps} opt-steps "
          f"(peak={_peak_lr:.0e}, end={_end_lr:.0e})", flush=True)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init((body_p, th_p, sh_p))
    ppo_step = make_ppo_update(body, target_head, ship_head, optimizer)

    # ── Resume from full checkpoint ──
    import pickle as _pickle
    start_upd = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            # Auto-find latest full checkpoint in save_dir if path not explicit
            resume_path = sorted((save_dir / "snapshots").glob("*.ckpt.pkl"))[-1]
        print(f"[resume] loading {resume_path}", flush=True)
        with open(resume_path, "rb") as _f:
            ckpt = _pickle.load(_f)
        body_p  = ckpt["body"]
        th_p    = ckpt["th"]
        sh_p    = ckpt["sh"]
        opt_state = ckpt["opt_state"]
        key     = ckpt["key"]
        start_upd = int(ckpt["upd"]) + 1
        total_env_steps = int(ckpt.get("total_env_steps", 0))
        print(f"[resume] restored upd={start_upd-1}, continuing from upd {start_upd}", flush=True)

    # CSV log
    log_path = save_dir / "train.csv"
    if not log_path.exists():
        with open(log_path, "w") as f:
            f.write("upd,pg_loss,vf_loss,ent_t,ent_s,approx_kl,clip_frac,n_fleets,sps,wall_sec\n")
    snap_dir = save_dir / "snapshots"; snap_dir.mkdir(exist_ok=True)

    # Train end-to-end
    t_start = time.time()
    total_env_steps = 0
    n_envs_int = args.n_envs
    for upd in range(start_upd, args.total_updates):
        t0 = time.time()
        # P0 #A fix part 3: host-side reset of any done envs BEFORE this rollout.
        # env_jax.step has no auto-reset; without this the trainer rolls forward
        # on post-terminal sticky-done state for the rest of training.
        done_np = np.asarray(state_batch.done)
        if done_np.any():
            n_reset = int(done_np.sum())
            new_states = []
            for i in range(n_envs_int):
                if done_np[i]:
                    seed_i = int(1000 + args.seed * 1000000 + total_env_steps + i)
                    new_states.append(reset_from_np(env_v3.reset(seed=seed_i), jr.PRNGKey(seed_i)))
                else:
                    new_states.append(jax.tree_util.tree_map(lambda x: x[i], state_batch))
            state_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *new_states)
            if upd % 20 == 0 or n_reset > n_envs_int // 4:
                print(f"  [reset] upd {upd}: reset {n_reset}/{n_envs_int} done envs", flush=True)

        # ── PFSP snapshot pool ──
        # Append current params to pool every POOL_SNAP_INTERVAL upds after
        # POOL_START_UPD. Drop oldest beyond POOL_MAX_SIZE.
        if upd >= POOL_START_UPD and upd % POOL_SNAP_INTERVAL == 0:
            snap_pool.append((
                jax.tree_util.tree_map(lambda x: jnp.array(x), body_p),
                jax.tree_util.tree_map(lambda x: jnp.array(x), th_p),
                jax.tree_util.tree_map(lambda x: jnp.array(x), sh_p),
            ))
            snap_pool_wr.append(0.5)   # initial WR estimate
            snap_pool_n.append(0)
            if len(snap_pool) > POOL_MAX_SIZE:
                snap_pool.pop(0); snap_pool_wr.pop(0); snap_pool_n.pop(0)
            print(f"  [pool] upd {upd}: +1 snap → size {len(snap_pool)}", flush=True)

        # Sample one snap from pool via PFSP weights: w_i = (1 - wr_i)^p + eps.
        # Harder snaps (low learner-WR-against) get higher sample probability.
        sampled_snap_idx = -1
        if snap_pool:
            weights = np.array([(1.0 - wr) ** PFSP_P + PFSP_EPS for wr in snap_pool_wr], dtype=np.float64)
            weights /= weights.sum()
            sampled_snap_idx = int(np.random.choice(len(snap_pool), p=weights))
            snap_params = snap_pool[sampled_snap_idx]
            chosen_step = rollout_step_snap
        else:
            snap_params = None
            chosen_step = rollout_step_continuous

        state_batch, key, buf = collect_rollout_jax(
            state_batch, body_p, th_p, sh_p, key, chosen_step, t_rollout=args.t_rollout,
            snap_params=snap_params,
        )

        # PFSP WR EMA update — if we used a snap, parse this rollout's wins
        if sampled_snap_idx >= 0:
            r_arr = np.asarray(buf["reward"])  # (T, N, 2)
            d_arr = np.asarray(buf["done"])    # (T, N)
            # For each env, find first done; learner = P0, snap = P1
            wins_learner = 0; n_games = 0
            for env_i in range(n_envs_int):
                dones_env = d_arr[:, env_i]
                if not dones_env.any(): continue
                t_first = int(np.argmax(dones_env))
                r0 = float(r_arr[t_first, env_i, 0])
                r1 = float(r_arr[t_first, env_i, 1])
                if r0 == 0.0 and r1 == 0.0: continue
                n_games += 1
                if r0 > r1: wins_learner += 1
            if n_games > 0:
                alpha = 0.1
                cur_wr = wins_learner / n_games
                snap_pool_wr[sampled_snap_idx] = (1 - alpha) * snap_pool_wr[sampled_snap_idx] + alpha * cur_wr
                snap_pool_n[sampled_snap_idx] += n_games
        # Compute final bootstrap value
        feats_last = featurize_jax_2seats(state_batch)
        _, last_value_2N = body.apply(body_p, feats_last["planet_feat"], feats_last["planet_mask"],
                                       feats_last["fleet_feat"], feats_last["fleet_mask"], feats_last["global_feat"])
        # GAE: split per seat (2N → 2 × N)
        values_2N = buf["value"]  # (T, 2N)
        rewards_2N = jnp.tile(buf["reward"][:, None, :], (1, JAX_MAX_PLANETS, 1)).mean(axis=1)  # not quite — reward shape (T, N, 2 players)
        # Actually buf["reward"] = state.rewards stored per step → (T, n_envs, N_PLAYERS=2)
        # We need per-seat rewards: rewards[:, :, seat]
        r_p0 = buf["reward"][:, :, 0]  # (T, N)
        r_p1 = buf["reward"][:, :, 1]
        sticky_dones = buf["done"]      # (T, N) — sticky from env_jax (done stays True forever)

        # ── P0 #A fix: convert sticky_done → first_done ─────────────────
        # env_jax doesn't auto-reset; once done it stays done. We need:
        #   first_done   — True only on the transition tick (T,N)
        #   valid_row    — True for rows BEFORE+AT first done (training mask)
        prev_done_int = jnp.concatenate([
            jnp.zeros_like(sticky_dones[:1], dtype=jnp.int32),
            jnp.cumsum(sticky_dones[:-1].astype(jnp.int32), axis=0),
        ], axis=0)
        prev_done = prev_done_int > 0      # rows AFTER first done — post-terminal noise
        first_done = sticky_dones & (~prev_done)
        valid_row = ~prev_done             # rows up to AND INCLUDING first-done
        valid_2seat = jnp.concatenate([valid_row, valid_row], axis=1)  # (T, 2N) — reused by adv-norm and PPO own-mask
        # PFSP-pool active when a snap was sampled: P1 plays with frozen-snap
        # params → PPO must not train on P1 samples (their old_lp came from
        # snap, not current). Zero-out the P1 half of valid_2seat.
        if sampled_snap_idx >= 0:
            valid_2seat = valid_2seat.at[:, n_envs_int:].set(False)
        dones = first_done                 # GAE + PBRS bootstrap use the real transition

        # Reward only on the transition tick (drop sticky duplicates)
        r_p0 = jnp.where(first_done, r_p0, 0.0)
        r_p1 = jnp.where(first_done, r_p1, 0.0)

        # ── P0 #L fix: terminal ties = both seats +1 in env. Self-play
        # would learn "stalemate is OK". Zero both on tie.
        terminal_tie = first_done & (r_p0 > 0.0) & (r_p1 > 0.0)
        r_p0 = jnp.where(terminal_tie, 0.0, r_p0)
        r_p1 = jnp.where(terminal_tie, 0.0, r_p1)

        if BIAS_PBRS:
            gf_2N = buf["gf"]
            phi_p0 = gf_2N[:, :n_envs_int, 1] - gf_2N[:, :n_envs_int, 2]
            phi_p1 = gf_2N[:, n_envs_int:, 1] - gf_2N[:, n_envs_int:, 2]
            # P0 #M fix: bootstrap from real post-rollout Φ (not zeros), and
            # zero Φ_next on first_done transition ticks (Ng-1999 needs
            # Φ(terminal)=0 only at TRUE terminal, not chunk boundary).
            gf_last = feats_last["global_feat"]
            phi_p0_boot = gf_last[:n_envs_int, 1] - gf_last[:n_envs_int, 2]
            phi_p1_boot = gf_last[n_envs_int:, 1] - gf_last[n_envs_int:, 2]
            # If the last tick of this chunk was a real done, post-rollout
            # state was reset before computing phi_boot — set Φ=0 there.
            phi_p0_boot = jnp.where(sticky_dones[-1], 0.0, phi_p0_boot)
            phi_p1_boot = jnp.where(sticky_dones[-1], 0.0, phi_p1_boot)
            phi_p0_next = jnp.concatenate([phi_p0[1:], phi_p0_boot[None]], axis=0)
            phi_p1_next = jnp.concatenate([phi_p1[1:], phi_p1_boot[None]], axis=0)
            # And zero Φ_next on every mid-chunk first_done — telescoping demands it.
            phi_p0_next = jnp.where(first_done, 0.0, phi_p0_next)
            phi_p1_next = jnp.where(first_done, 0.0, phi_p1_next)
            gamma = 0.99
            F_p0 = gamma * phi_p0_next - phi_p0
            F_p1 = gamma * phi_p1_next - phi_p1
            # PBRS only on valid rows; post-terminal rows must contribute 0
            r_p0 = jnp.where(valid_row, r_p0 + F_p0, 0.0)
            r_p1 = jnp.where(valid_row, r_p1 + F_p1, 0.0)
        # Split values_2N: (T, N) for p0, (T, N) for p1
        v_p0 = values_2N[:, :n_envs_int]
        v_p1 = values_2N[:, n_envs_int:]
        lv_p0 = last_value_2N[:n_envs_int]
        lv_p1 = last_value_2N[n_envs_int:]
        adv_p0, ret_p0 = compute_gae(r_p0, v_p0, dones, lv_p0)
        adv_p1, ret_p1 = compute_gae(r_p1, v_p1, dones, lv_p1)
        adv = jnp.concatenate([adv_p0, adv_p1], axis=1).reshape(-1)
        # P0 #A fix part 4: normalize advantages OVER VALID ROWS ONLY.
        # Post-terminal rows have adv ≈ 0 (reward 0, V ≈ 0), which collapses
        # batch std and blows up normalized adv on the few real rows. Mask
        # them out for the moments, zero out their advantage post-norm.
        valid_2seat_flat = valid_2seat.reshape(-1).astype(jnp.float32)
        v_count = valid_2seat_flat.sum().clip(min=1.0)
        adv_mean = (adv * valid_2seat_flat).sum() / v_count
        adv_var = ((adv - adv_mean) ** 2 * valid_2seat_flat).sum() / v_count
        adv_std = jnp.sqrt(adv_var) + 1e-8
        adv = jnp.where(valid_2seat_flat > 0, (adv - adv_mean) / adv_std, 0.0)
        ret = jnp.concatenate([ret_p0, ret_p1], axis=1).reshape(-1)
        # Flatten buf for PPO: (T*2N, ...)
        T = args.t_rollout
        pf = buf["pf"].reshape(T * 2 * n_envs_int, JAX_MAX_PLANETS, PLANET_FEAT_DIM)
        pm = buf["pm"].reshape(T * 2 * n_envs_int, JAX_MAX_PLANETS)
        ff = buf["ff"].reshape(T * 2 * n_envs_int, JAX_MAX_FLEETS, FLEET_FEAT_DIM)
        fm = buf["fm"].reshape(T * 2 * n_envs_int, JAX_MAX_FLEETS)
        gf = buf["gf"].reshape(T * 2 * n_envs_int, GLOBAL_FEAT_DIM)
        own = buf["own"].reshape(T * 2 * n_envs_int, JAX_MAX_PLANETS)
        # P0 #A fix part 2: zero out own_mask on post-terminal rows so PPO loss
        # contributes 0 from those samples (matches log_prob_per_shot * own_flat).
        # valid_2seat computed in reward block above.
        own = own & valid_2seat.reshape(-1)[:, None]
        sun_inv = buf["sun"].reshape(T * 2 * n_envs_int, JAX_MAX_PLANETS, JAX_MAX_PLANETS)
        t_idx = buf["t_idx"].reshape(T * 2 * n_envs_int, JAX_MAX_PLANETS)
        s_idx = buf["s_idx"].reshape(T * 2 * n_envs_int, JAX_MAX_PLANETS)
        old_lp = buf["lp"].reshape(-1)

        # Single PPO update on whole batch (1 epoch, 1 minibatch for speed)
        # Value warmup: first 30 upd only update V (policy frozen — let V converge first)
        # Value warmup — 30 upd of V-only. Tutorial skips this (small batch +
        # 4 epoch lets V catch up fast). We have 4096 batch + 2 epoch — V
        # alone needs a head start, else advantages stay noisy and policy
        # collapses on noise. Re-enabled after observing repeated NOOP-crash.
        value_only = upd < 30
        # ent_coef = 0.10 — modest exploration pressure. Bigger ent_coef can't
        # save us when masks are heavy (softmax already degenerate); rely on
        # PBRS + relaxed MIN_SHIPS to keep mask non-degenerate.
        ent_t_coef = 0.10
        ent_s_coef = 0.10
        # PPO_EPOCHS × minibatch loop — tutorial-style gradient density.
        # buf_size = T * 2 * n_envs (16384 by default). MINIBATCH_SIZE=2048 →
        # 8 minibatches per epoch × 4 epochs = 32 ppo_step calls.
        buf_size = T * 2 * n_envs_int
        n_mb = max(1, buf_size // MINIBATCH_SIZE)
        for epoch_i in range(PPO_EPOCHS):
            key, perm_key = jr.split(key)
            perm = jr.permutation(perm_key, buf_size)
            for mb_i in range(n_mb):
                idx = perm[mb_i * MINIBATCH_SIZE : (mb_i + 1) * MINIBATCH_SIZE]
                body_p, th_p, sh_p, opt_state, metrics = ppo_step(
                    body_p, th_p, sh_p, opt_state,
                    pf[idx], pm[idx], ff[idx], fm[idx], gf[idx], own[idx], sun_inv[idx],
                    t_idx[idx], s_idx[idx], old_lp[idx], adv[idx], ret[idx],
                    jnp.float32(ent_t_coef), jnp.float32(ent_s_coef),
                    jnp.float32(0.5), jnp.float32(0.2),
                    jnp.bool_(value_only),
                )
        metrics["loss"].block_until_ready()
        dt_upd = time.time() - t0
        env_steps = args.n_envs * args.t_rollout
        total_env_steps += env_steps
        wall = time.time() - t_start
        sps = total_env_steps / wall
        n_fleets = int(state_batch.fleet_active.sum())
        if upd % 20 == 0:
            print(f"[upd {upd:6d}] {dt_upd*1000:.0f}ms vo={int(value_only)} (fleets={n_fleets}) "
                  f"pg={metrics['pg_loss']:.4f} vf={metrics['vf_loss']:.4f} "
                  f"ent_t={metrics['ent_t']:.3f} ent_s={metrics['ent_s']:.3f} "
                  f"kl={metrics['approx_kl']:+.4f} cf={metrics['clip_frac']:.3f} SPS={sps:.0f}", flush=True)

        with open(log_path, "a") as f:
            f.write(f"{upd},{float(metrics['pg_loss']):.6f},{float(metrics['vf_loss']):.6f},"
                    f"{float(metrics['ent_t']):.6f},{float(metrics['ent_s']):.6f},"
                    f"{float(metrics['approx_kl']):.6f},{float(metrics['clip_frac']):.6f},"
                    f"{n_fleets},{sps:.1f},{wall:.1f}\n")

        if upd > 0 and upd % 200 == 0:
            import pickle
            from training.v92 import bias_config as _bc
            action_space = {
                "USE_SHIP_HEAD": _bc.USE_SHIP_HEAD,
                "K_NEAREST": _bc.K_NEAREST,
                "MIN_SHIPS_FLOOR": 10,
                "active_biases": _bc.active_biases(),
            }
            # Lightweight param-only snapshot (for eval / submission)
            snap_path = snap_dir / f"v92_jax_upd{upd:06d}.pkl"
            with open(snap_path, "wb") as f:
                pickle.dump({
                    "body": body_p, "th": th_p, "sh": sh_p, "upd": upd,
                    "action_space": action_space,
                }, f)
            # Full resume checkpoint: params + opt_state + key + upd
            ckpt_path = snap_dir / f"v92_jax_upd{upd:06d}.ckpt.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "body": body_p, "th": th_p, "sh": sh_p,
                    "opt_state": opt_state,
                    "key": key,
                    "upd": upd,
                    "action_space": action_space,
                    "total_env_steps": total_env_steps,
                }, f)
            print(f"[snap] saved → {snap_path} + {ckpt_path.name}", flush=True)

            # ── INLINE EVAL every 200 upd (replaces watcher subprocess) ──
            try:
                from training.v92.eval_batched import eval_snapshot as _eval_snapshot
                t_ev = time.time()
                # ow_proto re-enabled via per-game single-worker pools.
                # n_games=4 + argmax-only (no sample) → ~4× faster eval
                # vs prior 8-games-both-modes config.
                ev_results = _eval_snapshot(body, target_head, ship_head, body_p, th_p, sh_p,
                                             opponents=["starter", "v14", "lb1224", "ow_proto"],
                                             n_games_per_opp=4, base_seed=90000 + upd * 31,
                                             also_sample=False)
                print(f"  [eval] upd {upd} done in {time.time()-t_ev:.0f}s", flush=True)
                # Append to eval.csv. Argmax-only schema (sample dropped).
                opps = ["starter", "v14", "lb1224", "ow_proto"]
                ev_csv = save_dir / "eval.csv"
                if not ev_csv.exists():
                    with open(ev_csv, "w") as f:
                        f.write("upd," + ",".join(f"wr_{o}" for o in opps) + ","
                                + ",".join(f"s0_{o}" for o in opps) + ","
                                + ",".join(f"s1_{o}" for o in opps) + ",wall_sec\n")
                with open(ev_csv, "a") as f:
                    row = [str(upd)]
                    row += [f"{ev_results[o].get('wr', 0):.3f}" for o in opps]
                    row += [str(ev_results[o].get('s0', 0)) for o in opps]
                    row += [str(ev_results[o].get('s1', 0)) for o in opps]
                    row.append(f"{time.time()-t_start:.1f}")
                    f.write(",".join(row) + "\n")
            except Exception as e:
                print(f"  [eval] upd {upd} FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
