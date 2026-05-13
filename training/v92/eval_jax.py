"""Eval JAX policy vs heuristic agents.

Uses NumPy env_v3 (byte-exact with kaggle) for the actual gameplay.
Bridges JAX policy → kaggle-style action.

Verifies that each opponent agent can `act()` at BOTH seats (0 and 1).
"""
import os, sys, math, argparse, time, pickle
sys.path.insert(0, "/home/lab/orbit-war")
import numpy as np
import jax
import jax.numpy as jnp

from training.v92 import env_v3
from training.v92.env_jax import (
    reset_from_np, MAX_PLANETS as JAX_MAX_PLANETS, MAX_FLEETS as JAX_MAX_FLEETS,
    BOARD_SIZE, CENTER, SUN_RADIUS,
)
from training.v92.features import (
    PLANET_FEAT_DIM, FLEET_FEAT_DIM, GLOBAL_FEAT_DIM, N_SHIP_BUCKETS, SHIP_FRACS,
)
from training.v92.policy_jax import V92PolicyJAX, TargetHead, ShipHead, init_policy
from training.v92.env import MAX_PLANETS as NP_MAX_PLANETS, MAX_FLEETS as NP_MAX_FLEETS


# Top-10-ish heuristic opponents
TOP_OPPONENTS = [
    "starter",
    "v14",
    "lb928",
    "lb1224",
    "ow_proto",                # has its own loader path
    "nb_sundancer_v2",
    "nb_orbit_star_wars_1224",
    "nb_marco_dg_v3_3",
    "nb_1103_peaking_bot",
    "nb_rudra_1049",
]


def get_opp_callable(name: str):
    """Return (callable, cleanup_fn). Callable takes single obs dict and returns moves."""
    if name == "ow_proto":
        from public_agents.nb_ow_proto_1000 import agent as fn
        return fn, lambda: None
    from public_agents._base import get_agent
    ba = get_agent(name).start(n_workers=1)
    def call(obs):
        return ba.act([obs], [int(obs.get("player", 0))])[0]
    return call, lambda: ba.stop()


def state_to_obs(np_state, player: int):
    """Convert env_v3 NumPy state to kaggle-style obs dict for opponent.

    Includes initial_planets (for rotation lookup) and comets (for capture).
    """
    pact = np_state.planets_active
    fact = np_state.fleets_active

    planets = []
    initial_planets = []
    for i in range(NP_MAX_PLANETS):
        if not pact[i]: continue
        p = list(np_state.planets[i])
        ip = list(np_state.initial_planets[i])
        p[0]=int(p[0]); p[1]=int(p[1]); p[5]=int(p[5]); p[6]=int(p[6])
        ip[0]=int(ip[0]); ip[1]=int(ip[1]); ip[5]=int(ip[5]); ip[6]=int(ip[6])
        planets.append(p)
        initial_planets.append(ip)

    fleets = []
    for i in range(NP_MAX_FLEETS):
        if not fact[i]: continue
        f = list(np_state.fleets[i])
        f[0]=int(f[0]); f[1]=int(f[1]); f[5]=int(f[5]); f[6]=int(f[6])
        fleets.append(f)

    # Reconstruct comets: group by spawn step (each spawn = up to 4 planet ids)
    # env_v3 state has: comet_planet_ids (MAX_C,), comet_paths (MAX_C, MAX_PATH_LEN, 2),
    # comet_path_len (MAX_C,), comet_path_idx (MAX_C,)
    comets = []
    comet_planet_ids = []
    from collections import defaultdict
    groups_by_idx = defaultdict(list)  # path_index → list of (pid, path, path_len)
    for c in range(len(np_state.comet_planet_ids)):
        pid = int(np_state.comet_planet_ids[c])
        if pid < 0: continue
        idx = int(np_state.comet_path_idx[c])
        plen = int(np_state.comet_path_len[c])
        path = [[float(np_state.comet_paths[c, k, 0]), float(np_state.comet_paths[c, k, 1])] for k in range(plen)]
        comet_planet_ids.append(pid)
        # Group by current path_index to reconstruct "groups"
        groups_by_idx[idx].append((pid, path, plen))
    for idx_key, items in groups_by_idx.items():
        comets.append({
            "planet_ids": [it[0] for it in items],
            "paths": [it[1] for it in items],
            "path_index": idx_key,
        })

    return {
        "step": int(np_state.step),
        "angular_velocity": float(np_state.angular_velocity),
        "planets": planets,
        "initial_planets": initial_planets,
        "fleets": fleets,
        "next_fleet_id": int(np_state.next_fleet_id),
        "comets": comets,
        "comet_planet_ids": comet_planet_ids,
        "player": int(player),
    }


def smoke_opponent_at_both_seats(name: str, n_steps: int = 50):
    """Verify an opponent can produce actions at seat 0 AND seat 1 in env_v3."""
    print(f"\n[smoke] {name}", flush=True)
    try:
        call, cleanup = get_opp_callable(name)
    except Exception as e:
        return {"name": name, "load_ok": False, "err": str(e)}
    out = {"name": name, "load_ok": True, "seat0_ok": False, "seat1_ok": False,
           "seat0_n_moves": 0, "seat1_n_moves": 0}
    try:
        for seat in [0, 1]:
            state = env_v3.reset(seed=42, num_agents=2)
            n_moves_total = 0
            err = None
            for t in range(n_steps):
                obs = state_to_obs(state, seat)
                try:
                    moves = call(obs) or []
                    n_moves_total += len(moves)
                except Exception as e:
                    err = f"step {t}: {e}"
                    break
                # Other seat plays NOOP
                if seat == 0:
                    state = env_v3.step(state, [moves, []])
                else:
                    state = env_v3.step(state, [[], moves])
                if state.done: break
            if err:
                out[f"seat{seat}_err"] = err
            else:
                out[f"seat{seat}_ok"] = True
                out[f"seat{seat}_n_moves"] = n_moves_total
    finally:
        try: cleanup()
        except Exception: pass
    return out


# ── JAX policy → env_v3 action bridge ──

def featurize_np_state_for_jax(np_state, player: int):
    """Convert env_v3 NumPy state into JAX feature dict (single env).

    Note: JAX env_jax has MAX_PLANETS=60 (matches NumPy now).
    """
    P = np_state.planets               # (60, 7)
    PA = np_state.planets_active
    F = np_state.fleets
    FA = np_state.fleets_active

    is_mine = (P[:, 1] == player) & PA
    is_enemy = (P[:, 1] != player) & (P[:, 1] != -1) & PA
    is_neutral = (P[:, 1] == -1) & PA

    pf = np.zeros((JAX_MAX_PLANETS, PLANET_FEAT_DIM), dtype=np.float32)
    pf[..., 0] = is_mine
    pf[..., 1] = is_enemy
    pf[..., 2] = is_neutral
    pf[..., 3] = P[:, 5] / 100.0
    pf[..., 4] = P[:, 6]
    pf[..., 5] = (P[:, 2] - CENTER) / CENTER
    pf[..., 6] = (P[:, 3] - CENTER) / CENTER
    pf[..., 7] = P[:, 4]
    # my_nearest_dist
    large_d = float(BOARD_SIZE * 2)
    pos = P[:, 2:4]
    diff = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt((diff * diff).sum(axis=-1))
    mine_mask = is_mine[None, :]
    dists_masked = np.where(mine_mask, dists, large_d)
    nearest = dists_masked.min(axis=-1)
    pf[..., 8] = np.clip(nearest / BOARD_SIZE, 0.0, 2.0)
    # is_largest_enemy
    enemy_ships = np.where(is_enemy, P[:, 5], -1.0)
    if is_enemy.any():
        largest_idx = int(enemy_ships.argmax())
        pf[largest_idx, 9] = 1.0

    own_mask = is_mine & (P[:, 5] >= 1)

    # NP env keeps NP_MAX_FLEETS=3000 slots; JAX policy expects JAX_MAX_FLEETS=256.
    # Compact active fleets to the first JAX_MAX_FLEETS slots.
    ff = np.zeros((JAX_MAX_FLEETS, FLEET_FEAT_DIM), dtype=np.float32)
    active_idx = np.where(FA)[0][:JAX_MAX_FLEETS]
    n_act = len(active_idx)
    if n_act > 0:
        Fc = F[active_idx]
        f_is_mine = (Fc[:, 1] == player)
        safe_ships = np.maximum(Fc[:, 6], 1.0)
        log_norm = np.log(safe_ships) / math.log(1000)
        f_speed = np.minimum(1.0 + 5.0 * np.power(log_norm, 1.5), 6.0)
        tgt_mask = is_enemy | is_neutral
        fdiff = Fc[:, None, 2:4] - pos[None, :, :]
        fdists = np.sqrt((fdiff * fdiff).sum(axis=-1))
        fdists_masked = np.where(tgt_mask[None, :], fdists, BOARD_SIZE * 2)
        fdist = fdists_masked.min(axis=-1)
        f_eta = fdist / np.maximum(f_speed, 0.1)
        ff[:n_act, 0] = f_is_mine
        ff[:n_act, 1] = np.clip(f_eta / 50.0, 0.0, 2.0)
        ff[:n_act, 2] = Fc[:, 6] / 100.0
        ff[:n_act, 3] = Fc[:, 2] / BOARD_SIZE * 2 - 1
        ff[:n_act, 4] = Fc[:, 3] / BOARD_SIZE * 2 - 1
    f_is_mine = (F[:, 1] == player) & FA  # full-size for global aggregates below

    gf = np.zeros(GLOBAL_FEAT_DIM, dtype=np.float32)
    gf[0] = np_state.step / 500.0
    my_p = (P[:, 5] * is_mine).sum()
    my_f = (F[:, 6] * f_is_mine).sum()
    en_p = (P[:, 5] * is_enemy).sum()
    f_is_en = (F[:, 1] != player) & FA
    en_f = (F[:, 6] * f_is_en).sum()
    gf[1] = (my_p + my_f) / 500.0
    gf[2] = (en_p + en_f) / 500.0
    gf[3] = is_mine.sum() / 30.0
    gf[4] = is_enemy.sum() / 30.0

    fm = np.zeros(JAX_MAX_FLEETS, dtype=bool)
    fm[:n_act] = True

    # R20+: compute sun_block exactly like rollout_jax.py (single env).
    # Mirrors training-time mask so eval/argmax cannot pick sun-blocked targets.
    sun_xy = np.array([CENTER, CENTER], dtype=np.float32)
    src_pos = pos[:, None, :].astype(np.float32)              # (P, 1, 2)
    tgt_pos = pos[None, :, :].astype(np.float32)              # (1, P, 2)
    d_vec = tgt_pos - src_pos                                 # (P, P, 2)
    l2 = (d_vec * d_vec).sum(axis=-1)                         # (P, P)
    pv = sun_xy[None, None, :] - src_pos                      # (P, 1, 2)
    t_proj = np.clip(
        (pv * d_vec).sum(axis=-1) / np.where(l2 == 0, 1.0, l2),
        0.0, 1.0,
    )
    proj = src_pos + t_proj[..., None] * d_vec                # (P, P, 2)
    diff2 = sun_xy[None, None, :] - proj
    dist_sun = np.sqrt((diff2 * diff2).sum(axis=-1))          # (P, P)
    sun_block = dist_sun < (SUN_RADIUS + 1.0)                 # (P, P)

    return {
        "planet_feat": pf[None, ...],
        "planet_mask": PA[None, ...].copy(),
        "fleet_feat": ff[None, ...],
        "fleet_mask": fm[None, ...],
        "global_feat": gf[None, ...],
        "own_mask": own_mask[None, ...],
        "sun_block": sun_block[None, ...],
    }


def jax_policy_action(body, target_head, ship_head, body_p, th_p, sh_p,
                       np_state, player: int, deterministic: bool = True):
    """Convert env_v3 state → JAX features → policy forward → kaggle action list."""
    feats = featurize_np_state_for_jax(np_state, player)
    pf = jnp.array(feats["planet_feat"]); pm = jnp.array(feats["planet_mask"])
    ff = jnp.array(feats["fleet_feat"]); fm = jnp.array(feats["fleet_mask"])
    gf = jnp.array(feats["global_feat"]); own = jnp.array(feats["own_mask"])
    sun = jnp.array(feats["sun_block"])

    p_h, value = body.apply(body_p, pf, pm, ff, fm, gf)
    # Per src
    own_flat = own[0]
    own_idxs = np.where(np.asarray(own_flat))[0]
    moves = []
    P_np = np_state.planets
    PA_np = np_state.planets_active

    # R20 audit #8: honor USE_SHIP_HEAD. When off (default), ship_head was
    # never trained and reading its logits = garbage. Use the same sniper
    # formula the training materializer uses.
    from training.v92.bias_config import (
        USE_SHIP_HEAD as _USE_SHIP_HEAD,
        BIAS_REQ_SHIPS as _BIAS_REQ_SHIPS,
    )
    MIN_SHIPS_FLOOR = 10
    LN1000 = math.log(1000.0)
    # R20 audit #1: compute ALL (src, tgt) logits in one TargetHead call.
    # featurize_np_state_for_jax provides `sun` already shaped (1, S, T).
    tgt_logits_all = target_head.apply(th_p, p_h[0:1], p_h[0:1], pm[0:1], sun)  # (1, S, T+1)
    for si in own_idxs.tolist():
        src_h = p_h[0:1, si]
        tgt_log = tgt_logits_all[:, si, :]  # (1, T+1)
        if _USE_SHIP_HEAD:
            ship_log = ship_head.apply(sh_p, src_h)
        if deterministic:
            t_idx = int(jnp.argmax(tgt_log, axis=-1).item())
            if _USE_SHIP_HEAD:
                s_idx = int(jnp.argmax(ship_log, axis=-1).item())
        else:
            t_idx = int(jax.random.categorical(jax.random.PRNGKey(np_state.step * 31 + si), tgt_log[0]).item())
            if _USE_SHIP_HEAD:
                s_idx = int(jax.random.categorical(jax.random.PRNGKey(np_state.step * 31 + si + 1), ship_log[0]).item())
        if t_idx >= JAX_MAX_PLANETS:
            continue
        if _USE_SHIP_HEAD and s_idx == 0:
            continue
        if not PA_np[t_idx] or t_idx == si:
            continue
        src_ships = int(P_np[si, 5])
        if src_ships < 1: continue
        if _USE_SHIP_HEAD:
            frac = SHIP_FRACS[s_idx]
            send = max(1, int(src_ships * frac))
        else:
            tgt_ships = int(P_np[t_idx, 5])
            base = max(tgt_ships + 1, MIN_SHIPS_FLOOR)
            if _BIAS_REQ_SHIPS:
                dx_a = P_np[t_idx, 2] - P_np[si, 2]
                dy_a = P_np[t_idx, 3] - P_np[si, 3]
                dist = math.sqrt(dx_a * dx_a + dy_a * dy_a)
                speed = min(1.0 + 5.0 * (math.log(max(base, 1.0)) / LN1000) ** 1.5, 6.0)
                tick_arrival = math.floor(dist / max(speed, 0.1))
                tgt_prod = float(P_np[t_idx, 6])
                needed = base + tick_arrival * tgt_prod
            else:
                needed = base
            needed = max(int(needed), MIN_SHIPS_FLOOR)
            send = needed
        send = min(send, src_ships)
        if send < 1:
            continue
        dx = P_np[t_idx, 2] - P_np[si, 2]
        dy = P_np[t_idx, 3] - P_np[si, 3]
        angle = math.atan2(dy, dx)
        moves.append([int(P_np[si, 0]), float(angle), send])
    return moves


def run_game(body, target_head, ship_head, body_p, th_p, sh_p,
              opp_call, policy_seat: int, seed: int):
    """One game: JAX policy at policy_seat vs opp at other seat."""
    state = env_v3.reset(seed=seed, num_agents=2)
    while not state.done:
        pol_moves = jax_policy_action(body, target_head, ship_head, body_p, th_p, sh_p,
                                       state, policy_seat, deterministic=True)
        opp_obs = state_to_obs(state, 1 - policy_seat)
        # R20 #6 fix: don't swallow opponent exceptions. Pre-fix, a broken
        # opponent became a no-op → inflated WR + silently corrupted model
        # selection. Raise with context so the bad opponent is visible.
        try:
            opp_moves = opp_call(opp_obs) or []
        except Exception as e:
            raise RuntimeError(
                f"opponent failed: seed={seed}, step={state.step}, "
                f"policy_seat={policy_seat}"
            ) from e
        if policy_seat == 0:
            actions = [pol_moves, opp_moves]
        else:
            actions = [opp_moves, pol_moves]
        state = env_v3.step(state, actions)
    rwd = state.rewards
    return rwd[policy_seat], rwd[1 - policy_seat]


def eval_suite(body, target_head, ship_head, body_p, th_p, sh_p,
                n_games: int = 6, opponents=TOP_OPPONENTS, base_seed: int = 90001):
    """Eval JAX policy vs full opponent list. Each opp plays both seats."""
    results = {}
    for opp_name in opponents:
        print(f"\n[eval] vs {opp_name}", flush=True)
        try:
            call, cleanup = get_opp_callable(opp_name)
        except Exception as e:
            print(f"  load fail: {e}")
            results[opp_name] = {"err": "load_fail"}
            continue
        wins = 0; losses = 0; draws = 0
        seat_wins = {0: 0, 1: 0}
        try:
            for g in range(n_games):
                seat = g % 2
                seed = base_seed + g * 7919
                pr, or_ = run_game(body, target_head, ship_head, body_p, th_p, sh_p,
                                    call, seat, seed)
                if pr > or_:
                    wins += 1; seat_wins[seat] += 1
                elif pr < or_: losses += 1
                else: draws += 1
        finally:
            cleanup()
        wr = wins / max(1, n_games)
        print(f"  result: {wins}W/{losses}L/{draws}D ({wr:.0%})  seat0_wins={seat_wins[0]}/{n_games//2} seat1_wins={seat_wins[1]}/{n_games//2}")
        results[opp_name] = {"wins": wins, "losses": losses, "draws": draws,
                              "wr": wr, "seat0_wins": seat_wins[0], "seat1_wins": seat_wins[1]}
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to policy pickle (None = random init)")
    ap.add_argument("--mode", choices=["smoke", "eval"], default="smoke")
    ap.add_argument("--n-games", type=int, default=6)
    args = ap.parse_args()

    if args.mode == "smoke":
        # Just verify each opponent can act at both seats
        results = []
        for name in TOP_OPPONENTS:
            r = smoke_opponent_at_both_seats(name, n_steps=30)
            results.append(r)
        print("\n=== Smoke summary ===")
        for r in results:
            if not r["load_ok"]:
                print(f"  {r['name']:30s} LOAD_FAIL: {r.get('err', '')[:80]}")
            else:
                s0 = "OK" if r["seat0_ok"] else f"FAIL: {r.get('seat0_err', '?')[:60]}"
                s1 = "OK" if r["seat1_ok"] else f"FAIL: {r.get('seat1_err', '?')[:60]}"
                print(f"  {r['name']:30s} seat0={s0}, seat1={s1}  moves: s0={r['seat0_n_moves']} s1={r['seat1_n_moves']}")
    else:
        # Full eval
        import jax.random as jr
        key = jr.PRNGKey(0)
        body, target_head, ship_head, body_p, th_p, sh_p = init_policy(key)
        if args.ckpt:
            with open(args.ckpt, "rb") as f:
                ck = pickle.load(f)
            body_p = ck["body"]; th_p = ck["th"]; sh_p = ck["sh"]
            print(f"Loaded ckpt: upd={ck.get('upd', '?')}")
        results = eval_suite(body, target_head, ship_head, body_p, th_p, sh_p, n_games=args.n_games)
        print("\n=== Eval summary ===")
        for opp, r in results.items():
            if "err" in r:
                print(f"  vs {opp:30s} ERR")
            else:
                print(f"  vs {opp:30s}: {r['wr']:.0%} ({r['wins']}W/{r['losses']}L/{r['draws']}D) seat0={r['seat0_wins']}/{args.n_games//2} seat1={r['seat1_wins']}/{args.n_games//2}")
