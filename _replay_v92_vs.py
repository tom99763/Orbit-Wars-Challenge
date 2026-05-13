"""Generate Kaggle-format replay JSON + offline HTML viewer for a v92 JAX ckpt
playing against starter (or any registered agent).

Usage:
  python3 _replay_v92_vs.py \\
    --ckpt save/v92_exp/AS1_K_4/snapshots/v92_jax_upd000400.pkl \\
    --opp starter --our-seat 0 --seed 42 \\
    --out viewer/v92_AS1_K_4_upd400_vs_starter

  cd viewer/v92_AS1_K_4_upd400_vs_starter && python3 -m http.server 8765
  → http://localhost:8765/play.html
"""
from __future__ import annotations
import argparse, json, math, pickle, shutil, sys
from pathlib import Path

sys.path.insert(0, "/home/lab/orbit-war")

import numpy as np
import jax
import jax.numpy as jnp

import kaggle_environments
from kaggle_environments.envs.orbit_wars.orbit_wars import starter_agent

# Set env vars BEFORE importing bias_config so K_NEAREST etc. propagate
import os
# These will be loaded from the ckpt's training extras if present (we passed
# K_NEAREST=4 to AS1_K_4 training). Default 7 matches baseline.

from training.v92.env import MAX_PLANETS, MAX_FLEETS, MAX_COMETS, MAX_PATH_LEN, State
from training.v92.env_jax import (
    MAX_PLANETS as JAX_MAX_PLANETS, MAX_FLEETS as JAX_MAX_FLEETS,
    BOARD_SIZE, CENTER,
)
from training.v92.features import (
    PLANET_FEAT_DIM, FLEET_FEAT_DIM, GLOBAL_FEAT_DIM, N_SHIP_BUCKETS, SHIP_FRACS,
)
from training.v92.policy_jax import V92PolicyJAX, TargetHead, ShipHead, init_policy
from training.v92.eval_jax import featurize_np_state_for_jax, jax_policy_action


def obs_to_state(obs: dict) -> State:
    """Construct a minimal env_v3 State from a Kaggle obs dict.

    Only the fields read by `featurize_np_state_for_jax` need to be valid:
    - planets, planets_active, fleets, fleets_active
    - step, angular_velocity
    Other fields get safe defaults.
    """
    planets = np.zeros((MAX_PLANETS, 7), dtype=np.float64)
    pact = np.zeros((MAX_PLANETS,), dtype=bool)
    for i, p in enumerate(obs.get("planets") or []):
        if i >= MAX_PLANETS: break
        planets[i] = p
        pact[i] = True
    fleets = np.zeros((MAX_FLEETS, 7), dtype=np.float64)
    fact = np.zeros((MAX_FLEETS,), dtype=bool)
    for i, f in enumerate(obs.get("fleets") or []):
        if i >= MAX_FLEETS: break
        fleets[i] = f
        fact[i] = True
    return State(
        planets=planets,
        planets_active=pact,
        initial_planets=planets.copy(),
        fleets=fleets,
        fleets_active=fact,
        comet_paths=np.zeros((MAX_COMETS, MAX_PATH_LEN, 2), dtype=np.float64),
        comet_path_len=np.zeros((MAX_COMETS,), dtype=np.int32),
        comet_path_idx=np.full((MAX_COMETS,), -1, dtype=np.int32),
        comet_planet_ids=np.full((MAX_COMETS,), -1, dtype=np.int32),
        step=int(obs.get("step", 0)),
        angular_velocity=float(obs.get("angular_velocity", 0.03)),
        next_fleet_id=0,
        next_planet_id=int(planets[:, 0].max()) + 1 if pact.any() else 0,
        done=False,
        rewards=np.zeros(2, dtype=np.float32),
        seed=0,
    )


def make_v92_jax_agent(ckpt_path: str):
    """Returns a kaggle-compatible agent(obs, config) closure."""
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    body_p = ckpt["body"]; th_p = ckpt["th"]; sh_p = ckpt["sh"]
    upd = ckpt.get("upd", "?")
    print(f"  loaded v92 ckpt {ckpt_path} (upd={upd})")

    key = jax.random.PRNGKey(0)
    body, target_head, ship_head, _, _, _ = init_policy(key)

    def agent(obs, config=None):
        if not obs or not obs.get("planets"):
            return []
        seat = int(obs.get("player", 0))
        state = obs_to_state(obs)
        try:
            moves = jax_policy_action(
                body, target_head, ship_head, body_p, th_p, sh_p,
                state, seat, deterministic=True,
            )
        except Exception as e:
            print(f"[v92 agent] step {state.step} seat {seat}: {e}")
            return []
        return moves
    return agent


def find_player_assets() -> Path:
    # Try simulation/ first (fresh scrapes), then existing replays/ dirs
    for root in ["/home/lab/orbit-war/simulation",
                 "/home/lab/orbit-war/replays",
                 "/home/lab/orbit-war/viewer"]:
        for pa in sorted(Path(root).glob("*/player_assets"), reverse=True):
            if (pa / "index.html").exists():
                return pa
    raise RuntimeError("no player_assets bundle found in simulation/, replays/, or viewer/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="v92 .pkl snapshot")
    ap.add_argument("--opp", default="starter", help="opponent: starter")
    ap.add_argument("--our-seat", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="our display name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Build agents
    print(f"loading our agent (v92 ckpt: {args.ckpt})")
    our_agent = make_v92_jax_agent(args.ckpt)
    our_name = args.name or f"v92:{Path(args.ckpt).stem}"

    if args.opp == "starter":
        opp_agent = starter_agent; opp_name = "starter"
    else:
        raise ValueError(f"unknown opp {args.opp}")

    if args.our_seat == 0:
        agents = [our_agent, opp_agent]; names = [our_name, opp_name]
    else:
        agents = [opp_agent, our_agent]; names = [opp_name, our_name]
    print(f"P0={names[0]}  P1={names[1]}")

    cfg = {"actTimeout": 60, "runTimeout": 1800, "seed": args.seed}
    env = kaggle_environments.make("orbit_wars", configuration=cfg, debug=True)
    env.run(agents)
    last = env.steps[-1]
    r0 = float(last[0]["reward"] or 0); r1 = float(last[1]["reward"] or 0)
    winner = 0 if r0 > r1 else (1 if r1 > r0 else -1)
    print(f"steps: {len(env.steps)}, rewards: P0={r0} P1={r1}, winner={'P'+str(winner) if winner>=0 else 'tie'}")

    rep = env.toJSON()
    replay = rep if isinstance(rep, dict) else json.loads(rep)
    replay.setdefault("info", {})
    replay["info"]["TeamNames"] = names
    (out / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    print(f"saved replay.json ({(out/'replay.json').stat().st_size//1024} KB)")

    pa_src = find_player_assets()
    pa_dst = out / "player_assets"
    if not pa_dst.exists():
        shutil.copytree(pa_src, pa_dst)
        print(f"copied player_assets from {pa_src}")

    title = f"{names[0]} vs {names[1]}"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  html,body{{margin:0;height:100%;background:#0a0a0a;color:#eee;font-family:system-ui,sans-serif}}
  header{{padding:8px 16px;background:#181818;border-bottom:1px solid #333;display:flex;gap:24px;align-items:center}}
  h1{{font-size:14px;margin:0;font-weight:500}}
  .res{{color:{'#9bd' if winner==0 else '#f96' if winner==1 else '#aaa'}}}
  iframe{{width:100%;height:calc(100vh - 40px);border:0}}
</style></head><body>
<header>
  <h1>{title}</h1>
  <span>steps: {len(env.steps)}</span>
  <span class="res">winner: {'P0 ('+names[0]+')' if winner==0 else ('P1 ('+names[1]+')' if winner==1 else 'tie')}</span>
  <span>rewards: {r0:+.0f} / {r1:+.0f}</span>
</header>
<iframe id="player" src="player_assets/index.html"></iframe>
<script>
  const agents = [
    {{ index: 0, name: {json.dumps(names[0])} }},
    {{ index: 1, name: {json.dumps(names[1])} }}
  ];
  fetch("replay.json").then(r => r.json()).then(env => {{
    const f = document.getElementById('player');
    const send = () => f.contentWindow.postMessage(
      {{ type: 'update', environment: env, agents: agents }}, '*');
    f.addEventListener('load', send);
    if (f.contentDocument && f.contentDocument.readyState === 'complete') send();
    setTimeout(send, 1000); setTimeout(send, 3000);
  }});
</script></body></html>
"""
    (out / "play.html").write_text(html, encoding="utf-8")
    print(f"\nopen with:\n  cd {out} && python3 -m http.server 8765\n  → http://localhost:8765/play.html")


if __name__ == "__main__":
    main()
