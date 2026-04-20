"""Async eval watcher for variant_picker_dual_stream_ppo (v2).

Adapts the v1 watcher to DualStreamAgent: loads new architecture, rasterizes
obs for spatial stream, and runs greedy argmax picker vs raw lb-1200 in both
2P and 4P games.

Usage:
  python training/variant_picker_v2_eval_watcher.py \\
      --ckpt training/checkpoints/variant_picker_v2.pt \\
      --out .variant_picker_v2_eval.csv \\
      --interval 120 --n-2p 10 --n-4p 10
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from kaggle_environments import make
from featurize import (featurize_step, PLANET_DIM, FLEET_DIM, GLOBAL_DIM, HISTORY_K)
from training.lb1200_agent import agent as lb1200_agent, Planet as _Planet
from training.dual_stream_model import (
    DualStreamAgent, rasterize_obs, N_SPATIAL_CHANNELS,
)
from training.variant_picker_dual_stream_ppo import (
    generate_variants, K_VARIANTS, GRID,
)


def make_greedy_picker(net: DualStreamAgent):
    """Return agent callable that picks variant via argmax (greedy)."""
    session = {}

    def _reset_session():
        session.clear()
        session["obs_history"] = collections.deque(maxlen=HISTORY_K)
        session["action_history"] = collections.deque(maxlen=HISTORY_K)
        session["last_actions_by_planet"] = {}
        session["cum_stats"] = {"total_ships_sent": 0, "total_actions": 0}
        session["last_step"] = -1
        session["ang_vel"] = None
        session["init_planets"] = None
        session["init_by_id_nt"] = None

    _reset_session()

    def agent(obs, config=None):
        step = int(obs.get("step", 0)) if isinstance(obs, dict) \
               else int(getattr(obs, "step", 0) or 0)
        if step < session.get("last_step", -1):
            _reset_session()
        session["last_step"] = step

        primary = lb1200_agent(obs, config) or []
        if not primary:
            return []

        raw_planets = obs.get("planets", []) if isinstance(obs, dict) \
                      else getattr(obs, "planets", []) or []
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) \
                     else getattr(obs, "fleets", []) or []
        my_player = int(obs.get("player", 0)) if isinstance(obs, dict) \
                    else int(getattr(obs, "player", 0) or 0)

        if session["ang_vel"] is None:
            session["ang_vel"] = float(obs.get("angular_velocity", 0.0)
                                       if isinstance(obs, dict)
                                       else getattr(obs, "angular_velocity", 0.0)
                                       or 0.0)
            session["init_planets"] = obs.get("initial_planets", []) if isinstance(obs, dict) \
                                      else getattr(obs, "initial_planets", []) or []
            if session["init_planets"]:
                session["init_by_id_nt"] = {int(p[0]): _Planet(
                    int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                    float(p[4]), int(p[5]), int(p[6]))
                    for p in session["init_planets"]}
            else:
                session["init_by_id_nt"] = {}

        n_players_guess = 2 if len(set(int(p[1]) for p in raw_planets if int(p[1]) >= 0)) <= 2 \
                          else 4

        variants = generate_variants(
            primary, raw_planets, raw_fleets, my_player,
            ang_vel=session["ang_vel"],
            initial_by_id=session["init_by_id_nt"],
        )

        step_dict = {
            "step": step, "planets": raw_planets, "fleets": raw_fleets,
            "action": primary,
            "my_total_ships": sum(p[5] for p in raw_planets if p[1] == my_player),
            "enemy_total_ships": 0, "my_planet_count": 0,
            "enemy_planet_count": 0, "neutral_planet_count": 0,
        }
        feat = featurize_step(
            step_dict, my_player, session["ang_vel"], n_players_guess,
            session["init_planets"],
            last_actions_by_planet=session["last_actions_by_planet"],
            cumulative_stats=session["cum_stats"],
            obs_history=list(session["obs_history"]),
            action_history=list(session["action_history"]),
        )
        spatial = rasterize_obs(obs, my_player, grid=GRID)

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
            logits, _ = net(
                torch.from_numpy(pl).unsqueeze(0),
                torch.from_numpy(pmask).unsqueeze(0),
                torch.from_numpy(fl).unsqueeze(0),
                torch.from_numpy(fmask).unsqueeze(0),
                torch.from_numpy(feat["globals"]).unsqueeze(0),
                torch.from_numpy(spatial).unsqueeze(0),
            )
            vi = int(logits[0].argmax().item())

        session["obs_history"].append({"planets": raw_planets, "step": step})
        return variants[vi]

    return agent


def run_eval(net: DualStreamAgent, n_2p: int, n_4p: int) -> dict:
    picker = make_greedy_picker(net)
    variant_hits = [0] * K_VARIANTS

    # 2P
    w2 = l2 = 0
    for i in range(n_2p):
        seat = i % 2
        env = make("orbit_wars", debug=False)
        env.reset(num_agents=2)
        agents = [lb1200_agent, lb1200_agent]
        agents[seat] = picker
        env.run(agents)
        r_me = env.state[seat].reward or 0
        r_opp = env.state[1 - seat].reward or 0
        if r_me > r_opp: w2 += 1
        else: l2 += 1

    # 4P (1 picker + 3 lb-1200)
    w4 = l4 = 0
    for i in range(n_4p):
        seat = i % 4
        env = make("orbit_wars", debug=False)
        env.reset(num_agents=4)
        agents = [lb1200_agent] * 4
        agents[seat] = picker
        env.run(agents)
        rewards = [env.state[j].reward or 0 for j in range(4)]
        if rewards[seat] == max(rewards) and rewards.count(max(rewards)) == 1:
            w4 += 1
        else:
            l4 += 1

    return {"w2p": w2, "n2p": n_2p, "w4p": w4, "n4p": n_4p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=".variant_picker_v2_eval.csv")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--n-2p", type=int, default=10)
    ap.add_argument("--n-4p", type=int, default=10)
    args = ap.parse_args()

    if not pathlib.Path(args.out).exists():
        with open(args.out, "w") as f:
            csv.writer(f).writerow(
                ["timestamp", "ckpt_mtime_iso", "iter",
                 "w2p", "n2p", "w4p", "n4p",
                 "winrate_2p", "winrate_4p", "overall_winrate"])

    last_mtime = 0
    print(f"[v2-eval-watcher] polling {args.ckpt} every {args.interval}s",
          flush=True)
    while True:
        ckpt_path = pathlib.Path(args.ckpt)
        if ckpt_path.exists():
            mtime = ckpt_path.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                try:
                    ckpt = torch.load(args.ckpt, map_location="cpu",
                                      weights_only=False)
                except Exception as e:
                    print(f"[watcher] load fail: {e}", flush=True)
                    time.sleep(args.interval); continue
                net = DualStreamAgent(
                    planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM,
                    global_dim=GLOBAL_DIM, n_variants=K_VARIANTS,
                )
                try:
                    net.load_state_dict(ckpt["model"])
                except RuntimeError as e:
                    print(f"[watcher] state_dict mismatch: {e}", flush=True)
                    time.sleep(args.interval); continue
                net.eval()
                iter_ = ckpt.get("iter", -1)
                print(f"[watcher] new ckpt iter {iter_} — running eval "
                      f"({args.n_2p}x 2P + {args.n_4p}x 4P)...", flush=True)
                t0 = time.time()
                r = run_eval(net, args.n_2p, args.n_4p)
                dt = time.time() - t0
                wr2 = r["w2p"] / max(1, r["n2p"])
                wr4 = r["w4p"] / max(1, r["n4p"])
                overall = (r["w2p"] + r["w4p"]) / max(1, r["n2p"] + r["n4p"])
                print(f"[watcher] iter {iter_}: "
                      f"2P={r['w2p']}/{r['n2p']}={wr2:.2f}  "
                      f"4P={r['w4p']}/{r['n4p']}={wr4:.2f}  "
                      f"overall={overall:.2f}  [{dt:.0f}s]", flush=True)
                with open(args.out, "a") as f:
                    csv.writer(f).writerow([
                        datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        datetime.datetime.fromtimestamp(mtime).isoformat(),
                        iter_, r["w2p"], r["n2p"], r["w4p"], r["n4p"],
                        wr2, wr4, overall])
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
