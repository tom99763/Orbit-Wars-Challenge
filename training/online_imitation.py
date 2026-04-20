"""DAgger-style online imitation learning.

Each game has MIXED seats: some played by our model, some by lb-1200. At every
step, for every seat we also CALL LB-1200 ON THE OBS to get the oracle label.
This guarantees the action_history feature at training time reflects the SAME
distribution as inference (model's own prior actions) for model-seats, while
lb-1200-seats provide stable on-policy baseline data.

Training sample: (obs + sliding-window history built from whatever happened so
far, lb-1200's oracle answer as label).

Usage:
  python training/online_imitation.py \
      --target-games 20000 --workers 8 --four-player-prob 0.5 \
      --out training/checkpoints/imitation_v1.pt
"""
from __future__ import annotations

import argparse
import collections
import io
import math
import multiprocessing as mp
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from featurize import (featurize_step, nearest_target_index, ship_bucket_idx,
                       HISTORY_K, PLANET_DIM, FLEET_DIM, GLOBAL_DIM,
                       SHIPS_BUCKETS)
from training.model import OrbitAgent, sun_blocker_mask


_worker_state: dict = {}


def _worker_init(model_kwargs: dict):
    torch.set_num_threads(1)
    _worker_state["model"] = OrbitAgent(**model_kwargs)
    _worker_state["model"].eval()


def _model_pick_action(feat: dict, planets_raw: list, ang_vel: float,
                      init_planets: list, comets: list, comet_ids: set,
                      temperature: float = 0.8):
    """Run model on a single seat's featurized obs, return env-format action
    list [[src_pid, angle, ships], ...] using physics aim."""
    pf = feat["planets"]; ff = feat["fleets"]; g = feat["globals"]
    pxy = feat["planet_xy"]; omask = feat["action_mask_owned"]
    P = pf.shape[0]
    if P == 0 or not omask.any():
        return []
    planets_t = torch.from_numpy(pf).unsqueeze(0)
    planet_xy_t = torch.from_numpy(pxy).unsqueeze(0)
    planet_mask = torch.ones((1, P), dtype=torch.bool)
    if ff.shape[0] > 0:
        fleets_t = torch.from_numpy(ff).unsqueeze(0)
        fleet_mask = torch.ones((1, ff.shape[0]), dtype=torch.bool)
    else:
        fleets_t = torch.zeros((1, 1, FLEET_DIM))
        fleet_mask = torch.zeros((1, 1), dtype=torch.bool)
    globals_t = torch.from_numpy(g).unsqueeze(0)
    tgt_mask = sun_blocker_mask(planet_xy_t, planet_mask)
    with torch.no_grad():
        tgt_logits, bkt_logits, _ = _worker_state["model"](
            planets_t, planet_mask, fleets_t, fleet_mask, globals_t, tgt_mask)
    tgt_l = tgt_logits[0].numpy()
    bkt_l = bkt_logits[0].numpy()

    from physics_aim import compute_aim_angle
    moves = []
    owned = np.where(omask)[0]
    for si in owned:
        row = tgt_l[si]
        if temperature > 0:
            z = row / max(temperature, 1e-6)
            z = z - z.max()
            p = np.exp(z); p = p / p.sum()
            tgt_class = int(np.random.choice(len(p), p=p))
        else:
            tgt_class = int(row.argmax())
        if tgt_class == 0:
            continue
        tgt_idx = tgt_class - 1
        if tgt_idx >= len(planets_raw):
            continue
        row_b = bkt_l[si]
        if temperature > 0:
            z = row_b / max(temperature, 1e-6)
            z = z - z.max()
            p = np.exp(z); p = p / p.sum()
            bkt_idx = int(np.random.choice(len(p), p=p))
        else:
            bkt_idx = int(row_b.argmax())
        src = planets_raw[si]
        tgt = planets_raw[tgt_idx]
        garrison = int(src[5])
        num_ships = max(1, int(round(SHIPS_BUCKETS[bkt_idx] * garrison)))
        if num_ships <= 0 or num_ships > garrison:
            continue
        angle = compute_aim_angle(src, tgt, num_ships,
                                  ang_vel, init_planets, comets, comet_ids)
        moves.append([int(src[0]), float(angle), int(num_ships)])
    return moves


def _worker_play(task: dict) -> list:
    from kaggle_environments import make
    from training.lb1200_agent import agent as lb1200_agent

    buf = io.BytesIO(task["weights"])
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    _worker_state["model"].load_state_dict(sd)

    n_players = task["n_players"]
    model_seat_prob = task.get("model_seat_prob", 0.5)

    env = make("orbit_wars", debug=False)
    env.reset(num_agents=n_players)
    obs0 = env.state[0].observation
    ang_vel = float(obs0.get("angular_velocity") or 0.0)
    init_planets = list(obs0.get("initial_planets") or [])
    init_ids = {int(p[0]) for p in init_planets}
    comet_ids: set = set()

    seat_types = ["model" if random.random() < model_seat_prob else "lb1200"
                  for _ in range(n_players)]

    obs_history: collections.deque = collections.deque(maxlen=HISTORY_K)
    action_history = {s: collections.deque(maxlen=HISTORY_K)
                      for s in range(n_players)}
    last_actions = {s: {} for s in range(n_players)}
    cum_stats = {s: {"total_ships_sent": 0, "total_actions": 0}
                 for s in range(n_players)}

    samples = []
    step_idx = 0
    while not env.done:
        shared_obs = env.state[0].observation
        planets = list(shared_obs.get("planets") or [])
        fleets = list(shared_obs.get("fleets") or [])
        comets = list(shared_obs.get("comets") or [])
        for p in planets:
            if int(p[0]) not in init_ids:
                comet_ids.add(int(p[0]))

        actions = [None] * n_players
        for seat in range(n_players):
            seat_obs = env.state[seat].observation
            try:
                oracle_action = lb1200_agent(seat_obs) or []
            except Exception:
                oracle_action = []

            step_dict = {
                "step": step_idx,
                "planets": planets, "fleets": fleets,
                "action": oracle_action,
                "my_total_ships": sum(p[5] for p in planets if p[1] == seat)
                                 + sum(f[6] for f in fleets if f[1] == seat),
                "enemy_total_ships": sum(p[5] for p in planets
                                         if p[1] != seat and p[1] != -1)
                                    + sum(f[6] for f in fleets if f[1] != seat),
                "my_planet_count": sum(1 for p in planets if p[1] == seat),
                "enemy_planet_count": sum(1 for p in planets
                                          if p[1] != seat and p[1] != -1),
                "neutral_planet_count": sum(1 for p in planets if p[1] == -1),
            }
            feat = featurize_step(
                step_dict, seat, ang_vel, n_players, init_planets, comet_ids,
                last_actions_by_planet=last_actions[seat],
                cumulative_stats=cum_stats[seat],
                obs_history=list(obs_history),
                action_history=list(action_history[seat]),
            )
            if len(feat["src_planet_idx"]) > 0:
                samples.append(feat)

            if seat_types[seat] == "lb1200":
                game_action = oracle_action
            else:
                game_action = _model_pick_action(
                    feat, planets, ang_vel, init_planets, comets, comet_ids,
                    temperature=0.8)
            actions[seat] = game_action

            for move in game_action:
                if len(move) != 3:
                    continue
                from_id, angle, ships = move
                src_planet = None
                for p in planets:
                    if int(p[0]) == int(from_id):
                        src_planet = p
                        break
                if src_planet is None:
                    continue
                tgt_i = nearest_target_index(src_planet, angle, planets)
                tgt_pid = int(planets[tgt_i][0]) if tgt_i is not None else -1
                garrison = int(src_planet[5]) + int(ships)
                bkt_idx = ship_bucket_idx(int(ships), max(1, garrison))
                prev = last_actions[seat].get(int(from_id), (-1, 0, -1, 0))
                last_actions[seat][int(from_id)] = (
                    tgt_pid, bkt_idx, step_idx, prev[3] + 1)
                cum_stats[seat]["total_ships_sent"] += int(ships)
                cum_stats[seat]["total_actions"] += 1
                action_history[seat].append(
                    (int(from_id), tgt_pid, bkt_idx, step_idx))

        obs_history.append({"planets": planets, "step": step_idx})
        env.step(actions)
        step_idx += 1
    return samples


def train_step(model, optimizer, samples, device):
    if not samples:
        return None
    B = len(samples)
    P = max(s["planets"].shape[0] for s in samples)
    Fmax = max(max(s["fleets"].shape[0] for s in samples), 1)
    planet_dim = samples[0]["planets"].shape[1]
    fleet_dim = next((s["fleets"].shape[1] for s in samples
                      if s["fleets"].shape[0] > 0), FLEET_DIM)
    g_dim = samples[0]["globals"].shape[0]

    pl = np.zeros((B, P, planet_dim), dtype=np.float32)
    pmask = np.zeros((B, P), dtype=bool)
    fl = np.zeros((B, Fmax, fleet_dim), dtype=np.float32)
    fmask = np.zeros((B, Fmax), dtype=bool)
    gl = np.zeros((B, g_dim), dtype=np.float32)
    flat_b, flat_src, flat_tgt, flat_bkt = [], [], [], []
    for i, s in enumerate(samples):
        np_ = s["planets"].shape[0]
        pl[i, :np_] = s["planets"]
        pmask[i, :np_] = True
        nf = s["fleets"].shape[0]
        if nf > 0:
            fl[i, :nf] = s["fleets"]
            fmask[i, :nf] = True
        gl[i] = s["globals"]
        srcs = s["src_planet_idx"]; tgts = s["target_planet_idx"]; bkts = s["ships_bucket"]
        valid = (srcs < np_) & (tgts < np_)
        for j in range(len(srcs)):
            if valid[j]:
                flat_b.append(i); flat_src.append(int(srcs[j]))
                flat_tgt.append(int(tgts[j])); flat_bkt.append(int(bkts[j]))
    if not flat_b:
        return None

    planets = torch.from_numpy(pl).to(device)
    planet_mask = torch.from_numpy(pmask).to(device)
    fleets = torch.from_numpy(fl).to(device)
    fleet_mask = torch.from_numpy(fmask).to(device)
    globals_ = torch.from_numpy(gl).to(device)
    fb = torch.tensor(flat_b, dtype=torch.long, device=device)
    fs = torch.tensor(flat_src, dtype=torch.long, device=device)
    ft = torch.tensor(flat_tgt, dtype=torch.long, device=device)
    fk = torch.tensor(flat_bkt, dtype=torch.long, device=device)

    tgt_logits, bkt_logits, _ = model(
        planets, planet_mask, fleets, fleet_mask, globals_, target_mask=None)
    picked_tgt = tgt_logits[fb, fs]
    picked_bkt = bkt_logits[fb, fs]
    tgt_labels = ft + 1
    ce_tgt = F.cross_entropy(picked_tgt, tgt_labels)
    ce_bkt = F.cross_entropy(picked_bkt, fk)
    loss = ce_tgt + 0.5 * ce_bkt

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": loss.item(),
        "ce_tgt": ce_tgt.item(),
        "ce_bkt": ce_bkt.item(),
        "acc_tgt": (picked_tgt.argmax(-1) == tgt_labels).float().mean().item(),
        "acc_bkt": (picked_bkt.argmax(-1) == fk).float().mean().item(),
        "n_labels": len(flat_b),
    }


def serialize_state(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-games", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--four-player-prob", type=float, default=0.5)
    ap.add_argument("--model-seat-prob", type=float, default=0.5,
                    help="Probability each seat is played by the model (rest lb-1200)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--grad-steps-per-iter", type=int, default=4)
    ap.add_argument("--buffer-size", type=int, default=5000)
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_base = pathlib.Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    kwargs = dict(planet_dim=PLANET_DIM, fleet_dim=FLEET_DIM, global_dim=GLOBAL_DIM,
                  d_model=128, n_heads=4, n_layers=4, n_buckets=4)
    model = OrbitAgent(**kwargs).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"warm-started from {args.init_from}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(args.workers, initializer=_worker_init, initargs=(kwargs,))

    print(f"DAgger imitation: target={args.target_games} workers={args.workers} "
          f"device={device}  model_seat_prob={args.model_seat_prob} "
          f"buf={args.buffer_size}", flush=True)

    buffer: collections.deque = collections.deque(maxlen=args.buffer_size)
    games_done = 0
    iter_count = 0
    t0 = time.time()

    try:
        while games_done < args.target_games:
            weights_bytes = serialize_state(model.cpu())
            model.to(device)
            tasks = [{"n_players": 4 if random.random() < args.four_player_prob else 2,
                      "model_seat_prob": args.model_seat_prob,
                      "weights": weights_bytes}
                     for _ in range(args.workers)]
            results = pool.map(_worker_play, tasks)
            for game_samples in results:
                for s in game_samples:
                    buffer.append(s)
            games_done += args.workers

            stats_agg = {"loss": 0, "ce_tgt": 0, "ce_bkt": 0,
                         "acc_tgt": 0, "acc_bkt": 0, "n_labels": 0, "n_steps": 0}
            for _ in range(args.grad_steps_per_iter):
                if len(buffer) < args.batch:
                    continue
                batch = random.sample(list(buffer), args.batch)
                stats = train_step(model, optimizer, batch, device)
                if stats:
                    for k, v in stats.items():
                        stats_agg[k] = stats_agg.get(k, 0) + v
                    stats_agg["n_steps"] += 1

            iter_count += 1
            dt = time.time() - t0
            rate = games_done / max(dt, 1)
            ns = max(stats_agg["n_steps"], 1)
            print(f"[iter {iter_count:04d}] games {games_done}/{args.target_games}  "
                  f"rate={rate:.2f}g/s  buf={len(buffer)}  "
                  f"loss={stats_agg['loss']/ns:.4f}  "
                  f"acc_tgt={stats_agg['acc_tgt']/ns:.3f}  "
                  f"acc_bkt={stats_agg['acc_bkt']/ns:.3f}  [{dt:.0f}s]",
                  flush=True)

            if iter_count % args.snapshot_every == 0 or games_done >= args.target_games:
                torch.save({"model": model.state_dict(), "kwargs": kwargs,
                            "iter": iter_count, "games_done": games_done,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                           out_base)
                print(f"  → saved {out_base}", flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
