"""Phase-2 PPO self-play trainer for Orbit Wars.

Single-env sequential rollouts (can be vectorised later). Opponent pool
sampled via PFSP. Optional teacher-KL anchor to a frozen BC checkpoint.

Design:
  - We control seat 0; opponent sits in seat 1. Alternated per rollout.
  - Rollout loop: encode obs → sample per-owned-planet (target, bucket)
    → step env with our + opponent actions → store transition.
  - PPO update: CE over target+bucket classes for our seat's actions,
    clipped ratio, GAE advantages.
  - Teacher-KL: KL(π_current || π_teacher) over target_logits, bucket_logits.
  - Opponent pool: start with {random, starter, bc_v1 frozen}. After
    N=5 updates we optionally snapshot current policy into the pool.

Usage:
  python training/ppo_selfplay.py \
      --bc-ckpt training/checkpoints/bc_v2.pt \
      --out training/checkpoints/ppo_v1.pt \
      --updates 20 --eps-per-update 4 \
      --device cuda --mem-fraction 0.20
"""

from __future__ import annotations

import argparse
import copy
import datetime
import math
import pathlib
import random
import sys
import time
from collections import defaultdict, deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent, starter_agent

from training.agent import _encode_obs, load_agent, SHIPS_BUCKETS
from training.model import OrbitAgent, sun_blocker_mask


# -----------------------------------------------------------
# Encoding helpers (mirror training/agent.py _encode_obs but return tensors)
# -----------------------------------------------------------

def obs_to_tensors(obs: dict, device: str = "cpu") -> dict:
    """Run the inference encoder and return a dict of tensors on `device`."""
    (pf, pxy, pids, omask, ff, g, planets_raw, player) = _encode_obs(obs)
    P = pf.shape[0]
    F_ = ff.shape[0]

    planets = torch.from_numpy(pf).unsqueeze(0).to(device)
    planet_xy = torch.from_numpy(pxy).unsqueeze(0).to(device)
    planet_mask = torch.ones((1, P), dtype=torch.bool, device=device)
    if F_ > 0:
        fleets = torch.from_numpy(ff).unsqueeze(0).to(device)
        fleet_mask = torch.ones((1, F_), dtype=torch.bool, device=device)
    else:
        fleets = torch.zeros((1, 1, 9), dtype=torch.float32, device=device)
        fleet_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)
    globals_ = torch.from_numpy(g).unsqueeze(0).to(device)
    owned_mask = torch.from_numpy(omask).unsqueeze(0).to(device)

    return {
        "planets": planets, "planet_xy": planet_xy, "planet_mask": planet_mask,
        "fleets": fleets, "fleet_mask": fleet_mask, "globals": globals_,
        "owned_mask": owned_mask,
        "planets_raw": planets_raw, "player": player,
    }


def sample_action(model: OrbitAgent, tensors: dict, device: str,
                  temperature: float = 1.0, greedy: bool = False):
    """Return:
      env_moves: list to pass to env.step (format [from_id, angle, ships])
      logprob: scalar tensor (sum across sampled per-planet sub-actions)
      value: scalar tensor
      per_planet: dict with tensors needed later for PPO update:
        src_indices (K,), target_sample (K,), bucket_sample (K,)
    """
    tgt_mask = sun_blocker_mask(tensors["planet_xy"], tensors["planet_mask"])
    tgt_logits, bkt_logits, value = model(
        tensors["planets"], tensors["planet_mask"],
        tensors["fleets"], tensors["fleet_mask"],
        tensors["globals"], tgt_mask,
    )
    # tgt_logits: [1, P, P+1]; bkt_logits: [1, P, 4]; value: [1]
    owned = tensors["owned_mask"][0]  # [P]
    src_idx = torch.where(owned)[0]   # [K]

    if src_idx.numel() == 0:
        return [], torch.zeros((), device=device), value.squeeze(0), None

    tgt_logits = tgt_logits[0] / temperature  # [P, P+1]
    bkt_logits = bkt_logits[0] / temperature  # [P, 4]

    t_rows = tgt_logits[src_idx]     # [K, P+1]
    b_rows = bkt_logits[src_idx]     # [K, 4]

    if greedy:
        t_sample = t_rows.argmax(-1)
        b_sample = b_rows.argmax(-1)
    else:
        t_sample = torch.distributions.Categorical(logits=t_rows).sample()
        b_sample = torch.distributions.Categorical(logits=b_rows).sample()

    # Per-planet log-probs, summed for the joint action
    t_logp = F.log_softmax(t_rows, dim=-1).gather(1, t_sample.unsqueeze(1)).squeeze(1)
    b_logp = F.log_softmax(b_rows, dim=-1).gather(1, b_sample.unsqueeze(1)).squeeze(1)
    logprob = (t_logp + b_logp).sum()

    # Translate to env moves: tgt_sample == 0 means "pass", else planet index
    planets_raw = tensors["planets_raw"]
    env_moves = []
    for i in range(src_idx.numel()):
        src_i = int(src_idx[i].item())
        tgt_c = int(t_sample[i].item())
        if tgt_c == 0:
            continue  # pass
        tgt_i = tgt_c - 1
        if tgt_i >= len(planets_raw):
            continue
        src = planets_raw[src_i]
        tgt = planets_raw[tgt_i]
        frac = SHIPS_BUCKETS[int(b_sample[i].item())]
        garrison = int(src[5])
        ships = max(1, int(round(frac * garrison)))
        if ships <= 0 or ships > garrison:
            continue
        angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
        env_moves.append([int(src[0]), float(angle), int(ships)])

    per_planet = {
        "src_idx": src_idx.detach().cpu(),
        "tgt_sample": t_sample.detach().cpu(),
        "bkt_sample": b_sample.detach().cpu(),
    }
    return env_moves, logprob.detach(), value.squeeze(0).detach(), per_planet


def evaluate_logprob_value(model: OrbitAgent, tensors: dict, per_planet: dict,
                           device: str):
    """During PPO update, re-run the model and compute log-prob + value
    for the stored action."""
    tgt_mask = sun_blocker_mask(tensors["planet_xy"], tensors["planet_mask"])
    tgt_logits, bkt_logits, value = model(
        tensors["planets"], tensors["planet_mask"],
        tensors["fleets"], tensors["fleet_mask"],
        tensors["globals"], tgt_mask,
    )
    src_idx = per_planet["src_idx"].to(device)
    tgt_s = per_planet["tgt_sample"].to(device)
    bkt_s = per_planet["bkt_sample"].to(device)
    if src_idx.numel() == 0:
        return torch.zeros((), device=device), value.squeeze(0), None
    t_rows = tgt_logits[0][src_idx]
    b_rows = bkt_logits[0][src_idx]
    t_logp = F.log_softmax(t_rows, dim=-1).gather(1, tgt_s.unsqueeze(1)).squeeze(1)
    b_logp = F.log_softmax(b_rows, dim=-1).gather(1, bkt_s.unsqueeze(1)).squeeze(1)
    logprob = (t_logp + b_logp).sum()
    ent = -(F.softmax(t_rows, -1) * F.log_softmax(t_rows, -1)).sum(-1).mean() \
          - (F.softmax(b_rows, -1) * F.log_softmax(b_rows, -1)).sum(-1).mean()
    return logprob, value.squeeze(0), (t_rows, b_rows, ent)


# -----------------------------------------------------------
# Opponent pool
# -----------------------------------------------------------

class OpponentPool:
    def __init__(self):
        self.entries: list[tuple[str, object]] = [
            ("random", random_agent),
            ("starter", starter_agent),
        ]
        self.wins: dict[str, int] = defaultdict(int)
        self.games: dict[str, int] = defaultdict(int)

    def add(self, name: str, agent_fn):
        self.entries.append((name, agent_fn))

    def sample_pfsp(self) -> tuple[str, object]:
        # P ∝ (1 - winrate)^2, clamped [0.1, 1]
        weights = []
        for name, _ in self.entries:
            g = self.games[name]
            wr = (self.wins[name] / g) if g > 0 else 0.5
            w = max(0.1, (1.0 - wr)) ** 2
            weights.append(w)
        total = sum(weights)
        probs = [w / total for w in weights]
        idx = np.random.choice(len(self.entries), p=probs)
        return self.entries[idx]

    def record(self, name: str, won: bool):
        self.games[name] += 1
        if won:
            self.wins[name] += 1

    def summary(self) -> dict:
        out = {}
        for name, _ in self.entries:
            g = self.games[name]
            out[name] = {"games": g,
                         "win_rate": (self.wins[name] / g) if g > 0 else None}
        return out


# -----------------------------------------------------------
# Rollout
# -----------------------------------------------------------

def _shared_obs(state, player_slot: int) -> dict:
    """Pull the shared world state (on seat 0) and relabel player."""
    raw = state[0].observation
    try:
        d = {k: raw[k] for k in raw}
    except Exception:
        d = dict(raw) if isinstance(raw, dict) else raw
    d["player"] = player_slot
    return d


def run_episode(model: OrbitAgent, opponent_fns: list, device: str,
                shape_reward: bool = True, my_seat: int = 0,
                n_players: int = 2):
    """Play one episode. `opponent_fns` must have length n_players-1
    (one callable per non-learner seat, in seat order excluding my_seat).
    Returns learner's rollout buffer + win flag.
    """
    assert n_players in (2, 4)
    assert len(opponent_fns) == n_players - 1
    agents = ["random"] * n_players  # placeholder; we step manually
    env = make("orbit_wars", debug=False, configuration={})
    # Force the number of agents by passing agents list at reset
    env.reset(num_agents=n_players)

    obs_list, lp_list, val_list, act_list, rew_list = [], [], [], [], []
    prev_my_total = None

    # Map from seat index to opponent_fn (seat != my_seat)
    op_by_seat: dict[int, object] = {}
    op_iter = iter(opponent_fns)
    for s in range(n_players):
        if s != my_seat:
            op_by_seat[s] = next(op_iter)

    while not env.done:
        state = env.state

        # Learner
        my_obs = _shared_obs(state, my_seat)
        tensors = obs_to_tensors(my_obs, device=device)
        env_moves, logp, value, per_planet = sample_action(
            model, tensors, device=device
        )

        # Opponents
        op_actions = {}
        for seat, fn in op_by_seat.items():
            op_obs = _shared_obs(state, seat)
            try:
                op_actions[seat] = fn(op_obs) or []
            except Exception:
                op_actions[seat] = []

        actions = [None] * n_players
        actions[my_seat] = env_moves
        for seat, a in op_actions.items():
            actions[seat] = a
        env.step(actions)

        # Per-step shaping reward — enemies = all other seats, not just seat 0
        my_total = sum(p[5] for p in (state[0].observation.get('planets') or [])
                       if p[1] == my_seat) + sum(
            f[6] for f in (state[0].observation.get('fleets') or [])
            if f[1] == my_seat
        )
        if shape_reward:
            if prev_my_total is not None:
                shaped = max(-1.0, min(1.0, (my_total - prev_my_total) / 100.0))
            else:
                shaped = 0.0
            prev_my_total = my_total
        else:
            shaped = 0.0

        if per_planet is not None:
            obs_list.append(tensors)
            lp_list.append(logp)
            val_list.append(value)
            act_list.append(per_planet)
            rew_list.append(shaped)

    # Terminal reward
    final_r = env.state[my_seat].reward or 0
    if rew_list:
        rew_list[-1] = float(final_r) + (rew_list[-1] if shape_reward else 0.0)
    won = final_r > 0

    return {
        "obs": obs_list,
        "logp": torch.stack(lp_list) if lp_list else torch.zeros(0, device=device),
        "val": torch.stack(val_list) if val_list else torch.zeros(0, device=device),
        "act": act_list,
        "rew": torch.tensor(rew_list, dtype=torch.float32, device=device),
        "won": won,
        "final_r": final_r,
    }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                gamma: float = 0.997, lam: float = 0.95):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        lastgae = delta + gamma * lam * lastgae
        advantages[t] = lastgae
    returns = advantages + values
    return advantages, returns


# -----------------------------------------------------------
# PPO update
# -----------------------------------------------------------

def ppo_update(model: OrbitAgent, teacher: OrbitAgent | None,
               rollouts: list[dict], optimizer: torch.optim.Optimizer,
               device: str,
               clip_coef: float = 0.2, vf_coef: float = 0.5,
               ent_coef: float = 0.01, teacher_kl_coef: float = 0.1,
               epochs: int = 4, minibatches: int = 4):
    # Flatten rollouts into per-step buffers
    obs_all, lp_all, val_all, act_all, rew_all = [], [], [], [], []
    for ro in rollouts:
        if len(ro["obs"]) == 0:
            continue
        obs_all.extend(ro["obs"])
        lp_all.append(ro["logp"])
        val_all.append(ro["val"])
        act_all.extend(ro["act"])
        adv, ret = compute_gae(ro["rew"], ro["val"])
        # Attach adv, ret to a unified array
        for i in range(len(ro["obs"])):
            rew_all.append({"adv": adv[i].detach(), "ret": ret[i].detach()})

    if not obs_all:
        return {"n": 0}

    logp_old = torch.cat(lp_all).detach()
    val_old = torch.cat(val_all).detach()
    adv_t = torch.stack([r["adv"] for r in rew_all])
    ret_t = torch.stack([r["ret"] for r in rew_all])
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    N = len(obs_all)
    opt = optimizer
    stats = defaultdict(list)
    for _ in range(epochs):
        idx = np.random.permutation(N)
        mb_size = max(1, N // minibatches)
        for mb_start in range(0, N, mb_size):
            mb = idx[mb_start:mb_start + mb_size]
            policy_loss = 0.0
            value_loss = 0.0
            ent_sum = 0.0
            kl_sum = 0.0
            count = 0
            for i in mb:
                tensors = obs_all[i]
                per_planet = act_all[i]
                lp_new, v_new, extra = evaluate_logprob_value(
                    model, tensors, per_planet, device
                )
                if extra is None:
                    continue
                t_rows, b_rows, ent = extra
                ratio = torch.exp(lp_new - logp_old[i])
                adv = adv_t[i]
                p1 = ratio * adv
                p2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * adv
                pl = -torch.min(p1, p2)
                vl = 0.5 * (v_new - ret_t[i]) ** 2
                policy_loss = policy_loss + pl
                value_loss = value_loss + vl
                ent_sum = ent_sum + ent
                if teacher is not None and teacher_kl_coef > 0:
                    with torch.no_grad():
                        t_tgt_mask = sun_blocker_mask(
                            tensors["planet_xy"], tensors["planet_mask"])
                        t_tgt, t_bkt, _ = teacher(
                            tensors["planets"], tensors["planet_mask"],
                            tensors["fleets"], tensors["fleet_mask"],
                            tensors["globals"], t_tgt_mask,
                        )
                        src_idx = per_planet["src_idx"].to(device)
                        t_tgt_rows = t_tgt[0][src_idx]
                        t_bkt_rows = t_bkt[0][src_idx]
                    # KL(student || teacher) — both target + bucket heads
                    kl_tgt = F.kl_div(F.log_softmax(t_rows, -1),
                                      F.softmax(t_tgt_rows, -1),
                                      reduction="batchmean")
                    kl_bkt = F.kl_div(F.log_softmax(b_rows, -1),
                                      F.softmax(t_bkt_rows, -1),
                                      reduction="batchmean")
                    kl_sum = kl_sum + 0.5 * (kl_tgt + kl_bkt)
                count += 1
            if count == 0:
                continue
            loss = (policy_loss + vf_coef * value_loss
                    - ent_coef * ent_sum + teacher_kl_coef * kl_sum) / count
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            stats["policy_loss"].append(policy_loss.detach().item() / count)
            stats["value_loss"].append(value_loss.detach().item() / count)
            stats["ent"].append(ent_sum.detach().item() / count)
            stats["kl_teacher"].append(
                kl_sum.detach().item() / count if isinstance(kl_sum, torch.Tensor)
                else 0.0)

    return {"n": N, **{k: float(np.mean(v)) for k, v in stats.items()}}


# -----------------------------------------------------------
# Driver
# -----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-ckpt", required=True,
                    help="Starting + teacher BC checkpoint (.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--eps-per-update", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-fraction", type=float, default=0.20)
    ap.add_argument("--teacher-kl-coef", type=float, default=0.5,
                    help="Decays linearly to 0.05 over full run")
    ap.add_argument("--add-self-every", type=int, default=5,
                    help="Snapshot current policy into pool every N updates")
    ap.add_argument("--snapshot-every", type=int, default=5,
                    help="Save a checkpoint every N updates")
    ap.add_argument("--four-player-prob", type=float, default=0.5,
                    help="Fraction of rollouts run as 4p games (rest 2p)")
    args = ap.parse_args()

    if args.device.startswith("cuda"):
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)

    # Load model + teacher
    ckpt = torch.load(args.bc_ckpt, map_location=args.device, weights_only=False)
    model = OrbitAgent(**ckpt["kwargs"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    teacher = OrbitAgent(**ckpt["kwargs"]).to(args.device)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    pool = OpponentPool()
    # Add the frozen BC as an opponent too
    bc_frozen_agent = load_agent(args.bc_ckpt, device=args.device)
    pool.add("bc_frozen", bc_frozen_agent)

    # Single persistent optimizer — keeps Adam moments across updates
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4,
                                  weight_decay=1e-5)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"PPO self-play  updates={args.updates} "
          f"eps/update={args.eps_per_update} device={args.device}", flush=True)

    for upd in range(args.updates):
        t0 = time.time()
        rollouts = []
        mode_counts = {"2p": 0, "4p": 0}
        for ep in range(args.eps_per_update):
            # 50/50 split 2p vs 4p to expose policy to both regimes
            n_players = 4 if random.random() < args.four_player_prob else 2
            mode_counts[f"{n_players}p"] += 1
            my_seat = random.randint(0, n_players - 1)
            op_names_fns = [pool.sample_pfsp() for _ in range(n_players - 1)]
            op_fns = [f for (_, f) in op_names_fns]
            try:
                ro = run_episode(model, op_fns, device=args.device,
                                 my_seat=my_seat, n_players=n_players)
            except Exception as e:
                print(f"  episode err ({n_players}p): {e}", flush=True)
                continue
            for (name, _) in op_names_fns:
                pool.record(name, ro["won"])
            rollouts.append(ro)
        # Decay teacher KL linearly
        kl_coef = max(0.05, args.teacher_kl_coef * (1.0 - upd / args.updates))
        stats = ppo_update(model, teacher, rollouts, optimizer=optimizer,
                           device=args.device, teacher_kl_coef=kl_coef)

        dt = time.time() - t0
        w = sum(1 for r in rollouts if r["won"])
        print(f"[upd {upd:03d}] {dt:.1f}s  modes={mode_counts}  kl={kl_coef:.3f}  "
              f"stats={ {k: round(v,3) for k,v in stats.items() if k != 'n'} }  "
              f"won={w}/{len(rollouts)}",
              flush=True)

        # Snapshot current policy into pool
        if (upd + 1) % args.add_self_every == 0:
            snap_model = OrbitAgent(**ckpt["kwargs"]).to(args.device)
            snap_model.load_state_dict(copy.deepcopy(model.state_dict()))
            snap_model.eval()
            for p in snap_model.parameters():
                p.requires_grad_(False)
            snap_name = f"self_u{upd+1:03d}"

            def _snap_agent(obs, _m=snap_model, _dev=args.device):
                return _inline_agent(_m, obs, _dev)

            pool.add(snap_name, _snap_agent)
            print(f"  → snapshotted pool entry {snap_name}", flush=True)

        if (upd + 1) % args.snapshot_every == 0 or upd == args.updates - 1:
            torch.save({
                "model": model.state_dict(),
                "kwargs": ckpt["kwargs"],
                "updates": upd + 1,
                "pool_summary": pool.summary(),
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            }, out_path)
            print(f"  → saved {out_path}", flush=True)

    print(f"\ndone  pool_summary={pool.summary()}", flush=True)
    return 0


def _inline_agent(model, obs, device):
    obs_d = obs if isinstance(obs, dict) else dict(obs)
    tensors = obs_to_tensors(obs_d, device=device)
    if not tensors["owned_mask"].any():
        return []
    with torch.no_grad():
        env_moves, _, _, _ = sample_action(model, tensors, device=device,
                                           greedy=True)
    return env_moves


if __name__ == "__main__":
    sys.exit(main())
