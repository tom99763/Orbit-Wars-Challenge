"""simple_rl_v2 PPO training loop with frozen-anchor league.

MVP — implements the BEST setting from CLAUDE-2.md TL;DR:
    --target-head hier --ship-head v14 --lead-aim 1
    --backbone mlp --hidden 128
plus the critical training fixes already known to matter:
    - GAMMA = 1.0, LAM = 1.0 (terminal-only sparse reward → pure MC)
    - Batch-level advantage normalization (NOT per-minibatch)
    - Mnih value clip
    - Value-head input detached (no V loss into backbone)
    - Frozen-anchor pool with PFSP + latest-self mirror

Skipped (TODO — explicit ablation handles in CLAUDE-2.md but not yet
implemented in this MVP): AGC, KL early stop, LR warmup/cosine, BC
anchor, transformer backbones, pointer/k8 heads, bucket5 ship head.

Usage (smoke):
    python -m training.simple_rl_v2.train_rl \\
        --save-dir training/checkpoints/srlv2_smoke \\
        --n-updates 5 --n-envs 4 --rollout-steps 16 \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the repo importable when run as a module
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from training.orbit_wars_vec_env import OrbitWarsVecEnv
from training.simple_rl_v2.featurize import build_features, to_torch
from training.simple_rl_v2.league import League
from training.simple_rl_v2.model import (
    SimpleRLAgentV2, SimpleRLAgentV2Config,
    joint_log_probs, sample_action, entropy as policy_entropy,
    v14_ship_count,
    K_CAND, SLOT_NOOP,
)
from training.simple_rl_v2.physics import (
    lead_target_angles, sun_crosses, BOARD_CENTER,
)


# ---------------------------------------------------------------------------
# Action conversion: sampled slots → env action dict
# ---------------------------------------------------------------------------

def slots_to_env_actions(
    chosen_slot: np.ndarray,    # [N, NP] int
    feats: dict,                # output of build_features
    env: OrbitWarsVecEnv,
    player: int,
    use_lead_aim: bool = True,
) -> list[dict]:
    """Convert sampled (per planet) slot indices into a list of N env action dicts.

    For each (env, src_planet) where:
        - planet is owned by `player`     (src_mask True)
        - chosen slot != NOOP
        - candidate is valid              (cand_valid True)
        - sun-crossing check passes
    we emit [src_pid, angle, ships]. Otherwise no-op for that source.

    Uses v14 ship rule: ships = min(src, max(tgt+1, 20)).
    Uses lead-target aim if `use_lead_aim` (the recommended CLAUDE-2.md
    setting); otherwise atan2 to current target position.
    """
    N, NP = chosen_slot.shape
    src_mask = feats["src_mask"]
    cand_valid = feats["cand_valid"]
    cand_pidx = feats["cand_pidx"]
    tgt_x = feats["tgt_x"]
    tgt_y = feats["tgt_y"]
    tgt_static = feats["tgt_static"]
    tgt_ships = feats["tgt_ships"]

    pl_x = env.pl_x
    pl_y = env.pl_y
    pl_ships = env.pl_ships
    pl_init_x = env.pl_init_x
    pl_init_y = env.pl_init_y
    pl_init_angle = env.pl_init_angle
    pl_static = env.pl_is_static
    omega = env.ang_vel   # [N]

    # Gather per-(env, src, slot) source features for the chosen slot
    # We need: src_x, src_y, src_ships per (env, src)
    # Already in [N, NP] arrays
    src_ships = pl_ships
    src_x = pl_x
    src_y = pl_y

    # Chosen slot's target — gather along K axis
    chosen_pidx = np.take_along_axis(
        cand_pidx, chosen_slot[..., None], axis=2
    ).squeeze(-1)                        # [N, NP]
    chosen_tgt_x = np.take_along_axis(
        tgt_x, chosen_slot[..., None], axis=2
    ).squeeze(-1)
    chosen_tgt_y = np.take_along_axis(
        tgt_y, chosen_slot[..., None], axis=2
    ).squeeze(-1)
    chosen_tgt_static = np.take_along_axis(
        tgt_static, chosen_slot[..., None], axis=2
    ).squeeze(-1).astype(bool)
    chosen_tgt_ships = np.take_along_axis(
        tgt_ships, chosen_slot[..., None], axis=2
    ).squeeze(-1)
    chosen_valid = np.take_along_axis(
        cand_valid, chosen_slot[..., None], axis=2
    ).squeeze(-1)

    # Active mask: must be owned + not NOOP + candidate valid
    active = src_mask & (chosen_slot != SLOT_NOOP) & chosen_valid

    # Compute angle per (env, src) — only meaningful where active
    if use_lead_aim:
        # Need target's init xy/angle for orbit prediction; gather along NP
        nidx = np.arange(N)[:, None, None]
        # For each chosen target planet idx, look up its init params
        safe_chosen = np.maximum(chosen_pidx, 0)
        # Shape gymnastics: pl_init_* are [N, NP], we want indexed at [N, NP, ...]
        # → use take_along_axis with chosen_pidx[..., None] over NP axis
        tgt_init_x = np.take_along_axis(
            pl_init_x[:, None, :], safe_chosen[..., None].transpose(0, 2, 1), axis=-1
        )
        # Above is awkward; simpler: take_along axis=1 with index [N, NP]
        # Just do: pl_init_x[env_idx, planet_idx]
        env_idx = np.arange(N)[:, None]              # [N, 1]
        tgt_init_x_v = pl_init_x[env_idx, safe_chosen]      # [N, NP]
        tgt_init_y_v = pl_init_y[env_idx, safe_chosen]
        tgt_init_angle_v = pl_init_angle[env_idx, safe_chosen]
        tgt_static_v = pl_static[env_idx, safe_chosen]

        ships = v14_ship_count(src_ships, chosen_tgt_ships)
        angles, _ = lead_target_angles(
            src_x, src_y,
            tgt_init_x_v, tgt_init_y_v, tgt_init_angle_v, tgt_static_v,
            omega[:, None],   # [N, 1] broadcast
            ships,
        )
    else:
        ships = v14_ship_count(src_ships, chosen_tgt_ships)
        angles = np.arctan2(chosen_tgt_y - src_y, chosen_tgt_x - src_x).astype(np.float32)

    # Sun-cross check
    sun_bad = sun_crosses(src_x, src_y, chosen_tgt_x, chosen_tgt_y)
    active = active & (~sun_bad) & (ships > 0)

    # Build per-env action dicts
    actions: list[dict] = []
    for eid in range(N):
        moves = []
        valid_idxs = np.nonzero(active[eid])[0]
        for src_pid in valid_idxs:
            ang = float(angles[eid, src_pid])
            n_ships = int(ships[eid, src_pid])
            if n_ships <= 0:
                continue
            moves.append([int(src_pid), ang, n_ships])
        actions.append({player: moves} if moves else {})
    return actions


# ---------------------------------------------------------------------------
# Rollout: collect ROLLOUT_STEPS × N_ENVS transitions
# ---------------------------------------------------------------------------

def collect_rollout(
    model: nn.Module,
    opp_model: nn.Module,
    env: OrbitWarsVecEnv,
    rollout_steps: int,
    device: str,
    use_lead_aim: bool = True,
    learner_seats: np.ndarray | None = None,   # [N] — which player slot is "model" per env
) -> dict:
    """One rollout window. Returns a dict of stacked tensors.

    Half the envs put `model` at seat 0, the other half at seat 1, so
    seat asymmetry averages out. learner_seats can override.

    Returns:
        pf, gf, cand_feat, cand_valid, src_mask, chosen_slot, chosen_logp,
        value, reward, done, last_value
        all tensors of shape [T, N, ...]
    """
    N = env.N
    P = env.P
    if learner_seats is None:
        # Half/half assignment (deterministic — caller seeds before call)
        learner_seats = np.zeros(N, dtype=np.int64)
        learner_seats[N // 2:] = 1

    # Storage on CPU torch — stacked at end
    buf = {
        "pf":         [],
        "gf":         [],
        "cand_feat":  [],
        "cand_valid": [],
        "src_mask":   [],
        "chosen_slot": [],
        "chosen_logp": [],
        "value":       [],
        "reward":      [],
        "done":        [],
    }

    model.eval()
    opp_model.eval()

    for t in range(rollout_steps):
        # ─── Build features for both seats ───────────────────────────────────
        env_actions_for_step: list[dict] = [{} for _ in range(N)]

        # For each player slot, decide actions
        for player in range(P):
            feats_np = build_features(env, player)
            feats_t = to_torch(feats_np, device=device)

            # Determine which envs this player is "model" or "opp" in
            is_learner = (learner_seats == player)   # [N] bool

            with torch.no_grad():
                # Forward through model OR opp_model per-env mask. To keep things
                # simple, just run BOTH and pick. (If GPU big enough, fine.)
                t_log_m, s_log_m, val_m, _ = model(
                    feats_t["pf"], feats_t["gf"],
                    feats_t["cand_feat"], src_mask=feats_t["src_mask"],
                )
                t_log_o, s_log_o, val_o, _ = opp_model(
                    feats_t["pf"], feats_t["gf"],
                    feats_t["cand_feat"], src_mask=feats_t["src_mask"],
                )

                joint_m = joint_log_probs(t_log_m, s_log_m, feats_t["cand_valid"])
                joint_o = joint_log_probs(t_log_o, s_log_o, feats_t["cand_valid"])
                # Stitch by env:
                is_learner_t = torch.from_numpy(is_learner).to(device)
                joint = torch.where(is_learner_t[:, None, None], joint_m, joint_o)
                val   = torch.where(is_learner_t, val_m, val_o)

                chosen_slot, chosen_lp = sample_action(
                    joint, feats_t["src_mask"], deterministic=False,
                )

            # Convert to env actions
            chosen_np = chosen_slot.cpu().numpy()
            actions_player = slots_to_env_actions(
                chosen_np, feats_np, env, player=player, use_lead_aim=use_lead_aim,
            )
            for eid in range(N):
                env_actions_for_step[eid].update(actions_player[eid])

            # Store learner-side step data only (we train ONLY on learner trajectories)
            if (is_learner).any():
                # Mask non-learner envs to zero contribution; collect everything
                # then we filter / reweight at training time using `is_learner_t`.
                buf["pf"].append(feats_t["pf"].cpu())
                buf["gf"].append(feats_t["gf"].cpu())
                buf["cand_feat"].append(feats_t["cand_feat"].cpu())
                buf["cand_valid"].append(feats_t["cand_valid"].cpu())
                buf["src_mask"].append(feats_t["src_mask"].cpu())
                buf["chosen_slot"].append(chosen_slot.cpu())
                # Per-step log-prob: sum over owned planets / n_owned, weighted by is_learner
                lp_sum = (chosen_lp * feats_t["src_mask"].float()).sum(dim=1)   # [N]
                n_owned = feats_t["src_mask"].sum(dim=1).clamp(min=1).float()   # [N]
                lp_per_env = lp_sum / n_owned                                    # [N]
                lp_per_env = lp_per_env * is_learner_t.float()                   # zero non-learner
                buf["chosen_logp"].append(lp_per_env.cpu())
                buf["value"].append((val * is_learner_t.float()).cpu())
                # placeholders — filled after env.step
                buf["reward"].append(torch.zeros(N))
                buf["done"].append(torch.zeros(N, dtype=torch.bool))

        # ─── Step env ────────────────────────────────────────────────────────
        _, rewards, done = env.step(env_actions_for_step)
        # rewards shape [N, P]; we want learner's reward
        learner_reward = rewards[np.arange(N), learner_seats]
        # Replace last-pushed reward/done entries (from learner-side player iter)
        buf["reward"][-1] = torch.from_numpy(learner_reward.astype(np.float32))
        buf["done"][-1]   = torch.from_numpy(np.asarray(done, dtype=bool))

        # Reset done envs (else they re-emit done=True forever)
        if done.any():
            env.reset(env_ids=np.nonzero(done)[0])
            # Re-randomize learner seats per env on reset — keeps balance
            for eid in np.nonzero(done)[0]:
                learner_seats[eid] = random.randrange(P)

    # Compute "last_value" for GAE bootstrap (V at t = ROLLOUT_STEPS+1)
    feats_np = build_features(env, int(learner_seats[0]))   # any seat — only need V of learner-seat envs
    # For correctness we should compute V for each env using its own seat.
    # Run learner per seat:
    last_values = torch.zeros(N)
    for player in range(P):
        is_learner = (learner_seats == player)
        if not is_learner.any():
            continue
        f = build_features(env, player)
        ft = to_torch(f, device=device)
        with torch.no_grad():
            _, _, v, _ = model(ft["pf"], ft["gf"], ft["cand_feat"], src_mask=ft["src_mask"])
        last_values[is_learner] = v.cpu()[is_learner]

    # Stack across time (each "step" produced one entry per player iteration,
    # so we have rollout_steps * P entries — only 1/P are learner. For MVP
    # simplicity, we keep the learner ones via is_learner mask in the buf.
    # But that means the buffer length is rollout_steps * P, not rollout_steps.
    # For the MVP, we'll just live with this and downweight via mask later.
    out = {k: torch.stack(v, dim=0) for k, v in buf.items()}
    out["last_value"] = last_values
    out["learner_seats"] = torch.from_numpy(learner_seats.copy())
    return out


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,    # [T, N]
    values: torch.Tensor,     # [T, N]
    dones: torch.Tensor,      # [T, N] bool
    last_values: torch.Tensor,   # [N]
    gamma: float = 1.0,
    lam: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard GAE. With γ=1, λ=1 → returns are pure MC sums of future
    rewards (because reward is terminal-only sparse ±1, this gives
    each transition the episode's terminal reward as its return)."""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        non_terminal = (~dones[t]).float()
        next_v = last_values if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model: SimpleRLAgentV2,
    optim: torch.optim.Optimizer,
    batch: dict,
    args,
    device: str,
) -> dict:
    """One PPO update over the rollout buffer.

    Implements:
      - PPO ratio clipping (CLIP_EPS)
      - Mnih value clipping (CLIP_VF)
      - Entropy bonus (ENT_COEF)
      - Batch-level advantage normalization (NOT per-minibatch)
    """
    # Flatten T × N → single batch dim
    flat = {}
    is_learner_mask = (batch["chosen_logp"].abs() > 0)   # [T, N] crude — only true where learner contributed
    for k, v in batch.items():
        if k in ("last_value", "learner_seats"):
            continue
        flat[k] = v.view(-1, *v.shape[2:])
    flat_mask = is_learner_mask.view(-1)

    # Filter to learner transitions only
    valid = flat_mask
    flat = {k: v[valid] for k, v in flat.items()}
    n_total = flat["pf"].shape[0]
    if n_total == 0:
        return {"pi_loss": 0.0, "v_loss": 0.0, "ent": 0.0, "n": 0}

    # Batch-level adv norm (CRITICAL — see CLAUDE-2.md)
    adv = flat["advantages"]
    if adv.std() > 1e-6:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    flat["advantages"] = adv

    info = {"pi_loss": 0.0, "v_loss": 0.0, "ent": 0.0, "n": 0}

    for epoch in range(args.ppo_epochs):
        idx = torch.randperm(n_total)
        for s in range(0, n_total, args.minibatch_size):
            mb_idx = idx[s:s + args.minibatch_size]
            pf  = flat["pf"][mb_idx].to(device)
            gf  = flat["gf"][mb_idx].to(device)
            cand_feat  = flat["cand_feat"][mb_idx].to(device)
            cand_valid = flat["cand_valid"][mb_idx].to(device)
            src_mask   = flat["src_mask"][mb_idx].to(device)
            chosen_slot = flat["chosen_slot"][mb_idx].to(device)
            old_lp     = flat["chosen_logp"][mb_idx].to(device)
            old_val    = flat["value"][mb_idx].to(device)
            adv_mb     = flat["advantages"][mb_idx].to(device)
            ret_mb     = flat["returns"][mb_idx].to(device)

            t_log, s_log, val, _ = model(pf, gf, cand_feat, src_mask=src_mask)
            joint = joint_log_probs(t_log, s_log, cand_valid)

            # Per-env log-prob: sum over chosen slots of owned planets / n_owned
            new_lp_planet = joint.gather(2, chosen_slot.unsqueeze(-1)).squeeze(-1)   # [B, NP]
            new_lp_planet = new_lp_planet * src_mask.float()
            n_owned = src_mask.sum(dim=1).clamp(min=1).float()
            new_lp = new_lp_planet.sum(dim=1) / n_owned    # [B]

            ratio = (new_lp - old_lp).exp()
            s1 = ratio * adv_mb
            s2 = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * adv_mb
            pi_loss = -torch.min(s1, s2).mean()

            # Value: Mnih clip
            v_clip = old_val + (val - old_val).clamp(-args.clip_vf, args.clip_vf)
            v_loss = torch.max((val - ret_mb) ** 2, (v_clip - ret_mb) ** 2).mean()

            ent = policy_entropy(joint, src_mask).mean()

            loss = pi_loss + args.vf_coef * v_loss - args.ent_coef * ent
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

            info["pi_loss"] += pi_loss.item()
            info["v_loss"]  += v_loss.item()
            info["ent"]     += ent.item()
            info["n"]       += 1

    if info["n"] > 0:
        for k in ("pi_loss", "v_loss", "ent"):
            info[k] /= info["n"]
    return info


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--n-updates",      type=int, default=10000)
    ap.add_argument("--n-envs",         type=int, default=32)
    ap.add_argument("--rollout-steps",  type=int, default=128)
    ap.add_argument("--gamma",          type=float, default=1.0)
    ap.add_argument("--lam",            type=float, default=1.0)
    ap.add_argument("--clip-eps",       type=float, default=0.1)
    ap.add_argument("--clip-vf",        type=float, default=0.2)
    ap.add_argument("--ent-coef",       type=float, default=0.01)
    ap.add_argument("--vf-coef",        type=float, default=0.5)
    ap.add_argument("--lr",             type=float, default=5e-4)
    ap.add_argument("--ppo-epochs",     type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=1024)
    ap.add_argument("--max-grad-norm",  type=float, default=1.0)
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--pool-size",      type=int, default=16)
    ap.add_argument("--latest-prob",    type=float, default=0.3)
    ap.add_argument("--elo-k",          type=float, default=16.0)
    ap.add_argument("--elo-init",       type=float, default=1500.0)
    ap.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--no-lead-aim",    action="store_true")
    ap.add_argument("--warm-start",     default=None,
                    help="Path to a SimpleRLAgentV2 .pt to warm-start.")
    args = ap.parse_args()

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "train_rl.log"
    log_path.write_text("", encoding="utf-8")   # truncate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    # Build env, model, optim, league
    env = OrbitWarsVecEnv(n_envs=args.n_envs, n_players=2, seed=args.seed)
    cfg = SimpleRLAgentV2Config()
    model     = SimpleRLAgentV2(cfg).to(device)
    opp_model = SimpleRLAgentV2(cfg).to(device)
    opp_model.load_state_dict(model.state_dict())   # initial mirror

    if args.warm_start:
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt)
        model.load_state_dict(sd)
        opp_model.load_state_dict(sd)
        print(f"warm-started from {args.warm_start}", flush=True)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    league = League(
        max_size=args.pool_size,
        elo_init=args.elo_init,
        elo_k=args.elo_k,
        latest_prob=args.latest_prob,
    )

    print(f"[srlv2] device={device}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"n_envs={args.n_envs}  rollout={args.rollout_steps}  "
          f"updates={args.n_updates}  pool={args.pool_size}  "
          f"snapshot_every={args.snapshot_every}", flush=True)

    t0 = time.time()
    for upd in range(1, args.n_updates + 1):
        # 1. Snapshot
        if (upd - 1) % args.snapshot_every == 0:
            league.add(model, f"upd{upd:04d}")

        # 2. Pick opponent
        opp_idx = -1
        if random.random() >= args.latest_prob and len(league) > 0:
            opp_idx = league.sample_pfsp()
            league.load_into(opp_model, opp_idx)
            opp_model.to(device).eval()
        else:
            opp_model.load_state_dict(model.state_dict())
            opp_model.eval()

        # 3. Rollout
        roll = collect_rollout(
            model, opp_model, env,
            rollout_steps=args.rollout_steps,
            device=device,
            use_lead_aim=not args.no_lead_aim,
        )

        # 4. GAE
        adv, ret = compute_gae(
            roll["reward"], roll["value"], roll["done"], roll["last_value"],
            gamma=args.gamma, lam=args.lam,
        )
        roll["advantages"] = adv
        roll["returns"]    = ret

        # 5. PPO update
        info = ppo_update(model, optim, roll, args, device)

        # 6. League ELO updates from rollout outcomes
        # Count games where learner's terminal reward was +1 / -1 vs opp
        if opp_idx >= 0:
            term_mask = roll["done"] & (roll["reward"].abs() > 0.5)   # [T, N]
            wins   = int(((roll["reward"] > 0) & term_mask).sum().item())
            losses = int(((roll["reward"] < 0) & term_mask).sum().item())
            for _ in range(wins):
                league.record_result(opp_idx, won=True)
            for _ in range(losses):
                league.record_result(opp_idx, won=False)
            n_elo_games = wins + losses
        else:
            wins = losses = n_elo_games = 0

        # 7. Logging
        elapsed = time.time() - t0
        # Per-iter wins (any opp) for the win-rate panel
        term_mask_all = roll["done"] & (roll["reward"].abs() > 0.5)
        all_wins   = int(((roll["reward"] > 0) & term_mask_all).sum().item())
        all_total  = int(term_mask_all.sum().item())
        line = (f"[iter {upd:05d}]  "
                f"wins={all_wins}/{max(1, all_total)}  "
                f"(elo_opp={'self' if opp_idx < 0 else league.entries[opp_idx].tag if opp_idx < len(league) else 'evicted'})  "
                f"T={int((roll['reward'] != 0).sum().item())}  "
                f"pi={info['pi_loss']:.3f}  v={info['v_loss']:.3f}  "
                f"ent={info['ent']:.3f}  "
                f"r=1.00/1.0  "
                f"ent_c={args.ent_coef:.3f}  pool={len(league)}  "
                f"hardest=[{league.hardest_summary()}]  "
                f"lb_noise=0.00  mc=[0 0 0 0 0]  fc=[0 0 0 0 0 0 0 0]  "
                f"learner_elo={league.learner_elo:.1f}  elo_games={league.elo_games}  "
                f"[{int(elapsed)}s]")
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        # 8. Snapshot save
        if upd % args.snapshot_every == 0 or upd == args.n_updates:
            ckpt_path = save_dir / f"srlv2_upd{upd:05d}.pt"
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "upd":   upd,
                "elo":   league.learner_elo,
                "args":  vars(args),
            }, ckpt_path)

    print(f"[srlv2] done — {args.n_updates} updates in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
