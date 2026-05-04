# simple_rl_v2 — Reproduction Guide

This directory contains a self-play PPO agent for Kaggle Orbit Wars, with a
fully-ablated training methodology. The goal: ELO that **continuously rises**
(like AlphaStar `league/elo_learner` plot), not oscillates around 1500.

> **TL;DR for next session**: best setting is `--target-head hier --ship-head v14 --lead-aim 1 --backbone mlp --hidden 128` plus all training fixes (batch adv-norm, KL stop, V-clip, AGC, PFSP league). See `train_rl.py` defaults. 600-upd ablation has high seed variance; need 3000+ upd or multi-seed averaging for stable conclusions.

---

## Files

| File | Purpose |
|---|---|
| `model.py` | `SimpleRLAgentV2` with configurable target/ship/backbone heads |
| `train_rl.py` | Main PPO training loop with all toggles |
| `league.py` | PFSP pool + frozen-anchor ELO bookkeeping |
| `physics.py` | Lead-target aim + sun-crossing masking (numpy vectorized) |
| `eval_h2h.py` | Pairwise head-to-head evaluation |
| `eval_elo_full.py` | Full round-robin with proper Elo (start=600, K=32) |
| `run_ablations.sh` | 9-run training-fix + action-design ablation orchestration |
| `run_backbones.sh` | 4-run backbone (MLP/attn/GTrXL/MLP-big) ablation |
| `run_long.sh` | Long-form 100k-update training |
| `parse_results.py` | Parses `train_rl.log` files into comparison tables |

---

## Quickest reproduction

```bash
# Single best-setting run, ~14 hr at 10k upds:
python -m training.simple_rl_v2.train_rl \
    --save-dir save/best_run \
    --target-head hier --ship-head v14 --lead-aim 1 \
    --backbone mlp --hidden 128 \
    --adv-norm batch --use-kl-stop 1 --use-agc 1 --use-v-clip 1 \
    --league pfsp \
    --n-updates 10000 --device cuda:0
```

---

## Vec Environment (`training/orbit_wars_vec_env.py`)

Custom **numpy-vectorized** Orbit Wars sim. Replaces `kaggle_environments` for
training (~30-80× faster at N_ENVS=32). Game mechanics matched to
`lb1200_agent.py` constants (sun radius 10, fleet speed formula, combat rules).

### API used by trainer

```python
env = OrbitWarsVecEnv(N_ENVS)
env.reset(env_ids=None)    # all or subset
pf, pad_mask, src_mask, val_mask, gf, ships = env.get_batched_features(player_idx)
#   pf       [N, NP, 8]    planet features (xy/100, log1p(ships)/log1p(1000), prod/5,
#                          is_me, is_enemy, is_neutral, valid)
#   src_mask [N, NP]       owned planets
#   gf       [N, 6]        global resource features
rewards, done = env.step_fast(actions)
#   actions: list of N dicts {player_idx: [[src_id, angle_rad, n_ships], ...]}
#   rewards [N, 2]  done [N]
obs = env.get_obs_dict(eid)   # Kaggle-compatible (only used for rule-based opponents)
```

### Direct attribute access (for fast feature engineering)

```python
env.pl_x, env.pl_y       # [N, NP] coords (orbiting + static)
env.pl_ships, env.pl_prod # [N, NP]
env.pl_valid, env.pl_owner, env.pl_radius
env.pl_is_static         # [N, NP] bool — these don't rotate
env.ang_vel              # [N] per-env angular velocity for orbiting planets
env.NP                   # max planets per env (40)
env.step_num             # [N] current step
```

### Game characteristics

- 500 turns max, terminal reward ±1 (most ships at end wins)
- 20-40 planets per game, comets spawn at steps 50/150/250/350/450
- Sparse reward: r=0 every step except terminal
- Half of planets orbit (inner), half static (outer)
- Fleet speed `1 + 5·(log(ships)/log(1000))^1.5` capped at 6

---

## Training Architecture (`train_rl.py`)

### Core loop

```
for upd in range(n_updates):
    # 1. Maybe snapshot model into league pool, sample new opponent
    if upd % SNAPSHOT_EVERY == 0: pool.add(model.state_dict())
    opp_idx = league.sample_pfsp()
    model_opp.load_state_dict(pool.entry(opp_idx).state_dict)

    # 2. Rollout: 128 steps × 32 envs (= 4096 transitions)
    #    Half envs: model plays P0, opp plays P1
    #    Half envs: model plays P1, opp plays P0
    #    Stores (pf, gf, src_mask, cand_feat, cand_valid, tgt, ship, lp, val, rew, done)

    # 3. GAE bootstrap (gamma=1, lambda=1 for pure MC)
    returns, advantages = compute_gae(rewards, values, dones, last_val, gamma=1, lam=1)

    # 4. Batch-level adv normalization (CRITICAL — see Findings below)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 5. PPO update: 4 epochs × 4 minibatches, with:
    #    - Mnih value clipping
    #    - Approx-KL early stop at 1.5 × target_kl
    #    - AGC (adaptive gradient clipping per tensor)

    # 6. Update learner ELO from games against current opp
```

### Action heads (combinable)

| `--target-head` | Description |
|---|---|
| `k8` | Categorical over K=8 nearest candidates (NOOP, 3 enemy, 3 neutral, 1 friend) |
| `pointer` | Categorical over all NP planets + 1 NOOP, dot-product attention |
| `hier` | **Best** — 4-way action type (NOOP/SNIPE/EXPAND/REINFORCE) × sub-target |

| `--ship-head` | Description |
|---|---|
| `v14` | **Best** — fixed sniper rule `min(src, max(tgt+1, 20))` |
| `bucket5` | Categorical over {snipe, 25%, 50%, 75%, 100%} |

| `--backbone` | Description |
|---|---|
| `mlp` | **Best** — 2-layer MLP per planet, hidden=128 |
| `mlp_big` | Same but hidden=256 (more params, similar performance) |
| `attn` | 2-layer Pre-LN self-attention. **Collapses** without gating |
| `gtrxl` | Self-attention + GRU-style gating (bias init = -2). Recovers from `attn` failure but still worse than MLP at 600 upd |

`physics.lead_target_angles()` predicts where target will be at fleet
arrival time, using fleet_speed × dist iteration. **Huge effect** —
without lead aim, agent loses ~95% of games to lead-aim version.

### Hyperparameters (defaults work)

```python
N_ENVS         = 32     # half P0, half P1
ROLLOUT_STEPS  = 128
GAMMA          = 1.0    # must be for terminal-only reward
LAM            = 1.0    # pure MC return for completed episodes
CLIP_EPS       = 0.1    # PPO ratio clip (transformer-friendly default)
CLIP_VF        = 0.2    # value clip (Mnih)
TARGET_KL      = 0.015  # 1.5× early stop
ENT_COEF       = 0.01
VF_COEF        = 0.5
AGC_FACTOR     = 0.1    # NFNets paper uses 0.01 — TOO TIGHT for small model
LR             = 5e-4   # Higher than typical 3e-4 worked better
LR_WARMUP      = 50     # Most LLM warmup tutorials say 1000 — way too long for 0.05M model
LR_FINAL       = 1e-4   # cosine target
PPO_EPOCHS     = 4
MINIBATCH_SIZE = 1024

POOL_SIZE      = 16     # frozen anchor pool
SNAPSHOT_EVERY = 50     # add learner snapshot every N upds
LATEST_PROB    = 0.3    # 30% games vs latest-self mirror (no ELO update)
PFSP_P         = 2.0    # weight = (1 - wr)^p + 0.05
ELO_K          = 16.0
```

---

## League System (`league.py`)

The key to **continuously rising ELO**. Every Claude session that doesn't
implement this correctly will see oscillation around 1500.

### Mechanism

1. Pool stores frozen state_dicts. Each entry has a `snapshot_elo` field set
   to the **learner's ELO at snapshot time**. Anchor ELOs NEVER update.
2. Learner has its own ELO (starts 1500), updated only on games vs anchors.
3. Every SNAPSHOT_EVERY=50 upds, current model is frozen into pool with
   current learner_elo.
4. Per rollout: PFSP-sample one anchor, weight ∝ (1 - wr_vs_anchor)^2 + 0.05,
   focusing on opponents we lose to. 30% chance of latest-self mirror (skip
   ELO update).
5. After each game: standard Elo update for learner only,
   `learner_elo += K × (score - expected)` where expected uses anchor ELO.

### Why this gives the "purple→cyan" line transition (image reference)

- Pure self-play oscillates: opponent improves at same rate, WR ≈ 50% always.
- Fixed external opponent (e.g. `starter_agent`) saturates: once you beat
  it 100%, ELO plateaus.
- **Frozen-anchor pool**: as learner improves, more anchors of varying
  strengths exist. Beating old self gives Elo gain; failing to beat recent
  self stays at 50%. Net: monotonic rise as long as learner improves.

### **DO NOT**:
- Add `starter_agent` or rule-based opponents to the pool. User explicitly
  vetoed: "你把 starter 加入到對手一定是錯的".
- Update anchor ELOs (then it's just regular self-play).
- Use uniform pool sampling with no PFSP — anchors you've already mastered
  waste rollouts.

---

## Evaluation (`eval_h2h.py` + `eval_elo_full.py`)

**League ELO is noisy** (within-run, vs evolving pool). Always do
post-training H2H to get ground truth.

```bash
# Pairwise H2H (used per-batch in run_ablations.sh)
python -m training.simple_rl_v2.eval_h2h \
    save/run_A/rl_iter000600.pt save/run_B/rl_iter000600.pt \
    --games 10

# Full round-robin Elo (used at end of ablation)
python -m training.simple_rl_v2.eval_elo_full \
    ckpt1.pt ckpt2.pt ... \
    --games 20 --start-elo 600 --k 32
```

### Eval matrix design

- N games per pair, **half as P0 half as P1** (handles seat asymmetry)
- Default 20 games per pair × 36 pairs (9-way) = 720 games, ~8 min on cuda:0
- Elo computed online: each game's outcome processed sequentially through
  Elo update. Both players' ratings move (this is "real" Elo, unlike
  league ELO which only updates learner).

### Loading ckpts (`load_model` in `eval_h2h.py`)

Reads `args` dict from ckpt to reconstruct:
- `target_head` (k8/pointer/hier)
- `ship_head` (v14/bucket5)
- `hidden`
- `backbone` (mlp/attn/gtrxl/mlp_big)
- `lead_aim` (0/1)

There's a state_dict key remapper for legacy MLP ckpts (when `src_enc` was
`nn.Sequential` not `MLPBackbone(net=...)`).

---

## Findings & Ablation Results

### Phase 1 — training fixes (all use k8 + v14 + lead-aim, 600 upd)

| Tag | Setup | League ELO | H2H Elo (start 600) |
|---|---|---|---|
| T0 | minibatch adv-norm, no KL stop, no AGC, no V-clip, uniform pool | 1662 | 368 |
| T1 | + batch adv-norm + KL stop + V-clip | 1718 | 379 |
| T2 | + AGC | 1685 | 507 |
| T3 | + PFSP league | 1893 | 614 |

**Lessons**:
- Batch adv-norm + KL stop + V-clip together = ~+130 ELO over T0
- AGC's effect is noisy in league ELO but H2H confirms +130 ELO over T1
- PFSP > uniform sampling once pool diversifies (upd 200+)

### Phase 2 — action design (all use T3 fixes, 600 upd)

| Tag | Action | League ELO | H2H Elo (9-way, start 600) |
|---|---|---|---|
| Am1 | T3 but **direct angle** (no lead) | 1667 | **337** |
| A0 | k8 + v14 + lead | 1826 | 650 |
| A1 | k8 + bucket5 + lead | 1658 | 854 |
| A2 | pointer + v14 + lead | 1707 | 542 |
| A3 | **hier + v14 + lead** | 1846 | **1147** |

**Lessons**:
- **Lead-target aim is the single biggest fix** (+800 H2H Elo). Without it,
  fleets miss orbiting planets.
- **Hierarchical action type = best** (NOOP/SNIPE/EXPAND/REINFORCE +
  sub-target). Semantic structure helps PPO learn.
- **Pointer head underperforms K=8** at 600 upd (might need more upd)
- **Ship bucket5 (A1)** somewhat worse on H2H vs hier (854 vs 1147),
  surprising given v14 ship rule is fixed. Suggests learning the ship
  count adds noise without much gain when target choice is good.

### Backbone ablation (all use A3 setting, 600 upd)

| Tag | Backbone | params | League ELO | H2H Elo (5-way, start 600) |
|---|---|---|---|---|
| B0 | MLP h=128 | 55k | 1962 | 327 |
| B1 | Self-attn 2L Pre-LN | 304k | 1653 | 370 |
| B2 | GTrXL gated | 370k | 1738 | 619 |
| B3 | MLP h=256 | 209k | 2043 | 777 |
| (A3 from prev ablation, same setting as B0) | MLP h=128 | 55k | — | **907** |

**Lessons**:
- **Plain attention (B1) collapses without gating**, confirming GTrXL
  literature. Stable Pre-LN alone isn't enough for RL.
- **GTrXL > plain attn** (+250 Elo) — gating is the active ingredient.
- **MLP > all transformers at 600 upd**. Probably need much longer training
  for transformer to overtake MLP, or much larger problem.
- **B0 vs A3 (same setting different seed) differs by 580 Elo**. Conclusion:
  600 upd is **way under-trained**, results dominated by seed noise.

### What DIDN'T work (warning to next session)

1. **AGC clip_factor=0.01** (NFNets default) — too tight for 0.05M model. Use 0.1.
2. **LR warmup 1000** — too long for tiny model. Use 50.
3. **LR=1e-4** — too low. Use 5e-4.
4. **Per-minibatch advantage normalization** — collapses pi_loss with sparse
   reward + GAMMA=1. Use batch-level.
5. **`env.reset()` at start of each rollout** — kills in-progress games.
   Only reset done envs after each step.
6. **Adding `starter_agent` to pool** — user vetoed. League must be self-snapshot only.
7. **`pointer + bucket5`** combo — ship features need slot context. Disabled.
8. **GAMMA=0.99 with sparse terminal reward** — return decays to 0.007 by step 499.
9. **LAM=0.95** for sparse reward — propagation decays to 0.001 over 128 steps.
   Use LAM=1.0 (pure MC).
10. **OrbitWars `kaggle_environments`** — 30-80× slower than custom vec env.
    Don't use for training.

---

## Implementation Quirks

### Action conversion (`to_actions_vec`)

Per (env, src_planet) pair, given chosen target index:

```python
# 1. Get target planet idx
if target_head == "pointer":
    ti = chosen_k    # already a planet idx (NP = NOOP)
else:  # k8 or hier
    ti = cand_pidx[b, src, chosen_k]    # cand_pidx[..., 0] = -1 (NOOP)

if ti < 0: skip   # NOOP

# 2. Compute ship count
if ship_head == "v14":
    send = min(src_ships, max(tgt_ships + 1, 20))
elif ship_head == "bucket5":
    send = bucket_to_count(chosen_ship_bucket, src_ships, tgt_ships)

# 3. Compute angle (lead-target physics OR direct)
if lead_aim:
    angle = lead_target_angles(src_xy, tgt_xy, target_static, omega, send)
else:
    angle = atan2(tgt_y - src_y, tgt_x - src_x)

# 4. Sun-cross masking
if sun_crosses(src_xy, tgt_xy):
    skip   # don't waste fleet
```

### Hierarchical action probability (`_logits_hier`)

Returns log-probs not logits, but Categorical(logits=log_probs) works fine
since softmax of log_probs that sum to 0 is invariant.

```
type_logp[B, NP, 4] = log_softmax(type_head(enc))
sub_logp[B, NP, K]  = log_softmax(scorer(enc, cand_feat) within type group)
joint[B, NP, K]     = type_logp[type_for_k] + sub_logp - logsumexp(over k)
```

### Build candidates (`build_candidates_vec`)

Fully numpy-vectorized using `np.argpartition` (O(B·NP·NP) compute, but no
Python loop over batch). 0.3 ms per call for B=16, NP=40.

K=8 slot layout:
- Slot 0: NOOP (always valid, zero feat, is_noop=1)
- Slot 1-3: 3 nearest enemies
- Slot 4-6: 3 nearest neutrals
- Slot 7: 1 nearest friend (excluding self)

Candidates with no available target (e.g., game has only 2 friends → friend
slot 7 invalid for one of them) are masked with `-inf` logit.

---

## Speed Benchmarks (cuda:0, single process)

| Config | s/upd | Notes |
|---|---|---|
| MLP h=128 (A3, B0) | 5.0-5.7 | 4096 transitions, 4 PPO epochs, ~14 games completed |
| MLP h=256 (B3) | 6.0 | More forward time, same data |
| Attn h=128 (B1) | 6.0 | 2 attention layers |
| GTrXL h=128 (B2) | 6.4 | Adds ~0.4s for gating |
| Episode-based (deprecated) | 21.4 | Wait for ALL 32 games to finish |

3-parallel on cuda:0 → individual run becomes ~7-8s/upd (interference from
shared GPU + CPU build_candidates). Total throughput ~3× higher.

---

## Things to try next (open problems)

1. **Long training** (10k-100k upd) to see real ELO ceiling. 600 upd shows
   strong seed noise. The user image suggests ~3500 ELO achievable.
2. **Multi-seed averaging** for reliable ablation (3 seeds × 600 upd is
   probably more informative than 1 seed × 1800 upd).
3. **PopArt value normalization** — recommended by MAPPO paper for
   evolving opponent distributions. Not yet tried.
4. **PPG (Phasic Policy Gradient)** — separate aux phase for V learning
   without trashing policy. ~25% extra compute but better V.
5. **Symlog two-hot critic** (DreamerV3) — scale-invariant V regression,
   avoids reward-scale sensitivity.
6. **Warmstart from BC** — would need expert trajectories. We have lb1200
   self-play replays in `trajectories/`, but BC is sometimes useless if
   architecture differs from expert.
7. **Set Transformer ISAB** — proper permutation-invariant set encoder
   (current attention is just standard MHSA). Could be more sample
   efficient at variable planet counts.

---

## Operational notes

- Long-running training: `bash run_long.sh` (100k upd × 3 settings, ~7 days).
  Check progress with `tail -f save/long_*/A3/train_rl.log`.
- All logs follow this format: `[HH:MM:SS] upd N pi=X vf=Y ent=Z kl=W
  adv_std=A ret_std=B learner_elo=E pool=P games=G lr=L t=Ts`
- Checkpoint naming: `rl_iter{N:06d}.pt` saved every 100 upds + final
  `rl_iter{N:06d}_final.pt`. Both contain `{model, opt, upd, elo, args}`.
- Disk space: ~5MB per ckpt × 1000 ckpts per 100k run × 3 runs = 15GB.
  Tune `SAVE_EVERY` in `train_rl.py` if low disk.
- `args` dict in ckpt is critical for `load_model` to work. Don't strip it.

---

## Research references (consulted before this implementation)

1. **AlphaStar** (Vinyals et al. 2019, Nature) — League training + PFSP
2. **The 37 Implementation Details of PPO** (Huang et al. 2022, ICLR Blog) —
   batch adv-norm, value clip, KL early stop, init scale
3. **Stabilizing Transformers for RL / GTrXL** (Parisotto et al. 2020) —
   gated residuals
4. **NFNets / AGC** (Brock et al. 2021) — adaptive per-tensor grad clip
5. **PFSP** in AlphaStar paper §4 — `f_hard(p) = (1-p)^p_exp` weighting
6. **TLeague** (Tencent 2020) — full distributed league framework reference
7. **Phasic Policy Gradient** (Cobbe et al. 2021, ICML) — aux V phase
8. **DreamerV3** (Hafner et al. 2024) — symlog + twohot critic

User vetoed:
- Fixed external opponents in pool (e.g., starter_agent) — "你把 starter 加入到對手一定是錯的"
- Reward shaping (sticking with sparse terminal ±1)
