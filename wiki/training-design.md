# Orbit Wars — Training Design (v1)

*Last updated: 2026-04-19.*

Concrete plan for producing a competitive neural-net agent. Sources and
trade-offs in `rl-methods.md`; this doc commits to choices.

---

## 0. Decisions in one screen

- **Observation encoder**: Set Transformer over planets + fleets, with
  global features appended. Permutation-invariant.
- **Action head**: per owned planet, predict `(target_planet_id_or_pass,
  ships_bucket)`; recover `angle = atan2(target - src)` at execute time.
  Entirely discrete + maskable.
- **Phase 1 — BC** on top-10 winner trajectories (from `trajectories/`).
  ≤ 1 GPU-day. Gives us a non-random starting point.
- **Phase 2 — PPO self-play** warm-started from BC. Teacher KL against
  the frozen BC model for 20M steps of shaped reward, then sparse.
  Add PFSP opponent sampling from a checkpoint pool.
- **Phase 3 — distillation** of a rules-based teacher (`lb-928` planner)
  into the neural net, and optional Gumbel-MuZero inference-time search
  via LightZero.
- **Evaluation harness**: local matchups against `random`, `starter`,
  the `lb-928` planner, and frozen past selves; ELO-style tracker. Then
  online: submit, scrape leaderboard+episodes (`scrape_submission.py`).
- **Framework**: CleanRL PPO (hacked), PyTorch, single 24 GB GPU.

---

## 1. Data pipeline (already in place)

- `scrape_api.py` — hourly, top-10 × 5 recent episodes/team → `simulation/`.
- `parse_replays.py` → `trajectories/<date>/<HH-MM>/*.pkl` (winner
  flag + per-step `(planets, fleets, action)`).
- For BC, filter `index.csv` by `winner==True`.

---

## 2. Observation encoding

One tensor per entity type, pooled via attention.

### Planet features (per planet, variable count ~20-40)
| Field | Encoding |
|---|---|
| `id` | Ignored (set model — position is attention-derived) |
| `owner` | One-hot (self / each opponent / neutral) — 3 or 5 dim |
| `x`, `y` | 2× Fourier features (4 dim) + raw `(x-50)/50` |
| `r` (radius) | Raw, log-scaled |
| `ships` | `log1p(ships)/8` |
| `production` | One-hot over [1..5] |
| `is_comet` | bool (from `comet_planet_ids`) |
| `is_static` | `orbital_r + r >= 50` flag |
| `my_dist_to_sun` | norm |
| `rotation_phase` | `sin, cos` of current angle (angular_velocity × step) |

### Fleet features (per fleet, variable count)
| Field | Encoding |
|---|---|
| `owner` | One-hot |
| `x`, `y` | Fourier features |
| `angle` | `(sin, cos)` |
| `ships` | `log1p/8` |
| `from_planet_id` | Looked up into planet embedding (cross-attention) |

### Global features (scalar, broadcast)
- `step / 500`
- `angular_velocity`
- One-hot over `{2p, 4p}`
- `remainingOverageTime / 2`
- `player` slot (one-hot 0..3)
- Flags: comet-window active? (`step in [45..55, 145..155, ...]`)
- Running tallies: `my_total_ships`, `enemy_total_ships`, planet counts

### Encoder (PyTorch sketch)

```
planet_tokens = MLP(planet_features)                  # [P, 128]
fleet_tokens  = MLP(fleet_features)                   # [F, 128]
global_token  = MLP(global_features)                  # [1, 128]
tokens = cat([planet_tokens, fleet_tokens, global_token])
for _ in range(N_layers):
    tokens = TransformerEncoderBlock(tokens)
critic_value = global_token.out → scalar
policy_logits_per_planet = my_planet_tokens.out → (target_planet, ships_bucket)
```

Share the encoder across policy and critic. 4-6 layers, 128 hidden,
~2-4 M params to start. Small, fast, easy to iterate.

---

## 3. Action representation

An action step is `List[[src_planet_id, angle, ships]]`. We produce it
as follows:

1. For each planet `p` owned by us with ≥ 1 ship, predict
   - `target_logits` over `{pass, planet_0, ..., planet_{P-1}}`
     (mask `pass` only if garrison ≥ 20 to encourage early play; mask
     all self-target options; mask neutral/enemy based on current state).
   - `ships_bucket` over `{25%, 50%, 75%, 100%}` (4 classes).
2. If `target == pass` → no move from this planet this turn.
3. Otherwise: angle = `atan2(target.y - p.y, target.x - p.x)` at
   execute time, `num_ships = round(bucket_frac × p.ships)`.

**Why not continuous angle head?** Because every observed winning move
in our scraped replays is effectively planet-directed; forcing free-
angle exploration wastes sample efficiency.

**Action mask** comes from the env: sources with 0 ships, destinations
that would cross the sun (point-to-segment distance < 10) can be
masked pre-softmax to cut search.

---

## 4. Phase 1 — Behaviour Cloning

**Input:** `trajectories/<date>/<HH-MM>/*.pkl` filtered to winners
(optionally to top-3 teams only).

**Target:** for each `(step, my_planets_with_actions)`, the target
is the expert action tuple per source planet.

**Loss:** cross-entropy on `target_planet_id` + cross-entropy on
`ships_bucket`, averaged over source planets that issued moves +
cross-entropy on pass for planets that didn't.

**Expected quality:** should beat `random`+`starter`, should tie or
lose narrowly vs. `lb-928`. That's the starting line.

**Compute:** ~1-3 hours on a single GPU with ~50k–200k expert
`(obs, actions)` pairs (we already have thousands per snapshot × hourly).

---

## 5. Phase 2 — PPO self-play

Warm-start from Phase 1. Environment is a PettingZoo `ParallelEnv`
wrapping `kaggle_environments.make('orbit_wars')`.

### Reward shaping (first 20 M steps)
Per-turn, per-agent:
- `+0.01 × Δ(my_total_ships)` clipped to ±1
- `+0.05 × Δ(my_planet_count)`
- `+0.02 × Δ(my_production_rate)`
- `−0.1` per ship wasted hitting the sun (detect by `fleets` disappearing
  outside enemy-planet range)
- `+1` / `−1` terminal (same as true reward)

After 20 M steps → drop shaping; keep only terminal ±1.

### PPO hyperparameters (starting point — CleanRL defaults)
- `num_envs = 64` parallel games
- `num_steps = 256` per rollout
- `update_epochs = 4`, `num_minibatches = 4`
- `lr = 2.5e-4` → linear decay
- `gamma = 0.997` (episode horizon 500 → effective)
- `gae_lambda = 0.95`
- `ent_coef = 0.01`
- `clip_coef = 0.2`
- `vf_coef = 0.5`
- `max_grad_norm = 0.5`
- **Teacher-KL coef = 0.5** decayed to 0.05 over first 5 M steps.

### Self-play opponent sampling (PFSP)
- Pool: current policy + last 8 checkpoints (snapshot every 1 M steps).
- Sample opponent ∝ `(1 − win_rate_vs_them)^2` — prioritise hard
  matchups. Fall back to uniform if pool < 4.

### Curriculum
1. First 1 M steps: play against `starter_agent` only.
2. 1-5 M: mix `starter`, `random`, last checkpoint 50/25/25.
3. 5-20 M: pure PFSP against checkpoint pool.
4. 20 M+: add the `lb-928` planner as a league member.

### Compute budget
Target 50 M env steps (~ 500k games) in a week on a single GPU. Env
steps are cheap (~0.5ms each). Bottleneck will be policy forward pass —
keep the model small (≤ 4 M params) and batch efficiently.

---

## 6. Phase 3 — optional MCTS at inference (Gumbel MuZero)

If Phase 2 plateaus below the `lb-928` rules planner:

- Train Gumbel MuZero via LightZero with a ≤ 32-action sample width
  (actions sampled from our learned policy prior).
- Sim budget per turn: calibrate to fit 1.0 s on the Kaggle grader
  (~100-300 sims realistic).
- Infer with MCTS; train value+policy from MCTS targets.

Treat this as a **separate track** — the baseline PPO agent should still
be shippable independently.

---

## 7. Evaluation

### Local (fast, anytime)
- Round-robin vs. `{random, starter, lb-928, bc_v1, ppo_<ckpt>}` at
  N=200 games each, both seats. Output ELO.
- Script: `eval_local.py` (to write; consumes
  `kaggle_environments.make('orbit_wars').run([...])`).

### Online (slow, truthful)
- Submit via `kaggle competitions submit orbit-wars -f main.py`.
- Record `submissionId` → pass to `scrape_submission.py` (to write),
  which records rating trajectory + recent matchups hourly.
- Cross-reference against the hourly top-10 snapshot to see which
  opponents we're drawing and what they're scoring.

### Regression gate before any new submission
```
new_agent beats lb-928  ≥ 55% of N=200 games  (both seats)
  AND beats previous submission ≥ 55%
```

Anything failing this stays off the ladder.

---

## 8. Directory layout for the training code

```
training/
  env.py            # PettingZoo wrapper around kaggle_environments
  encoder.py        # Set Transformer torso
  policy.py         # Action head + masking
  bc.py             # Phase 1 trainer
  ppo_selfplay.py   # Phase 2 trainer (CleanRL-derived)
  league.py         # PFSP opponent pool
  eval_local.py     # Round-robin evaluator
  config/           # YAML configs per run
  checkpoints/      # Saved weights (gitignored)

submission/
  bc_v1/            # per-submission folder
    main.py         # → imports model from a bundled .pt
    model.pt
    notebook.ipynb  # Kaggle kernel submission version
```

---

## 9. Status & next action (2026-04-19)

- [x] Scraper + parser producing `trajectories/` hourly.
- [x] Env installed, smoke-tested.
- [x] Research + design written.
- [ ] **Next:** wait for the first few hourly scrapes to land so we have
  a sizeable BC dataset, then implement `training/env.py` +
  `training/bc.py` + `training/eval_local.py`. These three are enough
  to get `bc_v1` submitted and start populating a submission tracker.
