# v92 — Baseline 訓練

## 什麼是 baseline？

Baseline 是「**零 inductive bias**」的參考點。沒有任何 game-specific 特徵注入，NN 必須從稀疏的 ±1 reward 自己摸索所有策略。它的唯一作用是**當對照組**——只有拿 baseline WR 跟 variant 比，才能判斷某個 inductive bias 是否真的有幫助。

### Baseline 的設定

| 項目 | 值 |
|------|-----|
| Inductive biases | **全部關閉**（`env_extras={}`） |
| 候選行星數 K | 7（K_NEAREST 預設值） |
| Ship count rule | sniper rule：`ships = max(target.ships + 1, 20)` |
| PFSP pool | 停用（`POOL_START_UPD = 10^12`，實際上永不啟動） |
| LR schedule | constant 3e-4（Adam） |
| PPO epochs | 4 |
| Minibatch size | 2048 |
| n_envs × t_rollout | 128 × 64 = 8192 samples/update |
| 預期 SPS | ~1800–2000（無競爭時）；~700（tucker_sed 搶 CPU 時） |
| Eval 頻率 | 每 200 updates，inline，vs starter / v14 / lb1224 / ow_proto |
| 每次 eval | 每個對手 2 games（2 seats），共 8 games |

**注意**：每次 eval 只有 2 games per opponent，variance 極高（一局 = ±25% WR）。判斷學習趨勢要看連續 3–4 個 eval 點的走向，不是單一數字。

---

## 如何啟動 baseline 訓練

### 最簡單：只跑 baseline

```bash
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lab/miniconda3/envs/tom/lib:$LD_LIBRARY_PATH \
  nohup /home/lab/miniconda3/envs/tom/bin/python training/v92/run_experiments.py \
    --base-save save/v92_exp \
    --n-envs 128 --t-rollout 64 \
    --updates-per-exp 1500 --baseline-updates 1500 \
    --max-wall-sec 5400 \
    --experiments baseline \
    >> save/v92_exp/runner.log 2>&1 &
echo "Runner PID: $!"
```

結果存在 `save/v92_exp/baseline/`：
- `train.csv` — 每 update 的 pi_loss / v_loss / ent / sps
- `eval.csv` — 每 200 upd 的 WR vs 各對手
- `run.log` — train_jax stdout

### 跑 baseline + 所有 variant（完整實驗）

Baseline 列在 `--experiments` 第一位時，runner 會先完成 baseline 再跑各 variant：

```bash
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lab/miniconda3/envs/tom/lib:$LD_LIBRARY_PATH \
  nohup /home/lab/miniconda3/envs/tom/bin/python training/v92/run_experiments.py \
    --base-save save/v92_exp \
    --n-envs 128 --t-rollout 64 \
    --updates-per-exp 1500 --baseline-updates 1500 \
    --max-wall-sec 5400 \
    --experiments baseline A3_gang_up A2_crash_exploit A1_defense_look \
      T2_prod_cp T2_threat_inflow T2_capture_cost \
      AS8_doomed AS6_hostility AS5_score_cand AS4_intercept \
      AS3_ship_head AS2_req_ships AS1_K_12 AS1_K_4 \
    >> save/v92_exp/runner.log 2>&1 &
echo "Runner PID: $!"
```

### 跳過已完成的實驗（--skip）

S3_sundancer 和 S1_stage_onehot 已跑完，用 `--skip` 避免重跑：

```bash
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/lab/miniconda3/envs/tom/lib:$LD_LIBRARY_PATH \
  nohup /home/lab/miniconda3/envs/tom/bin/python training/v92/run_experiments.py \
    --base-save save/v92_exp \
    --n-envs 128 --t-rollout 64 \
    --updates-per-exp 1500 --baseline-updates 1500 \
    --max-wall-sec 5400 \
    --skip S3_sundancer S1_stage_onehot \
    >> save/v92_exp/runner.log 2>&1 &
echo "Runner PID: $!"
```

---

## 監控 baseline 訓練

```bash
# 看目前 upd / ent / sps（最後 10 行）
grep "baseline t=" save/v92_exp/runner.log | tail -10

# 看 eval 結果（每 200 upd 更新一次）
cat save/v92_exp/baseline/eval.csv

# 看完整 train 曲線
cat save/v92_exp/baseline/train.csv | column -t -s, | tail -20

# 確認 process 還活著
ps aux | grep -E "run_experiments|train_jax" | grep -v grep
```

### 何時算 baseline 「夠穩定」？

Baseline 是對照組，不需要很強，只需要：
1. `ent_t > 0.5` 全程（未 collapse）
2. `wr_starter` 在至少一個 eval 點達到 ≥ 25%
3. 無 crash / 無 pi_loss > 5.0

達到以上條件，baseline 結果就可以當作 variant 比較的基準。

---

# v92 — Inductive Bias 設計 (重要)

practitioner: 「直接觀察遊戲 + 做分析，找出哪些資訊有用，進一步縮小 policy search space」

下面是給 v92 的 game-specific inductive bias，分 3 個層次注入：

## A. Observation Features (16-dim per planet, 大量 game-knowledge)

基本（已在 v90.5 用過）:
```
[my_ships/200, enemy_ships/200, neutral_ships/200,
 prod/5, radius/5,
 pos_x/50, pos_y/50, step/500,
 is_static (r+radius >= 50),
 is_comet, comet_life_remaining/50]
```

新加 game-derived (這才是 inductive bias 重點)：
```
threat_score        = Σ enemy_fleet.ships / (eta+1)    # 受威脅程度
defense_eta         = min(eta) over incoming enemy fleets  # 最快敵到達時間
my_fleet_inbound    = Σ my_fleet.ships heading here    # 已派出的支援
capture_cost        = (enemy_ships + 1) + prod*eta_nearest  # 需要多少艦才打下
distance_to_my_nearest  / 100                          # 我方最近距離
distance_to_enemy_nearest / 100                        # 敵方最近距離
```

**這些 features 直接 encode 戰術 reasoning**，比讓 NN 自己學「fleet 在動 + ships > threshold = threat」省好幾百萬 sample。

## B. Action Space — Mask 大量無效動作

```
target_mask (per src):
  - Sun collision targets → -∞
  - ETA > game_remaining_steps → -∞
  - Self-reinforce → -∞  (already)
  - Out-of-bounds → -∞

ship_bucket_mask:
  - src.ships < 5 → only allow bucket 0 (NOOP)
  - src under threat → mask high buckets (don't empty self)
```

**Sun mask 是 free win** — 不增加 param，直接刪 ~30% bad actions。

## C. Env-Level — Symmetry & Augmentation

```
Symmetric reflections: 4-fold board symmetry
  → can train on (state, action) AND (rotated_state, rotated_action)
  → 4× effective data

POV normalization (optional):
  → always rotate board so "my home" in fixed quadrant
  → kills positional bias
```

## D. Architecture Inductive Bias

Already planned:
- Entity transformer (permutation invariant over planets)
- Per-source factored action (decompose joint distribution)
- Single Dense heads (regularization via simplicity)

額外加：
- **Per-planet positional encoding** = sin/cos of orbital angle (helps attention)
- **Mask attention to comets** when no comets exist
- **Skip connection** from raw features to head (residual)

---

# v92 — Orbit Wars JAX Env Spec

Extracted from `kaggle_environments/envs/orbit_wars/orbit_wars.py` (812 lines).
Goal: byte-exact reimplementation in pure jax.numpy for ~10K SPS.

## Constants (all integer / float, fixed)

```
BOARD_SIZE = 100.0          CENTER = 50.0
SUN_RADIUS = 10.0           ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0          COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7        MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10      MIN_STATIC_GROUPS = 3
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]
shipSpeed = 6 (configurable)
cometSpeed = 4
episodeSteps = 500
```

## State Schema (fixed-size for JAX)

```
planets:           (MAX_PLANETS=60, 7)  [id, owner, x, y, r, ships, prod]
planets_active:    (MAX_PLANETS,) bool
initial_planets:   (MAX_PLANETS, 7)     (snapshot of t=0 + spawn-time comets)
fleets:            (MAX_FLEETS=400, 7)  [id, owner, x, y, angle, from_id, ships]
fleets_active:     (MAX_FLEETS,) bool
comet_paths:       (MAX_COMETS=20, MAX_PATH_LEN=40, 2)  pre-computed at spawn
comet_path_idx:    (MAX_COMETS,) int32  -1 = not spawned
comet_planet_ids:  (MAX_COMETS,) int32  -1 = unused slot
step:              int32
angular_velocity:  float32
next_fleet_id:     int32
active_players:    (N_PLAYERS,) bool
rewards:           (N_PLAYERS,) float32   computed at terminal
done:              bool
```

## reset(seed) — episode init

1. RNG = `random.Random(seed)` (Python's stdlib for byte-parity)
2. `angular_velocity = rng.uniform(0.025, 0.05)`
3. `generate_planets(rng)` — see below
4. Pick random home group, assign:
   - 2P: planet[base+0].owner=0, ships=10; planet[base+3].owner=1, ships=10
   - 4P: planet[base+i].owner=i, ships=10 for i in 0..3
5. `next_fleet_id = 0`, `step = 0`, all comets/fleets empty

**Note**: Initial state generation is rejection-sample loop (up to 5000 attempts).
Best run on CPU once per episode, then transfer to GPU JAX state.

### generate_planets(rng)

Two phases:

**Phase 1** (lines 76-122): static planet groups
- Loop max 5000 attempts to get `>= 3` static groups
- For each: `prod = rng.randint(1, 5)`, `r = 1 + log(prod)`
- Polar coords in Q1: `angle ∈ [0, π/2]`, `orbital_r ∈ [ROTATION_LIMIT-r, max_constrained]`
- Q1 position: `x = CENTER + r·cos(angle), y = CENTER + r·sin(angle)`
- Symmetric replication to 4 quadrants (group of 4 planets)
- `ships = min(rng.randint(5,99), rng.randint(5,99))` (skewed low)
- Reject if overlaps existing planets within `PLANET_CLEARANCE`

**Phase 2** (lines 124-188): orbiting planets
- Continue until `num_q1*4` planets AND at least 1 orbiting
- Polar coords in Q1 `(CENTER+15, BOARD-r-5)²` square
- Same group-of-4 symmetric replication
- `ships = rng.randint(5, 30)` (lower than Phase 1)
- Cross-checks: rotating vs static planet pairs need
  `abs(orb_r1 - orb_r2) >= r1 + r2 + PLANET_CLEARANCE`

### Comet spawn schedule

At steps in `COMET_SPAWN_STEPS`:
1. Derive per-spawn RNG: `random.Random(f"orbit_wars-comet-{seed}-{step+1}")`
2. `generate_comet_paths(...)` — up to 300 attempts:
   - `e ∈ [0.75, 0.93]` eccentricity, `a ∈ [60, 150]` semi-major
   - `phi ∈ [π/6, π/3]` orientation
   - Sample ellipse densely (5000 pts), resample at `comet_speed=4` arc-length
   - Extract on-board contiguous segment (5-40 points)
   - Build 4 symmetric copies
   - Validate no collision with sun / static planets / orbiting planets
3. If valid → 4 comets added with random `ships = min4(rng.randint(1,99))` (heavily skewed low)

## step(state, actions) — Per-turn dynamics

### 0. Spawn comets (if `step+1 ∈ COMET_SPAWN_STEPS`)
- Add 4 comet "planets" at off-board placeholder `(-99,-99)` until first path advance
- Use deterministic seed-based RNG

### 1. Process moves (per player)
```
For each move [from_id, angle, ships]:
    if from_planet.owner != player: skip
    if from_planet.ships < ships or ships <= 0: skip
    from_planet.ships -= ships
    start_x = from_planet.x + cos(angle) * (from_planet.radius + 0.1)
    start_y = from_planet.y + sin(angle) * (from_planet.radius + 0.1)
    create fleet [next_id, player, start_x, start_y, angle, from_id, ships]
    next_fleet_id += 1
```

### 2. Production
```
For each planet with owner != -1: planet.ships += planet.production
```

### 3. Compute planet motion (end-of-tick positions)

**Static planet** (`orb_r + radius >= ROTATION_LIMIT`):
- `new_pos = old_pos` (no movement)

**Orbiting planet**:
- `initial_p` from initial_planets snapshot
- `r = hypot(init.x - CENTER, init.y - CENTER)`
- `initial_angle = atan2(init.y - CENTER, init.x - CENTER)`
- `current_angle = initial_angle + angular_velocity * step`
- `new_pos = (CENTER + r·cos, CENTER + r·sin)`

**Comet planet**: position from pre-computed `comet_paths[path_index]`

### 4. Fleet movement
```
For each fleet:
    speed = 1 + (max_speed - 1) * (log(ships)/log(1000))^1.5
    speed = min(speed, max_speed)
    new_pos = old_pos + speed * (cos(angle), sin(angle))
    
    For each planet (in order, FIRST hit wins):
        if swept_pair_hit(fleet_old, fleet_new, planet_old, planet_new, planet.radius):
            → fleet goes to combat list, removed
            break
    else if out of bounds OR point_to_segment_dist(SUN, fleet_old, fleet_new) < SUN_R:
        → removed
```

`swept_pair_hit`: solve quadratic for relative position of two moving segments.

### 5. Apply planet motion
- All planets `(x, y) ← new_pos`

### 6. Combat resolution (per planet)
```
For planet, fleets_arrived:
    ships_per_owner = sum ships by fleet.owner
    sorted = sort desc by ships
    if len > 1:
        survivor_ships = top.ships - second.ships
        if tie: survivor_ships = 0
        survivor_owner = top.owner if survivor_ships > 0 else -1
    else:
        survivor_owner = top.owner; survivor_ships = top.ships
    
    if survivor_ships > 0:
        if planet.owner == survivor_owner:
            planet.ships += survivor_ships
        else:
            planet.ships -= survivor_ships
            if planet.ships < 0:
                planet.owner = survivor_owner
                planet.ships = abs(...)
```

### 7. Termination check
```
terminate = (step >= episodeSteps - 2)
         OR (len(alive_players) <= 1, where alive = has planet or fleet)
```

### 8. Reward (at terminal only)
```
scores[i] = sum(planet.ships if owner==i) + sum(fleet.ships if owner==i)
max_score = max(scores)
For each player i:
    reward[i] = +1 if scores[i] == max_score AND max_score > 0 else -1
```

## JAX Implementation Strategy

### Variable → Fixed-size

| Original | JAX |
|----------|-----|
| `planets: list (variable)` | `(MAX_PLANETS, 7) + active mask` |
| `fleets: list (variable)` | `(MAX_FLEETS, 7) + active mask` |
| `comet_paths: list per group` | `(MAX_COMETS, MAX_PATH_LEN, 2)` pre-allocated |
| `combat_lists[pid] = list` | `(MAX_FLEETS,) target_planet_idx for each fleet hit` |
| `next_fleet_id` ↑ | wrap modulo MAX_FLEETS (since old fleets exit) |

### Vectorization opportunities

- **Production**: `planets.ships += where(owner != -1, prod, 0)`
- **Planet motion**: vectorize over MAX_PLANETS
- **Fleet movement**: vectorize over MAX_FLEETS
- **Swept-pair hit**: vectorize over MAX_FLEETS × MAX_PLANETS pairs (mask)
- **Combat**: scatter-add by planet_id, then per-planet resolution

### Tricky parts

1. **Combat sequential ordering**: original code does "first hit wins" per fleet.
   - In JAX, compute ALL fleet-planet hits, then per-fleet take min-eta planet.
   - Verify byte-parity since order of planets matters.

2. **Comet path indexing**: each comet has its own `path_index`. Pack into `(MAX_COMETS,) int32`.

3. **Initial state**: CPU pre-generate (rejection sampling), then `jnp.array(...)` transfer.
   - reset() is ~1-2ms on CPU, negligible vs game steps.

4. **Done condition**: `len(alive_players) <= 1` requires set-counting. Use:
   `n_alive = sum(any(planets.owner == i and active) | any(fleets.owner == i and active) for i in players)`

### Parity Test Plan ⚠ MANDATORY GATE

**任何一個 mismatch 都不能進 Phase 1 PPO 訓練。**

```
For 1000 random seeds:
    1. Reset kaggle env with seed
    2. Reset jax env with seed
    3. Run 500 steps, both with same random action policy (seeded)
    4. After each step, compare (full state, not just observable):
       - planets: id, owner, x, y, r, ships, prod  (floats ε=1e-6)
       - fleets:  id, owner, x, y, angle, from_id, ships
       - comet_paths, comet_path_idx
       - step, angular_velocity, next_fleet_id
       - rewards (final game)
       - done flag (terminal step)
    5. If ANY diff: fail with detailed diff dump (which field, which step, which seed)
    6. PASS criterion: ALL 1000 seeds × 500 steps × all fields match
```

**Why strict**: 
- A 1e-4 position drift at step 1 accumulates over 500 steps → completely different game
- Combat resolution is non-commutative under reordering → exact sequence matters
- RL training on a "close but not identical" env produces a policy that loses on Kaggle
- Better to spend 2 days debugging parity than 5 days training on a phantom env

**Implementation order**:
1. Implement reset → match initial state (planets, comets initial schedule)
2. Implement step components one-by-one, parity-test after each:
   - production only → diff
   - planet motion only → diff
   - fleet movement only → diff
   - combat → diff
   - termination → diff
3. Build full step → 1000-seed test
4. Iterate until ALL pass

## Benchmarks

```
Target:
  1 env step:        ~50 μs   (jit + GPU)
  256 envs × step:   ~12 ms
  Total SPS:         ~20,000

Baseline (Python kaggle env):
  1 env step:        ~3 ms
  16 envs sequential: ~50 ms
  Total SPS:         ~300
  
Speedup target: 60-100×
```

## Engineering Order (Day 1-7)

```
Day 1: Read source ✓. Write SPEC ✓ (this doc).
Day 2: Implement state schema + reset() (CPU init + JAX transfer)
Day 3: Implement step() — vectorize fleet movement + collision
Day 4: Combat resolution + termination
Day 5: Parity test infra (1000 seeds × 500 steps comparison)
Day 6: Debug parity failures + iterate
Day 7: Benchmark + JIT optimize, hit 10K SPS target
```

## Open questions for future

- **Reward shaping?** Spec says ±1 sparse only. v92 plan agrees.
- **Action space**: still factored (target, ship_bucket)? Yes per v92 spec.
- **Multi-env batching strategy**: vmap over (state, action) batch dim? Yes.
- **JAX env library**: use `jax-marl` / `pgx` patterns? Probably write from scratch for control.
