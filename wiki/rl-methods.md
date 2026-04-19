# RL Methods Survey for Orbit Wars

*Last updated: 2026-04-19. Sources at bottom.*

This document synthesises what RL approaches are viable for Orbit Wars
(2-or-4 player, 500-step, continuous-state planetary RTS) and what past
Kaggle simulation competitions have shown works. The companion
`training-design.md` turns the conclusions here into a concrete training
plan.

---

## 1. What Orbit Wars actually is, RL-wise

Structural features that drive every design choice:

| Property | Value | RL implication |
|---|---|---|
| Players | 2 or 4 (selfish) | Self-play or league; zero-sum for 2p, general-sum for 4p |
| Episode length | 500 steps | Short — end-to-end RL per-episode feasible |
| Act timeout | 1 second/turn | Tight — MCTS must be cheap; Gumbel-style search, not deep AlphaZero |
| State | Continuous board, variable # planets (20-40 + comets), variable # fleets | Permutation-invariant encoder required (Set Transformer / attention over entities) |
| Action | List of `[src_planet_id, angle_rad, num_ships]`. Variable # actions/turn | Multi-discrete + continuous with action masking |
| Reward | ±1 at terminal (winner = argmax total ships) | Sparse — reward shaping essential for stage-1 RL |
| Simulator | Deterministic given seed; `kaggle_environments.make('orbit_wars')` | Re-playable, seeded, cheap to run (~ms/step) |
| Data | Top-10 Kaggle replays scraped hourly (`simulation/`, `trajectories/`) | Offline RL / BC / distillation are all viable starting points |

The *continuous angle* looks intimidating but in practice, strong agents
(see `notebooks/lb-928-7-physics-accurate-planner.ipynb`) compute the
angle from `(src_planet, target_planet)` — the true underlying action is
effectively discrete: **which planet do you shoot at, with what fraction
of your garrison**. A useful simplification is to model the action head
as `(target_planet_id, ships_bucket)` and recover the angle at execution.

---

## 2. Track record on Kaggle simulation competitions

Since 2020 Kaggle has run one RTS-style Simulation competition per year
(Halite → Lux AI → Kore → Lux AI S2 → **Orbit Wars 2026**). Lessons:

- **Rules-based agents won Halite, Kore, and Lux AI S2.** The current
  Orbit Wars leaderboard is dominated by one — see
  `notebooks/lb-928-7-physics-accurate-planner.ipynb` (a single-file,
  71 KB deterministic physics planner sitting in top 10).
- **Only Lux AI Season 1 was won by deep RL** — Pressman et al., 2021.
  Their recipe is the closest published analogue to what Orbit Wars
  needs and is worth imitating closely (see §4).
- **microRTS 2023/24 was won by `RAISocketAI`** using the combination
  *"behavioural cloning → DRL fine-tune → map-specific transfer"*. This
  is the pattern our data pipeline was designed to support.

Implication: a realistic Orbit Wars agent is **either** a strong
rules-based planner **or** a BC-seeded, self-play-trained neural net
distilled against a rules-based teacher. The middle ground (pure-from-
scratch RL) tends to lose.

---

## 3. Algorithm families and where they fit

### 3.1 Policy-gradient (PPO / IMPALA)
- **PPO** is the de-facto default. Stable, well-understood, supports
  action masking and multi-discrete. MAPPO (shared critic, decentralised
  actors) works well in cooperative/competitive MARL.
- **IMPALA + UPGO + V-trace** was the Lux AI S1 winning backbone.
  Distributed actor→learner with importance-sampled off-policyness.
  Useful if we want to farm out game rollouts across CPU cores.
- **Teacher KL** (freeze an older / supervised model, add
  `KL(π‖π_teacher)` to loss) was a critical stabiliser for Lux S1.
  It prevents policy collapse and strategic cycling. Recommended for
  our self-play phase, anchored against the BC model.

### 3.2 MCTS / AlphaZero family
- **AlphaZero** — needs a simulator (we have it), but expensive per-move.
  Not practical inside a 1-second turn at useful sim counts.
- **MuZero** — learns a latent dynamics model; removes the need for a
  simulator but adds training complexity.
- **Sampled MuZero** — extends MuZero to continuous / large action
  spaces by sampling K actions from the policy prior and searching among
  those. Fits Orbit Wars if we refuse to discretise actions.
- **Gumbel MuZero** — beats MuZero *when sim count is low*, which is
  precisely our 1-second regime. Best MCTS candidate for Orbit Wars.
- **LightZero** (OpenDILab, NeurIPS 2023) has PyTorch implementations
  of all four and a single config flag for continuous-vs-discrete
  action spaces. If we want MCTS in the stack, this is the library.

### 3.3 Offline RL / Imitation from our replays
- **Behaviour Cloning (BC)** — supervised `(obs, action)` classification
  on winner-only trajectories in `trajectories/`. Cheap, fast, and —
  per BAIR 2022 — *optimal when the data is near-expert and complete*.
  Our top-10 data is near-expert by definition.
- **Offline RL (CQL, IQL, TD3+BC)** — beats BC when data is noisy or
  mixed. Less important here because we filter to winners.
- **Decision Transformer** — autoregressive sequence model conditioned
  on return-to-go. Cute but the horizon (500 steps × 20-40 entities)
  makes context expensive.

### 3.4 Self-play schemes
- **Naïve self-play** (current policy vs. current policy) → strategy
  cycles.
- **Fictitious Self-Play (FSP)** — random past checkpoint.
- **Prioritised FSP (PFSP)** — AlphaStar-style; sample opponent with
  probability ∝ win rate against them. Much stronger signal.
- **League training** — main agents, main exploiters, league exploiters
  (AlphaStar). Overkill for week-of-compute budget but worth knowing.

### 3.5 Architecture for the observation
Planets and fleets are **sets** of entity feature vectors with varying
cardinality. The right encoder is attention-based:

- **Set Transformer** (Lee et al., ICML 2019) — permutation-invariant
  encoder with induced-set attention blocks.
- **Graph Transformer / GCNT** — edges between entities (e.g., pairs
  of planets, fleet→target relations) — captures relational structure
  and scales.
- Concrete shape: embed each planet/fleet with a small MLP, cross-
  attend over the set, append global features (step, angular_velocity,
  remaining time), pool for critic and per-entity policy heads.

Avoid the Lux S1 fully-convolutional ResNet pattern — our board is
*continuous* (no natural grid), and forcing a raster discards precision.

---

## 4. Frameworks — what to actually use

| Framework | Strengths | Weaknesses | Use for Orbit Wars? |
|---|---|---|---|
| **CleanRL** | Single-file PPO / DQN / TD3 / PPG; easy to hack in action masking, teacher KL, custom action heads | No multi-agent abstractions built-in | **Yes** — PPO prototype & self-play loop |
| **Stable-Baselines3** | Cleanest end-user API; quickest "try PPO on this env" | Single-agent only; you hack MARL on top | Maybe — sanity-check runs |
| **RLlib (Ray)** | First-class league / self-play (`self_play_league_based_with_open_spiel.py`), APPO, IMPALA | Heavy, lots of YAML, learning curve | **Only if** we need distributed rollouts later |
| **PettingZoo** | Standard multi-agent env API | Not an algorithm library | **Yes** — wrap the orbit_wars env as a PettingZoo ParallelEnv |
| **LightZero** | MuZero / Sampled / Gumbel / EfficientZero, PyTorch, continuous action flag | Steeper learning curve; large repo | Optional Phase-3 — add MCTS at inference if compute allows |

**Recommended stack:** `kaggle_environments.make('orbit_wars')` wrapped
into a PettingZoo ParallelEnv, trained with CleanRL PPO, custom Set-
Transformer torso, action head = `(target_planet_id | no-op,
ships_bucket)` per owned planet with masking, teacher-KL against a BC
model trained on `trajectories/winners`. Optional: later drop in
Gumbel MuZero via LightZero if a rules-based teacher is still beating us.

---

## 5. Key references

- **AlphaStar** — PPO + league + PFSP + teacher KL; the template
  everything else imitates.
  [DeepMind blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
  · [Nature paper PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)
- **Lux AI S1 winner (Pressman, 2021)** — IMPALA + UPGO + TD-λ + teacher KL,
  fully-conv ResNet, staged reward shaping.
  [Repo & README](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/README.md)
- **RAISocketAI (microRTS 2023/24)** — BC warm-start + DRL fine-tune +
  map transfer.
  [arXiv 2402.08112](https://arxiv.org/abs/2402.08112)
- **MAPPO** — PPO just works in cooperative/competitive MARL.
  [arXiv 2103.01955](https://arxiv.org/abs/2103.01955)
- **When to prefer Offline RL over BC (BAIR 2022)** — BC is optimal
  when expert data is near-perfect; offline RL wins on noisy data.
  [BAIR blog](https://bair.berkeley.edu/blog/2022/04/25/rl-or-bc/)
  · [arXiv 2204.05618](https://arxiv.org/abs/2204.05618)
- **Sampled MuZero** — MCTS in continuous action spaces.
  [Hubert et al., ICML 2021 PDF](https://proceedings.mlr.press/v139/hubert21a/hubert21a.pdf)
- **Gumbel MuZero** — better MCTS with low sim budget (our 1s case).
  [LightZero impl](https://github.com/opendilab/LightZero/blob/main/lzero/policy/gumbel_muzero.py)
- **LightZero** — unified PyTorch benchmark for MCTS+RL.
  [NeurIPS 2023, repo](https://github.com/opendilab/LightZero)
- **Set Transformer** — permutation-invariant entity encoder.
  [Lee et al., ICML 2019](https://arxiv.org/abs/1810.00825)
- **CleanRL** — hackable single-file PPO and friends.
  [Repo](https://github.com/vwxyzjn/cleanrl)
  · [PettingZoo CleanRL tutorial](https://pettingzoo.farama.org/tutorials/cleanrl/implementing_PPO/)
- **RLlib league example** — reference self-play orchestration.
  [self_play_league_based_with_open_spiel.py](https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/self_play_league_based_with_open_spiel.py)
