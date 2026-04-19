# Strategic Scenarios — Issues and Policies

*Last updated: 2026-04-19.*

Enumerates the game-state situations a competitive Orbit Wars agent has
to handle differently, and how we intend to teach each one. Pairs with
`training-design.md` (base stack) and `match-quality.md` (metrics used
to detect each scenario).

---

## 1. 2-player vs 4-player are genuinely different games

This is the single biggest regime split and the one most agents
(including naive BC) miss.

### 1.1 Key structural differences
- **Home geometry.** 2p: Q1/Q4 of a random group (could be static or
  orbiting). 4p: *only* the y=x diagonal group can host all 4 homes
  symmetrically; the env re-routes there if the picked group is
  orbiting. Agents must be **position-aware differently**.
- **Zero-sum vs general-sum.** 2p is strictly zero-sum → min-max
  optimal. 4p has diplomacy / kingmaking / turtling dynamics.
- **Comet competition.** 4 comets spawn per window; in 2p one player
  can reasonably grab multiple, in 4p the per-player share is ~1.
- **"Attack the weakest" vs "attack the target".** In 2p the target
  is obvious; in 4p, picking which of 3 opponents to pressure is a
  first-order decision.

### 1.2 Proposed handling
- **Explicit `is_4p` flag in the global features** (already in
  `featurize.py` globals[7]). Encoder / policy should condition on it.
- **Train separate heads per mode,** sharing the encoder. Loss is
  `is_4p ? head_4p : head_2p` — avoids the policy averaging across
  regimes. Cheap: two linear layers.
- **Reward shaping that's mode-aware.** In 4p, add a tiny bonus for
  being ranked ≥ 2nd in total ships — avoids all-out attacks that
  let a third player win.
- **Self-play pool must contain 4p games.** If we only self-play 2p,
  the 4p head will drift. Alternate env modes in rollout batches.

---

## 2. Headwinds / comeback scenario

"Opponent conquered several planets; you're down in total ships; how do
you use comets and late-game fleets to claw back?"

### 2.1 Signals we can detect
- `P(win_t) < 0.3` for ≥ 20 turns → we're behind, stable.
- `Δships_lead < 0 and |Δships_lead| increasing` → losing momentum.
- `step ∈ [40..55, 140..155, …]` → comet window (the main
  redistribute-resources moment each 100 turns).

### 2.2 Policies the agent should have
- **Don't commit to low-value attacks when behind.** Attacking neutrals
  is fine; attacking well-defended enemy planets just bleeds more ships.
- **Comet rush.** In the 11-turn window around a comet spawn, divert
  production to the nearest comet before the enemy does. A captured
  comet = free ships + income.
- **Turtle-then-counter.** Consolidate on ≤ 3 production-5 planets,
  wait for opponent to over-extend, then strike.
- **Denial targeting.** If behind by ships, trade the enemy's
  production down before trading garrisons. A single lost production-5
  planet on their side is worth ~50 ships over the remaining game.

### 2.3 Proposed training
- **Curriculum scenes.** Initialise mid-game states where we're behind
  (10+ ships behind, ≥ 2 planets lost) and train with dense rewards
  for closing the gap. Generate these from our scraped *comeback*-tagged
  games (we already label these in `analytics_match.csv`).
- **Clutch bonus.** During PPO, multiply the reward by 2.0 when the
  agent's `P(win)` is below 0.3 at action time. Over-rewards crawling
  back; has to be capped so the agent doesn't prefer being behind.
- **Behavioural cloning of come-backers.** Up-weight `quality_tag=="comeback"`
  trajectories in the BC loss. `featurize.py` will expose the tag
  in `index.csv` so the dataloader can sample accordingly.

---

## 3. Comet windows as defensive tool (the user's prompt)

"Your opponent conquered some planets, how do you use your comets to
defend when you are in headwinds and comeback?"

Two distinct uses of comets:

### 3.1 Offensive comet grabs
- Send a force to the spawn point ~3 turns before spawn.
- If your force arrives too early, it wastes turns; too late, enemy
  got it.
- Training signal: `quality_tag == 'comeback' AND comet_grabs >= 1` —
  these are the trajectories worth copying.

### 3.2 Defensive comet use
- Comet captured ≠ auto-defence. Ships on a comet can't easily come
  back (comet is on an ellipse). But:
  - Grabbing the comet denies ~50+ ships to the opponent.
  - The comet's path can intersect enemy fleets — sweep mechanics
    (planets sweeping over fleets) may destroy attacking forces if
    timed right.
  - Comet's orbit puts it near the enemy home side ~once, so a
    forward-deployed garrison can launch short-range attacks.

### 3.3 How to teach it
- Add an auxiliary prediction head: "next-spawn comet positions". The
  env's comet spawns are deterministic given seed + angular_velocity;
  a small supervised loss teaches the encoder to track them.
- Reward shape (Phase 2): +0.1 per comet ship captured, +0.05 per
  attacker ship destroyed by a planet-sweep.

---

## 4. Opening book (turns 0-40)

- Home planet starts with 10 ships.
- Static production-5 planets are the highest-value early targets.
- The `starter` agent shows the canonical opening: snipe nearest
  static non-owned planet with ~half garrison.

**Issue:** BC from winners will learn *an* opening, but not necessarily
the best. Expert openings diverge.

**Handling:** train on winner trajectories, evaluate the resulting
opening-40-turn win-rate vs. each of {random, starter, lb-928}.
If weak, add supervised distillation from `lb-928`'s openings (it's
deterministic and strong in the opening phase).

---

## 5. Late game (turns 400-500)

- Production matters most (accumulated over remaining turns).
- Risk aversion: don't launch attacks that won't resolve before step 498.
- Ships in flight at game end count toward your total — don't leave
  ships on long flights through the sun.

**Handling:** encode `remaining_turns_norm` (already in globals[2]).
Add a training constraint: **no fleet's ETA > remaining_turns**.
Enforceable as action mask.

---

## 6. Sun-kills are a major mistake

The sun is at (50, 50), radius 10. Point-to-segment distance < 10 kills
the fleet. Naive angle-based policies frequently shoot through the sun.

**Handling:** action mask should pre-reject any `(src, target)` pair
whose connecting line segment passes within 10 of the center. This is
cheap (O(N²) over planet pairs, precomputable per step). Add this to
`training/policy.py` masking.

---

## 7. Issue list (short form, for training checklist)

| # | Scenario | Detector (in our data) | Handling |
|---|---|---|---|
| 1 | 2p vs 4p regime split | `n_players` | Conditioned policy (shared encoder, per-mode head) |
| 2 | Headwind comeback | `quality_tag == comeback`, `pwin < 0.3` | Clutch reward + curriculum init + BC upweight |
| 3 | Comet windows | `step ∈ spawn_windows ± 5` | Comet grab reward, position prediction aux loss |
| 4 | Opening (turns 0-40) | `step < 40` | Winner BC + optional `lb-928` distillation |
| 5 | Late game (400-500) | `step > 400` | Action mask on fleet ETA > remaining turns |
| 6 | Sun-kill avoidance | `|centre − segment| < 10` | Action mask (pre-softmax) |
| 7 | Kingmaking (4p only) | 3rd place → choose who to attack | Mode-aware reward (rank ≥ 2nd bonus) |
| 8 | Blowout-in-progress | `pwin ≥ 0.9 early`, `quality_tag == blowout` | Drop from BC; capped reward in PPO so agent doesn't learn to tilt |

---

## 8. Concrete follow-ups

- [ ] `analyze.py` → add `quality_tag` to per-episode CSV (done, check)
- [ ] `featurize.py` → expose `quality_tag`, `comeback_magnitude` in
      `index.csv` so BC can sample by tag
- [ ] `training/policy.py` → implement sun-kill mask + fleet-ETA mask
- [ ] `training/bc.py` → sampler weights: `{epic: 1.5, comeback: 2.0,
      standard: 1.0, close: 1.2, blowout: 0.0}`
- [ ] `training/ppo_selfplay.py` → clutch bonus, comet-grab bonus,
      mode-aware rewards
