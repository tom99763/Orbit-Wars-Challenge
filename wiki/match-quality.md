# Match Quality — Research Notes

*Last updated: 2026-04-19. Captured so we don't re-search each time the
analytics agent is touched.*

Purpose: when we scrape a replay, we want a number that answers *"was
this an epic game?"* — so we can prioritise high-quality games for BC
training, for PFSP opponent selection (reward the agent when it wins
a *close* game, penalise boring wins), and for human inspection of
replays.

---

## 1. Established metrics from sports / esports analytics

### 1.1 Game Excitement Index (GEI) — Yale / nflWAR school
- Definition: `GEI = Σ_t |ΔP(win_t)|` — sum of absolute changes in
  the win-probability of any team across every play/turn.
- Normalisation: divide by a reference game length so an overtime
  (or in our case, an accidentally long) match doesn't automatically
  look more exciting.
- High GEI ↔ lots of back-and-forth in win probability.
- **Applied to Orbit Wars:** compute `P(win_i | state_t)` per player
  each turn with a light classifier, sum `|ΔP|` across all 500 turns.
- Source: [Yale Sports Analytics](https://sports.sites.yale.edu/game-excitement-index-part-ii)

### 1.2 Win Probability Added (WPA)
- Used by FiveThirtyEight (NBA), Riot/AWS (2023 LoL Worlds), nflWAR.
- `WPA(action) = P(win | after) − P(win | before)`. Summable across
  a game → per-player contribution.
- **Applied to Orbit Wars:** attribute `ΔP(win)` of each step to the
  agent who took non-trivial actions that turn. Gives *per-agent
  clutch score*, not just per-match excitement.
- Sources:
  [Riot/AWS](https://aws.amazon.com/blogs/gametech/riot-games-and-aws-bring-esports-win-probability-stat-to-2023-league-of-legends-world-championships-broadcasts/),
  [FiveThirtyEight NBA methodology](https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)

### 1.3 Leverage
- `leverage(state_t) = stddev of P(win_end | state_t)` over simulated
  continuations — how much does what happens in the next few plays
  *matter*? High leverage = pivot moments.
- In our world: the turn when a comet spawns + someone fluffs the
  grab will often be high leverage.

### 1.4 Comeback index (MOBA literature)
- `comeback_magnitude = max_deficit − final_margin` where both are
  signed for the eventual winner. If the winner was down by 40 ships
  and finished up by 10, CM = 50.
- MOBA-specific: feature engineering around bounty gold, neutral
  objectives. For Orbit Wars the analogous features are
  *ship_lead*, *planet_count_lead*, *production_rate_lead*,
  *comet_captures*, *sun_kills_against_opponent*.
- Source: [MOBA comeback-victory prediction (MDPI Electronics 2025)](https://www.mdpi.com/2079-9292/14/7/1445)

### 1.5 Blowout detection (Dota, StarCraft literature)
- Inverse of GEI — a game where one side's win probability approaches
  1 within the first ~30% of the timeline and never dips.
- Useful negative filter: *don't* train BC on these because the
  winner's actions are uninformative after the lead is locked in.
- Sources:
  [Trouncing in Dota 2 (AAAI 2021)](https://cdn.aaai.org/ojs/7444/7444-52-10777-1-2-20200923.pdf),
  [SC2EGSet Nature 2023](https://www.nature.com/articles/s41597-023-02510-7)

### 1.6 APM (Actions Per Minute)
- Classic RTS skill proxy. Weak on its own — high APM ≠ good decisions.
- Useful in combination: *effective actions per minute* = APM × win_rate.

---

## 2. Computing P(win) in Orbit Wars

We don't have a learned win-prob model. Options, cheapest first:

- **Ship-based sigmoid (our MVP):** fit a 1-feature logistic on all
  scraped winner trajectories: `P(win | state) ≈ σ(β · ship_lead_t)`
  where `ship_lead = my_total - max(opponent_total)`. A few hours of
  scraped data gives enough to fit.
- **Feature logistic:** `[ship_lead, planet_lead, production_lead,
  step, is_4p]`. Still tractable closed-form.
- **Learned neural WP model:** tiny MLP over the same encoder planned
  for the BC model. Overkill for v1 but the eventual home if we want
  accurate leverage / WPA.

Use the same P(win) estimator for both GEI and WPA so they stay
internally consistent.

---

## 3. Our per-match scorecard

For each replay we'll emit:

```
{
  "episode_id": int,
  "gei": float,                    # Σ|ΔP(win)| normalised
  "comeback_magnitude": float,     # winner's (max_deficit − final_margin)
  "num_lead_changes": int,         # argmax P(win) flips
  "decisive_turn": int,            # first turn where eventual winner's
                                   # P(win) exceeds 0.9 and never dips
                                   # below. Earlier = more decisive.
  "action_density_per_agent": [float],
  "comet_captures_per_agent": [int],
  "sun_kills_per_agent": [int],
  "quality_tag": "epic" | "comeback" | "close" | "standard" | "blowout"
}
```

**Epic tag thresholds (to tune on our data):**
- `epic` ← `comeback_magnitude ≥ 30` AND `num_lead_changes ≥ 3`
- `blowout` ← `decisive_turn ≤ 150` AND `gei < 2.0`
- `comeback` ← `comeback_magnitude ≥ 20`, not already epic
- `close` ← `|final_margin| ≤ 0.1 × total_ships`
- `standard` ← otherwise

---

## 4. Per-agent advanced stats (beyond win rate)

### Offense
- **Attack efficiency** = `enemy_ships_killed / (my_ships_committed_to_attack)`.
- **Successful conquests** = planets captured that were held at game end.
- **Contested planets taken** = conquests where the prior owner was an
  enemy (not neutral). Neutral grabs don't count as offense.
- **First-strike rate** = fraction of attacks where we got there before
  the opponent could react.

### Defense
- **Defense rate** = `successful_repels / incoming_attacks`.
- **Loss rate under pressure** = planets lost when an enemy fleet of
  size ≥ our garrison was within 30 units.
- **Reinforcement speed** = median turns between "enemy fleet detected
  heading here" and "defensive fleet launched here".
- **Comet defense** = comet-grabs that happened while our other home
  planets were under attack (indicates multi-tasking).

### Economy
- **Production lead** = area under `production_rate(my) - production_rate(enemy)`.
- **Expansion pacing** = turn at which we owned 50% of our final planet
  count. Earlier = more aggressive opener.

### Coordination / waste
- **Sun kills per 100 fleets** = ships lost to the sun (bad!).
- **Overcommit rate** = fraction of fleets where `sent > needed+10%`.
- **Concurrent actions per turn** = median of `len(action)` when
  `len(action) > 0`.

### Clutch
- **WPA from behind** = total WPA earned at turns where our P(win) < 0.4.
  Agents who turn games around score highly.

---

## 5. Storage plan

Analytics land next to the trajectory pickles:

```
processed/<date>/
    analytics.csv        # one row per (episode, agent) with all metrics
    episode_gei.csv      # one row per episode with match-level metrics
    win_prob_model.pkl   # the fitted logistic (refreshed daily)
```

Computed by `analyze.py` (to be written) after `parse_replays.py` and
before `featurize.py` — so we can use the quality tag inside
`featurize.py` to filter BC training data (drop blowouts, upweight
comebacks).
