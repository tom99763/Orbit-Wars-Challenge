# Orbit Wars — Project Instructions for Claude Code

This workspace competes in Kaggle's **Orbit Wars** Featured Simulation Competition
(https://www.kaggle.com/competitions/orbit-wars). It is NOT a supervised-learning
competition: the deliverable is a `main.py` that exports `agent(obs) -> list[moves]`
playing a 2D real-time strategy game (planets orbiting a sun, fleets, comets,
2-or-4 players, 500 turns). Winner = most total ships at game end.

Claude Code, please follow the conventions below when working in this project.

---

## Repository layout

```
main.py                          Orbit Wars agent — the Kaggle submission entry point
pipeline.py                      End-to-end: scrape top-N replays, then parse into trajectories
scrape_top5.py                   Playwright scraper (name is legacy; TOP_N defaults to 10)
parse_replays.py                 Turns raw replay JSONs into clean per-(episode, agent) pickles
cron_wrapper.sh                  Cron entry point: self-expiry, lock file, logging
install_cron.sh / uninstall_cron.sh
setup.sh                         One-shot environment bootstrap

notebooks/*.ipynb                ALL Jupyter notebooks live here (flat)
simulation/<YYYY-MM-DD>/<HH-MM>/ Raw scrape output (per snapshot)
    leaderboard.html, .png       Full page + screenshot at scrape time
    player_assets/               Kaggle visualizer bundle (iframe + JS) for offline replay
    rank0X_<team>/               One folder per top-N team in this snapshot
        replay_<episode_id>.json Full per-step game replay (Kaggle Environments schema)
        episode_<id>.json        Episode metadata
        play_<id>.html           Offline-playable wrapper (iframe + postMessage)
        page.html, player.png    Captured dialog state + screenshot
trajectories/<YYYY-MM-DD>/<HH-MM>/
    <episode_id>__<team_slug>.pkl   Cleaned per-(episode, agent) trajectory
    index.csv                       Summary (team, winner, reward, seed, …)
```

---

## Non-obvious conventions Claude Code MUST follow

### 1. Notebooks belong in `notebooks/`
All `*.ipynb` files live under `notebooks/`, flat layout. Never create a notebook
at the repo root or nest sub-folders inside `notebooks/`. If you find a notebook
outside `notebooks/`, move it in.

### 2. Scraped data is timestamped and append-only
Never overwrite an existing `simulation/<date>/<HH-MM>/` folder — each cron tick
is a snapshot used for longitudinal analysis of how top agents evolve. Use
`pipeline.py` (which picks one `datetime.now()` at start and passes matching
`--out-dir` / `--sim-dir` args to both stages) so scrape and parse output land in
the same `<date>/<HH-MM>/` subdirectory.

### 3. Playwright needs `LD_LIBRARY_PATH` on this host
The Chromium binary Playwright installs looks for `libnspr4.so`, `libnss3.so`,
`libasound.so.2`. On hosts where these aren't under `/usr/lib`, install them
into the conda env (`setup.sh` does this) and launch scrapers with
`LD_LIBRARY_PATH=<env_prefix>/lib`. `pipeline.py` and `cron_wrapper.sh` set this
automatically. If a user runs a scraper directly, they must export it first.

### 4. Trajectory dedup by `episode_id`
4-player games are scraped under multiple `rank0X_<team>/` folders (each
participating top-N team sees it as their last game). The parser dedups by
`episode_id` but still emits one trajectory per *(episode, agent)* — so a
4-player game yields 4 pickles, not 1. Don't try to dedup further.

### 5. Different scraping snapshots = different top-N
The leaderboard is live. The team at rank 3 at 14:00 may not be at rank 3 at
14:30. `index.csv` within each snapshot is authoritative *for that snapshot*.

### 6. This is an agent competition
Do not recommend supervised-learning workflows (feature engineering, CV,
training models on the provided data). There is no train.csv/test.csv. The
Kaggle "dataset" is just a starter agent, an agents.md tutorial, and a README.
Focus on game-strategy code in `main.py`. For offline RL / imitation learning,
use the trajectories emitted by `parse_replays.py`.

### 7. Submission format
Single-file: `kaggle competitions submit orbit-wars -f main.py -m "msg"`.
Multi-file: bundle with `main.py` at the root of a tar.gz, then submit the
tar.gz. Never put helpers inside `notebooks/` and expect the submission to find
them.

---

## Running the pipeline

```bash
# One-shot
python3 pipeline.py

# Schedule every 30 min for 7 days (self-expiring cron)
./install_cron.sh --every 30 --days 7

# Check status
crontab -l
tail -f .cron.log
date -d @"$(cat .cron_expiry)"        # remaining lifetime

# Stop early
./uninstall_cron.sh
```

---

## Leaderboard scraping — how it actually works

`scrape_top5.py` uses Playwright (headless Chromium) to:

1. Load https://www.kaggle.com/competitions/orbit-wars/leaderboard
2. Wait for React hydration (detect by checking for a known top-team name in
   `page.content()`).
3. Read the top-N team names from the DOM. The leaderboard is an
   `ul[role="list"]` with `li` children; each `li`'s `innerText` starts with the
   rank number followed by the team name.
4. For each team, find the row's `.google-symbols` icon whose `innerText` is
   `"live_tv"` (the "View episodes from this team's highest scoring agent" button)
   and click it. The dialog is a Material UI `.MuiModal-root`.
5. Inside the dialog, click the first `li` whose `innerText` contains `"ago"`
   (the most recent episode). This fires two internal API calls:
       POST https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisode
       POST https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay
6. A `response` event listener captures both JSON bodies.
7. Close with Escape, move on to the next team.
8. Once per run, download the iframe visualizer bundle from
   https://www.kaggleusercontent.com/episode-visualizers/orbit_wars/default/index.html
   plus the referenced `./assets/*.js`. Save to `player_assets/`.
9. Emit a `play_<id>.html` wrapper per episode — it iframes the local visualizer
   and `postMessage`s `{type:'update', environment: <replayJson>}` into it.
10. Open the wrapper via `python3 -m http.server` (not `file://`) to play back
    offline with full animation controls.

If Kaggle changes the leaderboard markup, the selectors in `scrape_top5.py` are
the things to adjust. Common failure signatures:
- Empty `top = []` → the `ul[role="list"]` detector didn't match; inspect DOM.
- `click result: no_icon` → the `live_tv` button's markup changed.
- `dialog selector matched: None` → `.MuiModal-root` is no longer the dialog class.

---

## Trajectory schema (parse_replays.py output)

Each pickle is a dict with:

```
episode_id          int
team_name           str
agent_idx           int          0-based player slot in the game
opponents           list[str]
n_players           int          2 or 4
final_reward        int          terminal reward from replay.rewards (winner = max)
final_status        str          "DONE" / "ERROR" / …
winner              bool
config              dict         game config including seed — replays are reproducible
angular_velocity    float
initial_planets     list         for trajectory reconstruction
n_steps             int
steps               list[dict]   one dict per turn, see below
```

Each `steps[t]` dict:

```
step                    int
planets                 list of [id, owner, x, y, radius, ships, production]
fleets                  list of [id, owner, x, y, angle, from_id, ships]
action                  list of [from_id, angle_rad, num_ships]   — what THIS agent did
reward                  float   — 0 except at terminal (±1)
status                  str     — "ACTIVE" / "DONE" / …
done                    bool
my_ships_on_planets     int     } convenience scalars for reward shaping /
my_ships_in_fleets      int     } feature engineering. Derived from planets+fleets
my_total_ships          int     } filtered by agent_idx.
my_planet_count         int
enemy_ships_on_planets  int
enemy_ships_in_fleets   int
enemy_total_ships       int
enemy_planet_count      int
neutral_planet_count    int
num_fleets_on_board     int
num_actions             int
```

### Loading

```python
import pickle, pandas as pd, pathlib
date_dir = pathlib.Path("trajectories/2026-04-18/21-30")
idx = pd.read_csv(date_dir / "index.csv")
winners = [pickle.load(open(date_dir / r.file, "rb"))
           for _, r in idx[idx.winner].iterrows()]
# Behavioral cloning pairs from expert (winning) agents
pairs = [(s, s["action"]) for t in winners for s in t["steps"] if s["action"]]
```

---

## Game mechanics (for strategy work in main.py)

- Board: 100×100 continuous; sun radius 10 at (50, 50). Fleets crossing the sun die.
- Planets: 20–40, symmetric across the center; inner planets orbit at
  `angular_velocity` rad/turn; outer planets are static.
- Planet radius = `1 + ln(production)`; production ∈ [1, 5] ships/turn.
- Home planets start with 10 ships; neutrals with 5–99 (skewed low).
- Fleet speed: `1.0 + 5.0 * (log(ships) / log(1000))^1.5`, capped at 6. A fleet of
  1 ship moves 1/turn; ~500 ships → ~5; ~1000 → 6.
- Action: `[[from_planet_id, angle_rad, num_ships], …]`; must own the source
  planet; cannot over-allocate; can issue multiple per turn.
- Comets spawn at steps 50, 150, 250, 350, 450 — groups of 4 on elliptical paths,
  capturable while on the board.
- Combat: largest attacking force minus second-largest survives; if attacker
  ≠ planet owner and surviving ships exceed the garrison, ownership flips.
- Game ends at step 500 or when only one player has any planets/fleets.

Full observation fields are documented in `README.md` (copy of the competition
README downloaded via `kaggle competitions download -c orbit-wars`).

---

## Kaggle setup

- `~/.kaggle/kaggle.json` must be for the account that joined the competition.
  A 403 on `kaggle competitions download orbit-wars` means the active API token
  is for a different account than the one that accepted the rules.
- Verify membership: `kaggle competitions list -s orbit-wars` → `userHasEntered`
  must be `True`.
- The installed `kaggle` CLI version 2.0.1 lacks `episodes` / `replay` / `logs` /
  `pages` subcommands. This kit bypasses them entirely — replays are captured
  by Playwright from the live leaderboard, and the internal Protobuf API
  requires a `submissionId` that isn't exposed on the public CSV leaderboard.
