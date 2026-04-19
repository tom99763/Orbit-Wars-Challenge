# Orbit Wars Kit — Quickstart

A shareable workspace for Kaggle's **Orbit Wars** competition. Bundles a
scheduled leaderboard scraper, a replay-to-trajectory parser, and conventions
for agent development with Claude Code.

## What you get

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project rules auto-loaded by Claude Code |
| `main.py` | Starter Orbit Wars agent (the Kaggle submission) |
| `pipeline.py` | One-shot: scrape top-10 replays + parse to trajectories |
| `scrape_top5.py` | Playwright scraper (TOP_N defaults to 10 — name is legacy) |
| `parse_replays.py` | Raw replay JSON → per-(episode, agent) pickle + index.csv |
| `cron_wrapper.sh` | Cron entry: self-expiry, lock, logging |
| `install_cron.sh` | Install a self-expiring cron job |
| `uninstall_cron.sh` | Remove the cron job |
| `setup.sh` | Install playwright, chromium, system libs, kaggle CLI |
| `README.md` | Official competition rules (game mechanics) |
| `agents.md` | Official competition starter guide |

## Prerequisites

- Python 3.10+ (a conda env is the smoothest path — `setup.sh` uses conda-forge
  to install chromium's system libs without sudo)
- A Kaggle account that has joined https://www.kaggle.com/competitions/orbit-wars/rules
- `~/.kaggle/kaggle.json` for that account (`chmod 600`)
- Linux or WSL (cron needs a running `cron`/`crond` service)

## Install

```bash
# Inside the python env you want to use
./setup.sh
```

`setup.sh` installs `playwright`, Chromium, the Playwright-required system libs
via conda-forge, and the `kaggle` CLI. It also prints Kaggle credential setup
steps if `~/.kaggle/kaggle.json` is missing.

## Verify

```bash
# Your Kaggle account must already have joined the competition
kaggle competitions list -s orbit-wars
#   userHasEntered should be True
```

## One-shot scrape + parse

```bash
python3 pipeline.py
```

Outputs:
```
simulation/<YYYY-MM-DD>/<HH-MM>/
    leaderboard.html/.png
    player_assets/                   (offline visualizer bundle)
    rank0X_<team>/replay_<id>.json   (full replay)
    rank0X_<team>/play_<id>.html     (offline-playable wrapper)

trajectories/<YYYY-MM-DD>/<HH-MM>/
    <episode_id>__<team_slug>.pkl    (cleaned per-(episode, agent))
    index.csv                         (team, winner, reward, seed, …)
```

## Schedule it

```bash
# Every 30 min for 7 days (the defaults)
./install_cron.sh

# Custom
./install_cron.sh --every 15 --days 3
```

The cron wrapper:
- Self-expires — removes itself from your crontab after N days
- Skips overlapping runs via an atomic lockfile
- Logs each run to `.cron.log`

Manage:
```bash
crontab -l                              # see the installed entry
tail -f .cron.log                       # watch live
date -d @"$(cat .cron_expiry)"          # remaining lifetime
./uninstall_cron.sh                     # stop early
echo $(date -d "+14 days" +%s) > .cron_expiry   # extend
```

## Play a replay offline

The `play_<id>.html` wrappers iframe the Kaggle visualizer and `postMessage`
the replay in. **The iframe must be served over http/https, not `file://`**:

```bash
cd simulation/<date>/<HH-MM>/
python3 -m http.server 8765
# then open  http://localhost:8765/rank01_<team>/play_<id>.html
```

## Use trajectories for RL / BC

```python
import pickle, pandas as pd, pathlib

date_dir = pathlib.Path("trajectories/2026-04-18/21-30")
idx = pd.read_csv(date_dir / "index.csv")

# Load all winning trajectories (expert demonstrations)
winners = [pickle.load(open(date_dir / row.file, "rb"))
           for _, row in idx[idx.winner].iterrows()]

# (state, action) pairs for behavioral cloning
pairs = [(step, step["action"])
         for traj in winners
         for step in traj["steps"]
         if step["action"]]            # skip idle turns
```

See `CLAUDE.md` for the trajectory schema and scraping internals.

## Submit an agent

```bash
# Single-file submission
kaggle competitions submit orbit-wars -f main.py -m "my strategy v1"

# Multi-file (main.py at the tar.gz root)
tar -czf submission.tar.gz main.py helpers.py weights.pkl
kaggle competitions submit orbit-wars -f submission.tar.gz -m "multi-file v1"
```

Test locally first:

```python
from kaggle_environments import make
env = make("orbit_wars", debug=True)
env.run(["main.py", "random"])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
```
