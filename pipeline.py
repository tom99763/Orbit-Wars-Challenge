"""End-to-end pipeline: scrape top-N replays, then parse into trajectories.

Picks ONE wall-clock timestamp at start (local time) and passes the derived
output paths to both stages so the scrape and the parser share the same
`<date>/<HH-MM>` folder.

Output layout:
    simulation/<YYYY-MM-DD>/<HH-MM>/       raw replay JSONs + offline player
    trajectories/<YYYY-MM-DD>/<HH-MM>/     cleaned per-(episode, agent) pickles + index.csv
"""

import datetime
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent
# Prepend the current Python's lib dir so Playwright's bundled Chromium can
# find system libs (nspr/nss/alsa) installed into the same (conda) env.
CONDA_LIB = str(pathlib.Path(sys.prefix) / "lib")


def run(cmd: list[str]) -> None:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    ld = CONDA_LIB if not existing else f"{CONDA_LIB}:{existing}"
    env = {**os.environ, "LD_LIBRARY_PATH": ld}
    print(f"\n▶ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> int:
    # Per-day rollup. Hourly cron ticks accumulate into the same folder;
    # scrape_api.py dedups downloads by episode_id and parse_replays.py
    # re-parses everything so the trajectory set always reflects what's on
    # disk.
    now = datetime.datetime.now()
    date = now.date().isoformat()
    sim_dir = ROOT / "simulation" / date
    traj_dir = ROOT / "trajectories" / date
    proc_dir = ROOT / "processed" / date

    print(f"orbit-wars pipeline  date={date}", flush=True)
    print(f"  sim_dir  = {sim_dir}", flush=True)
    print(f"  traj_dir = {traj_dir}", flush=True)
    print(f"  proc_dir = {proc_dir}", flush=True)

    run([sys.executable, str(ROOT / "scrape_api.py"),
         "--out-dir", str(sim_dir)])
    run([sys.executable, str(ROOT / "parse_replays.py"),
         "--sim-dir", str(sim_dir), "--out-dir", str(traj_dir)])
    run([sys.executable, str(ROOT / "analyze.py"),
         "--traj-dir", str(traj_dir), "--out-dir", str(proc_dir)])
    run([sys.executable, str(ROOT / "featurize.py"),
         "--traj-dir", str(traj_dir), "--out-dir", str(proc_dir)])

    n_replays = len(list(sim_dir.glob("*/*/replay.json")))
    n_trajs = len(list(traj_dir.glob("*.pkl")))
    n_proc = len(list(proc_dir.glob("*.npz")))
    print(f"\npipeline complete  replays={n_replays}  trajectories={n_trajs}  "
          f"processed={n_proc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
