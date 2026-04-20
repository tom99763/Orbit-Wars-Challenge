"""Daily scheduler that fires training/nightly_offline_update.sh at 00:00.

Sibling to scheduler.py (hourly pipeline) but with a separate lock, log, and
fixed daily cadence. Self-expires after --days days.

Usage:
  nohup /home/lab/miniconda3/envs/tom/bin/python training/offline_scheduler.py \
        --days 90 > .offline_scheduler.stdout 2>&1 &

Stop:
  kill $(cat .offline_scheduler.pid)
"""
from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "training" / "nightly_offline_update.sh"
LOG = ROOT / ".offline_scheduler.log"
PID_FILE = ROOT / ".offline_scheduler.pid"
LOCK = ROOT / ".offline_scheduler.lock"


def log(msg: str) -> None:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def acquire_lock(clobber: bool) -> bool:
    if LOCK.exists():
        try:
            owner = int((LOCK / "pid").read_text().strip())
            os.kill(owner, 0)
            return False
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            if clobber:
                import shutil
                shutil.rmtree(LOCK, ignore_errors=True)
            else:
                return False
    try:
        LOCK.mkdir()
    except FileExistsError:
        return False
    (LOCK / "pid").write_text(str(os.getpid()))
    return True


def release_lock() -> None:
    import shutil
    shutil.rmtree(LOCK, ignore_errors=True)


INTERVAL_HR = 5


def next_fire_time(start_anchor: datetime.datetime) -> datetime.datetime:
    """Next :00 boundary that's a multiple of INTERVAL_HR since the anchor.
    Ensures the first run starts at the anchor (tomorrow 00:00) and then
    marches every INTERVAL_HR. Skips past any fires already in the past."""
    now = datetime.datetime.now()
    t = start_anchor
    while t <= now:
        t += datetime.timedelta(hours=INTERVAL_HR)
    return t


def tomorrow_midnight() -> datetime.datetime:
    now = datetime.datetime.now()
    return (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0)


def run_update() -> int:
    log(f"firing {SCRIPT}")
    with LOG.open("a") as fh:
        r = subprocess.run(["bash", str(SCRIPT)], stdout=fh, stderr=subprocess.STDOUT)
    log(f"update exit code {r.returncode}")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90,
                    help="Self-expire after this many days")
    ap.add_argument("--now", action="store_true",
                    help="Fire one update immediately, then run at 00:00 nightly")
    ap.add_argument("--force", action="store_true",
                    help="Clobber stale lock")
    args = ap.parse_args()

    if not acquire_lock(clobber=args.force):
        log("ERROR: offline scheduler already running — abort")
        return 1
    PID_FILE.write_text(str(os.getpid()))

    deadline = datetime.datetime.now() + datetime.timedelta(days=args.days)
    log(f"offline scheduler started pid={os.getpid()} expires={deadline.isoformat(timespec='seconds')}")

    def _shutdown(signum, frame):
        log(f"signal {signum} — shutting down")
        release_lock()
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Anchor the schedule: first fire at tomorrow 00:00, then every INTERVAL_HR.
    anchor = tomorrow_midnight()
    log(f"schedule anchor: {anchor.isoformat(timespec='seconds')} "
        f"+ every {INTERVAL_HR}h")

    try:
        if args.now:
            try:
                run_update()
            except Exception as e:
                log(f"update raised: {e}")

        while True:
            if datetime.datetime.now() >= deadline:
                log("expired — exiting")
                break
            target = next_fire_time(anchor)
            wait = (target - datetime.datetime.now()).total_seconds()
            log(f"sleeping {wait/3600:.2f} hr until {target.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(max(wait, 1))
            if datetime.datetime.now() >= deadline:
                break
            try:
                run_update()
            except Exception as e:
                log(f"update raised: {e}")
    finally:
        release_lock()
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
