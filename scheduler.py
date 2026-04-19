"""Poor-man's cron for WSL2 / systemd-less hosts.

Runs pipeline.py at a fixed cadence (default hourly, on the hour). Holds
a lock file so only one scheduler runs at a time. Self-expires after
--days days. Logs every tick to .scheduler.log next to this file.

Usage:

  # Start (foreground, backgroundable with nohup)
  nohup /home/lab/miniconda3/envs/tom/bin/python scheduler.py \
        --every 60 --days 14 \
      > .scheduler.stdout 2>&1 &

  # Helper wrapper:
  ./run_scheduler.sh

  # Stop:
  cat .scheduler.pid  # PID
  kill $(cat .scheduler.pid)

  # Status:
  ps -p $(cat .scheduler.pid)
  tail -f .scheduler.log

Designed to be safe-ish: if pipeline.py fails, we log and continue —
next tick will retry. We don't abort on transient errors because hourly
cadence means a hiccup is cheap to miss.
"""

import argparse
import datetime
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).parent
LOG = ROOT / ".scheduler.log"
PID_FILE = ROOT / ".scheduler.pid"
LOCK = ROOT / ".scheduler.lock"


def log(msg: str) -> None:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def acquire_lock(clobber: bool) -> bool:
    """Exclusive lock via an atomic mkdir. If an old lock exists but the
    owning PID is dead, clobber it."""
    if LOCK.exists():
        # Check if owner is still alive
        try:
            owner = int((LOCK / "pid").read_text().strip())
            os.kill(owner, 0)  # throws if not running
            return False  # still alive
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


def wait_until_next_tick(every_min: int) -> None:
    """Sleep until the next top-of-interval boundary (e.g. every 60 →
    next xx:00:00)."""
    now = datetime.datetime.now()
    minutes_since_hour = now.minute + now.second / 60.0
    delta_min = every_min - (minutes_since_hour % every_min)
    if delta_min < 0.5:  # already within the tick window → just go
        delta_min = every_min  # wait a full period
    target = now + datetime.timedelta(minutes=delta_min)
    log(f"sleeping {delta_min:.1f} min until {target.strftime('%H:%M:%S')}")
    time.sleep(delta_min * 60)


def run_pipeline() -> int:
    """Launch pipeline.py with the env set up the same way cron_wrapper.sh did."""
    py = sys.executable
    env = {**os.environ}
    prefix = subprocess.run(
        [py, "-c", "import sys; print(sys.prefix)"],
        capture_output=True, text=True, check=True
    ).stdout.strip()
    ld = f"{prefix}/lib"
    if env.get("LD_LIBRARY_PATH"):
        ld = f"{ld}:{env['LD_LIBRARY_PATH']}"
    env["LD_LIBRARY_PATH"] = ld
    env["PATH"] = f"{prefix}/bin:{env.get('PATH', '')}"

    log(f"running: {py} pipeline.py  (LD_LIBRARY_PATH={ld})")
    with open(LOG, "a") as fh:
        r = subprocess.run(
            [py, str(ROOT / "pipeline.py")],
            env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT
        )
    log(f"pipeline exit code {r.returncode}")
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=60,
                    help="Tick cadence in minutes (default 60)")
    ap.add_argument("--days", type=int, default=14,
                    help="Self-expire after this many days (default 14)")
    ap.add_argument("--now", action="store_true",
                    help="Fire one tick immediately, then settle into the rhythm")
    ap.add_argument("--force", action="store_true",
                    help="Ignore stale lock")
    args = ap.parse_args()

    if not acquire_lock(clobber=args.force):
        log("ERROR: scheduler already running (lock held) — abort")
        return 1

    # Write PID for external stop
    PID_FILE.write_text(str(os.getpid()))

    deadline = datetime.datetime.now() + datetime.timedelta(days=args.days)
    log(f"scheduler started pid={os.getpid()} every={args.every}min "
        f"expires={deadline.isoformat(timespec='seconds')}")

    def _shutdown(signum, frame):
        log(f"signal {signum} received — shutting down")
        release_lock()
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        if args.now:
            run_pipeline()

        while True:
            if datetime.datetime.now() >= deadline:
                log("expired — exiting")
                break
            wait_until_next_tick(args.every)
            if datetime.datetime.now() >= deadline:
                log("expired during sleep — exiting")
                break
            try:
                run_pipeline()
            except Exception as e:
                log(f"pipeline raised: {e}")
    finally:
        release_lock()
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
