#!/bin/bash
# Cron wrapper for the Orbit Wars pipeline.
#
# - Self-expires: reads .cron_expiry (epoch seconds) in the project root;
#   removes itself from crontab when now >= expiry.
# - Non-overlapping: skips if a previous run is still in flight.
# - Auto-detects project root and Python env — portable across users.
# - Logs to .cron.log next to this script.
#
# Install by running `install_cron.sh`.

set -u

# Resolve the script's own directory so the wrapper is path-independent.
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOG="$ROOT/.cron.log"
EXPIRY_FILE="$ROOT/.cron_expiry"
LOCK="$ROOT/.cron.lock"
MARKER="cron_wrapper.sh"  # unique substring used for self-removal

cd "$ROOT" || { echo "[$(date -Is)] cannot cd to $ROOT" >> "$LOG"; exit 1; }

# Expiry check
if [ -f "$EXPIRY_FILE" ]; then
    now_ts=$(date +%s)
    exp_ts=$(cat "$EXPIRY_FILE" 2>/dev/null || echo 0)
    if [ "$now_ts" -ge "$exp_ts" ]; then
        echo "[$(date -Is)] expired (expiry=$exp_ts, now=$now_ts) — removing cron entry" >> "$LOG"
        ( crontab -l 2>/dev/null | grep -v -F "$MARKER" ) | crontab -
        rm -f "$EXPIRY_FILE"
        exit 0
    fi
fi

# Non-overlap lock (atomic mkdir)
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "[$(date -Is)] previous run still active, skipping" >> "$LOG"
    exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# Auto-detect the Python env (miniconda/venv/etc). Users should install
# playwright + chromium deps into the same env that invoked install_cron.sh;
# the env's prefix was recorded at install time in .cron_python.
if [ -f "$ROOT/.cron_python" ]; then
    PY="$(cat "$ROOT/.cron_python")"
else
    PY="$(command -v python3)"
fi
if [ ! -x "$PY" ]; then
    echo "[$(date -Is)] python not executable: $PY" >> "$LOG"
    exit 1
fi

# Chromium needs LD_LIBRARY_PATH pointing at the env's lib dir
PY_PREFIX="$("$PY" -c 'import sys;print(sys.prefix)')"
export LD_LIBRARY_PATH="$PY_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$PY_PREFIX/bin:$PATH"

echo "[$(date -Is)] ---- pipeline start (python=$PY) ----" >> "$LOG"
if "$PY" "$ROOT/pipeline.py" >> "$LOG" 2>&1; then
    echo "[$(date -Is)] ---- pipeline ok ----" >> "$LOG"
else
    rc=$?
    echo "[$(date -Is)] ---- pipeline FAILED rc=$rc ----" >> "$LOG"
fi
