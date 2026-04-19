#!/bin/bash
# Install the Orbit Wars scrape pipeline as a cron job.
#
# Usage:  install_cron.sh [--every MINUTES] [--days N]
#    --every MINUTES  cadence in minutes (default 60 = hourly; also valid:
#                     1, 2, 3, 5, 6, 10, 15, 20, 30)
#    --days   N       self-expire after N days (default 7)
#
# Idempotent — removes any existing orbit-wars cron entry before installing.

set -eu

EVERY=60
DAYS=7
while [ $# -gt 0 ]; do
    case "$1" in
        --every) EVERY="$2"; shift 2 ;;
        --days)  DAYS="$2";  shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
WRAPPER="$ROOT/cron_wrapper.sh"
MARKER="cron_wrapper.sh"

[ -x "$WRAPPER" ] || chmod +x "$WRAPPER"

# Record which python interpreter the wrapper should use (the one invoking
# this installer, which should have playwright + chromium deps installed).
PY="$(command -v python3)"
echo "$PY" > "$ROOT/.cron_python"

# Compute expiry
expiry=$(date -d "+${DAYS} days" +%s)
echo "$expiry" > "$ROOT/.cron_expiry"

# Build the crontab line
if [ "$EVERY" = "60" ]; then
    SCHED="0 * * * *"
else
    SCHED="*/${EVERY} * * * *"
fi
LINE="$SCHED $WRAPPER"

# Remove any prior orbit-wars entry, then append ours
( crontab -l 2>/dev/null | grep -v -F "$MARKER" ; echo "$LINE" ) | crontab -

echo "installed: $LINE"
echo "python:    $PY"
echo "expires:   $(date -d @"$expiry")"
echo "log:       $ROOT/.cron.log"
echo "remove:    $ROOT/uninstall_cron.sh"
