#!/bin/bash
# Launch scheduler.py with the right Python + LD_LIBRARY_PATH, detached.
#
# Usage:
#   ./run_scheduler.sh [--every 60] [--days 14] [--now]
#
# Writes .scheduler.log (structured) and .scheduler.stdout (subprocess raw).

set -eu
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PY="/home/lab/miniconda3/envs/tom/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
export LD_LIBRARY_PATH="$("$PY" -c 'import sys;print(sys.prefix)')/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ARGS="$*"
if [ -z "$ARGS" ]; then
    ARGS="--every 60 --days 14"
fi

if [ -f "$ROOT/.scheduler.pid" ] && kill -0 "$(cat "$ROOT/.scheduler.pid" 2>/dev/null)" 2>/dev/null; then
    echo "scheduler already running (pid $(cat "$ROOT/.scheduler.pid")). To stop: kill \$(cat $ROOT/.scheduler.pid)"
    exit 1
fi

nohup "$PY" "$ROOT/scheduler.py" $ARGS > "$ROOT/.scheduler.stdout" 2>&1 &
NEW_PID=$!
echo "started scheduler pid=$NEW_PID  args=$ARGS"
echo "  log:    tail -f $ROOT/.scheduler.log"
echo "  stop:   kill $NEW_PID"
echo "  status: ps -p $NEW_PID"
