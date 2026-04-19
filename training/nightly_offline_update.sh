#!/bin/bash
# Nightly continual AWAC: ingest today's trajectories + warm-start from current
# offline_v1.pt + train a few more epochs. Safe to run repeatedly; idempotent
# per-date on the dataset side because output is keyed by episode filename.
#
# Intended to be fired by training/offline_scheduler.py at 00:00 daily.

set -u
ROOT="/home/lab/orbit-war"
PY="/home/lab/miniconda3/envs/tom/bin/python"
export LD_LIBRARY_PATH="/home/lab/miniconda3/envs/tom/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

TS="$(date -Iseconds)"
LOG="$ROOT/.offline_nightly.log"
CKPT="training/checkpoints/offline_v1.pt"

echo "[$TS] === nightly offline update starting ===" | tee -a "$LOG"

TODAY="$(date +%Y-%m-%d)"
YEST="$(date -d 'yesterday' +%Y-%m-%d)"

# Process today + yesterday (late trajectories may land in yesterday's folder)
for D in "$YEST" "$TODAY"; do
    TRAJ_DIR="$ROOT/trajectories/$D"
    OUT_DIR="$ROOT/offline/$D"
    if [ ! -d "$TRAJ_DIR" ]; then
        echo "[$(date -Iseconds)] skip $D (no trajectories dir)" >> "$LOG"
        continue
    fi
    echo "[$(date -Iseconds)] building offline dataset for $D" >> "$LOG"
    $PY training/build_offline_dataset.py \
        --traj-dir "$TRAJ_DIR" \
        --out-dir "$OUT_DIR" \
        >> "$LOG" 2>&1
done

# Require initial ckpt to exist (must be created by a manual bootstrap run first)
if [ ! -f "$CKPT" ]; then
    echo "[$(date -Iseconds)] ERROR: $CKPT missing — bootstrap with a full offline_awac.py run first" | tee -a "$LOG"
    exit 1
fi

echo "[$(date -Iseconds)] continuing AWAC from $CKPT (3 epochs)" >> "$LOG"
$PY training/offline_awac.py \
    --data-dir offline \
    --out "$CKPT" \
    --init-from "$CKPT" \
    --epochs 3 --batch 128 --lr 1e-4 \
    >> "$LOG" 2>&1

echo "[$(date -Iseconds)] === nightly done ===" | tee -a "$LOG"
