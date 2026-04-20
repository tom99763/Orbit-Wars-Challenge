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
TODAY="$(date +%Y-%m-%d)"
YEST="$(date -d 'yesterday' +%Y-%m-%d)"
LATEST="training/checkpoints/offline_v1.pt"                # stable symlink, inference target
DATED="training/checkpoints/offline_v1_${TODAY}.pt"        # versioned, permanent

echo "[$TS] === nightly offline update starting (${TODAY}) ===" | tee -a "$LOG"

# Scrape fresh leaderboard + parse replays (via pipeline.py).
# TRAINING STAGES BELOW ARE DISABLED per user 2026-04-19: only scrape for now.
echo "[$(date -Iseconds)] scraping leaderboard via pipeline.py" >> "$LOG"
$PY pipeline.py >> "$LOG" 2>&1 || echo "[$(date -Iseconds)] pipeline.py failed" >> "$LOG"
echo "[$(date -Iseconds)] === 5h tick done (scrape only) ===" | tee -a "$LOG"
exit 0

# ---- Disabled stages below (dataset build + AWAC + online) ----
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

# Require the stable symlink/file to exist — bootstrap with a full
# offline_awac.py run first (which creates offline_v1.pt directly).
if [ ! -e "$LATEST" ]; then
    echo "[$(date -Iseconds)] ERROR: $LATEST missing — bootstrap with a full offline_awac.py run first" | tee -a "$LOG"
    exit 1
fi

echo "[$(date -Iseconds)] continuing AWAC from $LATEST -> $DATED (3 epochs)" >> "$LOG"
$PY training/offline_awac.py \
    --data-dir offline \
    --out "$DATED" \
    --init-from "$LATEST" \
    --epochs 3 --batch 128 --lr 1e-4 \
    >> "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ] && [ -f "$DATED" ]; then
    ln -sf "$(basename "$DATED")" "$LATEST"
    echo "[$(date -Iseconds)] rotated offline: $LATEST -> $DATED" >> "$LOG"
else
    echo "[$(date -Iseconds)] OFFLINE FAILED (rc=$RC); keeping previous $LATEST — SKIPPING ONLINE" >> "$LOG"
    echo "[$(date -Iseconds)] === nightly done (offline only, failed) ===" | tee -a "$LOG"
    exit $RC
fi

# -------- Online finetune stage --------
# Seeds from last night's online ckpt if present, else today's fresh offline.
ONLINE_LATEST="training/checkpoints/online_v1.pt"
ONLINE_DATED="training/checkpoints/online_v1_${TODAY}.pt"
if [ -e "$ONLINE_LATEST" ]; then
    ONLINE_SEED="$ONLINE_LATEST"
else
    ONLINE_SEED="$LATEST"
fi

echo "[$(date -Iseconds)] online finetune from $ONLINE_SEED -> $ONLINE_DATED (30 iters)" >> "$LOG"
$PY training/online_impala_v4.py \
    --bc-ckpt "$ONLINE_SEED" \
    --out "$ONLINE_DATED" \
    --iters 30 --workers 4 --four-player-prob 0.2 \
    --lb928-prob 0.0 --starter-prob 0.5 \
    --random-opp-prob 0.0 --random-opp-end 0.0 --random-opp-decay-iters 30 \
    --planet-action-noise 0.10 \
    --temp-start 1.0 --temp-end 0.8 \
    --ent-high 0.005 --ent-low 0.005 --ent-switch-iter 0 \
    --shape-decay-iters 15 \
    --lr 5e-5 --grad-steps-per-iter 2 --snapshot-every 10 \
    >> "$LOG" 2>&1
RC2=$?
if [ $RC2 -eq 0 ] && [ -f "$ONLINE_DATED" ]; then
    ln -sf "$(basename "$ONLINE_DATED")" "$ONLINE_LATEST"
    echo "[$(date -Iseconds)] rotated online: $ONLINE_LATEST -> $ONLINE_DATED" >> "$LOG"
else
    echo "[$(date -Iseconds)] ONLINE FAILED (rc=$RC2); keeping previous $ONLINE_LATEST" >> "$LOG"
fi

echo "[$(date -Iseconds)] === nightly done ===" | tee -a "$LOG"
