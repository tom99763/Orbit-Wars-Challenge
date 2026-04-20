#!/bin/bash
# Continual online↔offline alternation, forever.
#
# Each loop iter:
#   1. pipeline.py — scrape + parse new leaderboard replays into trajectories/{date}/
#   2. build_offline_dataset.py — today+yesterday trajectories → offline/{date}/*.npz
#   3. offline_awac.py — 3 epochs, warm from online_v2.pt (previous online),
#      save as offline_v1_{date}_loop{N}.pt + update offline_v1.pt symlink
#   4. online_impala_v4.py — 200 iter × 8 workers = 1600 games, warm from
#      offline_v1.pt, save as online_v2_{date}_loop{N}.pt + update online_v2.pt
#
# Chain: online_v2(N-1) → AWAC + new data → offline_v1(N) → IMPALA → online_v2(N)
# Both offline and online gains are preserved across loops.

set -u
ROOT=/home/lab/orbit-war
PY=/home/lab/miniconda3/envs/tom/bin/python
export LD_LIBRARY_PATH=/home/lab/miniconda3/envs/tom/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
LOG="$ROOT/.continual_loop.log"

echo "[$(date -Iseconds)] continual_loop start pid=$$" | tee -a "$LOG"

# Wait for any currently-running online_impala_v4 to finish before first loop
while pgrep -f "online_impala_v4.py" > /dev/null; do
    echo "[$(date -Iseconds)] waiting for existing online run..." >> "$LOG"
    sleep 300
done

ITER=1
while true; do
    TODAY="$(date +%Y-%m-%d)"
    YEST="$(date -d 'yesterday' +%Y-%m-%d)"
    OFFLINE_LATEST="training/checkpoints/offline_v1.pt"
    ONLINE_LATEST="training/checkpoints/online_v2.pt"
    OFFLINE_DATED="training/checkpoints/offline_v1_${TODAY}_loop${ITER}.pt"
    ONLINE_DATED="training/checkpoints/online_v2_${TODAY}_loop${ITER}.pt"

    echo "[$(date -Iseconds)] === loop iter $ITER start ===" | tee -a "$LOG"

    # 1. Scrape latest leaderboard
    echo "[$(date -Iseconds)] [stage 1/4] scraping leaderboard" >> "$LOG"
    $PY pipeline.py >> "$LOG" 2>&1 || echo "[$(date -Iseconds)] pipeline.py failed" >> "$LOG"

    # 2. Build today/yesterday offline datasets
    for D in "$YEST" "$TODAY"; do
        TRAJ_DIR="$ROOT/trajectories/$D"
        if [ -d "$TRAJ_DIR" ]; then
            echo "[$(date -Iseconds)] [stage 2/4] building dataset $D" >> "$LOG"
            $PY training/build_offline_dataset.py \
                --traj-dir "$TRAJ_DIR" \
                --out-dir "$ROOT/offline/$D" \
                >> "$LOG" 2>&1
        fi
    done

    # 3. AWAC warm from previous ONLINE (not offline), 3 epochs
    if [ -e "$ONLINE_LATEST" ]; then
        AWAC_SEED="$ONLINE_LATEST"
    else
        AWAC_SEED="$OFFLINE_LATEST"
    fi
    echo "[$(date -Iseconds)] [stage 3/4] AWAC from $AWAC_SEED -> $OFFLINE_DATED" >> "$LOG"
    $PY training/offline_awac.py \
        --data-dir offline \
        --out "$OFFLINE_DATED" \
        --init-from "$AWAC_SEED" \
        --epochs 3 --batch 128 --lr 1e-4 \
        >> "$LOG" 2>&1
    RC=$?
    if [ $RC -eq 0 ] && [ -f "$OFFLINE_DATED" ]; then
        ln -sf "$(basename $OFFLINE_DATED)" "$OFFLINE_LATEST"
        echo "[$(date -Iseconds)] offline rotated -> $OFFLINE_DATED" >> "$LOG"
    else
        echo "[$(date -Iseconds)] AWAC FAILED (rc=$RC), skipping online stage" >> "$LOG"
        sleep 600
        ITER=$((ITER+1))
        continue
    fi

    # 4. Online 1600-game run warm from just-updated OFFLINE
    echo "[$(date -Iseconds)] [stage 4/4] online from $OFFLINE_LATEST -> $ONLINE_DATED" >> "$LOG"
    $PY training/online_impala_v4.py \
        --bc-ckpt "$OFFLINE_LATEST" \
        --out "$ONLINE_DATED" \
        --iters 200 --workers 8 --four-player-prob 0.2 \
        --lb928-prob 0.0 --starter-prob 0.5 \
        --random-opp-prob 0.0 --random-opp-end 0.0 --random-opp-decay-iters 30 \
        --planet-action-noise 0.02 \
        --temp-start 0.8 --temp-end 0.7 \
        --ent-high 0.003 --ent-low 0.003 --ent-switch-iter 0 \
        --shape-decay-iters 1 --shape-weight-end 0.2 \
        --lr 5e-5 --grad-steps-per-iter 2 --snapshot-every 10 \
        --beat-starter-bonus 0.5 --beat-lb928-bonus 1.0 \
        >> "$LOG" 2>&1
    RC=$?
    if [ $RC -eq 0 ] && [ -f "$ONLINE_DATED" ]; then
        ln -sf "$(basename $ONLINE_DATED)" "$ONLINE_LATEST"
        echo "[$(date -Iseconds)] online rotated -> $ONLINE_DATED" >> "$LOG"
    else
        echo "[$(date -Iseconds)] ONLINE FAILED (rc=$RC)" >> "$LOG"
    fi

    echo "[$(date -Iseconds)] === loop iter $ITER done ===" | tee -a "$LOG"
    ITER=$((ITER+1))
done
