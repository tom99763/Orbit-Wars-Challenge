#!/bin/bash
# Polls the imitation log, and whenever a new iter that is a multiple of 50
# appears, runs a 20-game eval vs starter using the latest checkpoint.
# Appends one CSV row per run to .eval_every_50.csv.
#
# Intended to run in background:
#   nohup bash training/eval_every_50.sh > .eval_every_50.stdout 2>&1 &
set -u
ROOT=/home/lab/orbit-war
PY=/home/lab/miniconda3/envs/tom/bin/python
export LD_LIBRARY_PATH=/home/lab/miniconda3/envs/tom/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"

TRAIN_LOG="$ROOT/.imitation_then_eval.log"
CKPT_LATEST="$ROOT/training/checkpoints/imitation_v1_2026-04-20.pt"
OUT_CSV="$ROOT/.eval_every_50.csv"
OUT_DETAIL="$ROOT/.eval_every_50.detail.log"
STATE="$ROOT/.eval_every_50.state"   # last iter evaluated

# Ensure header
if [ ! -f "$OUT_CSV" ]; then
    echo "iter,datetime,n_games,wins,win_rate,seat0_w,seat0_n,seat1_w,seat1_n" > "$OUT_CSV"
fi

# Read last evaluated iter; 0 if none.
LAST_EVALED=$(cat "$STATE" 2>/dev/null || echo 0)

echo "[$(date -Iseconds)] eval_every_50 watcher started (last_evaled=$LAST_EVALED)" | tee -a "$OUT_DETAIL"

while true; do
    # Find the highest iter seen in the log so far.
    LATEST=$(grep -oP '\[iter \K[0-9]+' "$TRAIN_LOG" 2>/dev/null | tail -1)
    if [ -z "$LATEST" ]; then sleep 60; continue; fi
    # Strip leading zeros
    LATEST_NUM=$((10#$LATEST))
    # Find next unevaluated multiple of 50
    NEXT_TARGET=$(( (LAST_EVALED / 50 + 1) * 50 ))
    if [ "$LATEST_NUM" -lt "$NEXT_TARGET" ]; then
        sleep 120
        continue
    fi
    # Target reached. Confirm the checkpoint was just saved by checking log.
    # With --snapshot-every 10, every 10-iter boundary saves. 50 is a multiple of 10.
    # Small grace period for file flush.
    sleep 5

    if [ ! -f "$CKPT_LATEST" ]; then
        echo "[$(date -Iseconds)] ckpt missing: $CKPT_LATEST; retrying" | tee -a "$OUT_DETAIL"
        sleep 120
        continue
    fi

    STAMP=$(date -Iseconds)
    echo "[$STAMP] iter=$NEXT_TARGET → running eval..." | tee -a "$OUT_DETAIL"

    # CPU-only to not steal GPU from training; kaggle_env is CPU anyway
    CUDA_VISIBLE_DEVICES="" \
        $PY training/eval_suite.py \
            --ckpt "$CKPT_LATEST" \
            --n-games 20 --four-player-prob 0.0 \
            --opponents starter \
            > "$OUT_DETAIL.tmp" 2>&1
    RC=$?
    cat "$OUT_DETAIL.tmp" >> "$OUT_DETAIL"

    if [ $RC -ne 0 ]; then
        echo "[$(date -Iseconds)] iter=$NEXT_TARGET EVAL FAILED rc=$RC" | tee -a "$OUT_DETAIL"
    else
        # Parse result lines:
        #   "[starter] 3/20 (15.0%)  [Xs]"
        #   "  2P seat 0: 2/13"
        #   "  2P seat 1: 1/7"
        STARTER_LINE=$(grep -oP '\[starter\] \K[0-9]+/[0-9]+ \([0-9.]+%\)' "$OUT_DETAIL.tmp" | head -1)
        WINS=$(echo "$STARTER_LINE" | grep -oP '^\K[0-9]+')
        N=$(echo "$STARTER_LINE" | grep -oP '/\K[0-9]+' | head -1)
        RATE=$(echo "$STARTER_LINE" | grep -oP '\(\K[0-9.]+(?=%\))' )

        S0=$(grep -oP '2P seat 0: \K[0-9]+/[0-9]+' "$OUT_DETAIL.tmp" | head -1)
        S1=$(grep -oP '2P seat 1: \K[0-9]+/[0-9]+' "$OUT_DETAIL.tmp" | head -1)
        S0_W=${S0%/*}; S0_N=${S0#*/}
        S1_W=${S1%/*}; S1_N=${S1#*/}

        echo "$NEXT_TARGET,$STAMP,$N,$WINS,$RATE,$S0_W,$S0_N,$S1_W,$S1_N" >> "$OUT_CSV"
        echo "[$(date -Iseconds)] iter=$NEXT_TARGET → $WINS/$N ($RATE%)  s0=$S0  s1=$S1" | tee -a "$OUT_DETAIL"
    fi

    rm -f "$OUT_DETAIL.tmp"
    LAST_EVALED=$NEXT_TARGET
    echo "$LAST_EVALED" > "$STATE"
done
