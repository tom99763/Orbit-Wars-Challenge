#!/bin/bash
# Phase 2 → Phase 3.5 auto-chain.
# 1. Online imitation: 20k lb-1200 self-play games streamed into BC training
# 2. Eval vs starter / lb928 / lb1200 (50 games each)
# 3. Log + report

set -u
ROOT=/home/lab/orbit-war
PY=/home/lab/miniconda3/envs/tom/bin/python
export LD_LIBRARY_PATH=/home/lab/miniconda3/envs/tom/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
LOG="$ROOT/.imitation_then_eval.log"
TODAY=$(date +%Y-%m-%d)
CKPT_DATED="training/checkpoints/imitation_v1_${TODAY}.pt"
CKPT_LATEST="training/checkpoints/imitation_v1.pt"

echo "[$(date -Iseconds)] === Phase 2: online imitation (20k games) ===" | tee -a "$LOG"
$PY training/online_imitation.py \
    --target-games 20000 --workers 8 --four-player-prob 0.5 \
    --out "$CKPT_DATED" \
    --lr 3e-4 --batch 128 --grad-steps-per-iter 4 \
    --buffer-size 5000 --snapshot-every 50 \
    >> "$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -f "$CKPT_DATED" ]; then
    echo "[$(date -Iseconds)] IMITATION FAILED rc=$RC" | tee -a "$LOG"
    exit $RC
fi
ln -sf "$(basename $CKPT_DATED)" "$CKPT_LATEST"
echo "[$(date -Iseconds)] imitation done, saved $CKPT_DATED + symlink" | tee -a "$LOG"

echo "[$(date -Iseconds)] === Phase 3.5: eval suite (50 games × 3 opps) ===" | tee -a "$LOG"
$PY training/eval_suite.py \
    --ckpt "$CKPT_LATEST" \
    --n-games 50 --four-player-prob 0.2 \
    --opponents starter,lb928,lb1200 \
    >> "$LOG" 2>&1

echo "[$(date -Iseconds)] === auto-chain done ===" | tee -a "$LOG"
