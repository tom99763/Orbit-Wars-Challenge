#!/bin/bash
# RL launcher — called once by the session /loop when BC has produced a valid
# teacher.pkl + passed GATE2. Starts the full-Lux v92 PPO self-play run with
# teacher-KL distillation on GPU 0 (per user 2026-05-22: GPU 1 is the user's
# BirdClef train_tucker_sed job @89% util — orbit-war MUST stay off GPU1).
# XLA_PYTHON_CLIENT_MEM_FRACTION caps JAX's GPU0 preallocation so it coexists
# with BirdClef's small GPU0 footprint instead of greedily grabbing 75%.
#
# Feature flags MUST be IDENTICAL to the BC collect+train flags, or the frozen
# teacher-net's first layer shape-mismatches the RL features → crash at init.
#   USE_SHIP_HEAD=1 N_SHIP_BUCKETS=11 + 10 BIAS_* flags  → feature_dims (18,11).
#
# Run as full 100k from the start; the session /loop gates at ~upd 300 (= the
# "smoke") and kills if it diverges, else lets it run to 100k (= auto-full).
#
# ── Multi-GPU (data-parallel) — OPT-IN ────────────────────────────────────────
# When GPU1 becomes free (BirdClef done), launch with N_GPUS=2 in the env:
#     N_GPUS=2 N_ENVS=256 bash training/v92/launch_rl.sh
# It will set CUDA_VISIBLE_DEVICES=0,1 and pass --n-gpus 2 to the trainer
# (env batch sharded along n_envs axis; params + opt_state + last_best
# replicated; PPO minibatch runs replicated-per-device — see train_jax.py
# comments for details). Default = single-GPU on 0.
# Hardware monitor (CPU/RAM/VRAM/util) lands in $SAVE/sys.csv (--monitor-every).
set -e
cd /home/lab/orbit-war
SAVE=save/v92_rl_teacher
LOG=save/teacher/rl_train.log
TEACHER=save/teacher/teacher.pkl

[ -f "$TEACHER" ] || { echo "[launch_rl] ABORT: $TEACHER missing" | tee -a "$LOG"; exit 1; }

# Multi-GPU opt-in (env vars). Defaults preserve single-GPU on GPU0.
N_GPUS=${N_GPUS:-1}
N_ENVS=${N_ENVS:-128}
MONITOR_EVERY=${MONITOR_EVERY:-50}
if [ "$N_GPUS" -gt 1 ]; then
  # Build a comma-list of device indices 0..N_GPUS-1
  CUDA_DEVS=$(seq -s, 0 $((N_GPUS - 1)))
else
  CUDA_DEVS=0
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVS XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
USE_SHIP_HEAD=1 N_SHIP_BUCKETS=11 \
BIAS_DEFENSE_LOOK=1 BIAS_CRASH_EXPLOIT=1 BIAS_GANG_UP=1 BIAS_INDIRECT_VAL=1 \
BIAS_STAGE_ONEHOT=1 BIAS_SUNDANCER_PHASE=1 BIAS_CAPTURE_COST=1 BIAS_THREAT_INFLOW=1 \
BIAS_PROD_CP=1 BIAS_MIN_SHIP_FLOOR=1 \
TEACHER_KL=1 TEACHER_CKPT="$TEACHER" \
GAMMA=0.9999 GAE_LAMBDA=0.85 ZEROSUM_VALUE=1 HUBER_VALUE=1 \
PURE_SELFPLAY=1 LAST_BEST_GATE=1 BIAS_SIGMOID_GAP_REWARD=1 \
PYTHONPATH=/home/lab/orbit-war \
nohup /home/lab/miniconda3/envs/tom/bin/python training/v92/train_jax.py \
  --n-envs "$N_ENVS" --t-rollout 64 --total-updates 100000 --save-dir "$SAVE" \
  --ent-coef 0.01 --eval-every 1000 \
  --n-gpus "$N_GPUS" --monitor-every "$MONITOR_EVERY" \
  >> "$LOG" 2>&1 &

RLPID=$!
echo "$RLPID" > save/teacher/rl.pid
echo "save/teacher/rl_train.log" > save/teacher/rl.logpath
echo "[launch_rl] $(date '+%F %T') RL started PID $RLPID on GPU(s) $CUDA_DEVS (N_GPUS=$N_GPUS, N_ENVS=$N_ENVS), save=$SAVE, teacher-KL ON, full-Lux. total=100000, monitor every $MONITOR_EVERY upd." | tee -a "$LOG"
echo "$RLPID"
