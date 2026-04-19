# alphazero_v1_2026-04-19

## What

First AlphaZero self-play run on top of `bc_v2`. Policy-only deployment (mode A).
See `/home/lab/orbit-war/wiki/rl-methods.md` and `wiki/training-design.md` for
context.

## Config

```
training/alphazero.py
  --bc-ckpt      training/checkpoints/bc_v2.pt
  --out          training/checkpoints/alphazero_v1.pt
  --iters        125
  --workers      4
  --n-sims       10       # Sampled MCTS simulations per MCTS call
  --k-samples    4        # joint actions sampled per MCTS node
  --four-player-prob 0.5  # half games 2p, half 4p
  --grad-steps-per-iter 2
  --lr           1e-4
```

Plus:
- Schema-validation bypass for env.step action checks (2-3× speedup)
- Per-seat value-head feedback (winners AND losers contribute value loss)
- Policy loss from learner seat only (MCTS visits as target)
- Learner seat randomised per game

## Files in this dir

```
run.log      → symlink to /home/lab/orbit-war/.alphazero_run.log
monitor.csv  → symlink to /home/lab/orbit-war/.alphazero_monitor.csv
model.pt     → symlink to /home/lab/orbit-war/training/checkpoints/alphazero_v1.pt
```

After the run finishes, use `training/generate_report.py` to produce an HTML
summary.

## Starting context

- Bootstrap: `bc_v2` — Set-Transformer 0.9 M params, 20 % vs starter,
  0 % vs lb-928
- Submission for bc_v2 (submissionId 51835195) had ERROR status on Kaggle
- Train time estimate: iter time ~1-3 min (4 workers), 125 iters =
  ~2-6 hours wall time depending on machine contention
