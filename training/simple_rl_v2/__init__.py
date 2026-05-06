"""simple_rl_v2 — self-play PPO with frozen-anchor league for Orbit Wars.

MVP implementation following CLAUDE-2.md spec. Targets the AlphaStar-style
"purple→cyan" monotonically-rising league/elo_learner curve.

Modules:
    physics — lead-target aim + sun-cross masking (numpy)
    model   — SimpleRLAgentV2 with MLP backbone + hierarchical action head
              + v14 ship rule + value head
    league  — PFSP opponent pool + frozen-anchor ELO bookkeeping
    train_rl — PPO training loop with batch adv-norm + value clip
"""
