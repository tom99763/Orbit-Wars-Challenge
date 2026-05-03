"""Benchmark adapter for the improved lb1200_lookahead_agent (v2).

v2 changes vs the existing submission/lb1200_lookahead/main.py:
  - LOOKAHEAD_HORIZON 30 → 50
  - Production score added (W_MY_PRODUCTION, W_ENEMY_PRODUCTION)
  - Comet-window bonus on my-side scoring
  - 2 extra variants (drop strongest, drop two weakest)
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from training.lb1200_lookahead_agent import agent  # noqa: F401
