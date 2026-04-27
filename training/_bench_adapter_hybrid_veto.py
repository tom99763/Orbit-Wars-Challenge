"""Tiny adapter so training/benchmark.py can load the lb1200_with_k14_veto
hybrid as if it were a single-file `agent` module.

Edit the CKPT_PATH below to point at the checkpoint to wrap.
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from training.hybrid_agent import load_hybrid_agent

CKPT_PATH = "training/checkpoints/k14_vec_pfsp_v2.pt"
agent = load_hybrid_agent(CKPT_PATH, strategy="lb1200_with_k14_veto")
