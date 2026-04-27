"""Adapter for the k14_with_lb1200_fill hybrid; see _bench_adapter_hybrid_veto.py."""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from training.hybrid_agent import load_hybrid_agent

CKPT_PATH = "training/checkpoints/k14_vec_pfsp_v2.pt"
agent = load_hybrid_agent(CKPT_PATH, strategy="k14_with_lb1200_fill")
