"""Kaggle submission entry point — k14 + lb1200 hybrid agent.

Strategy: lb1200_with_k14_veto. lb1200 proposes actions; k14's mode
head argmax acts as a binary act/pass gate, dropping any proposed
action where the trained policy thinks the source planet should pass.
Tactical execution (target choice, aim, ship count) stays with lb1200.

Bundle layout (must be preserved when packaging tar.gz):
    main.py                                 (this file)
    model.pt                                (k14 DualStreamK13Agent state_dict)
    featurize.py                            (state featurizer)
    training/__init__.py                    (empty)
    training/model.py                       (SetAttentionBlock)
    training/dual_stream_model.py           (SpatialCNN, ScalarMLP, rasterize_obs)
    training/lb1200_agent.py                (lb1200 rule-based agent + helpers)
    training/physics_action_helper_k13.py   (mode mask, top-K cands, materialize)
    training/physics_picker_k13_ppo.py      (DualStreamK13Agent class)
    training/hybrid_agent.py                (load_hybrid_agent)

To swap checkpoints: replace model.pt with a different DualStreamK13Agent
state_dict — same architecture, just different trained weights.
"""
import pathlib
import sys

# Bundle root is this file's directory; insert FIRST so our `training/`
# subdir wins over any same-named module that might be on the grader's
# path. Without this, `from training.lb1200_agent import ...` fails.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from training.hybrid_agent import load_hybrid_agent

_CKPT = _HERE / "model.pt"
_AGENT = load_hybrid_agent(
    str(_CKPT),
    strategy="lb1200_with_k14_veto",
    device="cpu",
)


def agent(obs, config=None):
    return _AGENT(obs, config)
