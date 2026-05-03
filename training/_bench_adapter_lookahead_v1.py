"""Benchmark adapter for the existing committed submission/lb1200_lookahead
single-file (the version that scored 836.4 on Kaggle on 2026-04-27 12:11).

Loads it as a fresh module under a unique label so it doesn't collide with
submission/raw_lb1200/main.py (both have stem "main").
"""
import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

_TARGET = _REPO / "submission" / "lb1200_lookahead" / "main.py"
_spec = importlib.util.spec_from_file_location("lookahead_v1_kaggle", _TARGET)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
agent = _mod.agent
