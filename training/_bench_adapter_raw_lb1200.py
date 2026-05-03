"""Benchmark adapter for submission/raw_lb1200/main.py.

Loads with a unique module name to avoid path-stem collision with
submission/lb1200_lookahead/main.py in benchmark.py's label table.
"""
import importlib.util
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))

_TARGET = _REPO / "submission" / "raw_lb1200" / "main.py"
_spec = importlib.util.spec_from_file_location("raw_lb1200_kaggle", _TARGET)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
agent = _mod.agent
