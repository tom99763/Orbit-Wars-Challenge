"""Build a self-contained Kaggle submission bundling lb-1200 + lookahead.

Cross-platform Python rewrite of build_submission_lookahead.sh (which had
hardcoded Linux paths /home/lab/orbit-war and conda env). Same behaviour:
inlines training/lb1200_agent.py + training/lb1200_lookahead_agent.py into
one self-contained submission/lb1200_lookahead/main.py.

Output:
  submission/lb1200_lookahead/main.py   (self-contained, ~113 KB)

Submit (single-file, no tar.gz needed for lookahead):
  kaggle competitions submit orbit-wars \\
      -f submission/lb1200_lookahead/main.py \\
      -m "<msg>"
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_main_py() -> pathlib.Path:
    base_path = REPO_ROOT / "training" / "lb1200_agent.py"
    wrap_path = REPO_ROOT / "training" / "lb1200_lookahead_agent.py"
    out_path = REPO_ROOT / "submission" / "lb1200_lookahead" / "main.py"

    base_src = base_path.read_text(encoding="utf-8")
    wrap_src = wrap_path.read_text(encoding="utf-8")

    # 1. Rename base's `agent` → `_lb1200_base_agent`
    base_src = re.sub(
        r"^def agent\(obs, config=None\):",
        "def _lb1200_base_agent(obs, config=None):",
        base_src,
        count=1,
        flags=re.MULTILINE,
    )
    base_src = base_src.replace(
        '__all__ = ["agent", "build_world"]',
        '__all__ = ["_lb1200_base_agent", "build_world"]',
    )

    # 2. Strip wrapper's `from training.lb1200_agent import (...)` block
    wrap_src = re.sub(
        r"from training\.lb1200_agent import \([\s\S]*?\)",
        "# [inlined above: _lb1200_base_agent, build_world, "
        "simulate_planet_timeline, etc.]",
        wrap_src,
        count=1,
    )
    # Strip wrapper's `from __future__` (must be file-top; base_src has it)
    wrap_src = re.sub(
        r"^from __future__ import [^\n]+\n",
        "",
        wrap_src,
        count=1,
        flags=re.MULTILINE,
    )

    # 3. Strip `agent_debug` — Kaggle's grader may pick up any name starting
    # with "agent" and the debug version returns a tuple (breaks env).
    wrap_src = re.sub(
        r"# Convenience: a \"debug\".*?(?=\n__all__|\Z)",
        "",
        wrap_src,
        count=1,
        flags=re.DOTALL,
    )
    wrap_src = re.sub(
        r'__all__ = \["agent", "agent_debug"\]',
        '__all__ = ["agent"]',
        wrap_src,
        count=1,
    )

    # 4. Wrapper internally references `lb1200_base_agent` (alias) — point
    # those at the inlined `_lb1200_base_agent`
    wrap_src = wrap_src.replace("lb1200_base_agent", "_lb1200_base_agent")

    combined = (
        "# Auto-generated Kaggle submission — lb-1200 + shallow lookahead.\n"
        "# Do NOT edit directly. Regenerate via "
        "training/build_submission_lookahead.py.\n\n"
        + base_src
        + "\n\n# ============================================================\n"
        + "# Lookahead wrapper (from training/lb1200_lookahead_agent.py)\n"
        + "# ============================================================\n\n"
        + wrap_src
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(combined, encoding="utf-8")
    return out_path


def smoke_test(main_py: pathlib.Path) -> None:
    """Verify the generated file imports and exports a callable agent."""
    spec = importlib.util.spec_from_file_location(
        "kaggle_submission_main", main_py,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise RuntimeError("generated main.py missing `agent` export")
    if not callable(mod.agent):
        raise RuntimeError("`agent` exists but is not callable")


def main() -> int:
    print("inlining lb-1200 + lookahead wrapper into single file ...")
    out = build_main_py()
    size_kb = out.stat().st_size / 1024
    print(f"  → {out.relative_to(REPO_ROOT)} ({size_kb:.1f} KB)")

    print("smoke-testing import + agent callable ...")
    smoke_test(out)
    print("  ✓ agent() importable")

    print("\nReady to submit:")
    print(
        '  "E:/ANACONDA_NEW/Scripts/kaggle.exe" competitions submit orbit-wars \\'
    )
    print(f"      -f {out.relative_to(REPO_ROOT)} \\")
    print('      -m "lb-1200 + lookahead v2 (prod-aware + comet-bonus + h=50)"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
