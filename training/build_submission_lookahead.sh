#!/bin/bash
# Build a self-contained Kaggle submission bundling lb-1200 + lookahead wrapper.
# Outputs:
#   submission/lb1200_lookahead/main.py       (self-contained, importable by Kaggle)
#   submission/lb1200_lookahead/submission.tar.gz
#
# After running, submit via:
#   kaggle competitions submit orbit-wars -f submission/lb1200_lookahead/submission.tar.gz -m "<msg>"

set -u
ROOT=/home/lab/orbit-war
OUT=$ROOT/submission/lb1200_lookahead
mkdir -p "$OUT"

PY=/home/lab/miniconda3/envs/tom/bin/python
cd "$ROOT"

"$PY" - <<'EOF'
import re
import pathlib

ROOT = pathlib.Path(".")
out = ROOT / "submission" / "lb1200_lookahead" / "main.py"

# Read lb-1200 base (big file, contains function named `agent`)
base_src = (ROOT / "training" / "lb1200_agent.py").read_text()

# Read lookahead wrapper (imports from training.lb1200_agent)
wrap_src = (ROOT / "training" / "lb1200_lookahead_agent.py").read_text()

# --- Step 1: rename base `agent` to `_lb1200_base_agent` in base_src ---
# The base file has exactly one top-level `def agent(obs, config=None):`
# Replace it with a renamed version.
base_src = re.sub(
    r"^def agent\(obs, config=None\):",
    "def _lb1200_base_agent(obs, config=None):",
    base_src,
    count=1,
    flags=re.MULTILINE,
)

# Also rewrite __all__ if present (so we don't accidentally export the old name)
base_src = base_src.replace(
    '__all__ = ["agent", "build_world"]',
    '__all__ = ["_lb1200_base_agent", "build_world"]',
)

# --- Step 2: strip the wrapper's `from training.lb1200_agent import (...)` ---
wrap_src = re.sub(
    r"from training\.lb1200_agent import \([\s\S]*?\)",
    "# [inlined above: _lb1200_base_agent, build_world, simulate_planet_timeline, etc.]",
    wrap_src,
    count=1,
)

# Also strip `from __future__` since it must be at file top (base_src already has it)
wrap_src = re.sub(
    r"^from __future__ import [^\n]+\n",
    "",
    wrap_src,
    count=1,
    flags=re.MULTILINE,
)

# CRITICAL: strip agent_debug function — Kaggle may scan for any callable named
# `agent*` and accidentally pick this one up, which returns a tuple (breaks env).
# Match `def agent_debug(...):` through the next top-level `def` or `__all__`.
wrap_src = re.sub(
    r"# Convenience: a \"debug\".*?(?=\n__all__|\Z)",
    "",
    wrap_src,
    count=1,
    flags=re.DOTALL,
)
# Also simplify __all__ to expose only `agent`
wrap_src = re.sub(
    r'__all__ = \["agent", "agent_debug"\]',
    '__all__ = ["agent"]',
    wrap_src,
    count=1,
)

# The wrapper uses `lb1200_base_agent` — rename its internal references to match
wrap_src = wrap_src.replace("lb1200_base_agent", "_lb1200_base_agent")

# The wrapper also uses symbols imported from lb1200_agent: build_world,
# simulate_planet_timeline, etc. Those are already defined top-level in base_src
# so no further action needed.

# --- Step 3: emit combined file ---
combined = (
    "# Auto-generated Kaggle submission — lb-1200 + shallow lookahead.\n"
    "# Do not edit directly. Regenerate via training/build_submission_lookahead.sh.\n\n"
    + base_src
    + "\n\n# ============================================================\n"
    + "# Lookahead wrapper (from training/lb1200_lookahead_agent.py)\n"
    + "# ============================================================\n\n"
    + wrap_src
)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(combined)
print(f"wrote {out} ({len(combined)/1024:.1f} KB)")
EOF

# --- Step 4: smoke test — can Python import the file and call agent()? ---
"$PY" - <<'EOF'
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "kaggle_submission_main",
    pathlib.Path("submission/lb1200_lookahead/main.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert hasattr(mod, "agent"), "missing `agent` export"
assert callable(mod.agent), "`agent` is not callable"
print(f"smoke test: agent() importable, is callable ✓")
EOF

# --- Step 5: bundle tar.gz ---
tar czf "$OUT/submission.tar.gz" -C "$OUT" main.py
ls -lh "$OUT/submission.tar.gz"
echo
echo "Ready to submit:"
echo "  kaggle competitions submit orbit-wars -f $OUT/submission.tar.gz -m 'lb-1200 + shallow lookahead'"
