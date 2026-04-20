#!/bin/bash
# Build a raw lb-1200 submission (no wrapper, no modifications).
set -u
ROOT=/home/lab/orbit-war
OUT=$ROOT/submission/raw_lb1200
mkdir -p "$OUT"
PY=/home/lab/miniconda3/envs/tom/bin/python
cd "$ROOT"

# Just copy the file verbatim — lb1200_agent.py already exports `agent`.
cp training/lb1200_agent.py "$OUT/main.py"

"$PY" - <<'EOF'
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("ks", pathlib.Path("submission/raw_lb1200/main.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert callable(m.agent), "agent not callable"
print("smoke test: agent() importable ✓")
EOF

tar czf "$OUT/submission.tar.gz" -C "$OUT" main.py
ls -lh "$OUT/submission.tar.gz"

# Notebook version
"$PY" - <<'EOF'
import json, pathlib
main_src = pathlib.Path("submission/raw_lb1200/main.py").read_text()
nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {},
         "source": ["# Raw lb-1200 — Orbit Wars Submission\n\n"
                    "Baseline submission: the public lb-1200 rule-based agent, unmodified.\n\n"
                    "Submits via `%%writefile submission.py`. Save & Run All → Submit to Competition."]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": ["%%writefile submission.py\n" + main_src]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": ["import importlib.util\n"
                    "spec = importlib.util.spec_from_file_location('sub','submission.py')\n"
                    "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                    "assert callable(m.agent)\n"
                    "print('agent() ready')"]},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12"}},
    "nbformat": 4, "nbformat_minor": 4,
}
out = pathlib.Path("notebooks/raw-lb1200-submission.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"notebook: {out} ({out.stat().st_size/1024:.1f} KB)")
EOF

echo
echo "Submit with:"
echo "  [tar.gz]    kaggle competitions submit orbit-wars -f $OUT/submission.tar.gz -m 'raw lb-1200 baseline'"
echo "  [Notebook]  notebooks/raw-lb1200-submission.ipynb"
