#!/bin/bash
# One-shot bootstrap for the Orbit Wars kit:
#   - installs playwright + chromium
#   - installs system libs (nspr, nss, alsa) into the current (conda) env
#     so Playwright's chromium can launch on hosts without sudo
#   - installs the kaggle CLI
#
# Run inside the Python environment you want the pipeline to use.

set -eu

PY="$(command -v python3)"
PREFIX="$("$PY" -c 'import sys;print(sys.prefix)')"

echo "python:  $PY"
echo "prefix:  $PREFIX"

# Python deps
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install playwright kaggle

# Chromium binary
"$PY" -m playwright install chromium

# System libs for headless chromium — no sudo required if using conda
if command -v conda >/dev/null 2>&1; then
    echo "installing chromium system libs via conda-forge"
    conda install -y -c conda-forge nspr nss alsa-lib
else
    echo ""
    echo "WARNING: conda not found. If chromium fails to launch with"
    echo "  'libnspr4.so: cannot open shared object file', install the"
    echo "  system packages yourself (Debian/Ubuntu):"
    echo "    sudo apt-get install -y libnspr4 libnss3 libasound2"
fi

# Kaggle credentials reminder
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo ""
    echo "NOTE: ~/.kaggle/kaggle.json not found."
    echo "  1. Visit https://www.kaggle.com/settings/account → API → Create New Token"
    echo "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  3. chmod 600 ~/.kaggle/kaggle.json"
    echo "  4. Join the competition at https://www.kaggle.com/competitions/orbit-wars/rules"
fi

echo ""
echo "setup done. next:"
echo "  python3 pipeline.py        # one-shot scrape + parse"
echo "  ./install_cron.sh          # schedule every 30 min for 7 days"
