#!/bin/bash
# Remove the Orbit Wars cron entry and its metadata.

set -eu
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
MARKER="cron_wrapper.sh"

( crontab -l 2>/dev/null | grep -v -F "$MARKER" ) | crontab -
rm -f "$ROOT/.cron_expiry" "$ROOT/.cron_python" "$ROOT/.cron.lock"

echo "uninstalled. remaining crontab:"
crontab -l 2>/dev/null || echo "  (empty)"
