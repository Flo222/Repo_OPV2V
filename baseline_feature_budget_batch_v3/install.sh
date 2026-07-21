#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-/home/server/v2x_projects/OPV2V}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/scripts"
cp -f "$HERE/files/scripts/baseline_feature_budget_common.py" "$ROOT/scripts/"
cp -f "$HERE/files/scripts/discover_baseline_tx_hooks.py" "$ROOT/scripts/"
cp -f "$HERE/files/scripts/profile_baseline_feature_budget.py" "$ROOT/scripts/"
chmod +x "$ROOT/scripts/discover_baseline_tx_hooks.py" "$ROOT/scripts/profile_baseline_feature_budget.py"
python -m py_compile \
  "$ROOT/scripts/baseline_feature_budget_common.py" \
  "$ROOT/scripts/discover_baseline_tx_hooks.py" \
  "$ROOT/scripts/profile_baseline_feature_budget.py"
echo "Installed baseline feature-budget audit v3 into $ROOT/scripts"
