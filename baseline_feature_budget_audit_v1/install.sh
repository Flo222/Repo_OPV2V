#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$(pwd)}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)/files/scripts"
mkdir -p "$ROOT/scripts"
cp -f "$SRC_DIR/baseline_feature_budget_common.py" "$ROOT/scripts/"
cp -f "$SRC_DIR/discover_baseline_tx_hooks.py" "$ROOT/scripts/"
cp -f "$SRC_DIR/profile_baseline_feature_budget.py" "$ROOT/scripts/"
chmod +x "$ROOT/scripts/discover_baseline_tx_hooks.py" "$ROOT/scripts/profile_baseline_feature_budget.py"
python -m py_compile \
  "$ROOT/scripts/baseline_feature_budget_common.py" \
  "$ROOT/scripts/discover_baseline_tx_hooks.py" \
  "$ROOT/scripts/profile_baseline_feature_budget.py"
echo "Installed baseline feature budget audit into $ROOT/scripts"
