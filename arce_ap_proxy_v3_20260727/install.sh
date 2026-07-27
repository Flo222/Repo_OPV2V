#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
REPO_DIR="${REPO_DIR:-$PWD}"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH="$PKG_DIR/arce_ap_proxy_v3.patch"

case "$MODE" in
  --check|--apply) ;;
  *)
    echo "Usage: REPO_DIR=/path/to/OPV2V bash install.sh [--check|--apply]" >&2
    exit 2
    ;;
esac

required=(
  "opencood/comm/arce/policies/ap_proxy_features.py"
  "opencood/models/point_pillar_where2comm_arce.py"
  "opencood/tools/audit_arce_counterfactual.py"
  "opencood/tools/train_counterfactual_ap_proxies.py"
)

for rel in "${required[@]}"; do
  if [[ ! -f "$REPO_DIR/$rel" ]]; then
    echo "MISSING  $rel" >&2
    exit 1
  fi
done

cd "$REPO_DIR"
git apply --check "$PATCH"
echo "AP-proxy v3 preflight passed."

if [[ "$MODE" == "--check" ]]; then
  echo "No files were changed."
  exit 0
fi

backup="$REPO_DIR/refactor_backups/ap_proxy_v3_$(date +%Y%m%d_%H%M%S)"
changed=(
  "opencood/comm/arce/policies/ap_proxy_features.py"
  "opencood/models/point_pillar_where2comm_arce.py"
  "opencood/tools/audit_arce_counterfactual.py"
  "opencood/tools/train_counterfactual_ap_proxies.py"
  "opencood/tools/merge_counterfactual_proxy_datasets.py"
)
for rel in "${changed[@]}"; do
  if [[ -f "$REPO_DIR/$rel" ]]; then
    mkdir -p "$backup/$(dirname "$rel")"
    cp "$REPO_DIR/$rel" "$backup/$rel"
  fi
done

git apply "$PATCH"

python -m py_compile \
  opencood/comm/arce/policies/ap_proxy_features.py \
  opencood/models/point_pillar_where2comm_arce.py \
  opencood/tools/audit_arce_counterfactual.py \
  opencood/tools/train_counterfactual_ap_proxies.py \
  opencood/tools/merge_counterfactual_proxy_datasets.py

PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
  python "$PKG_DIR/tests/test_ap_proxy_features_v3.py"
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
  python "$PKG_DIR/tests/test_proxy_v3_split_metrics.py"
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
  python "$PKG_DIR/tests/test_proxy_v3_end_to_end.py"

echo "AP-proxy v3 installed."
echo "Backup: $backup"
