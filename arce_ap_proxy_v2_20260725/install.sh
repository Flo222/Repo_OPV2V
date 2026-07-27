#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
REPO_DIR="${REPO_DIR:-$PWD}"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH="$PKG_DIR/arce_ap_proxy_v2.patch"

case "$MODE" in
  --check|--apply) ;;
  *)
    echo "usage: REPO_DIR=/path/to/OPV2V bash install.sh [--check|--apply]" >&2
    exit 2
    ;;
esac

cd "$REPO_DIR"

required=(
  opencood/models/point_pillar_where2comm_arce.py
  opencood/tools/audit_arce_counterfactual.py
  opencood/tools/collect_delta_ap_proxy_dataset.py
)

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "MISSING  $path" >&2
    exit 1
  fi
  echo "READY    $path"
done

git apply --check "$PATCH"

if [[ "$MODE" == "--check" ]]; then
  echo "AP-proxy v2 preflight passed; no files were changed."
  exit 0
fi

backup="$REPO_DIR/refactor_backups/ap_proxy_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup/opencood/models" "$backup/opencood/tools"
cp opencood/models/point_pillar_where2comm_arce.py \
  "$backup/opencood/models/"
cp opencood/tools/audit_arce_counterfactual.py \
  "$backup/opencood/tools/"
cp opencood/tools/collect_delta_ap_proxy_dataset.py \
  "$backup/opencood/tools/"

git apply "$PATCH"

python -m py_compile \
  opencood/comm/arce/policies/ap_proxy_features.py \
  opencood/models/point_pillar_where2comm_arce.py \
  opencood/tools/audit_arce_counterfactual.py \
  opencood/tools/collect_delta_ap_proxy_dataset.py \
  opencood/tools/train_counterfactual_ap_proxies.py

PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
  python "$PKG_DIR/tests/test_ap_proxy_features.py"
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
  python "$PKG_DIR/tests/test_counterfactual_proxy_training.py"

echo "AP-proxy v2 installed."
echo "Backup: $backup"
