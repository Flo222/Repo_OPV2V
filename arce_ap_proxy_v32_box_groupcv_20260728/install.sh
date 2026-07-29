#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
REPO_DIR="${REPO_DIR:-$PWD}"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$PKG_DIR/files/opencood/tools/audit_ap_proxy_v32_box_groupcv.py"
TARGET="$REPO_DIR/opencood/tools/audit_ap_proxy_v32_box_groupcv.py"

case "$MODE" in
  --check|--apply)
    ;;
  *)
    echo "Usage: REPO_DIR=/path/to/OPV2V bash install.sh [--check|--apply]" >&2
    exit 2
    ;;
esac

for required in \
  "$SOURCE" \
  "$REPO_DIR/opencood/comm/arce/policies/ap_proxy_features.py" \
  "$REPO_DIR/opencood/tools/train_counterfactual_ap_proxies.py"
do
  if [ ! -f "$required" ]; then
    echo "MISSING  $required" >&2
    exit 1
  fi
done

if ! grep -q "PAIRED_SPATIAL_AP_PROXY_FEATURES" \
  "$REPO_DIR/opencood/comm/arce/policies/ap_proxy_features.py"
then
  echo "ERROR: AP-proxy v3 feature definitions are not installed." >&2
  exit 1
fi

python -m py_compile "$SOURCE"
echo "READY    opencood/tools/audit_ap_proxy_v32_box_groupcv.py"

if [ "$MODE" = "--check" ]; then
  echo "AP-proxy v3.2 box-summary preflight passed; no files were changed."
  exit 0
fi

BACKUP_DIR="$REPO_DIR/refactor_backups/ap_proxy_v32_box_groupcv_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
if [ -f "$TARGET" ]; then
  cp "$TARGET" "$BACKUP_DIR/audit_ap_proxy_v32_box_groupcv.py"
fi

cp "$SOURCE" "$TARGET"
python -m py_compile "$TARGET"
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
  python "$PKG_DIR/tests/test_ap_proxy_v32_box_groupcv.py"

echo "AP-proxy v3.2 decoded-box GroupKFold audit tool installed."
echo "Backup: $BACKUP_DIR"
