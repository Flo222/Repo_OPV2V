#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
REPO_DIR="${REPO_DIR:-$PWD}"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
FILES_DIR="$PKG_DIR/files"

case "$MODE" in
  --check|--apply)
    ;;
  *)
    echo "Usage: REPO_DIR=/path/to/OPV2V bash install.sh [--check|--apply]" >&2
    exit 2
    ;;
esac

required_repo=(
  "opencood/comm/arce/policies/ap_proxy_features.py"
  "opencood/tools/audit_arce_counterfactual.py"
  "opencood/tools/merge_counterfactual_proxy_datasets.py"
)
for rel in "${required_repo[@]}"; do
  if [[ ! -f "$REPO_DIR/$rel" ]]; then
    echo "MISSING  $rel" >&2
    exit 1
  fi
done

if ! grep -q "PAIRED_SPATIAL_AP_PROXY_FEATURES" \
  "$REPO_DIR/opencood/comm/arce/policies/ap_proxy_features.py"
then
  echo "ERROR: AP-proxy v3 feature definitions are not installed." >&2
  exit 1
fi

expected_collector_sha256="50ae37f0ffd567c39094f0f55cae96adf616c7325adc849dc337bdf2a2a56e4d"
installed_collector_sha256="28793e3eefd19f6b0b4b8ef23661dda234cae899fa910cb7ea8dde1379cb92c9"
actual_collector_sha256="$(
  sha256sum "$REPO_DIR/opencood/tools/audit_arce_counterfactual.py" \
    | awk '{print $1}'
)"
if [[ "$actual_collector_sha256" != "$expected_collector_sha256" \
  && "$actual_collector_sha256" != "$installed_collector_sha256" ]]; then
  echo "ERROR: audit_arce_counterfactual.py differs from the reviewed v3 source." >&2
  echo "Expected original:  $expected_collector_sha256" >&2
  echo "Expected installed: $installed_collector_sha256" >&2
  echo "Actual:   $actual_collector_sha256" >&2
  echo "No files were changed." >&2
  exit 1
fi

package_files=(
  "opencood/comm/arce/policies/decoded_box_proxy_features.py"
  "opencood/tools/audit_arce_counterfactual.py"
  "opencood/tools/merge_counterfactual_proxy_datasets_v33.py"
  "opencood/tools/audit_ap_proxy_v33_rich_box_groupcv.py"
)
for rel in "${package_files[@]}"; do
  if [[ ! -f "$FILES_DIR/$rel" ]]; then
    echo "MISSING PACKAGE FILE  $rel" >&2
    exit 1
  fi
  python -m py_compile "$FILES_DIR/$rel"
  echo "READY    $rel"
done

if [[ "$MODE" == "--check" ]]; then
  echo "AP-proxy v3.3 preflight passed; no files were changed."
  exit 0
fi

backup="$REPO_DIR/refactor_backups/ap_proxy_v33_rich_box_$(date +%Y%m%d_%H%M%S)"
for rel in "${package_files[@]}"; do
  target="$REPO_DIR/$rel"
  if [[ -f "$target" ]]; then
    mkdir -p "$backup/$(dirname "$rel")"
    cp "$target" "$backup/$rel"
  fi
done

for rel in "${package_files[@]}"; do
  target="$REPO_DIR/$rel"
  mkdir -p "$(dirname "$target")"
  cp "$FILES_DIR/$rel" "$target"
done

cd "$REPO_DIR"
python -m py_compile \
  opencood/comm/arce/policies/decoded_box_proxy_features.py \
  opencood/tools/audit_arce_counterfactual.py \
  opencood/tools/merge_counterfactual_proxy_datasets_v33.py \
  opencood/tools/audit_ap_proxy_v33_rich_box_groupcv.py

PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
  python "$PKG_DIR/tests/test_decoded_box_proxy_features_v33.py"
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
  python "$PKG_DIR/tests/test_ap_proxy_v33_rich_box_groupcv.py"

echo "AP-proxy v3.3 rich decoded-box offline audit installed."
echo "Backup: $backup"
