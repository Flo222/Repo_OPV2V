#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
if [ "$MODE" != "--check" ] && [ "$MODE" != "--apply" ]; then
  echo "Usage: REPO_DIR=/path/to/OPV2V bash install.sh --check|--apply" >&2
  exit 2
fi

PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$(pwd)}"

targets=(
  "opencood/comm/arce/arce_fixed_comm.py"
  "opencood/comm/arce/arce_c2mab_comm.py"
  "opencood/models/fuse_modules/where2comm_arce_fuse.py"
  "opencood/logs/main_opv2v_where2comm_grace_full/config.yaml"
  "opencood/comm/arce/policies/spatial_importance.py"
)

expected=(
  "bc67bbd60268b2b458a8125346cdcd33e75ddd018c867e5f54dc484a778e0734"
  "3f8ad4b86caa5654178625dcd49e93aeb075d01a45f44b8b782998b566cbb397"
  "316371e7fde16420c257bdb3ae33fb41cd6ac98fbb554daef18f8a581095f2d6"
  "a34bb1b8ace72d2827058effb436ae66bd3a40d0cbbf62c081b7786c2c7ced37"
  "ABSENT"
)

failed=0
for i in "${!targets[@]}"; do
  rel="${targets[$i]}"
  dst="$REPO_DIR/$rel"
  src="$PKG_DIR/files/$rel"
  want="${expected[$i]}"

  if [ ! -f "$src" ]; then
    echo "PACKAGE_MISSING  $rel"
    failed=1
    continue
  fi

  replacement_hash="$(sha256sum "$src" | awk '{print $1}')"
  if [ -f "$dst" ]; then
    current_hash="$(sha256sum "$dst" | awk '{print $1}')"
    if [ "$current_hash" = "$replacement_hash" ]; then
      echo "INSTALLED  $rel"
    elif [ "$want" != "ABSENT" ] && [ "$current_hash" = "$want" ]; then
      echo "READY      $rel"
    else
      echo "MISMATCH   $rel"
      echo "  current:    $current_hash"
      echo "  expected:   $want"
      echo "  replacement:$replacement_hash"
      failed=1
    fi
  elif [ "$want" = "ABSENT" ]; then
    echo "NEW        $rel"
  else
    echo "MISSING    $rel"
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "Stage 3A preflight failed; no files were changed." >&2
  exit 1
fi

if [ "$MODE" = "--check" ]; then
  echo "Stage 3A preflight passed; no files were changed."
  exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$REPO_DIR/refactor_backups/stage3a_spatial_importance_$STAMP"
mkdir -p "$BACKUP_DIR"

for rel in "${targets[@]}"; do
  dst="$REPO_DIR/$rel"
  src="$PKG_DIR/files/$rel"
  if [ -f "$dst" ]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$rel")"
    cp "$dst" "$BACKUP_DIR/$rel"
  fi
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
done

cd "$REPO_DIR"
python -m py_compile \
  opencood/comm/arce/policies/spatial_importance.py \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/arce_c2mab_comm.py \
  opencood/models/fuse_modules/where2comm_arce_fuse.py

python "$PKG_DIR/tests/test_spatial_importance.py"

echo "Stage 3A installed."
echo "Backup: $BACKUP_DIR"
echo "Run the server integration test next:"
echo "  python $PKG_DIR/tests/test_stage3a_runtime_integration.py"
