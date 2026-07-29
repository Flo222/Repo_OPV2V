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
  "opencood/comm/arce/zero_sparse_codec.py"
  "opencood/comm/arce/audit/compression_auditor.py"
  "opencood/logs/main_opv2v_where2comm_grace_full/config.yaml"
)

expected=(
  "76e6e9be283a3a90609c9f67359f7772a74136de6670c9911890daa935dd40f3"
  "ABSENT"
  "e32e7e7ec708ad596f33a012d9b94664d8c47f9fda0a2b43a1de2ac115492c2e"
  "4bb4fe79e31326a9b3642f76f6c73cb5078c09df4883039cad257ddd370526f3"
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
      echo "  current:     $current_hash"
      echo "  expected:    $want"
      echo "  replacement: $replacement_hash"
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
  echo "Stage 3B preflight failed; no files were changed." >&2
  exit 1
fi

if [ "$MODE" = "--check" ]; then
  echo "Stage 3B preflight passed; no files were changed."
  exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$REPO_DIR/refactor_backups/stage3b_zero_codec_$STAMP"
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
  opencood/comm/arce/zero_sparse_codec.py \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/audit/compression_auditor.py

python "$PKG_DIR/tests/test_zero_sparse_codec.py"

echo "Stage 3B installed."
echo "Backup: $BACKUP_DIR"
echo "Run the server integration test next:"
echo "  python $PKG_DIR/tests/test_stage3b_runtime_integration.py"
