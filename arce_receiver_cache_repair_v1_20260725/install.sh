#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
REPO_DIR="${REPO_DIR:-$PWD}"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"

FILES=(
  "opencood/comm/arce/arce_fixed_comm.py"
  "opencood/logs/main_opv2v_where2comm_grace_full/config.yaml"
)

ORIGINAL_SHA256=(
  "6a0472c456558a99d10f10fb370ac01468b58e9bdea5809aed5a2b10a9d4b15e"
  "a3111c6fc488655e7d2524cb311517516d8300489c9cd1750dd3dd343dafb613"
)

PATCHED_SHA256=(
  "d6af472cc5ab767ccecd98f46a7e9586d1e2dbc8ff10b3b9dcde56a7776e43e0"
  "a34bb1b8ace72d2827058effb436ae66bd3a40d0cbbf62c081b7786c2c7ced37"
)

if [[ "$MODE" != "--check" && "$MODE" != "--apply" ]]; then
  echo "Usage: REPO_DIR=/path/to/OPV2V bash install.sh [--check|--apply]" >&2
  exit 2
fi

status=()
for i in "${!FILES[@]}"; do
  rel="${FILES[$i]}"
  target="$REPO_DIR/$rel"
  source="$PKG_DIR/$rel"

  if [[ ! -f "$target" ]]; then
    echo "MISSING  $rel"
    status+=("failed")
    continue
  fi
  if [[ ! -f "$source" ]]; then
    echo "PACKAGE_MISSING  $rel"
    status+=("failed")
    continue
  fi

  current="$(sha256sum "$target" | awk '{print $1}')"
  replacement="$(sha256sum "$source" | awk '{print $1}')"
  if [[ "$replacement" != "${PATCHED_SHA256[$i]}" ]]; then
    echo "PACKAGE_HASH_MISMATCH  $rel"
    status+=("failed")
  elif [[ "$current" == "${ORIGINAL_SHA256[$i]}" ]]; then
    echo "READY    $rel"
    status+=("ready")
  elif [[ "$current" == "${PATCHED_SHA256[$i]}" ]]; then
    echo "ALREADY  $rel"
    status+=("already")
  else
    echo "CHANGED  $rel"
    echo "  expected original: ${ORIGINAL_SHA256[$i]}"
    echo "  current:           $current"
    status+=("failed")
  fi
done

if printf '%s\n' "${status[@]}" | grep -q '^failed$'; then
  echo "Preflight failed; no files were changed." >&2
  exit 1
fi

python -m py_compile "$PKG_DIR/opencood/comm/arce/arce_fixed_comm.py"

if [[ "$MODE" == "--check" ]]; then
  echo "Cache repair preflight passed; no files were changed."
  exit 0
fi

backup_dir="$REPO_DIR/refactor_backups/receiver_cache_repair_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for i in "${!FILES[@]}"; do
  rel="${FILES[$i]}"
  target="$REPO_DIR/$rel"
  source="$PKG_DIR/$rel"
  current="$(sha256sum "$target" | awk '{print $1}')"

  if [[ "$current" == "${PATCHED_SHA256[$i]}" ]]; then
    continue
  fi

  mkdir -p "$backup_dir/$(dirname "$rel")"
  cp "$target" "$backup_dir/$rel"
  cp "$source" "$target"
done

python -m py_compile "$REPO_DIR/opencood/comm/arce/arce_fixed_comm.py"
PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" \
  python "$PKG_DIR/tests/test_receiver_temporal_cache.py"

echo "Receiver temporal cache repair installed."
echo "Backup: $backup_dir"
