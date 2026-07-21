#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
if [[ "$MODE" != "--check" && "$MODE" != "--apply" ]]; then
  echo "usage: REPO_DIR=/path/to/OPV2V bash install.sh [--check|--apply]" >&2
  exit 2
fi

PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="${REPO_DIR:-$(pwd)}"
MANIFEST="$PKG_DIR/manifest.tsv"

if [[ ! -f "$REPO_DIR/opencood/comm/arce/arce_fixed_comm.py" ]]; then
  echo "not an OPV2V repository: $REPO_DIR" >&2
  exit 2
fi

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

status=0
while IFS=$'\t' read -r old_hash new_hash rel; do
  [[ -n "$rel" ]] || continue
  target="$REPO_DIR/$rel"
  source="$PKG_DIR/files/$rel"

  if [[ ! -f "$target" ]]; then
    echo "MISSING  $rel"
    status=1
    continue
  fi
  if [[ ! -f "$source" ]]; then
    echo "PACKAGE_MISSING  $rel"
    status=1
    continue
  fi

  current_hash="$(sha256_file "$target")"
  packaged_hash="$(sha256_file "$source")"
  if [[ "$packaged_hash" != "$new_hash" ]]; then
    echo "PACKAGE_HASH_ERROR  $rel"
    status=1
  elif [[ "$current_hash" == "$old_hash" ]]; then
    echo "READY    $rel"
  elif [[ "$current_hash" == "$new_hash" ]]; then
    echo "INSTALLED $rel"
  else
    echo "CONFLICT $rel"
    echo "  current: $current_hash"
    echo "  expected original: $old_hash"
    echo "  expected stage1:   $new_hash"
    status=1
  fi
done < "$MANIFEST"

if [[ "$status" -ne 0 ]]; then
  echo "Stage 1 preflight failed; no files were changed." >&2
  exit 1
fi

if [[ "$MODE" == "--check" ]]; then
  echo "Stage 1 preflight passed; no files were changed."
  exit 0
fi

backup_dir="$REPO_DIR/refactor_backups/stage1_priority_layout_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

while IFS=$'\t' read -r old_hash new_hash rel; do
  [[ -n "$rel" ]] || continue
  target="$REPO_DIR/$rel"
  source="$PKG_DIR/files/$rel"
  backup="$backup_dir/$rel"
  mkdir -p "$(dirname "$backup")"
  cp "$target" "$backup"
  cp "$source" "$target"
done < "$MANIFEST"

python -m py_compile \
  "$REPO_DIR/opencood/models/fuse_modules/where2comm_arce_fuse.py" \
  "$REPO_DIR/opencood/comm/arce/arce_c2mab_comm.py" \
  "$REPO_DIR/opencood/comm/arce/arce_fixed_comm.py"

echo "Stage 1 installed."
echo "Backup: $backup_dir"
