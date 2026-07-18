#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$(pwd)}"
TARGET="$(realpath "$TARGET")"

if [ ! -f "$TARGET/opencood/comm/arce/arce_fixed_comm.py" ]; then
  echo "Target does not look like Repo_OPV2V: $TARGET" >&2
  echo "Usage: bash install.sh /path/to/Repo_OPV2V" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$TARGET/backup_experiment1_compression_audit_$STAMP"
mkdir -p "$BACKUP/opencood/comm/arce" "$BACKUP/scripts"
cp "$TARGET/opencood/comm/arce/arce_fixed_comm.py" "$BACKUP/opencood/comm/arce/arce_fixed_comm.py"

for f in \
  scripts/run_experiment1_compression_audit.sh \
  scripts/summarize_experiment1_compression_audit.py \
  scripts/test_experiment1_compression_audit_unit.py; do
  if [ -f "$TARGET/$f" ]; then
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp "$TARGET/$f" "$BACKUP/$f"
  fi
done

mkdir -p "$TARGET/opencood/comm/arce/audit" "$TARGET/scripts"
cp "$PKG_DIR/files/opencood/comm/arce/arce_fixed_comm.py" "$TARGET/opencood/comm/arce/arce_fixed_comm.py"
cp "$PKG_DIR/files/opencood/comm/arce/audit/__init__.py" "$TARGET/opencood/comm/arce/audit/__init__.py"
cp "$PKG_DIR/files/opencood/comm/arce/audit/compression_auditor.py" "$TARGET/opencood/comm/arce/audit/compression_auditor.py"
cp "$PKG_DIR/files/scripts/"* "$TARGET/scripts/"
chmod +x \
  "$TARGET/scripts/run_experiment1_compression_audit.sh" \
  "$TARGET/scripts/summarize_experiment1_compression_audit.py" \
  "$TARGET/scripts/test_experiment1_compression_audit_unit.py"

cd "$TARGET"
python -m py_compile \
  opencood/comm/arce/audit/__init__.py \
  opencood/comm/arce/audit/compression_auditor.py \
  opencood/comm/arce/arce_fixed_comm.py \
  scripts/summarize_experiment1_compression_audit.py \
  scripts/test_experiment1_compression_audit_unit.py
python scripts/test_experiment1_compression_audit_unit.py

cat > "$TARGET/.experiment1_compression_audit_install.txt" <<EOF
installed_at=$STAMP
backup_dir=$BACKUP
EOF

echo
echo "Installed Experiment 1 compression audit."
echo "Backup: $BACKUP"
echo "Repository: $TARGET"
