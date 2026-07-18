#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$(pwd)}"
TARGET="$(realpath "$TARGET")"

for required in \
  opencood/tools/inference_arce.py \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/audit/compression_auditor.py \
  scripts/test_experiment1_compression_audit_unit.py; do
  if [ ! -f "$TARGET/$required" ]; then
    echo "Missing prerequisite file: $TARGET/$required" >&2
    echo "Install the Experiment-1 compression audit patch first." >&2
    exit 1
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$TARGET/backup_experiment2_compression_budget_audit_$STAMP"
mkdir -p \
  "$BACKUP/opencood/tools" \
  "$BACKUP/opencood/comm/arce/audit" \
  "$BACKUP/opencood/comm/arce" \
  "$BACKUP/scripts"

cp "$TARGET/opencood/tools/inference_arce.py" "$BACKUP/opencood/tools/inference_arce.py"
cp "$TARGET/opencood/comm/arce/arce_fixed_comm.py" "$BACKUP/opencood/comm/arce/arce_fixed_comm.py"
cp "$TARGET/opencood/comm/arce/audit/compression_auditor.py" \
   "$BACKUP/opencood/comm/arce/audit/compression_auditor.py"

for f in \
  scripts/run_experiment2_compression_budget_audit.sh \
  scripts/summarize_experiment2_compression_budget_audit.py \
  scripts/test_experiment2_budget_accounting_unit.py; do
  if [ -f "$TARGET/$f" ]; then
    cp "$TARGET/$f" "$BACKUP/$f"
  fi
done

if grep -q "attach_frame_identity_to_batch" "$TARGET/opencood/tools/inference_arce.py" \
  && grep -q "num_source_dropped_by_budget" "$TARGET/opencood/comm/arce/arce_fixed_comm.py" \
  && grep -q "require_no_budget_drop" "$TARGET/opencood/comm/arce/audit/compression_auditor.py"; then
  echo "Existing Experiment-2 core changes detected; skipping patch re-application."
else
  if ! patch --dry-run --batch --forward -p1 -d "$TARGET" < "$PKG_DIR/existing_files.patch" >/dev/null; then
    echo "Patch does not match the current repository state." >&2
    echo "No core file was changed. Backup is at: $BACKUP" >&2
    echo "The patch expects the Experiment-1 audit version of the three core files." >&2
    exit 1
  fi
  patch --batch --forward -p1 -d "$TARGET" < "$PKG_DIR/existing_files.patch"
fi

mkdir -p "$TARGET/scripts"
cp "$PKG_DIR/files/scripts/"* "$TARGET/scripts/"
chmod +x \
  "$TARGET/scripts/run_experiment2_compression_budget_audit.sh" \
  "$TARGET/scripts/summarize_experiment2_compression_budget_audit.py" \
  "$TARGET/scripts/test_experiment2_budget_accounting_unit.py"

cd "$TARGET"
python -m py_compile \
  opencood/tools/inference_arce.py \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/audit/compression_auditor.py \
  scripts/summarize_experiment2_compression_budget_audit.py \
  scripts/test_experiment2_budget_accounting_unit.py

# Both tests are dataset/checkpoint independent.
python scripts/test_experiment1_compression_audit_unit.py
python scripts/test_experiment2_budget_accounting_unit.py

cat > "$TARGET/.experiment2_compression_budget_audit_install.txt" <<EOF
installed_at=$STAMP
backup_dir=$BACKUP
EOF

echo
echo "Installed Experiment 2 compression-budget audit."
echo "Backup: $BACKUP"
echo "Repository: $TARGET"
