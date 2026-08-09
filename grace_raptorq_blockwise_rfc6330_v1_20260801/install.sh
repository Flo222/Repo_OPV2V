#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---check}"
REPO_DIR="${REPO_DIR:-$PWD}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH="$SCRIPT_DIR/grace_raptorq_blockwise_rfc6330.patch"

case "$MODE" in
  --check|--apply) ;;
  *) echo "Usage: REPO_DIR=/path/to/OPV2V bash install.sh [--check|--apply]" >&2; exit 2 ;;
esac

cd "$REPO_DIR"
git apply --check "$PATCH"

if [ "$MODE" = "--check" ]; then
  echo "RaptorQ patch preflight passed; no files were changed."
  exit 0
fi

python - <<'PY'
try:
    import raptorq
except ImportError as exc:
    raise SystemExit(
        "Missing raptorq==1.6.3. Install it first with: "
        "python -m pip install --only-binary=:all: raptorq==1.6.3"
    ) from exc
PY

BACKUP_DIR="refactor_backups/raptorq_blockwise_rfc6330_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

FILES=(
  opencood/comm/arce/arce_c2mab_comm.py
  opencood/comm/arce/arce_fixed_comm.py
  opencood/comm/arce/audit/compression_auditor.py
  opencood/comm/arce/audit/fec_recovery_auditor.py
  opencood/comm/arce/policies/action_space.py
  opencood/comm/arce/policies/communication_cost_estimator.py
  opencood/comm/fec/__init__.py
  opencood/logs/main_opv2v_where2comm_grace_full/config.yaml
)

for file in "${FILES[@]}"; do
  mkdir -p "$BACKUP_DIR/$(dirname "$file")"
  cp "$file" "$BACKUP_DIR/$file"
done

git apply "$PATCH"

python -m py_compile \
  opencood/comm/fec/fec_raptorq.py \
  opencood/comm/arce/priority_block_fec_transport.py \
  opencood/comm/arce/policies/action_space.py \
  opencood/comm/arce/policies/communication_cost_estimator.py \
  opencood/comm/arce/audit/fec_recovery_auditor.py \
  opencood/comm/arce/audit/compression_auditor.py \
  opencood/comm/arce/arce_fixed_comm.py \
  opencood/comm/arce/arce_c2mab_comm.py \
  scripts/preflight_grace_raptorq.py \
  scripts/test_grace_raptorq_planner_unit.py \
  scripts/test_grace_raptorq_transport.py

python scripts/test_grace_raptorq_planner_unit.py
python scripts/preflight_grace_raptorq.py
python scripts/test_grace_raptorq_transport.py

echo "RaptorQ blockwise transport installed and verified."
echo "Backup: $BACKUP_DIR"
