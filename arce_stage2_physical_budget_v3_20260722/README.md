# ARCE Stage 2: physical execution budget

This patch changes two production files only:

`opencood/comm/arce/arce_c2mab_comm.py`
`opencood/comm/arce/policies/c2mab_execution_record_builder.py`

It keeps proposal `estimated_tx_bytes` in the existing oracle cost path, but
stops using that estimate as the executor hard cap. Runtime execution accepts
only the existing physical per-link budget or the dedicated future field
`oracle_physical_allocation_bytes`.

Ambiguous `allocated_budget_bytes`, proposal budgets, cost estimates, and the
full frame budget are not accepted as per-link fallbacks. Before execution,
the selected links' physical budget sum must not exceed the frame budget.
After execution, actual transmitted bytes are checked against the same frame
budget.

The patch does not change UCB, reward, cache, Fixed, proposal construction,
oracle selection, channel-budget construction, or configuration.

## Apply on the server

Run from the OPV2V repository root after uploading and extracting this package:

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

PKG="$PWD/arce_stage2_physical_budget_v3_20260722"
BACKUP_DIR="refactor_backups/stage2_physical_budget_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp opencood/comm/arce/arce_c2mab_comm.py "$BACKUP_DIR/"
cp opencood/comm/arce/policies/c2mab_execution_record_builder.py "$BACKUP_DIR/"

git apply --check "$PKG/stage2_physical_budget.patch"
git apply "$PKG/stage2_physical_budget.patch"

python -m py_compile \
  opencood/comm/arce/arce_c2mab_comm.py \
  opencood/comm/arce/policies/c2mab_execution_record_builder.py
REPO_DIR="$PWD" python "$PKG/tests/test_stage2_physical_budget.py"
PYTHONPATH="$PWD:$PYTHONPATH" python "$PKG/tests/test_stage2_executor_coverage.py"
```

The first test runs nine field-source, finite-value, and frame-budget checks. Expected unit
coverage under the same 2048-byte physical budget and `C=64`:

- FP16: 16 complete spatial units
- INT8: 32 complete spatial units
- INT4: 64 complete spatial units

## Run the 100-frame integration audit

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
METHOD=arce RUN_AP=1 RUN_BW=1 MAX_FRAMES=100 \
TAG=stage2_physical_budget_ap_bw100 \
bash scripts/run_arce_pair_eval.sh

python "$PKG/audit_stage2_runtime.py" \
  outputs/arce_c2mab/stage2_physical_budget_ap_bw100/runtime_records.jsonl
```

Required audit conditions:

- `estimated_cost_used_as_execution_budget_count == 0`
- `not_decoupled_count == 0`
- `unexpected_quant_dependent_budget_source_count == 0`
- `link_over_physical_budget_count == 0`
- `frame_over_system_budget_count == 0` when frame caps are present in records
- Proposal estimates differ by quantization, while physical execution budgets
  come from `link_budget_bytes` under the current oracle.

`physical_budget_source_expected_quant_independent` is only a source-level
expectation. Numerical equality across quantization modes must still be checked
with the forced-action counterfactual test; the audit does not label it as a
proven property.

Do not start a full run until these checks pass.
