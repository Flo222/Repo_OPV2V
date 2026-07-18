# Compact-sparse source/parity budget-accounting fix

This is an incremental patch installed after the Experiment 2 and Experiment 2.5 audit patches.

## Problem fixed

In the compact-sparse / method-native payload path, the communication result still obeyed the finite budget, but the compression auditor could receive incomplete accounting fields through `comm_record["size"]`. This produced `null` budget values, zero transmitted-source counts, and `pass=false` despite valid inference.

## Fix

`ARCEFixedComm.communicate_feature()` now passes the already computed runtime values directly to `CompressionAuditor.record()`:

- frame/link bandwidth budget
- source/parity/encoded packet counts
- transmitted source/parity counts
- source/parity packets removed by budget
- transmitted source/parity bytes
- Bernoulli loss and FEC recovery outcomes

The auditor treats these runtime locals as authoritative and checks:

- `tx_source + budget_drop_source == source`
- `tx_parity + budget_drop_parity == parity`
- `budget_drop_source + budget_drop_parity == total_budget_drop`
- `tx_source == source_tx_mask.sum()`

No packet is added, removed, reordered, or changed by this patch.

## Install

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

unzip compact_sparse_budget_accounting_fix.zip
bash compact_sparse_budget_accounting_fix/install.sh \
  /home/server/v2x_projects/OPV2V
```

Expected final test line:

```text
Experiment 2 compact-sparse budget-accounting smoke test passed.
```

## Re-run 20-frame Experiment 2 check

Use a new output directory:

```bash
export MODEL_DIR=/home/server/v2x_projects/OPV2V/opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/correct_model_experiment2_budget100_fixed_test20
export MAX_SAMPLES=20
export NUM_WORKERS=0
export SAVE_TENSORS=1
export SAVE_FIRST_N_LINKS=12
export SYSTEM_BUDGET_MBPS=100
export TX_WINDOW_MS=100
export QUANT_MODES="fp32 fp16 int8 int4"

bash scripts/run_experiment2_compression_budget_audit.sh
```

Expected summary properties:

- `mean_bandwidth_budget_bytes` is finite, approximately 1,250,000 bytes per link/frame when one collaborator communicates.
- `budget_accounting_valid_ratio = 1.0`
- `sanity_pass_ratio = 1.0`
- `mean_tx_source_packets > 0`
- source transmission ratios approximately increase from FP32 to INT4
- parity counts remain zero because `rho=0`
