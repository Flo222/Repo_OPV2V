# Experiment 2.5 — Budget Retention Layout Audit

This patch is installed **after Experiment 2**. It does not change quantization,
packetization, budget selection, packet loss, recovery, fusion, or detection.
It only receives the existing `source_tx_mask` and maps selected source packets
back to the original `[C,H,W]` feature layout.

## Outputs per sender → ego link

- selected source packet ranges `[start, end)`
- selected byte ranges
- selected original-value ranges
- retained value ratio
- per-channel retention ratio `[C]`
- channel summary: fully/partially/zero-retained channel counts
- spatial retention summary
- front-half vs back-half channel retention bias

For the first `SAVE_FIRST_N_LINKS`, tensor snapshots additionally contain:

- `channel_retention_ratio_tensor: [C]`
- `spatial_retention_ratio_tensor: [H,W]`
- `retained_value_mask`

Optional PNG plots are generated when matplotlib is available.

## Install

```bash
bash install.sh /home/server/v2x_projects/OPV2V
```

## Two-frame check

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

export MODEL_DIR=/home/server/v2x_projects/OPV2V/opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/experiment25_budget_retention_test
export MAX_SAMPLES=2
export NUM_WORKERS=0
export SAVE_TENSORS=1
export SAVE_FIRST_N_LINKS=12
export SYSTEM_BUDGET_MBPS=300
export TX_WINDOW_MS=100

bash scripts/run_experiment25_budget_retention_audit.sh
```

Use a new `OUT_ROOT`; the original model/config/checkpoint remain untouched.
