# Baseline feature-budget audit v3

This package is prepared from the real one-frame hook discovery results.

## Selected communication tensors

- Where2Comm: first transmitted backbone scale only (`fusion_net.fuse_modules.0@input[0]`).
- V2X-ViT: 256-D feature tensor after removing the three repeated prior channels (`fusion_net.encoder.layers.0.0@input[0]`); add 12 metadata bytes per sender for velocity, delay and infrastructure flag.
- RoCooper: clean feature before its native impairment module (`comm_module@input[0]`).
- CoopDiff: student communication branch at three scales; teacher branch is excluded.
- CoSDH: three intermediate scales; intermediatelate dataset/inference is supported.

Native ARCE/Markov/RoCooper impairment is disabled only in the audit runtime so the tool measures the clean tensor presented to a plug-in communication module. Source models and configs are not modified.

## Install

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
unzip -o baseline_feature_budget_batch_v3.zip
bash baseline_feature_budget_batch_v3/install.sh /home/server/v2x_projects/OPV2V
```

## Re-run the three previously failed discoveries

```bash
bash baseline_feature_budget_batch_v3/run_discover_missing.sh
```

Expected reason for the old failures:

- CoSDH required the custom `intermediatelate` dataset/inference entry.
- V2X-Real Where2Comm had obsolete ARCE reward keys; v3 disables native ARCE while profiling clean feature shapes.

## Profile all ten baselines

```bash
export PROJECT_ROOT=/home/server/v2x_projects/OPV2V
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/baseline_feature_budget_profiles_v3
export MAX_FRAMES=20
export NUM_WORKERS=0
export SEED=2026
export QUANT_MODES=fp32,fp16,int8,int4
export RHOS=0,0.1,0.25,0.6
export PROFILES='good:27:0.05:10,medium:5:0.20:50,bad:1:0.35:100'
export TX_WINDOW_MS=100
export PACKET_SIZE_BYTES=1024
export BUDGET_SCOPE=both
export PACKETIZATION_MODE=concat
bash baseline_feature_budget_batch_v3/run_profile_selected.sh
```

Outputs include per-link feature sizes, current source-first budget fit, budget-feasible source/parity plans and combined summaries across all successful baselines.
