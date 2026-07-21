# Experiment 4: Joint Compression + Redundancy Audit

This patch extends the already-installed Experiment-3 read-only FEC audit.
It does not change quantization, packet ordering, budget scheduling, Bernoulli
loss, FEC encoding/decoding, feature reconstruction, fusion, or detection.

Experiment grid defaults:

- quantization: `fp16 int8 int4`
- PLR: `0.20 0.35`
- redundancy: `0 0.25 0.60`
- system budget: `100 Mbps`
- transmit window: `100 ms`
- cache: disabled
- Markov channel: disabled

The audit separates:

1. generated source/parity packets;
2. source/parity packets admitted by the finite budget;
3. transmitted packets lost by the Bernoulli channel;
4. source packets recovered by FEC;
5. source packets still missing;
6. direct-only and FEC feature NMSE/cosine;
7. final AP from inference logs.

## Install

```bash
unzip experiment4_joint_compression_redundancy_audit_patch.zip
bash experiment4_joint_compression_redundancy_audit_patch/install.sh \
  /home/server/v2x_projects/OPV2V
```

Experiment 3 must already be installed.

## Run the 20-frame validation grid

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

export MODEL_DIR=/home/server/v2x_projects/OPV2V/opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/correct_model_experiment4_joint_test20
export MAX_SAMPLES=20
export NUM_WORKERS=0
export SEED=2026
export QUANT_MODES="fp16 int8 int4"
export PLRS="0.20 0.35"
export RHOS="0 0.25 0.60"
export SYSTEM_BUDGET_MBPS=100
export TX_WINDOW_MS=100
export SAVE_TENSORS=1
export SAVE_FIRST_N_LINKS=12

bash scripts/run_experiment4_joint_compression_redundancy_audit.sh
```

The summary is written to:

- `experiment4_summary.json`
- `experiment4_summary.csv`
