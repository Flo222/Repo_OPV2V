# Experiment 3: pure FEC recovery audit

This package adds a read-only Experiment-3 audit to the current ARCE code.

## Goal

Hold compression and transport conditions fixed, provide enough bandwidth for
all source and repair packets, and vary only Bernoulli PLR and redundancy rho.
The audit compares:

- `F_quant`: quantized/dequantized payload before loss;
- `F_direct`: only directly received source packets, missing source packets zero-filled;
- `F_fec`: payload after the actual Raptor-like peeling decoder, remaining missing zero-filled.

The normal model output continues to use `F_fec`. `F_direct` exists only for
read-only diagnostics.

## Default grid

- Quantization: INT8
- PLR: 0.20, 0.35
- rho: 0, 0.10, 0.25, 0.60
- cache: 0
- delay: 0
- Markov: disabled
- frame budget: 10000 Mbps x 100 ms, sufficient for source + repair packets
- FEC for rho > 0: `raptor_sim` (real XOR repair packets + peeling decoder; not standard RaptorQ)

All conditions use the same inference seed. The source packets are systematic
and appear before repair packets. The current Bernoulli generator therefore
uses the same random draws for source packets across rho conditions. The
summary script verifies this per frame/link using a SHA-256 fingerprint of the
source loss mask.

## Install

```bash
bash experiment3_fec_recovery_audit_patch/install.sh /path/to/OPV2V
```

## Run

```bash
export MODEL_DIR=/home/server/v2x_projects/OPV2V/opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/correct_model_experiment3_fec
export MAX_SAMPLES=20
export PLRS="0.20 0.35"
export RHOS="0 0.10 0.25 0.60"
export QUANT_MODE=int8
export SYSTEM_BUDGET_MBPS=10000
export TX_WINDOW_MS=100
bash scripts/run_experiment3_fec_recovery_audit.sh
```

After a 20-frame smoke run passes, set `MAX_SAMPLES=200` and use a new output
directory.

## Main outputs

- `<condition>/audit/fec_recovery_audit.jsonl`
- `<condition>/inference.log`
- `experiment3_summary.json`
- `experiment3_summary.csv`

Important fields:

- `num_parity_packets`
- `num_direct_received_source_packets`
- `num_fec_recovered_source_packets`
- `num_missing_source_packets`
- `source_direct_recovery_ratio`
- `source_final_recovery_ratio`
- `direct_feature_error.nmse`
- `fec_feature_error.nmse`
- `fec_gain.nmse_reduction`
- `source_loss_fingerprint`

A valid run requires zero source/parity budget drops and identical source-loss
fingerprints across rho for the same PLR/frame/link.
