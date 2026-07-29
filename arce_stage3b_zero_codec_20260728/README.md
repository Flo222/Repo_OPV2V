# ARCE Stage 3B: Adaptive Unit Zero Codec

This package adds zero suppression after quantization and before FEC. It is
enabled only in the ARCE-C2MAB config. The Fixed config is not included and
continues to use its original byte-stream path.

For each prioritized KC spatial unit, the codec chooses the shorter of:

```text
dense quantized values
bitmap of nonzero channels + packed nonzero quantized values
```

Each transmitted record contains its stable spatial unit id. Records are
placed in fixed 1024-byte source packets before FEC, budget cropping, and
Bernoulli packet loss. A lost packet therefore cannot desynchronize later
records. Codec headers and bitmaps are included in the byte count.

If zero suppression does not reduce the number of physical source packets,
the codec falls back to the original dense byte stream. It therefore cannot
increase the source-packet count.

## Install

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

tar -xzf arce_stage3b_zero_codec_20260728.tar.gz
PKG="$PWD/arce_stage3b_zero_codec_20260728"

REPO_DIR="$PWD" bash "$PKG/install.sh" --check
REPO_DIR="$PWD" bash "$PKG/install.sh" --apply
python "$PKG/tests/test_stage3b_runtime_integration.py"
```

The integration test must show:

```text
ARCE zero codec: enabled True
Fixed zero codec: enabled False
Stage 3B runtime integration: PASS
```

## 100-frame validation

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

METHOD=arce RUN_AP=1 RUN_BW=1 MAX_FRAMES=100 \
TAG=stage3b_zero_codec_ap_bw100 \
bash scripts/run_arce_pair_eval.sh
```

Audit the real runtime records:

```bash
python "$PKG/audit_stage3b_runtime.py" \
  outputs/arce_c2mab/stage3b_zero_codec_ap_bw100/runtime_records.jsonl
```

Inspect these fields before any full run:

- `encoding_modes`: how often adaptive bitmap encoding was useful.
- `total_source_packet_savings`: physical source packets removed by the codec.
- `nonzero_value_ratio`: post-quantization nonzero ratio by quant mode.
- `source_packet_regression`: must never occur.
- `metadata_bytes`: must be positive for adaptive records and is included in
  encoded bytes.

Do not run the full dataset until the integration and runtime audits pass.

## Scope

Stage 3B does not change:

- sender selection or the greedy oracle;
- D-LinUCB/UCB;
- reward;
- action space;
- channel profiles or physical budgets;
- FEC and packet-loss models;
- receiver temporal-cache policy;
- Fixed baseline configuration.
