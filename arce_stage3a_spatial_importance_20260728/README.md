# ARCE Stage 3A: Sender-Local Spatial Importance

This package replaces the Where2Comm confidence score used by the ARCE-C2MAB
payload ordering path with a deterministic sender-local ARCE score.

For a spatial feature vector `X[:,h,w]`, the score is:

```text
rms(h,w) = sqrt(mean_c(X[c,h,w]^2))
priority = rms / max(rms)
```

Exact all-zero spatial units are excluded. Remaining complete spatial units
are sorted in descending priority order and retain stable flattened spatial
`unit_id` values.

The change is intentionally restricted to ARCE-C2MAB:

- ARCE bypasses the native Where2Comm mask and ranks the sender feature map.
- Fixed keeps its existing native Where2Comm mask and CK1 payload path.
- UCB, reward, action space, budgets, quantization, packetization, and cache are
  unchanged.

## Install

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

tar -xzf arce_stage3a_spatial_importance_20260728.tar.gz
PKG="$PWD/arce_stage3a_spatial_importance_20260728"

REPO_DIR="$PWD" bash "$PKG/install.sh" --check
REPO_DIR="$PWD" bash "$PKG/install.sh" --apply
python "$PKG/tests/test_stage3a_runtime_integration.py"
```

Do not proceed to the zero-suppression stage until the integration test and a
100-frame ARCE-only AP+BW run pass.

## 100-frame run

```bash
METHOD=arce RUN_AP=1 RUN_BW=1 MAX_FRAMES=100 \
TAG=stage3a_spatial_importance_ap_bw100 \
bash scripts/run_arce_pair_eval.sh
```

The runtime record must report:

```text
candidate_source = arce_nonzero_spatial_support
priority_source = arce_sender_feature_rms
layout = KC
```

Audit it with:

```bash
python "$PKG/audit_stage3a_runtime.py" \
  outputs/arce_c2mab/stage3a_spatial_importance_ap_bw100/runtime_records.jsonl
```

The Fixed config is not included in this package and must remain unchanged.
