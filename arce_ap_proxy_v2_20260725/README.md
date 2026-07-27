# ARCE AP-proxy v2

This patch changes only AP-proxy feature extraction, counterfactual data
export, and proxy training. It does not modify transport, cache, physical
budget allocation, reward coefficients, the action space, or D-LinUCB.

## Changes

1. One canonical PSM feature extractor is shared by training and runtime.
2. Class logits are always collapsed with `max(dim=1)` before statistics.
3. Exact collab/ego PSM identity enforces paired delta `0`.
4. The seven-action counterfactual audit exports one balanced CSV row per
   frame/action trial.
5. A new trainer fits absolute and paired-delta Random Forest proxies using
   a temporal frame holdout. All seven actions from a frame stay in the same
   split.
6. Both proxy loaders support explicit model payload metadata and
   `require_model`.

## Install

From the OPV2V repository root:

```bash
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

PKG="$PWD/arce_ap_proxy_v2_20260725"
sed -i 's/\r$//' "$PKG/install.sh"

REPO_DIR="$PWD" bash "$PKG/install.sh" --check
REPO_DIR="$PWD" bash "$PKG/install.sh" --apply
```

If `--check` fails, do not force-apply the patch. The server file differs
from the reviewed snapshot and must be reconciled first.

## Small collection smoke test

```bash
MAX_FRAMES=80 \
AUDIT_FRAMES=3 \
AUDIT_STRIDE=20 \
AUDIT_START=0 \
OUT_DIR=audit_runs/ap_proxy_v2_smoke \
bash scripts/run_arce_counterfactual_audit.sh
```

The output directory must contain:

```text
counterfactual_7action.json
counterfactual_proxy_dataset.csv
```

For three valid audited frames the CSV contains 21 data rows plus one
header. Frames without the requested sender are skipped, so the count can
be lower.

## Current-pipeline training data

Do not reuse the old proxy CSV. It was collected before the current
transport/cache pipeline and used a different delta feature definition.

Initial collection:

```bash
MAX_FRAMES=2170 \
AUDIT_FRAMES=120 \
AUDIT_STRIDE=18 \
AUDIT_START=0 \
OUT_DIR=audit_runs/ap_proxy_v2_counterfactual_sender1 \
bash scripts/run_arce_counterfactual_audit.sh
```

This samples up to 120 frames and runs all seven actions from identical
pre-frame channel/cache/bandit state. Aim for at least 80 valid audited
frames before treating the proxy as a candidate for the main experiment.

## Train

```bash
DATA=audit_runs/ap_proxy_v2_counterfactual_sender1/counterfactual_proxy_dataset.csv
MODEL_OUT=audit_runs/ap_proxy_v2_models
mkdir -p "$MODEL_OUT"

python opencood/tools/train_counterfactual_ap_proxies.py \
  --csv "$DATA" \
  --out_abs_model "$MODEL_OUT/ap_proxy_abs_rf_v2.pkl" \
  --out_delta_model "$MODEL_OUT/ap_proxy_delta_rf_v2.pkl" \
  --out_meta "$MODEL_OUT/ap_proxy_v2_meta.json" \
  --validation_fraction 0.2 \
  --seed 2026 \
  --n_estimators 500 \
  --max_depth 10 \
  --min_samples_leaf 2
```

The validation set is the last 20 percent of sampled frame IDs. It is not a
random row split.

## Initial acceptance gates

Inspect `ap_proxy_v2_meta.json`. Before enabling the new models, the
validation delta proxy should be materially above chance:

```text
send_sign_accuracy                 >= 0.65
pairwise_ranking_accuracy          >= 0.60
frame_ranking_spearman_mean        >= 0.40
top1_match_rate                    >= 0.30
```

These are engineering gates, not claimed theoretical thresholds. If they
fail, do not compensate by changing reward weights.

## Enable accepted models

Only after the validation gates pass:

```bash
python - <<'PY'
from pathlib import Path
import yaml

p = Path("opencood/logs/main_opv2v_where2comm_grace_full/config.yaml")
cfg = yaml.load(p.read_text(encoding="utf-8"), Loader=yaml.Loader)
arce = cfg["model"]["args"]["arce"]

arce["ap_proxy_reward"] = {
    "enabled": True,
    "model_path": "audit_runs/ap_proxy_v2_models/ap_proxy_abs_rf_v2.pkl",
    "require_model": True,
}
arce["delta_ap_proxy_reward"] = {
    "enabled": True,
    "model_path": "audit_runs/ap_proxy_v2_models/ap_proxy_delta_rf_v2.pkl",
    "require_model": True,
}

p.write_text(
    yaml.dump(cfg, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
print(yaml.dump(arce["ap_proxy_reward"], sort_keys=False))
print(yaml.dump(arce["delta_ap_proxy_reward"], sort_keys=False))
PY
```

Then repeat a 20-frame seven-action audit. Check runtime sign/ranking
metrics again before running rolling reward and D-LinUCB convergence audits.
