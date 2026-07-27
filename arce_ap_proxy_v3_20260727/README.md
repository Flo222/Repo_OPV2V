# ARCE AP-proxy v3

This is an incremental patch on top of the installed AP-proxy v2 plus the
global-delta hotfix.

It changes only:

- shared AP-proxy head features;
- model-side proxy feature extraction;
- counterfactual dataset export;
- dataset merge;
- proxy training and validation metrics.

It does not change reward weights, UCB, action space, physical budgets,
transport, or cache.

## Install

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

tar -xzf arce_ap_proxy_v3_20260727.tar.gz
PKG="$PWD/arce_ap_proxy_v3_20260727"

REPO_DIR="$PWD" bash "$PKG/install.sh" --check
REPO_DIR="$PWD" bash "$PKG/install.sh" --apply
```

## Smoke collection

```bash
MAX_FRAMES=60 AUDIT_FRAMES=3 AUDIT_STRIDE=18 AUDIT_START=0 \
SENDER_INDEX=1 OUT_DIR=audit_runs/ap_proxy_v3_smoke_sender1 \
bash scripts/run_arce_counterfactual_audit.sh

python - <<'PY'
import csv
from pathlib import Path

p = Path(
    "audit_runs/ap_proxy_v3_smoke_sender1/"
    "counterfactual_proxy_dataset.csv"
)
rows = list(csv.DictReader(p.open(encoding="utf-8")))
required = {
    "sequence_id",
    "sequence_frame_idx",
    "collab_reg_abs_mean",
    "ego_reg_abs_mean",
    "spatial_prob_l1_mean",
    "reg_diff_abs_mean",
}
missing = sorted(required.difference(rows[0] if rows else {}))
print("rows:", len(rows))
print("missing:", missing)
assert len(rows) == 21
assert not missing
print("AP-proxy v3 smoke: PASS")
PY
```

## Collect sender datasets

The v2 CSV cannot train v3 because it does not contain RM and spatial-pair
features. Collect new matched-state data.

```bash
MAX_FRAMES=2170 AUDIT_FRAMES=240 AUDIT_STRIDE=9 AUDIT_START=0 \
SENDER_INDEX=1 OUT_DIR=audit_runs/ap_proxy_v3_sender1 \
bash scripts/run_arce_counterfactual_audit.sh

MAX_FRAMES=2170 AUDIT_FRAMES=240 AUDIT_STRIDE=9 AUDIT_START=0 \
SENDER_INDEX=2 OUT_DIR=audit_runs/ap_proxy_v3_sender2 \
bash scripts/run_arce_counterfactual_audit.sh
```

Frames without the requested sender are skipped legally.

## Merge

```bash
python opencood/tools/merge_counterfactual_proxy_datasets.py \
  --inputs \
    audit_runs/ap_proxy_v3_sender1/counterfactual_proxy_dataset.csv \
    audit_runs/ap_proxy_v3_sender2/counterfactual_proxy_dataset.csv \
  --out_csv audit_runs/ap_proxy_v3_merged/counterfactual_proxy_dataset.csv \
  --out_meta audit_runs/ap_proxy_v3_merged/merge_meta.json
```

## Train and validate

Validation uses complete held-out OPV2V sequences, not rows sampled from the
same sequence.

```bash
python opencood/tools/train_counterfactual_ap_proxies.py \
  --csv audit_runs/ap_proxy_v3_merged/counterfactual_proxy_dataset.csv \
  --out_abs_model audit_runs/ap_proxy_v3_models/ap_proxy_abs_rf_v3.pkl \
  --out_delta_model audit_runs/ap_proxy_v3_models/ap_proxy_delta_rf_v3.pkl \
  --out_meta audit_runs/ap_proxy_v3_models/ap_proxy_v3_meta.json \
  --validation_fraction 0.2 \
  --tie_tolerance 0.01 \
  --seed 2026 \
  --n_estimators 500 \
  --max_depth 10 \
  --min_samples_leaf 2
```

Do not enable the models in the ARCE config until the sequence-holdout
metadata has been reviewed.
