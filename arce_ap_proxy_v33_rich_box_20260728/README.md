# ARCE AP-proxy v3.3 rich decoded-box counterfactual audit

This package extends the existing matched-state seven-action audit with richer
decoded-prediction features. It is an offline diagnostic package.

It does not modify:

- online reward or reward weights;
- D-LinUCB/UCB statistics;
- action space or oracle;
- transport, physical budgets, quantization, or cache;
- model checkpoints or runtime AP-proxy configuration.

It never writes a runtime `.pkl` model.

## What v3.3 collects

For each audited sequence/frame/sender, all seven actions still start from the
same copied Markov, cache, and bandit state. The collector adds:

- decoded score distribution and threshold counts;
- box center, distance, quadrant, area, and 10 m grid occupancy statistics;
- current-action versus matched-state sender no-send AABB-IoU matching;
- matched-box confidence change and center displacement;
- added and removed box counts and confidence statistics.

The current inference API does not expose predicted class labels. Matching is
therefore explicitly class-agnostic. Ground truth is used only to construct
offline labels, never as a proxy input feature.

The no-send decoded result is an offline counterfactual reference for the
audited sender. Other senders can still be active, so it is a sender-marginal
reference rather than a guaranteed ego-only frame. These paired features are
not available in the current one-action online reward path. A successful result
only justifies evaluating an additional matched no-send forward later; it does
not justify enabling a model immediately.

## Install

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

tar -xzf arce_ap_proxy_v33_rich_box_20260728.tar.gz
PKG="$PWD/arce_ap_proxy_v33_rich_box_20260728"

REPO_DIR="$PWD" bash "$PKG/install.sh" --check
REPO_DIR="$PWD" bash "$PKG/install.sh" --apply
```

The preflight verifies the reviewed collector SHA-256 before replacing it.
If it reports a mismatch, do not force installation; inspect the current file
against the uploaded v3.3 source first.

## Smoke collection

```bash
MAX_FRAMES=60 AUDIT_FRAMES=3 AUDIT_STRIDE=18 AUDIT_START=0 \
SENDER_INDEX=1 OUT_DIR=audit_runs/ap_proxy_v33_smoke_sender1 \
bash scripts/run_arce_counterfactual_audit.sh
```

Verify the new schema:

```bash
python - <<'PY'
import csv
from pathlib import Path

p = Path(
    "audit_runs/ap_proxy_v33_smoke_sender1/"
    "counterfactual_proxy_dataset.csv"
)
rows = list(csv.DictReader(p.open(encoding="utf-8")))
required = {
    "decoded_score_p90",
    "decoded_grid_10m_occupancy",
    "no_send_decoded_score_p90",
    "paired_delta_decoded_score_p90",
    "paired_match_iou_mean",
    "paired_added_count",
    "paired_removed_count",
}
missing = sorted(required.difference(rows[0] if rows else {}))
print("rows:", len(rows))
print("missing:", missing)
assert len(rows) == 21
assert not missing
print("AP-proxy v3.3 smoke: PASS")
PY
```

## Recollect sender datasets

The previous v3.2 CSV/JSON contains only aggregate box summaries and cannot be
used to reconstruct box matching. New seven-action collection is required.

```bash
MAX_FRAMES=2170 AUDIT_FRAMES=240 AUDIT_STRIDE=9 AUDIT_START=0 \
SENDER_INDEX=1 OUT_DIR=audit_runs/ap_proxy_v33_sender1 \
bash scripts/run_arce_counterfactual_audit.sh

MAX_FRAMES=2170 AUDIT_FRAMES=240 AUDIT_STRIDE=9 AUDIT_START=0 \
SENDER_INDEX=2 OUT_DIR=audit_runs/ap_proxy_v33_sender2 \
bash scripts/run_arce_counterfactual_audit.sh
```

Frames without the requested sender are skipped legally.

## Merge

```bash
python opencood/tools/merge_counterfactual_proxy_datasets_v33.py \
  --inputs \
    audit_runs/ap_proxy_v33_sender1/counterfactual_proxy_dataset.csv \
    audit_runs/ap_proxy_v33_sender2/counterfactual_proxy_dataset.csv \
  --out_csv audit_runs/ap_proxy_v33_merged/counterfactual_proxy_dataset.csv \
  --out_meta audit_runs/ap_proxy_v33_merged/merge_meta.json
```

## Grouped diagnostic audit

```bash
python opencood/tools/audit_ap_proxy_v33_rich_box_groupcv.py \
  --csv audit_runs/ap_proxy_v33_merged/counterfactual_proxy_dataset.csv \
  --out_dir audit_runs/ap_proxy_v33_rich_box_groupcv \
  --holdout_fraction 0.2 \
  --folds 4 \
  --selection_estimators 200 \
  --final_estimators 500 \
  --max_depths 4,6,8 \
  --min_samples_leaves 8,16,32 \
  --tie_tolerance 0.01
```

Primary output:

```text
audit_runs/ap_proxy_v33_rich_box_groupcv/
ap_proxy_v33_rich_box_groupcv_report.json
```

Compare these predefined families:

1. `head_only`
2. `simple_paired_box`
3. `rich_paired_box`
4. `head_plus_rich_paired_box`

The report remains `offline_diagnostic_only`. Review sequence-grouped CV,
reference holdout correlation, sign lift, pairwise action ranking, frame
Spearman, top-set rate, and selected-action regret before changing the online
AP-proxy or reward.
