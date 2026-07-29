# ARCE AP-proxy v3.2b paired decoded-box audit

This package adds one offline diagnostic tool:

`opencood/tools/audit_ap_proxy_v32b_paired_box_groupcv.py`

It does not modify runtime inference, reward, UCB, actions, cache, transport,
budgets, or model configuration. It does not save runtime model files.

## Purpose

The tool tests whether decoded predictions become useful for AP-proxy learning
when each collaborative action is paired with the no-send output from the same
sequence, frame, and sender.

It compares:

1. `head_only`
2. `current_box_only`
3. `paired_box_only`
4. `head_plus_paired_box`

The paired groups contain current-action, no-send/ego-reference, and difference
features for:

- predicted box count;
- non-empty prediction marker;
- mean prediction score;
- maximum prediction score;
- estimated prediction score sum.

## Important limitation

The no-send decoded reference comes from the seven-action counterfactual audit.
It is not available in the current single-action online reward path. Therefore
paired-box results are an offline upper-bound diagnostic and cannot be enabled
directly.

Every feature family is selected using GroupKFold on development sequences.
Each predeclared family is also evaluated on the existing reference holdout to
determine whether paired features are worth implementing online. These family
holdout comparisons are exploratory, not a final untouched paper test.

## Install

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

tar -xzf arce_ap_proxy_v32b_paired_box_groupcv_20260728.tar.gz
PKG="$PWD/arce_ap_proxy_v32b_paired_box_groupcv_20260728"

REPO_DIR="$PWD" bash "$PKG/install.sh" --check
REPO_DIR="$PWD" bash "$PKG/install.sh" --apply
```

## Run

```bash
python opencood/tools/audit_ap_proxy_v32b_paired_box_groupcv.py \
  --csv audit_runs/ap_proxy_v32b_paired_box/counterfactual_proxy_dataset_paired_box.csv \
  --out_dir audit_runs/ap_proxy_v32b_paired_box_groupcv \
  --holdout_fraction 0.2 \
  --folds 4 \
  --selection_estimators 200 \
  --final_estimators 500 \
  --max_depths 4,6,8 \
  --min_samples_leaves 8,16,32 \
  --tie_tolerance 0.01
```

Primary output:

`audit_runs/ap_proxy_v32b_paired_box_groupcv/ap_proxy_v32b_paired_box_groupcv_report.json`

The report always uses `status: offline_diagnostic_only`. Check
`paired_upper_bound_result` and `feature_family_gates` to determine whether an
online ego-only decode path is justified.
