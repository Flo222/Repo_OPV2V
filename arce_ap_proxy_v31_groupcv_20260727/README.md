# ARCE AP-proxy v3.1 GroupKFold audit

This package adds one offline audit tool:

`opencood/tools/audit_ap_proxy_v31_groupcv.py`

It does not modify runtime inference, reward, UCB, actions, cache, transport,
budgets, or model configuration.

## Purpose

The tool uses the existing merged v3 counterfactual CSV to:

1. keep the last 20% of complete sequences as a reference holdout;
2. run GroupKFold only on the remaining development sequences;
3. compare four feature families;
4. compare regularized random-forest settings;
5. train absolute, global-delta, and sender-marginal experimental models;
6. evaluate all selected models on the reference holdout;
7. emit a strict `do_not_enable` or `reference_holdout_passed` status.

The four feature families are:

- `v2_psm`
- `robust_psm`
- `robust_psm_rm`
- `full_v3`

The robust families exclude unstable maximum and p99 features.

## Install

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

tar -xzf arce_ap_proxy_v31_groupcv_20260727.tar.gz
PKG="$PWD/arce_ap_proxy_v31_groupcv_20260727"

REPO_DIR="$PWD" bash "$PKG/install.sh" --check
REPO_DIR="$PWD" bash "$PKG/install.sh" --apply
```

## Run

```bash
python opencood/tools/audit_ap_proxy_v31_groupcv.py \
  --csv audit_runs/ap_proxy_v3_merged/counterfactual_proxy_dataset.csv \
  --out_dir audit_runs/ap_proxy_v31_groupcv \
  --holdout_fraction 0.2 \
  --folds 4 \
  --selection_estimators 200 \
  --final_estimators 500 \
  --max_depths 4,6,8 \
  --min_samples_leaves 8,16,32 \
  --tie_tolerance 0.01
```

Primary output:

`audit_runs/ap_proxy_v31_groupcv/ap_proxy_v31_groupcv_report.json`

The generated `.pkl` files are experimental. Do not enable them in the main
configuration unless the report gate passes and an additional untouched test
is completed.
