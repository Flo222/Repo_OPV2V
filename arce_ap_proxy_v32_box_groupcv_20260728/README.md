# ARCE AP-proxy v3.2 decoded-box audit

This package adds one offline audit tool:

`opencood/tools/audit_ap_proxy_v32_box_groupcv.py`

It does not modify runtime inference, reward, UCB, actions, cache, transport,
budgets, or model configuration.

## Question tested

The tool compares three runtime-feasible feature families:

1. `head_only`: the robust PSM/RM features used by the v3.1 audit;
2. `box_summary_only`: decoded box count and confidence summaries;
3. `head_plus_box_summary`: both groups together.

Decoded-box inputs are:

- `decoded_num_pred_boxes`
- `decoded_has_predictions`
- `decoded_score_mean`
- `decoded_score_max`
- `decoded_score_sum_est`

The existing dataset contains decoded summaries only for the collaborative
action output. It does not contain decoded ego-only box summaries. Therefore
this audit does not claim to test a decoded-box paired delta.

## Validation protocol

- Last 20 percent of complete sequences: reference holdout.
- Remaining sequences: GroupKFold model selection.
- Targets: absolute quality, global delta, and sender-marginal delta.
- Candidate models must pass minimum CV eligibility checks before their
  ranking score is considered.
- If no candidate is eligible, the best model is retained only as a
  diagnostic fallback and the final gate cannot pass.

The generated pickle files are experimental and must not be enabled unless the
strict holdout gate passes and a separate untouched test also passes.

## Install

```bash
cd ~/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$PWD:$PYTHONPATH

tar -xzf arce_ap_proxy_v32_box_groupcv_20260728.tar.gz
PKG="$PWD/arce_ap_proxy_v32_box_groupcv_20260728"

REPO_DIR="$PWD" bash "$PKG/install.sh" --check
REPO_DIR="$PWD" bash "$PKG/install.sh" --apply
```

## Run

```bash
python opencood/tools/audit_ap_proxy_v32_box_groupcv.py \
  --csv audit_runs/ap_proxy_v32_box_summary/counterfactual_proxy_dataset_box.csv \
  --out_dir audit_runs/ap_proxy_v32_box_groupcv \
  --holdout_fraction 0.2 \
  --folds 4 \
  --selection_estimators 200 \
  --final_estimators 500 \
  --max_depths 4,6,8 \
  --min_samples_leaves 8,16,32 \
  --tie_tolerance 0.01
```

Primary output:

`audit_runs/ap_proxy_v32_box_groupcv/ap_proxy_v32_box_groupcv_report.json`

Interpretation:

- `eligible_candidate`: the selected model passed minimum development CV
  conditions.
- `diagnostic_fallback_no_eligible_candidate`: no candidate passed; do not
  enable the saved model.
- `reference_holdout_passed`: all strict holdout checks passed, but an
  untouched test is still required.
- `do_not_enable`: the proxy is not ready for runtime use.
