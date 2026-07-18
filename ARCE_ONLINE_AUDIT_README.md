# ARCE online evaluation and counterfactual audit

## What changes

1. AP and BW can be computed from one online trajectory and one model instance.
2. Rolling AP/BW/reward/action entropy/UCB statistics are exported.
3. Warm-up and post-warm-up candidate phases are reported separately.
4. A matched-state seven-action counterfactual audit checks feature, prediction,
   true frame quality, paired-delta proxy sign, and action ranking.
5. Counterfactual trials never update the bandit. Normal online execution is
   unchanged unless `set_forced_action()` is explicitly called by the audit.
6. Progress output reports elapsed time and throughput. Counterfactual output
   reports both stream frame FPS and model-forward FPS because an audited frame
   runs all seven actions plus the normal online action.

Single-frame truth is frame quality `TP/(TP+FP+FN)`, averaged over IoU
0.3/0.5/0.7. It is not called dataset AP. Dataset AP is still accumulated over
the complete online trajectory.

## Static checks

```bash
python -m py_compile \
  opencood/tools/arce_online_eval.py \
  opencood/tools/audit_arce_counterfactual.py \
  opencood/comm/arce/arce_c2mab_comm.py \
  opencood/comm/arce/policies/reward_update_manager.py \
  opencood/comm/arce/policies/c2mab_execution_record_builder.py \
  opencood/models/fuse_modules/where2comm_arce_fuse.py
```

## Six unified entry modes

When both `RUN_AP=1` and `RUN_BW=1`, the default is now one-pass evaluation.
AP-only and BW-only retain the previous paths.

```bash
# ARCE: AP+BW in one pass
METHOD=arce RUN_AP=1 RUN_BW=1 SINGLE_PASS_AP_BW=1 MAX_FRAMES=100 \
TAG=online_joint_arce_100 bash scripts/run_arce_pair_eval.sh

# ARCE: AP only / BW only
METHOD=arce RUN_AP=1 RUN_BW=0 MAX_FRAMES=100 \
TAG=ap_only_arce_100 bash scripts/run_arce_pair_eval.sh
METHOD=arce RUN_AP=0 RUN_BW=1 MAX_FRAMES=100 \
TAG=bw_only_arce_100 bash scripts/run_arce_pair_eval.sh

# Fixed baseline: AP+BW in one pass
METHOD=baseline RUN_AP=1 RUN_BW=1 SINGLE_PASS_AP_BW=1 MAX_FRAMES=100 \
TAG=online_joint_fixed_100 bash scripts/run_arce_pair_eval.sh

# Fixed baseline: AP only / BW only
METHOD=baseline RUN_AP=1 RUN_BW=0 MAX_FRAMES=100 \
TAG=ap_only_fixed_100 bash scripts/run_arce_pair_eval.sh
METHOD=baseline RUN_AP=0 RUN_BW=1 MAX_FRAMES=100 \
TAG=bw_only_fixed_100 bash scripts/run_arce_pair_eval.sh
```

Use `METHOD=both` to run ARCE and Fixed sequentially. Set
`SINGLE_PASS_AP_BW=0` only when reproducing the old two-pass protocol.

Important one-pass outputs:

- `final_summary.json`: dataset AP and actual BW from the same trajectory.
- `online_trace.jsonl`: per-frame actions, bytes, reward and true frame quality.
- `rolling_metrics.json`: rolling and warm-up/post-warm-up metrics.
- `bw_breakdown.json` and `byte_accounting_audit.json`: byte accounting.
- `reward_runtime_audit.json`: reward term audit.

The default `WARMUP_FRAMES=500` is provisional, not proof of convergence. It
can be changed on the command line. The rolling output includes selected-arm
LinUCB mean/bonus and the oracle's separate warm-up exploration bonus.

## Seven-action matched-state audit

Start with a small smoke run:

```bash
MAX_FRAMES=30 AUDIT_FRAMES=2 AUDIT_STRIDE=10 AUDIT_START=0 \
OUT_DIR=audit_runs/arce_counterfactual_smoke \
bash scripts/run_arce_counterfactual_audit.sh
```

Then sample across a longer online stream so cache1 is tested after cache state
has actually accumulated:

```bash
MAX_FRAMES=500 AUDIT_FRAMES=20 AUDIT_STRIDE=25 AUDIT_START=0 \
OUT_DIR=audit_runs/arce_counterfactual_20x7 \
bash scripts/run_arce_counterfactual_audit.sh
```

Inspect `counterfactual_7action.json`:

- Every audited frame should have `matched_channel_state: true`.
- Every action should have `policy_update_applied: false`.
- `feature_delta` and `psm_vs_no_send` show whether the action changes received
  features and dense predictions.
- `true_delta_quality` is measured against forced no-send on the same frame.
- `proxy_delta_quality` is the runtime paired-delta proxy.
- `sign_accuracy`, `proxy_true_delta_spearman`, pairwise ranking accuracy and
  top-1 match rate test whether the proxy is usable for action selection.
- `transport_feature_stages` traces each send action through dense input,
  method-native sparse payload, quantization, channel recovery and dense
  scatter. `first_zero_stage_counter` identifies the first stage that becomes
  all zero.

The same stage records are also written to
`counterfactual_transport_stages.jsonl`. The counterfactual runner defaults to
`PROGRESS_INTERVAL=10`; set it explicitly for a different interval.

Do not run a new full reward ablation until this audit confirms that actions
change the model output and that proxy sign/ranking quality is materially above
chance.
