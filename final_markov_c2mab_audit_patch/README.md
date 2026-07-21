## v3 compatibility fix

Normalizes saved configs where `arce.recovery` is a parameter mapping. The runtime copy uses `recovery: temporal_cache` (or the first enabled supported method), while preserving the full mapping under `recovery_config`. The source model directory is never modified.

# Final Markov + C2MAB Online Audit

This patch does **not** change the C2MAB action space, reward, UCB update,
source-first scheduler, quantization, FEC, cache, or fusion logic.

It adds a reproducible final evaluation layer:

- one continuous online/prequential C2MAB trajectory;
- link-level Good/Medium/Bad Markov channel;
- Good: 27 Mbps, PLR 0.05, delay 10 ms;
- Medium: 5 Mbps, PLR 0.20, delay 50 ms;
- Bad: 1 Mbps, PLR 0.35, delay 100 ms;
- transition matrix:
  - Good: `[0.85, 0.13, 0.02]`
  - Medium: `[0.10, 0.80, 0.10]`
  - Bad: `[0.03, 0.17, 0.80]`
- deterministic seeds and raw runtime record export;
- exact dataset AP and state-composition AP;
- action distribution by link state;
- requested-action versus executor-action consistency;
- source/parity budget, loss, and FEC execution summaries;
- empirical Markov transition matrix and state run lengths;
- warm-up / steady-state-candidate / tail summaries.

## Install

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

unzip final_markov_c2mab_audit_patch.zip
bash final_markov_c2mab_audit_patch/install.sh \
  /home/server/v2x_projects/OPV2V
```

## 200-frame smoke run

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

export SOURCE_MODEL_DIR=/home/server/v2x_projects/OPV2V/opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div
export TEST_DIR=/home/server/v2x_projects/OPV2V/opv2v_data_dumping/test
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/final_markov_c2mab_seed2026_test200
export SEED=2026
export MAX_FRAMES=200
export WARMUP_FRAMES=50
export NUM_WORKERS=0

bash scripts/run_final_markov_c2mab_audit.sh
```

## Full test trajectory

Use a fresh output directory. Do not resume the 200-frame policy state.
The policy must start from the same clean initialization and run continuously
through the whole test trajectory.

```bash
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/final_markov_c2mab_seed2026_full
export MAX_FRAMES=-1
export WARMUP_FRAMES=500
export WINDOW_SIZE=200
export WINDOW_STRIDE=200
export RESET_OUT=1

bash scripts/run_final_markov_c2mab_audit.sh
```

## Required log checks

The online evaluation log must show a checkpoint load, typically:

```text
resuming by loading epoch 20
```

The final audit should report:

```text
pass: true
known_state_ratio: 1.0
profile_match_ratio: 1.0
budget_exceeded_ratio: 0.0
execution_mismatch_ratio: 0.0
```

`multiple_actions_observed` and `all_expected_states_observed` are diagnostic,
not hard pass conditions for a short 200-frame smoke run.

## Main outputs

- `final_summary.json`: exact AP and total communication results.
- `online_trace.jsonl`: frame-level actions, reward, quality, and AP statistics.
- `runtime_records.jsonl`: full link/superarm/reward runtime records.
- `final_markov_c2mab_audit_summary.json`: consolidated audit.
- `state_action_summary.csv`: state-conditioned action distribution.
- `frame_state_perception.csv`: exact AP grouped by frame link-state composition and worst link state.
- `markov_transition.csv`: empirical transition matrix.
- `diagnostic_links_top100.csv`: high-recovery / high-clipping link cases.
- `rolling_metrics.json`: windowed AP, bandwidth, reward, entropy, UCB mean and bonus.

## Important interpretation

The Markov chain is link-scoped. A frame can contain collaborators in different
states. Therefore there is no mathematically valid single-link AP after fused
perception. The audit reports exact frame-level AP under:

1. the exact set of link states in the frame; and
2. the worst link state in the frame.

Link-level state statistics, action distributions, budgets, packet recovery,
and reward inputs remain exact per link.

## YAML compatibility fix

The preparation script reads the current project layout at `model.args.arce` and retains a fallback for older snapshots that placed ARCE at `model.args.where2comm_fusion.arce`.

## v4 reward-schema compatibility

Saved `final_proxy` configs with `alpha_*` keys are incompatible with the
current strict C2MAB reward implementation. The runtime preparation step now
uses `REWARD_PROFILE=r2b` by default and writes the current tested profile:

```yaml
mode: ap_delta_cost
lambda_abs: 0.1
lambda_delta: 3.0
lambda_cost: 0.1
lambda_delay: 0.0
lambda_quant: 0.0
lambda_violate: 0.0
stale_max_ms: 100.0
```

The original legacy reward mapping is retained verbatim in
`model_runtime/final_markov_manifest.json`. This is an explicit runtime reward
profile selection, not a one-to-one rename of the obsolete formula. Set
`REWARD_PROFILE=preserve` only when the source config already uses the current
`lambda_*` schema.
