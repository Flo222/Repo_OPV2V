# Final Markov+C2MAB compatibility patch v6

Built against the uploaded `reward_test` debug bundle at commit
`9372661ba6921a4fae14d4ac948acab82b86aa7d`.

## Fixes

1. Removes the unsupported `--reward-profile` argument from the
   `arce_online_eval.py` invocation. The profile remains an argument of the
   runtime-config preparation script only.
2. Maps the saved action-space keys to the names consumed by the current code:
   - `quant -> online_quant_modes`
   - `rho -> online_redundancy_ratios`
   - `send -> send_values`
   - `cache -> cache_values`
3. Preserves the saved six-dimensional context instead of silently allowing
   the current default to force seven dimensions:
   - sets `include_cav_confidence=false` for `context_dim=6`
   - maps `normalize_bandwidth_by_mbps -> b_max_mbps`
   - maps `normalize_delay_by_ms -> stale_max_ms`
4. Keeps the existing `scheduler.per_link_budget` value instead of rewriting it.
5. Adds a preflight that validates the prepared YAML, constructs the real
   C2MAB model, verifies the effective action grid/context, and loads the
   checkpoint before the 200-frame run begins.

## Install

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

unzip -o final_markov_c2mab_audit_patch_v6.zip
bash final_markov_c2mab_audit_patch_v6/install.sh \
  /home/server/v2x_projects/OPV2V
```

Expected ending:

```text
Final Markov+C2MAB interface compatibility test passed.
Final Markov+C2MAB audit smoke test passed.
Installed final Markov+C2MAB compatibility patch v6.
```

## Run

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
export WINDOW_SIZE=50
export WINDOW_STRIDE=50
export PROGRESS_INTERVAL=20
export REWARD_PROFILE=r2b
export RUN_MODEL_PREFLIGHT=1
export RESET_OUT=1

bash scripts/run_final_markov_c2mab_audit.sh
```

Before inference, `preflight_runtime.log` must show:

```text
"static_config_pass": true
"model_build_pass": true
"loaded_epoch": 20
"actual_quant_modes": ["fp16", "int4", "int8"]
"actual_redundancy_ratios": [0.0, 0.25, 0.5]
"actual_context_dim": 6
```

The quant list is sorted alphabetically by the preflight; this does not change
runtime action ordering.


## v6 fix

- Fixes the audit smoke-test fixture, which previously omitted the legacy
  context normalization fields while asserting their converted aliases.
- Explicitly materializes `context.b_max_mbps` and `context.stale_max_ms` using
  the current runtime defaults when neither current nor legacy keys exist.
