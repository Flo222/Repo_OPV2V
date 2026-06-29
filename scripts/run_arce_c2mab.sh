#!/usr/bin/env bash
set -euo pipefail

METHOD_NAME="${METHOD_NAME:-ARCE-C2MAB}"
MODEL_DIR="${MODEL_DIR:-opencood/logs/main_opv2v_where2comm_grace_full}"
OUT_DIR="${OUT_DIR:-outputs/arce_c2mab/default}"

export METHOD_NAME
export MODEL_DIR
export OUT_DIR

bash scripts/run_arce_single_eval.sh
