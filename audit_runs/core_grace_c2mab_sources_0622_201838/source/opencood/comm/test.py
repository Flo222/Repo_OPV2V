MODEL_DIR=opencood/logs/point_pillar_v2xvit_opv2v_2026_05_13_20_33_54_arce_eval

for state in good medium bad; do
  python opencood/tools/set_delay_by_channel_state.py \
    --config ${MODEL_DIR}/config.yaml \
    --state ${state}

  python opencood/tools/debug_v2xvit_delay_prior.py \
    --hypes_yaml ${MODEL_DIR}/config.yaml \
    --num_samples 5 \
    2>&1 | tee ${MODEL_DIR}/debug_delay_${state}.log

  python opencood/tools/inference_arce.py \
    --model_dir ${MODEL_DIR} \
    --fusion_method intermediate \
    --save_comm \
    --arce_enabled true \
    --arce_channel_state ${state} \
    --arce_late_policy allow \
    --num_workers 0 \
    --comm_log_dir ${MODEL_DIR}/eval_patchmax_delay_${state} \
    2>&1 | tee ${MODEL_DIR}/eval_patchmax_delay_${state}_stdout.log

python opencood/tools/inference_arce.py \
  --model_dir ${MODEL_DIR} \
  --fusion_method intermediate \
  --arce_enabled false \
  --num_workers 0 \
  2>&1 | tee ${MODEL_DIR}/eval_delay_only_no_arce_stdout.log

========== good AP ==========
The Average Precision at IOU 0.3 is 0.91, The Average Precision at IOU 0.5 is 0.88, The Average Precision at IOU 0.7 is 0.60
========== medium AP ==========
The Average Precision at IOU 0.3 is 0.88, The Average Precision at IOU 0.5 is 0.76, The Average Precision at IOU 0.7 is 0.50
========== bad AP ==========
The Average Precision at IOU 0.3 is 0.74, The Average Precision at IOU 0.5 is 0.62, The Average Precision at IOU 0.7 is 0.43

MODEL_DIR=opencood/logs/point_pillar_v2xvit_opv2v_2026_05_13_20_33_54_arce_eval

python opencood/tools/inference_arce.py \
  --model_dir ${MODEL_DIR} \
  --fusion_method intermediate \
  --save_comm \
  --arce_enabled true \
  --arce_late_policy allow \
  --num_workers 0 \
  --comm_log_dir ${MODEL_DIR}/eval_link_markov_patchmax_full \
  2>&1 | tee ${MODEL_DIR}/eval_link_markov_patchmax_full_stdout.log