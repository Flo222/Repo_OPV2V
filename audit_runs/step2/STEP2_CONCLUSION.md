# Step 2 结论：公平 Where2Comm-Markov baseline

## 1. Baseline 定义

本步骤建立了真正公平的 Where2Comm-Markov baseline：

Where2Comm mask
-> ARCE executor
-> fixed send=1 / FP32 / rho0 / cache0 / no FEC
-> same Markov / same budget / same true loss
-> compact_sparse token selection
-> AttentionFusion

## 2. 实验目录

opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0

## 3. 核心配置

- validate_dir = opv2v_data_dumping/test
- samples = 2170
- model = point_pillar_where2comm_arce
- policy = fixed
- fixed_action:
  - send = 1
  - quant_mode = fp32
  - rho = 0.0
  - redundancy_ratio = 0.0
  - fec_type = none
  - cache_enabled = 0
  - cache = 0
- Markov states = good / medium / bad
- compact_sparse.enabled = true

## 4. 代码修正

已对 opencood/comm/arce/arce_fixed_comm.py 做最小修正：

当 policy=fixed 且 fixed_action 存在时，communicate_feature 优先使用 fixed_action，而不是 FixedARCEPolicy 的默认 state-action 表。

该修正避免默认动作表继续选择：
- good_fp16_no_fec
- medium_int8_xor
- bad_int4_raptor_like

## 5. Probe 检查结果

probe 已确认：

- dataset_len = 2170
- records_total > 0
- Markov 状态包含 good / medium / bad
- tx_bytes > 0
- rx_bytes <= tx_bytes
- budget_violations = 0
- action 已变为 fixed_action:
  - send=1
  - fp32
  - rho0
  - no FEC
  - cache0

## 6. 全量 AP 结果

The Average Precision at IOU 0.3 is 0.776,
The Average Precision at IOU 0.5 is 0.760,
The Average Precision at IOU 0.7 is 0.578

## 7. 结论

Step 2 已完成。

该结果是当前最严格的公平 Where2Comm-Markov baseline，可作为后续 GRACE / C²MAB learned 策略的对比基准。

需要注意，之前的 0.823 / 0.809 / 0.655 并不是纯 FP32/rho0/cache0 baseline，而是仍受到 FixedARCEPolicy 默认状态动作表影响，因此不能作为严格 FP32 baseline。
