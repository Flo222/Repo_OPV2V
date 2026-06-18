# Step 3 结论：feature delta 与 budget consistency 仪表化

## 1. 阶段目标

本步骤验证两件事：

1. GRACE/ARCE 是否真实修改进入 AttentionFusion 前的 feature；
2. oracle estimated cost、allocated budget、executor actual tx_bytes 是否一致。

## 2. Feature delta 结果

Probe 配置：

- model_dir = opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0
- split = test
- dataset_len = 2170
- max_samples = 30

结果：

- found_count = 30
- max_abs_max = 3.8554556369781494
- mean_abs_avg = 0.001964524679351598
- nz_ratio_avg = 0.053031892081101734

结论：

arce_feature_delta 能够从模型输出中正常读取，且 max_abs、mean_abs、nz_ratio 均大于 0。这说明 ARCE executor 不是只做通信统计，而是确实修改了 Where2Comm mask 之后、AttentionFusion 之前的融合输入 feature。

## 3. Budget consistency 结果

Probe 配置：

- model_dir = opencood/logs/main_opv2v_where2comm_grace_full
- split = test
- dataset_len = 2170
- max_samples = 80

结果：

- found_count = 59
- viol_actual_over_alloc_gt_1.00 = 0
- viol_actual_over_alloc_gt_1.05 = 0
- actual_tx_sum = 1946624.0
- allocated_sum = 1946624.0
- estimated_sum = 1946624.0
- actual_over_alloc_max = 1.0
- actual_over_alloc_avg = 1.0
- actual_over_est_max = 1.0
- actual_over_est_avg = 1.0

结论：

当前 C²MAB/GRACE 路径下，proposal estimated cost、oracle allocated budget、executor actual tx_bytes 三者一致，没有发现预算超发问题。

## 4. 本步骤新增仪表化字段

### arce_feature_delta

位置：

opencood/models/fuse_modules/where2comm_arce_fuse.py

字段：

- max_abs
- mean_abs
- nz_ratio

### budget_consistency

位置：

opencood/comm/arce/arce_c2mab_comm.py

字段：

- estimated_tx_bytes
- allocated_budget_bytes
- actual_tx_bytes
- actual_over_est
- actual_over_allocated

## 5. 总结

Step 3 已完成。

当前可以证明：

1. GRACE/ARCE 会真实改写进入融合前的 feature；
2. C²MAB oracle 的 cost 估计、预算分配和 executor 实际发送量在小样本 probe 中保持一致；
3. 当前 AP 变化可以认为来自真实 feature 改写与真实通信约束，而不是单纯日志统计或预算未生效。
