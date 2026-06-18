# Step 4：message mask / patch selection 口径修正最终结论

## 1. 修正内容

本步骤将 Where2Comm + GRACE/ARCE 中 compact sparse token selection 的候选区域口径从：

message_scores = _confidence_maps

修正为：

message_scores = _confidence_maps * raw_masks

修正后含义为：

- Where2Comm binary raw mask 决定候选区域；
- confidence map 只在候选区域内部作为排序分数；
- budget-aware Top-K 只会从 Where2Comm 认为可通信的区域中选择 BEV tokens。

## 2. mask alignment probe 结果

使用 strict FP32 baseline 进行 80 个样本 probe：

- found_count = 59
- selected_tokens_total = 14396
- selected_tokens_outside_mask_total = 0
- outside_mask_ratio_max = 0.0
- outside_mask_ratio_avg = 0.0
- candidate_ratio_avg = 0.6709250208654342
- num_tokens_avg = 244.0

结论：selected tokens 没有 raw mask 外泄，compact sparse 已经被限制在 Where2Comm binary raw mask 候选区域内。

## 3. mask-aligned AP 结果

Strict FP32 baseline：

- AP@0.3 = 0.776
- AP@0.5 = 0.760
- AP@0.7 = 0.578

Mask-aligned GRACE current：

- AP@0.3 = 0.794
- AP@0.5 = 0.779
- AP@0.7 = 0.610

相对于 strict FP32 baseline，mask-aligned GRACE current 的提升为：

- AP@0.3: +0.018
- AP@0.5: +0.019
- AP@0.7: +0.032

## 4. 结论

Step 4 修改成功。

修正后，GRACE/ARCE 的 patch/token selection 口径与 Where2Comm 通信 mask 对齐。当前系统可以表述为：

Where2Comm 首先根据置信图生成二值通信候选区域；GRACE/ARCE 仅在该候选区域内根据置信度和预算约束选择 Top-K BEV tokens；随后对选中的 compact tokens 进行量化、分包、丢包/FEC 恢复与 scatter 回填，再进入 AttentionFusion。

## 5. 注意事项

当前 candidate_ratio_avg 约为 0.671，说明 Where2Comm threshold = 0.01 时候选区域较宽。这不是代码错误，但后续可以做 threshold sweep 消融。
