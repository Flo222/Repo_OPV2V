# Step 5：send=0 与 cache0/cache1 动作语义修正最终结论

## 1. 修正目标

本步骤修正 C2MAB/GRACE 动作空间中 send 与 cache 的真实执行语义。

核心目标：

1. send=0 不应只是 oracle 未选中后的 fallback，而应进入 LinUCB reward update；
2. cache_enabled=0 时不允许 bad state 自动走 previous_frame；
3. cache_enabled=1 时才允许使用 previous_frame / temporal cache。

## 2. 修改内容

### 2.1 send=0 进入 policy update

修改文件：

- opencood/comm/arce/arce_c2mab_comm.py

修改后，当 sender 未被 ego oracle 选中时：

- 生成 send0 no-send record；
- actual_transmitted_bytes = 0；
- selected_for_update = True；
- no_send_update = True；
- 将 send0 action 加入 pending_reward；
- update_with_proxy_reward() 中会执行 policy.update()。

### 2.2 cache0 禁止 previous_frame

修改文件：

- opencood/comm/arce/arce_fixed_comm.py

修改后：

- cache_enabled=0：feature_tx = current feature，temporal_source = current_cache_disabled；
- cache_enabled=1：才调用 _get_temporal_tx_feature()，允许 bad state 使用 previous_frame。

## 3. send=0 验证结果

500 样本 probe：

- superarm_has_dropped_sender = 117
- no_send_records = 164
- no_send_selected_for_update = 164
- reward_send0_links = 164
- reward_num_updated_sum = 823
- reward_update_records = 500

结论：

send=0 已经进入 LinUCB reward update，不再只是未选中后的无学习 fallback。

## 4. cache 语义验证结果

500 样本 probe：

- cache0_records = 459
- cache0_current_cache_disabled = 295
- cache0_previous_frame_violation = 0
- cache1_records = 364
- cache1_previous_frame = 86
- cache1_current = 278

结论：

cache0 不再出现 previous_frame；cache1 可以使用 previous_frame。cache 动作语义已经生效。

## 5. Step 5 结论

Step 5 修改成功。

当前动作语义可以表述为：

- send=1：执行真实通信，产生 tx/rx/packet/loss/FEC/cache 等记录，并进入 reward update；
- send=0：不发送，tx_bytes=0，输出 zero-fill，但仍作为 send0 arm 进入 reward update；
- cache0：严格禁用 previous_frame；
- cache1：允许根据状态使用 previous_frame / temporal cache。

## 6. Step 5 后 AP 结果

| Method | AP@0.3 | AP@0.5 | AP@0.7 |
|---|---:|---:|---:|
| Strict FP32 / rho0 / cache0 after Step5 | 0.781 | 0.767 | 0.585 |
| GRACE current after Step5 | 0.793 | 0.779 | 0.610 |
| Delta: GRACE - Strict | +0.012 | +0.012 | +0.025 |

对应日志：

- Strict FP32: `opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0/inference_step5_fp32_rho0_cache0_0617_084241.log`
- GRACE current: `opencood/logs/main_opv2v_where2comm_grace_full/inference_step5_grace_current_0617_084520.log`

结论：Step 5 修正后，strict baseline 与 GRACE 均可正常推理，GRACE 仍稳定优于 strict FP32 baseline。cache0 语义修正没有破坏主方法表现。
