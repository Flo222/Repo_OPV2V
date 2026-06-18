# Step 7-Cache Gating：reward 中 cache 项门控修正结论

## 1. 修改目的

当前 reward 中 cache_term 原本为：

cache_term = cache_quality if cache_enabled else 0

该设计会导致只要 cache_enabled=1 且 cache_quality 较高，就会获得 cache reward。即使当前链路接收质量 q_eff 已经较高，cache 仍可能被奖励。这会使 cache1 在部分场景下无条件占优，存在 reward 与最终 AP@0.7 脱钩的风险。

因此，本步骤将 cache reward 改为只在当前通信质量不足时发挥补偿作用。

## 2. 修改内容

修改文件：

opencood/comm/arce/policies/reward.py

修改后公式为：

cache_term = cache_quality × (1 - q_eff) × I(cache=1)

也就是：

- 当 q_eff 较高时，cache_term 自动降低；
- 当 q_eff 较低时，cache 才获得更高补偿 reward；
- 当 cache_enabled=0 时，cache_term 恒为 0。

## 3. 公式验证结果

500 帧 probe 结果如下：

- total_reward_items = 823
- cache1_items = 328
- cache0_items = 495
- max_abs_error_all = 0.0
- max_abs_error_cache1 = 0.0
- max_cache_term_cache0 = 0.0

说明当前代码中的 cache_term 与目标公式完全一致。

按 q_eff 分组后：

q_eff >= 0.8：

- cache_quality_mean = 0.7732
- cache_term_mean = 0.0576

0.4 < q_eff < 0.8：

- cache_quality_mean = 0.7046
- cache_term_mean = 0.2434

q_eff <= 0.4：

- cache_quality_mean = 0.6292
- cache_term_mean = 0.4617

结果说明链路质量越高，cache reward 越小；链路质量越低，cache reward 越大，符合补偿型 cache reward 的设计目标。

## 4. Reward 与 AP 相关性变化

修改前：

- reward_mean vs AP@0.7 Pearson = 0.2893
- reward_mean vs AP@0.7 Spearman = 0.2598

修改后：

- reward_mean vs AP@0.7 Pearson = 0.3287
- reward_mean vs AP@0.7 Spearman = 0.3160

说明 cache gating 后，reward 与 AP@0.7 的相关性有所提升。

## 5. 全量 AP 结果

使用 test split 2170 帧重新推理，结果为：

- AP@0.3 = 0.791
- AP@0.5 = 0.778
- AP@0.7 = 0.609

与 Step5 后 GRACE 结果相比：

- AP@0.3: 0.793 → 0.791
- AP@0.5: 0.779 → 0.778
- AP@0.7: 0.610 → 0.609

AP 变化极小，基本持平。

## 6. 结论

cache gating 修改有效。

该修改使 cache reward 从“只要启用 cache 就可能加分”变为“仅在当前通信质量不足时提供补偿”。500 帧相关性诊断显示 reward 与 AP@0.7 的一致性有所提升；全量 test 推理显示最终 AP 基本不下降。因此，该修改可以保留。

不过，当前 reward_mean 与 AP@0.7 的相关性仍低于 mean_conf 与 AP@0.7 的相关性，说明 reward 与最终检测指标之间仍存在目标偏差。后续需要继续处理 FEC reward 或进一步构建 AP proxy reward。
