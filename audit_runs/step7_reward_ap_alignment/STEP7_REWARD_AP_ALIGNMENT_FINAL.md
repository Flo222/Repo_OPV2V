# Step 7：Reward 与 AP 对齐修正最终结论

## 1. 阶段目标

本步骤用于修正 C²MAB reward 与最终检测指标 AP@0.7 不一致的问题。前期 500 帧相关性诊断显示，原始 reward_mean 与 AP@0.7 的相关性明显低于 mean_conf 与 AP@0.7 的相关性，说明原 reward 更偏向通信质量、FEC/cache 等通信层 proxy，而不是最终检测收益。

原始相关性结果：

- mean_conf vs AP@0.7：Pearson = 0.5330，Spearman = 0.5088
- reward_mean vs AP@0.7：Pearson = 0.2893，Spearman = 0.2598

因此，本步骤采用保守方式修正 reward 中的 cache 和 FEC 项。

## 2. Cache reward 修正

原始形式：

cache_term = cache_quality if cache_enabled else 0

问题是只要 cache_enabled=1 且 cache_quality 较高，就可能获得 cache reward，即使当前链路接收质量 q_eff 已经较高。

修改后：

cache_term = cache_quality × (1 - q_eff) × I(cache=1)

该设计使 cache reward 变成补偿型项：

- q_eff 高时，cache reward 自动降低；
- q_eff 低时，cache 才获得更高补偿；
- cache_enabled=0 时，cache_term 恒为 0。

500 帧验证结果：

- max_abs_error_all = 0.0
- max_abs_error_cache1 = 0.0
- max_cache_term_cache0 = 0.0

按 q_eff 分组：

q_eff >= 0.8：

- cache_quality_mean = 0.7732
- cache_term_mean = 0.0576

0.4 < q_eff < 0.8：

- cache_quality_mean = 0.7046
- cache_term_mean = 0.2434

q_eff <= 0.4：

- cache_quality_mean = 0.6292
- cache_term_mean = 0.4617

说明 cache reward 已经从无条件奖励变成低通信质量下的补偿项。

## 3. FEC reward 修正

原始形式：

fec_gain = fec_recovered / missing_by_loss

该形式奖励的是“丢失包中恢复了多少比例”。当丢失包数量较少时，即使 FEC 只恢复少量 packet，fec_gain 也可能接近 1，容易高估 FEC 对感知质量的实际贡献。

修改后：

fec_gain = fec_recovered / selected_src

也就是奖励“FEC 恢复了整条 source message 中多少有效 source packet”。

修改前 500 帧统计：

- rho0p25 fec_gain_mean = 0.2007，reward_mean = 0.1919
- rho0p5 fec_gain_mean = 0.2935，reward_mean = 0.1919

修改后 500 帧统计：

- rho0p25 fec_gain_mean = 0.0289，reward_mean = 0.1532
- rho0p5 fec_gain_mean = 0.0605，reward_mean = 0.1458

rho0 始终满足：

- fec_gain_mean = 0.0
- fec_gain_nonzero = 0

说明 FEC reward 没有无条件按 rho 加分，同时 conservative FEC 修正有效降低了 fec_gain 虚高问题。

## 4. Reward 与 AP 相关性变化

原始 reward：

- reward_mean vs AP@0.7：Pearson = 0.2893，Spearman = 0.2598

cache gating 后：

- reward_mean vs AP@0.7：Pearson = 0.3287，Spearman = 0.3160

cache gating + conservative FEC 后：

- reward_mean vs AP@0.7：Pearson = 0.3111，Spearman = 0.3002

说明 cache gating 明显提升 reward 与 AP 的一致性；conservative FEC 后相关性略低于 cache-only，但仍高于原始 reward，并且有效抑制了 FEC reward 虚高。

## 5. 全量 AP 结果

Step5 GRACE current：

- AP@0.3 = 0.793
- AP@0.5 = 0.779
- AP@0.7 = 0.610

Step7 cache-gated：

- AP@0.3 = 0.791
- AP@0.5 = 0.778
- AP@0.7 = 0.609

Step7 cache + conservative FEC：

- AP@0.3 = 0.793
- AP@0.5 = 0.780
- AP@0.7 = 0.611

最终结果显示，cache + conservative FEC 修正后，AP 不仅没有下降，AP@0.5 和 AP@0.7 还有轻微提升。

## 6. 最终结论

Step 7 reward 修正有效。

本步骤将 cache reward 从无条件奖励改为低 q_eff 补偿项，将 FEC reward 从“恢复丢失包比例”改为“恢复总 source packet 比例”。修改后，reward 与 AP@0.7 的相关性高于原始 reward，FEC/cache 不再明显虚高，全量 test AP 基本持平并略有提升。因此，当前 cache gating 与 conservative FEC reward 修改可以保留。

不过，当前 reward_mean 与 AP@0.7 的相关性仍明显低于 mean_conf 与 AP@0.7 的相关性，说明手工 reward 仍未完全对齐最终检测指标。后续更系统的方向是构建 AP proxy reward，使 bandit 的学习目标进一步从通信质量转向检测质量。
