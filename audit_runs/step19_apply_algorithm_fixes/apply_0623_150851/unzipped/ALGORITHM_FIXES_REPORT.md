# GRACE/C2MAB 主算法针对性修复说明

日期：2026-06-22

本次补丁只处理当前模块化后的主算法代码问题，不包含 `arce_fixed_comm.py` 的第二阶段深度模块化拆分。

## 1. 修复范围

本补丁包包含以下文件：

| 文件 | 处理内容 |
| --- | --- |
| `opencood/comm/arce/arce_c2mab_comm.py` | 主控层配置传递、ego 当前帧置信度、oracle 参数接入、no-send pending 记录 |
| `opencood/comm/arce/policies/c2mab_policy_bank.py` | D-LinUCB/CW-C2UCB 反馈权重配置接入 |
| `opencood/comm/arce/policies/ego_greedy_oracle.py` | oracle 代价、探索、覆盖阈值参数可配置 |
| `opencood/comm/arce/policies/reward_pending_builder.py` | pending reward 中保留真实 channel profile，并区分 no-send 反馈 |
| `opencood/comm/arce/policies/reward_update_manager.py` | 修复全 no-send 时 AP gain 被错误均分的问题 |

## 2. 主要问题与修复

### 2.1 C_ego 没有严格使用当前帧 ego 置信度

原逻辑中 `C_ego` 主要来自 `last_ego_confidence`，这更像上一轮 AP proxy/历史状态，不是当前帧 ego 本地检测置信度。

已修复为从 `local_cav_confidences` 中按 `ego_index` 读取当前帧 ego 置信度：

```python
ego_conf = get_cav_confidence(
    local_cav_confidences,
    int(ego_index),
    default=float(self.default_ego_confidence),
)
```

这样更符合“上下文由当前信道质量和当前协作有效性信息构成”的设计口径。

### 2.2 CW-C2UCB 反馈权重配置没有完整传入策略

原先 `feedback_weight_mode`、`feedback_weight_alpha`、`feedback_weight_floor`、`statistical_weight_alpha` 等配置容易停留在 YAML/外层配置中，没有稳定进入 `DiscountedLinUCB`。

已修复：

- `arce_c2mab_comm.py` 会把外层 `arce` 配置中的反馈权重字段合并进 `policy_cfg`。
- `c2mab_policy_bank.py` 支持读取以下配置入口：
  - 直接字段：`feedback_weight_mode` 等；
  - 嵌套字段：`feedback_weight`；
  - 兼容别名：`corrupted_feedback`、`cw_c2ucb`。

这使 `w` 的实现更接近 CW-C2UCB 思想：不是简单固定权重，而是允许按信道反馈质量对更新强度做加权。

### 2.3 oracle 中部分关键超参数写死

原先 greedy oracle 中存在硬编码：

- `min_marginal_coverage`
- `oracle_cost_lambda`
- `explore_warmup_pulls_per_quant`
- `explore_warmup_pulls_per_rho`
- `explore_warmup_pulls_per_cache`
- `explore_bonus`

已修复为可从配置传入，并在 oracle 输出记录中保留这些参数，便于后续实验审计。

这不是把 oracle 改成论文中的严格组合优化 oracle，而是把当前工程中的近似 greedy oracle 做到“参数可控、行为可解释、实验可复现”。

### 2.4 no-send 分支的反馈通道语义不清

原逻辑中 no-send 的 `q_recv=0` 可能被 fallback 成 `loss_rate=1`，这会让“没有发送”被误解成“物理链路 100% 丢包”。

已修复：

- no-send pending item 保留原始 `channel_profile`；
- 原始物理丢包率写入 `physical_loss_rate`；
- 用于反馈权重的 `loss_rate/plr` 设置为 `0.0`；
- 增加 `feedback_source = "no_send_no_physical_transmission"`。

这样 no-send 被视作“动作没有触发物理传输”，而不是一次失败通信。

### 2.5 全 no-send 时 AP gain 被错误分配

原先如果一帧中所有 pending 项的 raw weight 都为 0，会统一 fallback 成均分，这会导致全 no-send 的动作也吃到 AP proxy gain。

已修复：

- 如果 pending 全部是 `no_send_update=True`，保持 raw weight 为 0；
- 不再把本帧 AP gain 均分给没有实际协作贡献的 no-send 动作；
- 如果不是全 no-send，仍保留原先的 uniform fallback，避免真实发送项因统计缺失完全无法学习。

## 3. 与论文/设计的对齐口径

当前修复后的状态可以这样表述：

- D-LinUCB：仍是工程化的 discounted LinUCB，不是逐公式复现论文完整 regret 设定，但折扣更新、上下文更新、在线自适应方向是一致的。
- CW-C2UCB：`w` 已经从“配置存在但可能没有实际进入策略”修复为“能够进入 `DiscountedLinUCB` 更新逻辑”，更符合 corrupted/weighted feedback 的思想。
- oracle：仍是预算约束下的 greedy 近似 oracle，不是严格枚举/最优组合 oracle。这个不对齐是工程可扩展性的合理简化，但现在关键超参数已经外置，可用于实验中说明和调参。
- no-send：现在区分“未发送”和“链路失败”，避免污染信道反馈学习。

## 4. 验证情况

已对修复后的 Python 文件执行语法编译检查：

```text
python -m py_compile: passed
```

本补丁为覆盖式源码补丁。将本目录中的 `opencood/...` 文件覆盖到当前工程对应路径即可应用。
