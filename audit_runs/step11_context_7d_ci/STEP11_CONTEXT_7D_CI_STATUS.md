# Step 11: 7D Context with CAV Local Confidence C_i

## 修改目标

将 C2MAB context 从原 6D：

```text
[B_norm, p_loss, d_norm, C_ego, q_cache_i, comp_i_ego]
扩展为 7D：

[B_norm, p_loss, d_norm, C_ego, q_cache_i, comp_i_ego, C_i]

其中 C_i 表示 CAV i 自身本地检测置信度，由 CAV 的 pre-fusion dense head 输出 psm_single 计算得到，属于动作前可观测信息，可以作为 context。

当前实现

新增模块：

opencood/comm/arce/c2mab_local_confidence.py

核心函数：

local_cav_confidences_from_psm(psm_single, topk=50)
get_cav_confidence(local_cav_confidences, cav_idx)

修改链路：

point_pillar_where2comm_arce.py
  psm_single -> local_cav_confidences

where2comm_arce_fuse.py
  local_cav_confidences -> _maybe_arce_comm()

arce_c2mab_comm.py
  local_cav_confidences -> context_builder.build(..., cav_confidence=C_i)

context_builder.py
  include_cav_confidence=True -> output 7D context
20 帧验证结果
model_context_dim = 7
include_cav_confidence = true
context_len_counter = {"7": 20}
policy_update_context_dim_counter = {"7": 40}
errors = []

C_i 统计：

cav_conf_count = 20
cav_conf_min = 0.3754979372024536
cav_conf_max = 0.39766258001327515
cav_conf_mean = 0.38743205517530444
当前判断

Step 11 已完成核心闭环验证：

C_i 已成功接入 context 第 7 维；
D-LinUCB 已按 7D context 正常运行；
当前 C_i 来自 CAV 自身 pre-fusion dense head，不存在 post-fusion/AP 泄漏；
当前 C_i 分布较窄，后续如需增强区分度，可将 top50 mean 调整为 top20/top10 mean 或 active-region mean。
