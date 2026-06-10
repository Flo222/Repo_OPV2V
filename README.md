# Repo_OPV2V：Where2comm-ARCE / DC2MAB 协同感知通信鲁棒性项目

本仓库基于 OpenCOOD / OPV2V 框架，当前分支为：

```bash
dc2mab_pdf_strict
```

当前项目目标是：在自动驾驶多车协同目标检测中，以 **Where2comm** 作为基础协同感知方法，在其通信 mask 与 attention fusion 之间接入 **ARCE / DC2MAB 通信控制层**，研究不可靠 V2V 通信条件下的鲁棒协同感知。

当前实现的主线可以概括为：

```text
PointPillar Backbone
    ↓
Where2comm confidence map / communication mask
    ↓
ARCE / DC2MAB message transmission
    ↓
quantization / redundancy / packet loss / latency / recovery
    ↓
Where2comm attention fusion
    ↓
detection head
```

其中：

* **Where2comm**：决定“哪里值得通信”，生成空间通信 mask。
* **ARCE**：执行 packetization、量化、冗余、丢包、时延和恢复。
* **DC2MAB / C2MAB**：根据链路状态和感知上下文，动态选择每个协作者的通信动作。
* **Link-level Markov channel**：为每条 ego-to-CAV 链路生成 Good / Medium / Bad 三态通信状态。

---

## 1. 当前实现状态

当前分支已经从早期的 V2X-ViT 通信损伤实验，更新为 **Where2comm-based ARCE / DC2MAB** 实验框架。

主要已实现内容包括：

| 模块                                                    | 当前状态 |
| ----------------------------------------------------- | ---- |
| Where2comm 基础协同感知                                     | 已接入  |
| Where2comm mask 与 ARCE 对齐                             | 已实现  |
| ARCE fixed baseline                                   | 已实现  |
| ARCE random baseline                                  | 已实现  |
| DC2MAB / C2MAB 动态策略                                   | 已实现  |
| 36 维通信动作空间                                            | 已实现  |
| 6 维上下文建模                                              | 已实现  |
| ego-side greedy knapsack 选择                           | 已实现  |
| link-level Markov 信道                                  | 已实现  |
| Good / Medium / Bad 三态链路 profile                      | 已实现  |
| GE packet loss                                        | 已实现  |
| patch-level packetization                             | 已实现  |
| FP16 / INT8 / INT4 量化                                 | 已实现  |
| XOR / Raptor-like 冗余                                  | 已实现  |
| temporal cache / spatial interpolation / zero-fill 恢复 | 已实现  |
| 通信日志与状态级统计                                            | 已实现  |
| fixed / random / c2mab / ablation YAML                | 已实现  |

---

## 2. 当前核心方法

当前核心模型文件为：

```text
opencood/models/point_pillar_where2comm_arce.py
```

该模型的整体流程为：

```text
processed_lidar
    ↓
PillarVFE
    ↓
PointPillarScatter
    ↓
BaseBEVBackbone
    ↓
optional shrink_conv / compressor
    ↓
Where2comm confidence mask
    ↓
ARCE / C2MAB communication layer
    ↓
Where2comm attention fusion
    ↓
classification / regression heads
```

ARCE 的插入点在：

```text
Where2comm confidence mask
    ↓
masked message
    ↓
ARCE / C2MAB communication
    ↓
AttentionFusion
```

也就是说，当前实现并没有替换 Where2comm 的原始融合方式，而是在 Where2comm 选出的通信区域上进一步模拟和优化真实通信过程。

---

## 3. 当前主要目录结构

```text
Repo_OPV2V/
├── opencood/
│   ├── models/
│   │   ├── point_pillar_where2comm_arce.py
│   │   └── fuse_modules/
│   │       └── where2comm_arce_fuse.py
│   │
│   ├── comm/
│   │   ├── arce/
│   │   │   ├── arce_fixed_comm.py
│   │   │   ├── arce_c2mab_comm.py
│   │   │   ├── fixed_policy.py
│   │   │   ├── random_policy.py
│   │   │   └── policies/
│   │   │       ├── action_space.py
│   │   │       ├── context_builder.py
│   │   │       ├── discounted_linucb.py
│   │   │       ├── ego_greedy_oracle.py
│   │   │       ├── bandwidth_patch_selector.py
│   │   │       ├── complementarity.py
│   │   │       └── reward.py
│   │   │
│   │   ├── channel/
│   │   │   ├── channel_manager.py
│   │   │   ├── gilbert_elliott.py
│   │   │   ├── latency_model.py
│   │   │   └── markov_state.py
│   │   │
│   │   ├── fec/
│   │   ├── packet/
│   │   ├── recovery/
│   │   └── metrics/
│   │
│   ├── compression/
│   │   └── feature_quantizer.py
│   │
│   ├── hypes_yaml/
│   │   └── where2comm_arce_final/
│   │       ├── point_pillar_where2comm_arce_fixed.yaml
│   │       ├── point_pillar_where2comm_arce_random.yaml
│   │       ├── point_pillar_where2comm_arce_c2mab.yaml
│   │       ├── point_pillar_where2comm_arce_c2mab_comp.yaml
│   │       ├── point_pillar_where2comm_arce_c2mab_comp_div.yaml
│   │       ├── point_pillar_where2comm_arce_ablate_no_cache.yaml
│   │       ├── point_pillar_where2comm_arce_ablate_no_comp.yaml
│   │       ├── point_pillar_where2comm_arce_ablate_no_div.yaml
│   │       └── point_pillar_where2comm_arce_ablate_no_red.yaml
│   │
│   └── tools/
│       ├── inference_arce.py
│       ├── summarize_comm_logs.py
│       ├── summarize_c2mab_logs.py
│       ├── summarize_final_alignment_metrics.py
│       ├── export_statewise_eval_index.py
│       └── validate_where2comm_arce_config.py
│
├── scripts/
│   ├── run_where2comm_one_sample_check.sh
│   ├── run_where2comm_markov_main.sh
│   ├── run_where2comm_ablation.sh
│   └── run_extension_backbones.sh
│
└── README.md
```

---

## 4. 当前实验配置

最终实验配置主要位于：

```text
opencood/hypes_yaml/where2comm_arce_final/
```

推荐使用的配置如下：

| YAML                                                | 含义                     |
| --------------------------------------------------- | ---------------------- |
| `point_pillar_where2comm_arce_fixed.yaml`           | 固定通信动作基线               |
| `point_pillar_where2comm_arce_random.yaml`          | 随机通信动作基线               |
| `point_pillar_where2comm_arce_c2mab.yaml`           | 基础 DC2MAB / C2MAB 动态策略 |
| `point_pillar_where2comm_arce_c2mab_comp.yaml`      | 加入互补性项的 C2MAB 变体       |
| `point_pillar_where2comm_arce_c2mab_comp_div.yaml`  | 加入互补性与多样性约束的完整版本       |
| `point_pillar_where2comm_arce_ablate_no_cache.yaml` | 去掉 temporal cache 的消融  |
| `point_pillar_where2comm_arce_ablate_no_comp.yaml`  | 去掉互补性项的消融              |
| `point_pillar_where2comm_arce_ablate_no_div.yaml`   | 去掉多样性约束的消融             |
| `point_pillar_where2comm_arce_ablate_no_red.yaml`   | 去掉冗余机制的消融              |

推荐主实验对比顺序：

```text
Fixed
Random
C2MAB
C2MAB + Complementarity
C2MAB + Complementarity + Diversity
```

推荐消融实验：

```text
Ours full
No cache
No complementarity
No diversity
No redundancy
```

---

## 5. 当前通信设置

### 5.1 三态链路状态

当前使用三种通信状态：

| 状态     |      带宽 | 丢包率 |       时延范围 | jitter 范围 | 含义   |
| ------ | ------: | --: | ---------: | --------: | ---- |
| Good   | 27 Mbps |  3% |   15–25 ms |    0–5 ms | 较好链路 |
| Medium |  5 Mbps | 12% |  80–100 ms |   5–15 ms | 中等链路 |
| Bad    |  1 Mbps | 28% | 200–400 ms |  15–40 ms | 较差信道 |

这些状态会影响：

* 每条链路的带宽预算；
* packet loss；
* latency；
* stale penalty；
* DC2MAB 上下文；
* patch 选择和可行动作集合。

---

### 5.2 Link-level Markov 信道

当前数据集阶段使用 link-level Markov channel。

也就是说，每条 ego-to-CAV 链路都有独立的 Markov 状态链：

```text
scenario_i / ego_to_cav_j
    ↓
Good / Medium / Bad
    ↓
delay_slots / channel_state_id / delay_ms
```

当前数据集侧 Markov 转移矩阵为：

| 当前状态   | Good | Medium |  Bad |
| ------ | ---: | -----: | ---: |
| Good   | 0.85 |   0.13 | 0.02 |
| Medium | 0.10 |   0.80 | 0.10 |
| Bad    | 0.03 |   0.17 | 0.80 |

状态 ID 约定：

```text
-1: ego_or_padding
 0: good
 1: medium
 2: bad
```

注意：部分 YAML 中同时存在 `wild_setting.channel_state_markov` 和 `model.args.arce.channel` 两套 Markov 配置。当前实际数据加载阶段的链路状态主要由 `wild_setting.channel_state_markov` 注入；ARCE / C2MAB 再读取该链路状态进行通信决策。后续修改参数时，建议保持两处状态 profile 和转移矩阵一致，避免实验解释混乱。

---

### 5.3 时延模型

当前 ARCE 时延估计采用：

```text
d_total = d_tx + jitter + d_proc
```

其中：

```text
d_tx = 8 × transmitted_bytes / (bandwidth_mbps × 1e6) × 1000
```

默认关键参数：

| 参数                   |        当前设置 |
| -------------------- | ----------: |
| 感知频率                 |       10 Hz |
| frame interval       |      100 ms |
| tx window / deadline |      100 ms |
| processing delay     |       10 ms |
| stale threshold      |      300 ms |
| late policy          | allow_stale |

因此，当前不是简单固定延迟，而是结合：

```text
实际传输字节数
    × 当前链路带宽
    + jitter
    + 处理开销
```

来估计通信时延。

---

### 5.4 Packet / patch 切分

当前采用 patch-level packetization：

| 参数                        | 当前设置            |
| ------------------------- | --------------- |
| packetizer mode           | block           |
| block size                | 4 × 4           |
| pad boundary              | true            |
| align_to_where2comm_mask  | true            |
| patch source              | where2comm_mask |
| mask threshold            | 0.05            |
| patch selector            | score_per_byte  |
| ranking                   | score_per_byte  |
| enforce_link_budget       | true            |
| min effective patch ratio | 0.2             |
| min patch count           | 1               |
| metadata bytes per packet | 8               |

因此，当前通信不是默认传完整 feature map，而是：

```text
Where2comm 先生成通信 mask
    ↓
ARCE 只在 mask 对应的有效空间区域内选择 patch
    ↓
根据链路带宽预算选择单位字节价值最高的 patch
    ↓
仅对 selected patches 做量化、冗余、丢包和恢复
```

---

### 5.5 量化设置

当前 ARCE 动作空间支持三种量化模式：

| 量化模式 | 相对 FP32 大小 |
| ---- | ---------: |
| FP16 |        1/2 |
| INT8 |        1/4 |
| INT4 |        1/8 |

在动作成本估计中，通信字节数近似为：

```text
transmitted_bytes = raw_bytes × compression_ratio × (1 + redundancy_ratio)
```

---

### 5.6 冗余 / FEC 设置

当前冗余率为：

```text
rho ∈ {0.0, 0.25, 0.5}
```

含义：

|  rho | 含义     |
| ---: | ------ |
|  0.0 | 无额外冗余  |
| 0.25 | 25% 冗余 |
|  0.5 | 50% 冗余 |

当前主方法中，`rho > 0` 默认可映射到 Raptor-like / fountain-style 冗余；在 fixed baseline 或 FEC 消融中也可以使用 XOR FEC。

Fixed baseline 当前常用固定动作：

```yaml
send: 1
quant: int8
rho: 0.25
cache: 1
fec_type: xor
```

---

### 5.7 恢复机制

当前恢复机制包括：

| 恢复方式                  | 当前状态    |
| --------------------- | ------- |
| temporal cache        | enabled |
| spatial interpolation | enabled |
| zero fill             | enabled |
| temporal fusion       | enabled |
| delay penalty         | enabled |

恢复优先级可理解为：

```text
FEC decode
    ↓
temporal cache
    ↓
spatial interpolation
    ↓
zero-fill
```

其中 temporal fusion 当前使用：

```text
beta = 5.0
tau_stale = 300 ms
```

Bad 状态下如果 packet 丢失或延迟过大，不是简单丢弃整张 feature，而是先尝试冗余恢复，再尝试历史缓存和空间插值，最后才 zero-fill。

---

## 6. DC2MAB / C2MAB 设置

### 6.1 36 维动作空间

当前最终动作空间为：

```text
a = (send, quant, rho, cache)
```

具体为：

```text
send  ∈ {0, 1}
quant ∈ {fp16, int8, int4}
rho   ∈ {0.0, 0.25, 0.5}
cache ∈ {0, 1}
```

所以动作空间大小为：

```text
2 × 3 × 3 × 2 = 36
```

解释：

| 动作维度  | 含义                    |
| ----- | --------------------- |
| send  | 是否发送当前 CAV 的信息        |
| quant | 当前链路使用的量化精度           |
| rho   | 当前链路使用的冗余率            |
| cache | 是否允许使用 temporal cache |

未被 ego-side oracle 选中的 CAV 会执行 no-send fallback。

---

### 6.2 6 维上下文

当前 C2MAB 使用 6 维上下文：

```text
x_i^t = [B_norm, p_loss, d_norm, C_ego, q_cache, comp_i_ego]
```

对应含义：

| 维度           | 含义                  |
| ------------ | ------------------- |
| `B_norm`     | 归一化带宽               |
| `p_loss`     | 当前链路丢包率             |
| `d_norm`     | 归一化时延               |
| `C_ego`      | ego 当前检测置信度         |
| `q_cache`    | 当前 CAV 的缓存质量        |
| `comp_i_ego` | 当前 CAV 与 ego 的空间互补性 |

上下文会被裁剪到 `[0, 1]` 范围内。

---

### 6.3 C2MAB 策略

当前 C2MAB 采用：

```text
per-CAV Discounted LinUCB
    ↓
每个 CAV 生成候选动作 proposal
    ↓
ego-side greedy knapsack oracle
    ↓
在全局预算下选择 CAV-action pair
    ↓
未选中的 CAV 执行 no-send
```

主要参数：

| 参数               |              当前设置 |
| ---------------- | ----------------: |
| context dim      |                 6 |
| ridge lambda     |               1.0 |
| discount         |              0.97 |
| exploration beta |               1.0 |
| reward type      |       final_proxy |
| feedback         | link_proxy_reward |
| q_min            |               0.3 |
| max budget CAVs  |                 4 |
| budget scope     |   global_sum_link |

---

## 7. 安装与环境

### 7.1 克隆当前分支

```bash
git clone -b dc2mab_pdf_strict https://github.com/Flo222/Repo_OPV2V.git
cd Repo_OPV2V
```

### 7.2 创建环境

推荐使用仓库中的环境文件：

```bash
conda env create -f environment.yml
conda activate opencood
```

如果已有 OpenCOOD 环境，也可以补装依赖：

```bash
pip install -r requirements.txt
python setup.py develop
```

---

## 8. 数据集路径

当前 YAML 默认数据路径为：

```yaml
root_dir: /data/opv2x/train
validate_dir: /data/opv2x/validate
```

如果本机数据集路径不同，需要修改：

```text
opencood/hypes_yaml/where2comm_arce_final/*.yaml
```

中的：

```yaml
root_dir:
validate_dir:
```

当前配置默认：

```yaml
max_cav: 5
batch_size: 1
epoches: 50
```

---

## 9. 训练命令

### 9.1 Fixed baseline

```bash
python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/where2comm_arce_final/point_pillar_where2comm_arce_fixed.yaml
```

### 9.2 Random baseline

```bash
python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/where2comm_arce_final/point_pillar_where2comm_arce_random.yaml
```

### 9.3 C2MAB baseline

```bash
python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/where2comm_arce_final/point_pillar_where2comm_arce_c2mab.yaml
```

### 9.4 Ours full

```bash
python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/where2comm_arce_final/point_pillar_where2comm_arce_c2mab_comp_div.yaml
```

### 9.5 消融实验

```bash
python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/where2comm_arce_final/point_pillar_where2comm_arce_ablate_no_cache.yaml

python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/where2comm_arce_final/point_pillar_where2comm_arce_ablate_no_comp.yaml

python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/where2comm_arce_final/point_pillar_where2comm_arce_ablate_no_div.yaml

python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/where2comm_arce_final/point_pillar_where2comm_arce_ablate_no_red.yaml
```

---

## 10. 推理与日志

### 10.1 单模型推理

```bash
python opencood/tools/inference_arce.py \
  --model_dir opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div \
  --fusion_method intermediate \
  --save_comm \
  --comm_log_dir opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div/final_alignment_markov_seed2026 \
  --num_workers 4
```

### 10.2 单样本检查

```bash
bash scripts/run_where2comm_one_sample_check.sh \
  opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div
```

该脚本会执行：

```text
max_samples = 1
save_comm = true
comm_log_dir = <MODEL_DIR>/final_alignment_one_sample
```

适合检查：

* 模型能否正常 forward；
* ARCE 是否被启用；
* 通信日志是否生成；
* 每条链路是否存在 state / delay / loss / action 记录。

---

### 10.3 主实验批量推理

```bash
bash scripts/run_where2comm_markov_main.sh
```

脚本默认评估：

```text
point_pillar_where2comm_arce_fixed
point_pillar_where2comm_arce_random
point_pillar_where2comm_arce_c2mab
point_pillar_where2comm_arce_c2mab_comp
point_pillar_where2comm_arce_c2mab_comp_div
```

注意：运行前需要检查脚本中的 `MODEL_DIRS` 是否与本机训练输出目录一致。

---

### 10.4 消融实验批量推理

```bash
bash scripts/run_where2comm_ablation.sh
```

脚本默认评估：

```text
point_pillar_where2comm_arce_ablate_no_comp
point_pillar_where2comm_arce_ablate_no_div
point_pillar_where2comm_arce_ablate_no_cache
point_pillar_where2comm_arce_ablate_no_red
point_pillar_where2comm_arce_ours_full
```

同样需要先确认 `MODEL_DIRS` 路径是否存在。

---

## 11. 输出文件说明

推理时如果使用：

```bash
--save_comm
```

会保存通信日志。常见输出目录为：

```text
<MODEL_DIR>/final_alignment_markov_seed2026/
```

或：

```text
<MODEL_DIR>/final_alignment_one_sample/
```

主要输出文件包括：

| 文件                           | 含义                              |
| ---------------------------- | ------------------------------- |
| `arce_comm_records.jsonl`    | 每条链路的完整嵌套通信记录                   |
| `arce_comm_flat.jsonl`       | 每条链路的扁平化通信指标                    |
| `arce_comm_flat.csv`         | 方便统计和画图的 CSV                    |
| `arce_comm_summary.json`     | 总体通信 summary                    |
| `final_metrics_summary.json` | 最终对齐后的通信指标汇总                    |
| `statewise_eval_index.csv`   | 按 Good / Medium / Bad 状态划分的评估索引 |

---

## 12. 日志汇总命令

### 12.1 汇总 final alignment 指标

```bash
python opencood/tools/summarize_final_alignment_metrics.py \
  --comm-jsonl opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div/final_alignment_markov_seed2026/arce_comm_flat.jsonl \
  --out opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div/final_alignment_markov_seed2026/final_metrics_summary.json
```

### 12.2 导出状态级评估索引

```bash
python opencood/tools/export_statewise_eval_index.py \
  --comm-jsonl opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div/final_alignment_markov_seed2026/arce_comm_flat.jsonl \
  --out opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div/final_alignment_markov_seed2026/statewise_eval_index.csv
```

### 12.3 汇总普通通信日志

```bash
python opencood/tools/summarize_comm_logs.py \
  --log_dir opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div/final_alignment_markov_seed2026 \
  --all_outputs \
  --skip_bypassed
```

---

## 13. 推荐实验流程

建议按以下顺序检查和运行：

```text
1. 检查数据集路径 root_dir / validate_dir
2. 先跑 one-sample check
3. 确认 arce_comm_flat.jsonl 正常生成
4. 检查 link-level channel_state / delay_ms / delay_slots
5. 检查 action_id / quant / rho / cache / send
6. 跑 fixed / random / c2mab 小样本
7. 跑完整 validation set
8. 导出 final_metrics_summary.json
9. 导出 statewise_eval_index.csv
10. 汇总 AP、通信开销、状态级性能和消融结果
```

推荐单样本检查命令：

```bash
bash scripts/run_where2comm_one_sample_check.sh \
  opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div
```

推荐完整主实验命令：

```bash
bash scripts/run_where2comm_markov_main.sh
```

推荐完整消融实验命令：

```bash
bash scripts/run_where2comm_ablation.sh
```

---

## 14. 当前方法与旧版本差异

与早期版本相比，当前分支有以下关键变化：

| 项目     | 旧版本                             | 当前版本                                                     |
| ------ | ------------------------------- | -------------------------------------------------------- |
| 基础协同方法 | V2X-ViT / transformer fusion    | Where2comm                                               |
| 通信插入点  | fusion 前直接模拟通信                  | Where2comm mask 与 attention fusion 之间                    |
| 通信区域   | 更接近整张 feature 或粗粒度 feature      | Where2comm mask 对齐的 selected patches                     |
| 信道状态   | fixed Good / Medium / Bad sweep | link-level Markov 三态                                     |
| 丢包方式   | 简化 packet loss                  | GE packet loss + FEC decode                              |
| 动作策略   | fixed 为主                        | fixed / random / DC2MAB                                  |
| 动作空间   | 手动配置                            | 36 维 final action space                                  |
| 决策粒度   | 状态级或链路级固定策略                     | 每个 CAV、每条链路、每帧动态决策                                       |
| 上下文    | 无或较弱                            | 6 维通信-感知上下文                                              |
| 互补性    | 未显式建模                           | 支持 ego-CAV mask complementarity                          |
| 恢复机制   | 简化补全                            | cache / interpolation / zero-fill                        |
| 日志     | 普通通信统计                          | link / frame / state / action / patch / FEC / quality 记录 |

---

## 15. 当前汇报口径

推荐在论文或汇报中将当前方法描述为：

```text
Where2comm-based ARCE / DC2MAB for robust collaborative perception under unreliable V2V communication.
```

中文表述：

```text
面向不可靠车车通信的 Where2comm 协同感知鲁棒传输优化方法。
```

核心思想：

```text
Where2comm 决定哪里值得传；
ARCE 决定如何稳健地传；
DC2MAB 根据链路状态和感知上下文动态决定传不传、传多少、用什么量化精度、加多少冗余、是否使用历史缓存。
```

---

## 16. 注意事项

1. 当前根目录旧 README 可能仍包含 V2X-ViT 和 “C2MAB 尚未实现” 的旧描述，应以本 README 为准更新。
2. 当前 YAML 中可能同时存在 dataset-level Markov 和 ARCE-level Markov 配置，修改实验设置时应保持一致。
3. `scripts/*.sh` 中的 `MODEL_DIRS` 是示例路径，运行前需要改成本机实际训练输出目录。
4. 如果只想做快速检查，优先使用 `run_where2comm_one_sample_check.sh`，不要一开始直接跑完整验证集。
5. 如果要对比 Good / Medium / Bad 状态下性能，建议使用 `statewise_eval_index.csv` 和通信日志中的 `channel_state` 字段做状态级统计。
6. 如果要写论文或汇报，建议强调当前方法不是替代 Where2comm，而是作为 Where2comm 的通信鲁棒增强层。
