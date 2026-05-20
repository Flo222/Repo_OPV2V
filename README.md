# Repo_OPV2V：ARCE 通信鲁棒协同感知项目说明

> 当前项目目标：在 OpenCOOD / OPV2V 的 V2X-ViT 中接入 ARCE（Adaptive Rate–Redundancy Communication Enhancement）通信链路，用于研究不可靠信道下的协同感知鲁棒性。当前实现以固定策略 ARCE 为主，覆盖固定 Good / Medium / Bad 信道、Gilbert–Elliott 丢包、feature packetization、量化压缩、FEC、partial reconstruction、通信日志与信道 sweep。

---

## 1. 当前实现状态

当前仓库已经形成以下模块：

```text
Repo_OPV2V/
├── opencood/
│   ├── comm/
│   │   ├── channel/        # 固定信道、GE 丢包、时延模型、ChannelManager
│   │   ├── packet/         # feature packetization 与通信大小估计
│   │   ├── fec/            # None / XOR / Raptor-like FEC
│   │   ├── recovery/       # zero-fill / spatial interpolation / temporal cache / partial reconstruction
│   │   ├── arce/           # 固定 ARCE 策略与 ARCEFixedComm 主链路
│   │   └── metrics/        # 通信指标统计与日志保存
│   ├── compression/        # FP16 / INT8 / INT4 量化工具
│   ├── models/
│   │   └── point_pillar_transformer_opv2v_arce.py
│   ├── hypes_yaml/
│   │   └── point_pillar_v2xvit_opv2v_arce.yaml
│   └── tools/
│       ├── inference_arce.py
│       ├── summarize_comm_logs.py
│       └── eval_arce_channel_sweep.py
```

目前：

```text
阶段 A：V2X-ViT / OPV2V 基线搭好
阶段 B：固定策略 ARCE 通信链路搭好
阶段 C：通信日志与 Good / Medium / Bad sweep 工具搭好
阶段 D：C2MAB / Discounted LinUCB 动态策略尚未实现
```

---

## 2. 检查

### 2.1 已基本一致的部分

| PDF 设计点 | 当前代码对应模块 | 状态 |
|---|---|---|
| 固定 Good / Medium / Bad 信道 | `opencood/comm/channel/fixed_channel.py`、`channel_manager.py` | 已设计 |
| Gilbert–Elliott packet loss | `opencood/comm/channel/gilbert_elliott.py` | 已设计 |
| 随机时延 / 100 ms deadline / late policy | `opencood/comm/channel/latency_model.py`、`arce_fixed_comm.py` | 已设计 |
| feature patch / packetization | `opencood/comm/packet/packetizer.py` | 已设计 |
| FP16 / INT8 / INT4 量化 | `opencood/compression/feature_quantizer.py` | 已设计 |
| FEC: None / XOR / Raptor-like | `opencood/comm/fec/` | 已设计 |
| partial reconstruction: temporal / spatial / zero-fill | `opencood/comm/recovery/` | 已设计 |
| 每帧每链路通信日志 | `opencood/comm/metrics/comm_logger.py` | 已设计 |
| Good / Medium / Bad sweep | `opencood/tools/eval_arce_channel_sweep.py` | 已设计 |

### 2.2 与 未完成的部分

| PDF 设计点 | 当前状态 | 说明 |
|---|---|---|
| C2MAB / Discounted LinUCB 动态策略 | 未实现 | 当前只有 fixed policy；PDF 中的上下文空间、动作空间、proxy reward、UCB 选择尚未落地。 |
| Multi-CAV 背包式联合选择 | 未实现 | 当前 `ARCEFixedComm` 是逐 link 固定策略，尚未做全局带宽预算下的联合选择。 |
| 时序感知融合公式 `alpha_t * current + (1-alpha_t) * cache` | 部分实现 | 当前 temporal cache 主要是补缺失 packet，不是完整的质量加权融合。 |
| 标准 Raptor / RaptorQ | 部分实现 | 当前是真实 XOR fountain / Raptor-like peeling decoder，不是 RFC 标准 RaptorQ。 |
| PDF 中完整实验矩阵 | 部分实现 | 当前有通信 sweep 和日志，但完整 baseline 对比、C2MAB、AP 表格仍需实验脚本补全。 |

---

## 3. 当前项目的完整流程

### 3.1 模型主流程

`point_pillar_transformer_opv2v_arce.py` 的目标流程是：

```text
processed_lidar
→ PillarVFE
→ PointPillarScatter
→ BaseBEVBackbone
→ shrink_conv / optional naive compressor
→ ARCE communication layer
→ regroup to [B, max_cav, C, H, W]
→ concatenate prior_encoding
→ V2XTransformer fusion
→ classification / regression heads
```

ARCE 插入点在 V2X-ViT fusion 前，对每个非 ego agent 的 BEV intermediate feature 做通信模拟。

### 3.2 ARCE 通信链路流程

对一个非 ego feature `F ∈ R^{C×H×W}`，`ARCEFixedComm` 的目标流程是：

```text
1. ChannelManager.step()
   得到当前信道状态：Good / Medium / Bad

2. FixedARCEPolicy.select()
   根据信道状态选择动作：
   - quant_mode: fp16 / int8 / int4
   - fec_type: none / xor / raptor_sim
   - redundancy_ratio
   - recovery priority

3. FeaturePacketizer.packetize()
   [C, H, W] -> K 个 source packets

4. FeatureQuantizer.quantize_packets()
   FP32 feature packets -> FP16 / INT8 / INT4 packets

5. FEC.encode()
   K source packets -> N encoded packets
   N = K + parity / repair packets

6. ChannelManager.estimate_latency()
   根据 transmitted bytes、bandwidth、jitter 估算 total delay

7. ChannelManager.sample_packet_loss()
   对 N 个 encoded packets 采样 Gilbert–Elliott packet loss

8. late policy
   如果 late_policy = cache_only / drop 且 total_delay > deadline，当前帧整条消息视为丢失

9. FEC.decode()
   根据收到的 encoded packets 恢复 K 个 source packets

10. dequantize()
   量化 packets -> float packets

11. PartialReconstructor.recover_packets()
   temporal cache -> spatial interpolation -> zero-fill

12. FeaturePacketizer.unpacketize()
   K 个 recovered source packets -> [C, H, W]

13. 记录通信日志
   channel、action、packetization、quantization、FEC、recovery、size、latency
```

### 3.3 日志与统计流程

```text
inference_arce.py
→ 运行普通 OpenCOOD inference
→ 从模型中的 ARCEFixedComm 收集 records
→ CommLogger 保存 records/jsonl/csv/summary

summarize_comm_logs.py
→ 离线读取 arce_comm_records.jsonl 或 arce_comm_flat.jsonl
→ 生成 summary_from_logs.json / csv / markdown report

eval_arce_channel_sweep.py
→ 分别调用 inference_arce.py 跑 good / medium / bad
→ 每个信道单独保存 logs
→ 生成 sweep_summary.csv / sweep_report.md
```

---

## 4. 运行方式

### 4.1 训练无通信 / ARCE wrapper 基线

当前 YAML 默认 `arce.enabled: false`，因此可用于检查 OPV2V + V2X-ViT wrapper 是否正常。

```bash
python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_v2xvit_opv2v_arce.yaml
```

### 4.2 单信道 ARCE 推理

```bash
python opencood/tools/inference_arce.py \
  --model_dir opencood/logs/你的模型目录 \
  --fusion_method intermediate \
  --save_comm \
  --arce_channel_state medium
```

调试 5 个样本：

```bash
python opencood/tools/inference_arce.py \
  --model_dir opencood/logs/你的模型目录 \
  --fusion_method intermediate \
  --save_comm \
  --arce_channel_state medium \
  --max_samples 5
```

### 4.3 Good / Medium / Bad sweep

```bash
python opencood/tools/eval_arce_channel_sweep.py \
  --model_dir opencood/logs/你的模型目录 \
  --fusion_method intermediate \
  --states good medium bad \
  --skip_bypassed_comm
```

快速调试：

```bash
python opencood/tools/eval_arce_channel_sweep.py \
  --model_dir opencood/logs/你的模型目录 \
  --fusion_method intermediate \
  --states medium \
  --max_samples 5 \
  --overwrite
```

### 4.4 离线汇总通信日志

```bash
python opencood/tools/summarize_comm_logs.py \
  --model_dir opencood/logs/你的模型目录 \
  --all_outputs \
  --skip_bypassed
```

如果是 sweep 目录中的某个信道：

```bash
python opencood/tools/summarize_comm_logs.py \
  --log_dir opencood/logs/你的模型目录/arce_channel_sweep/medium_repeat00_seed0/comm_logs \
  --all_outputs \
  --skip_bypassed
```

---

## 5. 输出文件说明

### 5.1 `inference_arce.py` 输出

默认通信日志目录：

```text
<model_dir>/arce_comm_logs/
```

主要文件：

```text
arce_comm_records.jsonl         # 每条 link 的完整嵌套通信记录
arce_comm_flat.jsonl            # 每条 link 的扁平指标
arce_comm_flat.csv              # 方便画图的 CSV
arce_comm_summary.json          # 总体简洁 summary
arce_comm_summary_full.json     # by_frame / by_channel / by_quant / by_fec 分组 summary
```

### 5.2 `eval_arce_channel_sweep.py` 输出

默认目录：

```text
<model_dir>/arce_channel_sweep/
```

主要文件：

```text
sweep_runs.jsonl
sweep_summary.json
sweep_summary.csv
sweep_state_summary.csv
sweep_report.md
```

其中最常用的是：

```text
sweep_state_summary.csv
```

它按 Good / Medium / Bad 汇总通信指标。

---

## 6. 推荐实验顺序

按下面顺序推进：

```text
1. py_compile 检查所有新增 Python 文件
2. arce.enabled=false 跑 1~5 个样本，确认 V2X-ViT wrapper 正常
3. arce.enabled=true + mode=bypass，确认插入 ARCE 后不改变 shape
4. medium 信道 + max_samples=5，确认通信链路能闭环
5. 检查 arce_comm_records.jsonl 和 summary
6. 跑 good / medium / bad 小样本 sweep
7. 跑完整验证集 sweep
8. 汇总 AP 与通信开销
```

---

## 9. 当前结论

当前项目已经完成了固定策略 ARCE 的主要工程骨架，和设计 PDF 中的 ARCE 通信链路部分基本对应：信道建模、GE 丢包、packetization、量化、FEC、partial reconstruction、通信统计与信道 sweep 都已经有文件承载。

但还不是 PDF 中完整方法的最终实现，主要差距是：

```text
1. C2MAB / Discounted LinUCB 动态策略尚未实现；
2. Multi-CAV 联合带宽约束选择尚未实现；
3. temporal cache 目前偏补洞，不是完整时序质量加权融合；
4. 当前 Raptor 是 Raptor-like fountain code，不是标准 RaptorQ；
5. YAML 默认关闭 ARCE，需要补全 ARCE 配置；
6. GitHub 当前文件疑似换行损坏，需要先 py_compile 修复；
7. 模型 wrapper 与 ARCEFixedComm 的调用接口需要对齐。
```

因此，当前项目定位为：

```text
OPV2V / V2X-ViT 上的固定策略 ARCE 通信鲁棒性实验框架。
```

而不是完整的：

```text
ARCE + C2MAB 在线自适应策略最终版。
```
