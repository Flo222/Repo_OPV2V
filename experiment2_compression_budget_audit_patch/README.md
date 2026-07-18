# 实验二：有限预算下的压缩有效性验证

## 实验目的

在不引入丢包、FEC 冗余、缓存或新压缩方法的条件下，验证：

```text
FP32 / FP16 / INT8 / INT4
→ 源字节数和 source packet 数不同
→ 同一总帧预算下，被裁掉的 source packet 数不同
→ 接收 feature 完整度不同
→ 融合检测结果可能不同
```

固定条件：

- `PLR = 0`
- `rho = 0`
- `cache = 0`
- Markov 状态关闭
- 所有非 ego 协作者都发送
- 总帧预算固定，并在当前帧的非 ego 协作者之间等分

## 本补丁修改内容

### 1. 真实 `frame_id` 进入内部审计记录

修改：

```text
opencood/tools/inference_arce.py
```

推理入口先从以下字段解析数据集标识：

```text
frame_id → timestamp → sample_idx → sample_id → loader sample_index
```

随后在模型 forward 前写入：

```python
batch_data["ego"]["frame_id"]
batch_data["ego"]["audit_sample_index"]
```

这些是只读元数据，模型层不会使用它们进行计算，不改变预测结果。

### 2. 分开统计 source 与 parity 的预算裁剪

修改：

```text
opencood/comm/arce/arce_fixed_comm.py
```

新增字段：

```text
actual_num_transmitted_source_packets
actual_num_transmitted_parity_packets
actual_transmitted_source_bytes
actual_transmitted_parity_bytes
num_source_dropped_by_budget
num_parity_dropped_by_budget
```

并检查：

```text
tx_source + dropped_source = source_packets
tx_parity + dropped_parity = parity_packets
dropped_source + dropped_parity = total_budget_drop
```

实验二固定 `rho=0`，所以 parity 相关字段应全部为 0；这些字段主要也为后续冗余实验保留。

### 3. 扩展只读审计器

修改：

```text
opencood/comm/arce/audit/compression_auditor.py
```

实验一保持原有严格检查；实验二允许预算裁剪和 `F_quant != F_recv`，但仍要求：

- 真实 frame id 非空；
- 请求量化模式等于实际模式；
- source/parity 预算记账完全一致；
- 无 Bernoulli 丢包；
- 无 FEC parity；
- INT4 使用真实 packing。

### 4. 新增实验二脚本

```text
scripts/run_experiment2_compression_budget_audit.sh
scripts/summarize_experiment2_compression_budget_audit.py
scripts/test_experiment2_budget_accounting_unit.py
```

所有运行配置都写入独立的 `audit_runs` 临时目录，不修改训练模型目录中的 `config.yaml` 或 checkpoint。

## 安装

该补丁基于实验一代码，因此先确认实验一补丁已经安装。

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

unzip experiment2_compression_budget_audit_patch.zip
bash experiment2_compression_budget_audit_patch/install.sh \
  /home/server/v2x_projects/OPV2V
```

安装器会：

1. 备份三个被修改的核心文件；
2. 应用增量 patch；
3. 安装三个实验二脚本；
4. 执行 Python 编译检查；
5. 重新执行实验一 smoke test；
6. 执行实验二 source/parity 预算记账 smoke test。

正常结尾：

```text
Experiment 1 audit installation smoke test passed.
source-drop accounting OK
parity-drop accounting OK
Experiment 2 budget-accounting smoke test passed.
Installed Experiment 2 compression-budget audit.
```

## 先跑 2 帧

默认预算为：

```text
SYSTEM_BUDGET_MBPS = 300 Mbps
TX_WINDOW_MS = 100 ms
```

对应总帧预算：

```text
3.75 MB/frame
```

它会在每帧所有非 ego 协作者之间平均分配。这是为了做机制验证，不代表最终真实信道配置。

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

export MODEL_DIR=/home/server/v2x_projects/OPV2V/opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/experiment2_compression_budget_test
export MAX_SAMPLES=2
export NUM_WORKERS=0
export SAVE_TENSORS=1
export SAVE_FIRST_N_LINKS=12
export SYSTEM_BUDGET_MBPS=300
export TX_WINDOW_MS=100

bash scripts/run_experiment2_compression_budget_audit.sh
```

## 跑 200 帧

```bash
export MODEL_DIR=/home/server/v2x_projects/OPV2V/opencood/logs/point_pillar_where2comm_arce_c2mab_comp_div
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/experiment2_compression_budget_200
export MAX_SAMPLES=200
export NUM_WORKERS=0
export SAVE_TENSORS=1
export SAVE_FIRST_N_LINKS=12
export SYSTEM_BUDGET_MBPS=300
export TX_WINDOW_MS=100

bash scripts/run_experiment2_compression_budget_audit.sh
```

## 输出

```text
audit_runs/experiment2_compression_budget_200/
├── fp32/
│   ├── audit/compression_budget_audit.jsonl
│   ├── audit/tensor_snapshots/
│   ├── comm_logs/
│   ├── inference.log
│   └── model_runtime/config.yaml
├── fp16/
├── int8/
├── int4/
├── experiment2_summary.json
└── experiment2_summary.csv
```

## 关键指标

每帧、每个 `sender → ego` 链路记录：

```text
frame_id
source_packet_count
num_transmitted_source_packets
num_source_dropped_by_budget
num_transmitted_parity_packets
num_parity_dropped_by_budget
source_tx_ratio
source_recovery_ratio
budget_transport_error.NMSE
end_to_end_error.NMSE
actual_transmitted_bytes
bandwidth_budget_bytes
```

### 两种误差

纯预算损伤：

```text
budget_transport_error
= F_quant 与 F_recv 的差异
```

端到端损伤：

```text
end_to_end_error
= F_src 与 F_recv 的差异
= 量化误差 + 预算裁剪损伤
```

## 预期趋势

在同一帧预算下通常应看到：

```text
FP32 → FP16 → INT8 → INT4
source packet 数下降
source budget drop 下降
source_tx_ratio 上升
source_recovery_ratio 上升
```

但 INT4 自身量化误差很大，所以最终 AP 不一定最高。实验二的核心不是证明“越压缩 AP 越高”，而是证明压缩是否真实减少预算裁剪。

## 查看结果

```bash
column -s, -t \
  /home/server/v2x_projects/OPV2V/audit_runs/experiment2_compression_budget_200/experiment2_summary.csv \
  | less -S
```

完整 JSON：

```bash
python -m json.tool \
  /home/server/v2x_projects/OPV2V/audit_runs/experiment2_compression_budget_200/experiment2_summary.json \
  | less
```
