# 理想/Markov 信道每帧线路通信量统计

## 1. 主统计口径

主指标为：

```text
avg_total_tx_MB_per_frame
= 所有已评估帧的 transmitted_wire_bytes 总和 / 帧数 / 1,000,000
```

每帧把所有非 ego `sender -> ego` 链路相加。没有协作车或没有发送数据的帧仍保留在分母中。

输出字段：

- `source_payload_bytes`：基线完成自身压缩、掩码或编码后，分包前的有效负载。
- `transmitted_wire_bytes`：预算/FEC 后真正注入线路的数据量，包含固定包长填充；这是主通信开销。
- `received_wire_bytes`：丢包后成功到达的数据量。
- `budget_truncated_bytes`：源负载中因预算没有发送的部分。

默认网络单位使用十进制 MB：`1 MB = 1,000,000 bytes`。

## 2. 各基线理想信道取数点

- **V2X-ViT**：Transformer fusion 输入处，排除 ego、padding agent 和 3 个 prior channels；统计原生 compressor 后的协作特征。
- **Where2Comm**：原生 confidence mask 生成后，按真实二值 mask 统计选中位置；不再用 `dense_bytes * communication_rate` 估算。
- **CoSDH**：每个原生尺度的 communication mask 后统计；默认把同一 sender 的多尺度 payload 合并成一条链路字节流再分包。
- **CoopDiff**：只统计 student 分支各尺度进入 `fuse_modules` 前的非 ego 特征，不统计 teacher/蒸馏内部特征。
- **RoCooper**：统计进入 `comm_module` 前、完成 shrink/compression 后的非 ego 特征。

理想信道取数使用只读 PyTorch forward hooks，不修改张量和检测结果。

## 3. Markov 信道取数点

- V2X-ViT / Where2Comm：读取 ARCE fixed executor 的实际发送字节记录。
- CoSDH official fixed transport：读取 `transmitted_wire_bytes`。
- 旧 CoSDH / CoopDiff：本补丁新增预算后 payload 的固定包长统计字段。
- RoCooper：本补丁新增每条链路的 `wire_records`；分辨率压缩发生时按低分辨率特征计算发送量，后续 fading、block loss、delay、frame drop 不减少已经注入线路的发送量。

## 4. 稀疏位置元数据

Where2Comm 和理想 CoSDH 的原开源代码没有真正实现线路序列化，因此物理线路量必须明确位置元数据假设：

```text
--sparse_metadata indices  # 默认：每个选中位置 4-byte 线性索引
--sparse_metadata bitmask  # 每个 sender/scale 发送 H*W bit mask
--sparse_metadata none     # 仅统计特征值，接近很多论文的 feature-only 口径
```

推荐正式汇报使用 `indices`，并额外跑一遍 `none` 作为论文口径对照。不同实验必须使用相同设置。

## 5. 安装

```bash
cd /home/server/v2x_projects
unzip wire_bw_audit_patch_20260729.zip

conda activate opencood
bash wire_bw_audit_patch_20260729/install.sh \
  /home/server/v2x_projects/OPV2V
```

安装程序会：

1. 将原文件备份到项目根目录的 `.wire_bw_backup_时间戳/`；
2. 安装统一统计器、运行脚本和三个 Markov 记录修复；
3. 执行语法检查和 CPU 单元测试。

不需要新增 pip 依赖。

## 6. 单个基线一键统计

必须准备两个模型目录：

- `IDEAL_MODEL_DIR`：真正的 clean/ideal 配置；
- `MARKOV_MODEL_DIR`：真正启用固定 Markov 信道的配置。

`--channel_mode ideal` 只负责取数，**不会自动关闭模型目录里的 Markov 模块**，因此不能拿 Markov 模型目录冒充理想信道。

示例：

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

BASELINE=where2comm
DATASET=opv2v
IDEAL_MODEL_DIR=/path/to/where2comm_clean_log
MARKOV_MODEL_DIR=/path/to/where2comm_fixed_markov_log
OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/wire_bw/${DATASET}_${BASELINE}

PACKET_SIZE_BYTES=1024 \
SPARSE_METADATA=indices \
SEED=2026 \
bash scripts/run_wire_bw_pair.sh \
  "$BASELINE" "$DATASET" \
  "$IDEAL_MODEL_DIR" "$MARKOV_MODEL_DIR" \
  "$OUT_ROOT" 200
```

CoSDH、CoopDiff、RoCooper、V2X-ViT 只需替换 `BASELINE` 和两个模型目录：

```text
where2comm
v2xvit
rocooper
coopdiff
cosdh
```

若测试集路径需要覆盖：

```bash
TEST_DIR=/data/OPV2V/test \
bash scripts/run_wire_bw_pair.sh ...
```

若 checkpoint 不是自动选择最新数字 epoch：

```bash
IDEAL_EPOCH=20 MARKOV_EPOCH=20 \
bash scripts/run_wire_bw_pair.sh ...
```

## 7. 分开运行

理想信道：

```bash
python scripts/wire_bw_audit.py \
  --name opv2v_where2comm_ideal \
  --dataset opv2v \
  --baseline where2comm \
  --channel_mode ideal \
  --model_dir /path/to/clean_log \
  --epoch auto \
  --max_frames 200 \
  --packet_size_bytes 1024 \
  --sparse_metadata indices \
  --sparse_index_bytes 4 \
  --out_dir audit_runs/wire_bw/opv2v_where2comm/ideal
```

Markov 信道：

```bash
python scripts/wire_bw_audit.py \
  --name opv2v_where2comm_markov \
  --dataset opv2v \
  --baseline where2comm \
  --channel_mode markov \
  --model_dir /path/to/fixed_markov_log \
  --epoch auto \
  --max_frames 200 \
  --packet_size_bytes 1024 \
  --out_dir audit_runs/wire_bw/opv2v_where2comm/markov
```

Markov 模型应为固定策略/无 C2MAB 的基线。脚本默认检测到 C2MAB/DC2MAB 后退出；只有明确要统计策略模型时才加 `--allow_policy`。

## 8. 输出文件

每次运行产生：

```text
summary.json
per_frame_tx.csv
per_link_records.jsonl
record_sources.json
policy_audit.json
frame_errors.json
```

主要查看：

```text
summary.json -> avg_total_tx_MB_per_frame
```

`per_frame_tx.csv` 可用于画分布、均值、P95 或与 AP 对齐。配对脚本还会生成：

```text
wire_bw_pair_summary.csv
wire_bw_pair_summary.json
```

## 9. 公平比较检查

五个基线必须保持：

- 同一数据集 split；
- 同样的前 N 帧或完整 test；
- 同样的 `packet_size_bytes`；
- 同样的稀疏 metadata 口径；
- 同一 MB 定义；
- 理想与 Markov 分别使用正确模型目录；
- Markov 主指标使用 `transmitted_wire_bytes`，不能使用丢包后的 `received_wire_bytes`。
