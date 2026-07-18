# 实验一：纯压缩正确性审计补丁

## 目标

固定 `rho=0`、`cache=0`、`PLR=0` 和充分带宽，仅比较：

- FP32
- FP16
- INT8
- packed INT4

逐帧、逐 ego-sender 链路记录：

- 实际待传 payload 大小；
- 量化后有效字节；
- source packet 数与 padding；
- 量化 NMSE / cosine similarity；
- 无损信道后恢复 feature 是否与量化后 feature 一致；
- 请求量化模式是否等于实际执行模式；
- INT4 是否真实 packed。

审计功能默认关闭。只有临时运行配置中的 `arce.compression_audit.enabled=true` 时才执行。

## 安装

```bash
unzip experiment1_compression_audit_patch.zip
cd experiment1_compression_audit_patch
bash install.sh /home/server2/Repo_OPV2V-reward_test
```

安装脚本会：

1. 备份原 `arce_fixed_comm.py`；
2. 复制审计模块和运行脚本；
3. 执行 Python 编译检查；
4. 执行无需数据集和 checkpoint 的 smoke test。

## 运行

```bash
cd /home/server2/Repo_OPV2V-reward_test
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

MODEL_DIR=/path/to/your/where2comm_model_log \
OUT_ROOT=audit_runs/experiment1_compression_clean \
MAX_SAMPLES=200 \
NUM_WORKERS=0 \
SAVE_TENSORS=1 \
SAVE_FIRST_N_LINKS=12 \
bash scripts/run_experiment1_compression_audit.sh
```

脚本不会修改 `MODEL_DIR/config.yaml`。每种量化模式都会创建独立的临时 `model_runtime/config.yaml`，checkpoint 使用只读软链接。

## 输出

```text
audit_runs/experiment1_compression_clean/
├── fp32/
│   ├── audit/compression_audit.jsonl
│   ├── audit/tensor_snapshots/*.pt
│   ├── comm_logs/
│   ├── inference.log
│   └── model_runtime/config.yaml
├── fp16/
├── int8/
├── int4/
├── experiment1_summary.json
└── experiment1_summary.csv
```

## 通过条件

每种模式必须满足：

- 请求模式等于实际量化模式；
- `rho=0`，无 parity；
- 无预算裁剪；
- 无 Bernoulli 丢包；
- 无最终缺失 source packet；
- 通信后 compact feature 与量化后 feature 一致；
- INT4 使用 `packed_int4`。

跨模式应大致满足：

```text
有效字节：FP32 > FP16 > INT8 > INT4
量化误差：FP32 <= FP16 <= INT8 <= INT4
```

实际发送字节可能因 1024B packet padding 不严格单调，因此同时查看 `quantized_valid_stream_bytes` 与 `source_packet_count`。
