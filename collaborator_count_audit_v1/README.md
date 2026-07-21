# OPV2V / V2X-Real 测试集协作者数量统计

## 统计定义

正式统计采用模型运行时口径：

```text
有效协作者数 = record_len - 1
```

`record_len` 是 OpenCOOD 数据加载器完成通信距离、`max_cav`、样本有效性等处理后，真正进入模型的车辆总数。减去 ego 后得到协作者数。

## 安装

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

unzip -o collaborator_count_audit_v1.zip
bash collaborator_count_audit_v1/install.sh /home/server/v2x_projects/OPV2V
```

## 跑完整测试集

```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}

export PROJECT_ROOT=/home/server/v2x_projects/OPV2V
export OUT_ROOT=/home/server/v2x_projects/OPV2V/audit_runs/test_split_collaborator_counts
export NUM_WORKERS=0
export SEED=2026

bash collaborator_count_audit_v1/run_all.sh
```

默认代表配置：

```text
OPV2V:
/home/server/v2x_projects/OPV2V/opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0

V2X-Real:
/home/server/v2x_projects/OPV2V/opencood/logs/point_pillar_where2comm_v2xreal_markov_eval
```

## 输出

```text
test_split_collaborator_counts/
├── dataset_comparison.csv
├── opv2v/
│   ├── summary.json
│   ├── frames.csv
│   └── collaborator_distribution.csv
└── v2x_real/
    ├── summary.json
    ├── frames.csv
    └── collaborator_distribution.csv
```

`summary.json` 包含：测试集总帧数、有/无协作者帧数、协作者总链路数、每帧平均协作者数、有协作者帧平均协作者数、最大协作者数，以及0/1/2/3…个协作者的帧数分布。

## 打包

```bash
cd /home/server/v2x_projects/OPV2V/audit_runs
tar -czf test_split_collaborator_counts.tar.gz test_split_collaborator_counts
```
