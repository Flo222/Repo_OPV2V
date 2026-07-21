# 十个基线批量特征/预算审计

本包配合 `baseline_feature_budget_audit_v1.zip` 使用。

## 已修正的路径
- OPV2V CoopDiff 从 `.../config.yaml` 规范为其父日志目录。
- V2X-Real 五个路径补全 `/home/server/v2x_projects/OPV2V/opencood/logs/` 前缀。

## 1. 先安装基础审计器
```bash
cd /home/server/v2x_projects/OPV2V
unzip -o baseline_feature_budget_audit_v1.zip
bash baseline_feature_budget_audit_v1/install.sh /home/server/v2x_projects/OPV2V
```

## 2. 解压本包并校验路径
```bash
unzip -o baseline_feature_budget_batch_v2.zip
cd baseline_feature_budget_batch_v2
python validate_baseline_paths.py baselines.tsv path_validation.json
```

## 3. 批量发现通信hook
```bash
cd /home/server/v2x_projects/OPV2V
conda activate opencood
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PROJECT_ROOT=$(pwd)
export OUT_ROOT=$(pwd)/audit_runs/baseline_tx_hook_discovery
bash baseline_feature_budget_batch_v2/run_discover_all.sh
```

每个基线失败不会中断其他基线。结果：
- `discovery_status.tsv`
- `all_top_candidates.csv`
- 每个基线目录下的 `candidate_tx_hooks.csv/json` 和 `discover.log`

将整个 `baseline_tx_hook_discovery` 目录打包后分析，选定每个基线的hook。

## 4. 选定hook后做20帧正式统计
把精确module path写入 `selected_hooks.tsv`，多尺度用逗号分隔，然后：
```bash
export MAX_FRAMES=20
bash baseline_feature_budget_batch_v2/run_profile_selected.sh
```
