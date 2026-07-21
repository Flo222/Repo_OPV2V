# Baseline Feature/Budget Audit v1

Purpose: capture the real tensor immediately before a baseline's fusion module, then calculate payload size and packet fit for FP32/FP16/INT8/INT4 and rho values under Good/Medium/Bad budgets.

## Required inputs per baseline

Only the model log directory is mandatory. It must contain `config.yaml` and a checkpoint such as `net_epoch20.pth`. The test dataset path should already be correct in `config.yaml`; otherwise edit a copied config or point it to the test set first.

## 1. Discover hook module

```bash
python scripts/discover_baseline_tx_hooks.py \
  --model_dir /path/to/baseline_log \
  --out_dir /path/to/audit/discover_baseline \
  --max_frames 1 \
  --num_workers 0
```

Choose a candidate with `max_split_sender_count > 0`. Common examples:

- Where2Comm / V2X-ViT / RoCooper: `fusion_net`
- CoSDH: `fusion_net.0,fusion_net.1,...`
- CoopDiff: `fuse_modules.0,fuse_modules.1,...`

The discovery CSV is the source of truth because exact names differ by branch.

## 2. Profile real feature and budget fit

```bash
python scripts/profile_baseline_feature_budget.py \
  --model_dir /path/to/baseline_log \
  --hook_modules fusion_net \
  --out_dir /path/to/audit/profile_baseline \
  --max_frames 20 \
  --quant_modes fp32,fp16,int8,int4 \
  --rhos 0,0.1,0.25,0.6 \
  --profiles 'good:27:0.05:10,medium:5:0.20:50,bad:1:0.35:100' \
  --tx_window_ms 100 \
  --packet_size_bytes 1024 \
  --budget_scope both \
  --num_workers 0
```

Outputs:

- `feature_sizes_per_link.csv`: actual per-sender tensor shape, raw bytes, quantized payload and source packet count.
- `budget_fit_per_link.csv`: every sender × quant × rho × state calculation.
- `budget_fit_summary.csv`: mean source/parity fit ratios.
- `manifest.json`: assumptions and run parameters.

`budget_scope=both` reports:

- `per_link`: every sender gets the full 27/5/1 Mbps × 100 ms budget.
- `frame_shared`: one frame budget is divided by the number of active senders.

The current implementation is reproduced as source-first. The file also reports a budget-feasible source/parity plan for comparison, but does not change project behavior.
