import os
import csv
import math
import torch
import numpy as np
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils

model_dir = "opencood/logs/main_opv2v_where2comm_grace_full"
yaml_path = model_dir + "/config.yaml"
max_samples = 500
iou_thr = 0.7

out_dir = "audit_runs/step7_reward_conf_ap_corr"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "conf_ap_corr_500.csv")

def get_tp_fp_func():
    if hasattr(eval_utils, "caluclate_tp_fp"):
        return eval_utils.caluclate_tp_fp
    if hasattr(eval_utils, "calculate_tp_fp"):
        return eval_utils.calculate_tp_fp
    raise RuntimeError("Cannot find calculate_tp_fp / caluclate_tp_fp in eval_utils")

def frame_ap_from_stat(stat, thr=0.7):
    d = stat[thr]
    gt = int(d.get("gt", 0))
    if gt <= 0:
        return None

    tp = np.array(d.get("tp", []), dtype=np.float64)
    fp = np.array(d.get("fp", []), dtype=np.float64)
    score = np.array(d.get("score", []), dtype=np.float64)

    if len(score) == 0:
        return 0.0

    order = np.argsort(-score)
    tp = tp[order]
    fp = fp[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    rec = tp_cum / max(gt, 1)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

def rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j+1]] = avg_rank
        i = j + 1
    return ranks

def corr(x, y, spearman=False):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return float("nan")

    if spearman:
        x = rankdata(x)
        y = rankdata(y)

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])

hypes = yaml_utils.load_yaml(yaml_path)
dataset = build_dataset(hypes, visualize=False, train=False)

loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    collate_fn=dataset.collate_batch_test,
    shuffle=False,
    pin_memory=False,
)

model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(model_dir, model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

tp_fp_func = get_tp_fp_func()

rows = []

with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break

        batch_data = train_utils.to_device(batch_data, device)

        pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(
            batch_data,
            model,
            dataset
        )

        if pred_score is None or len(pred_score) == 0:
            scores_np = np.array([], dtype=np.float64)
        else:
            scores_np = pred_score.detach().cpu().numpy().astype(np.float64)

        n_pred = int(len(scores_np))
        n_gt = int(gt_box_tensor.shape[0]) if gt_box_tensor is not None else 0

        mean_conf = float(scores_np.mean()) if n_pred > 0 else 0.0
        max_conf = float(scores_np.max()) if n_pred > 0 else 0.0
        sum_conf = float(scores_np.sum()) if n_pred > 0 else 0.0

        stat = {
            iou_thr: {
                "tp": [],
                "fp": [],
                "gt": 0,
                "score": []
            }
        }

        try:
            tp_fp_func(pred_box_tensor, pred_score, gt_box_tensor, stat, iou_thr)
            frame_ap70 = frame_ap_from_stat(stat, iou_thr)
        except Exception as e:
            frame_ap70 = None
            print(f"[WARN] idx={idx} AP calc failed: {type(e).__name__}: {e}")

        rows.append({
            "idx": idx,
            "n_pred": n_pred,
            "n_gt": n_gt,
            "mean_conf": mean_conf,
            "max_conf": max_conf,
            "sum_conf": sum_conf,
            "frame_ap70": "" if frame_ap70 is None else frame_ap70,
        })

        if idx % 50 == 0:
            print(f"[probe] idx={idx}, n_pred={n_pred}, n_gt={n_gt}, mean_conf={mean_conf:.4f}, ap70={frame_ap70}")

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["idx", "n_pred", "n_gt", "mean_conf", "max_conf", "sum_conf", "frame_ap70"]
    )
    writer.writeheader()
    writer.writerows(rows)

valid = [r for r in rows if r["frame_ap70"] != ""]
ap = np.array([float(r["frame_ap70"]) for r in valid], dtype=np.float64)

metrics = {
    "mean_conf": np.array([float(r["mean_conf"]) for r in valid], dtype=np.float64),
    "max_conf": np.array([float(r["max_conf"]) for r in valid], dtype=np.float64),
    "sum_conf": np.array([float(r["sum_conf"]) for r in valid], dtype=np.float64),
    "n_pred": np.array([float(r["n_pred"]) for r in valid], dtype=np.float64),
    "n_gt": np.array([float(r["n_gt"]) for r in valid], dtype=np.float64),
}

print("\n===== CONF vs FRAME AP@0.7 CORRELATION =====")
print("model_dir =", model_dir)
print("max_samples =", max_samples)
print("valid_frames =", len(valid))
print("csv_path =", csv_path)

for name, x in metrics.items():
    print(
        f"{name:10s}  pearson={corr(x, ap, False): .4f}  spearman={corr(x, ap, True): .4f}"
    )

print("\n===== AP statistics =====")
print("ap70_mean =", float(np.mean(ap)) if len(ap) else None)
print("ap70_std  =", float(np.std(ap)) if len(ap) else None)
print("ap70_min  =", float(np.min(ap)) if len(ap) else None)
print("ap70_max  =", float(np.max(ap)) if len(ap) else None)
