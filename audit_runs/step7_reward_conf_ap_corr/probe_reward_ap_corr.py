import os
import csv
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
csv_path = os.path.join(out_dir, "reward_ap_corr_500.csv")

def get_tp_fp_func():
    if hasattr(eval_utils, "caluclate_tp_fp"):
        return eval_utils.caluclate_tp_fp
    if hasattr(eval_utils, "calculate_tp_fp"):
        return eval_utils.calculate_tp_fp
    raise RuntimeError("Cannot find calculate_tp_fp in eval_utils")

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
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)

    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j+1]] = (i + j) / 2.0 + 1.0
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

comm = model.arce_comm
tp_fp_func = get_tp_fp_func()

rows = []

with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break

        batch_data = train_utils.to_device(batch_data, device)

        rec_before = len(comm.get_records())

        pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(
            batch_data,
            model,
            dataset
        )

        rec_after = comm.get_records()[rec_before:]

        frame_rewards = []
        frame_q_eff = []
        frame_cost_norm = []
        frame_quant_loss = []
        frame_fec_gain = []
        frame_cache_term = []

        for r in rec_after:
            if "reward_update" not in r:
                continue
            ru = r["reward_update"]
            for lr in ru.get("link_rewards", []):
                if "reward" in lr:
                    frame_rewards.append(float(lr["reward"]))

                info = lr.get("info", lr)
                if isinstance(info, dict):
                    for key, arr in [
                        ("q_eff", frame_q_eff),
                        ("cost_norm", frame_cost_norm),
                        ("quant_loss", frame_quant_loss),
                        ("fec_gain", frame_fec_gain),
                        ("cache_term", frame_cache_term),
                    ]:
                        if key in info:
                            try:
                                arr.append(float(info[key]))
                            except Exception:
                                pass

        if pred_score is None or len(pred_score) == 0:
            scores_np = np.array([], dtype=np.float64)
        else:
            scores_np = pred_score.detach().cpu().numpy().astype(np.float64)

        mean_conf = float(scores_np.mean()) if len(scores_np) > 0 else 0.0
        sum_conf = float(scores_np.sum()) if len(scores_np) > 0 else 0.0
        n_pred = int(len(scores_np))
        n_gt = int(gt_box_tensor.shape[0]) if gt_box_tensor is not None else 0

        stat = {iou_thr: {"tp": [], "fp": [], "gt": 0, "score": []}}
        try:
            tp_fp_func(pred_box_tensor, pred_score, gt_box_tensor, stat, iou_thr)
            ap70 = frame_ap_from_stat(stat, iou_thr)
        except Exception as e:
            ap70 = None
            print(f"[WARN] idx={idx} AP calc failed: {type(e).__name__}: {e}")

        row = {
            "idx": idx,
            "frame_ap70": "" if ap70 is None else ap70,
            "mean_conf": mean_conf,
            "sum_conf": sum_conf,
            "n_pred": n_pred,
            "n_gt": n_gt,
            "reward_sum": float(np.sum(frame_rewards)) if frame_rewards else 0.0,
            "reward_mean": float(np.mean(frame_rewards)) if frame_rewards else 0.0,
            "reward_count": len(frame_rewards),
            "q_eff_mean": float(np.mean(frame_q_eff)) if frame_q_eff else 0.0,
            "cost_norm_mean": float(np.mean(frame_cost_norm)) if frame_cost_norm else 0.0,
            "quant_loss_mean": float(np.mean(frame_quant_loss)) if frame_quant_loss else 0.0,
            "fec_gain_mean": float(np.mean(frame_fec_gain)) if frame_fec_gain else 0.0,
            "cache_term_mean": float(np.mean(frame_cache_term)) if frame_cache_term else 0.0,
        }
        rows.append(row)

        if idx % 50 == 0:
            print(
                f"[probe] idx={idx}, ap70={ap70}, mean_conf={mean_conf:.4f}, "
                f"reward_mean={row['reward_mean']:.4f}, reward_count={row['reward_count']}"
            )

with open(csv_path, "w", newline="") as f:
    fieldnames = [
        "idx", "frame_ap70", "mean_conf", "sum_conf", "n_pred", "n_gt",
        "reward_sum", "reward_mean", "reward_count",
        "q_eff_mean", "cost_norm_mean", "quant_loss_mean", "fec_gain_mean", "cache_term_mean"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

valid = [r for r in rows if r["frame_ap70"] != ""]
ap = np.array([float(r["frame_ap70"]) for r in valid], dtype=np.float64)

metrics = [
    "mean_conf",
    "sum_conf",
    "n_pred",
    "reward_sum",
    "reward_mean",
    "reward_count",
    "q_eff_mean",
    "cost_norm_mean",
    "quant_loss_mean",
    "fec_gain_mean",
    "cache_term_mean",
]

print("\n===== REWARD / COMPONENT vs FRAME AP@0.7 CORRELATION =====")
print("model_dir =", model_dir)
print("max_samples =", max_samples)
print("valid_frames =", len(valid))
print("csv_path =", csv_path)

for name in metrics:
    x = np.array([float(r[name]) for r in valid], dtype=np.float64)
    print(f"{name:16s} pearson={corr(x, ap, False): .4f}  spearman={corr(x, ap, True): .4f}")

print("\n===== reward statistics =====")
rw = np.array([float(r["reward_mean"]) for r in valid], dtype=np.float64)
print("reward_mean_avg =", float(np.mean(rw)))
print("reward_mean_std =", float(np.std(rw)))
print("reward_nonzero_frames =", int(np.sum(np.abs(rw) > 1e-12)))
