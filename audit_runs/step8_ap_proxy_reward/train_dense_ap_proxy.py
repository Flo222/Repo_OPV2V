import csv
import json
import math
import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import inference_utils
from opencood.utils import eval_utils
from opencood.data_utils.datasets import build_dataset

model_dir = "opencood/logs/main_opv2v_where2comm_grace_full"
yaml_path = model_dir + "/config.yaml"
out_dir = Path("audit_runs/step8_ap_proxy_reward")
out_dir.mkdir(parents=True, exist_ok=True)

max_samples = 500
iou_thr = 0.7

def to_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def mean(xs):
    return float(np.mean(xs)) if xs else 0.0

def frame_ap70(pred_box_tensor, pred_score, gt_box_tensor):
    result_stat = {
        iou_thr: {
            "tp": [],
            "fp": [],
            "gt": 0,
            "score": [],
        }
    }

    calc_tp_fp = getattr(eval_utils, "caluclate_tp_fp", None)
    if calc_tp_fp is None:
        calc_tp_fp = getattr(eval_utils, "calculate_tp_fp")

    calc_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat,
        iou_thr,
    )

    # Current OpenCOOD eval_utils.calculate_ap may require
    # global_sort_detections as the third argument.
    try:
        ap, _, _ = eval_utils.calculate_ap(result_stat, iou_thr, False)
    except TypeError:
        ap, _, _ = eval_utils.calculate_ap(result_stat, iou_thr)
    return float(ap)

def extract_dense_features(output_dict):
    psm = output_dict["psm"]

    with torch.no_grad():
        prob = torch.sigmoid(psm).detach()

        # psm: [B, A, H, W]，先取每个 BEV cell 最强 anchor 置信度
        if prob.dim() == 4:
            dense = prob.max(dim=1)[0]
        else:
            dense = prob.reshape(prob.shape[0], -1)

        flat = dense.reshape(-1).float()

        if flat.numel() == 0:
            return {
                "dense_mean_conf": 0.0,
                "dense_max_conf": 0.0,
                "dense_sum_conf": 0.0,
                "dense_std_conf": 0.0,
                "dense_count_gt_03": 0.0,
                "dense_count_gt_05": 0.0,
                "dense_count_gt_07": 0.0,
                "dense_top10_mean": 0.0,
                "dense_top50_mean": 0.0,
            }

        top10 = torch.topk(flat, k=min(10, flat.numel())).values
        top50 = torch.topk(flat, k=min(50, flat.numel())).values

        return {
            "dense_mean_conf": float(flat.mean().cpu().item()),
            "dense_max_conf": float(flat.max().cpu().item()),
            "dense_sum_conf": float(flat.sum().cpu().item()),
            "dense_std_conf": float(flat.std(unbiased=False).cpu().item()),
            "dense_count_gt_03": float((flat > 0.3).sum().cpu().item()),
            "dense_count_gt_05": float((flat > 0.5).sum().cpu().item()),
            "dense_count_gt_07": float((flat > 0.7).sum().cpu().item()),
            "dense_top10_mean": float(top10.mean().cpu().item()),
            "dense_top50_mean": float(top50.mean().cpu().item()),
        }

def extract_reward_update_features(output_dict):
    ru = output_dict.get("arce_reward_update", None)
    if not isinstance(ru, dict):
        return {
            "reward_count": 0.0,
            "q_eff_mean": 0.0,
            "quant_loss_mean": 0.0,
            "fec_gain_mean": 0.0,
            "cache_term_mean": 0.0,
            "cost_norm_mean": 0.0,
        }

    link_rewards = ru.get("link_rewards", [])
    infos = []
    for lr in link_rewards:
        info = lr.get("info", lr)
        if isinstance(info, dict):
            infos.append(info)

    return {
        "reward_count": float(ru.get("num_updated", len(infos))),
        "q_eff_mean": mean([to_float(x.get("q_eff", 0.0)) for x in infos]),
        "quant_loss_mean": mean([to_float(x.get("quant_loss", 0.0)) for x in infos]),
        "fec_gain_mean": mean([to_float(x.get("fec_gain", 0.0)) for x in infos]),
        "cache_term_mean": mean([to_float(x.get("cache_term", 0.0)) for x in infos]),
        "cost_norm_mean": mean([
            to_float(x.get("normalized_cost", x.get("cost_norm", 0.0)))
            for x in infos
        ]),
    }

def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def rankdata(a):
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks

def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def eval_cv(model, X, y):
    pred = np.zeros_like(y, dtype=np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=2026)
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    return pred, {
        "pearson": pearson(pred, y),
        "spearman": spearman(pred, y),
        "rmse": rmse(pred, y),
        "mae": mae(pred, y),
    }

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

rows = []

with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break

        batch_data = train_utils.to_device(batch_data, device)

        output_dict = model(batch_data["ego"])

        dense_feats = extract_dense_features(output_dict)
        reward_feats = extract_reward_update_features(output_dict)

        # 优先用 dataset.post_process，避免二次 forward
        try:
            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
                batch_data,
                output_dict,
            )
        except Exception:
            # fallback：如果某些 dataset 不支持直接 post_process
            pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_intermediate_fusion(
                batch_data,
                model,
                dataset,
            )

        ap70 = frame_ap70(pred_box_tensor, pred_score, gt_box_tensor)

        row = {
            "idx": int(idx),
            "frame_ap70": float(ap70),
        }
        row.update(dense_feats)
        row.update(reward_feats)
        rows.append(row)

        if idx % 50 == 0:
            print(
                f"[dense_ap_proxy_data] idx={idx}, "
                f"ap70={ap70:.4f}, "
                f"dense_mean={dense_feats['dense_mean_conf']:.4f}, "
                f"reward_count={reward_feats['reward_count']:.0f}"
            )

csv_path = out_dir / "dense_ap_proxy_dataset_500.csv"
fieldnames = list(rows[0].keys())
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

feature_cols = [
    "dense_mean_conf",
    "dense_max_conf",
    "dense_sum_conf",
    "dense_std_conf",
    "dense_count_gt_03",
    "dense_count_gt_05",
    "dense_count_gt_07",
    "dense_top10_mean",
    "dense_top50_mean",
    "reward_count",
    "q_eff_mean",
    "quant_loss_mean",
    "fec_gain_mean",
    "cache_term_mean",
    "cost_norm_mean",
]

X = np.array([[to_float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)
y = np.array([to_float(r["frame_ap70"]) for r in rows], dtype=np.float64)

baseline_pred = X[:, feature_cols.index("dense_mean_conf")]

ridge_model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])

rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        random_state=2026,
        n_jobs=-1,
    )),
])

ridge_pred, ridge_res = eval_cv(ridge_model, X, y)
rf_pred, rf_res = eval_cv(rf_model, X, y)

baseline_res = {
    "pearson": pearson(baseline_pred, y),
    "spearman": spearman(baseline_pred, y),
    "rmse": rmse(baseline_pred, y),
    "mae": mae(baseline_pred, y),
}

final_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=5,
        random_state=2026,
        n_jobs=-1,
    )),
])
final_rf.fit(X, y)

model_path = out_dir / "ap_proxy_dense_rf.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_rf, f)

rf = final_rf.named_steps["rf"]
importance = {
    c: float(v) for c, v in zip(feature_cols, rf.feature_importances_)
}

meta = {
    "model_dir": model_dir,
    "dataset_csv": str(csv_path),
    "target_col": "frame_ap70",
    "feature_cols": feature_cols,
    "num_rows": int(len(rows)),
    "cv_results": {
        "dense_mean_baseline": baseline_res,
        "ridge_dense_ap_proxy": ridge_res,
        "random_forest_dense_ap_proxy": rf_res,
    },
    "feature_importance": importance,
    "model_path": str(model_path),
}

meta_path = out_dir / "ap_proxy_dense_rf_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

pred_path = out_dir / "dense_ap_proxy_predictions_500.csv"
with open(pred_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_ap70", "dense_mean_conf", "ridge_ap_hat", "rf_ap_hat"])
    for i in range(len(y)):
        writer.writerow([
            float(y[i]),
            float(baseline_pred[i]),
            float(ridge_pred[i]),
            float(rf_pred[i]),
        ])

print("\n===== DENSE AP PROXY RESULTS =====")
print("dataset_csv =", csv_path)
print("num_rows =", len(rows))
print("feature_cols =", feature_cols)

print("\nCV RESULTS")
for name, res in meta["cv_results"].items():
    print(name)
    for k, v in res.items():
        print(" ", k, "=", v)

print("\nRF FEATURE IMPORTANCE")
for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{k}\t{v:.6f}")

print("\nSaved:")
print(model_path)
print(meta_path)
print(pred_path)
