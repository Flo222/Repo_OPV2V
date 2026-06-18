import csv
import math
import json
import pickle
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

csv_path = Path("audit_runs/step7_reward_ap_alignment/reward_ap_corr_500_after_cache_and_fec_gating.csv")
out_dir = Path("audit_runs/step8_ap_proxy_reward")
out_dir.mkdir(parents=True, exist_ok=True)

target_col = "frame_ap70"

# 干净版特征：不使用 reward_sum / reward_mean，避免旧 reward 循环依赖。
feature_cols = [
    "mean_conf",
    "sum_conf",
    "n_pred",
    "reward_count",
    "q_eff_mean",
    "cost_norm_mean",
    "quant_loss_mean",
    "fec_gain_mean",
    "cache_term_mean",
]

def to_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return np.nan
        return v
    except Exception:
        return np.nan

rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    missing = [c for c in [target_col] + feature_cols if c not in fieldnames]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}; fields={fieldnames}")
    for r in reader:
        rows.append(r)

X = np.array([[to_float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)
y = np.array([to_float(r[target_col]) for r in rows], dtype=np.float64)

valid = np.isfinite(y)
X = X[valid]
y = y[valid]

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

def eval_cv(model, X, y, name):
    pred = np.zeros_like(y, dtype=np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=2026)

    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    res = {
        "pearson": pearson(pred, y),
        "spearman": spearman(pred, y),
        "rmse": rmse(pred, y),
        "mae": mae(pred, y),
    }
    return pred, res

mean_conf_pred = X[:, feature_cols.index("mean_conf")]
mean_conf_res = {
    "pearson": pearson(mean_conf_pred, y),
    "spearman": spearman(mean_conf_pred, y),
    "rmse": rmse(mean_conf_pred, y),
    "mae": mae(mean_conf_pred, y),
}

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

ridge_pred, ridge_res = eval_cv(ridge_model, X, y, "ridge")
rf_pred, rf_res = eval_cv(rf_model, X, y, "random_forest")

# 训练最终模型：全 500 帧训练，供后续在线 reward 读取。
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

model_path = out_dir / "ap_proxy_rf_clean.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_rf, f)

meta = {
    "csv_path": str(csv_path),
    "target_col": target_col,
    "feature_cols": feature_cols,
    "num_rows": int(len(y)),
    "cv_results": {
        "mean_conf_baseline": mean_conf_res,
        "ridge_ap_proxy_clean": ridge_res,
        "random_forest_ap_proxy_clean": rf_res,
    },
    "model_path": str(model_path),
}

# feature importance
rf = final_rf.named_steps["rf"]
meta["feature_importance"] = {
    c: float(v) for c, v in zip(feature_cols, rf.feature_importances_)
}

meta_path = out_dir / "ap_proxy_rf_clean_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

pred_path = out_dir / "ap_proxy_clean_predictions_500.csv"
with open(pred_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_ap70", "mean_conf", "ridge_ap_hat", "rf_ap_hat"])
    for i in range(len(y)):
        writer.writerow([
            float(y[i]),
            float(mean_conf_pred[i]),
            float(ridge_pred[i]),
            float(rf_pred[i]),
        ])

print("===== CLEAN AP PROXY TRAINING =====")
print("csv_path =", csv_path)
print("num_rows =", len(y))
print("target_col =", target_col)
print("feature_cols =", feature_cols)

print("\n===== CV RESULTS =====")
for name, res in meta["cv_results"].items():
    print(name)
    for k, v in res.items():
        print(" ", k, "=", v)

print("\n===== RF FEATURE IMPORTANCE =====")
for k, v in sorted(meta["feature_importance"].items(), key=lambda x: x[1], reverse=True):
    print(f"{k}\t{v:.6f}")

print("\nSaved:")
print(model_path)
print(meta_path)
print(pred_path)
