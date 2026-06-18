import csv
import math
import json
import numpy as np
from pathlib import Path

# 优先使用 Step7 修正后的 reward/AP 相关性 CSV
candidates = [
    Path("audit_runs/step7_reward_ap_alignment/reward_ap_corr_500_after_cache_and_fec_gating.csv"),
    Path("audit_runs/step7_reward_conf_ap_corr/reward_ap_corr_500.csv"),
]

csv_path = None
for p in candidates:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    raise FileNotFoundError("No reward_ap_corr_500 csv found.")

out_dir = Path("audit_runs/step8_ap_proxy_reward")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        rows.append(r)

print("===== AP PROXY OFFLINE DATASET =====")
print("csv_path =", csv_path)
print("num_rows =", len(rows))
print("fields =", fieldnames)

target_candidates = [
    "frame_ap70", "ap70", "AP70", "ap_70", "ap@0.7",
    "frame_AP@0.7", "AP@0.7"
]

target_col = None
for c in target_candidates:
    if c in fieldnames:
        target_col = c
        break

if target_col is None:
    # 兜底：找名字里同时含 ap 和 70/0.7 的列
    for c in fieldnames:
        lc = c.lower()
        if "ap" in lc and ("70" in lc or "0.7" in lc):
            target_col = c
            break

if target_col is None:
    raise RuntimeError("Cannot infer AP@0.7 target column from CSV fields.")

feature_candidates = [
    "mean_conf",
    "max_conf",
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

features = [c for c in feature_candidates if c in fieldnames]

if not features:
    raise RuntimeError("No usable feature columns found.")

def to_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return np.nan
        return v
    except Exception:
        return np.nan

X = np.array([[to_float(r.get(c, "")) for c in features] for r in rows], dtype=np.float64)
y = np.array([to_float(r.get(target_col, "")) for r in rows], dtype=np.float64)

valid = np.isfinite(y)
X = X[valid]
y = y[valid]

# 用列均值填充 nan；全 nan 列填 0
for j in range(X.shape[1]):
    col = X[:, j]
    good = np.isfinite(col)
    if good.any():
        m = float(np.mean(col[good]))
    else:
        m = 0.0
    col[~good] = m
    X[:, j] = col

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

def kfold_indices(n, k=5, seed=2026):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test

def fit_predict_ridge(X, y, k=5, lam=1.0):
    preds = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in kfold_indices(len(y), k=k):
        Xtr = X[train_idx]
        ytr = y[train_idx]
        Xte = X[test_idx]

        mu = Xtr.mean(axis=0)
        std = Xtr.std(axis=0)
        std[std < 1e-8] = 1.0

        Xtrn = (Xtr - mu) / std
        Xten = (Xte - mu) / std

        Xtrb = np.concatenate([np.ones((Xtrn.shape[0], 1)), Xtrn], axis=1)
        Xteb = np.concatenate([np.ones((Xten.shape[0], 1)), Xten], axis=1)

        I = np.eye(Xtrb.shape[1])
        I[0, 0] = 0.0
        w = np.linalg.solve(Xtrb.T @ Xtrb + lam * I, Xtrb.T @ ytr)
        preds[test_idx] = Xteb @ w

    return preds

print("\n===== TARGET / FEATURES =====")
print("target_col =", target_col)
print("features =", features)

baseline_pred = X[:, features.index("mean_conf")] if "mean_conf" in features else np.full_like(y, np.mean(y))
ridge_pred = fit_predict_ridge(X, y, k=5, lam=1.0)

results = {}

results["mean_conf_baseline"] = {
    "pearson": pearson(baseline_pred, y),
    "spearman": spearman(baseline_pred, y),
    "rmse": rmse(baseline_pred, y),
    "mae": mae(baseline_pred, y),
}

results["ridge_ap_proxy"] = {
    "pearson": pearson(ridge_pred, y),
    "spearman": spearman(ridge_pred, y),
    "rmse": rmse(ridge_pred, y),
    "mae": mae(ridge_pred, y),
}

rf_pred = None
rf_importance = None

try:
    from sklearn.ensemble import RandomForestRegressor

    rf_pred = np.zeros_like(y, dtype=np.float64)
    importances = []

    for train_idx, test_idx in kfold_indices(len(y), k=5):
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            random_state=2026,
            n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        rf_pred[test_idx] = model.predict(X[test_idx])
        importances.append(model.feature_importances_)

    rf_importance = np.mean(np.vstack(importances), axis=0)

    results["random_forest_ap_proxy"] = {
        "pearson": pearson(rf_pred, y),
        "spearman": spearman(rf_pred, y),
        "rmse": rmse(rf_pred, y),
        "mae": mae(rf_pred, y),
    }

except Exception as e:
    results["random_forest_ap_proxy"] = {
        "skipped": True,
        "reason": repr(e),
    }

print("\n===== AP PROXY CV RESULTS =====")
for name, res in results.items():
    print(name)
    for k, v in res.items():
        print(" ", k, "=", v)

if rf_importance is not None:
    print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")
    pairs = sorted(zip(features, rf_importance), key=lambda x: x[1], reverse=True)
    for f, imp in pairs:
        print(f"{f}\t{float(imp):.6f}")

with open(out_dir / "ap_proxy_cv_results.json", "w") as f:
    json.dump(
        {
            "csv_path": str(csv_path),
            "target_col": target_col,
            "features": features,
            "results": results,
            "rf_importance": (
                {f: float(i) for f, i in zip(features, rf_importance)}
                if rf_importance is not None else None
            ),
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

# 保存预测，方便后续画图或分析
pred_csv = out_dir / "ap_proxy_predictions_500.csv"
with open(pred_csv, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["y_ap70", "mean_conf_baseline", "ridge_ap_hat"]
    if rf_pred is not None:
        header.append("rf_ap_hat")
    writer.writerow(header)

    for i in range(len(y)):
        row = [float(y[i]), float(baseline_pred[i]), float(ridge_pred[i])]
        if rf_pred is not None:
            row.append(float(rf_pred[i]))
        writer.writerow(row)

print("\nSaved:")
print(out_dir / "ap_proxy_cv_results.json")
print(pred_csv)
