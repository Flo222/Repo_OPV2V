import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except Exception as e:
    raise RuntimeError(
        "缺少 sklearn，无法训练 RandomForest AP proxy。"
        "请先确认当前环境是否安装 scikit-learn。原始错误：%s" % repr(e)
    )

MODEL_DIR = "opencood/logs/main_opv2v_where2comm_grace_full"
YAML_PATH = MODEL_DIR + "/config.yaml"
OUT_DIR = Path("audit_runs/step8_ap_proxy_reward")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 500
IOU_THR = 0.70
SEED = 0

FEATURE_NAMES = [
    "dense_mean_conf",
    "dense_max_conf",
    "dense_sum_conf",
    "dense_std_conf",
    "dense_count_gt_03",
    "dense_count_gt_05",
    "dense_count_gt_07",
    "dense_top10_mean",
    "dense_top50_mean",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def dense_features_from_psm(psm):
    with torch.no_grad():
        conf = torch.sigmoid(psm.detach().float())

        # psm usually: [B, A, H, W]. Use max over anchor/channel dim.
        if conf.dim() >= 4:
            conf = conf.max(dim=1)[0]

        flat = conf.reshape(-1)
        if flat.numel() <= 0:
            return {k: 0.0 for k in FEATURE_NAMES}

        top10_k = min(10, int(flat.numel()))
        top50_k = min(50, int(flat.numel()))

        top10 = torch.topk(flat, k=top10_k).values
        top50 = torch.topk(flat, k=top50_k).values

        return {
            "dense_mean_conf": float(flat.mean().detach().cpu().item()),
            "dense_max_conf": float(flat.max().detach().cpu().item()),
            "dense_sum_conf": float(flat.sum().detach().cpu().item()),
            "dense_std_conf": float(flat.std(unbiased=False).detach().cpu().item()),
            "dense_count_gt_03": float((flat > 0.3).sum().detach().cpu().item()),
            "dense_count_gt_05": float((flat > 0.5).sum().detach().cpu().item()),
            "dense_count_gt_07": float((flat > 0.7).sum().detach().cpu().item()),
            "dense_top10_mean": float(top10.mean().detach().cpu().item()),
            "dense_top50_mean": float(top50.mean().detach().cpu().item()),
        }


def frame_ap70(dataset, batch_data, output_dict):
    # OpenCOOD post_process expects output_dict keyed by cav id, e.g. {"ego": model_output}.
    # The model forward on batch_data["ego"] only returns the ego branch output itself.
    wrapped_output_dict = {"ego": output_dict}
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
        batch_data, wrapped_output_dict
    )

    result_stat = {
        IOU_THR: {
            "tp": [],
            "fp": [],
            "gt": 0,
            "score": [],
        }
    }

    eval_utils.caluclate_tp_fp(
        pred_box_tensor,
        pred_score,
        gt_box_tensor,
        result_stat,
        IOU_THR,
    )

    if int(result_stat[IOU_THR]["gt"]) <= 0:
        return None

    try:
        ap, _, _ = eval_utils.calculate_ap(result_stat, IOU_THR, False)
    except TypeError:
        ap, _, _ = eval_utils.calculate_ap(result_stat, IOU_THR)

    if ap is None or not np.isfinite(float(ap)):
        return None

    return float(ap)


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rank_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return corr(rx, ry)


print("[1/4] Loading dataset/model...")
hypes = yaml_utils.load_yaml(YAML_PATH)
dataset = build_dataset(hypes, visualize=False, train=False)
model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(MODEL_DIR, model)
model = model.cuda().eval()

num_samples = min(MAX_SAMPLES, len(dataset))
rows = []

print("[2/4] Collecting dense-head features and AP@0.7 targets...")
for idx in range(num_samples):
    if idx % 50 == 0:
        print("[collect] idx =", idx, "valid_rows =", len(rows))

    try:
        batch_data = dataset.collate_batch_test([dataset[idx]])
        batch_data = train_utils.to_device(batch_data, torch.device("cuda"))

        with torch.no_grad():
            output_dict = model(batch_data["ego"])

        feats = dense_features_from_psm(output_dict["psm"])
        ap70 = frame_ap70(dataset, batch_data, output_dict)

        if ap70 is None:
            continue

        row = {
            "idx": int(idx),
            "ap70": float(ap70),
        }
        row.update(feats)
        rows.append(row)

    except Exception as e:
        print("[WARN] skip idx=%d due to %s: %s" % (idx, type(e).__name__, e))
        continue

if len(rows) < 30:
    raise RuntimeError("有效训练样本太少：%d。请检查 dataset.post_process 或 eval_utils。" % len(rows))

csv_path = OUT_DIR / "dense_ap_proxy_dataset_500.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["idx", "ap70"] + FEATURE_NAMES)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print("[3/4] Training RandomForest dense AP proxy...")
X = np.asarray([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=np.float64)
y = np.asarray([r["ap70"] for r in rows], dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=SEED,
    min_samples_leaf=3,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)

meta = {
    "model_type": "RandomForestRegressor",
    "target": "frame_ap70",
    "feature_names": FEATURE_NAMES,
    "num_rows": int(len(rows)),
    "num_train": int(len(y_train)),
    "num_test": int(len(y_test)),
    "train_pearson": corr(pred_train, y_train),
    "train_spearman": rank_corr(pred_train, y_train),
    "test_pearson": corr(pred_test, y_test),
    "test_spearman": rank_corr(pred_test, y_test),
    "dense_mean_baseline_test_pearson": corr(X_test[:, 0], y_test),
    "dense_mean_baseline_test_spearman": rank_corr(X_test[:, 0], y_test),
    "model_dir": MODEL_DIR,
    "yaml_path": YAML_PATH,
    "max_samples": int(MAX_SAMPLES),
    "iou_thr": float(IOU_THR),
}

model_path = OUT_DIR / "ap_proxy_dense_rf.pkl"
meta_path = OUT_DIR / "ap_proxy_dense_rf_meta.json"

with open(model_path, "wb") as f:
    pickle.dump(rf, f)

meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

print("[4/4] Saved AP proxy.")
print("dataset =", csv_path)
print("model   =", model_path)
print("meta    =", meta_path)
print(json.dumps(meta, indent=2, ensure_ascii=False))
