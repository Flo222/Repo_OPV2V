import torch
import numpy as np
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

model_dir = "opencood/logs/main_opv2v_where2comm_grace_full"
yaml_path = model_dir + "/config.yaml"
max_samples = 500

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

with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break
        batch_data = train_utils.to_device(batch_data, device)
        _ = model(batch_data["ego"])

        if idx % 100 == 0:
            print(f"[cache_gating_probe] idx={idx}")

records = comm.get_records()

all_items = []
cache1_items = []
cache0_items = []

for r in records:
    ru = r.get("reward_update", None)
    if not isinstance(ru, dict):
        continue

    for lr in ru.get("link_rewards", []):
        info = lr.get("info", lr)
        if not isinstance(info, dict):
            continue

        q_eff = float(info.get("q_eff", 0.0))
        cache_enabled = int(float(info.get("cache_enabled", 0.0)))
        cache_quality = float(info.get("cache_quality", 0.0))
        cache_term = float(info.get("cache_term", 0.0))

        expected = cache_quality * (1.0 - q_eff) if cache_enabled else 0.0
        err = abs(cache_term - expected)

        item = {
            "q_eff": q_eff,
            "cache_enabled": cache_enabled,
            "cache_quality": cache_quality,
            "cache_term": cache_term,
            "expected": expected,
            "err": err,
        }
        all_items.append(item)

        if cache_enabled:
            cache1_items.append(item)
        else:
            cache0_items.append(item)

def mean(xs):
    return float(np.mean(xs)) if xs else 0.0

def maxv(xs):
    return float(np.max(xs)) if xs else 0.0

high_q = [x for x in cache1_items if x["q_eff"] >= 0.8]
mid_q = [x for x in cache1_items if 0.4 < x["q_eff"] < 0.8]
low_q = [x for x in cache1_items if x["q_eff"] <= 0.4]

print("\n===== STEP7 CACHE GATING PROBE SUMMARY =====")
print("model_dir =", model_dir)
print("max_samples =", max_samples)
print("total_reward_items =", len(all_items))
print("cache1_items =", len(cache1_items))
print("cache0_items =", len(cache0_items))

print("\n===== formula check =====")
print("max_abs_error_all =", maxv([x["err"] for x in all_items]))
print("max_abs_error_cache1 =", maxv([x["err"] for x in cache1_items]))
print("max_cache_term_cache0 =", maxv([x["cache_term"] for x in cache0_items]))

print("\n===== cache1 grouped by q_eff =====")
print("high_q_count(q_eff>=0.8) =", len(high_q))
print("high_q_cache_term_mean =", mean([x["cache_term"] for x in high_q]))
print("high_q_cache_quality_mean =", mean([x["cache_quality"] for x in high_q]))

print("mid_q_count(0.4<q_eff<0.8) =", len(mid_q))
print("mid_q_cache_term_mean =", mean([x["cache_term"] for x in mid_q]))
print("mid_q_cache_quality_mean =", mean([x["cache_quality"] for x in mid_q]))

print("low_q_count(q_eff<=0.4) =", len(low_q))
print("low_q_cache_term_mean =", mean([x["cache_term"] for x in low_q]))
print("low_q_cache_quality_mean =", mean([x["cache_quality"] for x in low_q]))

print("\n===== sample cache1 records =====")
for x in cache1_items[:10]:
    print(x)
