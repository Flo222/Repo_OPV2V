import torch
from torch.utils.data import DataLoader
from collections import Counter

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
            print(f"[cache_probe] idx={idx}")

cnt = Counter()
bad_samples = []

for r in comm.get_records():
    action = r.get("action", None)
    if not isinstance(action, dict):
        continue

    cache_enabled = int(action.get("cache_enabled", action.get("cache", -1)))
    temporal_source = r.get("temporal_source", None)

    if cache_enabled == 0:
        cnt["cache0_records"] += 1
        if temporal_source == "current_cache_disabled":
            cnt["cache0_current_cache_disabled"] += 1
        if temporal_source == "previous_frame":
            cnt["cache0_previous_frame_violation"] += 1
            if len(bad_samples) < 5:
                bad_samples.append({
                    "action": action,
                    "temporal_source": temporal_source,
                    "channel_state": r.get("channel_state"),
                })

    if cache_enabled == 1:
        cnt["cache1_records"] += 1
        if temporal_source == "previous_frame":
            cnt["cache1_previous_frame"] += 1
        if temporal_source == "current_no_history":
            cnt["cache1_current_no_history"] += 1
        if temporal_source == "current":
            cnt["cache1_current"] += 1

print("\n===== CACHE SEMANTICS 500 SUMMARY =====")
print("model_dir =", model_dir)
print("max_samples =", max_samples)
for k in sorted(cnt.keys()):
    print(f"{k} = {cnt[k]}")

print("\n===== cache0 previous_frame violations =====")
for x in bad_samples:
    print(x)
