import torch
from torch.utils.data import DataLoader
from collections import Counter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

model_dir = "opencood/logs/main_opv2v_where2comm_grace_full"
yaml_path = model_dir + "/config.yaml"
max_samples = 80

def walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_dicts(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk_dicts(x)

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

cnt = Counter()
bad_cache0_prev = []
no_send_samples = []

with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break

        batch_data = train_utils.to_device(batch_data, device)
        out = model(batch_data["ego"])

        for d in walk_dicts(out):
            if d.get("no_send", False):
                cnt["no_send_records"] += 1

                if d.get("selected_for_update", False):
                    cnt["no_send_selected_for_update"] += 1

                if d.get("no_send_update", False):
                    cnt["no_send_update_true"] += 1

                if "no_send_update_error" in d:
                    cnt["no_send_update_error"] += 1

                if len(no_send_samples) < 5:
                    no_send_samples.append({
                        "idx": idx,
                        "action": d.get("action"),
                        "pdf_action": d.get("pdf_action"),
                        "selected_for_update": d.get("selected_for_update"),
                        "no_send_update": d.get("no_send_update"),
                        "err": d.get("no_send_update_error"),
                        "actual_transmitted_bytes": d.get("actual_transmitted_bytes"),
                    })

            if "reward_update" in d and isinstance(d["reward_update"], dict):
                ru = d["reward_update"]
                cnt["reward_update_records"] += 1
                cnt["reward_num_updated_sum"] += int(ru.get("num_updated", 0))
                for lr in ru.get("link_rewards", []):
                    aid = str(lr.get("action_id", "")).lower()
                    if "send0" in aid:
                        cnt["reward_send0_links"] += 1

            action = d.get("action", None)
            temporal_source = d.get("temporal_source", None)

            cache_enabled = None
            if isinstance(action, dict):
                cache_enabled = int(action.get("cache_enabled", action.get("cache", -1)))

            if cache_enabled == 0 and temporal_source == "previous_frame":
                cnt["cache0_previous_frame_violation"] += 1
                if len(bad_cache0_prev) < 5:
                    bad_cache0_prev.append({
                        "idx": idx,
                        "action": action,
                        "temporal_source": temporal_source,
                        "channel_state": d.get("channel_state"),
                    })

        if idx % 10 == 0:
            print(f"[probe] idx={idx}")

print("\n===== STEP5 SEND/CACHE PROBE SUMMARY =====")
print("model_dir =", model_dir)
print("dataset_len =", len(dataset))
print("max_samples =", max_samples)

for k in sorted(cnt.keys()):
    print(f"{k} = {cnt[k]}")

print("\n===== no-send samples =====")
for x in no_send_samples:
    print(x)

print("\n===== cache0 previous_frame violations =====")
for x in bad_cache0_prev:
    print(x)
