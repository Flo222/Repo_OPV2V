import torch
from torch.utils.data import DataLoader
from collections import Counter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

model_dir = "opencood/logs/main_opv2v_where2comm_grace_full"
yaml_path = model_dir + "/config.yaml"
max_samples = 80

hypes = yaml_utils.load_yaml(yaml_path)

# 临时强制低预算，逼迫 oracle 不能选完所有 sender
try:
    hypes["model"]["args"]["arce"]["system_budget_mbps"] = 0.05
except Exception:
    pass

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

def walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_dicts(v)
    elif isinstance(obj, list):
        for x in obj:
            yield from walk_dicts(x)

cnt = Counter()
send0_rewards = []

with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break

        batch_data = train_utils.to_device(batch_data, device)
        out = model(batch_data["ego"])

        for d in walk_dicts(out):
            if d.get("no_send", False):
                cnt["no_send_records_in_output"] += 1
            if d.get("selected_for_update", False):
                cnt["selected_for_update_records_in_output"] += 1
            if d.get("no_send_update", False):
                cnt["no_send_update_records_in_output"] += 1

            if "reward_update" in d and isinstance(d["reward_update"], dict):
                ru = d["reward_update"]
                cnt["reward_update_records"] += 1
                cnt["reward_num_updated_sum"] += int(ru.get("num_updated", 0))
                for lr in ru.get("link_rewards", []):
                    aid = str(lr.get("action_id", "")).lower()
                    if "send0" in aid:
                        cnt["reward_send0_links"] += 1
                        if len(send0_rewards) < 5:
                            send0_rewards.append(lr)

        if idx % 10 == 0:
            print(f"[low_budget_probe] idx={idx}")

print("\n===== LOW BUDGET SEND0 PROBE SUMMARY =====")
print("model_dir =", model_dir)
print("dataset_len =", len(dataset))
print("max_samples =", max_samples)
print("forced_system_budget_mbps = 0.05")

for k in sorted(cnt.keys()):
    print(f"{k} = {cnt[k]}")

print("\n===== send0 reward samples =====")
for x in send0_rewards:
    print(x)
