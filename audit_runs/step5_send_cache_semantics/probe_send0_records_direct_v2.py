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
comm.system_budget_mbps = 0.001

with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break
        batch_data = train_utils.to_device(batch_data, device)
        _ = model(batch_data["ego"])
        if idx % 10 == 0:
            print(f"[direct_v2] idx={idx}")

records = comm.get_records()
cnt = Counter()
samples = []

for r in records:
    if r.get("no_send", False):
        cnt["no_send_records"] += 1
        if r.get("selected_for_update", False):
            cnt["no_send_selected_for_update"] += 1
        if r.get("no_send_update", False):
            cnt["no_send_update_true"] += 1
        if "no_send_update_error" in r:
            cnt["no_send_update_error"] += 1
        if len(samples) < 8:
            samples.append({
                "action": r.get("action"),
                "pdf_action": r.get("pdf_action"),
                "selected_for_update": r.get("selected_for_update"),
                "no_send_update": r.get("no_send_update"),
                "err": r.get("no_send_update_error"),
                "actual_transmitted_bytes": r.get("actual_transmitted_bytes"),
            })

    if "reward_update" in r:
        ru = r["reward_update"]
        cnt["reward_update_records"] += 1
        cnt["reward_num_updated_sum"] += int(ru.get("num_updated", 0))
        for lr in ru.get("link_rewards", []):
            aid = str(lr.get("action_id", "")).lower()
            if "send0" in aid:
                cnt["reward_send0_links"] += 1

print("\n===== DIRECT V2 SEND0 RECORD PROBE SUMMARY =====")
print("model_dir =", model_dir)
print("max_samples =", max_samples)
print("forced_runtime_system_budget_mbps = 0.001")
print("total_records =", len(records))
for k in sorted(cnt.keys()):
    print(f"{k} = {cnt[k]}")

print("\n===== no-send record samples =====")
for x in samples:
    print(x)
