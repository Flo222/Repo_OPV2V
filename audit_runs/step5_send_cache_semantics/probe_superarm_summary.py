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

with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break
        batch_data = train_utils.to_device(batch_data, device)
        _ = model(batch_data["ego"])

records = comm.get_records()

cnt = Counter()
samples = []

for r in records:
    if "dc2mab_superarm" not in r:
        continue
    s = r["dc2mab_superarm"]

    nc = int(s.get("num_collaborators", -1))
    ns = int(s.get("num_selected", -1))
    cnt[f"collab_{nc}_selected_{ns}"] += 1

    if len(samples) < 15:
        samples.append({
            "frame_id": r.get("frame_id"),
            "num_collaborators": nc,
            "num_selected": ns,
            "budget_source": s.get("budget_source"),
            "budget_scope": s.get("budget_scope"),
            "budget_bytes": s.get("budget_bytes"),
            "used_budget_bytes": s.get("used_budget_bytes"),
            "selected_sender_ids": s.get("selected_sender_ids"),
            "selected_action_ids": s.get("selected_action_ids"),
            "link_states": s.get("link_states"),
        })

print("===== SUPERARM SUMMARY =====")
print("total_records =", len(records))
for k in sorted(cnt.keys()):
    print(k, "=", cnt[k])

print("\n===== samples =====")
for x in samples:
    print(x)
