import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

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
            print(f"[fec_reward_probe] idx={idx}")

records = comm.get_records()

items = []
by_rho = defaultdict(list)

for r in records:
    ru = r.get("reward_update", None)
    if not isinstance(ru, dict):
        continue

    for lr in ru.get("link_rewards", []):
        info = lr.get("info", lr)
        if not isinstance(info, dict):
            continue

        action_id = str(info.get("action_id", lr.get("action_id", "")))
        fec_gain = float(info.get("fec_gain", 0.0))
        q_eff = float(info.get("q_eff", 0.0))
        reward = float(info.get("reward", lr.get("reward", 0.0)))

        if "rho0p5" in action_id:
            rho = "rho0p5"
        elif "rho0p25" in action_id:
            rho = "rho0p25"
        elif "rho0" in action_id:
            rho = "rho0"
        else:
            rho = "unknown"

        item = {
            "action_id": action_id,
            "rho": rho,
            "fec_gain": fec_gain,
            "q_eff": q_eff,
            "reward": reward,
        }
        items.append(item)
        by_rho[rho].append(item)

def mean(xs):
    return float(np.mean(xs)) if xs else 0.0

def nonzero(xs):
    return int(sum(1 for x in xs if abs(float(x)) > 1e-12))

print("\n===== STEP7 FEC REWARD PROBE SUMMARY =====")
print("model_dir =", model_dir)
print("max_samples =", max_samples)
print("total_reward_items =", len(items))

print("\n===== by rho =====")
for rho in sorted(by_rho.keys()):
    xs = by_rho[rho]
    fecs = [x["fec_gain"] for x in xs]
    qs = [x["q_eff"] for x in xs]
    rs = [x["reward"] for x in xs]
    print(
        rho,
        "count =", len(xs),
        "fec_gain_mean =", mean(fecs),
        "fec_gain_nonzero =", nonzero(fecs),
        "q_eff_mean =", mean(qs),
        "reward_mean =", mean(rs),
    )

print("\n===== sample fec_gain > 0 =====")
shown = 0
for x in items:
    if x["fec_gain"] > 0:
        print(x)
        shown += 1
        if shown >= 12:
            break
if shown == 0:
    print("NO fec_gain > 0 samples found")
