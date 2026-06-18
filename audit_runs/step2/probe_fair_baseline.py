import torch
from collections import Counter
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

yaml_path = "opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0/config.yaml"
model_dir = "opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0"
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

state_counter = Counter()
action_counter = Counter()
quant_counter = Counter()
rho_counter = Counter()
cache_counter = Counter()

sum_tx = 0.0
sum_rx = 0.0
sum_budget = 0.0
violations = 0
records_total = 0


def walk_records(obj):
    if obj is None:
        return
    if isinstance(obj, dict):
        for key in ["records", "communication_records", "selected_records"]:
            if key in obj and isinstance(obj[key], list):
                for r in obj[key]:
                    yield r

        if "dc2mab_superarm" in obj:
            sa = obj.get("dc2mab_superarm") or {}
            for r in sa.get("selected_records", []) or []:
                yield r

        for v in obj.values():
            yield from walk_records(v)

    elif isinstance(obj, list):
        for x in obj:
            yield from walk_records(x)


with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break

        batch_data = train_utils.to_device(batch_data, device)
        out = model(batch_data["ego"])

        comm_info = out.get("comm_info", None)
        records = list(walk_records(comm_info))

        if idx % 10 == 0:
            print(f"[probe] idx={idx}, records={len(records)}")

        for r in records:
            if not isinstance(r, dict):
                continue

            records_total += 1

            state = str(
                r.get("channel_state")
                or r.get("state")
                or r.get("state_name")
                or r.get("active_channel_state")
                or "unknown"
            ).lower()
            state_counter[state] += 1

            action = r.get("action_id") or r.get("action") or r.get("policy_action") or ""
            action_counter[str(action)] += 1

            q = r.get("quant_mode") or r.get("quant") or ""
            if q:
                quant_counter[str(q).lower()] += 1

            rho = r.get("rho", r.get("redundancy_ratio", None))
            if rho is not None:
                try:
                    rho_counter[float(rho)] += 1
                except Exception:
                    rho_counter[str(rho)] += 1

            cache = r.get("cache_enabled", r.get("cache", None))
            if cache is not None:
                cache_counter[str(cache)] += 1

            tx = float(
                r.get("tx_bytes",
                r.get("actual_transmitted_bytes",
                r.get("transmitted_bytes", 0.0)))
            )
            rx = float(
                r.get("rx_bytes",
                r.get("actual_received_bytes",
                r.get("received_bytes", 0.0)))
            )
            budget = float(
                r.get("budget_bytes",
                r.get("allocated_budget_bytes",
                r.get("link_budget_bytes", 0.0)))
            )

            sum_tx += tx
            sum_rx += rx
            sum_budget += budget

            if budget > 0 and tx > budget + 1e-6:
                violations += 1

print("\n===== FAIR BASELINE PROBE SUMMARY =====")
print("dataset_len =", len(dataset))
print("max_samples =", max_samples)
print("records_total =", records_total)
print("state_counter =", state_counter)
print("action_counter_top10 =", action_counter.most_common(10))
print("quant_counter =", quant_counter)
print("rho_counter =", rho_counter)
print("cache_counter =", cache_counter)
print("sum_tx =", sum_tx)
print("sum_rx =", sum_rx)
print("sum_budget =", sum_budget)
print("rx_le_tx =", sum_rx <= sum_tx + 1e-6)
print("budget_violations =", violations)
print("budget_ratio =", sum_tx / max(sum_budget, 1.0))
