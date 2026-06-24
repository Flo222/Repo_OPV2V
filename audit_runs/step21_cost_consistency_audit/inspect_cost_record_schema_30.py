import json
from collections import Counter

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

MODEL_DIR = "opencood/logs/main_opv2v_where2comm_grace_full"
MAX_SAMPLES = 30
MAX_PRINT = 80

def move_to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_cuda(v) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_cuda(v) for v in x)
    return x

def walk(obj, path="root"):
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(obj, tuple):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}({i})")

def slim_dict(d):
    keep_keys = [
        "type", "record_type", "event", "action_id", "selected_action_id",
        "sender_id", "ego_id", "channel_state", "state_name",
        "tx_bytes", "rx_bytes", "actual_tx_bytes", "actual_transmitted_bytes",
        "actual_received_bytes", "estimated_tx_bytes", "estimated_cost_bytes",
        "cost_bytes", "allocated_budget_bytes", "budget_bytes", "link_budget_bytes",
        "selected_sender_ids", "selected_action_ids",
        "budget_consistency", "system_budget", "pdf_action", "dc2mab",
        "channel_profile"
    ]
    out = {}
    for k in keep_keys:
        if k in d:
            v = d[k]
            if isinstance(v, dict):
                out[k] = {kk: vv for kk, vv in v.items() if kk in [
                    "action_id", "send", "quant", "rho", "cache",
                    "state_name", "loss_rate", "plr", "bandwidth_mbps",
                    "delay_ms", "estimated_tx_bytes", "allocated_budget_bytes",
                    "actual_tx_bytes", "actual_transmitted_bytes",
                    "actual_over_est", "actual_over_allocated"
                ]}
            else:
                out[k] = v
    out["_keys"] = sorted(list(d.keys()))
    return out

hypes = yaml_utils.load_yaml(MODEL_DIR + "/config.yaml")
dataset = build_dataset(hypes, visualize=False, train=False)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=dataset.collate_batch_test,
)

model = train_utils.create_model(hypes).cuda()
_, model = train_utils.load_saved_model(MODEL_DIR, model)
model.eval()

seen_records = 0
printed = []
path_counter = Counter()
key_counter = Counter()

interesting_keys = {
    "actual_tx_bytes", "actual_transmitted_bytes", "estimated_tx_bytes",
    "estimated_cost_bytes", "cost_bytes", "allocated_budget_bytes",
    "budget_consistency", "pdf_action", "dc2mab", "selected_action_ids"
}

with torch.no_grad():
    for idx, batch in enumerate(loader):
        if idx >= MAX_SAMPLES:
            break
        batch = move_to_cuda(batch)
        _ = model(batch["ego"])

        records = model.arce_comm.get_records() if hasattr(model, "arce_comm") else []
        new_records = records[seen_records:] if len(records) >= seen_records else records
        seen_records = len(records)

        for path, d in walk(new_records):
            if not isinstance(d, dict):
                continue
            if any(k in d for k in interesting_keys):
                path_counter[path] += 1
                for k in d.keys():
                    key_counter[k] += 1
                if len(printed) < MAX_PRINT:
                    printed.append({
                        "path": path,
                        "record": slim_dict(d),
                    })

result = {
    "max_samples": MAX_SAMPLES,
    "records_seen": seen_records,
    "interesting_path_counter_top50": path_counter.most_common(50),
    "interesting_key_counter_top80": key_counter.most_common(80),
    "examples": printed,
}

print(json.dumps(result, indent=2, ensure_ascii=False))
