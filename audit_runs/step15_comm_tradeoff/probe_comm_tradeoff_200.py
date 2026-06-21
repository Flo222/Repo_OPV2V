import json
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

MODEL_DIR = "opencood/logs/main_opv2v_where2comm_grace_full"
MAX_SAMPLES = 200

BYTE_KEYS = [
    "cost_bytes",
    "actual_tx_bytes",
    "tx_bytes",
    "rx_bytes",
    "payload_bytes",
    "link_budget_bytes",
    "budget_bytes",
    "selected_bytes",
    "sent_bytes",
]

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

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)
    elif isinstance(obj, tuple):
        for v in obj:
            yield from walk(v)

def add_stat(stats, key, value):
    try:
        v = float(value)
    except Exception:
        return
    stats[key]["count"] += 1
    stats[key]["sum"] += v
    stats[key]["min"] = v if stats[key]["min"] is None else min(stats[key]["min"], v)
    stats[key]["max"] = v if stats[key]["max"] is None else max(stats[key]["max"], v)

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

action_counter = Counter()
send_counter = Counter()
quant_counter = Counter()
rho_counter = Counter()
cache_counter = Counter()
reward_type_counter = Counter()
record_key_counter = Counter()
byte_stats = defaultdict(lambda: {"count": 0, "sum": 0.0, "min": None, "max": None})

seen_records = 0

with torch.no_grad():
    for idx, batch in enumerate(loader):
        if idx >= MAX_SAMPLES:
            break

        batch = move_to_cuda(batch)
        _ = model(batch["ego"])

        records = []
        if hasattr(model, "arce_comm") and hasattr(model.arce_comm, "get_records"):
            records = model.arce_comm.get_records()

        new_records = records[seen_records:] if len(records) >= seen_records else records
        seen_records = len(records)

        for d in walk(new_records):
            if not isinstance(d, dict):
                continue

            for k in d.keys():
                record_key_counter[str(k)] += 1

            for k in BYTE_KEYS:
                if k in d:
                    add_stat(byte_stats, k, d[k])

            if "reward_type" in d:
                reward_type_counter[str(d["reward_type"])] += 1

            aid = d.get("action_id", None)
            if aid is None:
                continue

            aid = str(aid)
            action_counter[aid] += 1

            if aid.startswith("send0"):
                send_counter["0"] += 1
            elif aid.startswith("send1"):
                send_counter["1"] += 1

            parts = aid.split("_")
            if len(parts) >= 2:
                quant_counter[parts[1]] += 1

            for part in parts:
                if part.startswith("rho"):
                    rho_counter[part.replace("rho", "")] += 1
                if part.startswith("cache"):
                    cache_counter[part.replace("cache", "")] += 1

def pack_stats(v):
    return {
        "count": v["count"],
        "sum_MB": v["sum"] / 1024 / 1024,
        "mean_bytes": v["sum"] / max(v["count"], 1),
        "min_bytes": v["min"],
        "max_bytes": v["max"],
    }

result = {
    "max_samples": MAX_SAMPLES,
    "records_seen": seen_records,
    "action_counter_top30": action_counter.most_common(30),
    "send_counter": dict(send_counter),
    "quant_counter": dict(quant_counter),
    "rho_counter": dict(rho_counter),
    "cache_counter": dict(cache_counter),
    "reward_type_counter": dict(reward_type_counter),
    "byte_stats": {k: pack_stats(v) for k, v in byte_stats.items()},
    "record_key_counter_top80": record_key_counter.most_common(80),
}

print(json.dumps(result, indent=2, ensure_ascii=False))
