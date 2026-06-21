import json
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

MODEL_DIR = "opencood/logs/main_opv2v_where2comm_grace_full"
MAX_SAMPLES = 200

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

def pack(v):
    return {
        "count": v["count"],
        "sum_MB": v["sum"] / 1024 / 1024,
        "mean_bytes": v["sum"] / max(v["count"], 1),
        "min_bytes": v["min"],
        "max_bytes": v["max"],
    }

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

selected_action_counter = Counter()
selected_send_counter = Counter()
selected_quant_counter = Counter()
selected_rho_counter = Counter()
selected_cache_counter = Counter()
num_selected_counter = Counter()
frame_level_records = 0
actual_byte_stats = defaultdict(lambda: {"count": 0, "sum": 0.0, "min": None, "max": None})

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

            # 只统计真正完成通信/融合后的实际字节字段
            for k in ["actual_tx_bytes", "tx_bytes", "rx_bytes"]:
                if k in d:
                    add_stat(actual_byte_stats, k, d[k])

            # 只统计 frame-level 最终选择，不统计 ranked/candidate/debug
            if "selected_action_ids" not in d:
                continue

            selected = d.get("selected_action_ids") or []
            if not isinstance(selected, (list, tuple)):
                continue

            frame_level_records += 1
            num_selected_counter[str(len(selected))] += 1

            for aid in selected:
                aid = str(aid)
                selected_action_counter[aid] += 1

                if aid.startswith("send0"):
                    selected_send_counter["0"] += 1
                elif aid.startswith("send1"):
                    selected_send_counter["1"] += 1

                parts = aid.split("_")
                if len(parts) >= 2:
                    selected_quant_counter[parts[1]] += 1

                for part in parts:
                    if part.startswith("rho"):
                        selected_rho_counter[part.replace("rho", "")] += 1
                    if part.startswith("cache"):
                        selected_cache_counter[part.replace("cache", "")] += 1

result = {
    "max_samples": MAX_SAMPLES,
    "records_seen": seen_records,
    "frame_level_selected_records": frame_level_records,
    "selected_action_counter_top30": selected_action_counter.most_common(30),
    "selected_send_counter": dict(selected_send_counter),
    "selected_quant_counter": dict(selected_quant_counter),
    "selected_rho_counter": dict(selected_rho_counter),
    "selected_cache_counter": dict(selected_cache_counter),
    "num_selected_counter": dict(num_selected_counter),
    "actual_byte_stats": {k: pack(v) for k, v in actual_byte_stats.items()},
}

print(json.dumps(result, indent=2, ensure_ascii=False))
