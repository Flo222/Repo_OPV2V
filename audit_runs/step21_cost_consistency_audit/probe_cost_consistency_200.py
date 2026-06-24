import json
from collections import defaultdict, Counter

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

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def add(stats, key, val):
    v = to_float(val)
    if v is not None:
        stats[key].append(v)

def stat(xs):
    if not xs:
        return None
    xs = list(xs)
    return {
        "count": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
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

seen_records = 0
stats = defaultdict(list)
by_action = defaultdict(lambda: defaultdict(list))
by_state = defaultdict(lambda: defaultdict(list))
by_quant = defaultdict(lambda: defaultdict(list))
ratio_counter = Counter()
examples = []

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

            # 只看 selected / actually transmitted records，过滤纯候选 proposal
            actual = None
            for k in ["actual_tx_bytes", "actual_transmitted_bytes", "tx_bytes"]:
                if k in d:
                    actual = to_float(d.get(k))
                    if actual is not None:
                        break

            estimated = None
            for k in ["estimated_tx_bytes", "cost_bytes", "estimated_cost_bytes"]:
                if k in d:
                    estimated = to_float(d.get(k))
                    if estimated is not None:
                        break

            allocated = None
            for k in ["allocated_budget_bytes", "budget_bytes", "link_budget_bytes"]:
                if k in d:
                    allocated = to_float(d.get(k))
                    if allocated is not None:
                        break

            # 必须至少有 actual 和 estimated 之一才记录
            if actual is None and estimated is None:
                continue

            action_id = str(d.get("action_id", d.get("selected_action_id", "unknown")))
            channel_profile = d.get("channel_profile", {})
            state = "unknown"
            if isinstance(channel_profile, dict):
                state = str(channel_profile.get("state_name", channel_profile.get("channel_state", "unknown")))
            else:
                state = str(d.get("channel_state", "unknown"))

            quant = "unknown"
            for q in ["fp32", "fp16", "int8", "int4"]:
                if q in action_id:
                    quant = q

            if actual is not None:
                add(stats, "actual_tx_bytes", actual)
                add(by_action[action_id], "actual_tx_bytes", actual)
                add(by_state[state], "actual_tx_bytes", actual)
                add(by_quant[quant], "actual_tx_bytes", actual)

            if estimated is not None:
                add(stats, "estimated_tx_bytes", estimated)
                add(by_action[action_id], "estimated_tx_bytes", estimated)
                add(by_state[state], "estimated_tx_bytes", estimated)
                add(by_quant[quant], "estimated_tx_bytes", estimated)

            if allocated is not None:
                add(stats, "allocated_budget_bytes", allocated)
                add(by_action[action_id], "allocated_budget_bytes", allocated)
                add(by_state[state], "allocated_budget_bytes", allocated)
                add(by_quant[quant], "allocated_budget_bytes", allocated)

            if actual is not None and estimated not in (None, 0.0):
                r = actual / max(estimated, 1e-9)
                add(stats, "actual_over_estimated", r)
                add(by_action[action_id], "actual_over_estimated", r)
                add(by_state[state], "actual_over_estimated", r)
                add(by_quant[quant], "actual_over_estimated", r)
                if r < 0.5:
                    ratio_counter["actual_less_than_half_est"] += 1
                elif r > 2.0:
                    ratio_counter["actual_more_than_2x_est"] += 1
                else:
                    ratio_counter["actual_near_est"] += 1

            if actual is not None and allocated not in (None, 0.0):
                r2 = actual / max(allocated, 1e-9)
                add(stats, "actual_over_allocated", r2)
                add(by_action[action_id], "actual_over_allocated", r2)
                add(by_state[state], "actual_over_allocated", r2)
                add(by_quant[quant], "actual_over_allocated", r2)

            if len(examples) < 20 and (actual is not None or estimated is not None):
                examples.append({
                    "action_id": action_id,
                    "state": state,
                    "quant": quant,
                    "actual_tx_bytes": actual,
                    "estimated_tx_bytes": estimated,
                    "allocated_budget_bytes": allocated,
                    "actual_over_estimated": None if actual is None or estimated in (None, 0.0) else actual / max(estimated, 1e-9),
                    "actual_over_allocated": None if actual is None or allocated in (None, 0.0) else actual / max(allocated, 1e-9),
                })

def pack_group(g, topn=30):
    rows = []
    for k, fields in g.items():
        row = {"key": k}
        row["_count"] = max([len(v) for v in fields.values()] or [0])
        for f, xs in fields.items():
            row[f] = stat(xs)
        rows.append(row)
    rows.sort(key=lambda x: x["_count"], reverse=True)
    return rows[:topn]

result = {
    "max_samples": MAX_SAMPLES,
    "records_seen": seen_records,
    "overall": {k: stat(v) for k, v in stats.items()},
    "ratio_counter": dict(ratio_counter),
    "by_state": pack_group(by_state, topn=20),
    "by_quant": pack_group(by_quant, topn=20),
    "by_action_top30": pack_group(by_action, topn=30),
    "examples": examples,
}

print(json.dumps(result, indent=2, ensure_ascii=False))
