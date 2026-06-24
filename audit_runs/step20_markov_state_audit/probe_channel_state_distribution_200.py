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

def add_num(stats, key, val):
    try:
        stats[key].append(float(val))
    except Exception:
        pass

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

state_counter = Counter()
profile_counter = Counter()
loss_counter = Counter()
budget_state_counter = Counter()
temporal_counter = Counter()
stats_by_state = defaultdict(lambda: defaultdict(list))

with torch.no_grad():
    for idx, batch in enumerate(loader):
        if idx >= MAX_SAMPLES:
            break

        # 尝试直接看 data_dict 里有没有 channel_state_ids
        ego = batch.get("ego", {})
        for key in ["channel_state_ids", "channel_states", "link_state_ids", "link_states"]:
            if isinstance(ego, dict) and key in ego:
                v = ego[key]
                try:
                    if torch.is_tensor(v):
                        vals = v.detach().cpu().view(-1).tolist()
                    else:
                        vals = list(v) if isinstance(v, (list, tuple)) else [v]
                    for x in vals:
                        budget_state_counter[f"data_dict:{key}:{x}"] += 1
                except Exception:
                    budget_state_counter[f"data_dict:{key}:unreadable"] += 1

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

            # 抓各种 channel profile / state 字段
            prof = d.get("channel_profile")
            if isinstance(prof, dict):
                state = str(prof.get("state_name", prof.get("channel_state", "missing")))
                state_counter[state] += 1
                temporal_counter[str(prof.get("temporal_source", "missing"))] += 1

                lr = prof.get("loss_rate", prof.get("plr", None))
                bw = prof.get("bandwidth_mbps", None)
                delay = prof.get("delay_ms", prof.get("fixed_delay_ms", None))
                if lr is not None:
                    add_num(stats_by_state[state], "loss_rate", lr)
                    loss_counter[str(lr)] += 1
                if bw is not None:
                    add_num(stats_by_state[state], "bandwidth_mbps", bw)
                if delay is not None:
                    add_num(stats_by_state[state], "delay_ms", delay)

            for k in ["channel_state", "state_name", "sender_state", "link_state"]:
                if k in d:
                    profile_counter[f"{k}:{d[k]}"] += 1

def stat(xs):
    if not xs:
        return None
    return {
        "count": len(xs),
        "mean": sum(xs) / len(xs),
        "min": min(xs),
        "max": max(xs),
    }

result = {
    "max_samples": MAX_SAMPLES,
    "records_seen": seen_records,
    "data_dict_state_fields": dict(budget_state_counter),
    "channel_profile_state_counter": dict(state_counter),
    "raw_state_field_counter": dict(profile_counter),
    "loss_rate_counter": dict(loss_counter),
    "temporal_source_counter": dict(temporal_counter),
    "stats_by_state": {
        s: {k: stat(v) for k, v in fields.items()}
        for s, fields in stats_by_state.items()
    }
}

print(json.dumps(result, indent=2, ensure_ascii=False))
