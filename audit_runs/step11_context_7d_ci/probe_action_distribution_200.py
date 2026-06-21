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
quant_counter = Counter()
rho_counter = Counter()
cache_counter = Counter()
send_counter = Counter()
context_len_counter = Counter()
cav_conf_by_action = defaultdict(list)
policy_update_dim_counter = Counter()

with torch.no_grad():
    for idx, batch in enumerate(loader):
        if idx >= MAX_SAMPLES:
            break

        batch = move_to_cuda(batch)
        out = model(batch["ego"])
        comm_info = out.get("comm_info", {})

        for d in walk(comm_info):
            action = d.get("action", None)
            if isinstance(action, dict):
                name = str(action.get("name", action.get("action_id", "unknown")))
                action_counter[name] += 1
                quant_counter[str(action.get("quant_mode", "unknown"))] += 1
                rho_counter[str(action.get("redundancy_ratio", "unknown"))] += 1
                cache_counter[str(action.get("cache_enabled", "unknown"))] += 1
                send_counter[str(action.get("send", action.get("send_flag", "unknown")))] += 1

                ctx = d.get("context", None)
                if isinstance(ctx, dict) and "vector" in ctx:
                    vec = ctx["vector"]
                    context_len_counter[len(vec)] += 1
                    if len(vec) >= 7:
                        cav_conf_by_action[name].append(float(vec[6]))

            if "context_vector" in d:
                vec = d["context_vector"]
                context_len_counter[len(vec)] += 1

            pud = d.get("policy_update_debug", None)
            if isinstance(pud, dict) and "context_dim" in pud:
                policy_update_dim_counter[int(pud["context_dim"])] += 1

cav_conf_summary = {}
for k, vals in cav_conf_by_action.items():
    if vals:
        cav_conf_summary[k] = {
            "count": len(vals),
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

result = {
    "max_samples": MAX_SAMPLES,
    "model_context_dim": int(getattr(model.arce_comm, "context_dim", -1)),
    "include_cav_confidence": bool(getattr(model.arce_comm, "include_cav_confidence", False)),
    "action_counter_top20": action_counter.most_common(20),
    "quant_counter": dict(quant_counter),
    "rho_counter": dict(rho_counter),
    "cache_counter": dict(cache_counter),
    "send_counter": dict(send_counter),
    "context_len_counter": dict(context_len_counter),
    "policy_update_dim_counter": dict(policy_update_dim_counter),
    "cav_conf_by_action": cav_conf_summary,
}

print(json.dumps(result, indent=2, ensure_ascii=False))
