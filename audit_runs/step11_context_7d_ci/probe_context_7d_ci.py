import json
from collections import Counter

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

MODEL_DIR = "opencood/logs/main_opv2v_where2comm_grace_full"
MAX_SAMPLES = 20

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

summary = {
    "model_context_dim": None,
    "include_cav_confidence": None,
    "context_len_counter": Counter(),
    "policy_update_context_dim_counter": Counter(),
    "cav_conf_values": [],
    "context_examples": [],
    "errors": [],
}

if hasattr(model, "arce_comm"):
    summary["model_context_dim"] = int(getattr(model.arce_comm, "context_dim", -1))
    summary["include_cav_confidence"] = bool(getattr(model.arce_comm, "include_cav_confidence", False))

with torch.no_grad():
    for idx, batch in enumerate(loader):
        if idx >= MAX_SAMPLES:
            break

        batch = move_to_cuda(batch)
        out = model(batch["ego"])
        comm_info = out.get("comm_info", {})

        for d in walk(comm_info):
            if "context_vector" in d:
                vec = d["context_vector"]
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                summary["context_len_counter"][len(vec)] += 1
                if len(vec) >= 7:
                    summary["cav_conf_values"].append(float(vec[6]))
                if len(summary["context_examples"]) < 5:
                    summary["context_examples"].append(vec)

            ctx = d.get("context", None)
            if isinstance(ctx, dict) and "vector" in ctx:
                vec = ctx["vector"]
                summary["context_len_counter"][len(vec)] += 1
                if len(vec) >= 7:
                    summary["cav_conf_values"].append(float(vec[6]))
                if len(summary["context_examples"]) < 5:
                    summary["context_examples"].append(vec)

            pud = d.get("policy_update_debug", None)
            if isinstance(pud, dict) and "context_dim" in pud:
                summary["policy_update_context_dim_counter"][int(pud["context_dim"])] += 1

summary["context_len_counter"] = dict(summary["context_len_counter"])
summary["policy_update_context_dim_counter"] = dict(summary["policy_update_context_dim_counter"])

vals = summary["cav_conf_values"]
if vals:
    summary["cav_conf_count"] = len(vals)
    summary["cav_conf_min"] = min(vals)
    summary["cav_conf_max"] = max(vals)
    summary["cav_conf_mean"] = sum(vals) / len(vals)
else:
    summary["cav_conf_count"] = 0
    summary["cav_conf_min"] = None
    summary["cav_conf_max"] = None
    summary["cav_conf_mean"] = None

print(json.dumps(summary, indent=2, ensure_ascii=False))
