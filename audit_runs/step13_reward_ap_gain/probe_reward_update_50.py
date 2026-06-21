import json
from collections import Counter

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

MODEL_DIR = "opencood/logs/main_opv2v_where2comm_grace_full"
MAX_SAMPLES = 50

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

reward_types = Counter()
rewards = []
ap_gains = []
weighted_ap_gains = []
examples = []

with torch.no_grad():
    for idx, batch in enumerate(loader):
        if idx >= MAX_SAMPLES:
            break

        batch = move_to_cuda(batch)
        out = model(batch["ego"])

        # reward_update is appended into arce_comm records.
        records = []
        if hasattr(model, "arce_comm") and hasattr(model.arce_comm, "get_records"):
            records = model.arce_comm.get_records()

        for d in walk(records):
            ru = d.get("reward_update", None)
            if not isinstance(ru, dict):
                continue

            for lr in ru.get("link_rewards", []):
                if not isinstance(lr, dict):
                    continue

                rt = str(lr.get("reward_type", "missing"))
                reward_types[rt] += 1

                if "reward" in lr:
                    rewards.append(float(lr["reward"]))
                if "ap_proxy_gain" in lr:
                    ap_gains.append(float(lr["ap_proxy_gain"]))
                if "weighted_ap_proxy_gain" in lr:
                    weighted_ap_gains.append(float(lr["weighted_ap_proxy_gain"]))

                if len(examples) < 8:
                    examples.append({
                        "reward_type": lr.get("reward_type"),
                        "reward": lr.get("reward"),
                        "ap_proxy_gain": lr.get("ap_proxy_gain"),
                        "weighted_ap_proxy_gain": lr.get("weighted_ap_proxy_gain"),
                        "normalized_cost": lr.get("normalized_cost"),
                        "delay_norm": lr.get("delay_norm"),
                        "quant_loss": lr.get("quant_loss"),
                        "action_id": lr.get("action_id"),
                        "quant_mode": lr.get("quant_mode"),
                        "policy_update_debug": lr.get("policy_update_debug"),
                    })

def stat(xs):
    if not xs:
        return None
    return {
        "count": len(xs),
        "min": min(xs),
        "max": max(xs),
        "mean": sum(xs) / len(xs),
    }

result = {
    "max_samples": MAX_SAMPLES,
    "reward_type_counter": dict(reward_types),
    "reward_stat": stat(rewards),
    "ap_proxy_gain_stat": stat(ap_gains),
    "weighted_ap_proxy_gain_stat": stat(weighted_ap_gains),
    "examples": examples,
}

print(json.dumps(result, indent=2, ensure_ascii=False))
