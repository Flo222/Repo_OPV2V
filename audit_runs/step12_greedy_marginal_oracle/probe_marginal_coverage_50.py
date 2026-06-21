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

vals = []
overlaps = []
ego_init_counter = Counter()
examples = []

with torch.no_grad():
    for idx, batch in enumerate(loader):
        if idx >= MAX_SAMPLES:
            break
        batch = move_to_cuda(batch)
        out = model(batch["ego"])
        comm_info = out.get("comm_info", {})

        for d in walk(comm_info):
            ranked = d.get("ranked", None)
            if isinstance(ranked, list):
                for r in ranked:
                    if not isinstance(r, dict):
                        continue
                    if "marginal_coverage" in r:
                        vals.append(float(r["marginal_coverage"]))
                    if "overlap_with_selected" in r:
                        overlaps.append(float(r["overlap_with_selected"]))
                    if "ego_mask_initialized" in r:
                        ego_init_counter[str(bool(r["ego_mask_initialized"]))] += 1
                    if len(examples) < 10 and "marginal_coverage" in r:
                        examples.append({
                            "action_id": r.get("action_id"),
                            "ratio": r.get("ratio"),
                            "ucb": r.get("ucb"),
                            "gain": r.get("gain"),
                            "marginal_coverage": r.get("marginal_coverage"),
                            "overlap_with_selected": r.get("overlap_with_selected"),
                            "ego_mask_initialized": r.get("ego_mask_initialized"),
                            "exploration_bonus": r.get("exploration_bonus"),
                        })

result = {
    "max_samples": MAX_SAMPLES,
    "marginal_count": len(vals),
    "marginal_min": min(vals) if vals else None,
    "marginal_max": max(vals) if vals else None,
    "marginal_mean": sum(vals) / len(vals) if vals else None,
    "overlap_min": min(overlaps) if overlaps else None,
    "overlap_max": max(overlaps) if overlaps else None,
    "overlap_mean": sum(overlaps) / len(overlaps) if overlaps else None,
    "ego_mask_initialized_counter": dict(ego_init_counter),
    "examples": examples,
}

print(json.dumps(result, indent=2, ensure_ascii=False))
