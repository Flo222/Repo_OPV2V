import json
import torch
from pathlib import Path

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

yaml_path = "opencood/logs/main_opv2v_where2comm_grace_full/config.yaml"
model_dir = "opencood/logs/main_opv2v_where2comm_grace_full"

hypes = yaml_utils.load_yaml(yaml_path)
dataset = build_dataset(hypes, visualize=False, train=False)
model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(model_dir, model)
model = model.cuda().eval()

batch = dataset.collate_batch_test([dataset[0]])
batch = train_utils.to_device(batch, torch.device("cuda"))

with torch.no_grad():
    out = model(batch["ego"])

def brief(x, depth=0, max_depth=4):
    if depth > max_depth:
        return "..."
    if isinstance(x, dict):
        return {str(k): brief(v, depth + 1, max_depth) for k, v in list(x.items())[:30]}
    if isinstance(x, list):
        return {
            "__type__": "list",
            "__len__": len(x),
            "__first__": brief(x[0], depth + 1, max_depth) if x else None,
        }
    if torch.is_tensor(x):
        return {
            "__type__": "tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
        }
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(type(x))

res = {
    "top_keys": list(out.keys()),
    "comm_info": brief(out.get("comm_info", {}), max_depth=6),
}

# 额外递归查找 complementarity 字段出现在哪里
hits = []

def find_key(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{path}.{k}"
            if "complementarity" in str(k).lower():
                hits.append({"path": kp, "value": brief(v, max_depth=2)})
            find_key(v, kp)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            find_key(v, f"{path}[{i}]")

find_key(out)
res["complementarity_hits"] = hits[:80]

out_path = Path("audit_runs/step10_modular_refactor/inspect_step10_comm_info_structure.json")
out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
