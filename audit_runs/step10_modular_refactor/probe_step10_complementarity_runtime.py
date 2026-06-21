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

vals_raw, vals_norm, errors = [], [], []

for idx in range(min(30, len(dataset))):
    try:
        batch = dataset.collate_batch_test([dataset[idx]])
        batch = train_utils.to_device(batch, torch.device("cuda"))

        with torch.no_grad():
            out = model(batch["ego"])

        comm_info = out.get("comm_info", {})
        records = comm_info.get("arce_records", []) or []

        for r in records:
            dc = r.get("dc2mab", {})
            prop = dc.get("proposal", {}) if isinstance(dc, dict) else {}
            rec = prop.get("record", {}) if isinstance(prop, dict) else {}
            if "complementarity_raw" in rec:
                vals_raw.append(float(rec.get("complementarity_raw", 0.0)))
            if "complementarity_normalized" in rec:
                vals_norm.append(float(rec.get("complementarity_normalized", 0.0)))

    except Exception as e:
        errors.append({"idx": idx, "error": f"{type(e).__name__}: {e}"})
        break

def stat(xs):
    if not xs:
        return {"count": 0}
    return {
        "count": len(xs),
        "min": min(xs),
        "max": max(xs),
        "mean": sum(xs) / len(xs),
        "num_gt_0": sum(1 for x in xs if x > 0),
    }

res = {
    "num_frames": min(30, len(dataset)),
    "complementarity_raw": stat(vals_raw),
    "complementarity_normalized": stat(vals_norm),
    "errors": errors,
}

out_path = Path("audit_runs/step10_modular_refactor/probe_step10_complementarity_runtime.json")
out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=2))
