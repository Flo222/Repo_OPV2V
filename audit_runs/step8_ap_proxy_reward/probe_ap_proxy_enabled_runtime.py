import json
import torch
from pathlib import Path

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

MODEL_DIR = "opencood/logs/main_opv2v_where2comm_grace_full"
YAML_PATH = MODEL_DIR + "/config.yaml"

hypes = yaml_utils.load_yaml(YAML_PATH)
dataset = build_dataset(hypes, visualize=False, train=False)
model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(MODEL_DIR, model)
model = model.cuda().eval()

rows = []

for idx in range(5):
    batch_data = dataset.collate_batch_test([dataset[idx]])
    batch_data = train_utils.to_device(batch_data, torch.device("cuda"))

    with torch.no_grad():
        out = model(batch_data["ego"])

    comm_info = out.get("comm_info", {})
    reward_update = comm_info.get("arce_reward_update", {})
    ap_proxy = reward_update.get("ap_proxy_reward", {})
    ego_ap_proxy = reward_update.get("ego_ap_proxy_reward", {})

    rows.append({
        "idx": idx,
        "ap_proxy_enabled": ap_proxy.get("ap_proxy_enabled"),
        "ap_proxy_used": ap_proxy.get("ap_proxy_used"),
        "ap_proxy_error": ap_proxy.get("ap_proxy_error"),
        "collab_confidence_source": ap_proxy.get("collab_confidence_source"),
        "collab_confidence": ap_proxy.get("collab_confidence"),
        "ego_ap_proxy_used": ego_ap_proxy.get("ap_proxy_used"),
        "ego_confidence_source": ego_ap_proxy.get("collab_confidence_source"),
        "ego_confidence": ego_ap_proxy.get("collab_confidence"),
        "ap_proxy_delta": reward_update.get("ap_proxy_delta"),
        "num_updated": reward_update.get("num_updated"),
        "mean_reward": reward_update.get("mean_reward"),
    })

res = {
    "model_has_ap_proxy_enabled_attr": bool(getattr(model, "ap_proxy_enabled", False)),
    "model_ap_proxy_error": getattr(model, "ap_proxy_error", None),
    "rows": rows,
}

out_path = Path("audit_runs/step8_ap_proxy_reward/probe_ap_proxy_enabled_runtime.json")
out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, indent=2, ensure_ascii=False))
