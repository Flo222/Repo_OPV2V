import traceback
import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils

MODEL_DIR = "opencood/logs/main_opv2v_where2comm_grace_full"
YAML_PATH = MODEL_DIR + "/config.yaml"

hypes = yaml_utils.load_yaml(YAML_PATH)
dataset = build_dataset(hypes, visualize=False, train=False)
model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(MODEL_DIR, model)
model = model.cuda().eval()

idx = 0
batch_data = dataset.collate_batch_test([dataset[idx]])
batch_data = train_utils.to_device(batch_data, torch.device("cuda"))

with torch.no_grad():
    output_dict = model(batch_data["ego"])

print("output_dict keys:", output_dict.keys())
print("psm shape:", tuple(output_dict["psm"].shape))

try:
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
        batch_data, output_dict
    )
    print("post_process OK")
    print("pred_box:", None if pred_box_tensor is None else tuple(pred_box_tensor.shape))
    print("pred_score:", None if pred_score is None else tuple(pred_score.shape))
    print("gt_box:", None if gt_box_tensor is None else tuple(gt_box_tensor.shape))

    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    for thr in [0.3, 0.5, 0.7]:
        print("caluclate_tp_fp thr =", thr)
        eval_utils.caluclate_tp_fp(
            pred_box_tensor,
            pred_score,
            gt_box_tensor,
            result_stat,
            thr,
        )
        print("stat:", result_stat[thr])

    for thr in [0.3, 0.5, 0.7]:
        print("calculate_ap thr =", thr)
        try:
            ap, _, _ = eval_utils.calculate_ap(result_stat, thr, False)
        except TypeError:
            ap, _, _ = eval_utils.calculate_ap(result_stat, thr)
        print("AP", thr, "=", ap)

except Exception:
    print("===== FULL TRACEBACK =====")
    traceback.print_exc()
