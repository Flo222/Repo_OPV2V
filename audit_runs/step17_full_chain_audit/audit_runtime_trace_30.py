import json
import math
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

MODEL_DIR = "opencood/logs/main_opv2v_where2comm_grace_full"
MAX_SAMPLES = 30

errors = []
warnings = []

def fail(name, detail):
    errors.append({"name": name, "detail": detail})

def warn(name, detail):
    warnings.append({"name": name, "detail": detail})

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
    "max_samples": MAX_SAMPLES,
    "model_context_dim": getattr(getattr(model, "arce_comm", None), "context_dim", None),
    "include_cav_confidence": getattr(getattr(model, "arce_comm", None), "include_cav_confidence", None),
}

counters = {
    "ap_proxy_used": Counter(),
    "collab_confidence_source": Counter(),
    "reward_type": Counter(),
    "policy_update_context_dim": Counter(),
    "action_id": Counter(),
    "send": Counter(),
    "old_reward_field_seen": Counter(),
    "feedback_weight_formula_ok": Counter(),
}

numeric = defaultdict(list)
examples = {
    "ap_proxy": [],
    "reward": [],
    "feedback_weight": [],
}

seen_records = 0

with torch.no_grad():
    for idx, batch in enumerate(loader):
        if idx >= MAX_SAMPLES:
            break

        batch = move_to_cuda(batch)
        out = model(batch["ego"])

        records = []
        if hasattr(model, "arce_comm") and hasattr(model.arce_comm, "get_records"):
            records = model.arce_comm.get_records()

        # Only inspect newly appended records to avoid repeated counting.
        new_records = records[seen_records:] if len(records) >= seen_records else records
        seen_records = len(records)

        # Inspect both model output and ARCE records.
        runtime_objs = [out, new_records]

        for d in walk(runtime_objs):
            if not isinstance(d, dict):
                continue

            if "ap_proxy_used" in d:
                counters["ap_proxy_used"][str(bool(d["ap_proxy_used"]))] += 1
                if len(examples["ap_proxy"]) < 5:
                    examples["ap_proxy"].append({
                        "ap_proxy_used": d.get("ap_proxy_used"),
                        "collab_confidence_source": d.get("collab_confidence_source"),
                        "ego_confidence": d.get("ego_confidence"),
                        "collab_confidence": d.get("collab_confidence"),
                    })

            if "collab_confidence_source" in d:
                counters["collab_confidence_source"][str(d["collab_confidence_source"])] += 1

            ru = d.get("reward_update", None)
            if isinstance(ru, dict):
                for lr in ru.get("link_rewards", []):
                    if not isinstance(lr, dict):
                        continue

                    rt = str(lr.get("reward_type", "missing"))
                    counters["reward_type"][rt] += 1

                    aid = str(lr.get("action_id", "missing"))
                    counters["action_id"][aid] += 1
                    if aid.startswith("send0"):
                        counters["send"]["0"] += 1
                    elif aid.startswith("send1"):
                        counters["send"]["1"] += 1

                    for old_key in ["q_eff", "fec_gain", "cache_term", "alpha_q", "alpha_r", "alpha_t"]:
                        if old_key in lr:
                            counters["old_reward_field_seen"][old_key] += 1

                    for k in ["reward", "ap_proxy_gain", "weighted_ap_proxy_gain", "normalized_cost", "delay_norm", "quant_loss"]:
                        if k in lr:
                            try:
                                numeric[k].append(float(lr[k]))
                            except Exception:
                                pass

                    pud = lr.get("policy_update_debug", {})
                    if isinstance(pud, dict):
                        if "context_dim" in pud:
                            counters["policy_update_context_dim"][str(pud["context_dim"])] += 1

                        # Verify corrupted-feedback weight:
                        # C = loss_rate, w = exp(-C), if both fields appear.
                        cp = pud.get("channel_profile", {})
                        if isinstance(cp, dict) and "loss_rate" in cp and "feedback_weight" in pud:
                            try:
                                loss = float(cp["loss_rate"])
                                w = float(pud["feedback_weight"])
                                expected = math.exp(-loss)
                                ok = abs(w - expected) < 1e-6
                                counters["feedback_weight_formula_ok"][str(ok)] += 1
                                numeric["feedback_corruption_C_from_loss"].append(loss)
                                numeric["feedback_weight"].append(w)
                                if len(examples["feedback_weight"]) < 8:
                                    examples["feedback_weight"].append({
                                        "action_id": aid,
                                        "loss_rate_as_C": loss,
                                        "feedback_weight": w,
                                        "expected_exp_neg_C": expected,
                                        "formula_ok": ok,
                                    })
                            except Exception as e:
                                warn("feedback_weight_check_error", repr(e))

                    if len(examples["reward"]) < 8:
                        examples["reward"].append({
                            "action_id": aid,
                            "reward_type": lr.get("reward_type"),
                            "reward": lr.get("reward"),
                            "ap_proxy_gain": lr.get("ap_proxy_gain"),
                            "weighted_ap_proxy_gain": lr.get("weighted_ap_proxy_gain"),
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

# Strong assertions.
if summary["model_context_dim"] != 7:
    fail("model_context_dim", "expected 7, got {}".format(summary["model_context_dim"]))

if summary["include_cav_confidence"] is not True:
    fail("include_cav_confidence", "expected True, got {}".format(summary["include_cav_confidence"]))

if not counters["policy_update_context_dim"]:
    fail("policy_update_context_dim", "no policy_update_debug.context_dim found")
elif set(counters["policy_update_context_dim"].keys()) != {"7"}:
    fail("policy_update_context_dim", dict(counters["policy_update_context_dim"]))

if not counters["reward_type"]:
    fail("reward_type", "no reward_type found")
elif set(counters["reward_type"].keys()) != {"ap_proxy_gain_dominated"}:
    fail("reward_type", dict(counters["reward_type"]))

if counters["old_reward_field_seen"]:
    fail("old_reward_field_seen", dict(counters["old_reward_field_seen"]))

if counters["feedback_weight_formula_ok"].get("False", 0) > 0:
    fail("feedback_weight_formula", dict(counters["feedback_weight_formula_ok"]))

if not counters["feedback_weight_formula_ok"]:
    warn("feedback_weight_formula", "no feedback_weight/channel_profile pair found in debug")

if not counters["ap_proxy_used"]:
    warn("ap_proxy_used", "no ap_proxy_used debug field found at runtime")
elif counters["ap_proxy_used"].get("True", 0) == 0:
    fail("ap_proxy_used", dict(counters["ap_proxy_used"]))

result = {
    "summary": summary,
    "num_fail": len(errors),
    "num_warn": len(warnings),
    "failures": errors,
    "warnings": warnings,
    "counters": {k: dict(v) for k, v in counters.items()},
    "numeric_stats": {k: stat(v) for k, v in numeric.items()},
    "examples": examples,
}

print(json.dumps(result, indent=2, ensure_ascii=False))

if errors:
    raise SystemExit(1)
