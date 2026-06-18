import copy
import torch
from torch.utils.data import DataLoader
from collections import Counter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

model_dir = "opencood/logs/main_opv2v_where2comm_grace_full"
yaml_path = model_dir + "/config.yaml"
max_samples = 500

def find_oracle(comm):
    # 只查 comm 的一层属性，避免递归爆栈
    for name, obj in getattr(comm, "__dict__", {}).items():
        if hasattr(obj, "_quant_select_counts") and hasattr(obj, "_rho_select_counts") and hasattr(obj, "_cache_select_counts"):
            print("[INFO] found oracle attr:", name, type(obj))
            return obj
    print("[WARN] oracle not found from comm.__dict__")
    return None

def count_sum(d):
    if not isinstance(d, dict):
        return 0
    s = 0
    for v in d.values():
        if isinstance(v, dict):
            s += count_sum(v)
        else:
            try:
                s += int(v)
            except Exception:
                pass
    return s

def safe_pending_len(comm):
    rb = getattr(comm, "pending_reward", None)
    if rb is None:
        return -1
    for name in ["pending", "buffer", "items", "_pending", "_buffer"]:
        x = getattr(rb, name, None)
        if x is not None:
            try:
                return len(x)
            except Exception:
                pass
    return -2

hypes = yaml_utils.load_yaml(yaml_path)
dataset = build_dataset(hypes, visualize=False, train=False)
loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    collate_fn=dataset.collate_batch_test,
    shuffle=False,
    pin_memory=False,
)

model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(model_dir, model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

comm = model.arce_comm
oracle = find_oracle(comm)

if oracle is not None:
    quant_before = copy.deepcopy(oracle._quant_select_counts)
    rho_before = copy.deepcopy(oracle._rho_select_counts)
    cache_before = copy.deepcopy(oracle._cache_select_counts)
else:
    quant_before = rho_before = cache_before = {}

pending_after_each_frame = []

with torch.no_grad():
    for idx, batch_data in enumerate(loader):
        if idx >= max_samples:
            break
        batch_data = train_utils.to_device(batch_data, device)
        _ = model(batch_data["ego"])

        pending_after_each_frame.append(safe_pending_len(comm))

        if idx % 100 == 0:
            print(f"[learning_loop_probe] idx={idx}, pending_after={pending_after_each_frame[-1]}")

records = comm.get_records()

cnt = Counter()
debug_samples = []

for r in records:
    if "reward_update" not in r:
        continue

    ru = r["reward_update"]
    cnt["reward_update_records"] += 1
    cnt["reward_num_updated_sum"] += int(ru.get("num_updated", 0))

    for lr in ru.get("link_rewards", []):
        aid = str(lr.get("action_id", "")).lower()
        if "send0" in aid:
            cnt["send0_updated"] += 1
        elif "send1" in aid:
            cnt["send1_updated"] += 1

        info = lr.get("info", lr)
        dbg = None
        if isinstance(info, dict):
            dbg = info.get("policy_update_debug", None)

        if isinstance(dbg, dict):
            cnt["policy_update_debug_records"] += 1
            cnt[f"context_dim_{dbg.get('context_dim')}"] += 1
            cnt[f"policy_t_delta_{dbg.get('policy_t_delta')}"] += 1

            if len(debug_samples) < 8:
                debug_samples.append(dbg)
        else:
            cnt["missing_policy_update_debug"] += 1

if oracle is not None:
    quant_after = oracle._quant_select_counts
    rho_after = oracle._rho_select_counts
    cache_after = oracle._cache_select_counts

    quant_before_sum = count_sum(quant_before)
    rho_before_sum = count_sum(rho_before)
    cache_before_sum = count_sum(cache_before)

    quant_after_sum = count_sum(quant_after)
    rho_after_sum = count_sum(rho_after)
    cache_after_sum = count_sum(cache_after)
else:
    quant_before_sum = rho_before_sum = cache_before_sum = -1
    quant_after_sum = rho_after_sum = cache_after_sum = -1

pending_nonzero = sum(1 for x in pending_after_each_frame if x not in (0, -2))

print("\n===== STEP6 LEARNING LOOP PROBE SUMMARY =====")
print("model_dir =", model_dir)
print("max_samples =", max_samples)
print("total_records =", len(records))

for k in sorted(cnt.keys()):
    print(f"{k} = {cnt[k]}")

print("\n===== pending reward buffer =====")
print("pending_after_each_frame_unique =", sorted(set(pending_after_each_frame)))
print("pending_nonzero_frames =", pending_nonzero)

print("\n===== warm-up count sums =====")
print("quant_before_sum =", quant_before_sum)
print("quant_after_sum  =", quant_after_sum)
print("quant_delta      =", quant_after_sum - quant_before_sum)
print("rho_before_sum   =", rho_before_sum)
print("rho_after_sum    =", rho_after_sum)
print("rho_delta        =", rho_after_sum - rho_before_sum)
print("cache_before_sum =", cache_before_sum)
print("cache_after_sum  =", cache_after_sum)
print("cache_delta      =", cache_after_sum - cache_before_sum)

print("\n===== policy update debug samples =====")
for x in debug_samples:
    print(x)
