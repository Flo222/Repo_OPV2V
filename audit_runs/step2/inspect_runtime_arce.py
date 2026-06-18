import torch
import pprint
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils

yaml_path = "opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0/config.yaml"
model_dir = "opencood/logs/where2comm_markov_trueloss_fp32_rho0_cache0"

hypes = yaml_utils.load_yaml(yaml_path)
model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(model_dir, model)

def find_arce_modules(obj, prefix="model", depth=0, max_depth=5, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > max_depth:
        return []
    seen.add(id(obj))

    found = []
    name = obj.__class__.__name__.lower()
    if "arce" in name or hasattr(obj, "channel_profiles") or hasattr(obj, "action_policy"):
        found.append((prefix, obj))

    for attr in ["arce_comm", "arce_fixed_comm", "fusion_net", "fusion_net_v2", "where2comm_fuse", "fuse_modules"]:
        if hasattr(obj, attr):
            try:
                child = getattr(obj, attr)
                found.extend(find_arce_modules(child, prefix + "." + attr, depth + 1, max_depth, seen))
            except Exception:
                pass

    if hasattr(obj, "_modules"):
        for k, v in obj._modules.items():
            found.extend(find_arce_modules(v, prefix + "." + k, depth + 1, max_depth, seen))

    return found

mods = find_arce_modules(model)

print("===== FOUND ARCE-LIKE MODULES =====")
for i, (name, m) in enumerate(mods):
    print(f"\n--- [{i}] {name}: {m.__class__.__name__} ---")

    if hasattr(m, "config"):
        print("[config keys]")
        print(list(m.config.keys()) if isinstance(m.config, dict) else type(m.config))

        if isinstance(m.config, dict):
            print("[config.fixed_action]")
            pprint.pprint(m.config.get("fixed_action"))

            print("[config.fixed_policy]")
            pprint.pprint(m.config.get("fixed_policy"))

            print("[config.channel.profiles]")
            pprint.pprint((m.config.get("channel") or {}).get("profiles"))

            print("[config.profiles]")
            pprint.pprint(m.config.get("profiles"))

    if hasattr(m, "channel_profiles"):
        print("[runtime channel_profiles]")
        pprint.pprint(m.channel_profiles)

    if hasattr(m, "action_policy"):
        print("[runtime action_policy class]")
        print(m.action_policy.__class__.__name__)

        if hasattr(m.action_policy, "get_config"):
            print("[runtime action_policy.get_config()]")
            pprint.pprint(m.action_policy.get_config())

        if hasattr(m.action_policy, "actions_by_state"):
            print("[runtime action_policy.actions_by_state]")
            pprint.pprint(m.action_policy.actions_by_state)

        if hasattr(m.action_policy, "state_actions"):
            print("[runtime action_policy.state_actions]")
            pprint.pprint(m.action_policy.state_actions)
