import torch
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils

model_dir = "opencood/logs/main_opv2v_where2comm_grace_full"
yaml_path = model_dir + "/config.yaml"

hypes = yaml_utils.load_yaml(yaml_path)
model = train_utils.create_model(hypes)
_, model = train_utils.load_saved_model(model_dir, model)

print("===== candidate modules =====")
for name, module in model.named_modules():
    cls = module.__class__.__name__
    low = (name + " " + cls).lower()
    if "arce" in low or "comm" in low or "c2mab" in low:
        print(name, "=>", cls)

print("===== direct attrs =====")
for name in dir(model):
    if "arce" in name.lower() or "comm" in name.lower() or "fusion" in name.lower():
        try:
            obj = getattr(model, name)
            print(name, "=>", type(obj))
        except Exception as e:
            print(name, "=> ERROR", e)
