from __future__ import print_function
import os
import sys
import yaml

model_dir = sys.argv[1]
path = os.path.join(model_dir, "config.yaml")
with open(path, "r") as f:
    cfg = yaml.safe_load(f) or {}
fusion = cfg.get("fusion", {}) or {}
core = str(fusion.get("core_method", "")).lower()
# Generic OpenCOOD inference dispatch. Custom methods are tested with intermediate first.
if "latefusion" in core:
    print("late")
elif "earlyfusion" in core:
    print("early")
else:
    print("intermediate")
