import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(".").resolve()

FILES = {
    "model": Path("opencood/models/point_pillar_where2comm_arce.py"),
    "fuse": Path("opencood/models/fuse_modules/where2comm_arce_fuse.py"),
    "comm": Path("opencood/comm/arce/arce_c2mab_comm.py"),
    "context": Path("opencood/comm/arce/policies/context_builder.py"),
    "linucb": Path("opencood/comm/arce/policies/discounted_linucb.py"),
    "oracle": Path("opencood/comm/arce/policies/ego_greedy_oracle.py"),
    "reward": Path("opencood/comm/arce/policies/ap_gain_reward.py"),
    "corruption": Path("opencood/comm/arce/policies/feedback_corruption.py"),
}

errors = []
warnings = []
checks = []

def ok(name, detail=""):
    checks.append({"name": name, "status": "PASS", "detail": detail})

def fail(name, detail=""):
    errors.append({"name": name, "status": "FAIL", "detail": detail})

def warn(name, detail=""):
    warnings.append({"name": name, "status": "WARN", "detail": detail})

def read(path):
    if not path.exists():
        fail("file_exists:" + str(path), "missing")
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")

def require_contains(label, text, needle):
    if needle in text:
        ok(label, needle)
    else:
        fail(label, "missing: " + needle)

def require_not_contains(label, text, needle):
    if needle not in text:
        ok(label, "not found: " + needle)
    else:
        fail(label, "unexpected old/rejected logic remains: " + needle)

def py_compile(path):
    cmd = [sys.executable, "-m", "py_compile", str(path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode == 0:
        ok("py_compile:" + str(path))
    else:
        fail("py_compile:" + str(path), p.stderr[-1000:])

# 1. File existence and py_compile.
for key, path in FILES.items():
    if path.exists():
        ok("file_exists:" + key, str(path))
        py_compile(path)
    else:
        fail("file_exists:" + key, str(path))

texts = {k: read(v) for k, v in FILES.items()}

# 2. Step11: 7D context + C_i path.
require_contains("model imports local confidence", texts["model"], "local_cav_confidences_from_psm")
require_contains("model computes psm_single confidence", texts["model"], "local_cav_confidences")
require_contains("fuse accepts local_cav_confidences", texts["fuse"], "local_cav_confidences")
require_contains("comm passes cav_confidence", texts["comm"], "cav_confidence=get_cav_confidence")
require_contains("context builder supports C_i", texts["context"], "include_cav_confidence")
require_contains("context builder field cav_confidence", texts["context"], "cav_confidence")

# 3. AP proxy path.
require_contains("model has AP proxy pkl path", texts["model"], "ap_proxy_dense_rf.pkl")
require_contains("model debug ap_proxy_used", texts["model"], "ap_proxy_used")
require_contains("model dense_ap_proxy source", texts["model"], "dense_ap_proxy")

# 4. Step13 reward path.
require_contains("comm imports AP-gain reward", texts["comm"], "c2mab_ap_gain_reward")
require_contains("comm passes ap_proxy_gain=delta_conf", texts["comm"], "ap_proxy_gain=delta_conf")
require_not_contains("old mixed reward call removed", texts["comm"], "c2mab_link_proxy_reward(")
require_contains("new reward type exists", texts["reward"], "ap_proxy_gain_dominated")
require_not_contains("old reward positive fec/cache not in new reward", texts["reward"], "fec_gain")
require_not_contains("old reward positive q_eff not in new reward", texts["reward"], "q_eff")

# 5. Mainline oracle should be hard threshold 0.001, not rejected Step14 variants.
require_contains("oracle min_marginal_coverage present", texts["oracle"], "min_marginal_coverage")
m = re.search(r"min_marginal_coverage:\s*float\s*=\s*([0-9.eE+-]+)", texts["oracle"])
if m:
    val = float(m.group(1))
    if abs(val - 0.001) < 1e-12:
        ok("oracle min_marginal_coverage default", str(val))
    else:
        fail("oracle min_marginal_coverage default", "expected 0.001, got {}".format(val))
else:
    fail("oracle min_marginal_coverage default", "not found")

require_contains("oracle has hard threshold continue", texts["oracle"], "float(marginal_coverage) < float(self.min_marginal_coverage)")
require_not_contains("soft penalty removed", texts["oracle"], "marginal_soft_factor")
require_not_contains("learned gate removed from mainline", texts["oracle"], "marginal_learned_benefit_gate")
require_not_contains("learned mean removed from mainline", texts["oracle"], "min_learned_mean")

# 6. Step16 corruption-C path.
require_contains("linucb imports corruption weight", texts["linucb"], "channel_corruption_weight")
require_contains("linucb stores last corruption C", texts["linucb"], "last_feedback_corruption_C")
require_contains("linucb stores corruption info", texts["linucb"], "last_feedback_corruption_info")
require_contains("linucb accepts channel_profile", texts["linucb"], "channel_profile")
require_contains("corruption defines C field", texts["corruption"], "feedback_corruption_C")
require_contains("corruption formula exp(-alpha*C)", texts["corruption"], "exp(-alpha * C)")

# 7. Import and signature checks.
try:
    from opencood.comm.arce.policies.discounted_linucb import DiscountedLinUCB
    from opencood.comm.arce.policies.ap_gain_reward import c2mab_ap_gain_reward
    from opencood.comm.arce.policies.feedback_corruption import channel_corruption_weight
    ok("import core policy modules")
except Exception as e:
    fail("import core policy modules", repr(e))
    DiscountedLinUCB = None

if DiscountedLinUCB is not None:
    sig_init = str(inspect.signature(DiscountedLinUCB.__init__))
    sig_update = str(inspect.signature(DiscountedLinUCB.update))
    sig_weight = str(inspect.signature(DiscountedLinUCB._compute_feedback_weight))

    if "context_dim" in sig_init and "feedback_weight_mode" in sig_init:
        ok("DiscountedLinUCB.__init__ signature", sig_init)
    else:
        fail("DiscountedLinUCB.__init__ signature", sig_init)

    if "channel_profile" in sig_update:
        ok("DiscountedLinUCB.update accepts channel_profile", sig_update)
    else:
        fail("DiscountedLinUCB.update accepts channel_profile", sig_update)

    if "channel_profile" in sig_weight:
        ok("_compute_feedback_weight accepts channel_profile", sig_weight)
    else:
        fail("_compute_feedback_weight accepts channel_profile", sig_weight)

    # Minimal unit test for LinUCB update with 7D context.
    try:
        policy = DiscountedLinUCB(
            action_ids=["send1_int8_rho0_cache0_none"],
            context_dim=7,
            feedback_weight_mode="channel_quality",
        )
        ret = policy.update(
            "send1_int8_rho0_cache0_none",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            reward=0.05,
            channel_profile={"loss_rate": 0.2},
        )
        w = policy.last_feedback_weight["send1_int8_rho0_cache0_none"]
        c = policy.last_feedback_corruption_C["send1_int8_rho0_cache0_none"]
        if abs(c - 0.2) < 1e-8 and 0.0 < w <= 1.0:
            ok("LinUCB unit update 7D + corruption C", "C={}, w={}, ret={}".format(c, w, ret))
        else:
            fail("LinUCB unit update 7D + corruption C", "C={}, w={}".format(c, w))
    except Exception as e:
        fail("LinUCB unit update 7D + corruption C", repr(e))

# 8. Reward unit test.
try:
    from opencood.comm.arce.policies.ap_gain_reward import c2mab_ap_gain_reward
    r, info = c2mab_ap_gain_reward(
        ap_proxy_gain=0.1,
        contribution_weight=1.0,
        cost_bytes=100.0,
        budget_bytes=1000.0,
        delay_ms=20.0,
        budget_violation=False,
        quant_mode="int8",
    )
    if info.get("reward_type") == "ap_proxy_gain_dominated" and "ap_proxy_gain" in info:
        ok("AP-gain reward unit", "r={}, info={}".format(r, info))
    else:
        fail("AP-gain reward unit", str(info))
except Exception as e:
    fail("AP-gain reward unit", repr(e))

# 9. Corruption unit test.
try:
    from opencood.comm.arce.policies.feedback_corruption import channel_corruption_weight
    w, info = channel_corruption_weight(loss_rate=0.3, alpha=1.0, floor=0.05)
    if abs(info.get("feedback_corruption_C", -1) - 0.3) < 1e-8 and 0.0 < w <= 1.0:
        ok("corruption C unit", "w={}, info={}".format(w, info))
    else:
        fail("corruption C unit", "w={}, info={}".format(w, info))
except Exception as e:
    fail("corruption C unit", repr(e))

result = {
    "summary": {
        "num_pass": len(checks),
        "num_warn": len(warnings),
        "num_fail": len(errors),
    },
    "failures": errors,
    "warnings": warnings,
    "passes_tail": checks[-30:],
}

print(json.dumps(result, indent=2, ensure_ascii=False))

if errors:
    sys.exit(1)
