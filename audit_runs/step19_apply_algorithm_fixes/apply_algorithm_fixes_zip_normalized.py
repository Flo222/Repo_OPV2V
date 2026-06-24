from pathlib import Path
import zipfile
import shutil
import py_compile
import subprocess
import time

ROOT = Path(".").resolve()
PATCH_ZIP = ROOT / "algorithm_fixes_20260622.zip"

stamp = time.strftime("%m%d_%H%M%S")
work_dir = ROOT / "audit_runs" / "step19_apply_algorithm_fixes" / f"normalized_apply_{stamp}"
extract_dir = work_dir / "normalized_extract"
backup_dir = work_dir / "backup_before_apply"
extract_dir.mkdir(parents=True, exist_ok=True)
backup_dir.mkdir(parents=True, exist_ok=True)

if not PATCH_ZIP.exists():
    raise SystemExit(f"[ERROR] 找不到补丁包: {PATCH_ZIP}")

print("===== normalize unzip =====")
patched_files = []
with zipfile.ZipFile(PATCH_ZIP, "r") as zf:
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if name.endswith("/"):
            continue

        # 只保留 opencood/ 及其后面的路径
        if "opencood/" not in name:
            continue
        rel = name[name.index("opencood/"):]
        out = extract_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        patched_files.append(rel)

patched_files = sorted(set(patched_files))
if not patched_files:
    raise SystemExit("[ERROR] 补丁包中没有解析出 opencood/ 文件，请检查 zip 内容")

print("===== patch files =====")
for f in patched_files:
    print("[PATCH]", f)

print("\n===== backup current files =====")
for rel in patched_files:
    src = ROOT / rel
    if src.exists():
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print("[BACKUP]", rel)
    else:
        print("[NEW_FILE]", rel)

print("\n===== apply normalized patch =====")
for rel in patched_files:
    src = extract_dir / rel
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print("[APPLY]", rel)

print("\n===== py_compile =====")
failed = []
for rel in patched_files:
    if not rel.endswith(".py"):
        continue
    try:
        py_compile.compile(str(ROOT / rel), doraise=True)
        print("[PY_COMPILE_OK]", rel)
    except Exception as e:
        print("[PY_COMPILE_FAIL]", rel, repr(e))
        failed.append((rel, repr(e)))
if failed:
    raise SystemExit("[ERROR] py_compile failed")

print("\n===== grep key evidence =====")
targets = [
    "opencood/comm/arce/arce_c2mab_comm.py",
    "opencood/comm/arce/policies/c2mab_policy_bank.py",
    "opencood/comm/arce/policies/ego_greedy_oracle.py",
    "opencood/comm/arce/policies/reward_pending_builder.py",
    "opencood/comm/arce/policies/reward_update_manager.py",
]
patterns = [
    "no_send_no_physical_transmission",
    "physical_loss_rate",
    "statistical_weight_alpha",
    "feedback_weight_mode",
    "get_cav_confidence",
    "min_marginal_coverage",
    "explore_warmup_pulls_per_quant",
]
for t in targets:
    p = ROOT / t
    if not p.exists():
        print("[MISS]", t)
        continue
    text = p.read_text(errors="ignore")
    for pat in patterns:
        if pat in text:
            print(f"[FOUND] {pat} in {t}")

print("\n===== git diff stat =====")
subprocess.run(["git", "diff", "--stat", "--"] + targets, cwd=ROOT)

print("\n[DONE]")
print("[WORK_DIR]", work_dir)
print("[BACKUP_DIR]", backup_dir)
