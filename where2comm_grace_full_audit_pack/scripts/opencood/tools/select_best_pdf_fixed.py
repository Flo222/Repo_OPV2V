#!/usr/bin/env python
"""Select the best PDF-Fixed action from validation logs.

This parser scans inference logs for action id and AP lines.
It writes a small YAML snippet that can be pasted into the test config:

arce:
  pdf_fixed:
    selected_action_id: ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import yaml

AP_RE = re.compile(
    r"Average Precision at IOU 0\.3 is ([0-9.]+),\s*The Average Precision at IOU 0\.5 is ([0-9.]+),\s*The Average Precision at IOU 0\.7 is ([0-9.]+)"
)
ACTION_RE = re.compile(r"pdf_fixed_(send[01]_[A-Za-z0-9_]+)")


def parse_file(path: Path):
    text = path.read_text(errors="ignore")
    aps = AP_RE.findall(text)
    if not aps:
        return None
    ap03, ap05, ap07 = [float(x) for x in aps[-1]]
    m = ACTION_RE.search(text) or ACTION_RE.search(path.name)
    action = m.group(1) if m else path.stem
    return {"file": str(path), "action_id": action, "ap03": ap03, "ap05": ap05, "ap07": ap07}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True, help="directory containing validation log files")
    parser.add_argument("--metric", default="ap05", choices=["ap03", "ap05", "ap07"])
    parser.add_argument("--out", default="best_pdf_fixed.yaml")
    args = parser.parse_args()

    root = Path(args.logs)
    rows = []
    for p in list(root.rglob("*.log")) + list(root.rglob("*.txt")):
        r = parse_file(p)
        if r:
            rows.append(r)
    if not rows:
        raise SystemExit(f"No AP rows found under {root}")
    rows.sort(key=lambda r: r[args.metric], reverse=True)
    best = rows[0]
    out_cfg = {"arce": {"pdf_fixed": {"selected_action_id": best["action_id"]}}, "best": best}
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False, allow_unicode=True)
    print("[BEST]", best)
    print("[OK] saved", args.out)


if __name__ == "__main__":
    main()
