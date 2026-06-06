#!/usr/bin/env python
"""Generate 48 PDF-Fixed YAML files.

Usage:
  python opencood/tools/generate_pdf_fixed_yamls.py \
    --base opencood/hypes_yaml/point_pillar_v2xvit_opv2v_arce_dc2mab.yaml \
    --out opencood/hypes_yaml/arce_pdf_fixed_48
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from opencood.comm.arce.policies.action_space import build_pdf_action_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fec-mode", default="raptor_sim", choices=["raptor_sim", "xor"])
    args = parser.parse_args()

    base_path = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    actions = build_pdf_action_space(fec_mode=args.fec_mode)
    for action in actions:
        c = yaml.safe_load(yaml.safe_dump(cfg, sort_keys=False))
        arce = c.setdefault("arce", {})
        arce["mode"] = "pdf_fixed"
        arce["policy"] = "pdf_fixed_48"
        arce.setdefault("action_space", {})["fec_main"] = args.fec_mode
        arce["pdf_fixed"] = {"selected_action_id": action.action_id}
        c["name"] = f"{c.get('name', 'arce')}_pdf_fixed_{action.action_id}"
        out = out_dir / f"{c['name']}.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] generated {len(actions)} PDF-Fixed YAMLs in {out_dir}")


if __name__ == "__main__":
    main()
