from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            for item in _walk_dicts(child):
                yield item
    elif isinstance(value, list):
        for child in value:
            for item in _walk_dicts(child):
                yield item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_records_jsonl")
    args = parser.parse_args()

    path = Path(args.runtime_records_jsonl)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        wrapper = json.loads(line)
        for item in _walk_dicts(wrapper):
            compact = item.get("compact_sparse")
            if (
                isinstance(compact, dict)
                and compact.get("priority_source") is not None
            ):
                records.append(item)

    candidate_sources = Counter()
    priority_sources = Counter()
    layouts = Counter()
    candidate_ratios = []
    wrong = []

    for item in records:
        compact = item["compact_sparse"]
        candidate_sources[str(compact.get("candidate_source"))] += 1
        priority_sources[str(compact.get("priority_source"))] += 1
        layouts[str(compact.get("layout"))] += 1
        importance = compact.get("spatial_importance") or {}
        ratio = importance.get("candidate_ratio")
        if ratio is not None:
            candidate_ratios.append(float(ratio))

        if (
            compact.get("candidate_source")
            != "arce_nonzero_spatial_support"
            or compact.get("priority_source")
            != "arce_sender_feature_rms"
            or compact.get("layout") != "KC"
        ):
            wrong.append({
                "frame_id": item.get("frame_id"),
                "agent_index": item.get("agent_index"),
                "candidate_source": compact.get("candidate_source"),
                "priority_source": compact.get("priority_source"),
                "layout": compact.get("layout"),
            })

    result = {
        "communication_records": len(records),
        "candidate_sources": dict(candidate_sources),
        "priority_sources": dict(priority_sources),
        "layouts": dict(layouts),
        "candidate_ratio": {
            "n": len(candidate_ratios),
            "min": min(candidate_ratios) if candidate_ratios else None,
            "mean": (
                sum(candidate_ratios) / len(candidate_ratios)
                if candidate_ratios else None
            ),
            "max": max(candidate_ratios) if candidate_ratios else None,
        },
        "wrong_records": len(wrong),
        "wrong_examples": wrong[:10],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not records:
        raise RuntimeError("No compact ARCE communication records found.")
    if wrong:
        raise RuntimeError(
            "Found records outside the Stage 3A importance path."
        )
    print("Stage 3A runtime audit: PASS")


if __name__ == "__main__":
    main()
