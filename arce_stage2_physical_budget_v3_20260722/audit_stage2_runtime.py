from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": values[0],
        "mean": sum(values) / len(values),
        "max": values[-1],
    }


def quant_mode(record):
    mode = record.get("quant_mode")
    if mode:
        return str(mode).lower()
    action_id = str(record.get("action_id", "unknown"))
    for candidate in ("fp16", "int8", "int4", "fp32", "none"):
        if "_" + candidate + "_" in action_id:
            return candidate
    return "unknown"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python audit_stage2_runtime.py runtime_records.jsonl")

    path = Path(sys.argv[1])
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        root = json.loads(line)
        for item in walk(root):
            if isinstance(item.get("budget_consistency"), dict):
                records.append(item)

    by_quant = defaultdict(lambda: defaultdict(list))
    sources = Counter()
    estimated_as_budget = 0
    not_decoupled = 0
    unexpected_quant_dependent_source = 0
    over_budget = 0
    frame_actual = defaultdict(float)
    frame_cap = {}

    for item in records:
        budget = item["budget_consistency"]
        mode = quant_mode(item)
        fields = {
            "proposal_estimated_tx_bytes": budget.get("proposal_estimated_tx_bytes"),
            "physical_execution_budget_bytes": budget.get(
                "physical_execution_budget_bytes"
            ),
            "actual_tx_bytes": budget.get("actual_tx_bytes"),
            "actual_over_physical_budget": budget.get(
                "actual_over_physical_budget"
            ),
        }
        for name, value in fields.items():
            by_quant[mode][name].append(finite(value))

        source = str(budget.get("physical_execution_budget_source", "missing"))
        sources[source] += 1
        estimated_as_budget += int(
            bool(budget.get("estimated_cost_used_as_execution_budget", False))
        )
        not_decoupled += int(
            not bool(budget.get("execution_budget_decoupled_from_estimated_tx", False))
        )
        unexpected_quant_dependent_source += int(
            not bool(
                budget.get(
                    "physical_budget_source_expected_quant_independent", False
                )
            )
        )
        ratio = finite(budget.get("actual_over_physical_budget"))
        over_budget += int(ratio is not None and ratio > 1.000001)

        frame_id = item.get("frame_id")
        actual = finite(budget.get("actual_tx_bytes"))
        system = item.get("system_budget", {})
        cap = finite(
            system.get("system_budget_bytes", system.get("total_budget_bytes"))
            if isinstance(system, dict)
            else None
        )
        if frame_id is not None and actual is not None:
            frame_actual[str(frame_id)] += actual
        if frame_id is not None and cap is not None:
            frame_cap[str(frame_id)] = cap

    frame_over_budget = sum(
        1
        for frame_id, actual in frame_actual.items()
        if frame_id in frame_cap and actual > frame_cap[frame_id] + 1e-6
    )

    report = {
        "records": len(records),
        "physical_budget_sources": dict(sources),
        "estimated_cost_used_as_execution_budget_count": estimated_as_budget,
        "not_decoupled_count": not_decoupled,
        "unexpected_quant_dependent_budget_source_count": (
            unexpected_quant_dependent_source
        ),
        "link_over_physical_budget_count": over_budget,
        "auditable_frame_count": len(frame_cap),
        "frame_over_system_budget_count": frame_over_budget,
        "by_quant": {
            mode: {field: stats(values) for field, values in fields.items()}
            for mode, fields in sorted(by_quant.items())
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
