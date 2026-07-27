from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(
    os.environ.get("REPO_DIR", str(Path(__file__).resolve().parents[1]))
).resolve()
MODULE_PATH = (
    REPO_ROOT
    / "opencood"
    / "comm"
    / "arce"
    / "policies"
    / "c2mab_execution_record_builder.py"
)
SPEC = importlib.util.spec_from_file_location("stage2_execution_record_builder", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PhysicalExecutionBudgetTest(unittest.TestCase):
    def test_quant_dependent_estimates_do_not_change_physical_budget(self):
        estimates = {"fp16": 49152.0, "int8": 24576.0, "int4": 12288.0}
        allocations = {}
        for mode, estimate in estimates.items():
            selected = SimpleNamespace(
                record={
                    "quant_mode": mode,
                    "estimated_tx_bytes": estimate,
                    "link_budget_bytes": 62500.0,
                    "proposal_budget_bytes": 62500.0,
                }
            )
            allocations[mode] = MODULE.selected_allocated_budget_bytes(
                selected, total_budget_bytes=62500.0
            )
            self.assertEqual(
                MODULE.selected_allocation_source(selected), "link_budget_bytes"
            )

        self.assertEqual(set(allocations.values()), {62500.0})
        self.assertEqual(len(set(estimates.values())), 3)

    def test_explicit_oracle_allocation_has_priority(self):
        selected = SimpleNamespace(
            record={
                "oracle_physical_allocation_bytes": 32768.0,
                "link_budget_bytes": 62500.0,
                "estimated_tx_bytes": 8192.0,
            }
        )
        self.assertEqual(
            MODULE.selected_allocated_budget_bytes(selected, 62500.0), 32768.0
        )
        self.assertEqual(
            MODULE.selected_allocation_source(selected),
            "oracle_physical_allocation_bytes",
        )

    def test_physical_budget_is_clamped_to_frame_budget(self):
        selected = SimpleNamespace(record={"link_budget_bytes": 70000.0})
        self.assertEqual(
            MODULE.selected_allocated_budget_bytes(selected, 62500.0), 62500.0
        )

    def test_ambiguous_legacy_budget_fields_are_rejected(self):
        selected = SimpleNamespace(
            record={
                "allocated_budget_bytes": 12000.0,
                "proposal_budget_bytes": 12000.0,
                "estimated_tx_bytes": 12000.0,
            }
        )
        with self.assertRaises(RuntimeError):
            MODULE.selected_allocated_budget_bytes(selected, 62500.0)

    def test_multiple_selected_links_share_one_frame_budget(self):
        selected = {
            1: SimpleNamespace(record={"link_budget_bytes": 30000.0}),
            2: SimpleNamespace(record={"link_budget_bytes": 20000.0}),
            3: SimpleNamespace(record={"link_budget_bytes": 12500.0}),
        }
        plan = MODULE.selected_physical_budget_plan(selected, 62500.0)
        self.assertEqual(plan, {1: 30000.0, 2: 20000.0, 3: 12500.0})
        self.assertEqual(sum(plan.values()), 62500.0)

    def test_multiple_selected_links_cannot_duplicate_frame_budget(self):
        selected = {
            1: SimpleNamespace(record={"link_budget_bytes": 62500.0}),
            2: SimpleNamespace(record={"link_budget_bytes": 62500.0}),
        }
        with self.assertRaises(RuntimeError):
            MODULE.selected_physical_budget_plan(selected, 62500.0)

    def test_actual_frame_bytes_cannot_exceed_frame_budget(self):
        MODULE.validate_frame_actual_transmitted_bytes(62464.0, 62500.0)
        with self.assertRaises(RuntimeError):
            MODULE.validate_frame_actual_transmitted_bytes(62501.0, 62500.0)

    def test_non_finite_and_negative_budgets_fail_fast(self):
        for invalid in (-1.0, math.inf, -math.inf, math.nan):
            selected = SimpleNamespace(record={"link_budget_bytes": invalid})
            with self.assertRaises(ValueError):
                MODULE.selected_allocated_budget_bytes(selected, 62500.0)
            with self.assertRaises(ValueError):
                MODULE.validate_frame_actual_transmitted_bytes(invalid, 62500.0)

        selected = SimpleNamespace(record={"link_budget_bytes": 1024.0})
        for invalid_total in (-1.0, math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                MODULE.selected_allocated_budget_bytes(selected, invalid_total)

    def test_budget_audit_distinguishes_estimate_from_physical_cap(self):
        selected = SimpleNamespace(
            cost_bytes=12288.0,
            record={
                "estimated_tx_bytes": 12288.0,
                "estimated_encoded_bytes": 12288.0,
                "link_budget_bytes": 62500.0,
                "proposal_budget_bytes": 62500.0,
            },
        )
        audit = MODULE.build_budget_consistency(
            selected=selected,
            record={
                "size": {"actual_num_encoded_packets": 61},
                "packet": {"packet_size_bytes": 1024},
                "bandwidth_selection": {"num_missing_by_budget": 0},
            },
            allocated_budget_bytes=62500.0,
            tx_bytes=62464.0,
        )

        self.assertEqual(audit["proposal_estimated_tx_bytes"], 12288.0)
        self.assertEqual(audit["physical_execution_budget_bytes"], 62500.0)
        self.assertEqual(
            audit["physical_execution_budget_source"], "link_budget_bytes"
        )
        self.assertFalse(audit["estimated_cost_used_as_execution_budget"])
        self.assertTrue(audit["execution_budget_decoupled_from_estimated_tx"])
        self.assertTrue(
            audit["physical_budget_source_expected_quant_independent"]
        )


if __name__ == "__main__":
    unittest.main()
