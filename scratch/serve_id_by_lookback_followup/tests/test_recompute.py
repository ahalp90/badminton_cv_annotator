from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from serve_id_followup.recompute import build_outputs, compare_outputs
from serve_id_followup.rules import (
    other_side,
    paired_outcome,
    preferred_decision,
    rank1_sensitivity_decision,
    temporal_slot_is_correct,
)


class RuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = {
            "baseline_server": "Top",
            "baseline_frame": "100",
            "baseline_category": "visible_serve",
            "baseline_gt_label": "contact_1",
        }

    def test_other_side(self) -> None:
        self.assertEqual(other_side("Top"), "Bot")
        self.assertEqual(other_side("Bot"), "Top")
        with self.assertRaises(ValueError):
            other_side("unknown")

    def test_temporal_scoring(self) -> None:
        self.assertTrue(temporal_slot_is_correct("serve", "contact_1"))
        self.assertTrue(temporal_slot_is_correct("return", "contact_2"))
        self.assertFalse(temporal_slot_is_correct("serve", "contact_2"))

    def test_paired_outcomes(self) -> None:
        self.assertEqual(paired_outcome(True, False), "fix")
        self.assertEqual(paired_outcome(False, True), "damage")
        self.assertEqual(paired_outcome(True, True), "both_correct")
        self.assertEqual(paired_outcome(False, False), "both_wrong")

    def test_preferred_branches(self) -> None:
        incoming = {
            "sequential_category": "first_visible_post_serve_contact",
            "sequential_selected_player": "Top",
            "sequential_selected_frame": "120",
            "tolerance_10_sequential_selected_label": "contact_2",
        }
        decision = preferred_decision(incoming, self.baseline)
        self.assertEqual(decision.predicted_server, "Bot")
        self.assertEqual(decision.temporal_claim, "return")

        unresolved = dict(incoming, sequential_category="not_enough_shuttle_trajectory_to_tell")
        decision = preferred_decision(unresolved, self.baseline)
        self.assertEqual(decision.branch, "pr82_fallback")
        self.assertEqual(decision.predicted_server, "Top")

    def test_rank1_fallback(self) -> None:
        search = {
            "sequential_category": "not_enough_shuttle_trajectory_to_tell",
            "sequential_selected_player": "Top",
            "sequential_selected_frame": "120",
            "tolerance_10_sequential_selected_label": "unmatched",
        }
        rank1 = {"pre_verdict": "incoming", "player": "Top"}
        decision = rank1_sensitivity_decision(search, self.baseline, rank1)
        self.assertEqual(decision.branch, "fallback_rank1__incoming__other_side")
        self.assertEqual(decision.predicted_server, "Bot")


class IntegrationTests(unittest.TestCase):
    def test_checked_metrics_and_committed_outputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            metrics = build_outputs(root / "data", temporary)
            self.assertEqual(compare_outputs(root / "results", temporary), [])

        self.assertEqual(metrics["population"], 239)
        self.assertEqual(
            metrics["pr82_baseline_from_frozen_table"]["server_correct"], 163
        )
        self.assertEqual(metrics["preferred_server_rule"]["server_correct"], 170)
        self.assertEqual(
            metrics["preferred_server_rule"]["temporal_slot_correct"], 132
        )
        self.assertEqual(
            metrics["preferred_server_rule"][
                "joint_temporal_and_server_correct"
            ],
            117,
        )
        self.assertEqual(metrics["preferred_server_rule"]["fixes_vs_pr82"], 20)
        self.assertEqual(metrics["preferred_server_rule"]["damages_vs_pr82"], 13)
        self.assertEqual(
            metrics["rank1_fallback_sensitivity"]["server_correct"], 171
        )
        self.assertEqual(
            metrics["rank1_fallback_sensitivity"]["temporal_slot_correct"],
            131,
        )
        self.assertEqual(metrics["diagnostics"]["direct_160"]["coverage"], 160)
        self.assertEqual(
            metrics["diagnostics"]["direct_160"]["correct_server_sides"], 120
        )
        self.assertEqual(
            metrics["diagnostics"]["minimum_path_3_final_correct"], 166
        )


if __name__ == "__main__":
    unittest.main()
