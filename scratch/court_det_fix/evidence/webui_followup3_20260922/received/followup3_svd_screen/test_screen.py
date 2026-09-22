"""Focused checks; set TASK3_INPUT_ROOT to the unpacked input root."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import run_screen as screen

INPUTS = Path(os.environ.get("TASK3_INPUT_ROOT", "../task3_inputs")).resolve()


class ScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fit_groups = staticmethod(screen.load_fit_groups(INPUTS))

    def test_exact_ties_use_support_then_original_index(self):
        groups = [dict(group_index=i, line_count=n, algebraic_rms=r)
                  for i, n, r in ((8, 3, .2), (5, 4, .2), (2, 4, .2), (9, 2, .1))]
        self.assertEqual(screen.rank_groups(groups), [9, 2, 5, 8])
        self.assertEqual(screen.rank_groups(list(reversed(groups))), [9, 2, 5, 8])

    def test_nonfinite_residual_rejected(self):
        groups = [dict(group_index=0, line_count=3, algebraic_rms=float("nan"))]
        with self.assertRaisesRegex(ValueError, "Invalid SVD residual"):
            screen.rank_groups(groups)

    def test_two_line_support_uses_full_nullspace(self):
        lines = np.array([[1., 0., -2.], [0., 1., -3.]])
        groups = screen.fit_diagnostics(lines, np.ones((1, 2), dtype=bool), self.fit_groups)
        self.assertEqual(groups[0]["singular_values"][-1], 0.0)
        self.assertLess(groups[0]["algebraic_rms"], 1e-12)
        self.assertEqual(groups[0]["support_line_ids"], [0, 1])

    def test_invalid_line_normal_and_insufficient_support_rejected(self):
        lines = np.array([[0., 0., 1.], [0., 1., -3.]])
        with self.assertRaisesRegex(ValueError, "Zero line normal"):
            screen.fit_diagnostics(lines, np.ones((1, 2), dtype=bool), self.fit_groups)
        with self.assertRaisesRegex(ValueError, "at least two support lines"):
            screen.fit_diagnostics(np.array([[1., 0., 1.]]), np.ones((1, 1), dtype=bool), self.fit_groups)

    def test_failures_and_finite_unconverged_fits_stay_separate(self):
        pairs = [[i, j] for i in range(3) for j in range(3) if i != j]
        records = [dict(pair_id=i, groups=pair, max_corner_working_px=float(i + 1), converged=True)
                   for i, pair in enumerate(pairs)]
        records[0] = dict(pair_id=0, groups=pairs[0], failure="synthetic solver failure")
        records[1].update(max_corner_working_px=.5, converged=False)
        result = screen.evaluate([0, 1, 2], 3, records)
        self.assertEqual((result["failed"], result["unconverged"], result["converged"]), (1, 1, 4))
        self.assertEqual(result["best_finite"]["pair_id"], 1)
        self.assertEqual(result["best_converged"]["pair_id"], 2)
        self.assertEqual(result["unconverged_pair_ids"], [1])
        self.assertEqual(screen.fit_status({"max_corner_working_px": float("inf"), "converged": True}), "failed")
        with self.assertRaisesRegex(ValueError, "Incomplete ordered-pair cache"):
            screen.evaluate([0, 1, 2], 3, records[:-1])

    def test_changed_B_directions_are_rejected(self):
        case = screen.CASES[0]
        paths = {"e2": INPUTS / screen.RUN / "e2" / f"{case}.json.gz",
                 "e3": INPUTS / screen.RUN / "e3" / f"{case}.json.gz",
                 "baseline": INPUTS / screen.PREFIX / "frozen_views/baseline_directions" / f"{case}.json.gz"}
        data = {key: screen.read_json(path) for key, path in paths.items()}
        screen.validate_inputs(case, paths, data)
        corrupt = copy.deepcopy(data)
        corrupt["e2"]["arms"]["B"]["points_working"][0][0] += 1
        with self.assertRaisesRegex(ValueError, "B directions changed"):
            screen.validate_inputs(case, paths, corrupt)

    def test_full_main_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            result = screen.run(INPUTS, output)
            self.assertEqual([r["best_pairs_preserved_all"] for r in result["aggregate"]], [2, 3, 9, 9, 9])
            self.assertEqual(sum(len(c["groups"]) for c in result["cases"]), 144)
            self.assertEqual(result, screen.read_json(output / "results.json.gz"))
            self.assertTrue(all(c["svd_replay"]["within_tolerance"] for c in result["cases"]))
            self.assertTrue(all(c["budgets"][-1]["best_finite"] == c["baseline_best_finite"]
                                for c in result["cases"]))
            self.assertTrue(all(c["budgets"][-1]["best_converged"] == c["baseline_best_converged"]
                                for c in result["cases"]))
            self.assertEqual(sum(len(c["exact_ties"]) for c in result["cases"]), 0)
            self.assertEqual(sum(len(c["numerical_ties_at_1e_minus_12"]) for c in result["cases"]), 0)
            with self.assertRaisesRegex(ValueError, "Output directory is not empty"):
                screen.run(INPUTS, output)

    def test_missing_inputs_and_wrong_revision_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Input root does not exist"):
                screen.run(root / "missing", root / "results")
            (root / "PINNED_REVISION.txt").write_text("wrong\nrevision\n")
            with self.assertRaisesRegex(ValueError, "Wrong revision marker"):
                screen.run(root, root / "results")


if __name__ == "__main__":
    unittest.main(verbosity=2)
