from __future__ import annotations

# Import the standalone experiment after adding its folder to sys.path.
# ruff: noqa: E402

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import scene_aware_ransac as experiment


def test_window_crosses_only_strictly_internal_cut() -> None:
    boundaries = np.asarray([8, 16, 23], dtype=np.int64)

    assert not experiment.window_crosses_cut(8, 16, boundaries)
    assert experiment.window_crosses_cut(4, 20, boundaries)
    assert not experiment.window_crosses_cut(0, 8, boundaries)
    assert experiment.window_crosses_cut(16, 24, boundaries)
    assert not experiment.window_crosses_cut(24, 32, boundaries)
    assert not experiment.window_crosses_cut(0, 16, np.asarray([], dtype=np.int64))


def test_crossing_window_is_skipped_before_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    track = np.ones((experiment.audit.WINDOW, 3), dtype=np.float64)
    fit_calls: list[np.ndarray] = []

    def record_fit(points: np.ndarray, *args: Any) -> np.ndarray:
        fit_calls.append(points)
        return np.zeros(experiment.audit.WINDOW, dtype=np.float64)

    monkeypatch.setattr(experiment.audit, "fit_quadratic_ransac", record_fit)
    boundaries = np.asarray([experiment.audit.WINDOW // 2], dtype=np.int64)

    _fields, diagnostics = experiment.run_ransac_with_boundaries(
        track,
        boundaries,
        exclude_cut_crossing=True,
    )

    assert not fit_calls
    assert diagnostics.scheduled == 1
    assert diagnostics.cut_crossing_skipped == 1

    experiment.run_ransac_with_boundaries(
        track,
        boundaries,
        exclude_cut_crossing=False,
    )
    assert len(fit_calls) == 1


def test_candidate_vote_uses_half_of_eligible_windows() -> None:
    masked = np.asarray([False, False, False, True])
    eligible = np.asarray([1, 2, 3, 3], dtype=np.int16)
    votes = np.asarray([1, 1, 1, 3], dtype=np.int16)
    residual = np.zeros(4, dtype=np.float64)

    result = experiment.finish_ransac_fields(masked, eligible, votes, residual)

    assert result["candidate"].tolist() == [True, True, False, False]
    assert result["outlier_fraction"].tolist() == [1.0, 0.5, 1.0 / 3.0, 1.0]


def test_base30_contact_radii_use_project_rounding() -> None:
    assert [experiment.scaled_contact_radius(value, 25.0) for value in (5, 10, 15)] == [4, 8, 13]
    assert [experiment.scaled_contact_radius(value, 30.0) for value in (5, 10, 15)] == [5, 10, 15]


def test_mask_metrics_separates_contacts_at_risk_from_selected_frames() -> None:
    mask = np.zeros(20, dtype=bool)
    mask[[3, 8, 9, 15]] = True

    metrics = experiment.mask_metrics(
        mask,
        contacts={5, 10},
        final_contacts={10},
        spans=[(8, 10), (16, 18)],
        fps=30.0,
    )

    tolerance = metrics["contact_tolerances"]["base30_5"]
    assert metrics["selected_frames"] == 4
    assert metrics["visual_positive_spans_hit"] == 1
    assert tolerance["contacts_with_candidate"] == 2
    assert tolerance["final_contacts_with_candidate"] == 1
    assert tolerance["candidate_frames_near_contacts"] == 4
    assert tolerance["candidate_frames_near_final_contacts"] == 3


def test_deterministic_gzip_writer_has_stable_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    payload = {"b": [2, 1], "a": {"value": True}}

    experiment.write_json_gz(first, payload)
    experiment.write_json_gz(second, payload)

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as source:
        assert json.load(source) == payload
