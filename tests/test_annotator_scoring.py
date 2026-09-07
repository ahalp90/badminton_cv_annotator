"""Regression floors for the full annotator GT scoring harness."""
import math
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from annotator.calibration import gt_scoring
from annotator.calibration.fixtures import FIXTURES
from annotator.calibration.gt_scoring import (
    REFERENCE_SCORES,
    flatten_metrics,
    render_table,
    run_fixture,
)
from annotator.calibration.scoring import (
    CANONICAL_CONTACT_TOLERANCE_BASE30,
    GtRally,
    strict_contact_rows,
    wide_edge_contact_rows,
)
from annotator.point_winner import (
    SHIPPED_LANDING_FILTER_OPTIONS,
    Half,
    Landing,
    LandingFilterOptions,
)
from annotator.run_video import AnnotatorResult
from annotator.scene_courts import SceneCourt
from annotator.types import ContactCandidate

# Below 0.75x reference reads as a miswired chain, not tuning debt (ruled 2026-07-18,
# raised from the drafted 0.5).
FLOOR_MULTIPLIER = 0.75


def test_calibration_uses_shipped_landing_filter_options() -> None:
    assert gt_scoring.SHIPPED_LANDING_FILTER_OPTIONS is SHIPPED_LANDING_FILTER_OPTIONS
    assert SHIPPED_LANDING_FILTER_OPTIONS == LandingFilterOptions(7, 0.004, 5, 7, 0.75)


def test_canonical_tolerance_uses_the_shared_base30_value() -> None:
    assert gt_scoring.canonical_tolerance(30.0) == CANONICAL_CONTACT_TOLERANCE_BASE30


def test_score_video_uses_final_stroke_scene_for_gt_landing(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = SimpleNamespace(name="scene-fixture", video_id=1, fps=30.0, gt_set_dir=None)
    rally = GtRally("set1", 1, (5, 15))
    master = pd.DataFrame({
        "vid": [1, 1],
        "frame_num": [5, 15],
        "player_side": ["Top", "Bot"],
    })
    winner = gt_scoring.GtWinner(None, None, None, None, (20.0, 30.0), {}, False)
    reconciliation = gt_scoring.Reconciliation([winner], 1, 0, 0, 0, 0, 0)
    set_table = pd.DataFrame({
        "rally": [1, 1], "frame_num": [5, 15], "hit_height": [np.nan, np.nan],
    })
    result = AnnotatorResult(
        [(0, 20)],
        [],
        [],
        {0: [5, 15]},
        [Half.TOP],
        [2],
        [None],
        [Half.TOP],
        {},
        {0: Landing(18, (0.5, 0.75), Half.BOT, False, False)},
        {},
        {},
        [],
    )
    early = {"name": "early"}
    late = {"name": "late"}
    scenes = (
        SceneCourt(0, 10, early, (0.1, 0.2), 0.1),
        SceneCourt(10, 20, late, (0.1, 0.2), 0.1),
    )
    projected_courts: list[str] = []

    monkeypatch.setattr(gt_scoring, "load_gt_rallies", lambda _master, _video_id: [rally])
    monkeypatch.setattr(gt_scoring, "load_set_tables", lambda _fixture, _gt: {"set1": set_table})
    monkeypatch.setattr(gt_scoring, "reconcile_sets", lambda *_args: reconciliation)
    monkeypatch.setattr(
        gt_scoring, "classify_all", lambda _spans, _gt: [(gt_scoring.RallyBoundary.COVERED, 0)],
    )
    monkeypatch.setattr(gt_scoring, "score_boundaries", lambda _spans, _gt: {})
    monkeypatch.setattr(
        gt_scoring,
        "score_contacts",
        lambda *_args, **_kwargs: {"count_gate": {"covered": {"pass": 1, "total": 1}}},
    )

    def project(_pixels, _resolution, court):
        projected_courts.append(court["name"])
        return np.array([[0.5], [0.75 if court["name"] == "late" else 0.25]])

    monkeypatch.setattr(gt_scoring.point_winner, "project_pixels_to_court", project)
    scored = gt_scoring.score_video(fixture, result, master, {1: {"name": "global"}}, 5, scenes)

    assert projected_courts == ["late"]
    assert scored.rows[0].landing_gt == Half.BOT.value
    assert scored.rows[0].landing_correct is True

    gap = gt_scoring.score_video(
        fixture, result, master, {1: {"name": "global"}}, 5,
        (SceneCourt(0, 10, early, (0.1, 0.2), 0.1),),
    )
    assert gap.rows[0].landing_gt is None
    assert gap.rows[0].landing_pred == Half.BOT.value
    assert gap.rows[0].landing_correct is False
    assert gap.landing == gt_scoring.ColumnAgg(0, 1, 0, 1)


def _assert_floors(fixture, metrics: dict[str, int | float | None]) -> None:
    if REFERENCE_SCORES is None:
        raise AssertionError("REFERENCE_SCORES is not captured")
    for metric in ("covered_fraction", "contact_f1"):
        reference = REFERENCE_SCORES[fixture.name][metric]
        current = metrics[metric]
        if not isinstance(reference, (int, float)) or not math.isfinite(reference) or reference < 0:
            raise AssertionError(f"invalid reference {fixture.name} {metric}: {reference!r}")
        if not isinstance(current, (int, float)) or not math.isfinite(current):
            raise AssertionError(f"invalid current {fixture.name} {metric}: {current!r}")
        if current < FLOOR_MULTIPLIER * reference:
            raise AssertionError(
                f"{fixture.name} {metric}: {current!r} < floor {FLOOR_MULTIPLIER * reference!r}"
            )


@pytest.mark.slow
@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda fixture: fixture.name)
def test_annotator_gt_floors(fixture):
    if not os.environ.get("ANNOTATOR_FIXTURES_ROOT"):
        pytest.skip("ANNOTATOR_FIXTURES_ROOT is unset; external fixtures are unavailable")
    metrics = flatten_metrics(run_fixture(fixture))
    print(render_table({fixture.name: metrics}))
    _assert_floors(fixture, metrics)


def test_strict_contact_rows_keep_all_row_kinds_and_scaled_tolerances():
    rallies = [
        GtRally('set1', 1, (10, 15)),
        GtRally('set1', 2, (32, 55)),
        GtRally('set1', 3, (80,)),
    ]
    rows = strict_contact_rows(
        [(0, 20), (30, 40), (50, 60)],
        [
            ContactCandidate(0, 6, None, None, None),
            ContactCandidate(0, 24, None, None, None),
            ContactCandidate(1, 33, None, None, None),
            ContactCandidate(2, 54, None, None, None),
        ],
        rallies,
        fps=25.0,
    )

    base5 = [row for row in rows if row['tolerance_base30'] == 5]
    assert [(row['row_kind'], row['gt_frame'], row['candidate_frame'], row['offset_frames'])
            for row in base5[:3]] == [
        ('matched', 10, 6, -4), ('unmatched_gt', 15, None, None),
        ('unmatched_candidate', None, 24, None),
    ]
    assert {row['tolerance_frames'] for row in rows} == {4, 8}
    split_rows = [row for row in rows if row['rally_id'] == 2]
    missed_rows = [row for row in rows if row['rally_id'] == 3]
    assert all(row['row_kind'] == 'unmatched_gt' for row in split_rows)
    assert all(row['row_kind'] == 'unmatched_gt' for row in missed_rows)


def test_wide_edge_contact_rows_split_midpoint_and_omit_outside_candidates():
    rallies = [GtRally('set1', 1, (10,)), GtRally('set1', 2, (20,))]
    rows = wide_edge_contact_rows(
        rallies,
        [
            ContactCandidate(0, 15, None, None, None),
            ContactCandidate(0, 16, None, None, None),
            ContactCandidate(1, 29, None, None, None),
            ContactCandidate(1, 30, None, None, None),
        ],
        fps=30.0,
        n_frames=30,
    )

    assert [(row['window_start'], row['window_end']) for row in rows if row['row_kind'] != 'unmatched_candidate'] == [
        (0, 16), (16, 30),
    ]
    assert [(row['rally_id'], row['row_kind'], row['candidate_frame']) for row in rows] == [
        (0, 'matched', 15),
        (1, 'matched', 16),
        (1, 'unmatched_candidate', 29),
    ]


def test_contact_reports_use_source_order_for_repeated_set_rally_numbers():
    rallies = [GtRally('set1', 1, (10,)), GtRally('set2', 1, (20,))]
    contacts = [
        ContactCandidate(0, 10, None, None, None),
        ContactCandidate(1, 20, None, None, None),
    ]

    strict = strict_contact_rows([(0, 15), (15, 25)], contacts, rallies, fps=30.0)
    wide = wide_edge_contact_rows(rallies, contacts, fps=30.0, n_frames=30)

    assert {row['rally_id'] for row in strict} == {0, 1}
    assert {row['rally_id'] for row in wide} == {0, 1}


def test_strict_contact_rows_scale_at_30_fps_and_keep_tie_order():
    rallies = [GtRally('set1', 1, (10, 20))]
    rows = strict_contact_rows(
        [(0, 30)],
        [ContactCandidate(0, 15, None, None, None)],
        rallies,
        fps=30.0,
    )

    assert {row['tolerance_frames'] for row in rows} == {5, 10}
    assert [(row['tolerance_base30'], row['row_kind'], row['gt_frame'], row['candidate_frame']) for row in rows] == [
        (5, 'matched', 10, 15),
        (5, 'unmatched_gt', 20, None),
        (10, 'matched', 10, 15),
        (10, 'unmatched_gt', 20, None),
    ]


def test_wide_edge_contact_rows_use_one_candidate_for_duplicate_targets():
    rallies = [GtRally('set1', 1, (10,)), GtRally('set2', 1, (10,))]
    rows = wide_edge_contact_rows(
        rallies,
        [ContactCandidate(0, 10, None, None, None)],
        fps=30.0,
        n_frames=30,
    )

    assert [(row['window_id'], row['rally_id'], row['row_kind']) for row in rows] == [
        (0, 0, 'matched'),
        (1, 1, 'unmatched_gt'),
    ]


def test_wide_edge_contact_rows_keep_duplicate_first_and_last_strokes():
    rows = wide_edge_contact_rows(
        [GtRally('set1', 1, (10, 10))],
        [ContactCandidate(0, 10, None, None, None)],
        fps=30.0,
        n_frames=30,
    )

    assert [(row['edge'], row['row_kind']) for row in rows] == [
        ('first', 'matched'),
        ('last', 'unmatched_gt'),
    ]


def test_wide_edge_contact_rows_leave_abutting_windows_unchanged():
    rallies = [GtRally('set1', 1, (90,)), GtRally('set1', 2, (271,))]
    rows = wide_edge_contact_rows(
        rallies,
        [
            ContactCandidate(0, 180, None, None, None),
            ContactCandidate(1, 181, None, None, None),
        ],
        fps=30.0,
        n_frames=400,
    )

    assert [
        (row['window_start'], row['window_end'], row['candidate_frame'])
        for row in rows
    ] == [
        (0, 181, 180),
        (181, 362, 181),
    ]
