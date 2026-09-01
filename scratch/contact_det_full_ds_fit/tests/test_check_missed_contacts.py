from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scratch.contact_det_full_ds_fit.scripts import check_missed_contacts as checker
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import SCORE_DTYPE


def _scores(
    frames: list[int],
    values: list[float],
    kept: list[bool],
) -> np.ndarray:
    rows = np.empty(len(frames), dtype=SCORE_DTYPE)
    rows["fixture"] = b"sset_18"
    rows["interval_id"] = 0
    rows["frame"] = frames
    rows["fps"] = 25.0
    rows["contact_score"] = values
    rows["kept"] = kept
    return rows


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (_scores([], [], []), checker.NO_CANDIDATE),
        (_scores([99], [0.4], [False]), checker.BELOW_CUTOFF),
        (_scores([99], [0.95], [False]), checker.REMOVED_NEARBY),
        (_scores([99], [0.95], [True]), checker.KEPT_NEARBY),
        (
            _scores([98, 99], [0.99, 0.95], [False, True]),
            checker.KEPT_NEARBY,
        ),
    ],
)
def test_nearby_score_summary_uses_fixed_explanation_order(
    rows: np.ndarray,
    expected: str,
) -> None:
    matched = frozenset(int(frame) for frame in rows["frame"][rows["kept"]])
    summary = checker.nearby_score_summary(
        rows,
        100,
        2,
        0.9,
        matched_kept_frames=matched,
    )
    assert summary["explanation"] == expected


def test_nearby_score_summary_keeps_checkable_counts_and_best_row() -> None:
    rows = _scores([98, 99, 101], [0.95, 0.8, 0.95], [False, True, False])
    summary = checker.nearby_score_summary(
        rows,
        100,
        2,
        0.9,
        matched_kept_frames=frozenset({99}),
    )
    assert summary == {
        "explanation": checker.KEPT_NEARBY,
        "nearby_candidate_count": 3,
        "nearby_kept_count": 1,
        "nearby_at_or_above_cutoff_count": 2,
        "nearby_inside_section_count": None,
        "nearby_outside_section_count": None,
        "nearby_kept_inside_section_count": None,
        "nearby_kept_outside_section_count": None,
        "best_candidate_frame": 98,
        "best_candidate_score": 0.95,
        "best_candidate_frame_offset": -2,
        "best_candidate_absolute_frame_distance": 2,
        "best_candidate_kept": False,
    }


def test_nearby_score_summary_separates_kept_row_outside_section() -> None:
    rows = _scores([98, 101], [0.95, 0.5], [True, False])
    summary = checker.nearby_score_summary(
        rows,
        100,
        2,
        0.9,
        section_bounds=(100, 110),
        matched_kept_frames=frozenset(),
    )
    assert summary["explanation"] == checker.KEPT_OUTSIDE_SECTION
    assert summary["nearby_inside_section_count"] == 1
    assert summary["nearby_outside_section_count"] == 1
    assert summary["nearby_kept_inside_section_count"] == 0
    assert summary["nearby_kept_outside_section_count"] == 1


def test_kept_nearby_must_be_matched_to_another_label() -> None:
    rows = _scores([99], [0.95], [True])
    with pytest.raises(ValueError, match="not matched elsewhere"):
        checker.nearby_score_summary(rows, 100, 2, 0.9)


def test_named_hash_rejects_changed_chosen_run_result(tmp_path) -> None:
    path = tmp_path / "baseline_result.json"
    path.write_text("original")
    record = {
        "run_result_file": path.name,
        "run_result_sha256": checker._sha256(path),
    }
    path.write_text("changed")
    with pytest.raises(ValueError, match="chosen run result hash differs"):
        checker._check_named_hash(
            record,
            path,
            "run_result_file",
            "run_result_sha256",
            "chosen run result",
        )


def test_otherwise_correct_one_short_requires_a_nonempty_answerable_section() -> None:
    span = {
        "rally_id": "set1:1",
        "event_count": 3,
        "ground_truth_contacts": 4,
        "timing_matches": 3,
        "correct_side_answers": 3,
        "side_answerable": True,
        "rejection_reasons": ["missing_contact", "timing_mismatch"],
    }
    assert checker._is_otherwise_correct_one_short(span)
    assert not checker._is_otherwise_correct_one_short({**span, "event_count": 0})
    assert not checker._is_otherwise_correct_one_short(
        {**span, "rejection_reasons": ["missing_contact", "extra_event"]}
    )
    assert not checker._is_otherwise_correct_one_short({**span, "side_answerable": False})


def test_score_identity_check_rejects_repeated_video_frames() -> None:
    rows = _scores([98, 98], [0.2, 0.3], [False, False])

    class Video:
        fixture = "sset_18"
        fps = 25.0

    class Split:
        validation_videos = (Video(),)

    class Verified:
        split = Split()

    run = SimpleNamespace(events_by_fixture={"sset_18": ()})

    with pytest.raises(ValueError, match="score identities"):
        checker._check_score_identities(rows, Verified(), run)  # type: ignore[arg-type]
