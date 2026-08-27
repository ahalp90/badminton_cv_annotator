from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scratch.contact_det_full_ds_fit.scripts import (
    check_rally_start_candidates as checker,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import SCORE_DTYPE


def _score_rows(
    frames: list[int],
    scores: list[float],
    kept: list[bool],
) -> np.ndarray:
    rows = np.empty(len(frames), dtype=SCORE_DTYPE)
    rows["fixture"] = b"sset_18"
    rows["interval_id"] = 0
    rows["frame"] = frames
    rows["fps"] = 30.0
    rows["contact_score"] = scores
    rows["kept"] = kept
    return rows


def _candidate_inputs() -> checker.CandidateInputs:
    video = SimpleNamespace(fixture="sset_18", fps=30.0)
    split = SimpleNamespace(validation_videos=(video,))
    verified = SimpleNamespace(
        split=split,
        spans_by_fixture={
            "sset_18": (
                {"span_id": 0, "start_frame": 0, "end_frame": 20},
                {"span_id": 1, "start_frame": 40, "end_frame": 80},
            )
        },
    )
    event = SimpleNamespace(frame=50)
    saved_run = SimpleNamespace(
        run_id="hgb_reference_raw_more_negatives",
        events_by_fixture={"sset_18": (event,)},
    )
    return checker.CandidateInputs(
        summary={},
        run_result={"selected_duplicate_distance_at_30_fps": 6},
        score_rows=_score_rows(
            [10, 20, 30, 44, 50],
            [0.99, 0.8, 0.8, 0.95, 0.9],
            [False, False, False, False, True],
        ),
        verified_predictions=verified,
        saved_run=saved_run,
        intervals_by_fixture={"sset_18": ((0, 100),)},
    )


def _rally_result(span_count: int) -> dict[str, object]:
    return {
        "runs": [
            {
                "run_id": "hgb_reference_raw_more_negatives",
                "primary": {
                    "spans": [
                        {
                            "fixture": "sset_18",
                            "span_id": span_id,
                            "rally_id": f"set1:{span_id}",
                        }
                        for span_id in range(span_count)
                    ]
                },
            }
        ]
    }
def test_build_candidate_construction_uses_previous_section_end_and_fixed_order() -> None:
    construction = checker.build_candidate_construction(_candidate_inputs(), "abc1234")
    assert construction["counts"] == {
        "detected_sections": 2,
        "candidate_lists": 1,
        "sections_without_kept_contact": 1,
        "candidate_entries": 3,
        "fixed_contact_entries": 1,
        "earlier_candidate_entries": 2,
    }
    candidate_list = construction["candidate_lists"][0]
    assert candidate_list["prefix_start_frame"] == 20
    assert [candidate["frame"] for candidate in candidate_list["candidates"]] == [
        50,
        20,
        30,
    ]
    assert [candidate["is_fixed_contact"] for candidate in candidate_list["candidates"]] == [
        True,
        False,
        False,
    ]


def test_freeze_candidate_construction_writes_repeatable_gzip(tmp_path) -> None:
    output = tmp_path / "candidates.json.gz"
    first, first_hash = checker.freeze_candidate_construction(
        _candidate_inputs(),
        "abc1234",
        output,
    )
    first_bytes = output.read_bytes()
    second, second_hash = checker.freeze_candidate_construction(
        _candidate_inputs(),
        "abc1234",
        output,
    )
    assert first == second
    assert first_hash == second_hash
    assert output.read_bytes() == first_bytes


def test_build_candidate_construction_requires_six_frame_distance() -> None:
    inputs = _candidate_inputs()
    inputs = checker.CandidateInputs(
        inputs.summary,
        {"selected_duplicate_distance_at_30_fps": 5},
        inputs.score_rows,
        inputs.verified_predictions,
        inputs.saved_run,
        inputs.intervals_by_fixture,
    )
    with pytest.raises(ValueError, match="must be six frames"):
        checker.build_candidate_construction(inputs, "abc1234")


def test_build_candidate_construction_scales_six_frames_to_five_at_25_fps() -> None:
    inputs = _candidate_inputs()
    video = SimpleNamespace(fixture="sset_18", fps=25.0)
    verified = SimpleNamespace(
        split=SimpleNamespace(validation_videos=(video,)),
        spans_by_fixture=inputs.verified_predictions.spans_by_fixture,
    )
    inputs = checker.CandidateInputs(
        inputs.summary,
        inputs.run_result,
        inputs.score_rows,
        verified,
        inputs.saved_run,
        inputs.intervals_by_fixture,
    )
    construction = checker.build_candidate_construction(inputs, "abc1234")
    assert construction["candidate_lists"][0]["duplicate_distance_frames"] == 5


def test_measure_candidate_construction_applies_the_fixed_limits() -> None:
    candidate_lists = []
    targets = []
    for span_id in range(checker.TARGET_CONTACT_COUNT):
        missing_frame = 100 + span_id * 100
        candidate_frame = missing_frame if span_id < 50 else missing_frame - 20
        candidate_lists.append(
            {
                "fixture": "sset_18",
                "span_id": span_id,
                "candidates": [
                    {
                        "frame": missing_frame + 40,
                        "contact_score": 0.9,
                        "is_fixed_contact": True,
                    },
                    {
                        "frame": candidate_frame,
                        "contact_score": 0.6,
                        "is_fixed_contact": False,
                    },
                ],
            }
        )
        targets.append(
            {
                "fixture": "sset_18",
                "span_id": span_id,
                "missing_frame": missing_frame,
                "section_start_frame": missing_frame + 10,
                "missing_contact_type": "first",
            }
        )
    construction = {
        "counts": {
            "candidate_entries": 162,
            "fixed_contact_entries": 81,
            "earlier_candidate_entries": 81,
        },
        "candidate_lists": candidate_lists,
    }
    missed_result = {
        "details": {
            "otherwise_correct_one_short_sections": targets,
            "missed_contacts": {"5": [], "10": []},
        }
    }
    inputs = _candidate_inputs()
    measurement = checker.measure_candidate_construction(
        construction,
        missed_result,
        _rally_result(checker.TARGET_CONTACT_COUNT),
        inputs,
    )
    assert measurement["target_first_contacts"]["10"]["covered_contacts"] == 50
    assert measurement["target_first_contacts"]["10"][
        "covered_only_by_candidates_before_section"
    ] == 50
    assert measurement["added_candidates_per_covered_target_at_10_frames"] == 1.62
    assert measurement["all_limits_pass"] is True


def test_all_first_contact_coverage_does_not_borrow_from_another_section() -> None:
    construction = {
        "candidate_lists": [
            {
                "fixture": "sset_18",
                "span_id": 1,
                "candidates": [
                    {"frame": 100, "contact_score": 0.6, "is_fixed_contact": False}
                ],
            },
            {
                "fixture": "sset_18",
                "span_id": 2,
                "candidates": [
                    {"frame": 200, "contact_score": 0.6, "is_fixed_contact": False}
                ],
            },
        ]
    }
    details = [
        {
            "fixture": "sset_18",
            "rally_id": "set1:1",
            "frame": 200,
            "contact_type": "first",
            "tolerance_frames": 10,
        }
    ]
    coverage = checker._all_missed_first_coverage(
        details,
        construction,
        {("sset_18", "set1:1"): ("sset_18", 1)},
    )
    assert coverage["assigned_to_one_detected_section"] == 1
    assert coverage["covered_first_contacts"] == 0


def test_missed_contact_result_rejects_a_changed_menu_input(tmp_path) -> None:
    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "baseline_summary",
            "rally_result",
            "predictions",
            "menu_result",
            "run_result",
            "scores",
            "split",
            "raw_features",
            "shots_master",
        )
    }
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    missed_result_path = tmp_path / "missed.json.gz"
    checker._write_json(
        missed_result_path,
        {
            "schema": checker.missed_checker.CHECK_SCHEMA,
            "status": "complete",
            "run_id": "hgb_reference_raw_more_negatives",
            "inputs": {
                "baseline_summary_sha256": checker._sha256(paths["baseline_summary"]),
                "rally_result_sha256": checker._sha256(paths["rally_result"]),
                "rally_prediction_sha256": checker._sha256(paths["predictions"]),
                "menu_result_sha256": checker._sha256(paths["menu_result"]),
                "run_result_sha256": checker._sha256(paths["run_result"]),
                "score_sha256": checker._sha256(paths["scores"]),
                "split_sha256": checker._sha256(paths["split"]),
                "raw_feature_record_sha256": checker._sha256(paths["raw_features"]),
                "contact_label_sha256": checker._sha256(paths["shots_master"]),
            },
        },
    )
    missed_summary_path = tmp_path / "missed_summary.json"
    checker._write_json(
        missed_summary_path,
        {
            "result_file": missed_result_path.name,
            "result_sha256": checker._sha256(missed_result_path),
            "run_id": "hgb_reference_raw_more_negatives",
        },
    )
    paths["menu_result"].write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the fixed inputs"):
        checker.load_missed_contact_result(
            missed_summary_path,
            missed_result_path,
            _candidate_inputs(),
            paths["baseline_summary"],
            paths["rally_result"],
            paths["predictions"],
            paths["menu_result"],
            paths["run_result"],
            paths["scores"],
            paths["split"],
            paths["raw_features"],
            paths["shots_master"],
        )


def test_main_flow_freezes_candidates_before_opening_missed_detail(
    tmp_path,
    monkeypatch,
) -> None:
    input_paths = [tmp_path / f"input_{index}.json" for index in range(11)]
    for path in input_paths:
        path.write_text("{}", encoding="utf-8")
    construction_path = tmp_path / "construction.json.gz"
    output_path = tmp_path / "result.json.gz"
    inputs = _candidate_inputs()

    monkeypatch.setattr(checker, "load_candidate_inputs", lambda *args: inputs)

    def freeze(*args):
        construction_path.write_bytes(b"fixed")
        return {"candidate_lists": [], "counts": {}}, "construction-hash"

    monkeypatch.setattr(checker, "freeze_candidate_construction", freeze)

    def load_missed(*args):
        assert construction_path.read_bytes() == b"fixed"
        return {"details": {}}, {"runs": []}

    monkeypatch.setattr(checker, "load_missed_contact_result", load_missed)
    monkeypatch.setattr(checker, "measure_candidate_construction", lambda *args: {})

    checker.check_rally_start_candidates(
        input_paths[0],
        input_paths[1],
        input_paths[2],
        input_paths[3],
        input_paths[4],
        input_paths[5],
        input_paths[6],
        input_paths[7],
        input_paths[8],
        input_paths[9],
        input_paths[10],
        construction_path,
        output_path,
        "abc1234",
    )
    assert checker._read_json(output_path)[
        "saved_missed_contacts_opened_after_candidate_list_fixed"
    ] is True
