from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scratch.contact_det_full_ds_fit.scripts import (
    save_validation_rally_start_inputs as saver,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import SCORE_DTYPE


def _video() -> SimpleNamespace:
    return SimpleNamespace(
        fixture="sset_18",
        video_id=18,
        fps=30.0,
        width=1920,
        height=1080,
    )


def _score_rows() -> np.ndarray:
    rows = np.empty(3, dtype=SCORE_DTYPE)
    rows["fixture"] = b"sset_18"
    rows["interval_id"] = 0
    rows["frame"] = (1, 4, 10)
    rows["fps"] = 30.0
    rows["contact_score"] = (0.4, 0.6, 0.95)
    rows["kept"] = (False, False, True)
    return rows


def _candidate_list() -> dict[str, object]:
    return {
        "fixture": "sset_18",
        "span_id": 0,
        "section_start_frame": 5,
        "section_end_frame": 20,
        "interval_id": 0,
        "prefix_start_frame": 0,
        "fixed_contact_frame": 10,
        "duplicate_distance_frames": 6,
        "candidates": [
            {
                "frame": 10,
                "contact_score": 0.95,
                "is_fixed_contact": True,
                "kept": True,
                "predicted_side": "Top",
            },
            {
                "frame": 4,
                "contact_score": 0.6,
                "is_fixed_contact": False,
                "kept": False,
                "predicted_side": "Top",
            },
            {
                "frame": 1,
                "contact_score": 0.4,
                "is_fixed_contact": False,
                "kept": False,
                "predicted_side": None,
            },
        ],
    }


def test_replay_side_coverage_is_exact() -> None:
    frames = np.asarray([1, 4, 10], dtype=np.int32)

    sides = saver._checked_replay_sides(
        "sset_18",
        frames,
        np.zeros((20, 3)),
        SimpleNamespace(),
        np.zeros((20, 1, 4)),
        (0.4, 0.6),
        lambda frame, *_args: "Bottom" if frame == 4 else None,
    )

    assert sides == {1: None, 4: "Bot", 10: None}

    with pytest.raises(ValueError, match="replay frames repeat"):
        saver._checked_replay_sides(
            "sset_18",
            np.asarray([1, 1], dtype=np.int32),
            np.zeros((20, 3)),
            SimpleNamespace(),
            np.zeros((20, 1, 4)),
            (0.4, 0.6),
            lambda *_args: "Top",
        )


def test_saved_kept_contact_must_match_the_replayed_side() -> None:
    rows = _score_rows()
    spans = [{"span_id": 0, "start_frame": 5, "end_frame": 20}]
    raw_contacts = [
        {
            "frame": 10,
            "timing_score": 0.95,
            "predicted_side": "Top",
            "span_id": 0,
        }
    ]

    contacts = saver._saved_kept_contacts(
        "sset_18",
        rows,
        spans,
        raw_contacts,
        {10: "Top"},
    )

    assert contacts[0]["interval_id"] == 0
    assert contacts[0]["predicted_side"] == "Top"

    with pytest.raises(ValueError, match="saved kept contact differs"):
        saver._saved_kept_contacts(
            "sset_18",
            rows,
            spans,
            raw_contacts,
            {10: "Bot"},
        )


def test_combined_result_rejects_a_candidate_used_by_two_sections(
    monkeypatch,
) -> None:
    monkeypatch.setattr(saver, "EXPECTED_VIDEO_COUNT", 1)
    monkeypatch.setattr(saver, "EXPECTED_CANDIDATE_LIST_COUNT", 2)
    monkeypatch.setattr(saver, "EXPECTED_CANDIDATE_ENTRY_COUNT", 6)
    first = _candidate_list()
    second = _candidate_list()
    second["span_id"] = 1
    video = {
        "fixture": "sset_18",
        "candidate_lists": [first, second],
        "counts": {
            "detected_sections": 2,
            "sections_without_kept_contact": 0,
            "kept_contacts": 1,
            "candidate_lists": 2,
            "candidate_entries": 6,
            "earlier_candidate_entries": 4,
        },
    }

    with pytest.raises(ValueError, match="more than one section"):
        saver._assemble_result("abc1234", [], [video], ["sset_18"])


def test_full_validation_input_save_is_repeatable_and_label_free(
    tmp_path,
    monkeypatch,
) -> None:
    video = _video()
    rows = _score_rows()
    checked_run = SimpleNamespace(
        run=SimpleNamespace(run_id=saver.CHOSEN_RUN_ID),
        score_rows=rows,
    )
    feature_rows = np.empty(3, dtype=[("frame", "<i4")])
    feature_rows["frame"] = (1, 4, 10)
    verified = SimpleNamespace(
        runs=[checked_run],
        split=SimpleNamespace(validation_videos=[video]),
        raw_features=SimpleNamespace(rows=feature_rows),
    )
    paths = {
        name: tmp_path / name
        for name in (
            "config.json",
            "split.json",
            "raw.json",
            "common.json",
            "labels.csv",
            "baseline.json",
            "menu.json",
            "predictions.json.gz",
            "rallies.json.gz",
            "scores.npy.xz",
            "candidate_summary.json",
            "candidates.json.gz",
        )
    }
    for path in paths.values():
        path.write_bytes(b"checked")
    saver._write_json(
        paths["predictions.json.gz"],
        {
            "videos": [
                {
                    "fixture": video.fixture,
                    "spans": [{"span_id": 0, "start_frame": 5, "end_frame": 20}],
                }
            ],
            "runs": [
                {
                    "run_id": saver.CHOSEN_RUN_ID,
                    "videos": [
                        {
                            "fixture": video.fixture,
                            "contacts": [
                                {
                                    "frame": 10,
                                    "timing_score": 0.95,
                                    "predicted_side": "Top",
                                    "span_id": 0,
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    )
    files = saver.ValidationInputFiles(
        paths["config.json"],
        paths["split.json"],
        paths["raw.json"],
        paths["common.json"],
        paths["labels.csv"],
        paths["baseline.json"],
        paths["menu.json"],
        paths["predictions.json.gz"],
        paths["rallies.json.gz"],
        paths["scores.npy.xz"],
        paths["candidate_summary.json"],
        paths["candidates.json.gz"],
    )
    raw_candidate_list = _candidate_list()
    for candidate in raw_candidate_list["candidates"]:
        candidate.pop("kept")
        candidate.pop("predicted_side")
    feature_record = {"feature_summary": {"frame_count": 20}}
    monkeypatch.setattr(saver, "EXPECTED_VIDEO_COUNT", 1)
    monkeypatch.setattr(saver, "EXPECTED_CANDIDATE_LIST_COUNT", 1)
    monkeypatch.setattr(saver, "EXPECTED_CANDIDATE_ENTRY_COUNT", 3)
    monkeypatch.setattr(saver, "check_validation_reproduction", lambda *_args: ())
    monkeypatch.setattr(
        saver,
        "_build_validation_candidate_lists",
        lambda *_args: ([raw_candidate_list], 0),
    )
    monkeypatch.setattr(saver, "_video_record", lambda *_args: feature_record)
    monkeypatch.setattr(saver, "_checked_stage_files", lambda *_args: [])
    monkeypatch.setattr(
        saver,
        "_spans",
        lambda *_args: [{"span_id": 0, "start_frame": 5, "end_frame": 20}],
    )
    monkeypatch.setattr(saver, "_video_feature_rows", lambda *_args: feature_rows)
    monkeypatch.setattr(saver, "_check_centre_feature_values", lambda *_args: None)
    pose = SimpleNamespace(bboxes=np.zeros((20, 1, 4)))
    court = SimpleNamespace(
        evidence=SimpleNamespace(inputs=SimpleNamespace(net_band=(0.4, 0.6)))
    )
    loaded = (
        np.zeros((20, 3)),
        pose,
        court,
        [],
        SimpleNamespace(),
        SimpleNamespace(),
    )
    output_path = tmp_path / saver.RESULT_FILENAME

    saver.save_validation_rally_start_inputs(
        files,
        tmp_path,
        output_path,
        "abc1234",
        menu_loader=lambda *_args: verified,
        input_loader=lambda *_args: loaded,
        side_attributor=lambda frame, *_args: None if frame == 1 else "Top",
    )
    first_bytes = output_path.read_bytes()
    saver.save_validation_rally_start_inputs(
        files,
        tmp_path,
        output_path,
        "abc1234",
        menu_loader=lambda *_args: verified,
        input_loader=lambda *_args: loaded,
        side_attributor=lambda frame, *_args: None if frame == 1 else "Top",
    )

    saved = saver._read_json(output_path, "saved validation input")
    assert output_path.read_bytes() == first_bytes
    assert saved["status"] == "complete"
    assert saved["labels_read"] is False
    assert saved["counts"]["candidate_entries"] == 3
    assert (
        saved["videos"][0]["candidate_lists"][0]["candidates"][2]["predicted_side"]
        is None
    )


def test_changed_prediction_hash_stops_before_video_inputs(
    tmp_path,
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "config.json",
            "split.json",
            "raw.json",
            "common.json",
            "labels.csv",
            "baseline.json",
            "menu.json",
            "predictions.json.gz",
            "rallies.json.gz",
            "scores.npy.xz",
            "candidate_summary.json",
            "candidates.json.gz",
        )
    }
    for path in paths.values():
        path.write_bytes(b"checked")
    saver._write_json(
        paths["baseline.json"],
        {
            "chosen_run_id": saver.CHOSEN_RUN_ID,
            "prediction_file": paths["predictions.json.gz"].name,
            "prediction_sha256": "0" * 64,
            "result_file": paths["rallies.json.gz"].name,
            "result_sha256": "0" * 64,
            "chosen_run_files": {
                "score_file": paths["scores.npy.xz"].name,
                "score_sha256": "0" * 64,
            },
        },
    )
    files = saver.ValidationInputFiles(
        paths["config.json"],
        paths["split.json"],
        paths["raw.json"],
        paths["common.json"],
        paths["labels.csv"],
        paths["baseline.json"],
        paths["menu.json"],
        paths["predictions.json.gz"],
        paths["rallies.json.gz"],
        paths["scores.npy.xz"],
        paths["candidate_summary.json"],
        paths["candidates.json.gz"],
    )
    output_path = tmp_path / saver.RESULT_FILENAME

    with pytest.raises(ValueError, match="saved validation file hashes differ"):
        saver.save_validation_rally_start_inputs(
            files,
            tmp_path,
            output_path,
            "abc1234",
            menu_loader=lambda *_args: SimpleNamespace(),
            input_loader=lambda *_args: pytest.fail("large video input was loaded"),
        )

    saved = saver._read_json(output_path, "failed validation input")
    assert saved["status"] == "running"
    assert saved["labels_read"] is False
