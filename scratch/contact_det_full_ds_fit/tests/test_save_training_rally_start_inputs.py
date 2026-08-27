from __future__ import annotations

import copy
import lzma
from types import SimpleNamespace

import numpy as np
import pytest

from scratch.contact_det_full_ds_fit.scripts import (
    save_training_rally_start_inputs as saver,
)
from scratch.contact_det_full_ds_fit.scripts.score_training_videos import (
    COMBINED_SCORE_DTYPE,
    SCORE_DTYPE,
    ScoreGroup,
)


def _video(name: str, video_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        fixture=name,
        video_id=video_id,
        fps=30.0,
        width=1920,
        height=1080,
    )


def _combined_rows(
    video_name: str = "sset_01",
    frames: tuple[int, ...] = (0, 10),
) -> np.ndarray:
    rows = np.empty(len(frames), dtype=COMBINED_SCORE_DTYPE)
    rows["fixture"] = video_name.encode("ascii")
    rows["interval_id"] = 0
    rows["frame"] = frames
    rows["fps"] = 30.0
    rows["contact_score"] = (0.4, 0.95)[: len(frames)]
    rows["kept"] = (False, True)[: len(frames)]
    rows["group"] = b"A"
    return rows


def _checked_for_one_video() -> saver.CheckedTrainingScores:
    video = _video("sset_01")
    rows = _combined_rows()
    feature_rows = np.empty(2, dtype=[("frame", "<i4")])
    feature_rows["frame"] = (0, 10)
    feature_record = {
        "video": {"name": video.fixture},
        "feature_file": "videos/sset_01/contact_features.npy.xz",
        "feature_sha256": "a" * 64,
        "feature_summary": {
            "frame_count": 20,
            "rally_span_count": 1,
            "search_intervals": [[0, 20]],
        },
    }
    features = SimpleNamespace(
        record={"videos": [feature_record]},
        rows=feature_rows,
        video_ranges={video.fixture: (0, 2)},
    )
    group = ScoreGroup(
        "A", (video,), tuple(_video(f"train_{index}") for index in range(24))
    )
    return saver.CheckedTrainingScores(
        rows,
        {"A": group},
        features,
        ({"role": "fixed", "filename": "fixed.json", "sha256": "b" * 64},),
        {"A": {}},
    )


def test_combined_score_reader_requires_the_group_field(tmp_path) -> None:
    path = tmp_path / "scores.npy.xz"
    rows = np.empty(1, dtype=SCORE_DTYPE)
    with lzma.open(path, "wb") as destination:
        np.save(destination, rows, allow_pickle=False)

    with pytest.raises(ValueError, match="fields differ"):
        saver._read_combined_scores(path)


def test_combined_scores_must_equal_the_four_child_files(tmp_path) -> None:
    combined_chunks = []
    for index, group_name in enumerate(saver.GROUP_NAMES):
        combined = _combined_rows(f"sset_{index + 1:02d}", (index,))
        combined["group"] = group_name.encode("ascii")
        combined_chunks.append(combined)
        child = np.empty(1, dtype=SCORE_DTYPE)
        for field in SCORE_DTYPE.names or ():
            child[field] = combined[field]
        score_path = tmp_path / f"group_{group_name}" / saver.GROUP_SCORE_FILE
        score_path.parent.mkdir(parents=True)
        with lzma.open(score_path, "wb") as destination:
            np.save(destination, child, allow_pickle=False)
    combined_rows = np.concatenate(combined_chunks)
    combined_rows["contact_score"][2] = 0.2

    with pytest.raises(ValueError, match="group C combined score rows differ"):
        saver._check_child_score_rows(
            combined_rows,
            saver.TrainingScorePaths(tmp_path / "summary.json", tmp_path),
        )


def test_group_result_rejects_a_scored_video_in_its_training_list(tmp_path) -> None:
    scored = _video("sset_01")
    group = ScoreGroup("A", (scored,), (scored,))
    result = {
        "schema": saver.RESULT_SCHEMA,
        "status": "complete",
        "group": "A",
        "source_commit": "abc1234",
        "run_id": saver.CHOSEN_RUN_ID,
        "training_videos": [scored.fixture],
        "scored_videos": [scored.fixture],
    }
    paths = saver.TrainingScorePaths(tmp_path / "summary.json", tmp_path)
    files = SimpleNamespace()

    with pytest.raises(ValueError, match="model separation"):
        saver._check_group_result(
            "A",
            group,
            result,
            {},
            {},
            paths,
            files,
            SimpleNamespace(),
            SimpleNamespace(),
            "abc1234",
        )


def test_validation_gate_checks_the_frozen_list_and_side_replay(
    tmp_path,
    monkeypatch,
) -> None:
    paths_by_name = {
        name: tmp_path / name
        for name in (
            "menu.json",
            "common.json",
            "predictions.json.gz",
            "rallies.json.gz",
            "chosen.npy.xz",
            "candidate_summary.json",
            "candidates.json.gz",
            "baseline_summary.json",
        )
    }
    for name in ("menu.json", "common.json", "rallies.json.gz", "chosen.npy.xz"):
        paths_by_name[name].write_bytes(name.encode("ascii"))
    saved_predictions = {
        "labels_read": False,
        "runs": [{"run_id": saver.CHOSEN_RUN_ID, "videos": []}],
        "videos": [],
    }
    saver._write_json(paths_by_name["predictions.json.gz"], saved_predictions)
    candidate_lists = [
        {
            "fixture": "sset_18",
            "span_id": span_id,
            "candidates": [
                {
                    "frame": span_id * 3 + offset,
                    "contact_score": 0.5,
                    "is_fixed_contact": offset == 0,
                }
                for offset in range(3)
            ],
        }
        for span_id in range(615)
    ]
    saver._write_json(
        paths_by_name["candidates.json.gz"],
        {
            "counts": {
                "candidate_lists": 615,
                "sections_without_kept_contact": 62,
            },
            "candidate_lists": candidate_lists,
        },
    )
    saver._write_json(
        paths_by_name["candidate_summary.json"],
        {
            "run_id": saver.CHOSEN_RUN_ID,
            "construction_file": paths_by_name["candidates.json.gz"].name,
            "construction_sha256": saver._sha256(paths_by_name["candidates.json.gz"]),
        },
    )
    saver._write_json(
        paths_by_name["baseline_summary.json"],
        {
            "chosen_run_id": saver.CHOSEN_RUN_ID,
            "prediction_file": paths_by_name["predictions.json.gz"].name,
            "prediction_sha256": saver._sha256(paths_by_name["predictions.json.gz"]),
            "result_file": paths_by_name["rallies.json.gz"].name,
            "result_sha256": saver._sha256(paths_by_name["rallies.json.gz"]),
            "chosen_run_files": {
                "score_file": paths_by_name["chosen.npy.xz"].name,
                "score_sha256": saver._sha256(paths_by_name["chosen.npy.xz"]),
            },
        },
    )
    monkeypatch.setattr(
        saver,
        "build_validation_rally_predictions",
        lambda *_args: dict(saved_predictions),
    )
    monkeypatch.setattr(
        saver,
        "_build_validation_candidate_lists",
        lambda *_args: (candidate_lists, 62),
    )
    validation_paths = saver.ValidationPaths(
        paths_by_name["menu.json"],
        paths_by_name["common.json"],
        paths_by_name["predictions.json.gz"],
        paths_by_name["rallies.json.gz"],
        paths_by_name["chosen.npy.xz"],
        paths_by_name["candidate_summary.json"],
        paths_by_name["candidates.json.gz"],
    )

    records = saver.check_validation_reproduction(
        SimpleNamespace(),
        validation_paths,
        paths_by_name["baseline_summary.json"],
        tmp_path,
        "abc1234",
    )

    assert len(records) == 7


def test_training_video_save_is_repeatable_and_resume_rechecks_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    checked = _checked_for_one_video()
    video = checked.groups["A"].scored_videos[0]
    checked_files = [
        {
            "role": "annotation",
            "filename": "result.json.gz",
            "size_bytes": 4,
            "sha256": "c" * 64,
        }
    ]
    monkeypatch.setattr(saver, "_checked_stage_files", lambda *_args: checked_files)
    monkeypatch.setattr(saver, "_check_centre_feature_values", lambda *_args: None)
    pose = SimpleNamespace(bboxes=np.zeros((20, 1, 4)), kps=np.zeros((20, 1, 17, 2)))
    court = SimpleNamespace(
        evidence=SimpleNamespace(inputs=SimpleNamespace(net_band=(0.4, 0.6)))
    )
    annotation = SimpleNamespace(spans=((5, 20),))
    loaded = (
        np.zeros((20, 3)),
        pose,
        court,
        [],
        SimpleNamespace(),
        annotation,
    )
    output_path = tmp_path / "video.json.gz"

    first = saver.save_training_video(
        checked,
        video,
        "A",
        tmp_path,
        output_path,
        "abc1234",
        resume=False,
        input_loader=lambda *_args: loaded,
        side_attributor=lambda *_args: "Top",
    )
    first_bytes = output_path.read_bytes()
    second = saver.save_training_video(
        checked,
        video,
        "A",
        tmp_path,
        output_path,
        "abc1234",
        resume=True,
        input_loader=lambda *_args: pytest.fail("resume reloaded the large inputs"),
        side_attributor=lambda *_args: "Top",
    )

    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert first["labels_read"] is False
    assert first["counts"]["kept_contacts"] == 1
    assert first["candidate_lists"][0]["candidates"][0]["predicted_side"] == "Top"

    changed = copy.deepcopy(first)
    changed["candidate_lists"][0]["candidates"][0]["contact_score"] = 0.1
    saver._write_json(output_path, changed)
    with pytest.raises(ValueError, match="saved candidate fields differ"):
        saver.save_training_video(
            checked,
            video,
            "A",
            tmp_path,
            output_path,
            "abc1234",
            resume=True,
            input_loader=lambda *_args: pytest.fail("resume reloaded the large inputs"),
            side_attributor=lambda *_args: "Top",
        )


def test_failed_video_save_leaves_running_status(tmp_path, monkeypatch) -> None:
    checked = _checked_for_one_video()
    video = checked.groups["A"].scored_videos[0]
    monkeypatch.setattr(saver, "_checked_stage_files", lambda *_args: [])
    output_path = tmp_path / "video.json.gz"

    with pytest.raises(RuntimeError, match="stopped"):
        saver.save_training_video(
            checked,
            video,
            "A",
            tmp_path,
            output_path,
            "abc1234",
            resume=False,
            input_loader=lambda *_args: (_ for _ in ()).throw(RuntimeError("stopped")),
            side_attributor=lambda *_args: "Top",
        )

    assert saver._read_json(output_path, "failed video")["status"] == "running"


def test_stage_hash_failure_replaces_an_old_complete_status(
    tmp_path, monkeypatch
) -> None:
    checked = _checked_for_one_video()
    video = checked.groups["A"].scored_videos[0]
    output_path = tmp_path / "video.json.gz"
    saver._write_json(
        output_path,
        {
            "schema": saver.VIDEO_SCHEMA,
            "status": "complete",
            "source_commit": "abc1234",
        },
    )
    monkeypatch.setattr(
        saver,
        "_checked_stage_files",
        lambda *_args: (_ for _ in ()).throw(ValueError("stage hash differs")),
    )

    with pytest.raises(ValueError, match="stage hash differs"):
        saver.save_training_video(
            checked,
            video,
            "A",
            tmp_path,
            output_path,
            "abc1234",
            resume=True,
            input_loader=lambda *_args: pytest.fail("stage inputs were loaded"),
            side_attributor=lambda *_args: "Top",
        )

    assert saver._read_json(output_path, "failed resume")["status"] == "running"


def test_combined_check_rejects_a_candidate_used_by_two_sections() -> None:
    videos = [
        {
            "video": {"fixture": "sset_01"},
            "candidate_lists": [
                {"candidates": [{"frame": 10}]},
                {"candidates": [{"frame": 10}]},
            ],
        }
    ]

    with pytest.raises(ValueError, match="more than one section"):
        saver._check_combined_video_values(videos, ["sset_01"], [None])


def test_combined_check_requires_each_videos_fixed_group() -> None:
    videos = [
        {
            "group": "B",
            "video": {"fixture": "sset_01"},
            "candidate_lists": [],
        }
    ]

    with pytest.raises(ValueError, match="video groups differ"):
        saver._check_combined_video_values(videos, ["sset_01"], ["A"])
