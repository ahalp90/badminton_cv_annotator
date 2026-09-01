from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np
import pytest

from scratch.contact_det.scripts.freeze_tree_contact_features import REGION_FIELDS
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DevelopmentSplit,
    load_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    VerifiedFeatureDataset,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import ContactLabels
from scratch.contact_det_full_ds_fit.scripts.score_training_videos import (
    COMBINED_SCORE_DTYPE,
    COMBINED_SCORE_FILE,
    GROUP_RESULT_FILE,
    GROUP_SCORE_FILE,
    InputFiles,
    _sha256,
    combine_groups,
    load_score_groups,
    score_group,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
GROUPS_PATH = EXPERIMENT_ROOT / "records/training_video_score_groups.json"
SPLIT_PATH = EXPERIMENT_ROOT / "records/shuttleset_development_split.json"
CONFIG_PATH = EXPERIMENT_ROOT / "records/baseline_runs.json"
SUMMARY_PATH = EXPERIMENT_ROOT / "records/baseline_summary.json"
CHOSEN_RESULT_PATH = (
    EXPERIMENT_ROOT
    / "raw"
    / "baseline_runs"
    / "hgb_reference_raw_more_negatives"
    / "baseline_result.json"
)


def test_fixed_groups_cover_each_training_video_once() -> None:
    split = load_development_split(SPLIT_PATH)
    groups = load_score_groups(GROUPS_PATH, split)

    scored = [video.fixture for group in groups.values() for video in group.scored_videos]
    assert len(scored) == 32
    assert len(set(scored)) == 32
    for group in groups.values():
        training = {video.fixture for video in group.training_videos}
        scored_group = {video.fixture for video in group.scored_videos}
        validation = {video.fixture for video in split.validation_videos}
        assert len(training) == 24
        assert training.isdisjoint(scored_group)
        assert training.isdisjoint(validation)


class _FixedModel:
    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> None:
        assert len(matrix) == len(labels)
        assert set(labels.tolist()) == {0, 1}

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        contact_scores = np.where(matrix[:, 0] == 1.0, 0.95, 0.05)
        return np.column_stack([1.0 - contact_scores, contact_scores])


def _fake_features(feature_record: Path) -> VerifiedFeatureDataset:
    split = load_development_split(SPLIT_PATH)
    fields: list[tuple[str, str]] = [
        ("fixture", "S7"),
        ("interval_id", "<i4"),
        ("frame", "<i4"),
        ("fps", "<f4"),
        ("model_value", "<f4"),
    ]
    fields.extend((field, "u1") for field in REGION_FIELDS)
    rows = np.zeros(len(split.videos) * 4, dtype=np.dtype(fields))
    video_ranges: dict[str, tuple[int, int]] = {}
    for index, video in enumerate(split.videos):
        row_start = index * 4
        row_end = row_start + 4
        video_ranges[video.fixture] = (row_start, row_end)
        rows["fixture"][row_start:row_end] = video.fixture.encode("ascii")
        rows["interval_id"][row_start:row_end] = 0
        rows["frame"][row_start:row_end] = [10, 12, 20, 100]
        rows["fps"][row_start:row_end] = video.fps
        rows["model_value"][row_start:row_end] = [1.0, 0.0, 0.0, 0.0]
        rows[REGION_FIELDS[0]][row_start:row_end] = 1
    return VerifiedFeatureDataset(
        feature_record,
        {"source_commit": "a4c8ec3b"},
        split,
        rows,
        video_ranges,
        ("model_value",),
    )


def _fake_labels(split: DevelopmentSplit) -> ContactLabels:
    frames = {video.fixture: np.asarray([10], dtype=np.int32) for video in split.videos}
    first_contacts = {video.fixture: frozenset({10}) for video in split.videos}
    rally_counts = {video.fixture: 1 for video in split.videos}
    return ContactLabels(frames, first_contacts, rally_counts)


def _temporary_inputs(tmp_path: Path) -> InputFiles:
    feature_record = tmp_path / "contact_features_record.json"
    feature_record.write_text("{}\n", encoding="utf-8")
    contact_labels = tmp_path / "shots_master.csv"
    contact_labels.write_text("test labels\n", encoding="utf-8")
    paths_by_role = {
        "training score groups": GROUPS_PATH,
        "development split": SPLIT_PATH,
        "baseline settings": CONFIG_PATH,
        "raw feature record": feature_record,
        "baseline summary": SUMMARY_PATH,
        "chosen baseline result": CHOSEN_RESULT_PATH,
        "contact labels": contact_labels,
    }
    split = load_development_split(SPLIT_PATH)
    expected_counts = {video.fixture: 4 for video in split.training_videos}
    input_list = tmp_path / "training_video_score_inputs.json"
    input_list.write_text(
        json.dumps(
            {
                "schema": "contact-training-video-score-inputs/1",
                "expected_training_video_count": 32,
                "expected_candidate_score_row_count": 128,
                "candidate_row_condition": "At least one listed unsigned-byte field equals 1",
                "candidate_row_fields": list(REGION_FIELDS),
                "expected_score_rows_by_video": expected_counts,
                "files": [
                    {
                        "role": role,
                        "filename": path.name,
                        "sha256": _sha256(path),
                    }
                    for role, path in paths_by_role.items()
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return InputFiles(
        input_list,
        GROUPS_PATH,
        SPLIT_PATH,
        CONFIG_PATH,
        feature_record,
        SUMMARY_PATH,
        CHOSEN_RESULT_PATH,
        contact_labels,
    )


def test_feature_failure_happens_before_contact_labels_are_read(tmp_path: Path) -> None:
    files = _temporary_inputs(tmp_path)
    labels_read = False

    def fail_features(*_args: object) -> VerifiedFeatureDataset:
        raise ValueError("feature check failed")

    def read_labels(_path: Path, _split: DevelopmentSplit) -> ContactLabels:
        nonlocal labels_read
        labels_read = True
        raise AssertionError("labels must not be read")

    with pytest.raises(ValueError, match="feature check failed"):
        score_group(
            files,
            "A",
            tmp_path / "failed",
            "deadbee",
            feature_loader=fail_features,
            label_loader=read_labels,
        )

    assert labels_read is False
    saved = json.loads((tmp_path / "failed" / GROUP_RESULT_FILE).read_text(encoding="utf-8"))
    assert saved["status"] == "running"


def test_group_scores_repeat_and_combine_to_every_training_video(tmp_path: Path) -> None:
    files = _temporary_inputs(tmp_path)
    features = _fake_features(files.feature_record)
    feature_loader = lambda *_args: features
    label_loader = lambda _path, split: _fake_labels(split)
    model_factory = lambda _run, _seed: _FixedModel()

    first_a = tmp_path / "group_A_first"
    group_dirs = {name: tmp_path / f"group_{name}" for name in "ABCD"}
    score_group(
        files,
        "A",
        first_a,
        "deadbee",
        feature_loader=feature_loader,
        label_loader=label_loader,
        model_factory=model_factory,
    )
    for name, output_dir in group_dirs.items():
        score_group(
            files,
            name,
            output_dir,
            "deadbee",
            feature_loader=feature_loader,
            label_loader=label_loader,
            model_factory=model_factory,
        )

    assert (first_a / GROUP_SCORE_FILE).read_bytes() == (
        group_dirs["A"] / GROUP_SCORE_FILE
    ).read_bytes()
    assert (first_a / GROUP_RESULT_FILE).read_bytes() == (
        group_dirs["A"] / GROUP_RESULT_FILE
    ).read_bytes()

    result_path = combine_groups(
        files,
        group_dirs,
        tmp_path / "combined",
        "deadbee",
        feature_loader=feature_loader,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with lzma.open(tmp_path / "combined" / COMBINED_SCORE_FILE, "rb") as source:
        scores = np.load(source, allow_pickle=False)

    assert result["status"] == "complete"
    assert result["score_row_count"] == 128
    assert scores.dtype == COMBINED_SCORE_DTYPE
    assert len(set(np.char.decode(scores["fixture"], "ascii"))) == 32
    assert set(np.char.decode(scores["group"], "ascii")) == set("ABCD")
    assert int(scores["kept"].sum()) == 32

    second_result = combine_groups(
        files,
        group_dirs,
        tmp_path / "combined_second",
        "deadbee",
        feature_loader=feature_loader,
    )
    assert (tmp_path / "combined" / COMBINED_SCORE_FILE).read_bytes() == (
        tmp_path / "combined_second" / COMBINED_SCORE_FILE
    ).read_bytes()
    assert result_path.read_bytes() == second_result.read_bytes()

    group_d_result = group_dirs["D"] / GROUP_RESULT_FILE
    group_d_result.write_text('{"status": "running"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="group D result is not complete"):
        combine_groups(
            files,
            group_dirs,
            tmp_path / "combined",
            "deadbee",
            feature_loader=feature_loader,
        )
    failed_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert failed_result["status"] == "running"
