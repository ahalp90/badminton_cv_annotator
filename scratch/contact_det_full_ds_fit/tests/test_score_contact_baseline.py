from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scratch.contact_det_full_ds_fit.scripts.score_contact_baseline as baseline
from scratch.contact_det.scripts.freeze_tree_contact_features import REGION_FIELDS
from scratch.contact_det_full_ds_fit.scripts.baseline_config import load_baseline_config
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DevelopmentSplit,
    SplitRole,
    VideoSpec,
    load_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    VerifiedFeatureDataset,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    CandidateRows,
    ContactLabels,
    choose_training_rows,
    choose_validation_settings,
    remove_nearby_contacts,
    run_baseline,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXPERIMENT_ROOT / "records/baseline_runs.json"
SPLIT_PATH = EXPERIMENT_ROOT / "records/shuttleset_development_split.json"


def _video(name: str, video_id: int, role: SplitRole, fps: float = 30.0) -> VideoSpec:
    return VideoSpec(name, video_id, fps, 1920, 1080, role, "winner", "loser", "event", "round")


def _small_split() -> DevelopmentSplit:
    return DevelopmentSplit(
        dataset="ShuttleSet",
        excluded_video_ids=(),
        expected_train_count=1,
        expected_validation_count=1,
        videos=(
            _video("sset_01", 1, SplitRole.TRAIN),
            _video("sset_02", 2, SplitRole.VALIDATION),
        ),
    )


def test_training_rows_use_training_videos_only_and_are_repeatable() -> None:
    split = _small_split()
    frames = np.concatenate([np.arange(101), np.arange(100, 106)]).astype(np.int32)
    rows = np.empty(len(frames), dtype=[("frame", "<i4")])
    rows["frame"] = frames
    candidates = CandidateRows(rows, {"sset_01": (0, 101), "sset_02": (101, 107)})
    labels = ContactLabels(
        frames={"sset_01": np.asarray([2], dtype=np.int32)},
        first_contacts={"sset_01": frozenset({2})},
        rally_counts={"sset_01": 1},
    )
    config = load_baseline_config(CONFIG_PATH)
    run = config.runs[0]

    first = choose_training_rows(candidates, split, labels, config, run)
    second = choose_training_rows(candidates, split, labels, config, run)

    assert np.array_equal(first.selected, second.selected)
    assert np.array_equal(first.labels, second.labels)
    assert not first.selected[101:].any()
    assert first.video_counts["sset_01"] == {
        "positive": 3,
        "nearby_negative": 11,
        "sampled_other_negative": 25,
        "selected": 39,
    }


def test_nearby_contacts_are_removed_separately_in_each_search_interval() -> None:
    kept = remove_nearby_contacts(
        frames=np.asarray([10, 12, 15, 11]),
        interval_ids=np.asarray([0, 0, 0, 1]),
        scores=np.asarray([0.8, 0.9, 0.7, 0.6]),
        cutoff=0.5,
        distance=4,
    )

    assert kept.tolist() == [3, 1]


def test_validation_ties_choose_larger_distance_then_higher_cutoff() -> None:
    split = _small_split()
    scores = np.zeros(1, dtype=baseline.SCORE_DTYPE)
    scores["fixture"] = b"sset_02"
    scores["frame"] = 100
    scores["fps"] = 30
    scores["contact_score"] = 0.01
    labels = ContactLabels(
        frames={"sset_02": np.asarray([100], dtype=np.int32)},
        first_contacts={"sset_02": frozenset({100})},
        rally_counts={"sset_02": 1},
    )
    config = load_baseline_config(CONFIG_PATH)

    cutoff, distance, counts, predictions, kept = choose_validation_settings(
        scores,
        split,
        labels,
        config,
    )

    assert cutoff == 0.95
    assert distance == 6
    assert counts["f1"] == 0.0
    assert predictions["sset_02"].tolist() == []
    assert not kept.any()


def test_feature_failure_happens_before_contact_labels_are_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels_read = False

    def fail_feature_check(*_args: object, **_kwargs: object) -> VerifiedFeatureDataset:
        raise ValueError("feature check failed")

    def read_labels(_path: Path, _split: DevelopmentSplit) -> ContactLabels:
        nonlocal labels_read
        labels_read = True
        raise AssertionError("labels must not be read")

    monkeypatch.setattr(baseline, "load_verified_feature_dataset", fail_feature_check)
    old_result = tmp_path / "results" / "hgb_reference_raw_balanced" / baseline.RESULT_FILE
    old_result.parent.mkdir(parents=True)
    old_result.write_text('{"status": "complete"}', encoding="utf-8")

    with pytest.raises(ValueError, match="feature check failed"):
        run_baseline(
            CONFIG_PATH,
            "hgb_reference_raw_balanced",
            tmp_path / "features.json",
            SPLIT_PATH,
            tmp_path / "shots_master.csv",
            tmp_path / "results",
            "deadbee",
            label_loader=read_labels,
        )

    assert labels_read is False
    assert json.loads(old_result.read_text(encoding="utf-8"))["status"] == "running"


class _FixedModel:
    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> None:
        assert len(matrix) == len(labels)
        assert set(labels.tolist()) == {0, 1}

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        contact_scores = np.where(matrix[:, 0] == 1.0, 0.9, 0.1)
        return np.column_stack([1.0 - contact_scores, contact_scores])


def _full_fake_features(feature_record: Path) -> VerifiedFeatureDataset:
    split = load_development_split(SPLIT_PATH)
    fields: list[tuple[str, str]] = [
        ("fixture", "S7"),
        ("interval_id", "<i4"),
        ("frame", "<i4"),
        ("fps", "<f4"),
        ("model_value", "<f4"),
    ]
    fields.extend((field, "?") for field in REGION_FIELDS)
    rows = np.zeros(len(split.videos) * 2, dtype=np.dtype(fields))
    video_ranges: dict[str, tuple[int, int]] = {}
    for index, video in enumerate(split.videos):
        row_start = index * 2
        row_end = row_start + 2
        video_ranges[video.fixture] = (row_start, row_end)
        rows["fixture"][row_start:row_end] = video.fixture.encode("ascii")
        rows["interval_id"][row_start:row_end] = 0
        rows["frame"][row_start:row_end] = [10, 100]
        rows["fps"][row_start:row_end] = video.fps
        rows["model_value"][row_start:row_end] = [1.0, 0.0]
        rows[REGION_FIELDS[0]][row_start:row_end] = True
    return VerifiedFeatureDataset(
        feature_record,
        {"source_commit": "a4c8ec3b"},
        split,
        rows,
        video_ranges,
        ("model_value",),
    )


def _full_fake_labels(split: DevelopmentSplit) -> ContactLabels:
    frames = {video.fixture: np.asarray([10], dtype=np.int32) for video in split.videos}
    first_contacts = {video.fixture: frozenset({10}) for video in split.videos}
    rally_counts = {video.fixture: 1 for video in split.videos}
    return ContactLabels(frames, first_contacts, rally_counts)


def test_complete_result_contains_portable_inputs_and_saved_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_record = tmp_path / "contact_features_record.json"
    feature_record.write_text("{}", encoding="utf-8")
    shots_master = tmp_path / "shots_master.csv"
    shots_master.write_text("test labels", encoding="utf-8")
    fake_features = _full_fake_features(feature_record)
    monkeypatch.setattr(baseline, "load_verified_feature_dataset", lambda *_args: fake_features)

    result_path = run_baseline(
        CONFIG_PATH,
        "hgb_reference_raw_balanced",
        feature_record,
        SPLIT_PATH,
        shots_master,
        tmp_path / "results",
        "deadbee",
        label_loader=lambda _path, split: _full_fake_labels(split),
        model_factory=lambda _run, _seed: _FixedModel(),
    )

    result_text = result_path.read_text(encoding="utf-8")
    result: dict[str, Any] = json.loads(result_text)
    score_path = result_path.parent / baseline.SCORE_FILE
    with baseline.lzma.open(score_path, "rb") as source:
        scores = np.load(source, allow_pickle=False)
    assert result["status"] == "complete"
    assert result["training_videos"] == [video.fixture for video in fake_features.split.training_videos]
    assert result["validation_videos"] == [video.fixture for video in fake_features.split.validation_videos]
    assert result["validation_score_sha256"] == baseline._sha256(score_path)
    assert len(scores) == 16
    assert int(scores["kept"].sum()) == 8
    assert str(tmp_path) not in result_text


def test_run_adds_the_repository_source_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(baseline.sys, "path", [entry for entry in baseline.sys.path if entry != str(baseline.REPO_ROOT / "src")])
    monkeypatch.setattr(
        baseline,
        "load_verified_feature_dataset",
        lambda *_args: (_ for _ in ()).throw(ValueError("stop after import setup")),
    )

    with pytest.raises(ValueError, match="stop after import setup"):
        run_baseline(
            CONFIG_PATH,
            "hgb_reference_raw_balanced",
            tmp_path / "features.json",
            SPLIT_PATH,
            tmp_path / "shots_master.csv",
            tmp_path / "results",
            "deadbee",
        )

    assert baseline.sys.path[0] == str(baseline.REPO_ROOT / "src")
