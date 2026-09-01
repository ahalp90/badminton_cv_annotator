from __future__ import annotations

import json
import lzma
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scratch.contact_det.scripts.freeze_tree_contact_features import REGION_FIELDS
from scratch.contact_det_full_ds_fit.scripts import fit_final_contact_model as fitter
from scratch.contact_det_full_ds_fit.scripts import score_final_contact_groups as scorer
from scratch.contact_det_full_ds_fit.scripts.baseline_config import load_baseline_config
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    VideoSpec,
    load_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.feature_dataset import (
    VerifiedFeatureDataset,
)
from scratch.contact_det_full_ds_fit.scripts.score_contact_baseline import (
    SCORE_DTYPE,
    ContactLabels,
    _sha256,
    collect_candidate_rows,
    predictions_for_settings,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
GROUPS_PATH = EXPERIMENT_ROOT / "records/final_video_score_groups.json"
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


class _FixedModel:
    classes_ = np.asarray([0, 1])

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> None:
        assert len(matrix) == len(labels)
        assert set(labels.tolist()) == {0, 1}

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        contact_scores = np.where(matrix[:, 0] == 1.0, 0.95, 0.05)
        return np.column_stack([1.0 - contact_scores, contact_scores])


def _fake_features(feature_record: Path) -> VerifiedFeatureDataset:
    split = load_development_split(SPLIT_PATH)
    fields: list[tuple[str, str]] = [
        ("fixture", "S8"),
        ("interval_id", "<i4"),
        ("frame", "<i4"),
        ("fps", "<f8"),
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
        {
            "source_commit": "deadbee",
            "videos": [
                {
                    "video": {"name": video.fixture},
                    "feature_sha256": f"{video.video_id:064x}",
                }
                for video in split.videos
            ],
        },
        split,
        rows,
        video_ranges,
        ("model_value",),
    )


def _fake_labels(videos: tuple[VideoSpec, ...] | list[VideoSpec]) -> ContactLabels:
    frames = {video.fixture: np.asarray([10], dtype=np.int32) for video in videos}
    first_contacts = {video.fixture: frozenset({10}) for video in videos}
    rally_counts = {video.fixture: 1 for video in videos}
    return ContactLabels(frames, first_contacts, rally_counts)


def _write_old_validation_scores(path: Path, features: VerifiedFeatureDataset) -> None:
    groups = scorer.load_final_score_groups(GROUPS_PATH, features.split)
    candidates = collect_candidate_rows(features)
    raw_scores = scorer._score_group_rows(
        candidates,
        groups["V"],
        _FixedModel(),
        features.model_input_fields,
    )
    old_scores = np.empty(len(raw_scores), dtype=SCORE_DTYPE)
    for field in ("fixture", "interval_id", "frame", "fps", "contact_score"):
        old_scores[field] = raw_scores[field]
    _predictions, kept = predictions_for_settings(
        old_scores,
        groups["V"].scored_videos,
        scorer.SCORE_CUTOFF,
        scorer.DUPLICATE_DISTANCE_AT_30_FPS,
    )
    old_scores["kept"] = kept
    with lzma.open(path, "wb") as destination:
        np.save(destination, old_scores, allow_pickle=False)


def _temporary_inputs(
    tmp_path: Path,
) -> tuple[scorer.InputFiles, VerifiedFeatureDataset]:
    feature_record = tmp_path / "contact_features_record.json"
    feature_record.write_text("{}\n", encoding="utf-8")
    features = _fake_features(feature_record)
    contact_labels = tmp_path / "shots_master.csv"
    contact_labels.write_text("test labels\n", encoding="utf-8")
    validation_scores = tmp_path / "validation_contact_scores.npy.xz"
    _write_old_validation_scores(validation_scores, features)
    paths_by_role = {
        "final score groups": GROUPS_PATH,
        "development split": SPLIT_PATH,
        "baseline settings": CONFIG_PATH,
        "raw feature record": feature_record,
        "baseline summary": SUMMARY_PATH,
        "chosen baseline result": CHOSEN_RESULT_PATH,
        "contact labels": contact_labels,
        "chosen validation scores": validation_scores,
    }
    input_list = tmp_path / "final_contact_score_inputs.json"
    input_list.write_text(
        json.dumps(
            {
                "schema": scorer.INPUT_SCHEMA,
                "expected_video_count": 40,
                "expected_candidate_score_row_count": 160,
                "candidate_row_condition": scorer.CANDIDATE_ROW_CONDITION,
                "candidate_row_fields": list(REGION_FIELDS),
                "expected_score_rows_by_video": {
                    video.fixture: 4 for video in features.split.videos
                },
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
    return (
        scorer.InputFiles(
            input_list,
            GROUPS_PATH,
            SPLIT_PATH,
            CONFIG_PATH,
            feature_record,
            SUMMARY_PATH,
            CHOSEN_RESULT_PATH,
            contact_labels,
            validation_scores,
        ),
        features,
    )


def test_final_groups_cover_each_development_video_once() -> None:
    split = load_development_split(SPLIT_PATH)
    groups = scorer.load_final_score_groups(GROUPS_PATH, split)

    scored = [video.fixture for group in groups.values() for video in group.scored_videos]
    assert len(scored) == 40
    assert len(set(scored)) == 40
    for group in groups.values():
        training = {video.fixture for video in group.training_videos}
        scored_group = {video.fixture for video in group.scored_videos}
        assert len(training) == 32
        assert training.isdisjoint(scored_group)


def test_label_loader_reads_only_requested_videos(tmp_path: Path) -> None:
    split = load_development_split(SPLIT_PATH)
    requested = split.videos[:2]
    labels_path = tmp_path / "shots.csv"
    labels_path.write_text(
        "vid,set_id,rally,frame_num\n"
        f"{requested[0].video_id},1,1,20\n"
        f"{requested[0].video_id},1,1,10\n"
        f"{requested[1].video_id},1,2,30\n"
        "999,not-an-integer,not-an-integer,not-an-integer\n",
        encoding="utf-8",
    )

    labels = scorer.load_contact_labels_for_videos(labels_path, requested)

    assert list(labels.frames) == [video.fixture for video in requested]
    assert labels.frames[requested[0].fixture].tolist() == [10, 20]
    assert labels.first_contacts[requested[0].fixture] == frozenset({10})


def test_scores_every_video_once_and_repeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files, features = _temporary_inputs(tmp_path)
    monkeypatch.setattr(scorer, "EXPECTED_SCORE_ROW_COUNT", 160)
    feature_loader = lambda *_args: features
    label_loader = lambda _path, videos: _fake_labels(tuple(videos))
    model_factory = lambda _run, _seed: _FixedModel()
    group_dirs = {name: tmp_path / f"group_{name}" for name in scorer.GROUP_NAMES}

    for name, output_dir in group_dirs.items():
        scorer.score_group(
            files,
            name,
            output_dir,
            "deadbee",
            feature_loader=feature_loader,
            label_loader=label_loader,
            model_factory=model_factory,
        )
    repeated_v = tmp_path / "group_V_repeated"
    scorer.score_group(
        files,
        "V",
        repeated_v,
        "deadbee",
        feature_loader=feature_loader,
        label_loader=label_loader,
        model_factory=model_factory,
    )

    assert (group_dirs["V"] / scorer.GROUP_SCORE_FILE).read_bytes() == (
        repeated_v / scorer.GROUP_SCORE_FILE
    ).read_bytes()
    assert (group_dirs["V"] / scorer.GROUP_RESULT_FILE).read_bytes() == (
        repeated_v / scorer.GROUP_RESULT_FILE
    ).read_bytes()

    first_result = scorer.combine_groups(
        files,
        group_dirs,
        tmp_path / "combined_first",
        "deadbee",
        feature_loader=feature_loader,
        label_loader=label_loader,
    )
    second_result = scorer.combine_groups(
        files,
        group_dirs,
        tmp_path / "combined_second",
        "deadbee",
        feature_loader=feature_loader,
        label_loader=label_loader,
    )
    first_payload = json.loads(first_result.read_text(encoding="utf-8"))
    with lzma.open(tmp_path / "combined_first" / scorer.COMBINED_RAW_SCORE_FILE, "rb") as source:
        combined = np.load(source, allow_pickle=False)

    assert first_payload["status"] == "complete"
    assert first_payload["labels_read"] is True
    assert len(combined) == 160
    assert "kept" not in combined.dtype.names
    assert len(set(np.char.decode(combined["fixture"], "ascii"))) == 40
    assert set(np.char.decode(combined["source_group"], "ascii")) == set(
        scorer.GROUP_NAMES
    )
    assert first_result.read_bytes() == second_result.read_bytes()
    assert (
        tmp_path / "combined_first" / scorer.COMBINED_RAW_SCORE_FILE
    ).read_bytes() == (
        tmp_path / "combined_second" / scorer.COMBINED_RAW_SCORE_FILE
    ).read_bytes()
    assert (tmp_path / "combined_first" / scorer.FINAL_SCORE_FILE).read_bytes() == (
        tmp_path / "combined_second" / scorer.FINAL_SCORE_FILE
    ).read_bytes()

    group_a_scores_path = group_dirs["A"] / scorer.GROUP_SCORE_FILE
    with lzma.open(group_a_scores_path, "rb") as source:
        group_a_scores = np.load(source, allow_pickle=False)
    group_a_scores["source_group"][0] = b"B"
    with lzma.open(group_a_scores_path, "wb") as destination:
        np.save(destination, group_a_scores, allow_pickle=False)
    group_a_result_path = group_dirs["A"] / scorer.GROUP_RESULT_FILE
    group_a_result = json.loads(group_a_result_path.read_text(encoding="utf-8"))
    group_a_result["raw_score_sha256"] = _sha256(group_a_scores_path)
    group_a_result_path.write_text(
        json.dumps(group_a_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="group A source fields differ"):
        scorer.combine_groups(
            files,
            group_dirs,
            tmp_path / "combined_bad_source",
            "deadbee",
            feature_loader=feature_loader,
            label_loader=label_loader,
        )


def test_equal_metrics_choose_the_larger_distance_and_higher_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = np.zeros(1, dtype=scorer.RAW_SCORE_DTYPE)
    config = replace(
        load_baseline_config(CONFIG_PATH),
        score_cutoffs=(0.05, 0.95),
        duplicate_distances_at_30_fps=(4, 6),
    )
    metrics = {"f1": 0.5, "recall": 0.5, "precision": 0.5}
    monkeypatch.setattr(
        scorer,
        "predictions_for_settings",
        lambda *_args: ({}, np.zeros(1, dtype=bool)),
    )
    monkeypatch.setattr(scorer, "contact_counts", lambda *_args: metrics)

    results, cutoff, distance, *_rest = scorer.choose_final_settings(
        scores,
        (),
        ContactLabels({}, {}, {}),
        config,
    )

    assert len(results) == 4
    assert cutoff == 0.95
    assert distance == 6


def test_failed_combine_records_when_labels_were_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files, features = _temporary_inputs(tmp_path)
    monkeypatch.setattr(scorer, "EXPECTED_SCORE_ROW_COUNT", 160)
    feature_loader = lambda *_args: features
    label_loader = lambda _path, videos: _fake_labels(tuple(videos))
    model_factory = lambda _run, _seed: _FixedModel()
    group_dirs = {name: tmp_path / f"group_{name}" for name in scorer.GROUP_NAMES}
    for name, output_dir in group_dirs.items():
        scorer.score_group(
            files,
            name,
            output_dir,
            "deadbee",
            feature_loader=feature_loader,
            label_loader=label_loader,
            model_factory=model_factory,
        )

    with pytest.raises(RuntimeError, match="stop after labels"):
        scorer.combine_groups(
            files,
            group_dirs,
            tmp_path / "combined",
            "deadbee",
            feature_loader=feature_loader,
            label_loader=lambda _path, _videos: (_ for _ in ()).throw(
                RuntimeError("stop after labels")
            ),
        )

    payload = json.loads(
        (tmp_path / "combined" / scorer.COMBINED_RESULT_FILE).read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "failed"
    assert payload["labels_read"] is True


def test_final_model_uses_all_videos_and_reloads_with_the_same_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files, features = _temporary_inputs(tmp_path)
    monkeypatch.setattr(scorer, "EXPECTED_SCORE_ROW_COUNT", 160)
    monkeypatch.setattr(fitter, "EXPECTED_SCORE_ROW_COUNT", 160)
    feature_loader = lambda *_args: features
    label_loader = lambda _path, videos: _fake_labels(tuple(videos))
    model_factory = lambda _run, _seed: _FixedModel()
    group_dirs = {name: tmp_path / f"group_{name}" for name in scorer.GROUP_NAMES}
    for name, output_dir in group_dirs.items():
        scorer.score_group(
            files,
            name,
            output_dir,
            "deadbee",
            feature_loader=feature_loader,
            label_loader=label_loader,
            model_factory=model_factory,
        )
    combined_dir = tmp_path / "combined"
    combined_result = scorer.combine_groups(
        files,
        group_dirs,
        combined_dir,
        "deadbee",
        feature_loader=feature_loader,
        label_loader=label_loader,
    )

    result_path = fitter.fit_final_model(
        fitter.FinalFitFiles(
            files,
            combined_result,
            combined_dir / scorer.COMBINED_RAW_SCORE_FILE,
            combined_dir / scorer.FINAL_SCORE_FILE,
        ),
        tmp_path / "final_model",
        "deadbee",
        "cafebabe",
        feature_loader=feature_loader,
        label_loader=label_loader,
        model_factory=model_factory,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["status"] == "complete"
    assert result["labels_read"] is True
    assert result["score_source_commit"] == "deadbee"
    assert result["fit_source_commit"] == "cafebabe"
    assert len(result["training_selection"]["videos"]) == 40
    assert len(result["model_check_rows"]) == 80
    assert len(result["feature_file_sha256_by_video"]) == 40
    for index in range(0, 80, 2):
        assert result["model_check_rows"][index]["frame"] == 10
        assert result["model_check_rows"][index + 1]["frame"] == 100
    assert (tmp_path / "final_model" / fitter.MODEL_FILE).is_file()
