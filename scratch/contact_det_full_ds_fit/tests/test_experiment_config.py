from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DEVELOPMENT_SPLIT_SCHEMA,
    SplitRole,
    load_development_split,
    verify_accepted_development_split,
    verify_against_shuttleset_tables,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = EXPERIMENT_ROOT / "records/shuttleset_development_split.json"
ANNOTATION_ROOT = REPO_ROOT / "training" / "data" / "shuttleset" / "annotations"
VALIDATION_IDS = {18, 22, 24, 25, 30, 31, 39, 40}
EXCLUDED_IDS = {9, 10, 12, 27}


def _video(video_id: int, role: str) -> dict[str, object]:
    return {
        "fixture": f"sset_{video_id:02d}",
        "video_id": video_id,
        "fps": 25.0,
        "width": 1920,
        "height": 1080,
        "role": role,
        "winner": f"Winner {video_id}",
        "loser": f"Loser {video_id}",
        "tournament": "Tournament",
        "round": "Final",
    }


def _payload(videos: list[dict[str, object]]) -> dict[str, object]:
    train_count = sum(video["role"] == "train" for video in videos)
    validation_count = sum(video["role"] == "validation" for video in videos)
    return {
        "schema": DEVELOPMENT_SPLIT_SCHEMA,
        "dataset": "ShuttleSet",
        "excluded_video_ids": [],
        "expected_counts": {
            "train": train_count,
            "validation": validation_count,
            "total": len(videos),
        },
        "videos": videos,
    }


def _write_manifest(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_csv_rows(tmp_path: Path, name: str, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def test_canonical_split_matches_the_accepted_video_roles() -> None:
    split = load_development_split(CANONICAL_MANIFEST)
    verify_accepted_development_split(split)

    assert split.dataset == "ShuttleSet"
    assert set(split.excluded_video_ids) == EXCLUDED_IDS
    assert len(split.training_videos) == 32
    assert len(split.validation_videos) == 8
    assert {video.video_id for video in split.validation_videos} == VALIDATION_IDS
    assert {video.video_id for video in split.training_videos} == (
        {video.video_id for video in split.videos} - VALIDATION_IDS
    )
    assert all(video.role is SplitRole.TRAIN for video in split.training_videos)
    assert all(video.role is SplitRole.VALIDATION for video in split.validation_videos)


def test_saved_split_matches_the_shuttleset_tables() -> None:
    split = load_development_split(CANONICAL_MANIFEST)

    verify_against_shuttleset_tables(
        split,
        ANNOTATION_ROOT / "video_metadata.csv",
        ANNOTATION_ROOT / "set" / "match.csv",
    )


def test_split_rejects_duplicate_video_identity(tmp_path: Path) -> None:
    payload = _payload([_video(1, "train"), _video(1, "validation")])

    with pytest.raises(ValueError, match="must be unique"):
        load_development_split(_write_manifest(tmp_path, payload))


def test_split_rejects_an_excluded_video_in_a_role(tmp_path: Path) -> None:
    payload = _payload([_video(1, "train"), _video(2, "validation")])
    payload["excluded_video_ids"] = [2]

    with pytest.raises(ValueError, match="excluded video"):
        load_development_split(_write_manifest(tmp_path, payload))


def test_split_rejects_counts_that_do_not_match_the_rows(tmp_path: Path) -> None:
    payload = _payload([_video(1, "train"), _video(2, "validation")])
    payload["expected_counts"] = {"train": 2, "validation": 0, "total": 2}

    with pytest.raises(ValueError, match="actual split counts"):
        load_development_split(_write_manifest(tmp_path, payload))


def test_split_rejects_rows_out_of_video_id_order(tmp_path: Path) -> None:
    payload = _payload([_video(2, "train"), _video(1, "validation")])

    with pytest.raises(ValueError, match="sorted by video_id"):
        load_development_split(_write_manifest(tmp_path, payload))


def test_split_rejects_a_different_dataset(tmp_path: Path) -> None:
    payload = _payload([_video(1, "train"), _video(2, "validation")])
    payload["dataset"] = "Another dataset"

    with pytest.raises(ValueError, match="dataset must be ShuttleSet"):
        load_development_split(_write_manifest(tmp_path, payload))


def test_accepted_split_rejects_a_changed_validation_video(tmp_path: Path) -> None:
    payload = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    for video in payload["videos"]:
        if video["video_id"] == 1:
            video["role"] = "validation"
        elif video["video_id"] == 18:
            video["role"] = "train"
    split = load_development_split(_write_manifest(tmp_path, payload))

    with pytest.raises(ValueError, match="validation video IDs"):
        verify_accepted_development_split(split)


def test_accepted_split_rejects_a_changed_exclusion(tmp_path: Path) -> None:
    payload = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    payload["excluded_video_ids"] = [9, 10, 12, 99]
    split = load_development_split(_write_manifest(tmp_path, payload))

    with pytest.raises(ValueError, match="excluded video IDs"):
        verify_accepted_development_split(split)


def test_source_check_rejects_changed_video_metadata(tmp_path: Path) -> None:
    split = load_development_split(CANONICAL_MANIFEST)
    metadata_rows = _read_csv_rows(ANNOTATION_ROOT / "video_metadata.csv")
    metadata_rows[0]["width"] = "1280"
    metadata_path = _write_csv_rows(tmp_path, "video_metadata.csv", metadata_rows)

    with pytest.raises(ValueError, match="sset_01: split details differ"):
        verify_against_shuttleset_tables(split, metadata_path, ANNOTATION_ROOT / "set" / "match.csv")


def test_source_check_rejects_changed_match_metadata(tmp_path: Path) -> None:
    split = load_development_split(CANONICAL_MANIFEST)
    match_rows = _read_csv_rows(ANNOTATION_ROOT / "set" / "match.csv")
    match_rows[0]["winner"] = "Different player"
    match_path = _write_csv_rows(tmp_path, "match.csv", match_rows)

    with pytest.raises(ValueError, match="sset_01: split details differ"):
        verify_against_shuttleset_tables(split, ANNOTATION_ROOT / "video_metadata.csv", match_path)


def test_source_check_rejects_a_missing_metadata_row(tmp_path: Path) -> None:
    split = load_development_split(CANONICAL_MANIFEST)
    metadata_rows = _read_csv_rows(ANNOTATION_ROOT / "video_metadata.csv")[1:]
    metadata_path = _write_csv_rows(tmp_path, "video_metadata.csv", metadata_rows)

    with pytest.raises(ValueError, match="split video IDs differ"):
        verify_against_shuttleset_tables(split, metadata_path, ANNOTATION_ROOT / "set" / "match.csv")


def test_manifest_contains_no_machine_or_access_fields() -> None:
    payload = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    forbidden = {"path", "url", "host", "hostname", "credential", "access"}

    for video in payload["videos"]:
        assert forbidden.isdisjoint(video)
