"""Load and verify the fixed ShuttleSet development split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

DEVELOPMENT_SPLIT_SCHEMA = "contact-full-dataset-development-split/1"
FIXTURE_NAME = re.compile(r"^sset_[0-9]{2}$")
ACCEPTED_EXCLUDED_VIDEO_IDS = (9, 10, 12, 27)
ACCEPTED_VALIDATION_VIDEO_IDS = frozenset({18, 22, 24, 25, 30, 31, 39, 40})
ACCEPTED_VIDEO_IDS = frozenset(range(1, 45)) - frozenset(ACCEPTED_EXCLUDED_VIDEO_IDS)


class SplitRole(StrEnum):
    """A video's role while model settings are being chosen."""

    TRAIN = "train"
    VALIDATION = "validation"


@dataclass(frozen=True)
class VideoSpec:
    """Portable identity and capture metadata for one ShuttleSet video."""

    fixture: str
    video_id: int
    fps: float
    width: int
    height: int
    role: SplitRole
    winner: str
    loser: str
    tournament: str
    tournament_round: str


@dataclass(frozen=True)
class DevelopmentSplit:
    """The fixed train and validation videos used to choose the model design."""

    dataset: str
    excluded_video_ids: tuple[int, ...]
    expected_train_count: int
    expected_validation_count: int
    videos: tuple[VideoSpec, ...]

    @property
    def training_videos(self) -> tuple[VideoSpec, ...]:
        return tuple(video for video in self.videos if video.role is SplitRole.TRAIN)

    @property
    def validation_videos(self) -> tuple[VideoSpec, ...]:
        return tuple(video for video in self.videos if video.role is SplitRole.VALIDATION)

    @property
    def by_fixture(self) -> dict[str, VideoSpec]:
        return {video.fixture: video for video in self.videos}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _video_spec(raw: object, index: int) -> VideoSpec:
    row = _mapping(raw, f"videos[{index}]")
    expected_fields = {
        "fixture",
        "video_id",
        "fps",
        "width",
        "height",
        "role",
        "winner",
        "loser",
        "tournament",
        "round",
    }
    if set(row) != expected_fields:
        raise ValueError(f"videos[{index}] fields differ")

    video_id = _integer(row["video_id"], f"videos[{index}].video_id")
    fixture = _non_empty_string(row["fixture"], f"videos[{index}].fixture")
    if FIXTURE_NAME.fullmatch(fixture) is None or fixture != f"sset_{video_id:02d}":
        raise ValueError(f"videos[{index}] fixture does not match its video ID")

    fps_value = row["fps"]
    if isinstance(fps_value, bool) or not isinstance(fps_value, (int, float)):
        raise TypeError(f"videos[{index}].fps must be a number")
    fps = float(fps_value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"videos[{index}].fps must be positive and finite")

    width = _integer(row["width"], f"videos[{index}].width")
    height = _integer(row["height"], f"videos[{index}].height")
    if width <= 0 or height <= 0:
        raise ValueError(f"videos[{index}] resolution must be positive")

    try:
        role = SplitRole(row["role"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"videos[{index}].role differs") from error

    return VideoSpec(
        fixture=fixture,
        video_id=video_id,
        fps=fps,
        width=width,
        height=height,
        role=role,
        winner=_non_empty_string(row["winner"], f"videos[{index}].winner"),
        loser=_non_empty_string(row["loser"], f"videos[{index}].loser"),
        tournament=_non_empty_string(row["tournament"], f"videos[{index}].tournament"),
        tournament_round=_non_empty_string(row["round"], f"videos[{index}].round"),
    )


def load_development_split(path: Path) -> DevelopmentSplit:
    """Load a saved video split and check its required fields."""
    manifest_path = Path(path)
    payload = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "split file")
    expected_fields = {"schema", "dataset", "excluded_video_ids", "expected_counts", "videos"}
    if set(payload) != expected_fields:
        raise ValueError("split file fields differ")
    if payload["schema"] != DEVELOPMENT_SPLIT_SCHEMA:
        raise ValueError("split file version differs")

    dataset = _non_empty_string(payload["dataset"], "dataset")
    if dataset != "ShuttleSet":
        raise ValueError("dataset must be ShuttleSet")
    raw_excluded = payload["excluded_video_ids"]
    if not isinstance(raw_excluded, list):
        raise TypeError("excluded_video_ids must be a list")
    excluded_video_ids = tuple(_integer(value, "excluded video ID") for value in raw_excluded)
    if excluded_video_ids != tuple(sorted(set(excluded_video_ids))):
        raise ValueError("excluded_video_ids must be sorted and unique")

    expected_counts = _mapping(payload["expected_counts"], "expected_counts")
    if set(expected_counts) != {"train", "validation", "total"}:
        raise ValueError("expected_counts fields differ")
    expected_train_count = _integer(expected_counts["train"], "expected_counts.train")
    expected_validation_count = _integer(expected_counts["validation"], "expected_counts.validation")
    expected_total = _integer(expected_counts["total"], "expected_counts.total")
    if expected_train_count + expected_validation_count != expected_total:
        raise ValueError("expected split counts do not add to the total")

    raw_videos = payload["videos"]
    if not isinstance(raw_videos, list):
        raise TypeError("videos must be a list")
    videos = tuple(_video_spec(raw, index) for index, raw in enumerate(raw_videos))
    video_ids = tuple(video.video_id for video in videos)
    fixtures = tuple(video.fixture for video in videos)
    if video_ids != tuple(sorted(video_ids)):
        raise ValueError("videos must be sorted by video_id")
    if len(set(video_ids)) != len(videos) or len(set(fixtures)) != len(videos):
        raise ValueError("video IDs and fixture names must be unique")
    if set(video_ids) & set(excluded_video_ids):
        raise ValueError("an excluded video is present in the development split")

    training_count = sum(video.role is SplitRole.TRAIN for video in videos)
    validation_count = sum(video.role is SplitRole.VALIDATION for video in videos)
    if (training_count, validation_count, len(videos)) != (
        expected_train_count,
        expected_validation_count,
        expected_total,
    ):
        raise ValueError("actual split counts differ from expected_counts")

    return DevelopmentSplit(
        dataset=dataset,
        excluded_video_ids=excluded_video_ids,
        expected_train_count=expected_train_count,
        expected_validation_count=expected_validation_count,
        videos=videos,
    )


def _csv_rows_by_id(path: Path) -> dict[int, dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    by_id = {int(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"{Path(path).name} contains duplicate video IDs")
    return by_id


def verify_accepted_development_split(split: DevelopmentSplit) -> None:
    """Check that a valid file contains the accepted experiment split."""
    video_ids = {video.video_id for video in split.videos}
    validation_ids = {video.video_id for video in split.validation_videos}
    if split.excluded_video_ids != ACCEPTED_EXCLUDED_VIDEO_IDS:
        raise ValueError("excluded video IDs differ from the accepted split")
    if video_ids != ACCEPTED_VIDEO_IDS:
        raise ValueError("eligible video IDs differ from the accepted split")
    if validation_ids != ACCEPTED_VALIDATION_VIDEO_IDS:
        raise ValueError("validation video IDs differ from the accepted split")
    if split.expected_train_count != 32 or split.expected_validation_count != 8:
        raise ValueError("video counts differ from the accepted split")


def verify_against_shuttleset_tables(
    split: DevelopmentSplit,
    video_metadata_path: Path,
    match_path: Path,
) -> None:
    """Check the saved split against the ShuttleSet source tables."""
    metadata_by_id = _csv_rows_by_id(video_metadata_path)
    matches_by_id = _csv_rows_by_id(match_path)
    valid_metadata_ids = {
        video_id
        for video_id, row in metadata_by_id.items()
        if row["width"] and row["height"] and row["fps"]
    }
    split_ids = {video.video_id for video in split.videos}
    if split_ids != valid_metadata_ids:
        raise ValueError("split video IDs differ from the valid ShuttleSet metadata rows")

    for video in split.videos:
        metadata = metadata_by_id[video.video_id]
        match = matches_by_id[video.video_id]
        actual = (
            float(metadata["fps"]),
            int(metadata["width"]),
            int(metadata["height"]),
            match["winner"],
            match["loser"],
            match["tournament"],
            match["round"],
        )
        expected = (
            video.fps,
            video.width,
            video.height,
            video.winner,
            video.loser,
            video.tournament,
            video.tournament_round,
        )
        if actual != expected:
            raise ValueError(f"{video.fixture}: split details differ from the ShuttleSet tables")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--video-metadata", type=Path, required=True)
    parser.add_argument("--matches", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    split = load_development_split(arguments.manifest)
    verify_accepted_development_split(split)
    verify_against_shuttleset_tables(split, arguments.video_metadata, arguments.matches)
    print(
        f"verified {len(split.training_videos)} train and "
        f"{len(split.validation_videos)} validation videos"
    )


if __name__ == "__main__":
    main()
