"""Tests for manual broadcast-timeline state and guide loading."""

from pathlib import Path

import cv2
import pytest

from annotator.broadcast_timeline_labels import SceneTruth, VideoMetadata, make_interval
from annotator.manual_broadcast_timeline_annotator import (
    GuideInterval,
    TimelineSession,
    build_parser,
    commit_label,
    read_guides,
    read_scene_partition,
    video_metadata,
)


METADATA = VideoMetadata("sset_01", 25.0, 10)


def _scenes() -> list[GuideInterval]:
    return [
        GuideInterval(0, 3, "scene 0"),
        GuideInterval(3, 7, "scene 1"),
        GuideInterval(7, 10, "scene 2"),
    ]


def test_session_commits_unlabelled_gaps_in_order() -> None:
    session = TimelineSession(METADATA, [])

    first = session.commit_through(2, SceneTruth.LIVE)
    second = session.commit_through(5, SceneTruth.REPLAY, "repeat")

    assert (first.start_frame, first.end_frame) == (0, 3)
    assert (second.start_frame, second.end_frame) == (3, 6)
    assert session.first_gap() == 6
    assert [interval.truth for interval in session.intervals] == [SceneTruth.LIVE, SceneTruth.REPLAY]


def test_scene_mode_commits_exact_scenes_and_advances_to_the_next_midpoint() -> None:
    session = TimelineSession(METADATA, [])

    first, first_target = commit_label(session, 1, SceneTruth.LIVE, _scenes())
    assert (first.start_frame, first.end_frame, first_target) == (0, 3, 4)

    second, second_target = commit_label(session, 4, SceneTruth.REPLAY, _scenes())
    assert (second.start_frame, second.end_frame, second_target) == (3, 7, 8)

    third, third_target = commit_label(session, 8, SceneTruth.CUTAWAY, _scenes())
    assert (third.start_frame, third.end_frame, third_target) == (7, 10, None)
    session.validate_complete()


def test_scene_mode_keeps_explicit_selection_and_existing_interval_semantics() -> None:
    session = TimelineSession(METADATA, [])
    session.set_selection_start(0)

    partial, next_target = commit_label(session, 1, SceneTruth.LIVE, _scenes())
    changed, relabel_target = commit_label(session, 0, SceneTruth.OTHER, _scenes())

    assert partial == make_interval(METADATA, 0, 2, SceneTruth.LIVE)
    assert next_target is None
    assert changed == make_interval(METADATA, 0, 2, SceneTruth.OTHER)
    assert relabel_target is None


def test_number_key_semantics_relabel_an_existing_interval() -> None:
    interval = make_interval(METADATA, 0, 10, SceneTruth.LIVE, "keep this note")
    session = TimelineSession(METADATA, [interval])

    changed = session.commit_through(5, SceneTruth.LIVE_NON_STANDARD)

    assert changed == make_interval(METADATA, 0, 10, SceneTruth.LIVE_NON_STANDARD, "keep this note")
    session.validate_complete()


def test_explicit_selection_refuses_overlap() -> None:
    interval = make_interval(METADATA, 3, 6, SceneTruth.REPLAY)
    session = TimelineSession(METADATA, [interval])
    session.set_selection_start(1)

    with pytest.raises(ValueError, match="overlaps existing"):
        session.commit_through(4, SceneTruth.LIVE)


def test_delete_sets_up_exact_interval_replacement() -> None:
    intervals = [
        make_interval(METADATA, 0, 3, SceneTruth.LIVE),
        make_interval(METADATA, 3, 6, SceneTruth.REPLAY),
        make_interval(METADATA, 6, 10, SceneTruth.LIVE),
    ]
    session = TimelineSession(METADATA, intervals)

    removed = session.delete_at(4)
    replacement = session.commit_through(5, SceneTruth.CUTAWAY)

    assert removed.truth is SceneTruth.REPLAY
    assert replacement == make_interval(METADATA, 3, 6, SceneTruth.CUTAWAY)
    session.validate_complete()


def test_note_edit_and_first_gap_respect_partial_coverage() -> None:
    intervals = [make_interval(METADATA, 2, 5, SceneTruth.OTHER)]
    session = TimelineSession(METADATA, intervals, covered_start=2, covered_end=8)

    updated = session.set_note_at(3, "score graphic")

    assert updated.note == "score graphic"
    assert session.first_gap() == 5
    with pytest.raises(ValueError, match="partition ends"):
        session.validate_complete()


def test_session_rejects_an_interval_crossing_covered_boundary() -> None:
    interval = make_interval(METADATA, 0, 4, SceneTruth.LIVE)

    with pytest.raises(ValueError, match="crosses the covered range"):
        TimelineSession(METADATA, [interval], covered_start=2, covered_end=8)


def test_read_guides_converts_inclusive_gt_end(tmp_path: Path) -> None:
    path = tmp_path / "gt.csv"
    path.write_text("first,last\n2,4\n", encoding="utf-8")

    guides = read_guides(
        path,
        frame_count=10,
        start_column="first",
        end_column="last",
        label_column=None,
        end_inclusive=True,
    )

    assert [(guide.start_frame, guide.end_frame, guide.label) for guide in guides] == [(2, 5, "guide")]


def test_read_guides_requires_named_columns_and_valid_bounds(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    missing.write_text("start\n0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing guide columns"):
        read_guides(
            missing,
            frame_count=10,
            start_column="start",
            end_column="end",
            label_column=None,
        )

    outside = tmp_path / "outside.csv"
    outside.write_text("start,end,label\n0,11,replay\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        read_guides(
            outside,
            frame_count=10,
            start_column="start",
            end_column="end",
            label_column="label",
        )


def test_read_scene_partition_accepts_release_schema_and_clips_review_range(tmp_path: Path) -> None:
    path = tmp_path / "raw_cuts.csv"
    path.write_text(
        "scene_index,start_frame,end_frame\n0,0,3\n1,3,7\n2,7,10\n",
        encoding="utf-8",
    )

    scenes = read_scene_partition(
        path,
        frame_count=10,
        covered_start=2,
        covered_end=8,
        start_column="start_frame",
        end_column="end_frame",
    )

    assert scenes == [
        GuideInterval(2, 3, "scene 0"),
        GuideInterval(3, 7, "scene 1"),
        GuideInterval(7, 8, "scene 2"),
    ]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("0,0,3\n1,4,10\n", "gap"),
        ("0,0,4\n1,3,10\n", "overlap"),
        ("0,0,3\n1,3,9\n", "ends at 9"),
    ],
)
def test_read_scene_partition_rejects_incomplete_or_overlapping_scenes(
    tmp_path: Path,
    rows: str,
    message: str,
) -> None:
    path = tmp_path / "bad_scenes.csv"
    path.write_text("scene_index,start_frame,end_frame\n" + rows, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_scene_partition(
            path,
            frame_count=10,
            covered_start=0,
            covered_end=10,
            start_column="start_frame",
            end_column="end_frame",
        )


def test_parser_pins_required_source_identity_and_output() -> None:
    args = build_parser().parse_args([
        "--video",
        "video.mp4",
        "--video-id",
        "sset_01",
        "--out-csv",
        "labels.csv",
    ])

    assert args.video == Path("video.mp4")
    assert args.video_id == "sset_01"
    assert args.out_csv == Path("labels.csv")
    assert args.scene_csv is None
    assert args.start_frame == 0
    assert args.end_frame is None


def test_parser_accepts_scene_partition_columns() -> None:
    args = build_parser().parse_args([
        "--video",
        "video.mp4",
        "--video-id",
        "sset_01",
        "--out-csv",
        "labels.csv",
        "--scene-csv",
        "raw_cuts.csv",
        "--scene-start-col",
        "first",
        "--scene-end-col",
        "last_exclusive",
    ])

    assert args.scene_csv == Path("raw_cuts.csv")
    assert args.scene_start_col == "first"
    assert args.scene_end_col == "last_exclusive"


class _FakeCapture:
    def __init__(self, *, opened: bool = True, fps: float = 25.0, frame_count: float = 10.0) -> None:
        self.opened = opened
        self.values = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
        }

    def isOpened(self) -> bool:
        return self.opened

    def get(self, key: int) -> float:
        return self.values[key]


def test_video_metadata_uses_capture_values() -> None:
    metadata = video_metadata(_FakeCapture(), "sset_01")  # type: ignore[arg-type]

    assert metadata == METADATA


@pytest.mark.parametrize(
    "capture",
    [
        _FakeCapture(opened=False),
        _FakeCapture(fps=0.0),
        _FakeCapture(frame_count=0.0),
        _FakeCapture(frame_count=10.5),
    ],
)
def test_video_metadata_rejects_unusable_source_values(capture: _FakeCapture) -> None:
    with pytest.raises(ValueError):
        video_metadata(capture, "sset_01")  # type: ignore[arg-type]
