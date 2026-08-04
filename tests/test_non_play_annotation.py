"""Tests for non-play timeline state and guide loading."""

from pathlib import Path

import cv2
import pytest

from annotator.non_play_annotation import TimelineSession, build_parser, read_guides, video_metadata
from annotator.non_play_labels import SceneTruth, VideoMetadata, make_interval


METADATA = VideoMetadata("sset_01", 25.0, 10)


def test_session_commits_unlabelled_gaps_in_order() -> None:
    session = TimelineSession(METADATA, [])

    first = session.commit_through(2, SceneTruth.LIVE)
    second = session.commit_through(5, SceneTruth.REPLAY, "repeat")

    assert (first.start_frame, first.end_frame) == (0, 3)
    assert (second.start_frame, second.end_frame) == (3, 6)
    assert session.first_gap() == 6
    assert [interval.truth for interval in session.intervals] == [SceneTruth.LIVE, SceneTruth.REPLAY]


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
    assert args.start_frame == 0
    assert args.end_frame is None


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
