"""Tests for the read-only rally-start event review companion."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
import pytest

import annotator.rally_start_event_annotator as tool
import annotator.rally_start_events as events
from annotator.broadcast_timeline_labels import (
    SceneTruth,
    VideoMetadata,
    make_interval,
    write_label_csv,
)


METADATA = VideoMetadata("sset_01", 25.0, 100)


class FakeCapture:
    def __init__(self, *, fps: float = 25.0, frame_count: float = 100.0) -> None:
        self.fps = fps
        self.frame_count = frame_count
        self.released = False
        self.frame = 0

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == tool.cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == tool.cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        raise AssertionError(f"unexpected capture property {property_id}")

    def set(self, property_id: int, value: float) -> bool:
        assert property_id == tool.cv2.CAP_PROP_POS_FRAMES
        self.frame = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.zeros((288, 512, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _target() -> events.RallyStartTarget:
    return events.RallyStartTarget(
        key=events.RallyStartKey("sset_01", "set1", 1),
        metadata=METADATA,
        gt_first_frame=20,
        gt_first_type_en="short service",
        gt_first_flaw=False,
        timeline_truth="live",
        timeline_interval_start=10,
        timeline_interval_end=100,
        preceding_truth="replay",
        live_transition_frame=10,
        review_start_frame=0,
        review_end_frame=41,
        pilot_stratum="full-audit-only",
        note="set1 rally 1",
    )


def _target_row(target: events.RallyStartTarget) -> dict[str, str]:
    row = {column: "" for column in events.TARGET_COLUMNS}
    row.update({
        "video_id": target.key.video_id,
        "fps": str(target.metadata.fps),
        "frame_count": str(target.metadata.frame_count),
        "set_id": target.key.set_id,
        "rally": str(target.key.rally),
        "gt_first_frame": str(target.gt_first_frame),
        "gt_first_ball_round": "1",
        "gt_first_type_raw": "1",
        "gt_first_type_en": target.gt_first_type_en,
        "gt_first_flaw": "false",
        "gt_first_server": "1",
        "committed_status": events.EXPECTED_COMMITTED_STATUS,
        "later_strokes_matched": "2",
        "timeline_truth": target.timeline_truth,
        "timeline_interval_start": str(target.timeline_interval_start),
        "timeline_interval_end": str(target.timeline_interval_end),
        "preceding_truth": target.preceding_truth,
        "live_transition_frame": str(target.live_transition_frame),
        "frames_from_live_transition": str(
            target.gt_first_frame - target.live_transition_frame
        ),
        "review_start_frame": str(target.review_start_frame),
        "review_end_frame": str(target.review_end_frame),
        "pilot_stratum": target.pilot_stratum,
        "note": target.note,
        "review_status": "pending",
    })
    return row


def _reviewed(target: events.RallyStartTarget) -> events.RallyStartDecision:
    return events.RallyStartDecision(
        key=target.key,
        review_status=events.ReviewStatus.REVIEWED,
        serve_visibility=events.ServeVisibility.VISIBLE,
        visible_serve_frame=20,
        confidence=events.Confidence.CERTAIN,
        review_note="service contact is visible",
    )


def _args(tmp_path: Path, *, validate_only: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        video=tmp_path / "video.mp4",
        video_id="sset_01",
        timeline_csv=tmp_path / "timeline.csv.gz",
        targets_csv=tmp_path / "targets.csv.gz",
        seed_csv=tmp_path / "seed.csv.gz",
        decisions_csv=tmp_path / "sset_01_rally_start_decisions.csv.gz",
        jump_frames=25,
        validate_only=validate_only,
    )


def _write_inputs(tmp_path: Path) -> tuple[argparse.Namespace, events.RallyStartTarget]:
    args = _args(tmp_path)
    target = _target()
    with gzip.open(args.targets_csv, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=events.TARGET_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(_target_row(target))
    events.write_decision_csv(
        args.seed_csv,
        [events.RallyStartDecision.pending(target.key)],
        [target],
    )
    write_label_csv(
        args.timeline_csv,
        [
            make_interval(METADATA, 0, 10, SceneTruth.REPLAY),
            make_interval(METADATA, 10, 100, SceneTruth.LIVE),
        ],
        METADATA,
    )
    return args, target


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        (ord("1"), tool.AuditAction.VISIBLE),
        (ord("2"), tool.AuditAction.BROADCAST_OMITTED),
        (ord("3"), tool.AuditAction.OFF_FRAME),
        (ord("4"), tool.AuditAction.UNCERTAIN),
        (ord("c"), tool.AuditAction.CAPTURE_CONTACT),
        (ord("R"), tool.AuditAction.CAPTURE_RETURN),
        (ord("f"), tool.AuditAction.CAPTURE_FIRST_VISIBLE),
        (13, tool.AuditAction.SAVE),
        (ord("u"), tool.AuditAction.UNDO),
        (ord("["), tool.AuditAction.PREVIOUS_ROW),
        (ord("]"), tool.AuditAction.NEXT_ROW),
        (27, tool.AuditAction.CLEAR_DRAFT),
    ],
)
def test_keyboard_mapping_is_explicit(key: int, expected: tool.AuditAction) -> None:
    assert tool.action_for_key(key) is expected


def test_unknown_key_has_no_action() -> None:
    assert tool.action_for_key(-1) is None


@pytest.mark.parametrize("input_name", ("timeline_csv", "targets_csv", "seed_csv"))
def test_decision_output_cannot_alias_an_immutable_input(
    tmp_path: Path,
    input_name: str,
) -> None:
    args = _args(tmp_path)
    args.decisions_csv = getattr(args, input_name)

    with pytest.raises(ValueError, match="decision output aliases"):
        tool.validate_input_output_paths(args)


def test_decision_output_rejects_a_canonical_timeline_name(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.decisions_csv = tmp_path / "other_broadcast_timeline_labels.csv.gz"

    with pytest.raises(ValueError, match="protected canonical timeline name"):
        tool.validate_input_output_paths(args)


def test_normal_start_initializes_once_and_never_changes_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, target = _write_inputs(tmp_path)
    timeline_before = args.timeline_csv.read_bytes()
    capture = FakeCapture()
    gui_calls: list[tool.PreparedAudit] = []
    monkeypatch.setattr(tool.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        tool,
        "_run_gui",
        lambda _capture, prepared, _path, _jump: gui_calls.append(prepared),
    )

    assert tool.run_annotation_tool(args) == 0

    assert capture.released
    assert len(gui_calls) == 1
    pending = events.RallyStartDecision.pending(target.key)
    assert events.read_decision_csv(args.decisions_csv, [target]) == [pending]
    assert args.timeline_csv.read_bytes() == timeline_before

    decision_before = args.decisions_csv.read_bytes()
    capture.released = False
    assert tool.run_annotation_tool(args) == 0
    assert capture.released
    assert args.decisions_csv.read_bytes() == decision_before
    assert args.timeline_csv.read_bytes() == timeline_before


def test_validation_only_requires_complete_existing_decisions_and_opens_no_gui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, target = _write_inputs(tmp_path)
    args.validate_only = True
    events.write_decision_csv(args.decisions_csv, [_reviewed(target)], [target])
    decisions_before = args.decisions_csv.read_bytes()
    timeline_before = args.timeline_csv.read_bytes()
    capture = FakeCapture()
    monkeypatch.setattr(tool.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        tool,
        "_run_gui",
        lambda *_args: pytest.fail("validation-only opened the GUI"),
    )
    monkeypatch.setattr(
        tool,
        "initialize_decision_csv",
        lambda *_args: pytest.fail("validation-only initialized output"),
    )

    assert tool.run_annotation_tool(args) == 0

    assert capture.released
    assert args.decisions_csv.read_bytes() == decisions_before
    assert args.timeline_csv.read_bytes() == timeline_before


def test_validation_only_rejects_partial_or_missing_output_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, target = _write_inputs(tmp_path)
    args.validate_only = True
    capture = FakeCapture()
    monkeypatch.setattr(tool.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(
        tool,
        "_run_gui",
        lambda *_args: pytest.fail("validation-only opened the GUI"),
    )
    monkeypatch.setattr(
        tool,
        "initialize_decision_csv",
        lambda *_args: pytest.fail("validation-only initialized output"),
    )

    with pytest.raises(FileNotFoundError, match="requires an existing"):
        tool.run_annotation_tool(args)
    assert not args.decisions_csv.exists()

    events.write_decision_csv(
        args.decisions_csv,
        [events.RallyStartDecision.pending(target.key)],
        [target],
    )
    output_before = args.decisions_csv.read_bytes()
    with pytest.raises(ValueError, match="1 pending rows"):
        tool.run_annotation_tool(args)
    assert args.decisions_csv.read_bytes() == output_before


def test_gui_dispatch_save_undo_and_frame_clamp_never_change_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, target = _write_inputs(tmp_path)
    pending = events.RallyStartDecision.pending(target.key)
    events.initialize_decision_csv(args.decisions_csv, [pending], [target])
    timeline = tuple(
        [
            make_interval(METADATA, 0, 10, SceneTruth.REPLAY),
            make_interval(METADATA, 10, 100, SceneTruth.LIVE),
        ]
    )
    prepared = tool.PreparedAudit((target,), (pending,), timeline, METADATA)
    timeline_before = args.timeline_csv.read_bytes()
    trackbar_frames: list[int] = []
    monkeypatch.setattr(
        tool.cv2,
        "setTrackbarPos",
        lambda _name, _window, frame: trackbar_frames.append(frame),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "service contact is visible")
    gui = tool._AuditGui(FakeCapture(), prepared, args.decisions_csv, 25)

    gui.set_frame(-10)
    assert gui.frame == target.review_start_frame
    gui.set_frame(999)
    assert gui.frame == target.review_end_frame - 1
    gui.set_frame(20)
    assert not gui.handle(tool.AuditAction.VISIBLE)
    assert not gui.handle(tool.AuditAction.CAPTURE_CONTACT)
    assert not gui.handle(tool.AuditAction.NOTE)
    assert not gui.handle(tool.AuditAction.SAVE)

    reviewed = events.read_decision_csv(args.decisions_csv, [target])
    assert reviewed == [_reviewed(target)]
    assert args.timeline_csv.read_bytes() == timeline_before

    assert not gui.handle(tool.AuditAction.UNDO)
    assert events.read_decision_csv(args.decisions_csv, [target]) == [pending]
    assert args.timeline_csv.read_bytes() == timeline_before
    assert trackbar_frames == [0, 40, 20, 0, 0]


@pytest.mark.parametrize(
    ("capture", "message"),
    [
        (FakeCapture(fps=30.0), "video fps"),
        (FakeCapture(frame_count=99.0), "video frame_count"),
    ],
)
def test_video_metadata_must_match_targets_before_output_initialization(
    tmp_path: Path,
    capture: FakeCapture,
    message: str,
) -> None:
    args, _target_row_value = _write_inputs(tmp_path)

    with pytest.raises(ValueError, match=message):
        tool.prepare_audit(args, capture)  # type: ignore[arg-type]

    assert not args.decisions_csv.exists()


def test_status_lines_keep_identity_bounds_and_markers_visible() -> None:
    target = _target()
    pending = events.RallyStartDecision.pending(target.key)
    session = events.RallyStartAuditSession([target], [pending])
    session.select_visibility(events.ServeVisibility.VISIBLE)
    session.capture_visible_serve(20)
    session.set_note("contact visible")

    lines = tool.status_lines(session, 20, "live", "draft valid")

    assert all(len(line) <= tool.MAX_STATUS_CHARS for line in lines)
    assert "sset_01 set1 R1" in lines[0]
    assert "win 0-40" in lines[0]
    assert "frame 20 GT 20 live 10 live" in lines[1]
    assert "visible c:20 ret:- first:-" in lines[2]
    assert "certain note:yes" in lines[3]


def test_timeline_drawing_changes_only_the_display_array() -> None:
    target = _target()
    decision = _reviewed(target)
    intervals = (
        make_interval(METADATA, 0, 10, SceneTruth.REPLAY),
        make_interval(METADATA, 10, 100, SceneTruth.LIVE),
    )
    before_intervals = tuple(intervals)
    image = np.zeros((288, 512, 3), dtype=np.uint8)

    tool.draw_event_timeline(image, intervals, target, decision, frame=20)

    assert np.any(image[-22:, :])
    assert intervals == before_intervals
