"""Review rally-start event rows without exposing a timeline write path."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path

import cv2
import numpy as np

from annotator.broadcast_timeline_labels import (
    LabelInterval,
    SceneTruth,
    VideoMetadata,
    interval_index_at,
    read_label_csv,
    validate_partition,
)
from annotator.rally_start_events import (
    RallyStartAuditSession,
    RallyStartDecision,
    RallyStartTarget,
    ReviewStatus,
    ServeVisibility,
    UndoResult,
    initialize_decision_csv,
    read_decision_csv,
    read_target_csv,
    validate_decision,
    validate_decision_output_path,
    validate_targets,
)


WINDOW_NAME = "Rally-start event review"
MAX_STATUS_CHARS = 58
TRUTH_COLOURS = {
    SceneTruth.LIVE: (30, 150, 30),
    SceneTruth.LIVE_NON_STANDARD: (80, 180, 80),
    SceneTruth.REPLAY: (180, 80, 180),
    SceneTruth.CUTAWAY: (180, 120, 40),
    SceneTruth.OTHER: (100, 100, 100),
}


class AuditAction(StrEnum):
    """Semantic action produced by one keyboard event."""

    QUIT = "quit"
    CLEAR_DRAFT = "clear-draft"
    PREVIOUS_FRAME = "previous-frame"
    NEXT_FRAME = "next-frame"
    JUMP_BACK = "jump-back"
    JUMP_FORWARD = "jump-forward"
    VISIBLE = "visible"
    BROADCAST_OMITTED = "broadcast-omitted"
    OFF_FRAME = "off-frame"
    UNCERTAIN = "uncertain"
    CAPTURE_CONTACT = "capture-contact"
    CAPTURE_RETURN = "capture-return"
    CAPTURE_FIRST_VISIBLE = "capture-first-visible"
    NOTE = "note"
    SAVE = "save"
    UNDO = "undo"
    PREVIOUS_ROW = "previous-row"
    NEXT_ROW = "next-row"
    VALIDATE = "validate"
    HELP = "help"


KEY_ACTIONS = {
    ord("q"): AuditAction.QUIT,
    ord("Q"): AuditAction.QUIT,
    27: AuditAction.CLEAR_DRAFT,
    ord(","): AuditAction.PREVIOUS_FRAME,
    ord("."): AuditAction.NEXT_FRAME,
    ord("<"): AuditAction.JUMP_BACK,
    ord(">"): AuditAction.JUMP_FORWARD,
    ord("1"): AuditAction.VISIBLE,
    ord("2"): AuditAction.BROADCAST_OMITTED,
    ord("3"): AuditAction.OFF_FRAME,
    ord("4"): AuditAction.UNCERTAIN,
    ord("c"): AuditAction.CAPTURE_CONTACT,
    ord("C"): AuditAction.CAPTURE_CONTACT,
    ord("r"): AuditAction.CAPTURE_RETURN,
    ord("R"): AuditAction.CAPTURE_RETURN,
    ord("f"): AuditAction.CAPTURE_FIRST_VISIBLE,
    ord("F"): AuditAction.CAPTURE_FIRST_VISIBLE,
    ord("n"): AuditAction.NOTE,
    ord("N"): AuditAction.NOTE,
    10: AuditAction.SAVE,
    13: AuditAction.SAVE,
    ord("u"): AuditAction.UNDO,
    ord("U"): AuditAction.UNDO,
    ord("["): AuditAction.PREVIOUS_ROW,
    ord("]"): AuditAction.NEXT_ROW,
    ord("v"): AuditAction.VALIDATE,
    ord("V"): AuditAction.VALIDATE,
    ord("h"): AuditAction.HELP,
    ord("H"): AuditAction.HELP,
}


@dataclass(frozen=True)
class PreparedAudit:
    """Validated one-video inputs ready for validation or GUI review."""

    targets: tuple[RallyStartTarget, ...]
    decisions: tuple[RallyStartDecision, ...]
    timeline: tuple[LabelInterval, ...]
    metadata: VideoMetadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="source review video")
    parser.add_argument("--video-id", required=True, help="canonical source identifier")
    parser.add_argument(
        "--timeline-csv",
        type=Path,
        required=True,
        help="immutable canonical broadcast timeline",
    )
    parser.add_argument(
        "--targets-csv",
        type=Path,
        required=True,
        help="immutable full rally-start target table",
    )
    parser.add_argument(
        "--seed-csv",
        type=Path,
        required=True,
        help="immutable compact full-audit decision seed",
    )
    parser.add_argument(
        "--decisions-csv",
        type=Path,
        required=True,
        help="local compact decision output ending in .csv.gz",
    )
    parser.add_argument("--jump-frames", type=int, default=25, help="coarse seek distance")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="require a complete existing decision table and open no GUI",
    )
    return parser


def action_for_key(key: int) -> AuditAction | None:
    """Map an OpenCV key code to one semantic review action."""
    return KEY_ACTIONS.get(key)


def validate_input_output_paths(args: argparse.Namespace) -> None:
    """Keep the compact output distinct from every immutable input."""
    decision_path = validate_decision_output_path(Path(args.decisions_csv)).resolve()
    inputs = {
        "video": Path(args.video),
        "timeline": Path(args.timeline_csv),
        "targets": Path(args.targets_csv),
        "seed": Path(args.seed_csv),
    }
    for label, path in inputs.items():
        if decision_path == path.resolve():
            raise ValueError(f"decision output aliases the {label} input: {path}")


def video_metadata(capture: cv2.VideoCapture, video_id: str) -> VideoMetadata:
    """Read validated metadata from one open review video."""
    if not capture.isOpened():
        raise ValueError("could not open source video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(frame_count_raw) or frame_count_raw <= 0:
        raise ValueError(f"video frame count is invalid: {frame_count_raw!r}")
    frame_count = int(round(frame_count_raw))
    if not math.isclose(frame_count_raw, frame_count, abs_tol=1e-6):
        raise ValueError(f"video frame count is not integral: {frame_count_raw!r}")
    return VideoMetadata(video_id, fps, frame_count)


def _require_matching_metadata(
    actual: VideoMetadata,
    expected: VideoMetadata,
    source: str,
) -> None:
    if actual.video_id != expected.video_id:
        raise ValueError(
            f"{source} video_id {actual.video_id!r} != target {expected.video_id!r}"
        )
    if actual.frame_count != expected.frame_count:
        raise ValueError(
            f"{source} frame_count {actual.frame_count} != target {expected.frame_count}"
        )
    if not math.isclose(actual.fps, expected.fps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{source} fps {actual.fps!r} != target {expected.fps!r}")


def _load_inputs(args: argparse.Namespace) -> tuple[
    list[RallyStartTarget],
    list[RallyStartDecision],
    list[LabelInterval],
    VideoMetadata,
]:
    targets = read_target_csv(Path(args.targets_csv))
    target_metadata = validate_targets(targets, str(args.targets_csv))
    if target_metadata.video_id != args.video_id:
        raise ValueError(
            f"--video-id {args.video_id!r} != target {target_metadata.video_id!r}"
        )
    seed = read_decision_csv(Path(args.seed_csv), targets)
    timeline = read_label_csv(Path(args.timeline_csv))
    timeline_metadata = validate_partition(timeline)
    _require_matching_metadata(timeline_metadata, target_metadata, "timeline")
    return targets, seed, timeline, target_metadata


def _load_or_initialize_decisions(
    args: argparse.Namespace,
    targets: Sequence[RallyStartTarget],
    seed: Sequence[RallyStartDecision],
) -> list[RallyStartDecision]:
    decision_path = Path(args.decisions_csv)
    if decision_path.exists():
        return read_decision_csv(decision_path, targets)
    if args.validate_only:
        raise FileNotFoundError(
            f"validation-only requires an existing decision table: {decision_path}"
        )
    initialize_decision_csv(decision_path, seed, targets)
    return read_decision_csv(decision_path, targets)


def prepare_audit(
    args: argparse.Namespace,
    capture: cv2.VideoCapture,
) -> PreparedAudit:
    """Validate inputs and safely initialize only a missing normal-mode output."""
    if args.jump_frames <= 0:
        raise ValueError(f"jump_frames must be positive, got {args.jump_frames}")
    validate_input_output_paths(args)
    targets, seed, timeline, expected_metadata = _load_inputs(args)
    actual_metadata = video_metadata(capture, args.video_id)
    _require_matching_metadata(actual_metadata, expected_metadata, "video")
    decisions = _load_or_initialize_decisions(args, targets, seed)
    return PreparedAudit(
        targets=tuple(targets),
        decisions=tuple(decisions),
        timeline=tuple(timeline),
        metadata=expected_metadata,
    )


def validate_complete_decisions(prepared: PreparedAudit) -> None:
    """Require every proposal to contain a valid reviewed decision."""
    pending = 0
    for row_number, (decision, target) in enumerate(
        zip(prepared.decisions, prepared.targets),
        start=2,
    ):
        validate_decision(decision, target, source=f"decision row {row_number}")
        pending += decision.review_status is ReviewStatus.PENDING
    if pending:
        raise ValueError(f"decision table is incomplete: {pending} pending rows")


def _timeline_x(frame: int, width: int, start: int, end: int) -> int:
    fraction = (frame - start) / (end - start)
    return int(round(np.clip(fraction, 0.0, 1.0) * (width - 1)))


def _draw_marker(
    image: np.ndarray,
    frame: int | None,
    target: RallyStartTarget,
    colour: tuple[int, int, int],
    top: int,
) -> None:
    if frame is None:
        return
    x = _timeline_x(
        frame,
        image.shape[1],
        target.review_start_frame,
        target.review_end_frame,
    )
    cv2.line(image, (x, top), (x, image.shape[0] - 1), colour, 2)


def draw_event_timeline(
    image: np.ndarray,
    intervals: Sequence[LabelInterval],
    target: RallyStartTarget,
    decision: RallyStartDecision,
    frame: int,
) -> None:
    """Draw immutable scene context and event markers inside one review window."""
    height, width = image.shape[:2]
    top = max(0, height - 22)
    image[top:height, :] = 0
    for interval in intervals:
        if interval.end_frame <= target.review_start_frame:
            continue
        if interval.start_frame >= target.review_end_frame:
            break
        left = _timeline_x(
            max(interval.start_frame, target.review_start_frame),
            width,
            target.review_start_frame,
            target.review_end_frame,
        )
        right = _timeline_x(
            min(interval.end_frame, target.review_end_frame),
            width,
            target.review_start_frame,
            target.review_end_frame,
        )
        cv2.rectangle(
            image,
            (left, top + 8),
            (max(left, right - 1), height - 1),
            TRUTH_COLOURS[interval.truth],
            -1,
        )
    _draw_marker(image, target.live_transition_frame, target, (255, 255, 0), top)
    _draw_marker(image, target.gt_first_frame, target, (255, 255, 255), top)
    _draw_marker(image, decision.visible_serve_frame, target, (0, 255, 0), top)
    _draw_marker(image, decision.broadcast_return_frame, target, (255, 0, 255), top)
    _draw_marker(image, decision.first_visible_rally_frame, target, (0, 165, 255), top)
    _draw_marker(image, frame, target, (0, 255, 255), top)


def _marker_text(value: int | None) -> str:
    return "-" if value is None else str(value)


def status_lines(
    session: RallyStartAuditSession,
    frame: int,
    truth: str,
    message: str,
) -> tuple[str, str, str, str]:
    """Build short status lines that keep row identity visible at 512 pixels."""
    target = session.current_target
    draft = session.draft
    visibility = "pending" if draft.serve_visibility is None else draft.serve_visibility.value
    confidence = "-" if draft.confidence is None else draft.confidence.value
    key = target.key
    lines = (
        f"{session.row_index + 1}/{len(session.targets)} {key.video_id} {key.set_id} R{key.rally} "
        f"win {target.review_start_frame}-{target.review_end_frame - 1}",
        f"frame {frame} GT {target.gt_first_frame} live {target.live_transition_frame} {truth}",
        f"{visibility} c:{_marker_text(draft.visible_serve_frame)} "
        f"ret:{_marker_text(draft.broadcast_return_frame)} "
        f"first:{_marker_text(draft.first_visible_rally_frame)}",
        f"{confidence} note:{'yes' if draft.review_note.strip() else 'no'} | {message}",
    )
    return (
        lines[0][:MAX_STATUS_CHARS],
        lines[1][:MAX_STATUS_CHARS],
        lines[2][:MAX_STATUS_CHARS],
        lines[3][:MAX_STATUS_CHARS],
    )


def draw_status(image: np.ndarray, lines: Sequence[str]) -> None:
    for row, line in enumerate(lines):
        y = 18 + row * 19
        cv2.putText(
            image,
            line,
            (7, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (7, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _read_frame(capture: cv2.VideoCapture, frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    if not ok or image is None:
        raise RuntimeError(f"could not decode frame {frame}")
    return image


def _print_help() -> None:
    print(
        "Controls:\n"
        "  ,/. one frame       </> coarse jump       [/] previous/next row\n"
        "  1 visible           2 broadcast-omitted   3 off-frame   4 uncertain\n"
        "  c contact frame     r broadcast return    f first visible rally\n"
        "  n review note       Enter save            u one-step undo\n"
        "  Esc clear draft     v validate draft      h help        q quit"
    )


class _AuditGui:
    def __init__(
        self,
        capture: cv2.VideoCapture,
        prepared: PreparedAudit,
        decision_path: Path,
        jump_frames: int,
    ) -> None:
        self.capture = capture
        self.timeline = prepared.timeline
        self.decision_path = Path(decision_path)
        self.jump_frames = jump_frames
        self.session = RallyStartAuditSession(
            prepared.targets,
            prepared.decisions,
            self.decision_path,
        )
        self.frame = self.session.current_target.review_start_frame
        self.message = "press h for controls"
        self._setting_trackbar = False

    def _clamp_frame(self, frame: int) -> int:
        target = self.session.current_target
        return min(max(frame, target.review_start_frame), target.review_end_frame - 1)

    def _truth_at_frame(self) -> str:
        index = interval_index_at(self.timeline, self.frame)
        return "none" if index is None else self.timeline[index].truth.value

    def redraw(self) -> None:
        image = _read_frame(self.capture, self.frame)
        draw_event_timeline(
            image,
            self.timeline,
            self.session.current_target,
            self.session.draft,
            self.frame,
        )
        draw_status(
            image,
            status_lines(
                self.session,
                self.frame,
                self._truth_at_frame(),
                self.message,
            ),
        )
        cv2.imshow(WINDOW_NAME, image)

    def set_frame(self, frame: int) -> None:
        self.frame = self._clamp_frame(frame)
        self._setting_trackbar = True
        try:
            cv2.setTrackbarPos("frame", WINDOW_NAME, self.frame)
        finally:
            self._setting_trackbar = False

    def on_trackbar(self, frame: int) -> None:
        if self._setting_trackbar:
            return
        clamped = self._clamp_frame(frame)
        if clamped != frame:
            self._setting_trackbar = True
            try:
                cv2.setTrackbarPos("frame", WINDOW_NAME, clamped)
            finally:
                self._setting_trackbar = False
        self.frame = clamped
        self.redraw()

    def _move_row(self, offset: int) -> None:
        self.session.move_row(offset)
        self.message = f"row {self.session.row_index + 1}"
        self.set_frame(self.session.current_target.review_start_frame)

    def _select_visibility(self, visibility: ServeVisibility) -> None:
        self.session.select_visibility(visibility)
        self.message = f"selected {visibility.value}"

    def _handle_state_or_marker(self, action: AuditAction) -> bool:
        if action is AuditAction.VISIBLE:
            self._select_visibility(ServeVisibility.VISIBLE)
        elif action is AuditAction.BROADCAST_OMITTED:
            self._select_visibility(ServeVisibility.BROADCAST_OMITTED)
        elif action is AuditAction.OFF_FRAME:
            self._select_visibility(ServeVisibility.OFF_FRAME)
        elif action is AuditAction.UNCERTAIN:
            self._select_visibility(ServeVisibility.UNCERTAIN)
        elif action is AuditAction.CAPTURE_CONTACT:
            self.session.capture_visible_serve(self.frame)
            self.message = f"contact {self.frame}"
        elif action is AuditAction.CAPTURE_RETURN:
            self.session.capture_broadcast_return(self.frame)
            self.message = f"return {self.frame}"
        elif action is AuditAction.CAPTURE_FIRST_VISIBLE:
            self.session.capture_first_visible_rally(self.frame)
            self.message = f"first visible {self.frame}"
        else:
            return False
        return True

    def handle(self, action: AuditAction) -> bool:
        """Apply one action and return true when the GUI should quit."""
        if action is AuditAction.QUIT:
            return True
        if self._handle_state_or_marker(action):
            return False
        if action is AuditAction.CLEAR_DRAFT:
            self.message = "draft cleared" if self.session.clear_draft() else "draft unchanged"
        elif action is AuditAction.PREVIOUS_FRAME:
            self.set_frame(self.frame - 1)
        elif action is AuditAction.NEXT_FRAME:
            self.set_frame(self.frame + 1)
        elif action is AuditAction.JUMP_BACK:
            self.set_frame(self.frame - self.jump_frames)
        elif action is AuditAction.JUMP_FORWARD:
            self.set_frame(self.frame + self.jump_frames)
        elif action is AuditAction.PREVIOUS_ROW:
            self._move_row(-1)
        elif action is AuditAction.NEXT_ROW:
            self._move_row(1)
        elif action is AuditAction.NOTE:
            self.session.set_note(input("review note (empty clears): ").strip())
            self.message = "note updated"
        elif action is AuditAction.SAVE:
            saved = self.session.save(self.decision_path)
            self.message = f"saved {saved.key.set_id} R{saved.key.rally}"
            self.set_frame(self.session.current_target.review_start_frame)
        elif action is AuditAction.UNDO:
            result = self.session.undo(self.decision_path)
            self.message = result.value
            if result is UndoResult.SAVED_ROW_RESTORED:
                self.set_frame(self.session.current_target.review_start_frame)
        elif action is AuditAction.VALIDATE:
            self.session.validate_draft()
            self.message = "draft valid"
        elif action is AuditAction.HELP:
            _print_help()
            self.message = "controls printed in terminal"
        return False

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            "frame",
            WINDOW_NAME,
            self.frame,
            max(self.session.current_target.metadata.frame_count - 1, 1),
            self.on_trackbar,
        )
        _print_help()
        self.redraw()
        try:
            while True:
                action = action_for_key(cv2.waitKeyEx(0))
                if action is None:
                    continue
                try:
                    if self.handle(action):
                        break
                except (EOFError, OSError, RuntimeError, ValueError) as exc:
                    self.message = f"error: {exc}"
                    print(self.message)
                self.redraw()
        finally:
            cv2.destroyWindow(WINDOW_NAME)


def _run_gui(
    capture: cv2.VideoCapture,
    prepared: PreparedAudit,
    decision_path: Path,
    jump_frames: int,
) -> None:
    _AuditGui(capture, prepared, decision_path, jump_frames).run()


def run_annotation_tool(args: argparse.Namespace) -> int:
    capture = cv2.VideoCapture(str(args.video))
    try:
        prepared = prepare_audit(args, capture)
        if args.validate_only:
            validate_complete_decisions(prepared)
            print(
                f"valid complete decisions: {len(prepared.decisions)} reviewed, "
                f"0 pending for {prepared.metadata.video_id}"
            )
            return 0
        reviewed = sum(
            decision.review_status is ReviewStatus.REVIEWED
            for decision in prepared.decisions
        )
        print(
            f"loaded {len(prepared.decisions)} decisions for {prepared.metadata.video_id}: "
            f"{reviewed} reviewed, {len(prepared.decisions) - reviewed} pending"
        )
        _run_gui(capture, prepared, Path(args.decisions_csv), args.jump_frames)
        return 0
    finally:
        capture.release()


def main() -> int:
    return run_annotation_tool(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
