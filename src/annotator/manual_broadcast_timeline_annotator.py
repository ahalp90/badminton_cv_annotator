"""OpenCV tool for manually labelling a broadcast scene timeline.

Run from the repository root::

    PYTHONPATH=src python -m annotator.manual_broadcast_timeline_annotator \
        --video path/to/sset_01_288p.mp4 --video-id sset_01 \
        --out-csv local_scratch/autograder_architecture/measurements/\
sset_01_broadcast_timeline_labels.csv

The current frame is inclusive when a number key commits a new interval. The
saved end is therefore ``current_frame + 1`` and remains half-open.

When ``--scene-csv`` is supplied, a number key on an unlabelled scene commits
that complete half-open scene and advances to the next unlabelled scene.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import dataclass
import gzip
import math
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np

from annotator.broadcast_timeline_labels import (
    LabelInterval,
    SceneTruth,
    VideoMetadata,
    interval_index_at,
    make_interval,
    read_label_csv,
    replace_interval,
    validate_intervals,
    validate_partition,
    write_label_csv,
)


WINDOW_NAME = "Manual broadcast timeline annotator"
LABEL_KEYS = {
    ord("1"): SceneTruth.LIVE,
    ord("2"): SceneTruth.LIVE_NON_STANDARD,
    ord("3"): SceneTruth.REPLAY,
    ord("4"): SceneTruth.CUTAWAY,
    ord("5"): SceneTruth.OTHER,
}
TRUTH_COLOURS = {
    SceneTruth.LIVE: (75, 180, 60),
    SceneTruth.LIVE_NON_STANDARD: (220, 180, 50),
    SceneTruth.REPLAY: (70, 70, 230),
    SceneTruth.CUTAWAY: (60, 150, 240),
    SceneTruth.OTHER: (160, 160, 160),
}


@dataclass(frozen=True)
class GuideInterval:
    """One scene, proposal, or GT interval drawn during review."""

    start_frame: int
    end_frame: int
    label: str
    note: str = ""


class TimelineSession:
    """Editable interval state kept independent from the OpenCV event loop."""

    def __init__(
        self,
        metadata: VideoMetadata,
        intervals: Sequence[LabelInterval],
        *,
        covered_start: int = 0,
        covered_end: int | None = None,
    ) -> None:
        self.metadata = metadata
        self.covered_start = covered_start
        self.covered_end = metadata.frame_count if covered_end is None else covered_end
        if not 0 <= self.covered_start < self.covered_end <= metadata.frame_count:
            raise ValueError(
                f"covered range [{self.covered_start}, {self.covered_end}) is outside "
                f"[0, {metadata.frame_count})"
            )
        self.intervals = list(intervals)
        validate_intervals(self.intervals, expected_metadata=metadata)
        for interval in self.intervals:
            overlaps = interval.start_frame < self.covered_end and interval.end_frame > self.covered_start
            contained = self.covered_start <= interval.start_frame and interval.end_frame <= self.covered_end
            if overlaps and not contained:
                raise ValueError(
                    f"interval [{interval.start_frame}, {interval.end_frame}) crosses the covered range boundary"
                )
        self.selection_start: int | None = None

    def selected_index(self, frame: int) -> int | None:
        return interval_index_at(self.intervals, frame)

    def set_selection_start(self, frame: int) -> None:
        self._check_frame(frame)
        self.selection_start = frame

    def clear_selection(self) -> None:
        self.selection_start = None

    def _check_frame(self, frame: int) -> None:
        if not self.covered_start <= frame < self.covered_end:
            raise ValueError(
                f"frame {frame} is outside [{self.covered_start}, {self.covered_end})"
            )

    def _gap_start_containing(self, frame: int) -> int:
        self._check_frame(frame)
        previous_end = self.covered_start
        for interval in self._covered_intervals():
            if frame < interval.start_frame:
                return previous_end
            if interval.start_frame <= frame < interval.end_frame:
                raise ValueError(f"frame {frame} is already labelled {interval.truth.value}")
            previous_end = interval.end_frame
        return previous_end

    def _covered_intervals(self) -> list[LabelInterval]:
        return [
            interval
            for interval in self.intervals
            if self.covered_start <= interval.start_frame and interval.end_frame <= self.covered_end
        ]

    def commit_through(self, frame: int, truth: SceneTruth, note: str = "") -> LabelInterval:
        """Commit through the displayed frame or relabel its existing interval."""
        self._check_frame(frame)
        selected = self.selected_index(frame)
        if self.selection_start is None and selected is not None:
            self.intervals = replace_interval(self.intervals, selected, truth=truth)
            return self.intervals[selected]

        start = self.selection_start
        if start is None:
            start = self._gap_start_containing(frame)
        end = frame + 1
        return self.commit_interval(start, end, truth, note)

    def commit_interval(
        self,
        start_frame: int,
        end_frame: int,
        truth: SceneTruth,
        note: str = "",
    ) -> LabelInterval:
        """Commit one exact half-open interval without crossing existing labels."""
        if not self.covered_start <= start_frame < end_frame <= self.covered_end:
            raise ValueError(
                f"selection [{start_frame}, {end_frame}) is outside "
                f"[{self.covered_start}, {self.covered_end})"
            )
        replacement = make_interval(self.metadata, start_frame, end_frame, truth, note)
        for interval in self.intervals:
            if replacement.start_frame < interval.end_frame and replacement.end_frame > interval.start_frame:
                raise ValueError(
                    f"selection [{start_frame}, {end_frame}) overlaps existing "
                    f"[{interval.start_frame}, {interval.end_frame})"
                )
        self.intervals.append(replacement)
        self.intervals.sort(key=lambda interval: interval.start_frame)
        validate_intervals(self.intervals, expected_metadata=self.metadata)
        self.selection_start = None
        return replacement

    def delete_at(self, frame: int) -> LabelInterval:
        self._check_frame(frame)
        index = self.selected_index(frame)
        if index is None:
            raise ValueError(f"frame {frame} has no interval to delete")
        removed = self.intervals.pop(index)
        self.selection_start = removed.start_frame
        return removed

    def set_note_at(self, frame: int, note: str) -> LabelInterval:
        self._check_frame(frame)
        index = self.selected_index(frame)
        if index is None:
            raise ValueError(f"frame {frame} has no interval to note")
        self.intervals = replace_interval(self.intervals, index, note=note)
        return self.intervals[index]

    def first_gap(self, from_frame: int | None = None) -> int | None:
        cursor = self.covered_start if from_frame is None else max(self.covered_start, from_frame)
        for interval in self._covered_intervals():
            if interval.end_frame <= cursor:
                continue
            if interval.start_frame > cursor:
                return cursor
            cursor = max(cursor, interval.end_frame)
        return cursor if cursor < self.covered_end else None

    def validate_complete(self) -> None:
        validate_partition(
            self._covered_intervals(),
            covered_start=self.covered_start,
            covered_end=self.covered_end,
            expected_metadata=self.metadata,
        )


def _open_guide_csv(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def read_guides(
    path: Path,
    *,
    frame_count: int,
    start_column: str,
    end_column: str,
    label_column: str | None,
    end_inclusive: bool = False,
) -> list[GuideInterval]:
    """Read optional proposal or GT spans without treating them as truth."""
    guides: list[GuideInterval] = []
    with _open_guide_csv(Path(path)) as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = {start_column, end_column}
        if not required.issubset(columns):
            raise ValueError(f"{path} is missing guide columns {sorted(required - columns)}")
        if label_column is not None and label_column not in columns:
            raise ValueError(f"{path} is missing guide label column {label_column!r}")
        for row_number, row in enumerate(reader, start=2):
            try:
                start = int(row[start_column])
                end = int(row[end_column]) + int(end_inclusive)
            except ValueError as exc:
                raise ValueError(f"{path} row {row_number} has non-integer frame bounds") from exc
            if not 0 <= start < end <= frame_count:
                raise ValueError(
                    f"{path} row {row_number} interval [{start}, {end}) is outside [0, {frame_count})"
                )
            label = str(row[label_column]) if label_column is not None else "guide"
            guides.append(GuideInterval(start, end, label, str(row.get("note", ""))))
    return guides


def read_scene_partition(
    path: Path,
    *,
    frame_count: int,
    covered_start: int,
    covered_end: int,
    start_column: str,
    end_column: str,
) -> list[GuideInterval]:
    """Read a complete half-open scene partition and clip it to the review range."""
    if not 0 <= covered_start < covered_end <= frame_count:
        raise ValueError(
            f"covered range [{covered_start}, {covered_end}) is outside [0, {frame_count})"
        )
    source_scenes = read_guides(
        path,
        frame_count=frame_count,
        start_column=start_column,
        end_column=end_column,
        label_column=None,
    )
    if not source_scenes:
        raise ValueError(f"{path} contains no scenes")

    expected_start = 0
    for row_number, scene in enumerate(source_scenes, start=2):
        if scene.start_frame != expected_start:
            relation = "gap" if scene.start_frame > expected_start else "overlap"
            raise ValueError(
                f"{path} row {row_number} creates a {relation}: "
                f"starts at {scene.start_frame}, expected {expected_start}"
            )
        expected_start = scene.end_frame
    if expected_start != frame_count:
        raise ValueError(f"{path} scene partition ends at {expected_start}, expected {frame_count}")

    scenes: list[GuideInterval] = []
    for scene_index, scene in enumerate(source_scenes):
        start = max(scene.start_frame, covered_start)
        end = min(scene.end_frame, covered_end)
        if start < end:
            scenes.append(GuideInterval(start, end, f"scene {scene_index}"))
    return scenes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True, help="source video")
    parser.add_argument("--video-id", required=True, help="canonical source identifier")
    parser.add_argument("--out-csv", type=Path, required=True, help="plain or gzip label CSV")
    parser.add_argument("--scene-csv", type=Path, help="optional complete half-open scene partition")
    parser.add_argument("--scene-start-col", default="start_frame")
    parser.add_argument("--scene-end-col", default="end_frame")
    parser.add_argument("--proposal-csv", type=Path, help="optional candidate intervals")
    parser.add_argument("--proposal-start-col", default="start_frame")
    parser.add_argument("--proposal-end-col", default="end_frame")
    parser.add_argument("--proposal-label-col", default="truth")
    parser.add_argument("--gt-csv", type=Path, help="optional GT rally extent guide")
    parser.add_argument("--gt-start-col", default="first_stroke_frame")
    parser.add_argument("--gt-end-col", default="last_stroke_frame")
    parser.add_argument("--start-frame", type=int, default=0, help="covered range start")
    parser.add_argument("--end-frame", type=int, help="covered range end, exclusive")
    parser.add_argument("--jump-frames", type=int, default=25, help="coarse seek distance")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the existing CSV against the video and exit",
    )
    return parser


def video_metadata(capture: cv2.VideoCapture, video_id: str) -> VideoMetadata:
    """Read and validate the metadata used by every output row."""
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


def _guide_at(guides: Sequence[GuideInterval], frame: int) -> list[GuideInterval]:
    return [guide for guide in guides if guide.start_frame <= frame < guide.end_frame]


def _scene_at(scenes: Sequence[GuideInterval], frame: int) -> GuideInterval:
    matches = _guide_at(scenes, frame)
    if len(matches) != 1:
        raise ValueError(f"frame {frame} belongs to {len(matches)} scene intervals, expected one")
    return matches[0]


def _scene_preview_frame(scene: GuideInterval) -> int:
    return (scene.start_frame + scene.end_frame - 1) // 2


def _next_unlabelled_scene_preview(
    session: TimelineSession,
    scenes: Sequence[GuideInterval],
    from_frame: int | None = None,
) -> int | None:
    gap = session.first_gap(from_frame)
    if gap is None:
        return None
    scene = _scene_at(scenes, gap)
    if scene.start_frame != gap:
        return gap
    return _scene_preview_frame(scene)


def commit_label(
    session: TimelineSession,
    frame: int,
    truth: SceneTruth,
    scenes: Sequence[GuideInterval],
) -> tuple[LabelInterval, int | None]:
    """Commit a manual interval or an exact scene, returning an optional next target."""
    if not scenes or session.selection_start is not None or session.selected_index(frame) is not None:
        return session.commit_through(frame, truth), None

    scene = _scene_at(scenes, frame)
    interval = session.commit_interval(scene.start_frame, scene.end_frame, truth)
    return interval, _next_unlabelled_scene_preview(session, scenes, interval.end_frame)


def _timeline_x(frame: int, width: int, start: int, end: int) -> int:
    fraction = (frame - start) / (end - start)
    return int(round(np.clip(fraction, 0.0, 1.0) * (width - 1)))


def draw_timeline(
    image: np.ndarray,
    session: TimelineSession,
    frame: int,
    scenes: Sequence[GuideInterval],
    proposals: Sequence[GuideInterval],
    gt_guides: Sequence[GuideInterval],
) -> None:
    """Draw compact truth, proposal, GT, and cursor tracks."""
    height, width = image.shape[:2]
    top = max(0, height - 20)
    image[top:height, :] = 0
    for interval in session.intervals:
        if interval.end_frame <= session.covered_start or interval.start_frame >= session.covered_end:
            continue
        left = _timeline_x(interval.start_frame, width, session.covered_start, session.covered_end)
        right = _timeline_x(interval.end_frame, width, session.covered_start, session.covered_end)
        cv2.rectangle(image, (left, top + 8), (max(left, right - 1), height - 1), TRUTH_COLOURS[interval.truth], -1)
    for scene in scenes[1:]:
        boundary = _timeline_x(scene.start_frame, width, session.covered_start, session.covered_end)
        cv2.line(image, (boundary, top + 5), (boundary, height - 1), (0, 200, 255), 1)
    for guide in proposals:
        left = _timeline_x(guide.start_frame, width, session.covered_start, session.covered_end)
        right = _timeline_x(guide.end_frame, width, session.covered_start, session.covered_end)
        cv2.line(image, (left, top + 4), (right, top + 4), (255, 0, 255), 2)
    for guide in gt_guides:
        left = _timeline_x(guide.start_frame, width, session.covered_start, session.covered_end)
        right = _timeline_x(guide.end_frame, width, session.covered_start, session.covered_end)
        cv2.line(image, (left, top + 1), (right, top + 1), (255, 255, 255), 1)
    cursor = _timeline_x(frame, width, session.covered_start, session.covered_end)
    cv2.line(image, (cursor, top), (cursor, height - 1), (0, 255, 255), 1)


def _draw_status(
    image: np.ndarray,
    session: TimelineSession,
    frame: int,
    scenes: Sequence[GuideInterval],
    proposals: Sequence[GuideInterval],
    gt_guides: Sequence[GuideInterval],
    message: str,
) -> None:
    interval_index = session.selected_index(frame)
    truth = session.intervals[interval_index].truth.value if interval_index is not None else "unlabelled"
    scene_text = ", ".join(guide.label for guide in _guide_at(scenes, frame)) or "none"
    proposal_text = ", ".join(guide.label for guide in _guide_at(proposals, frame)) or "none"
    gt_text = ", ".join(guide.label for guide in _guide_at(gt_guides, frame)) or "none"
    selection = "none" if session.selection_start is None else str(session.selection_start)
    lines = [
        f"frame {frame}/{session.metadata.frame_count - 1}  time {frame / session.metadata.fps:.3f}s",
        f"truth {truth}  selection_start {selection}  scene {scene_text}  proposal {proposal_text}  GT {gt_text}",
        message,
    ]
    for row, line in enumerate(lines):
        y = 24 + row * 24
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _print_help(*, scene_mode: bool) -> None:
    controls = "Controls:\n"
    if scene_mode:
        controls += "  scene mode: number labels the full unlabelled scene and advances\n"
    controls += (
        "  trackbar or ,/.       previous/next frame\n"
        "  </>                   coarse jump\n"
        "  s                     set interval start at current frame\n"
        "  1 live                2 live-non-standard\n"
        "  3 replay              4 cutaway              5 other\n"
        "  d                     delete interval under cursor\n"
        "  n                     edit note for interval under cursor\n"
        "  g                     jump to first unlabelled frame\n"
        "  j                     jump to next proposal or GT boundary\n"
        "  v                     validate the complete covered range\n"
        "  Esc                   clear explicit selection\n"
        "  h                     show this help\n"
        "  q                     quit"
    )
    print(controls)


def _read_frame(capture: cv2.VideoCapture, frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    if not ok or image is None:
        raise RuntimeError(f"could not decode frame {frame}")
    return image


def _load_intervals(path: Path, metadata: VideoMetadata) -> list[LabelInterval]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    intervals = read_label_csv(path)
    validate_intervals(intervals, expected_metadata=metadata)
    return intervals


def _load_guides(
    args: argparse.Namespace,
    metadata: VideoMetadata,
    session: TimelineSession,
) -> tuple[list[GuideInterval], list[GuideInterval], list[GuideInterval]]:
    scenes = []
    if args.scene_csv is not None:
        scenes = read_scene_partition(
            args.scene_csv,
            frame_count=metadata.frame_count,
            covered_start=session.covered_start,
            covered_end=session.covered_end,
            start_column=args.scene_start_col,
            end_column=args.scene_end_col,
        )
    proposals = []
    if args.proposal_csv is not None:
        proposals = read_guides(
            args.proposal_csv,
            frame_count=metadata.frame_count,
            start_column=args.proposal_start_col,
            end_column=args.proposal_end_col,
            label_column=args.proposal_label_col,
        )
    gt_guides = []
    if args.gt_csv is not None:
        gt_guides = read_guides(
            args.gt_csv,
            frame_count=metadata.frame_count,
            start_column=args.gt_start_col,
            end_column=args.gt_end_col,
            label_column=None,
            end_inclusive=True,
        )
    return scenes, proposals, gt_guides


def run_annotation_tool(args: argparse.Namespace) -> int:
    if args.jump_frames <= 0:
        raise ValueError(f"jump_frames must be positive, got {args.jump_frames}")
    capture = cv2.VideoCapture(str(args.video))
    try:
        metadata = video_metadata(capture, args.video_id)
        intervals = _load_intervals(args.out_csv, metadata)
        session = TimelineSession(
            metadata,
            intervals,
            covered_start=args.start_frame,
            covered_end=args.end_frame,
        )
        scenes, proposals, gt_guides = _load_guides(args, metadata, session)
        if args.validate_only:
            session.validate_complete()
            print(
                f"valid partition: {args.out_csv} covers "
                f"[{session.covered_start}, {session.covered_end})"
            )
            return 0
        _run_gui(capture, args.out_csv, session, scenes, proposals, gt_guides, args.jump_frames)
        return 0
    finally:
        capture.release()


def _run_gui(
    capture: cv2.VideoCapture,
    output_path: Path,
    session: TimelineSession,
    scenes: Sequence[GuideInterval],
    proposals: Sequence[GuideInterval],
    gt_guides: Sequence[GuideInterval],
    jump_frames: int,
) -> None:
    initial_scene_frame = _next_unlabelled_scene_preview(session, scenes) if scenes else None
    frame_state = {"index": session.covered_start if initial_scene_frame is None else initial_scene_frame}
    message_state = {"text": "press h for controls"}
    boundaries = sorted({
        frame
        for guide in (*proposals, *gt_guides)
        for frame in (guide.start_frame, guide.end_frame)
        if session.covered_start <= frame < session.covered_end
    })

    def redraw(frame: int) -> None:
        frame = min(max(frame, session.covered_start), session.covered_end - 1)
        frame_state["index"] = frame
        image = _read_frame(capture, frame)
        draw_timeline(image, session, frame, scenes, proposals, gt_guides)
        _draw_status(image, session, frame, scenes, proposals, gt_guides, message_state["text"])
        cv2.imshow(WINDOW_NAME, image)

    def save() -> None:
        write_label_csv(output_path, session.intervals, session.metadata)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(
        "frame",
        WINDOW_NAME,
        frame_state["index"],
        max(session.metadata.frame_count - 1, 1),
        redraw,
    )
    _print_help(scene_mode=bool(scenes))
    redraw(frame_state["index"])
    try:
        while True:
            key = cv2.waitKeyEx(0)
            frame = frame_state["index"]
            target = frame
            try:
                if key in (ord("q"), ord("Q")):
                    break
                if key == 27:
                    session.clear_selection()
                    message_state["text"] = "selection cleared"
                elif key == ord(","):
                    target = frame - 1
                elif key == ord("."):
                    target = frame + 1
                elif key == ord("<"):
                    target = frame - jump_frames
                elif key == ord(">"):
                    target = frame + jump_frames
                elif key in (ord("s"), ord("S")):
                    session.set_selection_start(frame)
                    message_state["text"] = f"selection starts at {frame}"
                elif key in LABEL_KEYS:
                    interval, next_scene_frame = commit_label(session, frame, LABEL_KEYS[key], scenes)
                    save()
                    if next_scene_frame is not None:
                        target = next_scene_frame
                    message_state["text"] = (
                        f"saved [{interval.start_frame}, {interval.end_frame}) {interval.truth.value}"
                    )
                elif key in (ord("d"), ord("D")):
                    removed = session.delete_at(frame)
                    save()
                    message_state["text"] = f"deleted [{removed.start_frame}, {removed.end_frame})"
                elif key in (ord("n"), ord("N")):
                    note = input("note (empty clears): ").strip()
                    interval = session.set_note_at(frame, note)
                    save()
                    message_state["text"] = f"saved note for [{interval.start_frame}, {interval.end_frame})"
                elif key in (ord("g"), ord("G")):
                    next_unlabelled = (
                        _next_unlabelled_scene_preview(session, scenes) if scenes else session.first_gap()
                    )
                    if next_unlabelled is None:
                        message_state["text"] = "covered range has no gaps"
                    else:
                        target = next_unlabelled
                        destination = "scene preview" if scenes else "gap start"
                        message_state["text"] = f"first unlabelled {destination} {next_unlabelled}"
                elif key in (ord("j"), ord("J")):
                    target = next(
                        (boundary for boundary in boundaries if boundary > frame),
                        boundaries[0] if boundaries else frame,
                    )
                    message_state["text"] = f"guide boundary {target}"
                elif key in (ord("v"), ord("V")):
                    session.validate_complete()
                    message_state["text"] = "complete partition is valid"
                elif key in (ord("h"), ord("H")):
                    _print_help(scene_mode=bool(scenes))
                    message_state["text"] = "controls printed in terminal"
            except (ValueError, IndexError) as exc:
                message_state["text"] = f"error: {exc}"
            target = min(max(target, session.covered_start), session.covered_end - 1)
            cv2.setTrackbarPos("frame", WINDOW_NAME, target)
            redraw(target)
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def main() -> int:
    return run_annotation_tool(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
