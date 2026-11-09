#!/home/ariel/.venvs/badminton-cicd/bin/python
"""Annotate labelled court intersections in the single native workspace.

Click points in any order, confirm their identities, then save four clicked
outer corners or fit at least five well-spread known intersections. Missing
corners are projected from the fitted homography. Output stays canonical:
corner_idx 0–3 is TL, TR, BR, BL regardless of input order.

Corner CSV columns::

    video,frame,corner_idx,x_px,y_px,x_norm,y_norm,orientation,corner_label,visible,source,fit_rms_px

Clicked outer corners use source=click. Derived corners use source=extrapolated
and all corners use visible=1 only when their coordinates lie inside the image.
Normalised coordinates remain unclamped. The landmark sidecar records actual
confirmed labelled points, one row per identity. Re-saving a frame replaces
both its corner rows and landmark rows. Review decisions remain separate.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

# Direct script launches need the scripts directory for the shared annotation helpers.
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from court_annotation import court_landmarks as court  # noqa: E402
from court_annotation.image_helpers import (  # noqa: E402
    CORNER_COLOUR,
    TEXT_COLOUR,
    loupe_origin,
    loupe_to_source,
    normalise_xy,
)
from court_annotation.review_queue import ReviewItem, ReviewQueue, load_candidates  # noqa: E402


def read_saved_courts(path: Path, video_basename: str) -> dict[int, list[tuple[float, float]]]:
    if not path.exists():
        return {}
    grouped: dict[int, list[tuple[int, tuple[float, float]]]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if Path(row["video"]).name == video_basename:
                point = (float(row["x_px"]), float(row["y_px"]))
                grouped.setdefault(int(row["frame"]), []).append((int(row["corner_idx"]), point))
    return {frame: [point for _, point in sorted(points)] for frame, points in grouped.items()}


def clock_label(frame: int, fps: float) -> str:
    seconds = frame / fps
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def previous_court(annotations: dict[int, list[tuple[float, float]]], frame: int
                   ) -> tuple[int, list[tuple[float, float]]] | None:
    earlier = [saved for saved in annotations if saved <= frame]
    if not earlier:
        return None
    saved = max(earlier)
    return saved, annotations[saved]


CSV_HEADER = (
    "video", "frame", "corner_idx", "x_px", "y_px", "x_norm", "y_norm", "orientation",
    "corner_label", "visible", "source", "fit_rms_px",
)
LANDMARK_CSV_HEADER = ("video", "frame", "landmark", "court_x_m", "court_y_m", "x_px", "y_px")
# Video menu entries show this much of a video's note.
NOTE_PREVIEW_CHARS = 40

# Sky blue from the Wong palette (#56B4E9 as BGR): landmarks read apart from the
# orange corner marks for protan vision.
LANDMARK_COLOUR = (233, 180, 86)


# --- Capture state machine (unit-tested) -----------------------------------

class CaptureState(StrEnum):
    SCRUB = "scrub"
    POINT_COARSE = "point_coarse"
    POINT_ADJUST = "point_adjust"


class ActionKind(StrEnum):
    """What the GUI shell should do after handling an event."""

    NOOP = "noop"  # event not valid in this state; ignore it
    REDRAW = "redraw"  # session advanced; close any loupe and redraw
    OPEN_LOUPE = "open_loupe"  # provisional point down; (re)open the loupe on it
    ADJUSTED = "adjusted"  # the provisional point moved; re-render the loupe
    COMMITTED = "committed"  # rows written; back to scrub
    ABORTED = "aborted"  # capture cancelled; back to scrub


@dataclass(frozen=True)
class SessionAction:
    """The session's reply to one event.

    :param kind: what the shell should do next
    :param message: console/status text, or None when nothing changed worth saying
    :param loupe_origin: crop top-left for OPEN_LOUPE, else None
    :param coarse_xy: the coarse click for OPEN_LOUPE (loupe crosshair), else None
    """

    kind: ActionKind
    message: str | None = None
    loupe_origin: tuple[int, int] | None = None
    coarse_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class FitResult:
    """A landmark homography fit and what it says about the corners.

    :param homography: (3, 3) court metres -> image pixels
    :param corners_px: (4, 2) all four corners projected through the fit, slot order
    :param rms_px: rms landmark reprojection residual
    :param max_px: worst landmark reprojection residual
    :param n_landmarks: pairs the fit used
    """

    homography: np.ndarray
    corners_px: np.ndarray
    rms_px: float
    max_px: float
    n_landmarks: int


class OffframeSession:
    """Label court intersections in any order, refine them, then save or fit corners."""

    def __init__(self, video: str, width: int, height: int, orientation: str,
                 csv_path: Path, half: int, zoom: int) -> None:
        self.video = video
        self.video_basename = Path(video).name
        self.width, self.height = width, height
        self.orientation = orientation
        self.csv_path = Path(csv_path)
        self.landmark_csv_path = self.csv_path.with_name(self.csv_path.stem + "_landmarks.csv")
        ensure_header(self.csv_path, CSV_HEADER)
        ensure_header(self.landmark_csv_path, LANDMARK_CSV_HEADER)
        self.annotated_frames = load_annotated_frames(self.csv_path, self.video_basename)
        self.half, self.zoom = half, zoom
        self.frame_idx = -1
        self.state = CaptureState.SCRUB
        self.selected_name: str | None = None
        self._editing_name: str | None = None
        self._pending_point: tuple[float, float] | None = None
        self._pending_origin: tuple[int, int] | None = None
        self._landmarks: list[tuple[str, float, float]] = []
        self._history: list[list[tuple[str, float, float]]] = []
        self.fit: FitResult | None = None

    @property
    def slots(self) -> tuple[tuple[float, float] | None, ...]:
        points = {name: (x, y) for name, x, y in self._landmarks}
        return tuple(points.get(name) for name in court.CORNER_LANDMARK_NAMES)

    @property
    def missing_corner_slots(self) -> tuple[int, ...]:
        return tuple(index for index, point in enumerate(self.slots) if point is None)

    @property
    def landmarks(self) -> tuple[tuple[str, float, float], ...]:
        return tuple(self._landmarks)

    @property
    def pending_point(self) -> tuple[float, float] | None:
        return self._pending_point

    @property
    def pending_origin(self) -> tuple[int, int] | None:
        return self._pending_origin

    @property
    def adjusting(self) -> bool:
        return self.state == CaptureState.POINT_ADJUST

    @property
    def can_save(self) -> bool:
        return (self.state == CaptureState.POINT_COARSE
                and (all(point is not None for point in self.slots) or self.fit is not None))

    def begin_capture(self, frame_idx: int) -> SessionAction:
        if self.state != CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        self._reset_capture()
        self.frame_idx = frame_idx
        self.state = CaptureState.POINT_COARSE
        message = "Click any court intersection, then choose its identity."
        if frame_idx in self.annotated_frames:
            message = f"Frame {frame_idx} is already annotated; saving replaces it. {message}"
        return SessionAction(ActionKind.REDRAW, message=message)

    def select_point(self, name: str | None) -> SessionAction:
        """Relabel a provisional point, or reopen an existing labelled point."""
        if self.state == CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        if name is not None and name not in court.LANDMARKS:
            raise ValueError(f"Unknown court point: {name}")
        self.selected_name = name
        if self.adjusting:
            return SessionAction(ActionKind.ADJUSTED, message="Point identity changed; its position is unchanged.")
        existing = next(((x, y) for label, x, y in self._landmarks if label == name), None)
        self._editing_name = name if existing is not None else None
        return self.coarse_click(*existing) if existing is not None else SessionAction(ActionKind.REDRAW)

    def begin_capture_prefilled(self, frame_idx: int, slots: list[tuple[float, float] | None],
                                landmarks: list[tuple[str, float, float]], source_frame: int) -> SessionAction:
        """Copy known identities without turning derived corners into clicked points."""
        if self.state != CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        self.begin_capture(frame_idx)
        points = {name: (x, y) for name, x, y in landmarks}
        for name, point in zip(court.CORNER_LANDMARK_NAMES, slots, strict=True):
            if point is not None:
                points[name] = point
        for name in court.LANDMARK_NAMES:
            if name in points:
                self._history.append(list(self._landmarks))
                x, y = points[name]
                self._landmarks.append((name, x, y))
        return SessionAction(ActionKind.REDRAW,
                             message=f"Copied labelled points from frame {source_frame}; review, fit if needed, then Save court.")

    def coarse_click(self, x: float, y: float) -> SessionAction:
        if self.state == CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        self.state = CaptureState.POINT_ADJUST
        self._pending_origin = loupe_origin(x, y, self.half, self.width, self.height)
        self._pending_point = (x, y)
        return SessionAction(ActionKind.OPEN_LOUPE, message=self._point_status(),
                             loupe_origin=self._pending_origin, coarse_xy=(x, y))

    def set_loupe_origin(self, origin: tuple[int, int]) -> None:
        """Move the refinement viewport without changing the provisional point."""
        self._pending_origin = origin

    def reposition_from_loupe(self, lx: float, ly: float) -> SessionAction:
        """Move the provisional point to a loupe click, sub-pixel.

        :param lx: click x in the zoomed loupe image
        :param ly: click y in the zoomed loupe image
        :return: ADJUSTED with the new position, or NOOP when nothing is pending
        """
        if not self.adjusting:
            return SessionAction(ActionKind.NOOP)
        # _pending_origin is always set while an adjust state is live.
        assert self._pending_origin is not None
        self._pending_point = loupe_to_source(lx, ly, self._pending_origin, self.zoom)
        return SessionAction(ActionKind.ADJUSTED, message=self._point_status())

    def nudge(self, dx: float, dy: float) -> SessionAction:
        """Shift the provisional point, for arrow-key fine adjustment.

        :param dx: shift in source px, positive right
        :param dy: shift in source px, positive down
        :return: ADJUSTED with the new position, or NOOP when nothing is pending
        """
        if not self.adjusting:
            return SessionAction(ActionKind.NOOP)
        assert self._pending_point is not None
        self._pending_point = (self._pending_point[0] + dx, self._pending_point[1] + dy)
        return SessionAction(ActionKind.ADJUSTED, message=self._point_status())

    def set_point(self, x: float, y: float) -> SessionAction:
        """Put the provisional point at typed source-pixel coordinates.

        :param x: typed x in source pixels
        :param y: typed y in source pixels
        :return: ADJUSTED with the new position, or NOOP when nothing is pending
        """
        if not self.adjusting:
            return SessionAction(ActionKind.NOOP)
        self._pending_point = (float(x), float(y))
        return SessionAction(ActionKind.ADJUSTED, message=self._point_status())

    def confirm_point(self) -> SessionAction:
        if not self.adjusting:
            return SessionAction(ActionKind.NOOP)
        if self.selected_name is None:
            return SessionAction(ActionKind.ADJUSTED, message="Choose which court intersection this point is before confirming.")
        assert self._pending_point is not None
        name = self.selected_name
        confirmed_index = court.LANDMARK_NAMES.index(name)
        self._history.append(list(self._landmarks))
        replaced_names = {name, self._editing_name}
        self._landmarks = [entry for entry in self._landmarks if entry[0] not in replaced_names]
        self._landmarks.append((name, *self._pending_point))
        self.fit = None
        self._editing_name = None
        self._clear_pending()
        self.state = CaptureState.POINT_COARSE
        if confirmed_index + 1 < len(court.LANDMARK_NAMES):
            self.selected_name = court.LANDMARK_NAMES[confirmed_index + 1]
            message = f"Point confirmed. {len(self._landmarks)} labelled points; {court.FIT_MIN_POINTS} needed to fit."
        else:
            self.selected_name = None
            message = "Last catalogue point confirmed. Choose another point or Save court."
        return SessionAction(ActionKind.REDRAW, message=message)

    def cancel_aim(self) -> SessionAction:
        if not self.adjusting:
            return SessionAction(ActionKind.NOOP)
        self._clear_pending()
        self._editing_name = None
        self.state = CaptureState.POINT_COARSE
        return SessionAction(ActionKind.REDRAW, message="Provisional point cancelled; confirmed points are unchanged.")

    def _point_status(self) -> str:
        assert self._pending_point is not None
        x, y = self._pending_point
        return f"Point at ({x:.2f}, {y:.2f}): choose its identity, refine, then Confirm point."

    def _clear_pending(self) -> None:
        self._pending_point = None
        self._pending_origin = None

    def undo_last(self) -> SessionAction:
        if self.adjusting:
            return self.cancel_aim()
        if self.state == CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        if not self._history:
            return SessionAction(ActionKind.REDRAW, message="No confirmed point to undo.")
        self._landmarks = self._history.pop()
        self.selected_name = None
        self.fit = None
        return SessionAction(ActionKind.REDRAW, message="Restored the previous labelled-point set.")

    def do_fit(self) -> SessionAction:
        """Fit the homography from the clicked landmarks, gated, and project corners.

        Refusals (fit stays None): under the point floor, a degenerate spread, a
        worst-point reprojection over the calibrated ceiling, or a projected quad
        that is not a convex behind-baseline court. A warn-level reprojection
        keeps the fit but says so; Save court accepts it explicitly.

        :return: REDRAW with the fit summary (or the reason it refused),
            or NOOP while a point is being refined or no capture is active
        """
        if self.state is not CaptureState.POINT_COARSE:
            return SessionAction(ActionKind.NOOP)
        if len(self._landmarks) < court.FIT_MIN_POINTS:
            return SessionAction(
                ActionKind.REDRAW,
                message=f"fit needs at least {court.FIT_MIN_POINTS} landmarks "
                        f"(4 fit exactly and hide a mislabel); have {len(self._landmarks)}",
            )
        court_pts = np.array([court.LANDMARKS[name] for name, _x, _y in self._landmarks], dtype=np.float64)
        image_pts = np.array([[x, y] for _name, x, y in self._landmarks], dtype=np.float64)
        try:
            homography, residuals = court.fit_homography(court_pts, image_pts)
        except ValueError as error:
            return SessionAction(ActionKind.REDRAW, message=f"fit failed: {error}")
        corners_px = court.project_corners(homography)
        if not court.quad_is_camera_valid(corners_px):
            return SessionAction(
                ActionKind.REDRAW,
                message="fit failed: projected quad is not a convex behind-baseline court "
                        "(check for a mislabelled crossing)",
            )
        verdict = court.check_fit(residuals, self.width)
        if verdict.level == "fail":
            return SessionAction(ActionKind.REDRAW, message=f"fit failed: {verdict.reason}")
        self.fit = FitResult(
            homography=homography,
            corners_px=corners_px,
            rms_px=verdict.rms_px,
            max_px=verdict.worst_px,
            n_landmarks=len(self._landmarks),
        )
        placed = ", ".join(
            f"{court.CORNER_NAMES[idx]}=({self.fit.corners_px[idx, 0]:.1f}, {self.fit.corners_px[idx, 1]:.1f})"
            for idx in self.missing_corner_slots
        )
        summary = f"fit: {self.fit.n_landmarks} landmarks, rms {self.fit.rms_px:.2f} px, max {self.fit.max_px:.2f} px"
        if placed:
            summary += f"; projected {placed}"
        if verdict.level == "warn":
            return SessionAction(ActionKind.REDRAW,
                                 message=f"fit WARN: {verdict.reason}. {summary}. Save court accepts this fit")
        return SessionAction(ActionKind.REDRAW, message=summary + ". choose Save court")

    def commit(self) -> SessionAction:
        if not self.can_save:
            return SessionAction(ActionKind.REDRAW, message="Confirm all four outer corners, or fit the labelled intersections first.")
        rows = build_slot_rows(self.video, self.frame_idx, self.slots, self.fit,
                               self.width, self.height, self.orientation)
        resolved = np.array([(row["x_px"], row["y_px"]) for row in rows], dtype=np.float64)
        if not court.quad_is_camera_valid(resolved):
            return SessionAction(ActionKind.REDRAW,
                                 message="Court is crossed, mirrored or upside-down. Check the point identities before saving.")
        replaced = upsert_csv(self.csv_path, rows, CSV_HEADER, self.video_basename, self.frame_idx)
        landmark_rows = build_landmark_rows(self.video, self.frame_idx, self._landmarks)
        upsert_csv(self.landmark_csv_path, landmark_rows, LANDMARK_CSV_HEADER, self.video_basename, self.frame_idx)
        frame = self.frame_idx
        self.annotated_frames.add(frame)
        self._reset_capture()
        self.state = CaptureState.SCRUB
        verb = "Replaced" if replaced else "Saved"
        return SessionAction(ActionKind.COMMITTED, message=f"{verb} court at frame {frame}.")

    def abort(self) -> SessionAction:
        if self.state == CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        self._reset_capture()
        self.state = CaptureState.SCRUB
        return SessionAction(ActionKind.ABORTED, message="Capture cancelled; nothing saved.")

    def _reset_capture(self) -> None:
        self.selected_name = None
        self._editing_name = None
        self._landmarks = []
        self._history = []
        self._clear_pending()
        self.fit = None


def build_slot_rows(
    video: str,
    frame: int,
    slots: list[tuple[float, float] | None] | tuple[tuple[float, float] | None, ...],
    fit: FitResult | None,
    width: int,
    height: int,
    orientation: str,
) -> list[dict[str, object]]:
    """Build one CSV row per corner slot, extrapolating the off-frame ones.

    :param video: source video path string, stored verbatim
    :param frame: frame index the corners belong to
    :param slots: 4 entries in slot order TL TR BR BL; None means derived from the fit
    :param fit: the landmark fit; required when any slot is None
    :param width: frame width in pixels, for the normalised columns
    :param height: frame height in pixels, for the normalised columns
    :param orientation: orientation flag stored on every row
    :return: list of 4 row dicts keyed by :data:`CSV_HEADER`
    :raises ValueError: on a slot count other than 4, or None slots without a fit
    """
    if len(slots) != 4:
        raise ValueError(f"expected 4 resolved slots, got {len(slots)}")
    if any(entry is None for entry in slots) and fit is None:
        raise ValueError("missing corners need a landmark fit to derive them")
    rows: list[dict[str, object]] = []
    for slot, clicked in enumerate(slots):
        if clicked is None:
            assert fit is not None  # guarded above
            x_px, y_px = float(fit.corners_px[slot, 0]), float(fit.corners_px[slot, 1])
            visible = int(0 <= x_px < width and 0 <= y_px < height)
            source, fit_rms = "extrapolated", f"{fit.rms_px:.3f}"
        else:
            x_px, y_px = clicked
            visible = int(0 <= x_px < width and 0 <= y_px < height)
            source, fit_rms = "click", ""
        x_norm, y_norm = normalise_xy(x_px, y_px, width, height)
        rows.append({
            "video": video,
            "frame": frame,
            "corner_idx": slot,
            "x_px": x_px,
            "y_px": y_px,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "orientation": orientation,
            "corner_label": court.CORNER_NAMES[slot],
            "visible": visible,
            "source": source,
            "fit_rms_px": fit_rms,
        })
    return rows


def build_landmark_rows(
    video: str, frame: int, landmarks: list[tuple[str, float, float]] | tuple[tuple[str, float, float], ...]
) -> list[dict[str, object]]:
    """Build one sidecar CSV row per clicked landmark.

    :param video: source video path string, stored verbatim
    :param frame: frame index the landmarks belong to
    :param landmarks: (name, x_px, y_px) clicks; names must be catalogue entries
    :return: row dicts keyed by :data:`LANDMARK_CSV_HEADER`
    """
    rows: list[dict[str, object]] = []
    for name, x_px, y_px in landmarks:
        court_x, court_y = court.LANDMARKS[name]
        rows.append({
            "video": video,
            "frame": frame,
            "landmark": name,
            "court_x_m": court_x,
            "court_y_m": court_y,
            "x_px": x_px,
            "y_px": y_px,
        })
    return rows


def ensure_header(csv_path: Path, header: tuple[str, ...]) -> None:
    """Refuse to build on a CSV whose columns are not this tool's.

    Appending 12-column rows under an 8-column header (or vice versa) corrupts
    the file silently, so a mismatch fails before any annotation effort is spent.

    :param csv_path: CSV to inspect; missing or empty files pass
    :param header: the expected column names
    :raises ValueError: on a different column set
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(newline="") as handle:
        found = tuple(csv.DictReader(handle).fieldnames or ())
    if found != header:
        raise ValueError(
            f"{csv_path} has columns {found}, expected {header}. It was written by a different "
            "tool or version; point --out-csv at a fresh file."
        )


def load_annotated_frames(csv_path: Path, video_basename: str) -> set[int]:
    """Frames already committed for this video in an existing corners CSV.

    :param csv_path: the corners CSV (validated by :func:`ensure_header` first)
    :param video_basename: video filename to match; paths in the CSV may differ
        by prefix across sessions, so only the basename is compared
    :return: committed frame indices, empty for a fresh file
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    frames: set[int] = set()
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if Path(str(row["video"])).name == video_basename:
                frames.add(int(row["frame"]))
    return frames


def load_prefill(
    csv_path: Path, landmark_csv_path: Path, video_basename: str, near_frame: int
) -> tuple[int, list[tuple[float, float] | None], list[tuple[str, float, float]]] | None:
    """The committed annotation nearest ``near_frame``, shaped as prefill data.

    Feeds the copy-forward flow for static cameras: clicked corners come back
    as points, extrapolated ones as None (off-frame again), and the frame's
    landmark clicks ride along for the re-fit.

    :param csv_path: the corners CSV
    :param landmark_csv_path: its landmarks sidecar
    :param video_basename: video filename to match
    :param near_frame: pick the annotated frame closest to this index
    :return: (source_frame, slots, landmarks), or None when this video has no
        complete committed frame yet
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    by_frame: dict[int, dict[int, dict[str, str]]] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if Path(str(row["video"])).name == video_basename:
                by_frame.setdefault(int(row["frame"]), {})[int(row["corner_idx"])] = row
    complete = {frame: rows for frame, rows in by_frame.items() if len(rows) == 4}
    if not complete:
        return None
    source_frame = min(complete, key=lambda frame: abs(frame - near_frame))
    slots: list[tuple[float, float] | None] = []
    for slot in range(4):
        row = complete[source_frame][slot]
        slots.append((float(row["x_px"]), float(row["y_px"])) if row["source"] == "click" else None)
    landmarks: list[tuple[str, float, float]] = []
    landmark_csv_path = Path(landmark_csv_path)
    if landmark_csv_path.exists() and landmark_csv_path.stat().st_size > 0:
        with landmark_csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if Path(str(row["video"])).name == video_basename and int(row["frame"]) == source_frame:
                    landmarks.append((str(row["landmark"]), float(row["x_px"]), float(row["y_px"])))
    return source_frame, slots, landmarks


def upsert_csv(
    csv_path: Path,
    rows: list[dict[str, object]],
    header: tuple[str, ...],
    video_basename: str,
    frame: int,
) -> int:
    """Write rows for one (video, frame), replacing any earlier rows for it.

    The whole file is rewritten through a sibling temp file and an atomic
    replace, so a crash mid-write cannot truncate existing annotations. Passing
    an empty ``rows`` deletes the frame's entries.

    :param csv_path: destination CSV
    :param rows: the frame's new row dicts keyed by ``header``
    :param header: the CSV's column names
    :param video_basename: video filename the rows belong to (basename match)
    :param frame: frame index the rows belong to
    :return: how many old rows were replaced or removed
    """
    csv_path = Path(csv_path)
    survivors: list[dict[str, str]] = []
    replaced = 0
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if Path(str(row["video"])).name == video_basename and int(row["frame"]) == frame:
                    replaced += 1
                else:
                    survivors.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_name(csv_path.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(survivors)
        writer.writerows(rows)
    tmp_path.replace(csv_path)
    return replaced


def draw_court_key(
    canvas: np.ndarray,
    origin: tuple[int, int],
    scale: float,
    highlight: str | None,
    placed: tuple[str, ...],
    corner_slot: int | None,
) -> None:
    """Draw a top-down court map with the 30 landmark crossings.

    Far end at the top, matching how the far baseline sits at the top of a
    normal court view. The crossing (or corner) currently being asked for is
    ringed; recorded landmarks are filled.

    :param canvas: BGR image to draw into
    :param origin: top-left px of the court rectangle on the canvas
    :param scale: px per court metre
    :param highlight: landmark name to ring, or None
    :param placed: landmark names already recorded this frame
    :param corner_slot: corner slot (0-3) to ring instead, or None
    """
    ox, oy = origin

    def to_px(court_x: float, court_y: float) -> tuple[int, int]:
        return (round(ox + court_x * scale), round(oy + court_y * scale))

    pad = 12
    right_bottom = to_px(court.COURT_WIDTH_M, court.COURT_LENGTH_M)
    cv2.rectangle(canvas, (ox - pad, oy - pad - 6), (right_bottom[0] + pad, right_bottom[1] + pad + 6),
                  (0, 0, 0), -1)
    line_colour = (120, 120, 120)
    for y_m in court.Y_LINES.values():
        cv2.line(canvas, to_px(0, y_m), to_px(court.COURT_WIDTH_M, y_m), line_colour, 1, cv2.LINE_AA)
    for name, x_m in court.X_LINES.items():
        if name == "centre":
            # The centre line stops at the short service lines; nothing is
            # painted through the mid-court.
            cv2.line(canvas, to_px(x_m, 0), to_px(x_m, court.Y_LINES["far_short_service"]),
                     line_colour, 1, cv2.LINE_AA)
            cv2.line(canvas, to_px(x_m, court.Y_LINES["near_short_service"]),
                     to_px(x_m, court.COURT_LENGTH_M), line_colour, 1, cv2.LINE_AA)
        else:
            cv2.line(canvas, to_px(x_m, 0), to_px(x_m, court.COURT_LENGTH_M), line_colour, 1, cv2.LINE_AA)
    cv2.line(canvas, to_px(0, court.NET_Y_M), to_px(court.COURT_WIDTH_M, court.NET_Y_M),
             (70, 70, 70), 1, cv2.LINE_AA)  # the net, for orientation only

    for name, (court_x, court_y) in court.LANDMARKS.items():
        point = to_px(court_x, court_y)
        if name in placed:
            cv2.circle(canvas, point, 3, LANDMARK_COLOUR, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, point, 2, (170, 170, 170), -1, cv2.LINE_AA)
    if highlight is not None:
        court_x, court_y = court.LANDMARKS[highlight]
        cv2.circle(canvas, to_px(court_x, court_y), 6, CORNER_COLOUR, 2, cv2.LINE_AA)
    if corner_slot is not None:
        court_x, court_y = court.CORNER_COURT_M[corner_slot]
        cv2.circle(canvas, to_px(float(court_x), float(court_y)), 6, CORNER_COLOUR, 2, cv2.LINE_AA)
    cv2.putText(canvas, "far", (ox, oy - pad + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                TEXT_COLOUR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "near", (ox, right_bottom[1] + pad + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                TEXT_COLOUR, 1, cv2.LINE_AA)


def open_video(video: Path) -> tuple[cv2.VideoCapture, int, float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        sys.exit(f"ERROR: could not open video {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or total <= 0:
        raise ValueError(f"{video} must report a positive frame rate and frame count")
    return capture, total, fps


def load_review(video: Path, args: argparse.Namespace, total: int, fps: float) -> ReviewQueue:
    candidates = args.candidate_dir / video.stem / "candidates.csv.gz" if args.candidate_dir else None
    items = load_candidates(candidates, video.name, fps, total, args.include_uncertain)
    if candidates is None:
        items += [ReviewItem(min(total - 1, round((index + 0.5) * total / max(1, args.frames))), "Sample court")
                  for index in range(max(1, args.frames))]
    return ReviewQueue(Path(args.out_csv).parent / f"{video.stem}_review.json.gz", video.name, items)


def pending_count(review: ReviewQueue) -> int:
    return sum(review.needs_review(item) for item in review.items)


def video_label(video: Path, review: ReviewQueue) -> str:
    pending = pending_count(review)
    label = f"{video.name} · {pending} {'stop' if pending == 1 else 'stops'} to review" if pending else f"{video.name} · reviewed"
    if not review.notes:
        return label
    note = review.notes if len(review.notes) <= NOTE_PREVIEW_CHARS else review.notes[:NOTE_PREVIEW_CHARS - 1] + "…"
    return f"{label} · “{note}”"


def batch_reviews(args: argparse.Namespace) -> list[ReviewQueue]:
    reviews = []
    for video in args.video:
        capture, total, fps = open_video(video)
        capture.release()
        reviews.append(load_review(video, args, total, fps))
    return reviews


def run_annotation_tool(args: argparse.Namespace) -> int:
    """Open one workspace at a time; the video picker chooses which opens next.

    :param args: parsed CLI namespace
    :return: process exit code
    """
    from court_annotation.native_gui import CourtWorkspace
    from PySide6 import QtWidgets

    names = [video.name for video in args.video]
    if args.start_video is not None:
        if args.start_video not in names:
            sys.exit(f"ERROR: --start-video {args.start_video} is not one of the --video files")
        index: int | None = names.index(args.start_video)
    else:
        index = next((position for position, review in enumerate(batch_reviews(args)) if pending_count(review)), 0)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    start_frame, geometry = args.start_frame, None
    while index is not None:
        # Recount every switch so the picker shows work finished in the previous video.
        reviews = batch_reviews(args)
        labels = [video_label(video, review) for video, review in zip(args.video, reviews, strict=True)]
        video, review = args.video[index], reviews[index]
        capture, total, fps = open_video(video)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            session = OffframeSession(str(video), width, height, args.orientation,
                                      Path(args.out_csv), args.loupe_size, args.loupe_zoom)
        except ValueError as error:
            sys.exit(f"ERROR: {error}")
        window = CourtWorkspace(capture, session, review, total, fps,
                                review.current.frame if start_frame is None else start_frame,
                                args.nudge_step, labels, index)
        if geometry is not None:
            window.restoreGeometry(geometry)
        window.show()
        try:
            application.exec()
        finally:
            capture.release()
        geometry = window.saveGeometry()
        index, start_frame = window.chosen_video, None
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, nargs="+", required=True,
                        help="One or more videos. With several, the workspace has a video picker.")
    parser.add_argument("--start-video", help="Filename of the --video to open first (default: first with pending stops).")
    parser.add_argument("--out-csv", type=Path, required=True,
                        help="Appended to; header written when new. Landmarks go to <stem>_landmarks.csv.")
    parser.add_argument("--orientation", type=str, default="portrait", help="Orientation flag stored per row.")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Override the first video's opening frame (default: its next pending stop).")
    parser.add_argument("--frames", type=int, default=2,
                        help="Target annotated frames per video; queue that many evenly "
                             "spaced spots (default 2).")
    parser.add_argument("--nudge-step", type=float, default=0.5,
                        help="Arrow-key nudge in source px while a placed point's loupe is open (default 0.5).")
    parser.add_argument("--loupe-size", type=int, default=20,
                        help="Loupe half-window in source pixels (default 20: 40 source pixels in the 320-pixel dock).")
    parser.add_argument("--loupe-zoom", type=int, default=8, help="Loupe magnification with NEAREST (default 8).")
    parser.add_argument("--candidate-dir", type=Path,
                        help="Camera scan folder holding <video stem>/candidates.csv.gz for every video.")
    parser.add_argument("--include-uncertain", action="store_true", help="Also queue uncertain scan transitions.")
    args = parser.parse_args()

    for video in args.video:
        if not video.exists():
            sys.exit(f"ERROR: --video does not exist: {video}")
    # Rows and review files are keyed by filename, so two videos sharing a stem would overwrite each other.
    stems = [video.stem for video in args.video]
    if len(stems) != len(set(stems)):
        sys.exit("ERROR: --video files must have unique filename stems")
    try:
        return run_annotation_tool(args)
    except KeyboardInterrupt:
        # Ctrl-C in the terminal instead of q in the window; nothing is lost,
        # every committed frame is already on disk.
        print("\ninterrupted; committed frames are already saved")
        return 130


if __name__ == "__main__":
    sys.exit(main())
