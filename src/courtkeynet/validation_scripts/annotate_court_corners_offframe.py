#!/usr/bin/env python3
"""Hand-annotate court corners when some corners sit beyond the visible frame.

Standalone sibling of ``annotate_court_corners.py``, built for footage where the
camera cuts off part of the court (common in amateur pulls). Corners are
captured in a fixed slot order and any slot can be marked off-frame instead of
clicked. Off-frame corners are then recovered by extrapolation: click four or
more painted line intersections (landmarks with known court coordinates, see
``court_landmarks.py``), fit the court-to-image homography, and the missing
corner is projected through it to its pixel position outside the image.

This file deliberately does not modify the original clicker; it imports its
pure helpers and keeps the same two-click loupe flow. Fold the two tools
together once the in-flight branches settle.

Slot order is TL TR BR BL, read on screen as far-left, far-right, near-right,
near-left. The court is symmetric end to end, so any consistent far/near
reading of a frame gives a valid homography.

The corners CSV is a superset of the original clicker's: the first eight
columns keep their names and meanings, so ``score_hand_corners.py`` reads these
files unchanged. Long form, one row per corner::

    video,frame,corner_idx,x_px,y_px,x_norm,y_norm,orientation,corner_label,visible,source,fit_rms_px

``corner_idx`` is the slot (0-3 = TL TR BR BL), ``corner_label`` names it,
``visible`` is 1/0, ``source`` is click or extrapolated, and ``fit_rms_px``
carries the landmark fit's rms residual on extrapolated rows. Extrapolated
``x_px``/``y_px`` (and the normalised columns) run outside the frame bounds by
design. Clicked landmarks are saved next to the corners CSV in
``<out-csv stem>_landmarks.csv``::

    video,frame,landmark,court_x_m,court_y_m,x_px,y_px

Annotation is resumable: the CSVs append across sessions, already-annotated
frames are flagged in the status bar, and committing a frame again replaces its
earlier rows (matched by video basename plus frame) instead of duplicating
them. Appending into a CSV written by a different tool or version is refused up
front, before any clicking happens.

Controls (a help panel opens at launch; ``h`` brings it back any time):

- trackbar, or ``,`` / ``.`` for -1 / +1 frame, ``<`` / ``>`` for -25 / +25
- ``c`` starts a capture on the current frame; corners are prompted in slot
  order, and ``x`` marks the prompted corner as off-frame
- placing a point: one click near it drops a provisional point and opens the
  loupe. Nothing locks until confirmed, so adjust freely: arrow keys nudge by
  ``--nudge-step`` px (``-`` / ``=`` halve / double the step), clicking in the
  loupe or on the video re-aims, and ``t`` takes typed "x y" source pixels.
  Enter or space confirms the point; ESC drops it and re-asks
- landmark mode (entered when any corner was off-frame, or always with
  ``--landmarks always``): ``n`` / ``p`` cycle the catalogue, points are placed
  the same way, ``f`` fits the homography, Enter at the prompt commits the frame
- ``j`` jumps to the next of ``--frames`` evenly spaced target frames, for
  spreading the annotations through the video; scrub locally if a target is
  blocked
- ``v`` copies the nearest annotated frame (static cameras): every corner and
  landmark arrives pre-aimed for an Enter-by-Enter confirm, off-frame marks
  re-apply, and the fit reruns on this frame's confirmed points
- mistakes: ``u`` undoes the last confirmed corner or landmark, stepping back
  out of landmark mode when none are left; ESC from a bare prompt abandons the
  frame; committing a frame again replaces its rows
- ``q`` quits

Usage::

    python src/courtkeynet/validation_scripts/annotate_court_corners_offframe.py \\
        --video path/to/clip.mp4 --out-csv hand_corners.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

# Sibling imports: this script runs from its own directory (python puts the
# script's dir on sys.path), and tests load it by file path, where __file__
# still resolves. The package __init__ pulls in torch, so the package route is
# avoided on purpose.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import court_landmarks as court  # noqa: E402
from annotate_court_corners import (  # noqa: E402
    CORNER_COLOUR,
    TEXT_COLOUR,
    loupe_origin,
    loupe_to_source,
    normalise_xy,
    read_frame,
    render_loupe,
)

CSV_HEADER = (
    "video", "frame", "corner_idx", "x_px", "y_px", "x_norm", "y_norm", "orientation",
    "corner_label", "visible", "source", "fit_rms_px",
)
LANDMARK_CSV_HEADER = ("video", "frame", "landmark", "court_x_m", "court_y_m", "x_px", "y_px")

# Sky blue from the Wong palette (#56B4E9 as BGR): landmarks read apart from the
# orange corner marks for protan vision.
LANDMARK_COLOUR = (233, 180, 86)


# --- Capture state machine (unit-tested) -----------------------------------

class CaptureState(StrEnum):
    """Where a capture is across the corner and landmark phases."""

    SCRUB = "scrub"  # free navigation, no capture in progress
    CORNER_COARSE = "corner_coarse"  # want a rough click or x for the prompted slot
    CORNER_ADJUST = "corner_adjust"  # provisional corner down; adjust, then confirm
    LANDMARK_COARSE = "landmark_coarse"  # want a rough click on the shown landmark
    LANDMARK_ADJUST = "landmark_adjust"  # provisional landmark down; adjust, then confirm


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
    """Slot-ordered corner capture with an off-frame escape hatch per corner.

    The corner phase prompts TL TR BR BL in turn; each is clicked (coarse then
    loupe refine, as in the original clicker) or marked off-frame with ``x``.
    When every slot is resolved the session either commits directly (all four
    clicked) or moves to the landmark phase, where clicked line intersections
    pin the homography that places the missing corners. The cv2 loop is a thin
    shell over this class.
    """

    def __init__(
        self,
        video: str,
        width: int,
        height: int,
        orientation: str,
        csv_path: Path,
        half: int,
        zoom: int,
        landmark_policy: str = "auto",
    ) -> None:
        """:param video: source video path string, stored on every CSV row
        :param width: frame width in pixels
        :param height: frame height in pixels
        :param orientation: orientation flag written to the CSV
        :param csv_path: corners CSV, appended to on each committed frame
        :param half: loupe half-window in source pixels
        :param zoom: loupe magnification
        :param landmark_policy: "auto" collects landmarks only when a corner is
            off-frame; "always" collects them on every frame (leave-one-out runs)
        """
        if landmark_policy not in ("auto", "always"):
            raise ValueError(f"landmark_policy must be auto or always, got {landmark_policy!r}")
        self.video = video
        self.video_basename = Path(str(video)).name
        self.width = width
        self.height = height
        self.orientation = orientation
        self.csv_path = Path(csv_path)
        self.landmark_csv_path = self.csv_path.with_name(self.csv_path.stem + "_landmarks.csv")
        # Refuse a foreign or older column set before any clicking happens.
        ensure_header(self.csv_path, CSV_HEADER)
        ensure_header(self.landmark_csv_path, LANDMARK_CSV_HEADER)
        self.annotated_frames = load_annotated_frames(self.csv_path, self.video_basename)
        self.half = half
        self.zoom = zoom
        self.landmark_policy = landmark_policy
        self.state = CaptureState.SCRUB
        self.frame_idx = -1
        # One entry per confirmed slot, slot order; None marks an off-frame corner.
        self._slots: list[tuple[float, float] | None] = []
        # The provisional point being adjusted, and its loupe crop origin.
        self._pending_point: tuple[float, float] | None = None
        self._pending_origin: tuple[int, int] | None = None
        # Confirmed landmarks as (name, x_px, y_px); replaced in place on a redo.
        self._landmarks: list[tuple[str, float, float]] = []
        self.cursor = 0  # index into court.LANDMARK_NAMES
        self.fit: FitResult | None = None
        # Copied-from-an-earlier-frame targets: one entry per slot (None = the
        # source corner was off-frame), and landmark aims consumed front to back.
        self._prefill_slots: list[tuple[float, float] | None] | None = None
        self._prefill_landmarks: list[tuple[str, float, float]] | None = None

    # --- read-only views for the shell and tests ---

    @property
    def slot(self) -> int:
        """:return: index of the corner slot currently being resolved (0-4)."""
        return len(self._slots)

    @property
    def slots(self) -> tuple[tuple[float, float] | None, ...]:
        """:return: resolved slots so far; None entries are off-frame corners."""
        return tuple(self._slots)

    @property
    def offframe_slots(self) -> tuple[int, ...]:
        """:return: slot indices marked off-frame."""
        return tuple(idx for idx, value in enumerate(self._slots) if value is None)

    @property
    def landmarks(self) -> tuple[tuple[str, float, float], ...]:
        """:return: clicked landmarks as (name, x_px, y_px), in click order."""
        return tuple(self._landmarks)

    @property
    def cursor_name(self) -> str:
        """:return: catalogue name the landmark cursor is on."""
        return court.LANDMARK_NAMES[self.cursor]

    def landmark_placed(self, name: str) -> bool:
        """:return: whether ``name`` already has a click recorded."""
        return any(existing == name for existing, _x, _y in self._landmarks)

    @property
    def pending_point(self) -> tuple[float, float] | None:
        """:return: the provisional point being adjusted, if any."""
        return self._pending_point

    @property
    def pending_origin(self) -> tuple[int, int] | None:
        """:return: the open loupe's crop origin, if a point is being adjusted."""
        return self._pending_origin

    @property
    def adjusting(self) -> bool:
        """:return: whether a provisional point is on the table."""
        return self.state in (CaptureState.CORNER_ADJUST, CaptureState.LANDMARK_ADJUST)

    # --- corner phase ---

    def begin_capture(self, frame_idx: int) -> SessionAction:
        """Start a fresh capture on ``frame_idx`` (only from scrub).

        :param frame_idx: the frame being annotated
        :return: REDRAW prompting for the TL corner, or NOOP if a capture is live
        """
        if self.state is not CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        self.frame_idx = frame_idx
        self._reset_capture()
        self.state = CaptureState.CORNER_COARSE
        prompt = self._corner_prompt()
        if frame_idx in self.annotated_frames:
            prompt = f"frame {frame_idx} is already annotated; committing replaces it. {prompt}"
        return SessionAction(ActionKind.REDRAW, message=prompt)

    def begin_capture_prefilled(
        self,
        frame_idx: int,
        slots: list[tuple[float, float] | None],
        landmarks: list[tuple[str, float, float]],
        source_frame: int,
    ) -> SessionAction:
        """Start a capture pre-aimed from an earlier frame's annotation.

        Made for static cameras: every corner (and landmark) arrives as a
        provisional point to confirm with Enter or adjust first, so a repeat
        frame takes a few keypresses while each value still passes under the
        annotator's eye. Source off-frame corners are re-marked automatically.

        :param frame_idx: the frame being annotated
        :param slots: 4 source corners in slot order; None for off-frame
        :param landmarks: source landmark clicks as (name, x_px, y_px)
        :param source_frame: where the copy came from, for the status message
        :return: the first pre-aimed point, or NOOP if a capture is live
        """
        if self.state is not CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        self.frame_idx = frame_idx
        self._reset_capture()
        self._prefill_slots = list(slots)
        self._prefill_landmarks = list(landmarks)
        self.state = CaptureState.CORNER_COARSE
        action = self._advance_corner()
        prefix = f"copied from frame {source_frame}; Enter confirms each point, or adjust first. "
        return SessionAction(action.kind, message=prefix + (action.message or ""),
                             loupe_origin=action.loupe_origin, coarse_xy=action.coarse_xy)

    def coarse_click(self, x: float, y: float) -> SessionAction:
        """Drop the provisional point at a main-window click and open the loupe.

        Valid from a coarse prompt (the first aim) and while adjusting: a second
        main-window click re-aims from scratch, loupe and all.

        :param x: click x in source pixels
        :param y: click y in source pixels
        :return: OPEN_LOUPE with the crop origin, or NOOP outside those states
        """
        if self.state is CaptureState.CORNER_COARSE:
            self.state = CaptureState.CORNER_ADJUST
        elif self.state is CaptureState.LANDMARK_COARSE:
            self.state = CaptureState.LANDMARK_ADJUST
        elif not self.adjusting:
            return SessionAction(ActionKind.NOOP)
        origin = loupe_origin(x, y, self.half, self.width, self.height)
        self._pending_origin = origin
        self._pending_point = (x, y)
        return SessionAction(
            ActionKind.OPEN_LOUPE, message=self._point_status(), loupe_origin=origin, coarse_xy=(x, y)
        )

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
        """Lock the provisional point in and move on (Enter or space).

        A corner fills the prompted slot, committing or entering landmark mode
        after the fourth. A landmark is recorded under the cursor's name,
        replacing an earlier take of the same crossing.

        :return: the advance action, or NOOP when nothing is pending
        """
        if not self.adjusting:
            return SessionAction(ActionKind.NOOP)
        assert self._pending_point is not None
        point = self._pending_point
        self._clear_pending()
        if self.state is CaptureState.CORNER_ADJUST:
            self._slots.append(point)
            return self._advance_corner()
        result = self._record_landmark(point)
        if self._prefill_landmarks:
            return self._offer_prefill_landmark()
        return result

    def cancel_aim(self) -> SessionAction:
        """Throw away the provisional point and re-ask for the same target (ESC).

        :return: REDRAW re-prompting the same target, or NOOP when nothing is pending
        """
        if self.state is CaptureState.CORNER_ADJUST:
            self.state = CaptureState.CORNER_COARSE
            message = f"cancelled. {self._corner_prompt()}"
        elif self.state is CaptureState.LANDMARK_ADJUST:
            self.state = CaptureState.LANDMARK_COARSE
            message = f"cancelled. click {self.cursor_name} again, or n/p to pick another"
        else:
            return SessionAction(ActionKind.NOOP)
        self._clear_pending()
        return SessionAction(ActionKind.REDRAW, message=message)

    def _point_status(self) -> str:
        assert self._pending_point is not None
        x, y = self._pending_point
        return f"{self._target_name()} at ({x:.2f}, {y:.2f}): arrows/click adjust, t types x y, Enter confirms"

    def _target_name(self) -> str:
        if self.state is CaptureState.CORNER_ADJUST:
            return court.CORNER_NAMES[self.slot].upper()
        return self.cursor_name

    def _clear_pending(self) -> None:
        self._pending_point = None
        self._pending_origin = None

    def mark_offframe(self) -> SessionAction:
        """Mark the prompted corner as extending beyond the visible frame.

        :return: REDRAW (next prompt or landmark mode), or NOOP outside a corner prompt
        """
        if self.state is not CaptureState.CORNER_COARSE:
            return SessionAction(ActionKind.NOOP)
        self._slots.append(None)
        return self._advance_corner()

    def _advance_corner(self) -> SessionAction:
        """Move to the next slot, or out of the corner phase after the fourth."""
        if self.slot < 4:
            if self._prefill_slots is not None:
                return self._offer_prefill_corner()
            self.state = CaptureState.CORNER_COARSE
            return SessionAction(ActionKind.REDRAW, message=self._corner_prompt())
        if not self.offframe_slots and self.landmark_policy == "auto":
            return self._commit()
        self.state = CaptureState.LANDMARK_COARSE
        seeded = self._seed_corner_landmarks()
        if self._prefill_landmarks:
            return self._offer_prefill_landmark()
        if self.offframe_slots:
            missing = ", ".join(court.CORNER_NAMES[idx] for idx in self.offframe_slots)
            note = f"off-frame: {missing}. "
        else:
            note = ""
        if seeded:
            note += f"your {seeded} clicked corners already count as landmarks. "
        return SessionAction(
            ActionKind.REDRAW,
            message=f"{note}landmark mode: n/p cycle, click to place, u undo, f fit, Enter commit",
        )

    def _seed_corner_landmarks(self) -> int:
        """Feed the clicked corners into the fit under their crossing names.

        The outer corners are catalogue crossings, so a corner the annotator
        already pinned through the loupe is a ready-made correspondence; asking
        for it again in landmark mode would be duplicate work. Only done when a
        corner is off-frame: on all-visible leave-one-out frames the fit must
        stay independent of the corner clicks it is measured against.

        :return: how many corners were added
        """
        if not self.offframe_slots:
            return 0
        seeded = 0
        for slot, clicked in enumerate(self._slots):
            if clicked is not None:
                self._landmarks.append((court.CORNER_LANDMARK_NAMES[slot], clicked[0], clicked[1]))
                seeded += 1
        if seeded:
            self._advance_cursor_to_unplaced()  # open on a crossing still needed
        return seeded

    def _offer_prefill_corner(self) -> SessionAction:
        """Aim the next copied corner; copied off-frame marks re-apply silently."""
        assert self._prefill_slots is not None
        value = self._prefill_slots[self.slot]
        if value is None:
            self._slots.append(None)
            return self._advance_corner()
        self.state = CaptureState.CORNER_ADJUST
        self._pending_point = value
        self._pending_origin = loupe_origin(value[0], value[1], self.half, self.width, self.height)
        return SessionAction(ActionKind.OPEN_LOUPE, message=self._point_status(),
                             loupe_origin=self._pending_origin, coarse_xy=value)

    def _offer_prefill_landmark(self) -> SessionAction:
        """Aim the next copied landmark under its own catalogue name."""
        assert self._prefill_landmarks
        name, x_px, y_px = self._prefill_landmarks.pop(0)
        self.cursor = court.LANDMARK_NAMES.index(name)
        self.state = CaptureState.LANDMARK_ADJUST
        self._pending_point = (x_px, y_px)
        self._pending_origin = loupe_origin(x_px, y_px, self.half, self.width, self.height)
        return SessionAction(ActionKind.OPEN_LOUPE, message=self._point_status(),
                             loupe_origin=self._pending_origin, coarse_xy=(x_px, y_px))

    def _corner_prompt(self) -> str:
        """:return: status text asking for the current corner slot."""
        name = court.CORNER_NAMES[self.slot].upper()
        description = court.CORNER_DESCRIPTIONS[self.slot]
        return f"click the {name} corner ({description}), or x if it is off-frame"

    # --- landmark phase ---

    def next_landmark(self) -> SessionAction:
        """Advance the catalogue cursor. :return: REDRAW, or NOOP outside landmark mode."""
        return self._move_cursor(+1)

    def prev_landmark(self) -> SessionAction:
        """Step the catalogue cursor back. :return: REDRAW, or NOOP outside landmark mode."""
        return self._move_cursor(-1)

    def _move_cursor(self, step: int) -> SessionAction:
        if self.state is not CaptureState.LANDMARK_COARSE:
            return SessionAction(ActionKind.NOOP)
        self.cursor = (self.cursor + step) % len(court.LANDMARK_NAMES)
        return SessionAction(ActionKind.REDRAW)

    def _record_landmark(self, point: tuple[float, float]) -> SessionAction:
        """Record a confirmed point against the cursor's landmark, replacing a redo."""
        name = self.cursor_name
        replaced = self.landmark_placed(name)
        self._landmarks = [entry for entry in self._landmarks if entry[0] != name]
        self._landmarks.append((name, point[0], point[1]))
        self.fit = None  # the landmark set changed, so any fit is stale
        self.state = CaptureState.LANDMARK_COARSE
        self._advance_cursor_to_unplaced()
        verb = "replaced" if replaced else "recorded"
        return SessionAction(
            ActionKind.REDRAW, message=f"{verb} {name} ({len(self._landmarks)} landmarks; 4 needed to fit)"
        )

    def _advance_cursor_to_unplaced(self) -> None:
        """Park the cursor on the next catalogue entry without a click, if any."""
        total = len(court.LANDMARK_NAMES)
        for step in range(1, total + 1):
            candidate = (self.cursor + step) % total
            if not self.landmark_placed(court.LANDMARK_NAMES[candidate]):
                self.cursor = candidate
                return

    def undo_last(self) -> SessionAction:
        """Step one action backwards: the last landmark, else the last corner.

        In landmark mode this drops the most recent landmark click; once none
        are left it steps back out of landmark mode and re-prompts the last
        corner. In the corner phase it re-prompts the last resolved corner,
        whether clicked or marked off-frame.

        :return: REDRAW describing what was undone (or that nothing could be),
            NOOP outside a capture
        """
        if self.adjusting:
            # One-key parity with the tour tool: a provisional point cancels
            # first, then confirmed points start popping.
            return self.cancel_aim()
        if self.state is CaptureState.LANDMARK_COARSE and self._landmarks:
            name, _x, _y = self._landmarks.pop()
            self.fit = None
            self.cursor = court.LANDMARK_NAMES.index(name)  # point back at what was undone
            return SessionAction(ActionKind.REDRAW, message=f"undid {name} ({len(self._landmarks)} landmarks)")
        if self.state in (CaptureState.LANDMARK_COARSE, CaptureState.CORNER_COARSE):
            if not self._slots:
                return SessionAction(ActionKind.REDRAW, message="nothing to undo")
            undone = self._slots.pop()
            self.fit = None
            self.state = CaptureState.CORNER_COARSE
            what = "off-frame mark" if undone is None else "corner"
            return SessionAction(ActionKind.REDRAW, message=f"undid {what}. {self._corner_prompt()}")
        return SessionAction(ActionKind.NOOP)

    def do_fit(self) -> SessionAction:
        """Fit the homography from the clicked landmarks, gated, and project corners.

        Refusals (fit stays None): under the point floor, a degenerate spread, a
        worst-point reprojection over the calibrated ceiling, or a projected quad
        that is not a convex behind-baseline court. A warn-level reprojection
        keeps the fit but says so; Enter then commits deliberately.

        :return: REDRAW with the fit summary (or the reason it refused),
            or NOOP outside landmark mode
        """
        if self.state is not CaptureState.LANDMARK_COARSE:
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
            for idx in self.offframe_slots
        )
        summary = f"fit: {self.fit.n_landmarks} landmarks, rms {self.fit.rms_px:.2f} px, max {self.fit.max_px:.2f} px"
        if placed:
            summary += f"; projected {placed}"
        if verdict.level == "warn":
            return SessionAction(ActionKind.REDRAW,
                                 message=f"fit WARN: {verdict.reason}. {summary}. Enter commits anyway")
        return SessionAction(ActionKind.REDRAW, message=summary + ". Enter commits")

    def commit(self) -> SessionAction:
        """Commit the frame from landmark mode (Enter).

        :return: COMMITTED once every off-frame corner has a fit behind it, REDRAW
            with the blocker when it does not, NOOP outside landmark mode
        """
        if self.state is not CaptureState.LANDMARK_COARSE:
            return SessionAction(ActionKind.NOOP)
        if self.offframe_slots and self.fit is None:
            return SessionAction(ActionKind.REDRAW, message="press f to fit before committing")
        return self._commit()

    # --- shared tail ---

    def _commit(self) -> SessionAction:
        """Write the corner rows (and any landmark rows), then reset to scrub.

        Rows for this (video basename, frame) replace any earlier ones, so a
        frame can be re-annotated across sessions without duplicating rows. The
        landmark sidecar is kept in step: a re-annotation with no landmarks also
        clears the frame's stale sidecar rows.
        """
        rows = build_slot_rows(
            self.video, self.frame_idx, self._slots, self.fit, self.width, self.height, self.orientation
        )
        replaced = upsert_csv(self.csv_path, rows, CSV_HEADER, self.video_basename, self.frame_idx)
        landmark_rows = build_landmark_rows(self.video, self.frame_idx, self._landmarks)
        if landmark_rows or self.landmark_csv_path.exists():
            upsert_csv(self.landmark_csv_path, landmark_rows, LANDMARK_CSV_HEADER,
                       self.video_basename, self.frame_idx)
        n_landmarks = len(self._landmarks)
        n_extrapolated = len(self.offframe_slots)
        frame_idx = self.frame_idx
        self.annotated_frames.add(frame_idx)
        self._reset_capture()
        self.state = CaptureState.SCRUB
        verb = "replaced" if replaced else "saved"
        message = f"{verb} frame {frame_idx}: 4 corners ({n_extrapolated} extrapolated)"
        if n_landmarks:
            message += f", {n_landmarks} landmarks"
        return SessionAction(ActionKind.COMMITTED, message=message)

    def abort(self) -> SessionAction:
        """Cancel the in-progress capture and drop back to scrub, writing nothing.

        :return: ABORTED when a capture was live, else NOOP
        """
        if self.state is CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        self._reset_capture()
        self.state = CaptureState.SCRUB
        return SessionAction(ActionKind.ABORTED, message="capture aborted")

    def _reset_capture(self) -> None:
        self._slots = []
        self._clear_pending()
        self._landmarks = []
        self.cursor = 0
        self.fit = None
        self._prefill_slots = None
        self._prefill_landmarks = None


# --- Pure helpers: CSV rows (unit-tested) -----------------------------------

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
    :param slots: 4 entries in slot order TL TR BR BL; None means off-frame
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
        raise ValueError("off-frame slots need a landmark fit to extrapolate from")
    rows: list[dict[str, object]] = []
    for slot, clicked in enumerate(slots):
        if clicked is None:
            assert fit is not None  # guarded above
            x_px, y_px = float(fit.corners_px[slot, 0]), float(fit.corners_px[slot, 1])
            visible, source, fit_rms = 0, "extrapolated", f"{fit.rms_px:.3f}"
        else:
            x_px, y_px = clicked
            visible, source, fit_rms = 1, "click", ""
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


# --- cv2 shell (thin driver over the session; not unit-tested) -------------

MAIN_WINDOW = "annotate court corners (off-frame capable)"
LOUPE_WINDOW = "refine (click the corner)"

KEY_PREV = ord(",")
KEY_NEXT = ord(".")
KEY_PREV_JUMP = ord("<")
KEY_NEXT_JUMP = ord(">")
KEY_CAPTURE = ord("c")
KEY_OFFFRAME = ord("x")
KEY_LANDMARK_NEXT = ord("n")
KEY_LANDMARK_PREV = ord("p")
KEY_UNDO = ord("u")
KEY_FIT = ord("f")
KEY_HELP = ord("h")
KEY_JUMP = ord("j")
KEY_COPY = ord("v")
KEY_TYPE = ord("t")
KEY_STEP_DOWN = ord("-")
KEY_STEP_UP = ord("=")
KEY_QUIT = ord("q")
KEY_ESC = 27
KEY_SPACE = 32
KEYS_ENTER = (13, 10)
KEYS_BACKSPACE = (8, 127)
TYPING_CHARS = "0123456789. ,-"
# Arrow keycodes after the & 0xFF mask (GTK and Qt HighGUI agree on the low byte).
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 81, 82, 83, 84
JUMP = 25

# A double-click's second press can land inside the loupe the first press just
# opened, pinning a point the user never chose. Clicks arriving this soon after
# the loupe opens are the double-click's echo, not a deliberate click. (Grafted
# from the tour tool, along with the beside-the-window loupe placement.)
LOUPE_CLICK_GUARD_S = 0.35

# Initial main-window cap (the frame still renders full-resolution inside a
# resizable window). Without the cap a 1920-wide video fills the screen and the
# beside-the-window loupe placement lands off-screen, where the WM dumps it
# anywhere, including behind. Grafted from the tour tool, which the loupe
# placement was calibrated against.
INIT_MAX_W = 1280
INIT_MAX_H = 720


def window_visible(name: str) -> bool:
    """:return: True when the named cv2 window is open and visible.

    Closing the LAST window via its X tears down Qt's GUI thread, and the
    property query then raises cv2.error instead of returning 0; that raise
    crashed the tour tool on quit, so it is caught and folded into "not
    visible". (Grafted from the tour tool.)

    :param name: the cv2 window name to query
    """
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False

HELP_LINES = (
    "c  start capture on this frame      q  quit",
    ", .  step 1 frame    < >  step 25    or drag the trackbar",
    "corner prompts (TL TR BR BL = far-left, far-right, near-right, near-left):",
    "   one click near the corner drops the point and opens the zoom window",
    "   x  the prompted corner is off-frame      u  undo the last corner",
    "adjust the dropped point until happy, nothing locks until you confirm:",
    "   arrows nudge (-/= step size), click in zoom or video re-aims,",
    "   t type exact x y, Enter or space CONFIRMS, ESC drops the point",
    "landmark mode:  n/p  cycle the named crossing, place points the same way",
    "   u  undo the last landmark    f  fit    Enter (at prompt)  commit frame",
    "j  jump to the next target frame (--frames evenly spaced per video)",
    "v  copy the nearest annotated frame: points arrive pre-aimed, Enter each",
    "ESC from a bare prompt abandons the frame",
    "the toolbar up top is OpenCV's own: its disk icon saves a SCREENSHOT",
    "   of the view, not your annotations",
    "press any key to close this help    h brings it back",
)

STRIP_HEIGHT = 44  # two status lines, at the 1920-wide reference


def hud_factor(width: int) -> float:
    """HUD size multiplier for a frame width.

    Text metrics below are calibrated at 1920 wide; fixed sizes become
    unreadable when the video is narrower or the window smaller (a tour-tool
    lesson). Clamped so tiny test frames stay legible and 4K does not shout.

    :param width: frame width in px
    """
    return min(max(width / 1920, 0.75), 2.0)
# cv2 draws int32 coordinates; a degenerate fit can project corners to absurd
# values, so drawn points are clamped well inside int32 while the CSV keeps the
# real numbers.
DRAW_CLAMP_PX = 100_000


COURT_KEY_SCALE = 20  # px per court metre in the in-window court map


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
        return (int(round(ox + court_x * scale)), int(round(oy + court_y * scale)))

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


def status_lines(session: OffframeSession, frame_idx: int, total: int, message: str) -> tuple[str, str]:
    """Compose the two status-strip lines.

    :param session: the live session
    :param frame_idx: current frame index
    :param total: total frames
    :param message: the most recent session message
    :return: (top line, bottom line)
    """
    done = " [annotated]" if frame_idx in session.annotated_frames else ""
    top = f"frame {frame_idx}/{total - 1}{done}   [{session.state}]   corners {session.slot}/4"
    if session.state in (CaptureState.LANDMARK_COARSE, CaptureState.LANDMARK_ADJUST):
        placed = " (placed)" if session.landmark_placed(session.cursor_name) else ""
        top += (
            f"   landmarks {len(session.landmarks)}"
            f"   at {session.cursor + 1}/{len(court.LANDMARK_NAMES)}: {session.cursor_name}{placed}"
        )
    return top, message


def draw_overlay(
    frame: np.ndarray,
    session: OffframeSession,
    frame_idx: int,
    total: int,
    message: str,
    show_help: bool = False,
    factor: float | None = None,
) -> np.ndarray:
    """Draw the status strip, placed corners, landmarks, and any fitted quad.

    :param frame: the current BGR frame
    :param session: the live session
    :param frame_idx: current frame index
    :param total: total frames, for the strip
    :param message: the most recent session message
    :param show_help: draw the key reference panel under the strip
    :param factor: HUD size multiplier; None derives it from the frame width.
        The shell passes one compensated for the window's display scale, since
        the frame renders full-resolution into a downscaled window
    :return: a copy of the frame with the overlay drawn
    """
    canvas = frame.copy()
    if session.fit is not None:
        # The fitted court outline; cv2 clips edges that run beyond the frame.
        quad = np.clip(session.fit.corners_px, -DRAW_CLAMP_PX, DRAW_CLAMP_PX)
        points = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], isClosed=True, color=LANDMARK_COLOUR, thickness=2, lineType=cv2.LINE_AA)
    for slot, clicked in enumerate(session.slots):
        if clicked is None:
            continue
        centre = (int(round(clicked[0])), int(round(clicked[1])))
        cv2.circle(canvas, centre, 4, CORNER_COLOUR, -1, cv2.LINE_AA)
        cv2.putText(canvas, court.CORNER_NAMES[slot], (centre[0] + 6, centre[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORNER_COLOUR, 1, cv2.LINE_AA)
    for index, (_name, x_px, y_px) in enumerate(session.landmarks):
        centre = (int(round(x_px)), int(round(y_px)))
        cv2.circle(canvas, centre, 3, LANDMARK_COLOUR, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), (centre[0] + 5, centre[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, LANDMARK_COLOUR, 1, cv2.LINE_AA)
    if session.pending_point is not None:
        px, py = (int(round(value)) for value in session.pending_point)
        cv2.drawMarker(canvas, (px, py), CORNER_COLOUR, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    if factor is None:
        factor = hud_factor(canvas.shape[1])
    strip_height = int(STRIP_HEIGHT * factor)
    if session.state is not CaptureState.SCRUB:
        # Court map in the top-right corner: which crossing or corner is wanted.
        key_width = int(court.COURT_WIDTH_M * COURT_KEY_SCALE)
        key_height = int(court.COURT_LENGTH_M * COURT_KEY_SCALE)
        key_x = canvas.shape[1] - key_width - 30
        key_y = strip_height + 30
        if key_x > 0 and key_y + key_height + 20 < canvas.shape[0]:
            in_landmarks = session.state in (CaptureState.LANDMARK_COARSE, CaptureState.LANDMARK_ADJUST)
            draw_court_key(
                canvas, (key_x, key_y), COURT_KEY_SCALE,
                highlight=session.cursor_name if in_landmarks else None,
                placed=tuple(name for name, _x, _y in session.landmarks),
                corner_slot=None if in_landmarks else min(session.slot, 3),
            )
    top, bottom = status_lines(session, frame_idx, total, message)
    font = 0.45 * factor
    thickness = 1 if factor < 1.5 else 2
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], strip_height), (0, 0, 0), -1)
    cv2.putText(canvas, top, (6, int(16 * factor)), cv2.FONT_HERSHEY_SIMPLEX, font,
                TEXT_COLOUR, thickness, cv2.LINE_AA)
    cv2.putText(canvas, bottom, (6, int(36 * factor)), cv2.FONT_HERSHEY_SIMPLEX, font,
                TEXT_COLOUR, thickness, cv2.LINE_AA)
    if show_help:
        line_height = int(18 * factor)
        panel_height = line_height * len(HELP_LINES) + int(12 * factor)
        panel_width = int(620 * factor)
        cv2.rectangle(canvas, (0, strip_height), (panel_width, strip_height + panel_height), (0, 0, 0), -1)
        for line_idx, line in enumerate(HELP_LINES):
            cv2.putText(canvas, line, (6, strip_height + int(22 * factor) + line_height * line_idx),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42 * factor, TEXT_COLOUR, thickness, cv2.LINE_AA)
    return canvas


def run_annotation_tool(args: argparse.Namespace) -> int:
    """Drive the scrub/capture loop over the cv2 windows and the OffframeSession.

    :param args: parsed CLI namespace
    :return: process exit code
    """
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        session = OffframeSession(
            str(args.video), width, height, args.orientation,
            Path(args.out_csv), args.loupe_size, args.loupe_zoom,
            landmark_policy=args.landmarks,
        )
    except ValueError as error:
        sys.exit(f"ERROR: {error}")
    # Mouse callbacks push their SessionAction here; the main loop drains it so
    # window creation/teardown stays on the loop thread.
    pending: list[SessionAction] = []
    frame_state = {"idx": max(0, min(args.start_frame, total - 1)), "frame": None}

    def load(idx: int) -> None:
        # Frozen during a capture, exactly as in the original clicker: clicks are
        # recorded against the frame the capture began on. ESC first, then scrub.
        if session.state is not CaptureState.SCRUB:
            return
        frame_state["idx"] = max(0, min(idx, total - 1))
        frame_state["frame"] = read_frame(cap, frame_state["idx"])

    help_state = {"visible": True}  # shown at launch; any key or click dismisses it
    loupe_state = {"opened_at": 0.0, "seen": False}

    def on_main_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if help_state["visible"]:
            help_state["visible"] = False  # a click dismisses the help; nothing else fires
            return
        # A main-window click aims the point; while adjusting it re-aims.
        if session.adjusting or session.state in (CaptureState.CORNER_COARSE, CaptureState.LANDMARK_COARSE):
            pending.append(session.coarse_click(float(x), float(y)))

    def on_loupe_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        # A loupe click repositions the provisional point; it confirms nothing.
        if event != cv2.EVENT_LBUTTONDOWN or not session.adjusting:
            return
        if time.monotonic() - loupe_state["opened_at"] < LOUPE_CLICK_GUARD_S:
            return  # a double-click's second press landing inside the just-opened loupe
        pending.append(session.reposition_from_loupe(float(x), float(y)))

    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    window_scale = min(1.0, INIT_MAX_W / width, INIT_MAX_H / height)
    cv2.resizeWindow(MAIN_WINDOW, int(width * window_scale), int(height * window_scale))
    cv2.setMouseCallback(MAIN_WINDOW, on_main_mouse)
    cv2.createTrackbar("frame", MAIN_WINDOW, frame_state["idx"], max(total - 1, 1), lambda idx: load(idx))
    load(frame_state["idx"])
    print(f"{args.video}: {width}x{height}, {total} frames.")
    print("  a help panel is open in the window; press h any time to bring it back")
    if session.annotated_frames:
        done = ", ".join(str(idx) for idx in sorted(session.annotated_frames))
        print(f"  resuming: frame(s) {done} already annotated for this video; re-committing one replaces it")

    # Evenly spaced capture targets for the j key; midpoints dodge intros and
    # outros at the ends of the video.
    n_targets = max(1, args.frames)
    anchors = sorted({min(total - 1, max(0, round((i + 0.5) * total / n_targets))) for i in range(n_targets)})

    def show_loupe(target_xy: tuple[float, float] | None, origin: tuple[int, int] | None, banner: str) -> None:
        # Reuses the base clicker's renderer; the crosshair tracks the placed
        # point, so nudges are visible at loupe magnification. A banner strip
        # says what the window wants, since it covers the main status bar. The
        # window itself is created and positioned by the open_loupe_window step.
        if frame_state["frame"] is None or origin is None:
            return
        synthetic = SessionAction(ActionKind.OPEN_LOUPE, loupe_origin=origin, coarse_xy=target_xy)
        zoomed = render_loupe(frame_state["frame"], synthetic, args.loupe_size, args.loupe_zoom)
        cv2.rectangle(zoomed, (0, zoomed.shape[0] - 24), (zoomed.shape[1], zoomed.shape[0]), (0, 0, 0), -1)
        cv2.putText(zoomed, banner, (6, zoomed.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    TEXT_COLOUR, 1, cv2.LINE_AA)
        cv2.imshow(LOUPE_WINDOW, zoomed)

    def open_loupe_window() -> None:
        # Create, position, and pin the loupe BEFORE its first show, so it never
        # flashes at a WM-default spot or spawns behind the main window. It sits
        # over the main window's LEFT side: the court map lives top-right, and
        # over-the-window can never land off-screen.
        cv2.namedWindow(LOUPE_WINDOW)
        cv2.setMouseCallback(LOUPE_WINDOW, on_loupe_mouse)
        try:
            main_x, main_y, main_w, _ = cv2.getWindowImageRect(MAIN_WINDOW)
            good_rect = main_w > 0
        except cv2.error:
            good_rect = False
        if good_rect:
            cv2.moveWindow(LOUPE_WINDOW, main_x + 16, max(main_y + 16, 0))
        else:
            cv2.moveWindow(LOUPE_WINDOW, 24, 24)
        try:
            # It overlaps the main window on purpose, so it must actually stay
            # in front; some WMs spawn unfocused windows behind.
            cv2.setWindowProperty(LOUPE_WINDOW, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    typing_state = {"active": False, "buffer": ""}
    step_state = {"px": args.nudge_step}

    def adjust_banner() -> str:
        if typing_state["active"]:
            return f"type x y: {typing_state['buffer']}_   (Enter applies, ESC cancels)"
        if session.state is CaptureState.CORNER_ADJUST:
            target = court.CORNER_NAMES[session.slot].upper()
        else:
            target = session.cursor_name
        return (f"{target}: arrows nudge {step_state['px']:g}px (-/= step), click re-aims, "
                f"t types, Enter/space CONFIRMS, ESC drops")

    def handle_typing(key: int) -> None:
        # Collect "x y" (or "x,y") in source pixels while typing mode is on.
        if key in KEYS_ENTER:
            typing_state["active"] = False
            parts = typing_state["buffer"].replace(",", " ").split()
            if len(parts) == 2:
                try:
                    pending.append(session.set_point(float(parts[0]), float(parts[1])))
                    return
                except ValueError:
                    pass
            pending.append(SessionAction(
                ActionKind.ADJUSTED, message=f"could not read x y from '{typing_state['buffer']}'"
            ))
        elif key == KEY_ESC:
            typing_state["active"] = False
            pending.append(SessionAction(ActionKind.ADJUSTED, message="typing cancelled"))
        elif key in KEYS_BACKSPACE:
            typing_state["buffer"] = typing_state["buffer"][:-1]
            pending.append(SessionAction(ActionKind.ADJUSTED))
        elif 0 <= key < 256 and chr(key) in TYPING_CHARS:
            typing_state["buffer"] += chr(key)
            pending.append(SessionAction(ActionKind.ADJUSTED))

    message = "c starts a capture; h shows the keys"
    main_seen = False
    loupe_open = False
    while True:
        frame = frame_state["frame"]
        if frame is None:
            # Never leave the window unpainted: a decode failure still gets a
            # canvas with the status strip, so the UI reads as alive and the
            # scrub keys visibly work.
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            message = f"frame {frame_state['idx']} failed to decode; scrub with , . or the trackbar"
        # HUD sizes compensate for the window's live display scale: the frame
        # renders full-resolution into a (usually) downscaled resizable window.
        try:
            _wx, _wy, shown_width, _wh = cv2.getWindowImageRect(MAIN_WINDOW)
        except cv2.error:
            shown_width = 0
        display_scale = shown_width / width if shown_width > 0 else window_scale
        factor = min(max(hud_factor(width) / max(display_scale, 0.2), 0.75), 3.0)
        cv2.imshow(MAIN_WINDOW,
                   draw_overlay(frame, session, frame_state["idx"], total, message,
                                help_state["visible"], factor=factor))

        while pending:
            action = pending.pop(0)
            if action.message:
                message = action.message
                if action.kind is not ActionKind.ADJUSTED:  # arrow-key repeats would spam the console
                    print(f"  {action.message}")
            if action.kind is ActionKind.OPEN_LOUPE:
                if not loupe_open:
                    open_loupe_window()
                loupe_open = True
                loupe_state["opened_at"] = time.monotonic()  # arms the double-click guard
                loupe_state["seen"] = False  # re-arm closed-detection for this loupe
                show_loupe(session.pending_point, session.pending_origin, adjust_banner())
            elif action.kind is ActionKind.ADJUSTED and loupe_open:
                show_loupe(session.pending_point, session.pending_origin, adjust_banner())
            elif action.kind is not ActionKind.NOOP and loupe_open:
                cv2.destroyWindow(LOUPE_WINDOW)
                loupe_open = False

        key = cv2.waitKey(20) & 0xFF
        # Not-visible only means "user closed it" AFTER the window has reported
        # visible once: a busy WM can read a not-yet-mapped window as closed,
        # and quitting on that reading kills the tool at launch.
        main_visible = window_visible(MAIN_WINDOW)
        main_seen = main_seen or main_visible
        if main_seen and not main_visible:
            break  # the main window's X quits; otherwise the next imshow reopens it
        if loupe_open:
            loupe_visible = window_visible(LOUPE_WINDOW)
            loupe_state["seen"] = loupe_state["seen"] or loupe_visible
            if loupe_state["seen"] and not loupe_visible:
                # The loupe's X, not a click: drop the aim like ESC would.
                loupe_open = False
                pending.append(session.cancel_aim())
        if typing_state["active"]:
            handle_typing(key)
        elif help_state["visible"]:
            if key != 255:
                # Any key closes the help panel, and fires nothing else on this
                # pass; the panel promises "press any key".
                help_state["visible"] = False
        elif key == KEY_QUIT:
            break
        elif key == KEY_ESC:
            # ESC steps back one level: drop a provisional point; from a bare
            # prompt or scrub it abandons the whole frame.
            pending.append(session.cancel_aim() if session.adjusting else session.abort())
        elif key in (KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN):
            dx = {KEY_LEFT: -step_state["px"], KEY_RIGHT: step_state["px"]}.get(key, 0.0)
            dy = {KEY_UP: -step_state["px"], KEY_DOWN: step_state["px"]}.get(key, 0.0)
            pending.append(session.nudge(dx, dy))
        elif key == KEY_SPACE:
            pending.append(session.confirm_point())
        elif key == KEY_TYPE and session.adjusting:
            typing_state["active"] = True
            typing_state["buffer"] = ""
            pending.append(SessionAction(ActionKind.ADJUSTED, message="type the exact x y, Enter applies"))
        elif key in (KEY_STEP_DOWN, KEY_STEP_UP):
            factor = 0.5 if key == KEY_STEP_DOWN else 2.0
            step_state["px"] = min(max(step_state["px"] * factor, 0.05), 8.0)
            pending.append(SessionAction(ActionKind.ADJUSTED, message=f"nudge step {step_state['px']:g} px"))
        elif key == KEY_HELP:
            help_state["visible"] = True
        elif key == KEY_JUMP and session.state is CaptureState.SCRUB:
            target = next((anchor for anchor in anchors if anchor > frame_state["idx"]), anchors[0])
            load(target)
            cv2.setTrackbarPos("frame", MAIN_WINDOW, frame_state["idx"])
            message = (f"target {anchors.index(target) + 1}/{len(anchors)} at frame {target}: "
                       f"scrub if blocked, c to capture, v to copy the nearest annotation")
        elif key == KEY_COPY and session.state is CaptureState.SCRUB:
            prefill = load_prefill(session.csv_path, session.landmark_csv_path,
                                   session.video_basename, frame_state["idx"])
            if prefill is None:
                message = "nothing to copy yet: annotate one frame by hand first"
            else:
                source_frame, slots, landmarks = prefill
                pending.append(session.begin_capture_prefilled(frame_state["idx"], slots, landmarks, source_frame))
        elif key == KEY_CAPTURE:
            pending.append(session.begin_capture(frame_state["idx"]))
        elif key == KEY_OFFFRAME:
            pending.append(session.mark_offframe())
        elif key == KEY_LANDMARK_NEXT:
            pending.append(session.next_landmark())
        elif key == KEY_LANDMARK_PREV:
            pending.append(session.prev_landmark())
        elif key == KEY_UNDO:
            pending.append(session.undo_last())
        elif key == KEY_FIT:
            pending.append(session.do_fit())
        elif key in KEYS_ENTER:
            # Enter confirms the point being adjusted; from the landmark prompt
            # (nothing pending) it commits the frame.
            pending.append(session.confirm_point() if session.adjusting else session.commit())
        elif key == KEY_PREV:
            load(frame_state["idx"] - 1)
            cv2.setTrackbarPos("frame", MAIN_WINDOW, frame_state["idx"])
        elif key == KEY_NEXT:
            load(frame_state["idx"] + 1)
            cv2.setTrackbarPos("frame", MAIN_WINDOW, frame_state["idx"])
        elif key == KEY_PREV_JUMP:
            load(frame_state["idx"] - JUMP)
            cv2.setTrackbarPos("frame", MAIN_WINDOW, frame_state["idx"])
        elif key == KEY_NEXT_JUMP:
            load(frame_state["idx"] + JUMP)
            cv2.setTrackbarPos("frame", MAIN_WINDOW, frame_state["idx"])

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True,
                        help="Appended to; header written when new. Landmarks go to <stem>_landmarks.csv.")
    parser.add_argument("--orientation", type=str, default="portrait", help="Orientation flag stored per row.")
    parser.add_argument("--landmarks", choices=("auto", "always"), default="auto",
                        help="auto: landmark mode only when a corner is off-frame; always: every frame "
                             "(for leave-one-out checks with check_extrapolation.py).")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame to open on (default 0).")
    parser.add_argument("--frames", type=int, default=2,
                        help="Target annotated frames per video; j jumps between that many evenly "
                             "spaced spots (default 2).")
    parser.add_argument("--nudge-step", type=float, default=0.5,
                        help="Arrow-key nudge in source px while a placed point's loupe is open (default 0.5).")
    parser.add_argument("--loupe-size", type=int, default=64,
                        help="Loupe half-window in source pixels (crop side is twice this; default 64).")
    parser.add_argument("--loupe-zoom", type=int, default=8, help="Loupe magnification with NEAREST (default 8).")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"ERROR: --video does not exist: {args.video}")
    try:
        return run_annotation_tool(args)
    except KeyboardInterrupt:
        # Ctrl-C in the terminal instead of q in the window; nothing is lost,
        # every committed frame is already on disk.
        print("\ninterrupted; committed frames are already saved")
        return 130


if __name__ == "__main__":
    sys.exit(main())
