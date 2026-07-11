#!/usr/bin/env python3
"""Hand-annotate the four court corners on chosen video frames.

Footage with no recorded homography (amateur pulls, freshly downloaded clips)
has no ground truth to score CourtKeyNet against. This tool closes that gap: an
OpenCV window scrubs the video, and on any frame you like you click the four
court corners and the annotation is appended to a CSV. ``score_hand_corners.py``
then scores the detector against that CSV.

Precision comes from a two-click-per-corner flow. A coarse click anywhere near
the corner opens a magnified loupe (a small source-pixel window scaled up with
NEAREST interpolation so the pixels stay blocky), and a second click inside the
loupe pins the corner to sub-source-pixel accuracy.

The CSV matches the frontend CourtBoundaryStep contract: normalised xy, any click
order (the scorer re-sorts to TL TR BR BL), and an orientation flag. Long form,
one row per corner::

    video,frame,corner_idx,x_px,y_px,x_norm,y_norm,orientation

The model is never loaded here; this tool only produces ground truth.

Controls:

- trackbar, or ``,`` / ``.`` for -1 / +1 frame, ``<`` / ``>`` for -25 / +25
- ``c`` starts a 4-corner capture on the current frame
- during capture: click a corner, then click again in the loupe to refine; ESC
  aborts the frame cleanly
- ``q`` quits

Usage::

    python src/courtkeynet/validation_scripts/annotate_court_corners.py \\
        --video path/to/clip.mp4 --out-csv hand_corners.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

CSV_HEADER = ("video", "frame", "corner_idx", "x_px", "y_px", "x_norm", "y_norm", "orientation")

# Orange from the Wong colourblind-safe palette (#D55E00 as BGR); reads clearly
# for protan vision against the green court and white lines.
CORNER_COLOUR = (0, 94, 213)
TEXT_COLOUR = (255, 255, 255)


# --- Pure helpers: coordinate maths (unit-tested) --------------------------

def normalise_xy(x_px: float, y_px: float, width: int, height: int) -> tuple[float, float]:
    """:return: (x_norm, y_norm) pixel coords divided by the frame extent.

    :param x_px: x in source pixels
    :param y_px: y in source pixels
    :param width: frame width in pixels
    :param height: frame height in pixels
    """
    return x_px / width, y_px / height


def loupe_origin(coarse_x: float, coarse_y: float, half: int, width: int, height: int) -> tuple[int, int]:
    """Top-left source pixel of the loupe crop, clamped so the window stays on-frame.

    The crop is a ``2*half`` square centred on the coarse click. Near a frame edge
    the origin is clamped inward, so the click no longer sits dead-centre in the
    loupe; that is fine because the refine click, not the coarse one, sets the
    corner. Origin is integer because the crop is taken by array slicing.

    :param coarse_x: coarse click x in source pixels
    :param coarse_y: coarse click y in source pixels
    :param half: half-window of the crop in source pixels (crop side is ``2*half``)
    :param width: frame width in pixels
    :param height: frame height in pixels
    :return: (origin_x, origin_y) integer top-left of the crop
    """
    win = 2 * half
    origin_x = int(round(coarse_x)) - half
    origin_y = int(round(coarse_y)) - half
    # Clamp into [0, dim - win]; max(dim - win, 0) guards a frame smaller than the crop.
    origin_x = min(max(origin_x, 0), max(width - win, 0))
    origin_y = min(max(origin_y, 0), max(height - win, 0))
    return origin_x, origin_y


def loupe_to_source(lx: float, ly: float, origin_xy: tuple[int, int], zoom: int) -> tuple[float, float]:
    """Map a click in the zoomed loupe image back to source pixels.

    The loupe shows the ``[origin, origin + 2*half]`` source window scaled up by
    ``zoom``, so a loupe click at ``(lx, ly)`` sits at ``origin + click/zoom`` in
    source pixels. Sub-source-pixel because ``lx/zoom`` is fractional.

    :param lx: click x in the zoomed loupe image
    :param ly: click y in the zoomed loupe image
    :param origin_xy: the loupe crop's source-pixel top-left
    :param zoom: integer magnification the loupe was drawn at
    :return: (x, y) in source pixels
    """
    origin_x, origin_y = origin_xy
    return origin_x + lx / zoom, origin_y + ly / zoom


def source_to_loupe(src_x: float, src_y: float, origin_xy: tuple[int, int], zoom: int) -> tuple[float, float]:
    """Inverse of :func:`loupe_to_source`: source pixel to zoomed loupe coords.

    Used to draw the coarse point inside the loupe. Exact inverse, so a source
    pixel round-trips through both functions unchanged.

    :param src_x: x in source pixels
    :param src_y: y in source pixels
    :param origin_xy: the loupe crop's source-pixel top-left
    :param zoom: integer magnification the loupe was drawn at
    :return: (x, y) in the zoomed loupe image
    """
    origin_x, origin_y = origin_xy
    return (src_x - origin_x) * zoom, (src_y - origin_y) * zoom


# --- Pure helpers: CSV rows (unit-tested) ----------------------------------

def build_corner_rows(
    video: str,
    frame: int,
    corners_px: list[tuple[float, float]],
    width: int,
    height: int,
    orientation: str,
) -> list[dict[str, object]]:
    """Build one CSV row per corner, ``corner_idx`` following click order.

    Click order is preserved (0-3); the scorer canonicalises to TL TR BR BL, so
    no sorting happens here.

    :param video: source video path string, stored verbatim
    :param frame: frame index the corners were clicked on
    :param corners_px: the 4 refined corners in click order, source pixels
    :param width: frame width in pixels, for the normalised columns
    :param height: frame height in pixels, for the normalised columns
    :param orientation: orientation flag stored on every row
    :return: list of 4 row dicts keyed by :data:`CSV_HEADER`
    """
    rows: list[dict[str, object]] = []
    for corner_idx, (x_px, y_px) in enumerate(corners_px):
        x_norm, y_norm = normalise_xy(x_px, y_px, width, height)
        rows.append({
            "video": video,
            "frame": frame,
            "corner_idx": corner_idx,
            "x_px": x_px,
            "y_px": y_px,
            "x_norm": x_norm,
            "y_norm": y_norm,
            "orientation": orientation,
        })
    return rows


def append_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """Append rows to the CSV, writing the header only when the file is new/empty.

    :param csv_path: destination CSV (opened in append mode)
    :param rows: row dicts keyed by :data:`CSV_HEADER`
    """
    csv_path = Path(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# --- Capture state machine (unit-tested) -----------------------------------

class CaptureState(StrEnum):
    """Where a capture is in the coarse/refine cycle."""

    SCRUB = "scrub"  # free navigation, no capture in progress
    AWAITING_COARSE = "awaiting_coarse"  # want a coarse click for the next corner
    AWAITING_REFINE = "awaiting_refine"  # loupe open, want a refine click


class ActionKind(StrEnum):
    """What the GUI shell should do after handling an event."""

    NOOP = "noop"  # event not valid in this state; ignore it
    AWAIT_COARSE = "await_coarse"  # capture started; wait for a coarse click
    OPEN_LOUPE = "open_loupe"  # coarse placed; open the loupe at loupe_origin
    CORNER_PLACED = "corner_placed"  # corner refined; more corners to go
    COMMITTED = "committed"  # 4th corner placed and rows appended; back to scrub
    ABORTED = "aborted"  # capture cancelled; back to scrub


@dataclass(frozen=True)
class SessionAction:
    """The session's reply to one event: a kind plus whatever the shell needs.

    :param kind: what the shell should do next
    :param corner_index: the corner the shell now waits on / just placed (0-4)
    :param loupe_origin: crop top-left for OPEN_LOUPE, else None
    :param coarse_xy: the coarse click for OPEN_LOUPE (loupe crosshair), else None
    :param corners: corners placed so far (CORNER_PLACED) or all four (COMMITTED)
    """

    kind: ActionKind
    corner_index: int = 0
    loupe_origin: tuple[int, int] | None = None
    coarse_xy: tuple[float, float] | None = None
    corners: tuple[tuple[float, float], ...] = field(default_factory=tuple)


class CaptureSession:
    """The corner-capture state machine; the cv2 loop is a thin shell over it.

    Methods take abstract events (:meth:`begin_capture`, :meth:`coarse_click`,
    :meth:`refine_click`, :meth:`abort`) and return a :class:`SessionAction`
    telling the shell what to render or do. On the fourth refined corner the
    frame's annotation is appended to the CSV and the session returns to scrub.
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
    ) -> None:
        """:param video: source video path string, stored on every CSV row
        :param width: frame width in pixels
        :param height: frame height in pixels
        :param orientation: orientation flag written to the CSV
        :param csv_path: destination CSV, appended to on each committed frame
        :param half: loupe half-window in source pixels
        :param zoom: loupe magnification
        """
        self.video = video
        self.width = width
        self.height = height
        self.orientation = orientation
        self.csv_path = Path(csv_path)
        self.half = half
        self.zoom = zoom
        self.state = CaptureState.SCRUB
        self.frame_idx = -1
        self._corners: list[tuple[float, float]] = []
        self._pending_origin: tuple[int, int] | None = None
        self._pending_coarse: tuple[float, float] | None = None

    @property
    def placed_corners(self) -> tuple[tuple[float, float], ...]:
        """:return: corners refined so far this frame, in click order."""
        return tuple(self._corners)

    def begin_capture(self, frame_idx: int) -> SessionAction:
        """Start a fresh 4-corner capture on ``frame_idx`` (only from scrub).

        :param frame_idx: the frame being annotated
        :return: AWAIT_COARSE for corner 0, or NOOP if a capture is already live
        """
        if self.state is not CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        self.frame_idx = frame_idx
        self._corners = []
        self._pending_origin = None
        self._pending_coarse = None
        self.state = CaptureState.AWAITING_COARSE
        return SessionAction(ActionKind.AWAIT_COARSE, corner_index=0)

    def coarse_click(self, x: float, y: float) -> SessionAction:
        """Record a coarse click and open the loupe around it.

        :param x: coarse click x in source pixels
        :param y: coarse click y in source pixels
        :return: OPEN_LOUPE with the crop origin, or NOOP if not awaiting a coarse click
        """
        if self.state is not CaptureState.AWAITING_COARSE:
            return SessionAction(ActionKind.NOOP)
        origin = loupe_origin(x, y, self.half, self.width, self.height)
        self._pending_origin = origin
        self._pending_coarse = (x, y)
        self.state = CaptureState.AWAITING_REFINE
        return SessionAction(
            ActionKind.OPEN_LOUPE,
            corner_index=len(self._corners),
            loupe_origin=origin,
            coarse_xy=(x, y),
        )

    def refine_click(self, lx: float, ly: float) -> SessionAction:
        """Pin the current corner from a loupe click; commit on the fourth.

        :param lx: refine click x in the zoomed loupe image
        :param ly: refine click y in the zoomed loupe image
        :return: CORNER_PLACED for corners 1-3, COMMITTED on the 4th, else NOOP
        """
        if self.state is not CaptureState.AWAITING_REFINE:
            return SessionAction(ActionKind.NOOP)
        # _pending_origin is always set while the state is AWAITING_REFINE.
        assert self._pending_origin is not None
        source_xy = loupe_to_source(lx, ly, self._pending_origin, self.zoom)
        self._corners.append(source_xy)
        self._pending_origin = None
        self._pending_coarse = None

        if len(self._corners) < 4:
            self.state = CaptureState.AWAITING_COARSE
            return SessionAction(
                ActionKind.CORNER_PLACED,
                corner_index=len(self._corners),
                corners=tuple(self._corners),
            )

        rows = build_corner_rows(
            self.video, self.frame_idx, self._corners, self.width, self.height, self.orientation
        )
        append_rows(self.csv_path, rows)
        committed = tuple(self._corners)
        self._corners = []
        self.state = CaptureState.SCRUB
        return SessionAction(ActionKind.COMMITTED, corner_index=4, corners=committed)

    def abort(self) -> SessionAction:
        """Cancel the in-progress capture and drop back to scrub, writing nothing.

        :return: ABORTED when a capture was live, else NOOP
        """
        if self.state is CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        self._corners = []
        self._pending_origin = None
        self._pending_coarse = None
        self.state = CaptureState.SCRUB
        return SessionAction(ActionKind.ABORTED)


# --- cv2 shell (thin driver over the session; not unit-tested) -------------

MAIN_WINDOW = "annotate court corners"
LOUPE_WINDOW = "refine (click the corner)"

# Key codes as returned by waitKey & 0xFF.
KEY_PREV = ord(",")
KEY_NEXT = ord(".")
KEY_PREV_JUMP = ord("<")
KEY_NEXT_JUMP = ord(">")
KEY_CAPTURE = ord("c")
KEY_QUIT = ord("q")
KEY_ESC = 27
JUMP = 25


def draw_scrub_overlay(frame: np.ndarray, session: CaptureSession, frame_idx: int, total: int) -> np.ndarray:
    """Draw the frame index strip and any corners placed so far this capture.

    :param frame: the current BGR frame
    :param session: the live capture session (for placed corners and state)
    :param frame_idx: current frame index
    :param total: total frames, for the strip
    :return: a copy of the frame with the overlay drawn
    """
    canvas = frame.copy()
    status = f"frame {frame_idx}/{total - 1}   [{session.state}]   corners {len(session.placed_corners)}/4"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(canvas, status, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOUR, 1, cv2.LINE_AA)
    for corner_idx, (x_px, y_px) in enumerate(session.placed_corners):
        centre = (int(round(x_px)), int(round(y_px)))
        cv2.circle(canvas, centre, 4, CORNER_COLOUR, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(corner_idx), (centre[0] + 6, centre[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORNER_COLOUR, 1, cv2.LINE_AA)
    return canvas


def render_loupe(frame: np.ndarray, action: SessionAction, half: int, zoom: int) -> np.ndarray:
    """Build the zoomed loupe image for a coarse click, with a centre crosshair.

    :param frame: the current BGR frame
    :param action: the OPEN_LOUPE action carrying the crop origin and coarse click
    :param half: loupe half-window in source pixels
    :param zoom: loupe magnification
    :return: the ``2*half*zoom`` square zoomed crop, BGR
    """
    origin = action.loupe_origin
    assert origin is not None  # OPEN_LOUPE always carries an origin
    origin_x, origin_y = origin
    win = 2 * half
    crop = frame[origin_y : origin_y + win, origin_x : origin_x + win]
    zoomed = cv2.resize(crop, (win * zoom, win * zoom), interpolation=cv2.INTER_NEAREST)
    if action.coarse_xy is not None:
        loupe_x, loupe_y = source_to_loupe(action.coarse_xy[0], action.coarse_xy[1], origin, zoom)
        cx, cy = int(round(loupe_x)), int(round(loupe_y))
        cv2.line(zoomed, (cx - 12, cy), (cx + 12, cy), CORNER_COLOUR, 1, cv2.LINE_AA)
        cv2.line(zoomed, (cx, cy - 12), (cx, cy + 12), CORNER_COLOUR, 1, cv2.LINE_AA)
    return zoomed


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    """Seek to and decode one frame; None on a failed read.

    :param cap: an open VideoCapture
    :param frame_idx: zero-based frame index
    :return: the BGR frame, or None past the end / on a decode failure
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def run_annotation_tool(args: argparse.Namespace) -> int:
    """Drive the scrub/capture loop over the cv2 window and the CaptureSession.

    :param args: parsed CLI namespace
    :return: process exit code
    """
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open video {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    session = CaptureSession(
        str(args.video), width, height, args.orientation,
        Path(args.out_csv), args.loupe_size, args.loupe_zoom,
    )
    # Mouse callbacks push their SessionAction here; the main loop drains it so
    # window creation/teardown stays on the loop thread.
    pending: list[SessionAction] = []
    frame_state = {"idx": max(0, min(args.start_frame, total - 1)), "frame": None}

    def load(idx: int) -> None:
        # Frozen during a capture: corners are recorded against the frame the capture
        # began on, so a mid-capture scrub (keys or trackbar) would silently attach
        # clicks from one frame to another frame's CSV rows. ESC first, then scrub.
        if session.state is not CaptureState.SCRUB:
            return
        frame_state["idx"] = max(0, min(idx, total - 1))
        frame_state["frame"] = read_frame(cap, frame_state["idx"])

    def on_main_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and session.state is CaptureState.AWAITING_COARSE:
            pending.append(session.coarse_click(float(x), float(y)))

    def on_loupe_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and session.state is CaptureState.AWAITING_REFINE:
            pending.append(session.refine_click(float(x), float(y)))

    cv2.namedWindow(MAIN_WINDOW)
    cv2.setMouseCallback(MAIN_WINDOW, on_main_mouse)
    cv2.createTrackbar("frame", MAIN_WINDOW, frame_state["idx"], max(total - 1, 1), lambda idx: load(idx))
    load(frame_state["idx"])
    print(f"{args.video}: {width}x{height}, {total} frames. c=capture  ,/. = -1/+1  </> = -25/+25  q=quit")

    loupe_open = False
    while True:
        frame = frame_state["frame"]
        if frame is not None:
            cv2.imshow(MAIN_WINDOW, draw_scrub_overlay(frame, session, frame_state["idx"], total))

        while pending:
            action = pending.pop(0)
            if action.kind is ActionKind.OPEN_LOUPE and frame is not None:
                cv2.namedWindow(LOUPE_WINDOW)
                cv2.setMouseCallback(LOUPE_WINDOW, on_loupe_mouse)
                cv2.imshow(LOUPE_WINDOW, render_loupe(frame, action, args.loupe_size, args.loupe_zoom))
                loupe_open = True
            elif action.kind in (ActionKind.CORNER_PLACED, ActionKind.COMMITTED, ActionKind.ABORTED):
                if loupe_open:
                    cv2.destroyWindow(LOUPE_WINDOW)
                    loupe_open = False
                if action.kind is ActionKind.COMMITTED:
                    print(f"  saved frame {session.frame_idx}: 4 corners -> {args.out_csv}")

        key = cv2.waitKey(20) & 0xFF
        if key == KEY_QUIT:
            break
        if key == KEY_ESC and session.state is not CaptureState.SCRUB:
            pending.append(session.abort())
        elif key == KEY_CAPTURE:
            pending.append(session.begin_capture(frame_state["idx"]))
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
    parser.add_argument("--out-csv", type=Path, required=True, help="Appended to; header written when new.")
    parser.add_argument("--orientation", type=str, default="portrait", help="Orientation flag stored per row.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame to open on (default 0).")
    parser.add_argument("--loupe-size", type=int, default=64,
                        help="Loupe half-window in source pixels (crop side is twice this; default 64).")
    parser.add_argument("--loupe-zoom", type=int, default=8, help="Loupe magnification with NEAREST (default 8).")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"ERROR: --video does not exist: {args.video}")
    return run_annotation_tool(args)


if __name__ == "__main__":
    sys.exit(main())
