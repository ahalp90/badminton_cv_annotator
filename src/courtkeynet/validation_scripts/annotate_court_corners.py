"""Retired tour annotator: helpers, sessions, and fit maths live on; the GUI is gone.

The runnable GUI retired on 2026-07-12 per docs/annotator_unification_brief.md;
``annotate_court_corners_offframe.py`` is THE court corner annotator (it grafted
this tool's point floor, reprojection gate, quad sanity check, loupe guard,
launch help, and window robustness). What remains here is everything other code
still imports:

- the loupe coordinate maths and ``render_loupe``/``read_frame``, which the
  off-frame tool reuses
- the corners-CSV and points-sidecar helpers (both formats stay byte-compatible;
  ``score_hand_corners.py`` and the eval consume them)
- ``build_point_table`` and the court-plan chain
- ``fit_corner_quad`` with its gates, and both capture state machines, pinned by
  ``tests/test_courtkeynet_annotation.py`` (including the torch-free regression)

The corners CSV stays the frontend CourtBoundaryStep contract: normalised xy,
any click order (the scorer re-sorts to TL TR BR BL), an orientation flag, one
row per corner::

    video,frame,corner_idx,x_px,y_px,x_norm,y_norm,orientation

Torch is never imported on this module's chain.
"""
from __future__ import annotations

import csv
import itertools
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# Resolve the repo root (three parents up) so the `src.` import below works when
# this file is run as a plain script; mirrors render_court_overlay.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.courtkeynet.court_corners import (  # noqa: E402
    CORNER_COURT_M,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    PAINTED_SEGMENTS_M,
)

CSV_HEADER = ("video", "frame", "corner_idx", "x_px", "y_px", "x_norm", "y_norm", "orientation")
# Sidecar written only by the intersection tour: one row per clicked intersection,
# with rms_px filled when a homography fit produced the corners (empty otherwise).
POINT_CSV_HEADER = ("video", "frame", "point_name", "x_px", "y_px", "rms_px")

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


def _append_rows(csv_path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    """Append rows to a CSV, writing the header only when the file is new/empty.

    Shared by the main corner CSV and the tour's points sidecar: both want the
    same append-with-header-once behaviour, only the column set differs. An
    existing file must already carry this column set: appending main rows onto a
    sidecar (or the reverse) would silently corrupt both, so a mismatched header
    raises instead.

    :param csv_path: destination CSV (opened in append mode)
    :param rows: row dicts keyed by ``fieldnames``
    :param fieldnames: the CSV column order/header for this file
    """
    csv_path = Path(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        with csv_path.open("r", newline="") as handle:
            existing = tuple(next(csv.reader(handle)))
        if existing != fieldnames:
            raise ValueError(
                f"{csv_path} carries header {existing}, not the expected {fieldnames}; "
                "refusing to append rows meant for the other CSV"
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def append_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """Append corner rows to the main CSV, header written only when new/empty.

    :param csv_path: destination CSV (opened in append mode)
    :param rows: row dicts keyed by :data:`CSV_HEADER`
    """
    _append_rows(csv_path, rows, CSV_HEADER)


def load_annotated_frames(csv_path: Path, video: str) -> set[int]:
    """Frame indices the main CSV already carries rows for, on this video.

    Warns the user before they double-annotate a frame: the scorer reads exactly
    four rows per (video, frame), and a duplicate frame poisons that read.

    :param csv_path: the main corner CSV; a missing or empty file has no annotations
    :param video: source video path string, matched by basename against the CSV's
        video column (tolerates a different path prefix, mirroring the scorer)
    :return: the set of already-annotated frame indices for this video
    """
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "video" not in reader.fieldnames or "frame" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} header {reader.fieldnames} is missing the video/frame columns "
                "load_annotated_frames needs"
            )
        video_name = Path(video).name
        return {int(row["frame"]) for row in reader if Path(row["video"]).name == video_name}


def build_point_rows(
    video: str,
    frame: int,
    clicked: list[tuple[str, float, float]],
    rms_px: float | None,
) -> list[dict[str, object]]:
    """Build the tour's sidecar rows, one per clicked intersection.

    :param video: source video path string, stored verbatim
    :param frame: frame index the tour ran on
    :param clicked: the clicked intersections as (point_name, x_px, y_px), tour order
    :param rms_px: fit reprojection RMS in pixels, or None when no fit ran (the
        four-corners-clicked path); None becomes an empty string in every row
    :return: list of row dicts keyed by :data:`POINT_CSV_HEADER`
    """
    rms_value: object = "" if rms_px is None else rms_px
    return [
        {"video": video, "frame": frame, "point_name": name, "x_px": x_px, "y_px": y_px, "rms_px": rms_value}
        for name, x_px, y_px in clicked
    ]


def append_point_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    """Append sidecar rows to the points CSV, header written only when new/empty.

    :param csv_path: destination points CSV (opened in append mode)
    :param rows: row dicts keyed by :data:`POINT_CSV_HEADER`
    """
    _append_rows(csv_path, rows, POINT_CSV_HEADER)


# --- Pure helpers: court plan + homography fit (unit-tested) ---------------

# Painted-line names keyed by the line's constant court coordinate (metres,
# rounded to 2 dp). The x-family runs along the court's length at a constant x;
# the y-family runs across it at a constant y. These keys must cover every
# constant in court_corners.PAINTED_SEGMENTS_M, or build_point_table raises: a missing
# key means the court model changed under the annotator.
X_LINE_NAMES = {
    0.0: "left doubles sideline",
    0.46: "left singles sideline",
    3.05: "centre line",
    5.64: "right singles sideline",
    6.1: "right doubles sideline",
}
Y_LINE_NAMES = {
    0.0: "far baseline",
    0.76: "far doubles long service line",
    4.72: "far short service line",
    8.68: "near short service line",
    12.64: "near doubles long service line",
    13.4: "near baseline",
}
# Corner labels in CORNER_COURT_M slot order (TL TR BR BL); the tour visits the
# four corners first, in this order.
CORNER_LABELS = ("top-left", "top-right", "bottom-right", "bottom-left")
# Fit reprojection floor, applied to the WORST clicked point rather than the
# mean: least squares smears one bad click's error across the good ones, so a
# 30 px misidentification among six clicks leaves the mean near 1 px while the
# worst point still sticks out. Hand clicks through the loupe land ~1-2 px, so
# 3 px flags a misidentified point without punishing honest jitter.
FIT_MAX_REPROJ_PX = 3.0


@dataclass(frozen=True)
class PlanPoint:
    """One clickable painted-line intersection in the BWF court plan.

    :param name: the label shown while the tour waits on this point
    :param court_xy: (x, y) court position in metres, the homography's domain point
    """

    name: str
    court_xy: tuple[float, float]


class _PlanLine(NamedTuple):
    """One painted line reduced to what the intersection algebra needs.

    :param const: the constant court coordinate (x for the x-family, y for the y)
    :param lo: low end of the painted extent along the varying axis, metres
    :param hi: high end of the painted extent along the varying axis, metres
    :param name: the human line name from the lookup
    """

    const: float
    lo: float
    hi: float
    name: str


def _plan_line_name(const: float, names: dict[float, str], family: str) -> str:
    """:return: the painted-line name for a constant coordinate, raising if unknown.

    :param const: the line's constant court coordinate in metres
    :param names: the family lookup (x-family or y-family)
    :param family: family label used only in the error message
    """
    key = round(const, 2)
    if key not in names:
        raise ValueError(
            f"court plan {family} line at {key} m has no name in the annotator's lookup; "
            f"court_corners.PAINTED_SEGMENTS_M changed under it"
        )
    return names[key]


def _split_painted_segments() -> tuple[list[_PlanLine], list[_PlanLine]]:
    """Split the imported painted segments into the x-family and y-family lines.

    Mirrors the fallback's own constant-x vs constant-y test. Each segment's name
    is resolved here, so a court-model change fails loud at table-build time.

    :return: (x-family lines, y-family lines)
    """
    x_lines: list[_PlanLine] = []
    y_lines: list[_PlanLine] = []
    for end_a, end_b in PAINTED_SEGMENTS_M:
        if end_a[0] == end_b[0]:  # constant x -> runs along the court's length
            const = float(end_a[0])
            lo, hi = sorted((float(end_a[1]), float(end_b[1])))
            x_lines.append(_PlanLine(const, lo, hi, _plan_line_name(const, X_LINE_NAMES, "x-family")))
        else:  # constant y -> runs across the court
            const = float(end_a[1])
            lo, hi = sorted((float(end_a[0]), float(end_b[0])))
            y_lines.append(_PlanLine(const, lo, hi, _plan_line_name(const, Y_LINE_NAMES, "y-family")))
    return x_lines, y_lines


def build_point_table() -> tuple[PlanPoint, ...]:
    """Derive the tour's clickable intersections from the BWF court plan.

    Every x-family line crossed with every y-family line gives a candidate point;
    it is painted (clickable) only where the crossing lies on BOTH segments'
    extents, endpoints inclusive. The four outer corners come first in TL TR BR BL
    order (CORNER_COURT_M slot order) with corner-style names; the rest follow in a
    fixed far-to-near, left-to-right scan so the tour order never shifts.

    :return: the ordered tour points, corners first
    """
    # Pin the annotator's assumption that the four projected corners are the
    # width x length rectangle. A fallback court-model change fails loud here
    # rather than silently mislabelling the corners the tour writes.
    expected_corners = np.array(
        [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
        dtype=np.float32,
    )
    if not np.allclose(CORNER_COURT_M, expected_corners):
        raise ValueError("fallback CORNER_COURT_M no longer matches COURT_WIDTH_M x COURT_LENGTH_M; court model changed")

    x_lines, y_lines = _split_painted_segments()
    # (court_xy, x-name, y-name) per painted intersection.
    intersections: list[tuple[tuple[float, float], str, str]] = []
    for x_line in x_lines:
        for y_line in y_lines:
            on_x_segment = x_line.lo <= y_line.const <= x_line.hi
            on_y_segment = y_line.lo <= x_line.const <= y_line.hi
            if on_x_segment and on_y_segment:
                intersections.append(((x_line.const, y_line.const), x_line.name, y_line.name))

    # Keyed by the rounded court coordinate so a corner from CORNER_COURT_M can be
    # matched back to its intersection without float-equality worries.
    by_key = {
        (round(court_xy[0], 2), round(court_xy[1], 2)): (court_xy, x_name, y_name)
        for court_xy, x_name, y_name in intersections
    }

    corner_points: list[PlanPoint] = []
    corner_keys: set[tuple[float, float]] = set()
    for slot, label in enumerate(CORNER_LABELS):
        corner_xy = (float(CORNER_COURT_M[slot][0]), float(CORNER_COURT_M[slot][1]))
        key = (round(corner_xy[0], 2), round(corner_xy[1], 2))
        court_xy, x_name, y_name = by_key[key]
        # Corners read baseline-then-sideline in the bracket, the way a coach names
        # a court corner (which baseline, which side).
        name = f"{label} corner ({y_name} meets {x_name})"
        corner_points.append(PlanPoint(name=name, court_xy=court_xy))
        corner_keys.add(key)

    remaining = [
        (court_xy, x_name, y_name)
        for court_xy, x_name, y_name in intersections
        if (round(court_xy[0], 2), round(court_xy[1], 2)) not in corner_keys
    ]
    # Fixed scan: far baseline to near (y up), left sideline to right (x up).
    remaining.sort(key=lambda item: (item[0][1], item[0][0]))
    other_points = [
        PlanPoint(name=f"{x_name} meets {y_name}", court_xy=court_xy)
        for court_xy, x_name, y_name in remaining
    ]
    return tuple(corner_points + other_points)


def _collinear(point_p: np.ndarray, point_q: np.ndarray, point_r: np.ndarray) -> bool:
    """:return: True when three plan points are collinear, within float tolerance.

    The signed twice-area of the triangle is zero for collinear triples, and those
    are not only same-painted-line triples: the court grid's even spacings put
    DIAGONAL triples on one line too. The y-lines 0.76 / 4.72 / 8.68 / 12.64 sit
    3.96 m apart and the sidelines 0 / 3.05 / 6.1 are evenly spaced, so e.g.
    (0.46, 0.76), (3.05, 4.72), (5.64, 8.68) is exactly collinear in decimal, but
    float subtraction leaves its cross at ~1e-15, which an exact == 0.0 misses.
    Hence the tolerance: genuinely non-collinear triples on the centimetre lattice
    have |cross| of at least ~1e-4, float fuzz sits ~1e-15, and 1e-6 splits the
    two with orders of magnitude to spare.
    """
    cross = (point_q[0] - point_p[0]) * (point_r[1] - point_p[1]) - (point_q[1] - point_p[1]) * (point_r[0] - point_p[0])
    return bool(abs(cross) < 1e-6)


def _has_general_position_quad(plan_pts: np.ndarray) -> bool:
    """:return: True when some 4 of the plan points have no 3 collinear.

    Four correspondences with no 3 collinear determine a well-posed homography.
    Without this a 3-collinear click set makes cv2.findHomography return
    plausible-looking garbage instead of failing, so the reprojection guard would
    be scoring a fit it should never have attempted. The point set is small, so
    the brute-force combination scan is cheap.

    :param plan_pts: (n, 2) clicked points in court metres
    """
    for subset in itertools.combinations(range(len(plan_pts)), 4):
        quad = plan_pts[list(subset)]
        if not any(_collinear(quad[a], quad[b], quad[c]) for a, b, c in itertools.combinations(range(4), 3)):
            return True
    return False


def _project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map points through a homography.

    :param homography: (3, 3) transform
    :param points: (n, 2) source points
    :return: (n, 2) float64 mapped points
    """
    reshaped = points.reshape(-1, 1, 2).astype(np.float64)
    mapped = cv2.perspectiveTransform(reshaped, homography)
    return mapped.reshape(-1, 2)


def _quad_is_camera_valid(quad: np.ndarray) -> bool:
    """:return: True when a projected quad is a convex behind-baseline TL TR BR BL rectangle.

    Local rather than imported: the fallback's convexity/order test is wired into
    its detection gate (CornerDetection flags), not exposed as a public quad
    predicate, and this module may only borrow the court constants. Convex means
    the four edge turns share one sign. Behind-baseline means the near baseline
    sits below the far one, so TL is above BL and TR above BR in image y (smaller
    y is higher up the frame).

    :param quad: (4, 2) projected corners in TL TR BR BL order
    """
    crosses = []
    for corner in range(4):
        here = quad[corner]
        nxt = quad[(corner + 1) % 4]
        after = quad[(corner + 2) % 4]
        crosses.append((nxt[0] - here[0]) * (after[1] - nxt[1]) - (nxt[1] - here[1]) * (after[0] - nxt[0]))
    convex = all(cross > 0 for cross in crosses) or all(cross < 0 for cross in crosses)
    top_left, top_right, bottom_right, bottom_left = quad
    behind_baseline = top_left[1] < bottom_left[1] and top_right[1] < bottom_right[1]
    return bool(convex and behind_baseline)


def fit_corner_quad(plan_pts_m: np.ndarray, clicked_px: np.ndarray) -> tuple[np.ndarray | None, float, str]:
    """Fit a plan->image homography from clicked points and project the four corners.

    Plain DLT least squares over all clicked correspondences (no RANSAC: hand
    clicks are few and deliberate). Returns None with a reason on any failure the
    caller must report and write nothing for: the solve collapsing, the worst
    point's reprojection over the floor (a misidentified point), or a projected
    quad that is not a convex behind-baseline rectangle. The mean RMS is computed
    and returned either way; it is the sidecar's bookkeeping figure, not the gate.

    :param plan_pts_m: (n, 2) clicked points in court metres
    :param clicked_px: (n, 2) the matching clicked image pixels
    :return: ((4, 2) corners TL TR BR BL or None, reprojection RMS px, failure reason)
    """
    homography, _ = cv2.findHomography(plan_pts_m, clicked_px, 0)
    if homography is None:
        return None, float("inf"), "homography solve returned None (clicked points too degenerate)"
    errors = np.linalg.norm(_project_points(homography, plan_pts_m) - clicked_px, axis=1)
    rms = float(np.sqrt(np.mean(errors ** 2)))
    worst = float(errors.max())
    if worst > FIT_MAX_REPROJ_PX:
        return None, rms, (f"worst point reprojects {worst:.1f} px off, over the "
                           f"{FIT_MAX_REPROJ_PX:.1f} px floor (a misidentified point?)")
    corners = _project_points(homography, CORNER_COURT_M.astype(np.float64)).astype(np.float32)
    if not _quad_is_camera_valid(corners):
        return None, rms, "projected court quad is not a convex behind-baseline TL TR BR BL rectangle"
    return corners, rms, ""


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
    COMMITTED = "committed"  # rows appended (corner capture or tour); back to scrub
    ABORTED = "aborted"  # capture cancelled or a tour commit failed; back to scrub
    UNDONE = "undone"  # last click un-placed / pending loupe cancelled; shell closes the loupe
    # Tour-only transitions; the loupe cycle (AWAIT_COARSE / OPEN_LOUPE) is shared.
    POINT_PLACED = "point_placed"  # tour point refined; advanced to the next
    POINT_SKIPPED = "point_skipped"  # tour point skipped; advanced to the next


@dataclass(frozen=True)
class SessionAction:
    """The session's reply to one event: a kind plus whatever the shell needs.

    :param kind: what the shell should do next
    :param corner_index: the corner the shell now waits on / just placed (0-4)
    :param loupe_origin: crop top-left for OPEN_LOUPE, else None
    :param coarse_xy: the coarse click for OPEN_LOUPE (loupe crosshair), else None
    :param corners: corners placed so far (CORNER_PLACED) or all four (COMMITTED)
    :param message: human-readable note for the shell to display, else empty
    """

    kind: ActionKind
    corner_index: int = 0
    loupe_origin: tuple[int, int] | None = None
    coarse_xy: tuple[float, float] | None = None
    corners: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    message: str = ""


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

    def undo(self) -> SessionAction:
        """Step back one click: a pending loupe cancels first, else the last corner pops.

        One press undoes one thing. ESC already covers leaving the capture, so undo
        with nothing placed and no loupe pending is a NOOP.

        :return: UNDONE with the corner count/corners after stepping back, else NOOP
        """
        if self.state is CaptureState.SCRUB:
            return SessionAction(ActionKind.NOOP)
        if self.state is CaptureState.AWAITING_REFINE:
            self._pending_origin = None
            self._pending_coarse = None
            self.state = CaptureState.AWAITING_COARSE
            return SessionAction(ActionKind.UNDONE, corner_index=len(self._corners), corners=tuple(self._corners))
        if not self._corners:
            return SessionAction(ActionKind.NOOP)
        self._corners.pop()
        return SessionAction(ActionKind.UNDONE, corner_index=len(self._corners), corners=tuple(self._corners))


@dataclass(frozen=True)
class TourAction:
    """The tour session's reply to one event: a kind plus what the shell needs.

    Shares the loupe fields (loupe_origin, coarse_xy) with :class:`SessionAction`
    so the shell renders the loupe the same way for either capture mode.

    :param kind: what the shell should do next
    :param point_index: the tour point now current (or just handled)
    :param point_name: that point's display name, for the overlay
    :param loupe_origin: crop top-left for OPEN_LOUPE, else None
    :param coarse_xy: the coarse click for OPEN_LOUPE (loupe crosshair), else None
    :param clicked: how many points have been clicked so far this tour
    :param skipped: how many points have been skipped so far this tour
    :param corners: the four committed corners (COMMITTED), else empty
    :param message: human-readable note for the shell to display, else empty
    """

    kind: ActionKind
    point_index: int = 0
    point_name: str = ""
    loupe_origin: tuple[int, int] | None = None
    coarse_xy: tuple[float, float] | None = None
    clicked: int = 0
    skipped: int = 0
    corners: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    message: str = ""


class IntersectionSession:
    """The intersection-tour state machine; the cv2 loop is a thin shell over it.

    Same coarse/refine loupe cycle as :class:`CaptureSession`, but instead of four
    fixed corners it walks the named painted-line intersections
    (:func:`build_point_table`). You click the visible ones and skip the rest; on
    'done' (or after the last point) it commits. If all four corners were clicked
    directly it writes them verbatim like the 'c' mode; otherwise it fits a
    homography from the clicked points against the BWF plan and writes the four
    projected corners. Every clicked point is recorded to a sidecar CSV too. Any
    commit failure prints the cause and writes nothing.
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
        point_table: tuple[PlanPoint, ...] | None = None,
    ) -> None:
        """:param video: source video path string, stored on every CSV row
        :param width: frame width in pixels
        :param height: frame height in pixels
        :param orientation: orientation flag written to the main CSV
        :param csv_path: destination main CSV; the sidecar sits beside it
        :param half: loupe half-window in source pixels
        :param zoom: loupe magnification
        :param point_table: the tour points; built from the BWF plan when omitted
        """
        self.video = video
        self.width = width
        self.height = height
        self.orientation = orientation
        self.csv_path = Path(csv_path)
        self.half = half
        self.zoom = zoom
        self.point_table = point_table if point_table is not None else build_point_table()
        self.state = CaptureState.SCRUB
        self.frame_idx = -1
        self._cursor = 0
        self._clicked: dict[int, tuple[float, float]] = {}
        self._skipped: set[int] = set()
        self._pending_origin: tuple[int, int] | None = None
        self._pending_coarse: tuple[float, float] | None = None

    @property
    def cursor(self) -> int:
        """:return: index into :attr:`point_table` of the point the tour waits on."""
        return self._cursor

    @property
    def clicked_count(self) -> int:
        """:return: how many tour points have been clicked so far."""
        return len(self._clicked)

    @property
    def skipped_count(self) -> int:
        """:return: how many tour points have been skipped so far."""
        return len(self._skipped)

    @property
    def clicked_points(self) -> tuple[tuple[float, float], ...]:
        """:return: the clicked pixels so far, in tour order."""
        return tuple(xy for _, xy in sorted(self._clicked.items()))

    def begin_tour(self, frame_idx: int) -> TourAction:
        """Start a fresh intersection tour on ``frame_idx`` (only from scrub).

        :param frame_idx: the frame being annotated
        :return: AWAIT_COARSE for the first point, or NOOP if a tour is already live
        """
        if self.state is not CaptureState.SCRUB:
            return TourAction(ActionKind.NOOP)
        self.frame_idx = frame_idx
        self._cursor = 0
        self._clicked = {}
        self._skipped = set()
        self._pending_origin = None
        self._pending_coarse = None
        self.state = CaptureState.AWAITING_COARSE
        return self._await_current(ActionKind.AWAIT_COARSE)

    def coarse_click(self, x: float, y: float) -> TourAction:
        """Record a coarse click for the current point and open the loupe around it.

        :param x: coarse click x in source pixels
        :param y: coarse click y in source pixels
        :return: OPEN_LOUPE with the crop origin, or NOOP if not awaiting a coarse click
        """
        if self.state is not CaptureState.AWAITING_COARSE:
            return TourAction(ActionKind.NOOP)
        origin = loupe_origin(x, y, self.half, self.width, self.height)
        self._pending_origin = origin
        self._pending_coarse = (x, y)
        self.state = CaptureState.AWAITING_REFINE
        return TourAction(
            ActionKind.OPEN_LOUPE,
            point_index=self._cursor,
            point_name=self.point_table[self._cursor].name,
            loupe_origin=origin,
            coarse_xy=(x, y),
            clicked=len(self._clicked),
            skipped=len(self._skipped),
        )

    def refine_click(self, lx: float, ly: float) -> TourAction:
        """Pin the current point from a loupe click and advance the tour.

        :param lx: refine click x in the zoomed loupe image
        :param ly: refine click y in the zoomed loupe image
        :return: POINT_PLACED (advanced), COMMITTED/ABORTED (last point), else NOOP
        """
        if self.state is not CaptureState.AWAITING_REFINE:
            return TourAction(ActionKind.NOOP)
        assert self._pending_origin is not None  # always set while AWAITING_REFINE
        source_xy = loupe_to_source(lx, ly, self._pending_origin, self.zoom)
        self._clicked[self._cursor] = source_xy
        self._pending_origin = None
        self._pending_coarse = None
        return self._advance(ActionKind.POINT_PLACED)

    def skip(self) -> TourAction:
        """Skip the current point (not visible / occluded) and advance the tour.

        Valid from either capture state: skipping with the loupe already open drops
        the pending coarse click so the point is genuinely skipped, not half-placed.

        :return: POINT_SKIPPED (advanced), COMMITTED/ABORTED (last point), else NOOP
        """
        if self.state not in (CaptureState.AWAITING_COARSE, CaptureState.AWAITING_REFINE):
            return TourAction(ActionKind.NOOP)
        self._pending_origin = None
        self._pending_coarse = None
        self._skipped.add(self._cursor)
        self.state = CaptureState.AWAITING_COARSE
        return self._advance(ActionKind.POINT_SKIPPED)

    def declare_done(self) -> TourAction:
        """Finish the tour early and commit whatever has been clicked so far.

        :return: COMMITTED on a good commit, ABORTED on a failed one, else NOOP
        """
        if self.state not in (CaptureState.AWAITING_COARSE, CaptureState.AWAITING_REFINE):
            return TourAction(ActionKind.NOOP)
        self._pending_origin = None
        self._pending_coarse = None
        return self._commit()

    def abort(self) -> TourAction:
        """Cancel the tour and drop back to scrub, writing nothing.

        :return: ABORTED when a tour was live, else NOOP
        """
        if self.state is CaptureState.SCRUB:
            return TourAction(ActionKind.NOOP)
        self._reset()
        return TourAction(ActionKind.ABORTED)

    def undo(self) -> TourAction:
        """Step back one point: a pending loupe cancels first, else the cursor steps back.

        Stepping back re-poses the previous point whether it was clicked or skipped.
        In-capture only, by ruling: after a commit (scrub) undo is a NOOP.

        :return: UNDONE re-awaiting the point stepped back to, else NOOP
        """
        if self.state is CaptureState.SCRUB:
            return TourAction(ActionKind.NOOP)
        if self.state is CaptureState.AWAITING_REFINE:
            self._pending_origin = None
            self._pending_coarse = None
            self.state = CaptureState.AWAITING_COARSE
            return self._await_current(ActionKind.UNDONE)
        if self._cursor == 0:
            return TourAction(ActionKind.NOOP)
        self._cursor -= 1
        # The stepped-back point is in at most one of the two records.
        self._clicked.pop(self._cursor, None)
        self._skipped.discard(self._cursor)
        return self._await_current(ActionKind.UNDONE)

    def _await_current(self, kind: ActionKind) -> TourAction:
        """:return: ``kind`` carrying the current point's name and tour progress."""
        point = self.point_table[self._cursor]
        return TourAction(
            kind,
            point_index=self._cursor,
            point_name=point.name,
            clicked=len(self._clicked),
            skipped=len(self._skipped),
        )

    def _advance(self, kind: ActionKind) -> TourAction:
        """Move past the current point; commit once the tour runs out.

        :param kind: the transition kind to report when more points remain
        :return: ``kind`` for the next point, or the commit's COMMITTED/ABORTED
        """
        self._cursor += 1
        if self._cursor >= len(self.point_table):
            return self._commit()
        self.state = CaptureState.AWAITING_COARSE
        return self._await_current(kind)

    def _commit(self) -> TourAction:
        """Run the commit checks and write the CSVs, or fail loud and write nothing.

        Four directly clicked corner points take priority over a fit: they are
        written verbatim like the 'c' mode, and any other clicked points only
        reach the sidecar.

        :return: COMMITTED with the four corners, or ABORTED on any failure
        """
        clicked_items = sorted(self._clicked.items())  # (tour index, pixel), tour order
        corners_clicked = all(slot in self._clicked for slot in range(4))

        if corners_clicked:
            # Direct path: write the four clicked corners exactly as the 'c' mode
            # does, no fit. The sidecar records every clicked point with no rms.
            if len(self._clicked) > 4:
                print(f"  frame {self.frame_idx}: all 4 corners clicked directly, so no fit ran; "
                      f"the {len(self._clicked) - 4} extra point(s) went to the sidecar only")
            corners_px = [self._clicked[slot] for slot in range(4)]
            append_rows(
                self.csv_path,
                build_corner_rows(self.video, self.frame_idx, corners_px, self.width, self.height, self.orientation),
            )
            self._write_sidecar(clicked_items, rms_px=None)
            return self._committed(tuple(corners_px))

        if len(self._clicked) < 5:
            # Four correspondences fit a homography exactly, so a 4-point fit has
            # zero residual even on garbage clicks and the rms guard goes blind.
            return self._fail(f"need the 4 corners clicked or at least 5 points; only {len(self._clicked)} clicked")

        plan_pts = np.array([self.point_table[idx].court_xy for idx, _ in clicked_items], dtype=np.float64)
        clicked_px = np.array([xy for _, xy in clicked_items], dtype=np.float64)
        if not _has_general_position_quad(plan_pts):
            return self._fail("clicked points are degenerate (no 4 with 3 non-collinear); cannot fit a homography")

        corners, rms, reason = fit_corner_quad(plan_pts, clicked_px)
        if corners is None:
            return self._fail(reason)
        corner_tuples = [(float(corner[0]), float(corner[1])) for corner in corners]
        append_rows(
            self.csv_path,
            build_corner_rows(self.video, self.frame_idx, corner_tuples, self.width, self.height, self.orientation),
        )
        self._write_sidecar(clicked_items, rms_px=rms)
        return self._committed(tuple(corner_tuples))

    def _write_sidecar(self, clicked_items: list[tuple[int, tuple[float, float]]], rms_px: float | None) -> None:
        """Append one sidecar row per clicked point beside the main CSV.

        :param clicked_items: (tour index, pixel) per clicked point, tour order
        :param rms_px: fit RMS when a fit ran, None for the direct-corners path
        """
        clicked = [(self.point_table[idx].name, xy[0], xy[1]) for idx, xy in clicked_items]
        append_point_rows(self._sidecar_path(), build_point_rows(self.video, self.frame_idx, clicked, rms_px))

    def _sidecar_path(self) -> Path:
        """:return: the points sidecar path, ``<stem>_points.csv`` beside the main CSV."""
        return self.csv_path.with_name(self.csv_path.stem + "_points.csv")

    def _committed(self, corners: tuple[tuple[float, float], ...]) -> TourAction:
        """Reset to scrub after a good commit and report it.

        :param corners: the four committed corners, TL TR BR BL
        """
        clicked, skipped = len(self._clicked), len(self._skipped)
        self._reset()
        return TourAction(ActionKind.COMMITTED, clicked=clicked, skipped=skipped, corners=corners)

    def _fail(self, message: str) -> TourAction:
        """Print the cause, write nothing, and drop back to scrub (fail loud).

        :param message: the reason the commit was rejected
        """
        print(f"  tour aborted on frame {self.frame_idx}: {message}")
        self._reset()
        return TourAction(ActionKind.ABORTED, message=message)

    def _reset(self) -> None:
        """Clear tour progress and return to scrub; frame_idx is kept for logging."""
        self._cursor = 0
        self._clicked = {}
        self._skipped = set()
        self._pending_origin = None
        self._pending_coarse = None
        self.state = CaptureState.SCRUB


# --- cv2 helpers shared with the off-frame tool -----------------------------

def render_loupe(frame: np.ndarray, action: SessionAction | TourAction, half: int, zoom: int) -> np.ndarray:
    """Build the zoomed loupe image for a coarse click, with a centre crosshair.

    :param frame: the current BGR frame
    :param action: the OPEN_LOUPE action carrying the crop origin and coarse click
        (from either capture mode; both carry the loupe fields)
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


