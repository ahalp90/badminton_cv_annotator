"""Reuse an earlier view's court when the same camera view returns.

try_reuse aligns a new view's image with an earlier view that has an accepted court,
using court_views' image alignment and its same-camera test. It carries the court's
corners through the fitted image warp, refits them to the new view's painted stripes
(stripe_refit) and checks the result as the detector checks courts: hard validity,
camera, players' feet and, when on, the upright-camera filter. It then compares the
paint support with the earlier court's and bounds how far the refit moved the court.

A failed step returns no court, so the caller runs the full search. Programming errors
propagate. MIN_PAINT_RATIO and MAX_REFIT_SHIFT_M are placeholders until measured on
known same-camera pairs on Carmack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np

from annotator import court_views
from shared.court import HOMOGRAPHY_RESOLUTION
from shared.court_model import COURT_LENGTH_M, COURT_WIDTH_M

from . import candidate_pool, proposals, stripe_refit
from .geometry import CORNER_COURT_M, project
from .paint_geometry import SIDE_PAINT_GAP_M, STRIPE_WIDTH_M

if TYPE_CHECKING:
    from .detect import LiveModules
    from .measurements import ViewContext

# UNVALIDATED placeholder. The reused court's final paint support must keep this share of
# the earlier court's.
MIN_PAINT_RATIO = 0.8
# UNVALIDATED placeholder. The closest parallel stripes, a doubles and a singles sideline,
# sit 0.46 m apart centre to centre. Half that separates a refit that stayed on the same
# stripes from one that slid onto the neighbouring stripe. It does not measure fit precision.
MAX_REFIT_SHIFT_M = (STRIPE_WIDTH_M + SIDE_PAINT_GAP_M) / 2
# Where refit movement is measured: every quarter of the court's width and eighth of its
# length, so the far baseline counts as much as the near one.
_ACROSS_M, _ALONG_M = np.meshgrid(np.linspace(0, COURT_WIDTH_M, 5), np.linspace(0, COURT_LENGTH_M, 9))
FLOOR_POINTS_M = np.column_stack((_ACROSS_M.ravel(), _ALONG_M.ravel()))
REUSE_KEY = "reuse"


@dataclass(frozen=True)
class KnownCourt:
    """A court accepted in an earlier view, kept to test later views against."""

    view_id: str
    image: np.ndarray  # (540, 960) uint8 grey frame at court_views.VIEW_RESOLUTION
    native_size: tuple[int, int]  # (width, height) of the view's frame
    corners_native_px: np.ndarray  # (4, 2) in CORNER_COURT_M order
    paint_score: float  # the accepted court's final paint support, on ReusedCourt.paint_score's scale


@dataclass(frozen=True)
class ReusedCourt:
    """A known court refitted to a new view's stripes that passed every check."""

    source_view_id: str
    corners_native_px: np.ndarray  # (4, 2) in the new view's native pixels
    paint_score: float  # the refitted court's q_paint10_span_weighted
    paint_ratio: float  # paint_score over the known court's
    max_floor_shift_m: float  # how far the refit moved the court, on the refitted court's floor


class ReuseAttempt(NamedTuple):
    court: ReusedCourt | None  # None: the caller runs the full search
    record: dict[str, Any]  # every value measured and the rejection reason, for artefacts and tuning


def view_image(native_frame: np.ndarray) -> np.ndarray:
    """The greyscale image court_views aligns: the BGR frame shrunk to VIEW_RESOLUTION."""
    small = cv2.resize(native_frame, court_views.VIEW_RESOLUTION, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def make_known_court(
    view_id: str, frame: np.ndarray, corners_native_px: np.ndarray, paint_score: float, *,
    alignment_image: np.ndarray | None = None,
) -> KnownCourt:
    """Keep an accepted court and its view's image for later reuse.

    :param frame: The view's BGR frame at native size, the one the court was found in.
    :param paint_score: CourtResult.paint_score: the final refit's q_paint10_span_weighted.
    :param alignment_image: Optional greyscale VIEW_RESOLUTION image for alignment.
    """
    if not paint_score > 0:
        raise ValueError(f"{view_id}: reuse compares paint support, so the known court needs positive paint, "
                         f"not {paint_score}")
    corners = np.array(corners_native_px, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError(f"{view_id}: known court corners must be four finite points, not {corners.tolist()}")
    image = view_image(frame) if alignment_image is None else alignment_image
    # Later views share these arrays, so a stray in-place write should fail loudly.
    corners.flags.writeable = False
    image.flags.writeable = False
    return KnownCourt(view_id, image, (frame.shape[1], frame.shape[0]), corners, float(paint_score))


def align_known_court(
    known: KnownCourt, native_frame: np.ndarray, alignment_image: np.ndarray | None = None,
) -> court_views.ViewAlignment | None:
    """Align the new view's frame to the known court's view inside the known court."""
    corners_refpx = known.corners_native_px * np.asarray(HOMOGRAPHY_RESOLUTION) / np.asarray(known.native_size)
    image = view_image(native_frame) if alignment_image is None else alignment_image
    return court_views.measure_view_alignment(known.image, image, corners_refpx)


def moved_corners_native(alignment: court_views.ViewAlignment, native_size: tuple[int, int]) -> np.ndarray:
    """The known court's corners carried into the new view, in its native pixels."""
    return alignment.moved_corners_refpx * np.asarray(native_size) / np.asarray(HOMOGRAPHY_RESOLUTION)


def floor_shift_m(before_working: np.ndarray, after_working: np.ndarray) -> tuple[float, np.ndarray]:
    """How far a refit moved a court, in metres on the refitted court's floor.

    Carry each FLOOR_POINTS_M point into the image with the earlier homography, then back
    to the floor with the refitted one. One image pixel covers more floor at the far end,
    so an equal pixel movement there counts for more metres.

    :param before_working: Court metres to working pixels before the refit.
    :param after_working: Court metres to working pixels after the refit.
    :return: The largest movement, and the floor point in metres where it happens.
    """
    round_trip = np.linalg.inv(after_working) @ before_working
    moved_m, _ = project(round_trip[None], FLOOR_POINTS_M)
    distances = np.linalg.norm(moved_m[0] - FLOOR_POINTS_M, axis=1)
    worst = int(distances.argmax())
    return float(distances[worst]), FLOOR_POINTS_M[worst]


def upright_court(
    homography_working: np.ndarray, corners_working: np.ndarray, size: tuple[int, int], max_horizon_tilt_deg: float,
) -> bool:
    """Apply the search's upright-camera filter to one court.

    The search tests a direction pair's two vanishing points. A court's own vanishing
    points are the images of its two floor directions, its homography's first two
    columns. For a court the search built, they are that pair's points.
    """
    vanishing_points = homography_working[:, :2].T  # (2, 3) homogeneous working px
    tilt = candidate_pool.horizon_tilt_deg(vanishing_points, size)
    if tilt is not None and tilt > max_horizon_tilt_deg:
        return False
    return bool(proposals.below_horizon(vanishing_points, corners_working[None], size)[0])


def rejected(record: dict[str, Any], reason: str) -> ReuseAttempt:
    record["rejection"] = reason
    return ReuseAttempt(None, record)


def try_reuse(
    known: KnownCourt, context: ViewContext, native_frame: np.ndarray, live: LiveModules, *,
    max_horizon_tilt_deg: float | None,
    require_people: bool = True,
    alignment_image: np.ndarray | None = None,
) -> ReuseAttempt:
    """Carry a known court into a new view, refit it to that view's stripes and check it.

    :param context: The new view's frozen context, built from its own lines, person
        boxes and standing feet.
    :param native_frame: The new view's BGR frame at native size.
    :param alignment_image: Optional greyscale VIEW_RESOLUTION image for alignment.
    :param max_horizon_tilt_deg: The detector's upright-camera limit. None only when the
        detector allows any camera roll, as Switches.upright_camera=False does.
    :return: The reused court, or None; the record says why and holds each measured value.
    """
    record: dict[str, Any] = {"source_view_id": known.view_id, "rejection": None}
    alignment = align_known_court(known, native_frame, alignment_image)
    if alignment is None:
        return rejected(record, "alignment_unmeasurable")
    record["alignment"] = {"correlation": float(alignment.correlation), "shift_refpx": float(alignment.shift_refpx)}
    if not alignment.matches:
        return rejected(record, "alignment_mismatch")

    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    moved_native = moved_corners_native(alignment, context.native_size)
    moved_working = moved_native / scale
    homography = cv2.getPerspectiveTransform(CORNER_COURT_M.astype(np.float32),
                                             moved_working.astype(np.float32)).astype(float)
    line_maps = live.scoring.view_line_maps(context)
    zone = live.runtime["zone"]
    with live.prepared_measurements(live.verifier):
        gates = live.runtime["gate_evidence"](moved_native, context.source, scale, context.size, context.families,
                                              line_maps, zone)
        # The moved court acts as the chosen parent in refit_chosen's scoring record.
        moved = {"origin_key": REUSE_KEY, "kind": "parent", "corners_px": moved_native.tolist(),
                 "homography_working": homography.tolist(), "gates": gates}
        valid, reason = live.verifier.hard_validity(moved)
        if not valid:
            return rejected(record, f"moved_court_{reason}")
        evidence, _ = live.verifier.measure_candidate(context, moved, {})
        record["moved_paint_score"] = evidence["q_paint10_span_weighted"]
        moved["evidence"] = {"stripe_assignments": evidence["stripe_assignments"]}
        refit = stripe_refit.refit_chosen({"parents": [moved], "valid_children": []}, REUSE_KEY, context,
                                          native_frame, live.verifier, live.runtime, line_maps, replay_check=False)
    corrected = refit["corrected"]
    if not corrected["valid"]:
        return rejected(record, f"refit_{corrected['validity_reason']}")

    measurement = corrected["measurement"]
    refit_working = np.asarray(corrected["corners_working_px"])
    refit_homography = np.asarray(corrected["homography_working"])
    shift_m, shift_at_m = floor_shift_m(homography, refit_homography)
    paint_score = measurement["paint_score"]
    paint_ratio = None if paint_score is None else paint_score / known.paint_score
    upright = max_horizon_tilt_deg is None or upright_court(
        refit_homography, refit_working, context.size, max_horizon_tilt_deg,
    )
    historical = measurement["historical"]
    record.update({
        "paint_score": paint_score, "paint_ratio": paint_ratio, "max_floor_shift_m": shift_m,
        "max_floor_shift_at_m": shift_at_m.tolist(),
        "max_corner_shift_working_px": float(np.linalg.norm(refit_working - moved_working, axis=1).max()),
        "historical": historical, "gates": measurement["gates"], "upright": upright,
    })
    # Record every value before deciding, so tuning can see how close each rejected view came.
    checks = (
        ("camera_implausible", historical["historical_camera"]),
        ("players_not_on_court", not require_people or historical["historical_fullcourt"]),
        ("camera_not_upright", upright),
        ("refit_moved_court", shift_m <= MAX_REFIT_SHIFT_M),
    )
    for reason, passed in checks:
        if not passed:
            return rejected(record, reason)
    # No visible marking in one direction leaves the paint score undefined.
    if paint_ratio is None:
        return rejected(record, "no_paint_support")
    if paint_ratio < MIN_PAINT_RATIO:
        return rejected(record, "paint_support_dropped")
    court = ReusedCourt(known.view_id, np.asarray(corrected["corners_native_px"]), paint_score, paint_ratio, shift_m)
    return ReuseAttempt(court, record)
