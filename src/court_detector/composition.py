"""Compose one scene's court from the best-painted markings across its searched frames.

Players hide different markings in different frames, so one frame's court can miss
paint that another frame shows. After a fresh search finds the middle frame's court,
the detector also searches the first and last frames of the middle frame's feet
window. This module then fits one court from whichever accepted frame shows each
marking best, and checks it in the middle frame.

1. Reference. The accepted frame with the highest own-frame score, the net choice's
   paint and line blend plus its net-post bonus, supplies coordinates and the fit's
   starting court. Exact ties go middle, first, last.
2. Alignment. Each other accepted frame is ECC-aligned to the reference inside its own
   court, with both frames' person boxes left out. A warp is usable when its
   correlation reaches the reuse check's level; camera movement is allowed.
3. Orientation. A court turned 180 degrees against the reference gets its corners
   rolled by two, so a marking name means the same painted line in every frame.
4. Donors. Each marking comes from the used court with the most q_paint10 on it. Its
   observed fragment samples outside person boxes are carried into reference pixels,
   each weighted by stripe_fitting.prepare's weight times the donor's q_paint10.
5. Fit. stripe_fitting.refine fits one court to the donated samples in the reference,
   and stripe_refit.fit_geometry checks the fit.
6. Middle frame. The court is carried into the middle frame and checked there: hard
   validity, camera, players' feet when required and the upright camera when on. Its
   paint support is measured there, on the final stripe refit's scale.

A failed step returns no composite, and the middle frame's own court stands.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np

from annotator import court_views
from shared.court import HOMOGRAPHY_RESOLUTION
from shared.court_model import CORNER_COURT_M

from . import net_choice, reuse, stripe_fitting, stripe_refit
from .detect import NET_OVERRUN_WORKING_PX, NET_WEIGHT
from .geometry import project
from .line_observations import MARKINGS
from .measurements import observable_points
from .paint_geometry import CENTRE_SEGMENTS_M

if TYPE_CHECKING:
    from .detect import LiveModules
    from .measurements import ViewContext

MIDDLE = "middle"
ENDPOINT_ROLES = ("first", "last")
# Every choice between frames breaks exact ties in this order.
FRAME_ROLES = (MIDDLE, *ENDPOINT_ROLES)
COMPOSITE_KEY = "composite"  # CourtResult.chosen_key of a composite court
# Rolling TL TR BR BL corners by two describes the same court turned 180 degrees.
HALF_TURN_ROLL = 2


@dataclass(frozen=True)
class SearchedFrame:
    """One scene frame's accepted court from a fresh search, with the inputs it was measured on."""

    role: str  # one of FRAME_ROLES
    native_frame: np.ndarray  # (height, width, 3) BGR, read-only
    context: ViewContext  # frozen; this frame's own lines and person boxes
    corners_native: np.ndarray  # (4, 2) the frame's accepted court
    paint_score: float | None  # that court's final stripe-refit paint support
    # measure_candidate's cache for this context, keyed by the homography's bytes
    evidence_cache: dict = field(default_factory=dict)

    @property
    def native_per_working(self) -> np.ndarray:
        return native_per_working(self.context)


@dataclass(frozen=True)
class UsedFrame:
    """An aligned frame, with its court in the reference court's orientation."""

    frame: SearchedFrame
    to_reference: np.ndarray  # (3, 3) this frame's working px to the reference's; identity for the reference
    corners_native: np.ndarray  # (4, 2) the frame's court in its native px, reordered if turned
    evidence: dict[str, Any]  # measure_candidate of that court in this frame


class Composite(NamedTuple):
    corners_native_px: np.ndarray  # (4, 2) in the middle frame's native px and its own court's corner order
    paint_score: float | None  # q_paint10_span_weighted measured in the middle frame
    # The aligned frames the composite was fitted from, in own-frame score order. view_pool.py
    # takes their donated samples without measuring them again.
    used_frames: tuple[UsedFrame, ...] = ()


def native_per_working(context: ViewContext) -> np.ndarray:
    """Native px per working px, one scale per axis."""
    return np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)


def court_homography(frame: SearchedFrame, corners_native: np.ndarray) -> np.ndarray:
    """Court metres to the frame's working px, built as the detector builds a court's evidence homography."""
    working = np.asarray(corners_native, dtype=float) / frame.native_per_working
    return cv2.getPerspectiveTransform(CORNER_COURT_M, working.astype(np.float32)).astype(float)


def measure(live: LiveModules, frame: SearchedFrame, corners_native: np.ndarray) -> dict[str, Any]:
    """A court's paint and line evidence and fragment assignments in one frame."""
    entry = {"homography_working": court_homography(frame, corners_native)}
    with live.prepared_measurements(live.verifier):
        evidence, _ = live.verifier.measure_candidate(frame.context, entry, frame.evidence_cache)
    return evidence


def carry(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Points through a homography, keeping their array shape."""
    points = np.asarray(points, dtype=float)
    moved, _ = project(np.asarray(homography, dtype=float)[None], points)
    return moved.reshape(points.shape)


def score_parts(paint: float, geometry: float, net_reward: float, geometry_weight: float) -> dict[str, float]:
    """The net choice's combined score: paint blended with line support, plus the weighted post reward."""
    evidence_value = net_choice.evidence_score({"paint_score": paint, "geometry_score": geometry}, geometry_weight)
    net_bonus = NET_WEIGHT * net_reward
    return {"evidence_score": evidence_value, "net_bonus": net_bonus, "combined_score": evidence_value + net_bonus}


def own_frame_score(live: LiveModules, frame: SearchedFrame, geometry_weight: float) -> dict[str, Any]:
    """A frame's court scored in its own frame. A missing paint or geometry score leaves no combined score."""
    evidence = measure(live, frame, frame.corners_native)
    net_state, posts = net_choice.net_posts(frame.corners_native, frame.context)
    row: dict[str, Any] = {"role": frame.role, "paint_score": frame.paint_score,
                           "geometry_score": evidence["q_geom_span_weighted"], "net_state": net_state,
                           "net_reward": net_choice.net_reward(net_state, posts, NET_OVERRUN_WORKING_PX)}
    if row["paint_score"] is not None and row["geometry_score"] is not None:
        row.update(score_parts(row["paint_score"], row["geometry_score"], row["net_reward"], geometry_weight))
    return row


def to_reference_working(warp_view: np.ndarray, working_size: tuple[int, int]) -> np.ndarray:
    """An ECC warp between VIEW_RESOLUTION images, as the same warp between working images.

    Working and view pixels differ by one scale per axis, the same for every frame of a video.
    """
    working_per_view = np.asarray(working_size, dtype=float) / np.asarray(court_views.VIEW_RESOLUTION, dtype=float)
    scale = np.diag([*working_per_view, 1.0])
    return scale @ np.asarray(warp_view, dtype=float) @ np.linalg.inv(scale)


def without_people(mask: np.ndarray, frame: SearchedFrame) -> np.ndarray:
    """Clear the frame's person boxes from a VIEW_RESOLUTION mask of that same frame, rounding outward."""
    view_per_working = np.asarray(court_views.VIEW_RESOLUTION, dtype=float) / np.asarray(frame.context.size, dtype=float)
    boxes = frame.context.mask_boxes * np.tile(view_per_working, 2)
    outward = np.column_stack((np.floor(boxes[:, :2]), np.ceil(boxes[:, 2:]))).astype(int)
    for x1, y1, x2, y2 in outward.tolist():
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, cv2.FILLED)
    return mask


def view_alignment_without_people(frame: SearchedFrame,
                                  reference: SearchedFrame) -> tuple[court_views.ViewAlignment | None, float]:
    """court_views.measure_view_alignment's ECC, with each image's person boxes cut from its own mask.

    Players move between samples, so their pixels disagree even when the camera stays still.
    The template (this frame) keeps measure_view_alignment's court polygon, less its own boxes.
    That polygon comes from this frame's court, so it sits in this frame's pixels. The input
    (the reference) keeps every pixel outside its own boxes. On each step ECC carries the
    input mask into template pixels through the current warp and uses pixels valid in both,
    so a moved camera moves the reference's boxes with it. A mask with too little left makes
    ECC fail, which reads as unmeasurable, as in measure_view_alignment.

    :return: The alignment, or None when ECC cannot measure one. Then the share of the court
        polygon valid in both masks at the identity warp, where ECC starts.
    """
    view_per_refpx = np.asarray(court_views.VIEW_RESOLUTION) / np.asarray(HOMOGRAPHY_RESOLUTION)
    native_size = np.asarray(frame.context.native_size, dtype=float)
    corners_refpx = np.asarray(frame.corners_native, dtype=float) * np.asarray(HOMOGRAPHY_RESOLUTION) / native_size
    corners = corners_refpx * view_per_refpx
    centre = corners.mean(axis=0)
    template_image, input_image = reuse.view_image(frame.native_frame), reuse.view_image(reference.native_frame)
    court = np.zeros(template_image.shape, np.uint8)
    polygon = centre + court_views.ALIGNMENT_MASK_SCALE * (corners - centre)
    cv2.fillConvexPoly(court, np.rint(polygon).astype(np.int32), 255)
    template_mask = without_people(court.copy(), frame)
    input_mask = without_people(np.full(input_image.shape, 255, np.uint8), reference)
    kept = float(np.count_nonzero(template_mask & input_mask) / np.count_nonzero(court))
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, court_views.ALIGNMENT_ITERATIONS,
                court_views.ALIGNMENT_EPSILON)
    try:
        correlation, warp = cv2.findTransformECCWithMask(
            template_image, input_image, template_mask, input_mask, np.eye(3, dtype=np.float32),
            cv2.MOTION_HOMOGRAPHY, criteria, court_views.ALIGNMENT_BLUR_SIZE,
        )
    except cv2.error:
        return None, kept
    moved = cv2.perspectiveTransform(corners[None].astype(np.float32), warp)[0]
    shift = np.linalg.norm((moved - corners) / view_per_refpx, axis=1).max()
    return court_views.ViewAlignment(correlation, warp, moved / view_per_refpx, shift), kept


def align(frame: SearchedFrame, reference: SearchedFrame) -> tuple[dict[str, Any], np.ndarray | None]:
    """Align a frame's image to the reference's inside the frame's own court, without people.

    :return: The alignment record, and the frame-to-reference working homography when usable.
    """
    # The warp maps the template (this frame) to the input (the reference).
    alignment, kept = view_alignment_without_people(frame, reference)
    if alignment is None:
        return {"mask_kept_fraction": kept, "usable": False, "skip_reason": "alignment_unmeasurable"}, None
    correlation = float(alignment.correlation)
    usable = correlation >= court_views.MIN_ALIGNMENT_CORRELATION
    record = {"correlation": correlation, "max_corner_shift_refpx": float(alignment.shift_refpx),
              "same_camera": alignment.matches, "mask_kept_fraction": kept, "usable": usable,
              "skip_reason": None if usable else "correlation_below_reuse_level"}
    return record, to_reference_working(alignment.warp, frame.context.size) if usable else None


def half_turn_roll(corners_in_reference: np.ndarray, reference_corners: np.ndarray) -> int:
    """0 when a court's corner order matches the reference court's, 2 when it is turned 180 degrees.

    Both courts' corners are in reference px. An exact tie keeps the saved order.
    """
    as_saved = np.linalg.norm(corners_in_reference - reference_corners, axis=1).max()
    turned = np.linalg.norm(np.roll(corners_in_reference, HALF_TURN_ROLL, axis=0) - reference_corners, axis=1).max()
    return HALF_TURN_ROLL if turned < as_saved else 0


def use_frame(live: LiveModules, frame: SearchedFrame, to_reference: np.ndarray,
              reference_corners_working: np.ndarray) -> tuple[UsedFrame, int]:
    """An aligned frame's court in the reference orientation, measured again in its own frame."""
    own = np.asarray(frame.corners_native, dtype=float)
    roll = half_turn_roll(carry(own / frame.native_per_working, to_reference), reference_corners_working)
    corners = np.roll(own, roll, axis=0)
    return UsedFrame(frame, to_reference, corners, measure(live, frame, corners)), roll


def corners_between(corners_native: np.ndarray, source: UsedFrame, target: UsedFrame) -> np.ndarray:
    """A court's corners carried from one used frame's native px to another's, through the reference."""
    if source is target:
        return np.asarray(corners_native, dtype=float)
    working = np.asarray(corners_native, dtype=float) / source.frame.native_per_working
    return carry(working, np.linalg.inv(target.to_reference) @ source.to_reference) * target.frame.native_per_working


def choose_donors(used: list[UsedFrame]) -> list[UsedFrame | None]:
    """Each marking's donor: the used court with the most q_paint10 on it.

    A marking without positive q_paint10 anywhere has no donor. used comes in own-frame
    score order, so the first of equal values wins.
    """
    donors: list[UsedFrame | None] = []
    for marking in range(len(MARKINGS)):
        best, best_q_paint = None, 0.0
        for candidate in used:
            q_paint = candidate.evidence["markings"][marking]["q_paint10"]
            if q_paint is not None and q_paint > best_q_paint:
                best, best_q_paint = candidate, q_paint
        donors.append(best)
    return donors


def donated_constraints(donor: UsedFrame, markings: list[int]) -> tuple[stripe_fitting.Constraints, dict[int, dict]]:
    """The donor's observed fragment samples on the given markings, in reference working px.

    stripe_fitting.prepare picks each assigned fragment's samples near its marking, with
    their geometrically assigned stripe positions. This fit adds no colour-polarity
    relabelling. Samples inside a person box are dropped. Each kept weight is prepare's weight times the donor's q_paint10 on that marking.

    :return: The constraints, and per marking index its kept and occluded samples and total weight.
    """
    context = donor.frame.context
    assignments = donor.evidence["stripe_assignments"]
    prepared = stripe_fitting.prepare(court_homography(donor.frame, donor.corners_native), context.observations,
                                      assignments, context.weights, centres=CENTRE_SEGMENTS_M)
    index_by_id = {int(raw_id): index for index, raw_id in enumerate(context.observations.fragment_ids)}
    # One marking index per prepared sample, from its fragment's assignment.
    sample_markings = np.asarray([assignments["marking"][index_by_id[int(raw_id)]]
                                  for raw_id in prepared.fragment_ids], dtype=int)
    donated = np.isin(sample_markings, markings)
    visible = observable_points(prepared.points, context.size, context.mask_boxes)
    kept = donated & visible
    q_paint = np.zeros(len(sample_markings))
    for marking in markings:
        q_paint[sample_markings == marking] = donor.evidence["markings"][marking]["q_paint10"]
    weights = prepared.weights * q_paint
    summary = {}
    for marking in markings:
        on_marking = sample_markings == marking
        summary[marking] = {"samples": int((on_marking & kept).sum()),
                            "occluded_samples": int((on_marking & ~visible).sum()),
                            "weight": float(weights[on_marking & kept].sum())}
    constraints = stripe_fitting.Constraints(
        carry(prepared.points[kept], donor.to_reference), prepared.intervals[kept], prepared.positions[kept],
        weights[kept], prepared.fragment_ids[kept], prepared.sample_ids[kept],
    )
    return constraints, summary


def joined(parts: list[stripe_fitting.Constraints]) -> stripe_fitting.Constraints:
    arrays = [np.concatenate([getattr(part, item.name) for part in parts])
              for item in dataclasses.fields(stripe_fitting.Constraints)]
    return stripe_fitting.Constraints(*arrays)


def donated_samples(used: list[UsedFrame]) -> tuple[list[dict[str, Any]], stripe_fitting.Constraints]:
    """Choose each marking's donor and join every donated sample into one set of constraints.

    :return: One row per marking naming its donor frame, and the joined constraints in reference px.
    """
    donors = choose_donors(used)
    parts, summaries = [], {}
    for candidate in used:
        # Every used frame adds its part, possibly empty, so the join always has one.
        markings = [marking for marking, donor in enumerate(donors) if donor is candidate]
        part, summary = donated_constraints(candidate, markings)
        parts.append(part)
        summaries.update(summary)
    rows = []
    for marking, name in enumerate(MARKINGS):
        donor = donors[marking]
        if donor is None:
            rows.append({"marking": name, "donor_role": None, "q_paint10": None, "samples": 0,
                         "occluded_samples": 0, "weight": 0.0})
            continue
        rows.append({"marking": name, "donor_role": donor.frame.role,
                     "q_paint10": donor.evidence["markings"][marking]["q_paint10"], **summaries[marking]})
    return rows, joined(parts)


def fit_in_reference(live: LiveModules, reference: UsedFrame, constraints: stripe_fitting.Constraints) -> dict[str, Any]:
    """Fit one court to the donated samples, starting from the reference court, in reference px.

    :return: stripe_refit.fit_geometry's record: the solver, rank, depth, convexity and
        hard-validity checks, and the fitted corners in the reference's native px.
    """
    context = reference.frame.context
    start = reference.corners_native / reference.frame.native_per_working
    fit = stripe_fitting.refine(start, constraints, context.size, use_positions=True, centres=CENTRE_SEGMENTS_M)
    with live.prepared_measurements(live.verifier):
        return stripe_refit.fit_geometry(fit, context, live.verifier, live.runtime, live.scoring.view_line_maps(context))


def check_in_frame(live: LiveModules, context: ViewContext, corners_native: np.ndarray, *, require_people: bool,
                   max_horizon_tilt_deg: float | None) -> tuple[dict[str, Any] | None, str | None]:
    """A court's final measurement in one frame, checked as the final stripe refit and reuse check courts.

    It needs only the frame's context, so callers can check a court without keeping the native image.

    :param require_people: Also require the players' feet on the court.
    :param max_horizon_tilt_deg: The upright-camera limit; None allows any camera roll.
    :return: stripe_refit.describe's measurement, or None when the corners cannot be measured,
        and the first failed check's reason, or None when every check passes.
    """
    working = np.asarray(corners_native, dtype=float) / native_per_working(context)
    if not np.isfinite(working).all() or not live.verifier.convex_corners(working):
        return None, "non_finite_or_non_convex_corners"
    homography = cv2.getPerspectiveTransform(CORNER_COURT_M, working.astype(np.float32)).astype(float)
    with live.prepared_measurements(live.verifier):
        measurement = stripe_refit.describe(working, context, live.verifier, live.runtime,
                                            live.scoring.view_line_maps(context))
    valid, reason = live.verifier.hard_validity({"homography_working": homography, "gates": measurement["gates"]})
    if not valid:
        return measurement, reason
    historical = measurement["historical"]
    upright = max_horizon_tilt_deg is None or reuse.upright_court(homography, working, context.size,
                                                                  max_horizon_tilt_deg)
    checks = (
        ("camera_implausible", historical["historical_camera"]),
        ("players_not_on_court", not require_people or historical["historical_fullcourt"]),
        ("camera_not_upright", upright),
    )
    for reason, passed in checks:
        if not passed:
            return measurement, reason
    return measurement, None


def fallback(record: dict[str, Any], reason: str) -> tuple[None, dict[str, Any]]:
    record["fallback_reason"] = reason
    return None, record


def compose_scene(live: LiveModules, frames: Sequence[SearchedFrame], *, geometry_weight: float,
                  require_people: bool, max_horizon_tilt_deg: float | None) -> tuple[Composite | None, dict[str, Any]]:
    """Compose one court from a scene's accepted frames and check it in the middle frame.

    :param frames: The accepted frames, one per role, the middle frame among them.
    :param geometry_weight: The net choice's share of the geometry score.
    :param require_people: Require the middle frame's feet on the composite, as the final refit does.
    :param max_horizon_tilt_deg: The upright-camera limit; None allows any camera roll.
    :return: The composite, or None when the middle frame's own court should stand; and a
        record of each step, whose fallback_reason says why no composite was returned.
    """
    record: dict[str, Any] = {"fallback_reason": None}
    if len(frames) < 2:
        return fallback(record, "too_few_accepted_frames")
    scores = [own_frame_score(live, frame, geometry_weight) for frame in frames]
    record["scores"] = scores
    if any("combined_score" not in row for row in scores):
        # Ranking only the scored frames would choose the reference from a smaller set.
        return fallback(record, "score_evidence_missing")
    order = sorted(range(len(frames)),
                   key=lambda index: (-scores[index]["combined_score"], FRAME_ROLES.index(frames[index].role)))
    ranked = [frames[index] for index in order]
    reference = ranked[0]
    record["reference"] = reference.role
    reference_working = reference.corners_native / reference.native_per_working

    used, alignments, rolls = [], {}, {}
    for frame in ranked:
        to_reference: np.ndarray | None = np.eye(3)
        if frame is not reference:
            alignments[frame.role], to_reference = align(frame, reference)
        if to_reference is None:
            continue
        item, rolls[frame.role] = use_frame(live, frame, to_reference, reference_working)
        used.append(item)
    record.update(alignments=alignments, half_turn_rolls=rolls, used_frames=[item.frame.role for item in used])
    if len(used) < 2:
        return fallback(record, "too_few_aligned_frames")
    middle = next((item for item in used if item.frame.role == MIDDLE), None)
    # Only an aligned middle frame gives a warp for middle-frame coordinates.
    if middle is None:
        return fallback(record, "middle_not_aligned")

    record["markings"], constraints = donated_samples(used)
    fit = fit_in_reference(live, used[0], constraints)
    record["fit"] = {"status": fit["status"], "sample_count": len(constraints.points), "valid": fit["valid"],
                     "validity_reason": fit["validity_reason"],
                     "corners_reference_native_px": fit.get("corners_native_px")}
    if not fit["valid"]:
        return fallback(record, f"fit_{fit['validity_reason']}")

    carried = corners_between(np.asarray(fit["corners_native_px"]), used[0], middle)
    # Back into the corner order of the middle frame's own court. Rolling by two undoes itself.
    corners = np.roll(carried, rolls[MIDDLE], axis=0)
    # The middle frame is checked even when it is the reference, so every composite passes the same checks.
    measurement, reason = check_in_frame(live, middle.frame.context, corners, require_people=require_people,
                                         max_horizon_tilt_deg=max_horizon_tilt_deg)
    record["middle"] = {"corners_native_px": corners.tolist(), "rejection": reason, "measurement": measurement}
    if reason is not None:
        return fallback(record, f"middle_{reason}")
    assert measurement is not None  # a measured court is the only one that can pass
    return Composite(corners, measurement["paint_score"], tuple(used)), record
