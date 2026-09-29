"""Compose one court per scene from the best-painted markings across full_three's accepted frames.

Players hide different markings in different frames, so each saved court can miss
paint that another frame shows. This experiment fits one extra court per scene from
the observed line samples of whichever accepted frame shows each marking best. It
then scores the saved courts and the composite on the same frames.

1. Reference. rescore.py's ``ranked`` choice, the net choice's combined score on the
   saved courts, sets the reference frame. It supplies coordinates and the fit's
   starting court, nothing more.
2. Alignment. Each other accepted frame is ECC-aligned to the reference inside its
   own saved court, as ``in_middle_frame`` does, but with both frames' person boxes left
   out. A warp is usable when its correlation reaches the reuse check's level; camera
   movement is allowed. Other frames are skipped.
3. Orientation. A court turned 180 degrees against the reference gets its corners
   rolled by two before measuring, so a marking name means the same painted line in
   every frame.
4. Evidence. Each used court's marking evidence and fragment assignments are measured
   again in its own frame, from the cached PNG, lines and same-frame person boxes.
5. Donors. Each marking comes from the used court with the most q_paint10 on it. Its
   observed fragment samples (stripe_fitting.prepare) outside person boxes are carried
   into reference pixels. Each keeps prepare's weight times the donor's q_paint10.
6. Fit. stripe_fitting.refine fits one court to the donated samples. The stripe refit's
   checks apply, plus the camera and upright checks. Player positions are not checked:
   the recovered people have boxes but no feet.
7. Comparison. The used frames' saved courts and the composite are carried into every
   used frame and scored there with the net choice's formula. The highest mean over
   those same frames wins, and an exact tie keeps a saved court.

Scores measure detector evidence, not accuracy. The source results stay untouched.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from annotator import court_views
from court_detector import reuse, stripe_fitting, stripe_refit
from court_detector.detect import (
    MAX_HORIZON_TILT_DEG,
    LiveModules,
    freeze_arrays,
    load_live_modules,
)
from court_detector.geometry import project
from court_detector.inputs import same_frame_provenance
from court_detector.line_observations import MARKINGS
from court_detector.measurements import ViewContext, observable_points
from court_detector.paint_geometry import CENTRE_SEGMENTS_M
from shared.court import HOMOGRAPHY_RESOLUTION
from shared.court_model import CORNER_COURT_M

from . import rescore
from .render import draw_outline
from .sampling import FRAME_ROLES, read_json

COMPOSE_SCHEMA = "court-scene-compose/2"
PEOPLE_SCHEMA = "court-scene-people/1"
METHOD = "full_three"
COMPOSITE = "composite"
# Rolling TL TR BR BL corners by two describes the same court turned 180 degrees.
HALF_TURN_ROLL = 2
# Saved evidence fields that a fresh measurement of the same court should reproduce.
SCORE_FIELDS = ("q_paint10_span_weighted", "q_geom_span_weighted")
MARKING_COUNT_FIELDS = ("visible_samples", "known_photometry_samples", "exclusive_fragment_count")

COMPARISON_FIELDS = (
    "video_id", "scene_id", "state", "accepted_count", "reference_role", "used_roles", "skipped_frames",
    "half_turned_roles", "reproduction_max_score_difference", "reproduction_mismatches", "donors",
    "fit_status", "fit_valid", "fit_reason", "fit_sample_count", "per_frame_scores", "common_frame_means",
    "paint_choice", "own_frame_choice", "common_frame_original", "composite_beats_originals", "final", "changed",
    "final_corners_reference_native_px",
)


@dataclass(frozen=True)
class Frame:
    """One accepted frame: its cached image, its rebuilt measurement context and its saved row."""

    saved: dict[str, Any]  # the source frame row; read only
    native_frame: np.ndarray  # (height, width, 3) BGR, the cached source PNG
    context: ViewContext
    # measure_candidate's cache for this context, keyed by the homography's bytes
    evidence_cache: dict = field(default_factory=dict)

    @property
    def role(self) -> str:
        return self.saved["role"]

    @property
    def native_per_working(self) -> np.ndarray:
        return np.asarray(self.context.native_size, dtype=float) / np.asarray(self.context.size, dtype=float)


@dataclass(frozen=True)
class UsedFrame:
    """An aligned frame, with its saved court in the reference court's orientation."""

    frame: Frame
    to_reference: np.ndarray  # (3, 3) this frame's working px to the reference's; identity for the reference
    corners_native: np.ndarray  # (4, 2) the saved court in this frame's native px, reordered if turned
    evidence: dict[str, Any]  # measure_candidate of that court in this frame


def read_people_cache(path: Path) -> dict[str, Any]:
    cache = read_json(path)
    if cache.get("schema") != PEOPLE_SCHEMA:
        raise ValueError(f"{path}: expected schema {PEOPLE_SCHEMA!r}, found {cache.get('schema')!r}")
    return cache


def candidate_label(frame_row: dict[str, Any]) -> str:
    return f"original_{frame_row['role']}"


def load_frame(live: LiveModules, saved: dict[str, Any], lines: dict[str, Any], people: dict[str, Any],
               frames_dir: Path, native_size: list[int]) -> Frame:
    """Rebuild a frame's measurement context from its cached PNG, lines and person boxes.

    The boxes were measured on the same image, so photometry hides them as the run did.
    No feet are known, so the player gates stay empty.
    """
    view_id = saved["view_id"]
    path = frames_dir / f"{view_id}.png"
    native_frame = cv2.imread(str(path))
    if native_frame is None:
        raise FileNotFoundError(path)
    height, width = native_frame.shape[:2]
    sizes = {"PNG": [width, height], "lines": lines[view_id]["native_size"], "people": people[view_id]["native_size"]}
    for name, size in sizes.items():
        if list(size) != list(native_size):
            raise ValueError(f"{view_id}: {name} size {size} differs from the video's {native_size}")
    source = {"id": view_id, "dimensions": {"width": width, "height": height},
              "segments_px": lines[view_id]["segments_native_px"], "bbox_px": people[view_id]["boxes_native_px"],
              "all_feet_px": [], "provenance": {"people_source": "not_supplied"}}
    native_frame.flags.writeable = False
    provenance = same_frame_provenance(view_id, saved["frame_index"])
    context = live.verifier.view_context(view_id, source, provenance, native_frame, view_id)
    freeze_arrays(context)
    return Frame(saved, native_frame, context)


def court_homography(frame: Frame, corners_native: np.ndarray) -> np.ndarray:
    """Court metres to the frame's working px, built as the run built it for its saved evidence."""
    working = np.asarray(corners_native, dtype=float) / frame.native_per_working
    return cv2.getPerspectiveTransform(CORNER_COURT_M, working.astype(np.float32)).astype(float)


def measure(live: LiveModules, frame: Frame, corners_native: np.ndarray) -> dict[str, Any]:
    """A court's paint and line evidence and fragment assignments in one frame."""
    entry = {"homography_working": court_homography(frame, corners_native)}
    with live.prepared_measurements(live.verifier):
        evidence, _ = live.verifier.measure_candidate(frame.context, entry, frame.evidence_cache)
    return evidence


def reproduction(saved: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    """How far a fresh measurement of a saved court is from its saved evidence.

    Recovered lines and boxes can differ from the run's, so differences are reported, not
    hidden. Score fields report their largest absolute difference. A score present on one
    side only, a differing sample count and differing occlusion awareness are listed by name.
    """
    scores = [(name, saved[name], measured[name]) for name in SCORE_FIELDS]
    mismatches = []
    if saved["photometry_occlusion_aware"] != measured["photometry_occlusion_aware"]:
        mismatches.append("photometry_occlusion_aware")
    for saved_marking, marking in zip(saved["markings"], measured["markings"], strict=True):
        name = marking["marking"]
        if saved_marking["marking"] != name:
            raise ValueError(f"saved marking {saved_marking['marking']!r} is where {name!r} belongs")
        scores.append((f"{name}.q_paint10", saved_marking["q_paint10"], marking["q_paint10"]))
        for count in MARKING_COUNT_FIELDS:
            if saved_marking[count] != marking[count]:
                mismatches.append(f"{name}.{count}")
    largest, largest_at = 0.0, None
    for name, before, after in scores:
        if before is None and after is None:
            continue
        if before is None or after is None:
            mismatches.append(name)
            continue
        if abs(after - before) > largest:
            largest, largest_at = abs(after - before), name
    return {"max_score_difference": largest, "largest_at": largest_at, "mismatches": mismatches}


def carry(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Points through a homography, keeping their array shape."""
    points = np.asarray(points, dtype=float)
    moved, _ = project(np.asarray(homography, dtype=float)[None], points)
    return moved.reshape(points.shape)


def to_reference_working(warp_view: np.ndarray, working_size: tuple[int, int]) -> np.ndarray:
    """An ECC warp between VIEW_RESOLUTION images, as the same warp between working images.

    Working and view pixels differ by one scale per axis, the same for every frame of a video.
    """
    working_per_view = np.asarray(working_size, dtype=float) / np.asarray(court_views.VIEW_RESOLUTION, dtype=float)
    scale = np.diag([*working_per_view, 1.0])
    return scale @ np.asarray(warp_view, dtype=float) @ np.linalg.inv(scale)


def without_people(mask: np.ndarray, frame: Frame) -> np.ndarray:
    """Clear the frame's person boxes from a VIEW_RESOLUTION mask of that same frame, rounding outward."""
    view_per_working = np.asarray(court_views.VIEW_RESOLUTION, dtype=float) / np.asarray(frame.context.size, dtype=float)
    boxes = frame.context.mask_boxes * np.tile(view_per_working, 2)
    outward = np.column_stack((np.floor(boxes[:, :2]), np.ceil(boxes[:, 2:]))).astype(int)
    for x1, y1, x2, y2 in outward.tolist():
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, cv2.FILLED)
    return mask


def view_alignment_without_people(frame: Frame,
                                  reference: Frame) -> tuple[court_views.ViewAlignment | None, float]:
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
    corners_refpx = np.asarray(frame.saved["corners_native_px"]) * np.asarray(HOMOGRAPHY_RESOLUTION) / native_size
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


def align(frame: Frame, reference: Frame) -> tuple[dict[str, Any], np.ndarray | None]:
    """Align a frame's image to the reference's inside the frame's saved court, without people.

    This is in_middle_frame's alignment with the reference in the middle frame's place, except
    that both images' person boxes are left out.

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


def use_frame(live: LiveModules, frame: Frame, to_reference: np.ndarray,
              reference_corners_working: np.ndarray) -> tuple[UsedFrame, int]:
    """An aligned frame's court in the reference orientation, measured again in its own frame."""
    saved = np.asarray(frame.saved["corners_native_px"], dtype=float)
    roll = half_turn_roll(carry(saved / frame.native_per_working, to_reference), reference_corners_working)
    corners = np.roll(saved, roll, axis=0)
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
    their stripe positions, as the stripe refit does. Samples inside a person box are
    dropped. Each kept weight is prepare's weight times the donor's q_paint10 on that marking.

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


def fit_composite(live: LiveModules, reference: UsedFrame, constraints: stripe_fitting.Constraints) -> dict[str, Any]:
    """Fit one court to the donated samples in reference px, then check it as reuse does.

    fit_geometry checks the solver, rank, depth, convexity and hard validity. Then the
    camera must be plausible and upright. Player positions are not checked.
    """
    context = reference.frame.context
    start = reference.corners_native / reference.frame.native_per_working
    fit = stripe_fitting.refine(start, constraints, context.size, use_positions=True, centres=CENTRE_SEGMENTS_M)
    with live.prepared_measurements(live.verifier):
        geometry = stripe_refit.fit_geometry(fit, context, live.verifier, live.runtime,
                                             live.scoring.view_line_maps(context))
    record = {"status": fit["status"], "sample_count": len(constraints.points), "fit": geometry["fit"],
              "valid": geometry["valid"], "validity_reason": geometry["validity_reason"],
              "corners_reference_native_px": geometry.get("corners_native_px")}
    if not geometry["valid"]:
        return record
    measurement = geometry["measurement"]
    upright = reuse.upright_court(np.asarray(geometry["homography_working"]),
                                  np.asarray(geometry["corners_working_px"]), context.size, MAX_HORIZON_TILT_DEG)
    record.update(camera_error=measurement["gates"]["camera_error"], upright=upright)
    if not measurement["historical"]["historical_camera"]:
        record.update(valid=False, validity_reason="camera_implausible")
    elif not upright:
        record.update(valid=False, validity_reason="camera_not_upright")
    return record


def compose_court(live: LiveModules, used: list[UsedFrame]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose each marking's donor, gather the donated samples and fit the composite.

    :return: One row per marking, and the fit record.
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
    return rows, fit_composite(live, used[0], joined(parts))


def score_in(live: LiveModules, target: UsedFrame, corners_native: np.ndarray) -> dict[str, Any]:
    """One court's combined score in one frame, measured on that frame's lines, paint and boxes."""
    evidence = measure(live, target.frame, corners_native)
    net = rescore.net_evidence(corners_native, target.frame.context)
    paint, geometry = evidence["q_paint10_span_weighted"], evidence["q_geom_span_weighted"]
    row: dict[str, Any] = {"frame_role": target.frame.role, "corners_native_px": np.asarray(corners_native).tolist(),
                           "paint_score": paint, "geometry_score": geometry, "net_state": net["net_state"],
                           "net_reward": net["net_reward"], "missing": []}
    if paint is None:
        row["missing"].append("paint_score")
    if geometry is None:
        row["missing"].append("geometry_score")
    if not row["missing"]:
        row.update(rescore.score_parts(paint, geometry, net["net_reward"]))
    return row


def evaluate(live: LiveModules, used: list[UsedFrame], skipped: list[dict[str, Any]],
             composite_corners: list | None) -> list[dict[str, Any]]:
    """Every candidate scored in every used frame, and its mean over exactly those frames.

    A candidate missing a score in any frame, or whose frame did not align, has no mean.
    """
    sources = [(candidate_label(item.frame.saved), item, item.corners_native) for item in used]
    if composite_corners is not None:
        sources.append((COMPOSITE, used[0], np.asarray(composite_corners, dtype=float)))
    rows = []
    for label, source, corners in sources:
        scores = [score_in(live, target, corners_between(corners, source, target)) for target in used]
        complete = all("combined_score" in score for score in scores)
        mean = float(np.mean([score["combined_score"] for score in scores])) if complete else None
        role = None if label == COMPOSITE else source.frame.role
        rows.append({"candidate": label, "role": role, "scores": scores, "mean_combined_score": mean})
    for frame_row in skipped:
        rows.append({"candidate": candidate_label(frame_row), "role": frame_row["role"], "scores": [],
                     "mean_combined_score": None, "unscored_reason": frame_row["alignment"]["skip_reason"]})
    return rows


def decide(evaluated: list[dict[str, Any]], own_frame_label: str) -> dict[str, Any]:
    """The highest common-frame mean wins.

    An exact tie keeps a saved court, and ties between saved courts go middle, first, last.

    If any saved court lacks a mean, the own-frame full-score choice stands: choosing among the
    scored courts alone would compare a different set of saved courts.
    """
    originals = [row for row in evaluated if row["candidate"] != COMPOSITE]
    composite = next((row for row in evaluated if row["candidate"] == COMPOSITE), None)
    if any(row["mean_combined_score"] is None for row in originals):
        return {"state": "original_scores_incomplete", "common_frame_original": None,
                "composite_beats_originals": None, "final": own_frame_label}
    best = min(originals, key=lambda row: (-row["mean_combined_score"], FRAME_ROLES.index(row["role"])))
    beats = None
    if composite is not None and composite["mean_combined_score"] is not None:
        beats = composite["mean_combined_score"] > best["mean_combined_score"]
    return {"state": "compared", "common_frame_original": best["candidate"], "composite_beats_originals": beats,
            "final": COMPOSITE if beats else best["candidate"]}


def prepare_frames(live: LiveModules, video: dict[str, Any], outcome: dict[str, Any], order: list[int],
                   lines: dict[str, Any], people: dict[str, Any],
                   frames_dir: Path) -> tuple[list[dict[str, Any]], list[UsedFrame]]:
    """Load, check and align every accepted frame.

    :param order: Accepted frame positions in own-frame score order; the first is the reference.
    :return: One record per accepted frame, and the frames the composition uses, both in that order.
    """
    frame_rows, used = [], []
    reference: Frame | None = None
    for position in order:
        frame = load_frame(live, outcome["frames"][position], lines, people, frames_dir, video["native_size"])
        saved_corners = np.asarray(frame.saved["corners_native_px"], dtype=float)
        row: dict[str, Any] = {"position": position, "role": frame.role, "view_id": frame.saved["view_id"],
                               "frame_index": frame.saved["frame_index"], "is_reference": reference is None,
                               "reproduction": reproduction(frame.saved["evidence"],
                                                            measure(live, frame, saved_corners))}
        if reference is None:
            reference = frame
            row["alignment"], to_reference = None, np.eye(3)
        else:
            row["alignment"], to_reference = align(frame, reference)
        row["half_turn_roll"] = None
        if to_reference is not None:
            reference_working = np.asarray(reference.saved["corners_native_px"]) / reference.native_per_working
            item, row["half_turn_roll"] = use_frame(live, frame, to_reference, reference_working)
            used.append(item)
        frame_rows.append(row)
    return frame_rows, used


def write_outlines(directory: Path, scene_id: str, reference: UsedFrame, evaluated: list[dict[str, Any]]) -> list[str]:
    """Each compared court as a plain 1 px red dashed outline on the reference frame, one image per court."""
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for row in evaluated:
        if not row["scores"]:
            continue  # an unaligned frame's court cannot be carried into the reference
        image = reference.frame.native_frame.copy()
        draw_outline(image, row["scores"][0]["corners_native_px"])
        name = f"{scene_id}__{row['candidate']}__on_{reference.frame.role}.png"
        if not cv2.imwrite(str(directory / name), image):
            raise OSError(f"Could not write {name}")
        names.append(name)
    return names


def final_fields(choice: dict[str, Any], evaluated: list[dict[str, Any]], outcome: dict[str, Any],
                 reference_view_id: str) -> None:
    """Add the final court's frame and corners, and its corners in the reference frame.

    A saved court keeps its saved corner order in its own frame. The composite's own
    frame is the reference. A court whose frame did not align has no reference corners.
    """
    row = next(item for item in evaluated if item["candidate"] == choice["final"])
    in_reference = row["scores"][0]["corners_native_px"] if row["scores"] else None
    if choice["final"] == COMPOSITE:
        view_id, corners = reference_view_id, in_reference
    else:
        saved = next(frame for frame in outcome["frames"] if candidate_label(frame) == choice["final"])
        view_id, corners = saved["view_id"], saved["corners_native_px"]
    choice.update(final_view_id=view_id, final_corners_native_px=corners,
                  final_corners_reference_native_px=in_reference)


def compose_scene(live: LiveModules, video: dict[str, Any], scene: dict[str, Any], lines: dict[str, Any],
                  people: dict[str, Any], frames_dir: Path, outlines_dir: Path) -> dict[str, Any]:
    """One scene's composite, its comparison with the saved courts, and the final choice."""
    record: dict[str, Any] = {"video_id": video["video_id"], "scene_id": scene["scene_id"],
                              "scene_status": scene["status"]}
    if scene["status"] != "analysed":
        record["state"] = "no_evaluation"
        return record
    outcome = scene["methods"][METHOD]
    rescore.check_lines_size(video, scene, lines)
    rescored = rescore.rescore_method(outcome, lines)
    record["rescore"] = rescored
    if rescored["state"] != "rescored":
        # no_evaluation, no_court or score_evidence_missing: the saved outcome stands.
        record["state"] = rescored["state"]
        if "selected" in rescored:
            paint_label = candidate_label(outcome["frames"][rescored["selected"]["paint"]])
            record["choice"] = {"paint": paint_label, "own_frame": None, "final": paint_label, "changed": False}
        return record
    paint_label = candidate_label(outcome["frames"][rescored["selected"]["paint"]])
    record["choice"] = {"paint": paint_label,
                        "own_frame": candidate_label(outcome["frames"][rescored["selected"]["ranked"]])}
    record["frames"], used = prepare_frames(live, video, outcome, rescored["ranked_order"], lines, people,
                                            frames_dir)
    record["reference"] = record["frames"][0]["role"]
    if len(used) < 2:
        record["state"] = "too_few_aligned_frames"
        own_frame_label = record["choice"]["own_frame"]
        own_frame = outcome["frames"][rescored["selected"]["ranked"]]
        record["choice"].update(final=own_frame_label, changed=own_frame_label != paint_label,
                                 final_view_id=own_frame["view_id"],
                                 final_corners_native_px=own_frame["corners_native_px"],
                                 final_corners_reference_native_px=own_frame["corners_native_px"])
        return record
    record["state"] = "composed"
    record["markings"], record["fit"] = compose_court(live, used)
    skipped = [row for row in record["frames"] if row["alignment"] is not None and not row["alignment"]["usable"]]
    composite_corners = record["fit"]["corners_reference_native_px"] if record["fit"]["valid"] else None
    evaluated = evaluate(live, used, skipped, composite_corners)
    record["evaluation"] = {"frames": [item.frame.role for item in used], "candidates": evaluated}
    choice = decide(evaluated, record["choice"]["own_frame"])
    final_fields(choice, evaluated, outcome, used[0].frame.saved["view_id"])
    record["choice"].update(choice, changed=choice["final"] != paint_label)
    if composite_corners is not None:
        record["outlines"] = write_outlines(outlines_dir, scene["scene_id"], used[0], evaluated)
    return record


def compose(live: LiveModules, results: dict[str, Any], lines: dict[str, Any], people: dict[str, Any],
            frames_dir: Path, outlines_dir: Path) -> list[dict[str, Any]]:
    """One record per scene. Reads the results and caches without changing them."""
    rows = []
    for video in results["videos"]:
        for scene in video["scenes"] + video["later_scenes"]:
            started = perf_counter()
            row = compose_scene(live, video, scene, lines, people, frames_dir, outlines_dir)
            row["seconds"] = perf_counter() - started
            print(f"{scene['scene_id']}: {row['state']}, final {row.get('choice', {}).get('final')}", flush=True)
            rows.append(row)
    return rows


def joined_text(items: list[str]) -> str:
    return ";".join(items)


def comparison_row(scene: dict[str, Any]) -> dict[str, Any]:
    """One compact CSV row per scene."""
    row: dict[str, Any] = {"video_id": scene["video_id"], "scene_id": scene["scene_id"], "state": scene["state"]}
    choice = scene.get("choice")
    if choice is not None:
        row.update(paint_choice=choice["paint"], own_frame_choice=choice["own_frame"], final=choice["final"],
                   changed=choice["changed"])
    if "rescore" in scene and "candidates" in scene["rescore"]:
        row["accepted_count"] = len(scene["rescore"]["candidates"])
    if "frames" not in scene:
        return row
    frames = scene["frames"]
    row.update(
        reference_role=scene["reference"],
        used_roles=joined_text([item["role"] for item in frames if item["half_turn_roll"] is not None]),
        skipped_frames=joined_text([f"{item['role']}:{item['alignment']['skip_reason']}" for item in frames
                                    if item["half_turn_roll"] is None]),
        half_turned_roles=joined_text([item["role"] for item in frames if item["half_turn_roll"] == HALF_TURN_ROLL]),
        reproduction_max_score_difference=max(item["reproduction"]["max_score_difference"] for item in frames),
        reproduction_mismatches=joined_text([f"{item['role']}:{name}" for item in frames
                                             for name in item["reproduction"]["mismatches"]]),
    )
    if scene["state"] != "composed":
        return row
    fit = scene["fit"]
    scores, means = [], []
    for candidate in scene["evaluation"]["candidates"]:
        means.append(f"{candidate['candidate']}={candidate['mean_combined_score']}")
        for score in candidate["scores"]:
            scores.append(f"{candidate['candidate']}@{score['frame_role']}={score.get('combined_score')}")
    row.update(
        donors=joined_text([f"{item['marking']}:{item['donor_role']}:{item['q_paint10']}:{item['samples']}"
                            for item in scene["markings"]]),
        fit_status=fit["status"], fit_valid=fit["valid"], fit_reason=fit["validity_reason"],
        fit_sample_count=fit["sample_count"], per_frame_scores=joined_text(scores),
        common_frame_means=joined_text(means),
        common_frame_original=choice["common_frame_original"],
        composite_beats_originals=choice["composite_beats_originals"],
        final_corners_reference_native_px=json.dumps(choice["final_corners_reference_native_px"]),
    )
    return row


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    with gzip.open(output_dir / "results.json.gz", "wt") as stream:
        json.dump(report, stream, allow_nan=False, separators=(",", ":"))
    with gzip.open(output_dir / "comparison.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_row(scene) for scene in report["scenes"])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True, help="the comparison's results.json.gz")
    parser.add_argument("--lines-cache", type=Path, required=True, help="cached DeepLSD lines by view_id")
    parser.add_argument("--people-cache", type=Path, required=True, help="cached same-frame person boxes by view_id")
    parser.add_argument("--frames-dir", type=Path, required=True, help="cached source PNGs named <view_id>.png")
    parser.add_argument("--output-dir", type=Path, required=True, help="a new directory for the outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists")
    results = read_json(args.results)
    if results["schema"] != rescore.SOURCE_SCHEMA:
        raise ValueError(f"{args.results}: expected schema {rescore.SOURCE_SCHEMA!r}, found {results['schema']!r}")
    lines = rescore.read_lines_cache(args.lines_cache)
    people = read_people_cache(args.people_cache)
    live = load_live_modules()
    args.output_dir.mkdir(parents=True)
    started = perf_counter()
    scenes = compose(live, results, lines["frames"], people["frames"], args.frames_dir, args.output_dir / "outlines")
    report = {
        "schema": COMPOSE_SCHEMA, "source_results": str(args.results), "source_finished": results["finished"],
        "lines_cache": str(args.lines_cache), "people_cache": str(args.people_cache),
        "frames_dir": str(args.frames_dir),
        # Kept apart from the source run's timing: none of these is part of an arm's measured time.
        "line_recovery_seconds": lines["recovery_seconds"], "people_recovery_seconds": people["recovery_seconds"],
        "compose_seconds": perf_counter() - started,
        "rule": {"method": METHOD, "geometry_weight": rescore.GEOMETRY_WEIGHT, "net_weight": rescore.NET_WEIGHT,
                 "net_overrun_working_px": rescore.NET_OVERRUN_WORKING_PX, "frame_tie_order": list(FRAME_ROLES),
                 "min_alignment_correlation": court_views.MIN_ALIGNMENT_CORRELATION,
                 "alignment_mask": "frame's saved court x1.08 without either frame's person boxes",
                 "max_horizon_tilt_deg": MAX_HORIZON_TILT_DEG,
                 "donor_weight": "stripe_fitting.prepare weight x donor q_paint10"},
        "scenes": scenes,
    }
    write_outputs(args.output_dir, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
