"""The original line-only court search, kept as the research comparison.

It extracts line fragments (or takes precomputed ones), builds rectangles from
line crossings, scores every placement of the court template on them and keeps
distinct courts. The search assumes an upright view from behind a baseline.
Scores measure image support; they are not calibrated probabilities.

The court detector does not use this search. It shares the court coordinates,
line grouping and scoring helpers in court_detector.geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from court_detector.geometry import (
    DEFAULT_SETTINGS,
    TEMPLATE_TRANSFORMS,
    UNIT_CORNERS,
    Candidate,
    Settings,
    _distance_maps,
    _filter_painted_stripes,
    _line_families,
    _merge_lines,
    _score,
    _wide_line_families,
)

# Canny and probabilistic Hough settings from the retired CourtKeyNet corner module.
CANNY_LO, CANNY_HI = 50, 150
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_PX = 40
HOUGH_MAX_GAP_PX = 15


@dataclass(frozen=True)
class Detection:
    candidates: tuple[Candidate, ...]
    accepted: bool
    reason: str
    score_gap: float | None
    segments_px: np.ndarray
    family_line_counts: tuple[int, int]
    hypotheses_scored: int


def _hough_segments(frame: np.ndarray) -> np.ndarray:
    """Canny and probabilistic Hough segments over the whole frame.

    :param frame: (H, W, 3) uint8 BGR frame
    :return: (m, 4) int segments (x1, y1, x2, y2), possibly empty
    """
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CANNY_LO, CANNY_HI)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_PX, maxLineGap=HOUGH_MAX_GAP_PX,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines.reshape(-1, 4)


def extract_segments(frame: np.ndarray, method: str) -> np.ndarray:
    """Extract full-frame fragments; no court mask or manual region is used."""
    if method in ("hough", "ridge"):
        segments = _hough_segments(frame).astype(np.float64)
        return _filter_painted_stripes(frame, segments) if method == "ridge" else segments
    if method == "lsd":
        lines = cv2.createLineSegmentDetector().detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
        if lines is None:
            return np.empty((0, 4), dtype=np.float64)
        segments = lines.reshape(-1, 4).astype(np.float64)
        length = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
        return segments[length >= 30]
    raise ValueError(f"unknown line extractor: {method}")


def _image_rectangles(
    x_lines: np.ndarray, y_lines: np.ndarray, size: tuple[int, int], settings: Settings,
) -> np.ndarray:
    width, height = size
    # Ordering at the image centre defines far/near and left/right for this view.
    x_order = np.argsort(-(x_lines[:, 1] * height / 2 + x_lines[:, 2]) / x_lines[:, 0])
    y_order = np.argsort(-(y_lines[:, 0] * width / 2 + y_lines[:, 2]) / y_lines[:, 1])
    x_lines, y_lines = x_lines[x_order], y_lines[y_order]
    intersections = np.cross(x_lines[:, None], y_lines[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        points = intersections[..., :2] / intersections[..., 2:]
    x_pairs = np.asarray(list(combinations(range(len(x_lines)), 2)))
    y_pairs = np.asarray(list(combinations(range(len(y_lines)), 2)))
    x_slots = x_pairs[:, [0, 1, 1, 0]][:, None, :]
    y_slots = y_pairs[:, [0, 0, 1, 1]][None, :, :]
    quads = points[x_slots, y_slots].reshape(-1, 4, 2)
    edges = np.roll(quads, -1, axis=1) - quads
    turns = edges[..., 0] * np.roll(edges[..., 1], -1, axis=1) - edges[..., 1] * np.roll(edges[..., 0], -1, axis=1)
    valid = np.isfinite(quads).all(axis=(1, 2)) & np.all(turns > 0, axis=1)
    quads = quads[valid]
    if len(quads) > settings.max_rectangles:
        generator = np.random.default_rng(settings.seed)
        quads = quads[generator.choice(len(quads), settings.max_rectangles, replace=False)]
    rectangles = []
    for quad in quads:
        if cv2.contourArea(quad.astype(np.float32)) >= 100:
            rectangles.append(cv2.getPerspectiveTransform(UNIT_CORNERS, quad.astype(np.float32)))
    return np.asarray(rectangles).reshape(-1, 3, 3)


def _retain(candidates: list[Candidate], proposed: list[Candidate], settings: Settings) -> list[Candidate]:
    retained: list[Candidate] = []
    for candidate in sorted(candidates + proposed, key=lambda item: -item.score):
        duplicate = False
        for previous in retained:
            separation = np.linalg.norm(candidate.corners_px - previous.corners_px, axis=1).max()
            if separation <= settings.distinct_corner_distance:
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
            if len(retained) == settings.keep_candidates:
                break
    return retained


def _separate_court(candidates: list[Candidate], size: tuple[int, int]) -> bool:
    """Two supported courts in different image regions leave the target unresolved."""
    width, height = size
    viewport = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    visible_regions: list[tuple[float, np.ndarray]] = []
    for candidate_index, candidate in enumerate(candidates):
        area, polygon = cv2.intersectConvexConvex(candidate.corners_px.astype(np.float32), viewport)
        if polygon is None or area <= 0:
            if candidate_index == 0:
                return True
            continue
        for previous_area, previous_polygon in visible_regions:
            overlap, _ = cv2.intersectConvexConvex(previous_polygon, polygon)
            if overlap < 0.5 * min(previous_area, area):
                return True
        visible_regions.append((area, polygon))
    return False


def detect(
    frame: np.ndarray, settings: Settings = DEFAULT_SETTINGS, *, segments_px: np.ndarray | None = None,
) -> Detection:
    """Return supported court hypotheses and an explicit ambiguity decision.

    :param frame: uint8 BGR image in source pixels; no reference geometry is accepted.
    :param segments_px: optional precomputed fragments, one native XYXY row per line.
    :return: candidates and line fragments in the original frame coordinates.
    """
    height, width = frame.shape[:2]
    scale = min(1.0, settings.max_dimension / max(width, height))
    working = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1 else frame
    native_scale = np.array([width / working.shape[1], height / working.shape[0]])
    size = (working.shape[1], working.shape[0])
    if segments_px is None:
        segments = extract_segments(working, settings.extractor)
    else:
        segments_px = np.asarray(segments_px, dtype=np.float64)
        if segments_px.ndim != 2 or segments_px.shape[1] != 4 or not np.isfinite(segments_px).all():
            raise ValueError("segments_px must contain finite native XYXY rows")
        if np.any(np.linalg.norm(segments_px[:, 2:] - segments_px[:, :2], axis=1) == 0):
            raise ValueError("segments_px must have positive length")
        segments = segments_px / np.tile(native_scale, 2)
    families = _wide_line_families(segments) if settings.wide_families else _line_families(segments)
    x_lines, y_lines = (_merge_lines(family, settings) for family in families)
    counts = (len(x_lines), len(y_lines))
    native_segments = segments * np.tile(native_scale, 2)
    if min(counts) < 2:
        return Detection((), False, "insufficient_lines", None, native_segments, counts, 0)
    rectangles = _image_rectangles(x_lines, y_lines, size, settings)
    maps = _distance_maps(families, size)
    candidates: list[Candidate] = []
    scored = 0
    for offset in range(0, len(rectangles), 8):
        homographies = (rectangles[offset:offset + 8, None] @ TEMPLATE_TRANSFORMS).reshape(-1, 3, 3)
        corners, scores, means, supported_counts = _score(homographies, maps, settings, (x_lines, y_lines))
        scored += len(scores)
        eligible = np.flatnonzero(scores >= 0)
        # Diversify before truncating: many hypotheses can describe the same court.
        proposed = []
        for index in eligible[np.argsort(-scores[eligible], kind="stable")]:
            proposed.append(Candidate(corners[index], float(scores[index]), tuple(means[index]),
                                      tuple(int(value) for value in supported_counts[index])))
        candidates = _retain(candidates, proposed, settings)
    gap = None if len(candidates) < 2 else candidates[0].score - candidates[1].score
    ambiguous_location = bool(candidates) and _separate_court(candidates, size)
    accepted = bool(candidates) and not ambiguous_location and (gap is None or gap >= settings.ambiguity_gap)
    reason = "accepted" if accepted else "ambiguous" if candidates else "unsupported"
    native_candidates = tuple(Candidate(candidate.corners_px * native_scale, candidate.score,
                                        candidate.family_support, candidate.supported_lines) for candidate in candidates)
    return Detection(native_candidates, accepted, reason, gap, native_segments, counts, scored)
