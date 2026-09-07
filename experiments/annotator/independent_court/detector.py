"""Experimental court proposals from image lines, without neural initialisation.

The search assumes an upright view from behind a baseline. It enumerates line
identities against the badminton template and keeps competing court placements.
Scores describe image support, not calibrated probabilities. Production does
not import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from courtkeynet.court_corners import (
    CORNER_COURT_M,
    PAINTED_SEGMENTS_M,
    _frame_segments,
)

SEGMENTS_M = np.asarray(PAINTED_SEGMENTS_M, dtype=np.float64)
X_COORDS = np.unique(SEGMENTS_M[:6, 0, 0])
Y_COORDS = np.unique(SEGMENTS_M[6:, 0, 1])
UNIT_CORNERS = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
RIDGE_SAMPLES = 24
RIDGE_CENTRE_SHIFTS = np.array([-4, -2, 0, 2, 4], dtype=np.float32)
RIDGE_SIDE_DISTANCE = 6.0
RIDGE_MIN_CONTRAST = 10.0
RIDGE_MIN_FRACTION = 0.4


@dataclass(frozen=True)
class Settings:
    """Initial engineering bounds; none represents learned confidence."""

    max_dimension: int = 960
    max_family_lines: int = 32
    max_rectangles: int = 4096
    seed: int = 20260907
    # Canny can place opposite edges of one painted stripe six pixels apart.
    merge_distance: float = 8.0
    merge_angle_deg: float = 3.0
    support_distance: float = 4.0
    samples_per_line: int = 24
    min_visible_span_fraction: float = 0.15
    min_family_support: float = 0.55
    min_supported_lines: int = 4
    ambiguity_gap: float = 0.035
    distinct_corner_distance: float = 12.0
    keep_candidates: int = 32
    extractor: str = "hough"
    wide_families: bool = False

    def __post_init__(self) -> None:
        if self.keep_candidates < 2:
            raise ValueError("keep_candidates must retain at least two courts for ambiguity checking")


@dataclass(frozen=True)
class Candidate:
    corners_px: np.ndarray
    score: float
    family_support: tuple[float, float]
    supported_lines: tuple[int, int]


DEFAULT_SETTINGS = Settings()


@dataclass(frozen=True)
class Detection:
    candidates: tuple[Candidate, ...]
    accepted: bool
    reason: str
    score_gap: float | None
    segments_px: np.ndarray
    family_line_counts: tuple[int, int]
    hypotheses_scored: int


def _template_transforms() -> np.ndarray:
    transforms = []
    for left, right in combinations(X_COORDS, 2):
        for top, bottom in combinations(Y_COORDS, 2):
            transforms.append([[1 / (right - left), 0, -left / (right - left)],
                               [0, 1 / (bottom - top), -top / (bottom - top)], [0, 0, 1]])
    return np.asarray(transforms)


TEMPLATE_TRANSFORMS = _template_transforms()


def project(homographies: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:return: projected points and homogeneous denominators for each hypothesis."""
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
    mapped = np.einsum("...ij,pj->...pi", homographies, homogeneous.reshape(-1, 3))
    denominator = mapped[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = mapped[..., :2] / denominator[..., None]
    return pixels, denominator


def extract_segments(frame: np.ndarray, method: str) -> np.ndarray:
    """Extract full-frame fragments; no court mask or manual region is used."""
    if method in ("hough", "ridge"):
        segments = _frame_segments(frame, np.full(frame.shape[:2], 255, dtype=np.uint8)).astype(np.float64)
        return _filter_painted_stripes(frame, segments) if method == "ridge" else segments
    if method == "lsd":
        lines = cv2.createLineSegmentDetector().detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
        if lines is None:
            return np.empty((0, 4), dtype=np.float64)
        segments = lines.reshape(-1, 4).astype(np.float64)
        length = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
        return segments[length >= 30]
    raise ValueError(f"unknown line extractor: {method}")


def _filter_painted_stripes(frame: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Keep bright stripes with darker pixels on both sides, regardless of colour.

    Canny marks stripe edges, so sample several nearby centres along the normal.
    This is an optional evidence filter; weak or crowded markings can be lost.
    Input fragments from the extractors have finite endpoints and positive length.
    """
    if not len(segments):
        return segments
    endpoints = segments.reshape(-1, 2, 2).astype(np.float32)
    vectors = endpoints[:, 1] - endpoints[:, 0]
    normals = np.stack((-vectors[:, 1], vectors[:, 0]), axis=1)
    normals /= np.linalg.norm(vectors, axis=1)[:, None]
    fractions = np.linspace(0, 1, RIDGE_SAMPLES, dtype=np.float32)
    centres = endpoints[:, None, 0] + vectors[:, None] * fractions[None, :, None]
    shifted = centres[:, :, None] + normals[:, None, None] * RIDGE_CENTRE_SHIFTS[None, None, :, None]
    sides = RIDGE_SIDE_DISTANCE * normals[:, None, None]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = frame.shape[:2]
    intensities = []
    in_frame = []
    # Each sample tests five possible centres and the pixels on either side.
    for points in (shifted, shifted - sides, shifted + sides):
        maps = points.reshape(len(segments), -1, 2)
        sampled = cv2.remap(grey, maps[..., 0], maps[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        intensities.append(sampled.reshape(points.shape[:-1]))
        in_frame.append(
            (points[..., 0] >= 0) & (points[..., 0] < width)
            & (points[..., 1] >= 0) & (points[..., 1] < height)
        )
    centre, first_side, second_side = intensities
    contrast = np.minimum(centre - first_side, centre - second_side)
    visible = in_frame[0] & in_frame[1] & in_frame[2]
    contrast = np.where(visible, contrast, -np.inf)
    ridge_samples = contrast.max(axis=2) >= RIDGE_MIN_CONTRAST
    return segments[ridge_samples.mean(axis=1) >= RIDGE_MIN_FRACTION]


def _line_families(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = segments[:, 2:] - segments[:, :2]
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = (angles + np.pi / 2) % np.pi - np.pi / 2
    # The dominant nearly horizontal direction proposes cross-court lines.
    bins = np.deg2rad(np.arange(-32, 34, 2))
    histogram, _ = np.histogram(angles, bins=bins, weights=lengths)
    peak = int(np.argmax(histogram))
    baseline_angle = (bins[peak] + bins[peak + 1]) / 2
    differences = np.abs(angles - baseline_angle)
    differences = np.minimum(differences, np.pi - differences)
    return segments[differences >= np.deg2rad(22)], segments[differences <= np.deg2rad(12)]


def _wide_line_families(segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exploratory overlapping groups preserve lines that fan out under perspective."""
    vectors = segments[:, 2:] - segments[:, :2]
    angles = (np.arctan2(vectors[:, 1], vectors[:, 0]) + np.pi / 2) % np.pi - np.pi / 2
    return segments[np.abs(angles) >= np.deg2rad(10)], segments[np.abs(angles) <= np.deg2rad(35)]


def _covered_length(points: np.ndarray, line: np.ndarray) -> float:
    intervals = np.sort((points @ np.array([line[1], -line[0]])).reshape(-1, 2), axis=1)
    intervals = intervals[np.argsort(intervals[:, 0])]
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            total += end - start
            start = next_start
        end = max(end, next_end)
    return float(total + end - start)


def _merge_lines(segments: np.ndarray, settings: Settings) -> np.ndarray:
    """Retain lines with the most covered length; gaps and duplicate edges add none."""
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    groups: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    for index in np.argsort(-lengths, kind="stable")[:300]:
        points = segments[index].reshape(2, 2)
        direction = (points[1] - points[0]) / lengths[index]
        target = None
        for group_index, line in enumerate(coefficients):
            parallel = abs(np.dot(direction, line[:2])) < np.sin(np.deg2rad(settings.merge_angle_deg))
            close = np.max(np.abs(points @ line[:2] + line[2])) <= settings.merge_distance
            if parallel and close:
                target = group_index
                break
        if target is None:
            groups.append(points)
            coefficients.append(np.zeros(3))
            target = len(groups) - 1
        else:
            groups[target] = np.concatenate((groups[target], points))
        dx, dy, centre_x, centre_y = cv2.fitLine(groups[target].astype(np.float32), cv2.DIST_L2, 0, .01, .01).ravel()
        coefficients[target] = np.array([-dy, dx, dy * centre_x - dx * centre_y], dtype=np.float64)
    extents = []
    for points, line in zip(groups, coefficients):
        extents.append(_covered_length(points, line))
    order = np.argsort(-np.asarray(extents), kind="stable")[:settings.max_family_lines]
    return np.asarray(coefficients, dtype=np.float64).reshape(-1, 3)[order]


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


def _distance_maps(families: tuple[np.ndarray, np.ndarray], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    maps = []
    for segments in families:
        mask = np.full((height, width), 255, dtype=np.uint8)
        for x1, y1, x2, y2 in np.rint(segments).astype(int):
            cv2.line(mask, (x1, y1), (x2, y2), 0, 1)
        maps.append(cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
    return np.stack(maps)


def _visible_samples(endpoints: np.ndarray, size: tuple[int, int], count: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip finite projected markings before sampling them uniformly in image space."""
    starts = endpoints[:, :, 0]
    vectors = endpoints[:, :, 1] - starts
    lower = np.zeros(starts.shape[:2])
    upper = np.ones(starts.shape[:2])
    visible = np.ones(starts.shape[:2], dtype=bool)
    for axis, limit in enumerate(size):
        stationary = np.abs(vectors[..., axis]) < 1e-8
        visible &= ~stationary | ((starts[..., axis] >= 0) & (starts[..., axis] <= limit - 1))
        divisor = np.where(stationary, 1, vectors[..., axis])
        first = -starts[..., axis] / divisor
        last = (limit - 1 - starts[..., axis]) / divisor
        lower = np.maximum(lower, np.where(stationary, -np.inf, np.minimum(first, last)))
        upper = np.minimum(upper, np.where(stationary, np.inf, np.maximum(first, last)))
    visible &= upper > lower
    clipped_length = (upper - lower) * np.linalg.norm(vectors, axis=-1)
    visible &= clipped_length >= 12
    fractions = lower[..., None] + (upper - lower)[..., None] * np.linspace(0, 1, count)
    samples = starts[..., None, :] + fractions[..., None] * vectors[..., None, :]
    return samples, visible


def _matched_line_counts(
    samples: np.ndarray,
    supported: np.ndarray,
    lines: tuple[np.ndarray, np.ndarray],
    tolerance: float,
) -> np.ndarray:
    """Count distinct observed lines, so nearby template lines cannot reuse one edge."""
    counts = []
    for family_index, observed in enumerate(lines):
        family_slice = slice(6 * family_index, 6 * (family_index + 1))
        endpoints = samples[:, family_slice][:, :, [0, -1]]
        distances = np.abs(np.einsum("nsed,ld->nsel", endpoints, observed[:, :2]) + observed[:, 2])
        residuals = distances.max(axis=2)
        matches = np.argmin(residuals, axis=2)
        best = np.take_along_axis(residuals, matches[..., None], axis=2)[..., 0]
        usable = supported[:, family_slice] & (best <= tolerance)
        used = matches[..., None] == np.arange(len(observed))
        counts.append(np.any(used & usable[..., None], axis=1).sum(axis=1))
    return np.stack(counts, axis=1)


def _score(
    homographies: np.ndarray,
    maps: np.ndarray,
    settings: Settings,
    lines: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, ...]:
    size = (maps.shape[2], maps.shape[1])
    corners, denominator = project(homographies, CORNER_COURT_M)
    valid = np.isfinite(corners).all(axis=(1, 2)) & np.all(denominator > 1e-6, axis=1)
    edges = np.roll(corners, -1, axis=1) - corners
    turns = edges[..., 0] * np.roll(edges[..., 1], -1, axis=1) - edges[..., 1] * np.roll(edges[..., 0], -1, axis=1)
    valid &= np.all(turns > 0, axis=1)
    visible_lower = np.maximum(corners.min(axis=1), 0)
    visible_upper = np.minimum(corners.max(axis=1), np.asarray(size) - 1)
    visible_span = (visible_upper - visible_lower) / np.asarray(size)
    valid &= np.all(visible_span >= settings.min_visible_span_fraction, axis=1)
    corners = corners[valid]
    homographies = homographies[valid]
    if not len(corners):
        return corners, np.empty(0), np.empty((0, 2)), np.empty((0, 2), dtype=int)
    endpoints, _ = project(homographies, SEGMENTS_M)
    samples, visible = _visible_samples(endpoints.reshape(-1, 12, 2, 2), size, settings.samples_per_line)
    samples = np.nan_to_num(samples, nan=0, posinf=0, neginf=0)
    pixel_x = np.clip(samples[..., 0], 0, size[0] - 1).astype(int)
    pixel_y = np.clip(samples[..., 1], 0, size[1] - 1).astype(int)
    families = np.repeat([0, 1], 6)[None, :, None]
    support = (maps[families, pixel_y, pixel_x] <= settings.support_distance).mean(axis=-1)
    support *= visible
    means = np.stack([support[:, :6].sum(axis=1) / np.maximum(visible[:, :6].sum(axis=1), 1),
                      support[:, 6:].sum(axis=1) / np.maximum(visible[:, 6:].sum(axis=1), 1)], axis=1)
    supported = support >= settings.min_family_support
    counts = _matched_line_counts(samples, supported, lines, settings.support_distance * 2)
    score = means.mean(axis=1)
    enough_lines = np.all(counts >= settings.min_supported_lines, axis=1)
    enough_support = np.all(means >= settings.min_family_support, axis=1)
    score[~(enough_lines & enough_support)] = -1
    return corners, score, means, counts


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
