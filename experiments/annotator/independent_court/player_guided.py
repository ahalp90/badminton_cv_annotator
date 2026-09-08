"""Fit complete court hypotheses against observed player positions over time.

Seed rectangles may describe service boxes. Player evidence is applied after
each seed has been assigned to the physical court template, before retention.
The original line-only detector remains the comparison implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import detector

COURT_SIZE_M = detector.CORNER_COURT_M.max(axis=0)


@dataclass(frozen=True)
class GuidedDetection:
    detection: detector.Detection
    hypotheses_generated: int
    hypotheses_with_players: int
    player_fractions: tuple[tuple[float, float], ...]


def player_fractions(
    homographies: np.ndarray, feet_px: np.ndarray, margin: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure observed presence within each complete projected court.

    :param homographies: Court metres to image pixels, one matrix per hypothesis.
    :param feet_px: (sampled frames, two player slots, image xy); NaN means missing.
    :return: Fractions with at least one player, and with one player in each half.
    """
    inverse = np.linalg.inv(homographies)
    homogeneous = np.concatenate((feet_px, np.ones((*feet_px.shape[:-1], 1))), axis=-1)
    mapped = np.einsum("hij,fpj->hfpi", inverse, homogeneous)
    with np.errstate(divide="ignore", invalid="ignore"):
        court = mapped[..., :2] / mapped[..., 2:] / COURT_SIZE_M
    inside = np.isfinite(court).all(axis=-1) & (court >= -margin).all(axis=-1) & (court <= 1 + margin).all(axis=-1)
    far = inside & (court[..., 1] < 0.5)
    near = inside & (court[..., 1] >= 0.5)
    return inside.any(axis=-1).mean(axis=-1), (far.any(axis=-1) & near.any(axis=-1)).mean(axis=-1)


def detect(
    frame: np.ndarray,
    feet_px: np.ndarray,
    settings: detector.Settings = detector.DEFAULT_SETTINGS,
    *,
    segments_px: np.ndarray | None = None,
    region_filter: bool = False,
    player_margin: float = 0.15,
) -> GuidedDetection:
    """Fit courts supported by lines and sustained observed player presence.

    :param feet_px: Native (frames, 2, 2) player observations; NaN marks gaps.
    :param region_filter: Restrict line evidence to a generous player region.
    :return: Retained courts, acceptance decision and player-support diagnostics.
    """
    feet = np.asarray(feet_px, dtype=np.float64)
    if feet.ndim != 3 or feet.shape[1:] != (2, 2) or not len(feet) or np.isinf(feet).any():
        raise ValueError("feet_px must be a nonempty (frames, 2, 2) array with finite points or NaN gaps")
    if not np.array_equal(np.isnan(feet[..., 0]), np.isnan(feet[..., 1])):
        raise ValueError("missing feet must have NaN in both coordinates")
    if not np.isfinite(player_margin) or player_margin < 0:
        raise ValueError("player_margin must be finite and non-negative")
    height, width = frame.shape[:2]
    scale = min(1.0, settings.max_dimension / max(width, height))
    working = cv2.resize(frame, (round(width * scale), round(height * scale))) if scale < 1 else frame
    size = (working.shape[1], working.shape[0])
    native_scale = np.array([width / size[0], height / size[1]])
    working_feet = feet / native_scale
    if segments_px is None:
        segments = detector.extract_segments(working, settings.extractor)
    else:
        native_segments = np.asarray(segments_px, dtype=np.float64)
        if native_segments.ndim != 2 or native_segments.shape[1] != 4 or not np.isfinite(native_segments).all():
            raise ValueError("segments_px must contain finite native XYXY rows")
        if np.any(np.linalg.norm(native_segments[:, 2:] - native_segments[:, :2], axis=1) == 0):
            raise ValueError("segments_px must have positive length")
        segments = native_segments / np.tile(native_scale, 2)
    if region_filter:
        observed = working_feet[np.isfinite(working_feet).all(axis=-1)]
        if len(observed):
            lower, upper = observed.min(axis=0), observed.max(axis=0)
            expansion = np.maximum((upper - lower) / 2, 0.30 * np.asarray(size))
            starts, ends = segments[:, :2], segments[:, 2:]
            # Bounding-box intersection preserves long fragments crossing the region.
            above_lower = (np.maximum(starts, ends) >= lower - expansion).all(axis=1)
            below_upper = (np.minimum(starts, ends) <= upper + expansion).all(axis=1)
            segments = segments[above_lower & below_upper]
    families = detector._wide_line_families(segments) if settings.wide_families else detector._line_families(segments)
    x_lines, y_lines = (detector._merge_lines(family, settings) for family in families)
    counts = (len(x_lines), len(y_lines))
    output_segments = segments * np.tile(native_scale, 2)
    if min(counts) < 2:
        result = detector.Detection((), False, "insufficient_lines", None, output_segments, counts, 0)
        return GuidedDetection(result, 0, 0, ())

    rectangles = detector._image_rectangles(x_lines, y_lines, size, settings)
    maps = detector._distance_maps(families, size)
    candidates: list[detector.Candidate] = []
    generated = 0
    with_players = 0
    scored = 0
    for offset in range(0, len(rectangles), 8):
        homographies = (rectangles[offset:offset + 8, None] @ detector.TEMPLATE_TRANSFORMS).reshape(-1, 3, 3)
        generated += len(homographies)
        one_fraction, two_fraction = player_fractions(homographies, working_feet, player_margin)
        supported_players = (one_fraction == 1.0) & (two_fraction >= 0.5)
        with_players += int(supported_players.sum())
        if not supported_players.any():
            continue
        corners, scores, means, supported_counts = detector._score(
            homographies[supported_players], maps, settings, (x_lines, y_lines),
        )
        scored += len(scores)
        eligible = np.flatnonzero(scores >= 0)
        proposed = []
        for index in eligible[np.argsort(-scores[eligible], kind="stable")]:
            proposed.append(detector.Candidate(corners[index], float(scores[index]), tuple(means[index]),
                                              tuple(int(value) for value in supported_counts[index])))
        candidates = detector._retain(candidates, proposed, settings)
    gap = None if len(candidates) < 2 else candidates[0].score - candidates[1].score
    ambiguous = bool(candidates) and detector._separate_court(candidates, size)
    accepted = bool(candidates) and not ambiguous and (gap is None or gap >= settings.ambiguity_gap)
    reason = "accepted" if accepted else "ambiguous" if candidates else "unsupported"
    native_candidates = tuple(detector.Candidate(candidate.corners_px * native_scale, candidate.score,
                                                candidate.family_support, candidate.supported_lines)
                              for candidate in candidates)
    fractions = []
    for candidate in native_candidates:
        transform = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, candidate.corners_px.astype(np.float32))
        one, two = player_fractions(transform[None], feet, player_margin)
        fractions.append((float(one[0]), float(two[0])))
    result = detector.Detection(native_candidates, accepted, reason, gap, output_segments, counts, scored)
    return GuidedDetection(result, generated, with_players, tuple(fractions))
