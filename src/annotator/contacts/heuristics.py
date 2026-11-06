"""Detect, gate and suppress heuristic contact candidates within supplied spans.

`rallies.find_rallies.segment_video` applies excluded-frame masking and finds
the spans before calling these contact rules.
"""

import numpy as np

from annotator.array_calc_helpers import rolling_mean, rolling_nanmean
from annotator.config import Thresholds
from annotator.types import ContactCandidate, SmoothingMode

FLOOR_EPS = 1e-4


def assemble_contacts(
    track: np.ndarray, spans: list[tuple[int, int]],
    thresholds: Thresholds, sticky_distances: np.ndarray | None,
    *, smoothing_mode: SmoothingMode,
) -> list[ContactCandidate]:
    """Detect, gate, and suppress contacts for already-selected spans."""
    contacts = []
    gated = []
    for rally_id, (start, end) in enumerate(spans):
        candidates = impulse_cell_candidates(track, start, end, thresholds, smoothing_mode=smoothing_mode)
        for frame, impulse in candidates:
            near_wrist = wrist_contact_near(sticky_distances, frame, thresholds.wrist_distance_body_heights)
            contacts.append(ContactCandidate(rally_id, frame, near_wrist, None))
            if near_wrist:
                gated.append((frame, impulse))
    if sticky_distances is None:
        return contacts

    # The second suppression stage compares gated contacts across rally boundaries.
    radius = thresholds.contact_suppression_radius_frames
    accepted = {frame for frame, _ in suppress_contact_flags(gated, radius=radius)}
    return [
        contact._replace(suppressed=contact.wrist_near and contact.contact_frame not in accepted)
        for contact in contacts
    ]


def impulse_cell_candidates(
    track: np.ndarray, start: int, end: int, thresholds: Thresholds, *,
    smoothing_mode: SmoothingMode,
) -> list[tuple[int, float]]:
    """Find raw impulse candidates and retain their impulse for suppression.

    The measured rule is pure impulse. It uses the three-frame visibility mask,
    a rolling impulse floor, and largest-impulse-first de-duplication at the
    three-frame boundary used at 25 fps.
    """
    measured = span_impulses(track, start, end, thresholds, smoothing_mode=smoothing_mode)
    if measured is None:
        return []
    impulses, ratios, around_visible = measured
    impulse_pass = ratios > thresholds.contact_impulse_multiple
    is_contact = impulse_pass & around_visible

    candidates = [(start + int(junction) + 1, float(impulses[junction])) for junction in np.flatnonzero(is_contact)]
    return suppress_contact_flags(candidates, radius=thresholds.contact_dedup_radius_frames)


def span_impulses(
    track: np.ndarray, start: int, end: int, thresholds: Thresholds, *,
    smoothing_mode: SmoothingMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Measure impulse magnitude, ratio to the local floor and junction visibility.

    Annotation prepares a replay-masked track before detecting contacts.
    Junction ``k`` sits at local frame ``k + 1`` and touches three straddling frames.
    Track xy inputs are finite. IGNORE_INVISIBLE makes a float copy whose invisible
    xy values are NaN. A NaN impulse means its smoothing window was unmeasurable;
    downstream comparisons drop the corresponding junction.

    :return: Magnitudes, ratios and three-frame visibility, each indexed by junction.
        Return None when the span is too short for smoothing and two differences.
    """
    smooth_window = thresholds.smooth_window
    span = track[start:end]
    if len(span) < smooth_window + 2:
        return None

    if smoothing_mode is SmoothingMode.ZERO_FILL:
        smooth_x = rolling_mean(span[:, 0], smooth_window)
        smooth_y = rolling_mean(span[:, 1], smooth_window)
    else:
        smooth_xy = span[:, :2].astype(float, copy=True)
        smooth_xy[span[:, 2] != 1] = np.nan
        smooth_x = rolling_nanmean(smooth_xy[:, 0], smooth_window)
        smooth_y = rolling_nanmean(smooth_xy[:, 1], smooth_window)
    velocity = np.diff(np.column_stack([smooth_x, smooth_y]), axis=0)
    impulses = np.linalg.norm(velocity[1:] - velocity[:-1], axis=1)
    visible = span[:, 2] == 1
    around_visible = visible[:-2] & visible[1:-1] & visible[2:]
    floors = rolling_floor(impulses, around_visible, thresholds.impulse_floor_half_window_frames)
    ratios = impulses / np.maximum(floors, FLOOR_EPS)
    return impulses, ratios, around_visible


def rolling_floor(
    values: np.ndarray, around_visible: np.ndarray,
    half_window: int,
) -> np.ndarray:
    """Median floor over visible junctions within a frame-count window.

    ``half_window`` counts junctions on each side at the video's frame rate. A junction with no
    visible neighbour in its window receives NaN and cannot pass the rule.
    """
    visible_values = np.where(around_visible, values, np.nan)
    floor = np.full(len(values), np.nan)
    for junction_index in range(len(values)):
        window_start = max(junction_index - half_window, 0)
        window_end = min(junction_index + half_window + 1, len(values))
        window = visible_values[window_start:window_end]
        if np.isfinite(window).any():
            floor[junction_index] = np.nanmedian(window)
    return floor


def suppress_contact_flags(
    flags: list[tuple[int, float]],
    radius: int,
) -> list[tuple[int, float]]:
    """Greedy argmax suppression over ``(frame, impulse)`` flags.

    Flags are ranked by descending impulse and then ascending frame. A flag is
    accepted only when it is at least ``radius`` frames from every accepted flag.
    """
    ordered = sorted(flags, key=lambda flag: (-flag[1], flag[0]))
    accepted: list[tuple[int, float]] = []
    for frame, impulse in ordered:
        if all(abs(frame - other_frame) >= radius for other_frame, _ in accepted):
            accepted.append((frame, impulse))
    return sorted(accepted)


def wrist_contact_near(
    sticky_distances: np.ndarray | None, contact_frame: int, threshold: float,
) -> bool | None:
    """Check wrist-to-shuttle distance at a contact, measured in player body heights.

    Return None when no distance array was supplied. With an array, a finite
    distance within the threshold passes; a missing or excessive distance fails.
    """
    if sticky_distances is None:
        return None
    distance = sticky_distances[contact_frame]
    return bool(np.isfinite(distance) and distance <= threshold)
