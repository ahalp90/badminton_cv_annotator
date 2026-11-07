"""Find contacts from sudden shuttle movement, wrist proximity and spacing.

``segment_video`` first masks excluded footage and chooses rally ranges.
These functions measure and filter contacts within those ranges.
"""

import numpy as np

from annotator.array_calc_helpers import rolling_mean, rolling_nanmean
from annotator.config import Thresholds
from annotator.types import ContactCandidate, SmoothingMode

FLOOR_EPS = 1e-4


def assemble_contacts(
    track: np.ndarray, spans: list[tuple[int, int]],
    thresholds: Thresholds, nearest_wrist_distances: np.ndarray | None,
    *, smoothing_mode: SmoothingMode,
) -> list[ContactCandidate]:
    """Find contacts within rally ranges and record the wrist and spacing checks."""
    contacts = []
    near_wrist_candidates = []
    for rally_id, (start, end) in enumerate(spans):
        candidates = impulse_contact_candidates(track, start, end, thresholds, smoothing_mode=smoothing_mode)
        for frame, impulse in candidates:
            near_wrist = wrist_contact_near(nearest_wrist_distances, frame, thresholds.wrist_distance_body_heights)
            contacts.append(ContactCandidate(rally_id, frame, near_wrist, None))
            if near_wrist:
                near_wrist_candidates.append((frame, impulse))
    if nearest_wrist_distances is None:
        return contacts

    # This spacing check also compares contacts across rally boundaries.
    radius = thresholds.contact_suppression_radius_frames
    accepted = {frame for frame, _ in keep_strongest_spaced_contacts(near_wrist_candidates, radius=radius)}
    return [
        contact._replace(temporal_proximity_suppressed=contact.wrist_near and contact.contact_frame not in accepted)
        for contact in contacts
    ]


def impulse_contact_candidates(
    track: np.ndarray, start: int, end: int, thresholds: Thresholds, *,
    smoothing_mode: SmoothingMode,
) -> list[tuple[int, float]]:
    """Find sudden changes in shuttle velocity, keeping the strongest nearby ones.

    Require visibility on all three frames around each velocity change. Compare
    its size with the local median, then apply the configured minimum spacing.
    """
    measured = span_impulses(
        track, start, end, smooth_window=thresholds.smooth_window,
        impulse_floor_half_window=thresholds.impulse_floor_half_window_frames,
        smoothing_mode=smoothing_mode,
    )
    if measured is None:
        return []
    impulses, ratios, around_visible = measured
    impulse_pass = ratios > thresholds.contact_impulse_multiple
    is_contact = impulse_pass & around_visible

    candidates = [
        (start + int(impulse_index) + 1, float(impulses[impulse_index])) for impulse_index in np.flatnonzero(is_contact)
    ]
    return keep_strongest_spaced_contacts(candidates, radius=thresholds.contact_dedup_radius_frames)


def span_impulses(
    track: np.ndarray, start: int, end: int, *,
    smooth_window: int, impulse_floor_half_window: int,
    smoothing_mode: SmoothingMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Measure changes in velocity and their size relative to nearby changes.

    Output index k corresponds to video frame start + k + 1, the middle of
    three frames. Return magnitudes, ratios to the local median and visibility
    across those three frames. Return None if fewer than smooth_window + 2 samples.

    IGNORE_INVISIBLE treats invisible xy values as NaN while smoothing.
    INCLUDE_INVISIBLE uses the supplied coordinates, including invisible frames.
    A window with no visible samples gives NaN in IGNORE_INVISIBLE mode.
    """
    span = track[start:end]
    if len(span) < smooth_window + 2:
        return None

    if smoothing_mode is SmoothingMode.INCLUDE_INVISIBLE:
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
    floors = rolling_floor(impulses, around_visible, impulse_floor_half_window)
    ratios = impulses / np.maximum(floors, FLOOR_EPS)
    return impulses, ratios, around_visible


def rolling_floor(
    values: np.ndarray, around_visible: np.ndarray,
    half_window: int,
) -> np.ndarray:
    """Local median impulse, using only fully visible three-frame samples.

    ``half_window`` counts samples on each side. Return NaN if none are usable;
    the contact then fails the impulse check.
    """
    visible_values = np.where(around_visible, values, np.nan)
    floor = np.full(len(values), np.nan)
    for impulse_index in range(len(values)):
        window_start = max(impulse_index - half_window, 0)
        window_end = min(impulse_index + half_window + 1, len(values))
        window = visible_values[window_start:window_end]
        if np.isfinite(window).any():
            floor[impulse_index] = np.nanmedian(window)
    return floor


def keep_strongest_spaced_contacts(
    flags: list[tuple[int, float]],
    radius: int,
) -> list[tuple[int, float]]:
    """Keep the largest impulses, with at least ``radius`` frames between contacts.

    Input pairs are (frame, impulse). Earlier frames win equal-impulse ties.
    Return accepted pairs in frame order.
    """
    ordered = sorted(flags, key=lambda flag: (-flag[1], flag[0]))
    accepted: list[tuple[int, float]] = []
    for frame, impulse in ordered:
        if all(abs(frame - other_frame) >= radius for other_frame, _ in accepted):
            accepted.append((frame, impulse))
    return sorted(accepted)


def wrist_contact_near(
    nearest_wrist_distances: np.ndarray | None, contact_frame: int, threshold: float,
) -> bool | None:
    """Check wrist-to-shuttle distance at a contact, measured in player body heights.

    Return None when no distance array was supplied. With an array, a finite
    distance within the threshold passes; a missing or excessive distance fails.
    """
    if nearest_wrist_distances is None:
        return None
    distance = nearest_wrist_distances[contact_frame]
    return bool(np.isfinite(distance) and distance <= threshold)
