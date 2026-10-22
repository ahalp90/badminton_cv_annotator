"""Rally contact candidate detection, gating, and suppression."""

import numpy as np

from ..config import PROXIMITY_MAX, Thresholds
from ..types import ContactCandidate, SmoothingMode
from .trajectory import _nan_rolling_mean, _rolling_mean

FLOOR_EPS = 1e-4


def span_impulses(
    track: np.ndarray, start: int, end: int, thresholds: Thresholds, *,
    smoothing_mode: SmoothingMode,
) -> np.ndarray | None:
    """Return ``|v_out - v_in|`` for each junction in one rally span.

    The track is already replay-masked when this is called from ``segment_video``.
    Junction ``k`` sits at local frame ``k + 1`` and touches three straddling frames.
    Track xy inputs are finite. IGNORE_INVISIBLE makes a float copy whose invisible
    xy values are NaN. A NaN output means its smoothing window was unmeasurable;
    downstream comparisons drop the corresponding junction.
    """
    smooth_window = thresholds.smooth_window
    span = track[start:end]
    if len(span) < smooth_window + 2:
        return None

    if smoothing_mode is SmoothingMode.ZERO_FILL:
        smooth_x = _rolling_mean(span[:, 0], smooth_window)
        smooth_y = _rolling_mean(span[:, 1], smooth_window)
    else:
        smooth_xy = span[:, :2].astype(float, copy=True)
        smooth_xy[span[:, 2] != 1] = np.nan
        smooth_x = _nan_rolling_mean(smooth_xy[:, 0], smooth_window)
        smooth_y = _nan_rolling_mean(smooth_xy[:, 1], smooth_window)
    velocity = np.diff(np.column_stack([smooth_x, smooth_y]), axis=0)
    return np.linalg.norm(velocity[1:] - velocity[:-1], axis=1)


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


def impulse_cell_candidates(
    track: np.ndarray, start: int, end: int, thresholds: Thresholds, *,
    smoothing_mode: SmoothingMode,
) -> list[tuple[int, float]]:
    """Find raw impulse candidates and retain their impulse for suppression.

    The measured rule is pure impulse. It uses the three-frame visibility mask,
    a rolling impulse floor, and largest-impulse-first de-duplication at the
    three-frame boundary used at 25 fps.
    """
    span = track[start:end]
    impulses = span_impulses(track, start, end, thresholds, smoothing_mode=smoothing_mode)
    if impulses is None:
        return []

    around_visible = (
        (span[:-2, 2] == 1) & (span[1:-1, 2] == 1) & (span[2:, 2] == 1)
    )
    half_window = thresholds.impulse_floor_half_window_frames
    dedup_radius = thresholds.contact_dedup_radius_frames
    floors = rolling_floor(impulses, around_visible, half_window)
    impulse_multiple = thresholds.contact_impulse_multiple
    impulse_pass = impulses / np.maximum(floors, FLOOR_EPS) > impulse_multiple
    is_contact = impulse_pass & around_visible

    candidate_local = np.flatnonzero(is_contact) + 1
    candidate_impulses = impulses[is_contact]
    kept: list[tuple[int, float]] = []
    # Stable sort: equal impulses keep the earlier frame, matching suppression's
    # (-impulse, frame) ordering. Exact ties occur in real data (sset_01 has one),
    # so an unstable sort here is a platform-dependent output.
    for candidate_index in np.argsort(-candidate_impulses, kind='stable'):
        local_frame = int(candidate_local[candidate_index])
        if all(
            abs(local_frame - other_frame) >= dedup_radius
            for other_frame, _other_impulse in kept
        ):
            kept.append((local_frame, float(candidate_impulses[candidate_index])))
    kept.sort()
    return [(start + local_frame, impulse) for local_frame, impulse in kept]


def contact_proximity_ok(
    track: np.ndarray, positions: np.ndarray | None, contact_frame: int
) -> bool | None:
    """Guardrail: does a tracked player sit near the shuttle at the contact frame?

    Never filters a contact; it annotates one. When no positions were supplied
    the check is unmeasured, which returns None (serialised blank downstream):
    a guardrail with no evidence must not read as a pass.

    :param track: `(t, 3)` whole-video track.
    :param positions: `(t, 2, 2)` `[slot, xy]` court positions, or None.
    :param contact_frame: whole-video frame index of the contact.
    :return: True/False when measured, None when no positions were supplied.
    """
    if positions is None:
        return None
    shuttle_xy = track[contact_frame, :2]  # (2,)
    player_xy = positions[contact_frame]  # (2, 2) [slot, xy]
    distances = np.linalg.norm(player_xy - shuttle_xy, axis=1)  # (2,) per slot
    if np.all(np.isnan(distances)):
        # Positions exist but both slots failed this frame: measured, unconfirmed.
        return False
    return bool(np.nanmin(distances) <= PROXIMITY_MAX)


def wrist_contact_near(
    sticky_distances: np.ndarray | None, contact_frame: int, threshold: float,
) -> bool | None:
    """The single-frame body-unit wrist gate on one contact, in player-box-height units.

    Mirrors `contact_proximity_ok`'s three-way verdict. None distances mean the gate never
    ran (no body-unit or pose/court inputs), which returns None (serialised blank
    downstream): raw candidates stand, per the recall-first convention. A NaN frame is
    measured-but-unconfirmed and fails closed to False, the measured arm's behaviour.
    """
    if sticky_distances is None:
        return None
    distance = sticky_distances[contact_frame]
    return bool(np.isfinite(distance) and distance <= threshold)


def suppress_contact_flags(
    flags: list[tuple[int, float]],
    radius: int,
) -> list[int]:
    """Greedy argmax suppression over ``(frame, impulse)`` flags.

    Flags are ranked by descending impulse and then ascending frame. A flag is
    accepted only when it is at least ``radius`` frames from every accepted flag.
    """
    ordered = sorted(flags, key=lambda flag: (-flag[1], flag[0]))
    accepted: list[int] = []
    for frame, _impulse in ordered:
        if all(abs(frame - other) >= radius for other in accepted):
            accepted.append(frame)
    return sorted(accepted)


def assemble_contacts(
    track: np.ndarray, positions: np.ndarray | None, spans: list[tuple[int, int]],
    thresholds: Thresholds, sticky_distances: np.ndarray | None,
    *, smoothing_mode: SmoothingMode,
) -> list[ContactCandidate]:
    """Detect, gate, and suppress contacts for already-selected spans."""
    raw_flags = [(rally_id, frame, impulse) for rally_id, (start, end) in enumerate(spans)
                 for frame, impulse in impulse_cell_candidates(
                     track, start, end, thresholds, smoothing_mode=smoothing_mode,
                 )]
    if sticky_distances is None:
        return [ContactCandidate(r, f, contact_proximity_ok(track, positions, f), None, None)
                for r, f, _ in raw_flags]
    wrist_distance = thresholds.wrist_distance_body_heights
    gated = [(f, impulse) for _r, f, impulse in raw_flags if wrist_contact_near(sticky_distances, f, wrist_distance)]
    radius = thresholds.contact_suppression_radius_frames
    accepted = set(suppress_contact_flags(gated, radius=radius))
    gate_frames = {frame for frame, _ in gated}
    return [ContactCandidate(r, f, contact_proximity_ok(track, positions, f), f in gate_frames,
                             f in gate_frames and f not in accepted)
            for r, f, _ in raw_flags]
