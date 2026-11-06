"""Measure shuttle-to-wrist gaps, attribute hitters and fit alternating player order."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from annotator.courts.frame_geometry import SceneCourt, court_at_frame
from annotator.types import (
    OTHER_HALF,
    WRIST_L,
    WRIST_R,
    ContactSequence,
    CourtHalf,
    FinishedRally,
    Slot,
    StickyResult,
)


def contact_player_evidence(
    frame: int, track: np.ndarray, sticky: StickyResult, bboxes: np.ndarray, scene_courts: Sequence[SceneCourt],
) -> tuple[CourtHalf | None, tuple[float, float] | None]:
    """Return the nearest player's half and optional TOP/BOT wrist gaps.

    Wrist gaps are in body-height units. A single tracked player can identify
    the hitter; comparing alternating orders needs one player on each half.
    Invisible shuttles and missing wrist evidence leave both results unknown.
    """
    if track[frame, 2] != 1:
        return None, None
    distances = sticky.distances_per_slot[frame]
    finite_distances = np.isfinite(distances)
    # Serve look-back frames before an accepted scene have no tracker distances.
    if not finite_distances.any():
        return None, None
    band_lo, band_hi = court_at_frame(scene_courts, frame).net_band

    # A finite value beats sentinels; exact ties retain the far/top tracker pick.
    nearest = int(np.nanargmin(distances))
    slot = sticky.picks[frame, nearest]
    foot_y = float(bboxes[frame, slot, 3])
    side = None
    if foot_y < band_lo:
        side = CourtHalf.TOP
    elif foot_y > band_hi:
        side = CourtHalf.BOT

    # A missing player or two picks on the same half cannot supply competing costs.
    if not finite_distances.all():
        return side, None
    first_foot, second_foot = bboxes[frame, sticky.picks[frame], 3]
    first_distance, second_distance = map(float, distances)
    if first_foot < band_lo and second_foot > band_hi:
        return side, (first_distance, second_distance)
    if second_foot < band_lo and first_foot > band_hi:
        return side, (second_distance, first_distance)
    return side, None


def fit_rally_sides(sequences: Sequence[ContactSequence]) -> tuple[FinishedRally, ...]:
    """Assign one alternating top/bottom order to each rally and retain its final hitter."""
    rallies = []
    for sequence in sequences:
        final_half = fit_alternation(
            [event.side for event in sequence.events], [event.wrist_distances for event in sequence.events],
        )
        if final_half is not None:
            fitted = alternating_player_sides(final_half, len(sequence.events))
            events = tuple(replace(event, side=side) for event, side in zip(sequence.events, fitted, strict=True))
            sequence = replace(sequence, events=events)
        rallies.append(FinishedRally(sequence, final_half))
    return tuple(rallies)


def fit_alternation(
    guesses: list[CourtHalf | None], wrist_distances: Sequence[tuple[float, float] | None] = (),
) -> CourtHalf | None:
    """Choose the alternating order that agrees with the most player guesses.

    Each known guess votes for the final striker implied by its position.
    Break equal votes with the lower summed wrist distance when available, then
    the first known guess. Return None if no player guess is known.
    """
    final_guesses = []
    for index, guess in enumerate(guesses):
        if guess is not None:
            remaining = len(guesses) - index - 1
            final_guesses.append(guess if remaining % 2 == 0 else OTHER_HALF[guess])
    if not final_guesses:
        return None
    top_votes = final_guesses.count(CourtHalf.TOP)
    bot_votes = final_guesses.count(CourtHalf.BOT)
    if top_votes == bot_votes:
        top_final_cost = bot_final_cost = 0.0
        for index, distances in enumerate(wrist_distances):
            if distances is None:
                continue
            top_final_distance, bot_final_distance = distances
            # An odd number of strokes before a TOP finish, BOT is hitting.
            if (len(guesses) - index - 1) % 2:
                top_final_distance, bot_final_distance = bot_final_distance, top_final_distance
            top_final_cost += top_final_distance
            bot_final_cost += bot_final_distance
        if top_final_cost != bot_final_cost:
            return CourtHalf.TOP if top_final_cost < bot_final_cost else CourtHalf.BOT
        return final_guesses[0]
    return CourtHalf.TOP if top_votes > bot_votes else CourtHalf.BOT


def alternating_player_sides(final_half: CourtHalf, n_strokes: int) -> list[CourtHalf]:
    """Alternating half per stroke for a rally whose LAST stroke is `final_half`.

    Stroke i counts back from the last (index n-1 = final_half); each step back flips halves.
    The two possible phases (final_half Top vs Bot) are exact complements of each other.
    """
    last = n_strokes - 1
    return [final_half if (last - i) % 2 == 0 else OTHER_HALF[final_half]
            for i in range(n_strokes)]


def measure_wrist_distances(
    track: np.ndarray,
    segments: list[tuple[int, int]],
    pose_kps: np.ndarray,
    resolution: tuple[float, float],
    half_window: int,
    evidence: StickyResult,
) -> None:
    """Measure shuttle gaps to each tracked player's nearer wrist.

    Store height-normalised gaps for every analysed frame with a picked player,
    and pixel gaps only where the shuttle is visible. Heights are the player's
    local mean box height. Retain the nearest finite normalised gap per frame.
    Update the supplied evidence in place; player tracking has already selected
    the player rows.
    """
    width, height = resolution
    for start, end in segments:
        for frame in range(start, end):
            shuttle_x = track[frame, 0] * width
            shuttle_y = track[frame, 1] * height
            for half in Slot:
                pick = evidence.picks[frame, half]
                if pick < 0:
                    continue
                wrists = pose_kps[frame, pick, (WRIST_L, WRIST_R), :]
                numerator = float(np.hypot(wrists[:, 0] - shuttle_x, wrists[:, 1] - shuttle_y).min())
                window = evidence.bbox_height[
                    max(start, frame - half_window):min(end, frame + half_window + 1), half
                ]
                if not np.isfinite(window).any():
                    raise ValueError(
                        f'sticky body-unit distance: no accepted finite height for slot {half} '
                        f'at frame {frame}'
                    )
                divisor = float(np.nanmean(window))
                if not np.isfinite(divisor) or divisor <= 0.0:
                    raise ValueError(
                        f'sticky body-unit distance: non-finite or non-positive body-scale '
                        f'denominator for slot {half} at frame {frame}'
                    )
                evidence.distances_per_slot[frame, half] = numerator / divisor
                if track[frame, 2] == 1:
                    evidence.wrist_dist_px[frame, half] = numerator

    for frame in np.flatnonzero(evidence.analysed):
        finite_distances = evidence.distances_per_slot[frame][np.isfinite(evidence.distances_per_slot[frame])]
        evidence.distances[frame] = float(finite_distances.min()) if len(finite_distances) else float('nan')
