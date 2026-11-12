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
    PlayerRow,
    TrackedPlayers,
    WristGaps,
)


def contact_player_evidence(
    frame: int, track: np.ndarray, players: TrackedPlayers, wrist_gaps: WristGaps,
    bboxes: np.ndarray, scene_courts: Sequence[SceneCourt],
) -> tuple[CourtHalf | None, tuple[float, float] | None]:
    """Guess the hitter from the nearest wrist; return Top/Bot distances if usable.

    Distances are in body heights. One tracked player can identify a hitter;
    comparing the two alternating orders needs a player on each half. Return
    (None, None) for an invisible shuttle or missing wrist evidence.
    """
    if track[frame, 2] != 1:
        return None, None
    distances = wrist_gaps.per_player[frame]
    finite_distances = np.isfinite(distances)
    # Serve look-back frames before court coverage begins have no tracker distances.
    if not finite_distances.any():
        return None, None
    band_lo, band_hi = court_at_frame(scene_courts, frame).net_band

    # Equal distances favour the top tracked player.
    nearest_player_row = int(np.nanargmin(distances))
    detection_index = players.detection_indices[frame, nearest_player_row]
    foot_y = float(bboxes[frame, detection_index, 3])
    side = None
    if foot_y < band_lo:
        side = CourtHalf.TOP
    elif foot_y > band_hi:
        side = CourtHalf.BOT

    # Comparing Top/Bot hitter distances needs one player on each half.
    if not finite_distances.all():
        return side, None
    top_row_foot_y, bottom_row_foot_y = bboxes[frame, players.detection_indices[frame], 3]
    top_row_distance, bottom_row_distance = map(float, distances)
    if top_row_foot_y < band_lo and bottom_row_foot_y > band_hi:
        return side, (top_row_distance, bottom_row_distance)
    if bottom_row_foot_y < band_lo and top_row_foot_y > band_hi:
        return side, (bottom_row_distance, top_row_distance)
    return side, None


def fit_rally_sides(sequences: Sequence[ContactSequence]) -> tuple[FinishedRally, ...]:
    """Choose one alternating player order per rally and assign its contact sides.

    Individual guesses vote for the order. Wrist distances break equal votes.
    Return new sequences with fitted sides; keep the original objects unchanged.
    If no hitter is known, keep the guesses and leave the final hitter unknown.
    """
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

    Each hitter guess votes for a final hitter, assuming alternating sides.
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
            # An odd number of hits before a Top finish puts this hit on Bot.
            if (len(guesses) - index - 1) % 2:
                top_final_distance, bot_final_distance = bot_final_distance, top_final_distance
            top_final_cost += top_final_distance
            bot_final_cost += bot_final_distance
        if top_final_cost != bot_final_cost:
            return CourtHalf.TOP if top_final_cost < bot_final_cost else CourtHalf.BOT
        return final_guesses[0]
    return CourtHalf.TOP if top_votes > bot_votes else CourtHalf.BOT


def alternating_player_sides(final_half: CourtHalf, n_strokes: int) -> list[CourtHalf]:
    """Sides in contact order, alternating backwards from the known final hitter."""
    last = n_strokes - 1
    return [final_half if (last - i) % 2 == 0 else OTHER_HALF[final_half]
            for i in range(n_strokes)]


def measure_wrist_distances(
    track: np.ndarray,
    segments: list[tuple[int, int]],
    pose_kps: np.ndarray,
    resolution: tuple[float, float],
    half_window: int,
    players: TrackedPlayers,
) -> WristGaps:
    """Measure each tracked player's nearer wrist gap, divided by nearby box height.

    Measure every tracked frame, including frames where the shuttle is hidden;
    callers apply visibility.
    """
    gaps = WristGaps(np.full(len(track), np.inf), np.full((len(track), 2), np.inf))
    width, height = resolution
    for start, end in segments:
        gaps.per_player[start:end] = np.nan
        for frame in range(start, end):
            shuttle_x = track[frame, 0] * width
            shuttle_y = track[frame, 1] * height
            for player_row in PlayerRow:
                detection_index = players.detection_indices[frame, player_row]
                if detection_index < 0:
                    continue
                wrists = pose_kps[frame, detection_index, (WRIST_L, WRIST_R), :]
                numerator = float(np.hypot(wrists[:, 0] - shuttle_x, wrists[:, 1] - shuttle_y).min())
                window = players.box_heights_px[
                    max(start, frame - half_window):min(end, frame + half_window + 1), player_row
                ]
                if not np.isfinite(window).any():
                    raise ValueError(
                        f'wrist distance: no accepted finite height for player row {player_row} '
                        f'at frame {frame}'
                    )
                divisor = float(np.nanmean(window))
                if not np.isfinite(divisor) or divisor <= 0.0:
                    raise ValueError(
                        f'wrist distance: non-finite or non-positive body-scale '
                        f'denominator for player row {player_row} at frame {frame}'
                    )
                gaps.per_player[frame, player_row] = numerator / divisor

        for frame in range(start, end):
            finite_distances = gaps.per_player[frame][np.isfinite(gaps.per_player[frame])]
            gaps.nearest[frame] = float(finite_distances.min()) if len(finite_distances) else float('nan')
    return gaps
