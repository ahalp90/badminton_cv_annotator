"""Select top/bottom players and follow their court positions across frames."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from shared.court import normalize_position, project_pixels_to_court

TOP_ROW = 0
BOTTOM_ROW = 1
DOUBLES_COUNT_MARGIN = 0.05
SITTING_THRESHOLD = -0.3


@dataclass(frozen=True)
class StickyAnchorParams:
    """Player selection and running-position settings, shared by both consumers."""

    prior_weight: float = 0.75
    ema_alpha: float = 0.1
    sanity_ceiling: float = 0.6
    generous_margin: float = 0.15
    score_filter: float = 0.2
    tiebreaker_tol: float = 0.05
    sitting_threshold: float = SITTING_THRESHOLD
    update_gate_eps: float = 0.01
    count_margin: float = DOUBLES_COUNT_MARGIN


class FramePlayers(NamedTuple):
    """Selected players in top/bottom row order, plus standing count for doubles.

    Detection indices refer to the original pose arrays. Missing players have
    index -1 and NaN court coordinates. Count includes unselected detections.
    """

    detection_indices: np.ndarray
    court_positions: np.ndarray
    standing_count: int


def track_frame_players(
    bboxes: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray,
    running_positions: np.ndarray,
    halfcourt_centres: np.ndarray,
    court_info: dict,
    resolution: tuple[float, float],
    params: StickyAnchorParams,
) -> FramePlayers:
    """Select players using court position and update running_positions in place.

    Supply only real detections for this frame. Initialise running_positions
    from halfcourt_centres at each new tracking interval. Missing selections
    reset that player's running position; off-court selections do not update it.
    """
    detection_indices = np.flatnonzero(scores > params.score_filter)
    boxes = bboxes[detection_indices].astype(np.float64)
    poses = keypoints[detection_indices].astype(np.float64)
    positions = project_bbox_bottom_centre(boxes, court_info, resolution)
    # Infinity remains a candidate; distance comparisons decide eligibility.
    valid_projection = ~np.isnan(positions).any(axis=1)
    detection_indices = detection_indices[valid_projection]
    boxes, poses, positions = boxes[valid_projection], poses[valid_projection], positions[valid_projection]

    anchors = params.prior_weight * halfcourt_centres + (1 - params.prior_weight) * running_positions
    distances = np.linalg.norm(positions[:, None, :] - anchors[None, :, :], axis=-1)
    sitting = is_sitting(poses, params.sitting_threshold)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    standing_count = count_standing_in_court(positions, sitting, params.count_margin)

    picks = [-1, -1]
    # Bottom chooses first when a detection is equally close to both players.
    for player_row, other_row in ((BOTTOM_ROW, TOP_ROW), (TOP_ROW, BOTTOM_ROW)):
        within_distance = distances[:, player_row] <= params.sanity_ceiling
        nearer_own_player = distances[:, player_row] <= distances[:, other_row]
        eligible = within_distance & nearer_own_player
        if picks[other_row] >= 0:
            eligible[picks[other_row]] = False
        if not eligible.any():
            continue
        eligible_indices = np.flatnonzero(eligible)
        winner = int(eligible_indices[np.argmin(distances[eligible_indices, player_row])])
        tied = eligible & (np.abs(distances[:, player_row] - distances[winner, player_row]) < params.tiebreaker_tol)
        if tied.sum() > 1:
            standing_tied = np.flatnonzero(tied & ~sitting)
            if len(standing_tied):
                winner = int(standing_tied[np.argmax(areas[standing_tied])])
        picks[player_row] = winner

    # Two off-court picks usually indicate a cutaway or bystanders.
    if picks[TOP_ROW] >= 0 and picks[BOTTOM_ROW] >= 0:
        top_inside = in_court(positions[picks[TOP_ROW]], params.generous_margin)
        bottom_inside = in_court(positions[picks[BOTTOM_ROW]], params.generous_margin)
        if not top_inside and not bottom_inside:
            picks = [-1, -1]

    selected_indices = np.full(2, -1, dtype=int)
    selected_positions = np.full((2, 2), np.nan)
    for player_row, pick in enumerate(picks):
        if pick < 0:
            running_positions[player_row] = halfcourt_centres[player_row]
            continue
        position = positions[pick]
        selected_indices[player_row] = detection_indices[pick]
        selected_positions[player_row] = position
        if in_court(position, params.update_gate_eps):
            running_positions[player_row] = (
                params.ema_alpha * position + (1 - params.ema_alpha) * running_positions[player_row]
            )
    return FramePlayers(selected_indices, selected_positions, standing_count)


def compute_halfcourt_centres(court_info: dict) -> np.ndarray:
    """Top/bottom halfcourt centres in normalised court coordinates."""
    left, right = court_info['border_L'], court_info['border_R']
    top, bottom = court_info['border_U'], court_info['border_D']
    centre_x = (left + right) / 2
    top_y = top + (bottom - top) / 4
    bottom_y = top + 3 * (bottom - top) / 4
    positions = np.array([[centre_x, centre_x], [top_y, bottom_y]], dtype=np.float64)
    return normalize_position(positions, court_info).T


def project_bbox_bottom_centre(
    bboxes: np.ndarray, court_info: dict, resolution: tuple[float, float],
) -> np.ndarray:
    """Use each box's bottom centre as its foot position; return court xy."""
    x1, _, x2, y2 = bboxes.T
    bottom_centres = np.stack([(x1 + x2) / 2, y2], axis=0)
    return project_pixels_to_court(bottom_centres, resolution, court_info).T


def in_court(position: np.ndarray, margin: float) -> bool:
    return bool(-margin <= position[0] <= 1 + margin and -margin <= position[1] <= 1 + margin)


def is_sitting(keypoints: np.ndarray, sitting_threshold: float) -> np.ndarray:
    """Identify seated poses from knees across, rather than below, the torso.

    Project hip-to-knee onto hip-to-shoulder. Standing ratios are around -0.8;
    seated ratios are near zero. A collapsed torso does not count as sitting.
    Input: (detections, COCO joints, xy). Return one boolean per detection.
    """
    shoulders = (keypoints[:, 5] + keypoints[:, 6]) / 2
    hips = (keypoints[:, 11] + keypoints[:, 12]) / 2
    knees = (keypoints[:, 13] + keypoints[:, 14]) / 2
    body_up = shoulders - hips
    knee_offset = knees - hips
    torso_length_sq = (body_up * body_up).sum(axis=1)
    collapsed_torso = torso_length_sq < 1e-6
    ratio = (knee_offset * body_up).sum(axis=1) / np.where(collapsed_torso, 1.0, torso_length_sq)
    return (ratio > sitting_threshold) & ~collapsed_torso


def count_standing_in_court(positions: np.ndarray, sitting: np.ndarray, margin: float) -> int:
    """Count standing candidates near the court, including players not selected.

    Seated officials can be close to the boundary; exclude them from doubles
    evidence. Positions and sitting flags must follow the same detection order.
    """
    in_margin = ((positions >= -margin) & (positions <= 1 + margin)).all(axis=1)
    return int((in_margin & ~sitting).sum())
