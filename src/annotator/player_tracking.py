"""Follow the top and bottom players using their court positions across frames.

The sticky picker favours continuity with earlier selections. This module
records player rows and positions; contact attribution measures wrist gaps
in a separate step.
"""

import numpy as np

from annotator.array_calc_helpers import true_runs
from annotator.courts.frame_geometry import SceneCourt, court_at_frame
from annotator.types import ANKLE_L, ANKLE_R, PlayerRow, TrackedPlayers
from shared.player_tracking import (
    StickyAnchorParams,
    compute_halfcourt_centres,
    track_frame_players,
)


def tracker_segments(
    scene_courts: tuple[SceneCourt, ...], court_present: np.ndarray, n_frames: int,
) -> list[tuple[int, int]]:
    """Keep frame ranges where an accepted court scene and court_present overlap."""
    if (
        not isinstance(court_present, np.ndarray)
        or court_present.shape != (n_frames,)
        or court_present.dtype != np.bool_
    ):
        raise ValueError('court_present must have shape (n_frames,) and bool dtype')
    segments = []
    for scene in scene_courts:
        start, end = max(0, scene.start_frame), min(n_frames, scene.end_frame)
        if start >= end:
            continue
        for run_start, run_end in true_runs(court_present[start:end]):
            segments.append((start + run_start, start + run_end))
    return segments


def track_sticky_players(
    n_frames: int,
    segments: list[tuple[int, int]],
    pose_bboxes: np.ndarray,
    pose_scores: np.ndarray,
    pose_kps: np.ndarray,
    pose_ndet: np.ndarray,
    resolution: tuple[float, float],
    scene_courts: tuple[SceneCourt, ...],
) -> TrackedPlayers:
    """Track players frame by frame; save their detection indices and body positions.

    Each segment must lie within one court scene, as produced by tracker_segments.
    """
    params = StickyAnchorParams()
    players = TrackedPlayers(
        detection_indices=np.full((n_frames, 2), -1, dtype=int),
        ankle_xy=np.full((n_frames, 2, 2), np.nan),
        box_heights_px=np.full((n_frames, 2), np.nan),
    )

    for start, end in segments:
        scene = court_at_frame(scene_courts, start)
        halfcourt_centres = compute_halfcourt_centres(scene.court_info)
        running_positions = halfcourt_centres.copy()
        for frame in range(start, end):
            detection_count = int(pose_ndet[frame])
            frame_players = track_frame_players(
                pose_bboxes[frame, :detection_count], pose_scores[frame, :detection_count],
                pose_kps[frame, :detection_count], running_positions, halfcourt_centres, scene.court_info, resolution, params,
            )
            for player_row in PlayerRow:
                detection_index = int(frame_players.detection_indices[player_row])
                if detection_index < 0:
                    continue
                players.detection_indices[frame, player_row] = detection_index
                box = pose_bboxes[frame, detection_index].astype(np.float64)
                players.box_heights_px[frame, player_row] = box[3] - box[1]
                ankles = pose_kps[frame, detection_index, (ANKLE_L, ANKLE_R), :]
                players.ankle_xy[frame, player_row] = ankles.mean(axis=0) / np.asarray(resolution)
    return players
