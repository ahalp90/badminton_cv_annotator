"""Collect classifier features using the shared player tracker."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from shared.player_tracking import (
    StickyAnchorParams,
    compute_halfcourt_centres,
    track_frame_players,
)

from .base import ClipContext, HeuristicOutput, RawClip


def _run_clip(
    raw: RawClip, ctx: ClipContext, normalize_joints: Callable[..., np.ndarray], params: StickyAnchorParams,
) -> HeuristicOutput:
    """Track players and collect court positions, normalised joints and failure flags."""
    court_info = ctx.all_court_info[ctx.vid]
    video_resolution = ctx.res_df.loc[ctx.vid]
    resolution = (float(video_resolution['width']), float(video_resolution['height']))
    halfcourt_centres = compute_halfcourt_centres(court_info)
    running_positions = halfcourt_centres.copy()
    num_frames = raw.kps.shape[0]
    failed = np.zeros(num_frames, dtype=bool)
    positions = np.zeros((num_frames, 2, 2), dtype=np.float64)
    joints = np.zeros((num_frames, 2, raw.kps.shape[2], 2), dtype=np.float64)
    overcount = np.zeros(num_frames, dtype=bool)

    for frame in range(num_frames):
        detection_count = int(raw.ndet[frame])
        players = track_frame_players(
            raw.bboxes[frame, :detection_count], raw.scores[frame, :detection_count],
            raw.kps[frame, :detection_count], running_positions, halfcourt_centres, court_info, resolution, params,
        )
        selected = players.detection_indices >= 0
        failed[frame] = not selected.all()
        # A fully failed frame contributes no doubles evidence.
        if not selected.any():
            continue
        overcount[frame] = players.standing_count > 2
        for player_row in np.flatnonzero(selected):
            detection_index = players.detection_indices[player_row]
            positions[frame, player_row] = players.court_positions[player_row]
            joints[frame, player_row] = normalize_joints(
                arr=raw.kps[frame, detection_index][None, :, :].astype(np.float64),
                bbox=raw.bboxes[frame, detection_index][None, :].astype(np.float64),
                v_height=None,
                center_align=True,
            )[0]
    return HeuristicOutput(pos=positions, joints=joints, failed=failed, overcount=overcount)


def apply(raw: RawClip, ctx: ClipContext, **hyperparams) -> HeuristicOutput:
    """Run the registered sticky_anchor tracker with optional setting overrides."""
    from preparing_data.prepare_train_on_shuttleset import normalize_joints

    return _run_clip(raw, ctx, normalize_joints, StickyAnchorParams(**hyperparams))
