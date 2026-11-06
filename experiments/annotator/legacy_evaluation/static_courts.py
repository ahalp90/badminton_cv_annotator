"""Static ShuttleSet court fixtures used by the historical measurement runner."""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.courts.prepare_courts import (
    DETECTOR_RESOLUTION,
    SCENE_VALID_MIN_FRACTION,
    CourtEvidenceResult,
    CourtInputs,
    CourtSceneRecord,
    SceneStatus,
    _scene_fraction,
    build_court_present,
    build_keep_vote,
    build_scene_rows,
)
from shared.camera_cuts import (
    COMPOSITION_CONTENT_THRESHOLD,
    MIN_SCENE_FRAMES_AT_30_FPS,
    detect_cuts,
)
from shared.court import HOMOGRAPHY_RESOLUTION, get_corner_camera, get_court_info
from shared.frame_timing import frames_at_fps

SHUTTLESET_TO_CLOCKWISE_CORNER_ORDER = (0, 1, 3, 2)


def _normalise_intervals(raw_cuts: Sequence[tuple[int, int]] | pd.DataFrame) -> list[tuple[int, int]]:
    """Return sorted, contiguous half-open intervals."""
    if isinstance(raw_cuts, pd.DataFrame):
        interval_values = raw_cuts[['start_frame', 'end_frame']].itertuples(index=False, name=None)
    else:
        interval_values = raw_cuts
    intervals = sorted((int(start), int(end)) for start, end in interval_values)
    if not intervals:
        raise ValueError('raw cuts must contain at least one scene')
    expected_start = intervals[0][0]
    if expected_start != 0:
        raise ValueError('raw cuts must begin at frame zero')
    for start, end in intervals:
        if start != expected_start or end <= start:
            raise ValueError('raw cuts must be non-empty, contiguous half-open intervals')
        expected_start = end
    return intervals



def build_raw_cut_intervals(video_path: Path, n_frames: int, fps: float) -> list[tuple[int, int]]:
    """Detect camera cuts once and cover the video with half-open scene intervals."""
    cut_frames = detect_cuts(
        video_path,
        expected_frames=n_frames,
        threshold=COMPOSITION_CONTENT_THRESHOLD,
        min_scene_len=frames_at_fps(MIN_SCENE_FRAMES_AT_30_FPS, fps),
    )
    cut_frames = np.asarray(cut_frames, dtype=int)
    boundaries = np.concatenate((np.array([0], dtype=int), cut_frames, np.array([n_frames], dtype=int)))
    intervals = [(int(start), int(end)) for start, end in itertools.pairwise(boundaries)]
    if any(start < 0 or end > n_frames or end <= start for start, end in intervals):
        raise ValueError('detected cuts do not form non-empty in-range intervals')
    if intervals[0][0] != 0 or intervals[-1][1] != n_frames:
        raise ValueError('detected cuts do not cover the video')
    if any(end != next_start for (_, end), (next_start, _) in itertools.pairwise(intervals)):
        raise ValueError('detected cuts contain a gap or overlap')
    return intervals



def _as_native_corners(corners_refpx: np.ndarray) -> np.ndarray:
    """Convert reference-space corners to native 512x288 video pixels."""
    scale = np.asarray(DETECTOR_RESOLUTION, dtype=float) / np.asarray(HOMOGRAPHY_RESOLUTION, dtype=float)
    return np.asarray(corners_refpx, dtype=float) * scale



def _static_corners_refpx(homography_row: pd.Series) -> np.ndarray:
    """Return static row corners in TL/TR/BR/BL order."""
    source_order = get_corner_camera(homography_row).T
    return source_order[list(SHUTTLESET_TO_CLOCKWISE_CORNER_ORDER)].copy()



def build_static_court_inputs(
    video_id: int | str,
    homo_df: pd.DataFrame,
    resolution: tuple[float, float],
    raw_cuts: Sequence[tuple[int, int]] | pd.DataFrame,
) -> CourtInputs:
    """Build static inputs from the existing ShuttleSet homography row."""
    intervals = _normalise_intervals(raw_cuts)
    active_corners = _static_corners_refpx(homo_df.loc[video_id])
    homography_rows = build_scene_rows(video_id, intervals, [active_corners] * len(intervals), resolution)
    return CourtInputs(
        resolution=tuple(map(float, resolution)),
        homography_rows=homography_rows,
    )



def build_static_court_evidence(
    video_id: int | str,
    homo_df: pd.DataFrame,
    resolution: tuple[float, float],
    raw_cuts: Sequence[tuple[int, int]] | pd.DataFrame,
    bboxes: np.ndarray,
    scores: np.ndarray,
    ndet: np.ndarray,
) -> CourtEvidenceResult:
    """Build masks and scene records using one supplied court calibration."""
    intervals = _normalise_intervals(raw_cuts)
    inputs = build_static_court_inputs(
        video_id, homo_df, resolution, intervals,
    )
    provisional_infos = [get_court_info(homo_df, video_id)] * len(intervals)
    keep_vote = build_keep_vote(
        bboxes, scores, ndet, resolution, intervals, provisional_infos,
    )
    scene_valid = [
        _scene_fraction(keep_vote, interval) >= SCENE_VALID_MIN_FRACTION
        for interval in intervals
    ]
    court_present = build_court_present(keep_vote, intervals, scene_valid)
    static_corners_px = _as_native_corners(_static_corners_refpx(homo_df.loc[video_id]))
    records = []
    for scene_index, ((start, end), valid) in enumerate(zip(intervals, scene_valid)):
        records.append(CourtSceneRecord(
            video_id=video_id,
            scene_index=scene_index,
            start_frame=start,
            end_frame=end,
            status=SceneStatus.COURT,
            analysed_frame=None,
            corners_native_px=static_corners_px,
            no_court_reason=None,
            reused_from=None,
            error=None,
            exactly_two_count=int(keep_vote[start:end].sum()),
            exactly_two_fraction=_scene_fraction(keep_vote, (start, end)),
            scene_valid=valid,
        ))
    return CourtEvidenceResult(inputs, tuple(records), keep_vote, court_present)
