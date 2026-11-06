"""Sticky player evidence shared by serve and contact rules."""

import sys
from pathlib import Path

import numpy as np

from annotator.array_calc_helpers import true_runs
from annotator.courts.frame_geometry import SceneCourt, court_at_frame
from annotator.types import ANKLE_L, ANKLE_R, Slot, StickyResult

# sticky_anchor is part of BST-X, not the scraper package. Keep the import seam
# at the package boundary so the picker remains the single implementation.
_BST_X_ROOT = Path(__file__).resolve().parents[1] / 'bst_x'
if str(_BST_X_ROOT) not in sys.path:
    sys.path.insert(0, str(_BST_X_ROOT))

from preparing_data.heuristics import sticky_anchor  # noqa: E402
from preparing_data.heuristics.base import RawClip  # noqa: E402


def tracker_segments(
    scene_courts: tuple[SceneCourt, ...], court_present: np.ndarray, n_frames: int,
) -> list[tuple[int, int]]:
    """Intersect validated court scenes with the aligned court-present runs."""
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
) -> StickyResult:
    """Run the sequential sticky picker and retain its frame-aligned evidence.

    Each segment must lie within one court scene, as produced by tracker_segments.
    ``measure_sticky_distances`` fills distances and wrist gaps afterwards.
    """
    params = sticky_anchor.StickyAnchorParams()
    raw = RawClip(
        kps=pose_kps,
        bboxes=pose_bboxes,
        scores=pose_scores,
        # The picker does not read keypoint scores, but RawClip requires the field.
        kp_scores=np.zeros((*pose_scores.shape, 17), dtype=np.float32),
        ndet=pose_ndet,
    )
    evidence = StickyResult(
        distances=np.full(n_frames, np.inf),
        picks=np.full((n_frames, 2), -1, dtype=int),
        standing_count=np.zeros(n_frames, dtype=int),
        ankle_pos=np.full((n_frames, 2, 2), np.nan),
        bbox_height=np.full((n_frames, 2), np.nan),
        # Per-slot fail-closed sentinel: +inf outside tracker segments, NaN once a
        # tracker segment is analysed but a slot carries no finite gap.
        distances_per_slot=np.full((n_frames, 2), np.inf, dtype=np.float64),
        wrist_dist_px=np.full((n_frames, 2), np.inf, dtype=np.float64),
        analysed=np.zeros(n_frames, dtype=bool),
    )

    for start, end in segments:
        scene = court_at_frame(scene_courts, start)
        halfcourt_centre = sticky_anchor.compute_halfcourt_centres(scene.court_info)
        ema = halfcourt_centre.copy()
        for frame in range(start, end):
            evidence.analysed[frame] = True
            evidence.distances_per_slot[frame] = np.nan
            evidence.wrist_dist_px[frame] = np.nan
            analysis = sticky_anchor.analyse_frame(
                raw, frame, ema, halfcourt_centre, scene.court_info, resolution, params,
            )
            evidence.standing_count[frame] = analysis.standing_in_court_count
            if analysis.picks is None:
                ema[:] = halfcourt_centre
                continue
            assert analysis.court_base_pos is not None
            assert analysis.bboxes is not None
            assert analysis.filtered_to_raw is not None
            for slot in Slot:
                pick = analysis.picks[slot]
                if pick < 0:
                    ema[slot] = halfcourt_centre[slot]
                    continue
                candidate_position = analysis.court_base_pos[pick]
                if sticky_anchor.in_generous_court(candidate_position, params.update_gate_eps):
                    ema[slot] = (
                        params.ema_alpha * candidate_position
                        + (1 - params.ema_alpha) * ema[slot]
                    )
                raw_slot = int(analysis.filtered_to_raw[pick])
                evidence.picks[frame, slot] = raw_slot
                box = analysis.bboxes[pick]
                evidence.bbox_height[frame, slot] = box[3] - box[1]
                ankles = pose_kps[frame, raw_slot, (ANKLE_L, ANKLE_R), :]
                evidence.ankle_pos[frame, slot] = ankles.mean(axis=0) / np.asarray(resolution)
    return evidence
