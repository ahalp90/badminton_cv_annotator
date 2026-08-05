"""Sticky player evidence shared by serve and contact rules."""

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Mapping, NamedTuple

import numpy as np

from ..types import ANKLE_L, ANKLE_R, Slot, StickyResult, WRIST_L, WRIST_R

# sticky_anchor is part of BST-X, not the scraper package. Keep the import seam
# at the package boundary so the picker remains the single implementation.
_BST_X_ROOT = Path(__file__).resolve().parents[2] / 'bst_x'
if str(_BST_X_ROOT) not in sys.path:
    sys.path.insert(0, str(_BST_X_ROOT))

from preparing_data.heuristics import sticky_anchor  # noqa: E402
from preparing_data.heuristics.base import ClipContext, RawClip  # noqa: E402


class CourtGeo(NamedTuple):
    """A court geometry used to filter person detections.

    The caller constructs it from tracked calibration geometry. Feet inside `net_band` claim
    NEITHER court half because the net line carries 3D model error.

    :param x_range: foot-point x bounds, pixels.
    :param y_range: foot-point y bounds, pixels.
    :param net_band: top/bottom half-split band (low, high), pixels.
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    net_band: tuple[float, float]


# Fixed body-height window for the compatibility path. Resolved config supplies it explicitly.
BODY_UNIT_HALF_WINDOW = 12


def court_scale_slots(
    frame_bboxes: np.ndarray, frame_scores: np.ndarray, court_geo: CourtGeo,
) -> np.ndarray:
    """Original pose-slot indices of the court-scale detections, ascending.

    The filter returns original pose-slot identities rather than recovering
    them through score equality, which can alias detections with tied scores.

    :param frame_bboxes: (16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param frame_scores: (16,) detection scores, NaN on padding slots.
    :param court_geo: the court geometry to filter against.
    :return: (k,) int slot indices into the frame's pose arrays.
    """
    valid = np.isfinite(frame_scores)
    x1, y1, x2, y2 = frame_bboxes[valid].T  # each (m,) pixels
    foot_x = (x1 + x2) / 2.0  # bottom-centre; foot y is y2
    x_lo, x_hi = court_geo.x_range
    y_lo, y_hi = court_geo.y_range
    in_court = (x_lo <= foot_x) & (foot_x <= x_hi) & (y_lo <= y2) & (y2 <= y_hi)
    return np.flatnonzero(valid)[in_court]


def tracker_segments(
    homography_rows: Iterable[Mapping[str, object]], court_present: np.ndarray, n_frames: int,
) -> list[tuple[int, int]]:
    """Return scene-row intervals intersected with court-present runs.

    :param homography_rows: Scene rows with named ``start_frame`` and ``end_frame`` bounds.
    :param court_present: `(n_frames,)` boolean court-detection mask.
    :param n_frames: Number of frames in the aligned video arrays.
    :return: Maximal court-present half-open intervals within each scene row.
    """
    if (
        not isinstance(court_present, np.ndarray)
        or court_present.shape != (n_frames,)
        or court_present.dtype != np.bool_
    ):
        raise ValueError('court_present must have shape (n_frames,) and bool dtype')

    parsed_rows = []
    for row in homography_rows:
        try:
            start = int(row['start_frame'])
            end = int(row['end_frame'])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('homography row bounds must be integers') from exc
        if end < start:
            raise ValueError('homography rows must not be reversed')
        parsed_rows.append((start, end))

    parsed_rows.sort()
    for previous, current in zip(parsed_rows, parsed_rows[1:]):
        if current[0] < previous[1]:
            raise ValueError('homography rows must not overlap')

    segments = []
    for row_start, row_end in parsed_rows:
        start = max(0, row_start)
        end = min(n_frames, row_end)
        if start >= end:
            continue
        run_start = None
        for frame in range(start, end):
            if court_present[frame] and run_start is None:
                run_start = frame
            elif not court_present[frame] and run_start is not None:
                segments.append((run_start, frame))
                run_start = None
        if run_start is not None:
            segments.append((run_start, end))
    return segments


@dataclass
class _StickyEvidence:
    """Mutable frame-aligned arrays filled by the two sticky analysis phases."""

    picks: np.ndarray
    standing_count: np.ndarray
    ankle_pos: np.ndarray
    bbox_height: np.ndarray
    distances_per_slot: np.ndarray
    wrist_dist_px: np.ndarray
    analysed: np.ndarray


def _track_sticky_players(
    n_frames: int,
    segments: list[tuple[int, int]],
    pose_bboxes: np.ndarray,
    pose_scores: np.ndarray,
    pose_kps: np.ndarray,
    pose_ndet: np.ndarray,
    gate_video_id: str,
    gate_court_info: dict[str, dict],
    gate_resolution_table: object,
    resolution: tuple[float, float],
) -> _StickyEvidence:
    """Run the sequential sticky picker and retain its frame-aligned evidence."""
    params = sticky_anchor.StickyAnchorParams()
    raw = RawClip(
        kps=pose_kps,
        bboxes=pose_bboxes,
        scores=pose_scores,
        # The picker does not read keypoint scores, but RawClip requires the field.
        kp_scores=np.zeros((*pose_scores.shape, 17), dtype=np.float32),
        ndet=pose_ndet,
    )
    ctx = ClipContext(gate_video_id, gate_court_info, gate_resolution_table)
    court_info = ctx.all_court_info[ctx.vid]
    halfcourt_centre = sticky_anchor.compute_halfcourt_centres(court_info)
    evidence = _StickyEvidence(
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
        ema = halfcourt_centre.copy()
        for frame in range(start, end):
            evidence.analysed[frame] = True
            evidence.distances_per_slot[frame] = np.nan
            evidence.wrist_dist_px[frame] = np.nan
            analysis = sticky_anchor.analyse_frame(raw, frame, ema, halfcourt_centre, ctx, params)
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


def _measure_sticky_distances(
    track: np.ndarray,
    segments: list[tuple[int, int]],
    pose_kps: np.ndarray,
    resolution: tuple[float, float],
    half_window: int,
    evidence: _StickyEvidence,
) -> None:
    """Fill body-unit and raw-pixel wrist distances for accepted sticky picks."""
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


def _collapse_sticky_distances(evidence: _StickyEvidence) -> np.ndarray:
    """Collapse finite per-slot distances to the nearest accepted wrist."""
    gaps = np.full(len(evidence.analysed), np.inf)
    for frame in np.flatnonzero(evidence.analysed):
        finite_distances = evidence.distances_per_slot[frame][
            np.isfinite(evidence.distances_per_slot[frame])
        ]
        gaps[frame] = float(finite_distances.min()) if len(finite_distances) else float('nan')
    return gaps


def build_sticky_result(
    track: np.ndarray, segments: list[tuple[int, int]],
    pose_bboxes: np.ndarray, pose_scores: np.ndarray, pose_kps: np.ndarray,
    pose_ndet: np.ndarray, gate_video_id: str,
    gate_court_info: dict[str, dict], gate_resolution_table: object,
    resolution: tuple[float, float], half_window: int = BODY_UNIT_HALF_WINDOW,
) -> StickyResult:
    """Run sticky player tracking and measure its contact and serve evidence."""
    evidence = _track_sticky_players(
        len(track),
        segments,
        pose_bboxes,
        pose_scores,
        pose_kps,
        pose_ndet,
        gate_video_id,
        gate_court_info,
        gate_resolution_table,
        resolution,
    )
    _measure_sticky_distances(track, segments, pose_kps, resolution, half_window, evidence)
    return StickyResult(
        _collapse_sticky_distances(evidence),
        evidence.picks,
        evidence.standing_count,
        evidence.ankle_pos,
        evidence.bbox_height,
        evidence.distances_per_slot,
        evidence.wrist_dist_px,
        evidence.analysed,
    )
