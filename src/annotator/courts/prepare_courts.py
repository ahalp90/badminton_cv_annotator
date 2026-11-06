"""Accept detected courts using the players visible in each camera scene.

A scene needs court corners and two in-court players in at least half its
frames. Keep accepted geometry for annotation and records of every scene
for inspection, including rejected scenes.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
import pandas as pd

from shared.court import (
    HOMOGRAPHY_RESOLUTION,
    project_pixels_to_court,
)
from shared.court_model import COURT_LENGTH_M, COURT_WIDTH_M

# court_detector.run_video.VIDEO_RESULT_SCHEMA. Importing run_video would set
# single-thread limits for this whole process.
DETECTOR_RESULT_SCHEMA = 'court-detector-video/1'
SCENE_ROW_COLUMNS = (
    'video_id', 'start_frame', 'end_frame',
    'upleft_x', 'upleft_y', 'upright_x', 'upright_y',
    'downleft_x', 'downleft_y', 'downright_x', 'downright_y',
)
# Scene-row column prefix and its index in the TL, TR, BR, BL corner order.
SCENE_COLUMNS_BY_CORNER = (
    ('upleft', 0),
    ('upright', 1),
    ('downleft', 3),
    ('downright', 2),
)
# Image size used by fixed-court experiment inputs.
DETECTOR_RESOLUTION = (512.0, 288.0)
PERSON_COURT_MARGIN = 0.10
SCENE_VALID_MIN_FRACTION = 0.5
UNIT_COURT_CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float32,
)


class SceneStatus(StrEnum):
    """Court detector result for one scene, before checking player occupancy.

    Only COURT scenes have corners. They still need the player check before
    annotation can use them.
    """

    COURT = 'court'
    NO_COURT = 'no_court'  # the detector found none; no_court_reason says why
    SCENE_TOO_SHORT_FOR_FEET = 'scene_too_short_for_feet'  # unanalysed: shorter than the foot window
    DETECTION_FAILED = 'detection_failed'  # the search or fit raised; error records it


@dataclass(frozen=True)
class CourtInputs:
    """Accepted scene ranges and their court corners, in video pixels.

    Annotation builds each scene's geometry from ``homography_rows`` using
    the video's width and height in ``resolution``.
    """

    resolution: tuple[float, float]
    homography_rows: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, 'homography_rows', self.homography_rows.copy(deep=True))


@dataclass(frozen=True)
class DetectorScene:
    """Validated detector output for one scene, before the player check.

    Frame range includes start, excludes end. Corners use original video pixels
    in top-left, top-right, bottom-right, bottom-left order; None for no court.
    """

    start_frame: int
    end_frame: int  # exclusive
    analysed_frame: int
    status: SceneStatus
    corners_native_px: np.ndarray | None  # (4, 2) TL, TR, BR, BL source-video pixels; court scenes only
    no_court_reason: str | None
    reused_from: str | None  # earlier view whose court was reused, or None for a full search
    error: str | None


@dataclass(frozen=True)
class CourtSceneRecord:
    """Detector result plus player counts and acceptance for one scene.

    Keep rejected scenes too, so saved outputs explain missing court coverage.
    ``status`` is the detector result; ``scene_valid`` also requires the player
    check to pass. ``exactly_two_fraction`` is the fraction of frames with
    exactly two in-court people.

    :param analysed_frame: Frame inspected by the detector; None for fixed courts.
    :param corners_native_px: (4, 2) corners in original video pixels, ordered
        top-left, top-right, bottom-right, bottom-left. None for no court.
    """

    video_id: int | str
    scene_index: int
    start_frame: int
    end_frame: int
    status: SceneStatus
    analysed_frame: int | None
    corners_native_px: np.ndarray | None
    no_court_reason: str | None
    reused_from: str | None
    error: str | None
    exactly_two_count: int
    exactly_two_fraction: float
    scene_valid: bool

    def __post_init__(self) -> None:
        if self.corners_native_px is not None:
            object.__setattr__(self, 'corners_native_px', np.asarray(self.corners_native_px).copy())


@dataclass(frozen=True)
class CourtEvidenceResult:
    """Accepted court inputs, frame masks and records of all scenes.

    ``inputs`` supplies accepted court geometry for annotation; None if no
    court was accepted. ``keep_vote`` marks frames with exactly two in-court
    people; ``court_present`` marks whole accepted scenes.

    The dataset builder saves ``scene_records`` for every scene, including
    rejections, so missing court coverage can be checked afterwards.
    """

    inputs: CourtInputs | None
    scene_records: tuple[CourtSceneRecord, ...]
    keep_vote: np.ndarray
    court_present: np.ndarray

    def __post_init__(self) -> None:
        for field_name in ('keep_vote', 'court_present'):
            values = np.asarray(getattr(self, field_name), dtype=np.bool_)
            object.__setattr__(self, field_name, np.ascontiguousarray(values).copy())


class NoAcceptedCourtError(ValueError):
    """No accepted court; ``result`` retains the scene records and player votes."""

    def __init__(self, result: CourtEvidenceResult, message: str) -> None:
        super().__init__(message)
        self.result = result


def build_court_detector_evidence(
    video_id: int | str,
    resolution: tuple[float, float],
    native_size: tuple[float, float],
    scenes: Sequence[DetectorScene],
    bboxes: np.ndarray,
    scores: np.ndarray,
    ndet: np.ndarray,
) -> CourtEvidenceResult:
    """Accept court scenes with exactly two in-court people in at least half the frames.

    Keep each accepted scene's own corner rows and mark all its frames in
    ``court_present``. Keep rejected scenes in the diagnostic records too.
    No accepted scene raises NoAcceptedCourtError with those records attached.

    :param resolution: Pose-video width and height in pixels.
    :param native_size: Original video size used by the detected corners.
    """
    intervals = [(scene.start_frame, scene.end_frame) for scene in scenes]
    corners_refpx = [
        None if scene.corners_native_px is None else _as_ref_corners(scene.corners_native_px, native_size)
        for scene in scenes
    ]
    scene_infos = [None if corners is None else detected_court_info(corners) for corners in corners_refpx]
    keep_vote = build_keep_vote(bboxes, scores, ndet, resolution, intervals, scene_infos)
    scene_valid = [
        corners is not None and _scene_fraction(keep_vote, interval) >= SCENE_VALID_MIN_FRACTION
        for corners, interval in zip(corners_refpx, intervals)
    ]
    court_present = build_court_present(keep_vote, intervals, scene_valid)
    records = []
    for scene_index, (scene, valid) in enumerate(zip(scenes, scene_valid)):
        interval = (scene.start_frame, scene.end_frame)
        records.append(CourtSceneRecord(
            video_id=video_id,
            scene_index=scene_index,
            start_frame=scene.start_frame,
            end_frame=scene.end_frame,
            status=scene.status,
            analysed_frame=scene.analysed_frame,
            corners_native_px=scene.corners_native_px,
            no_court_reason=scene.no_court_reason,
            reused_from=scene.reused_from,
            error=scene.error,
            exactly_two_count=int(keep_vote[scene.start_frame:scene.end_frame].sum()),
            exactly_two_fraction=_scene_fraction(keep_vote, interval),
            scene_valid=valid,
        ))
    accepted_indices = [index for index, valid in enumerate(scene_valid) if valid]
    if not accepted_indices:
        statuses = dict(Counter(scene.status.value for scene in scenes))
        result = CourtEvidenceResult(None, tuple(records), keep_vote, court_present)
        raise NoAcceptedCourtError(result, f'no scene has an accepted court; scene statuses: {statuses}')

    inputs = CourtInputs(
        resolution=tuple(map(float, resolution)),
        homography_rows=build_scene_rows(
            video_id, [intervals[index] for index in accepted_indices],
            [corners_refpx[index] for index in accepted_indices], resolution,
        ),
    )
    return CourtEvidenceResult(inputs, tuple(records), keep_vote, court_present)


def read_detector_scenes(
    result: Mapping[str, object], *, frame_count: int, native_size: tuple[float, float],
) -> tuple[DetectorScene, ...]:
    """Validate saved detector output and convert its scene rows to records.

    Require the expected schema, video size and frame count. Scenes must cover
    the whole video without gaps or overlaps, incl start, excl end. Only court
    scenes may have corners.

    :param native_size: Original video width and height in pixels.
    """
    if result.get('schema') != DETECTOR_RESULT_SCHEMA:
        raise ValueError(f"unsupported court detector schema {result.get('schema')!r}")
    if result.get('frame_count') != frame_count:
        raise ValueError(f"court detector frame count {result.get('frame_count')!r} differs from {frame_count}")
    if result.get('native_size') != list(native_size):
        raise ValueError(f"court detector native size {result.get('native_size')!r} differs from {list(native_size)}")
    rows = result.get('scenes')
    if not isinstance(rows, list) or not rows:
        raise ValueError('court detector result has no scene rows')
    scenes = []
    next_frame = 0
    for index, row in enumerate(rows):
        start_frame, end_frame = row['start_frame'], row['end_frame']
        if start_frame != next_frame or not start_frame < end_frame <= frame_count:
            raise ValueError(f'court detector scene {index} [{start_frame}, {end_frame}) must start at frame '
                             f'{next_frame} and end after it, by frame {frame_count}')
        next_frame = end_frame
        status = SceneStatus(row['status'])
        corners = row['corners_native_px']
        if (corners is not None) != (status is SceneStatus.COURT):
            raise ValueError(f'court detector scene {index} has status {status.value!r}; only court scenes '
                             'carry corners')
        if corners is not None:
            corners = np.asarray(corners, dtype=float)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                raise ValueError(f'court detector scene {index} corners must be four finite points')
        if status is SceneStatus.DETECTION_FAILED and not row.get('error'):
            raise ValueError(f'court detector scene {index} failed without an error')
        scenes.append(DetectorScene(
            start_frame, end_frame, row['frame_index'], status, corners,
            row.get('no_court_reason'), row.get('reused_from'), row.get('error'),
        ))
    if next_frame != frame_count:
        raise ValueError(f'court detector scenes end at frame {next_frame}, expected {frame_count}')
    return tuple(scenes)


# Accept scenes from player occupancy

def build_keep_vote(
    bboxes: np.ndarray,
    scores: np.ndarray,
    ndet: np.ndarray,
    resolution: tuple[float, float],
    scene_intervals: Sequence[tuple[int, int]],
    provisional_court_info: Sequence[dict[str, object] | None],
) -> np.ndarray:
    """Mark frames with exactly two detections inside the court plus its allowed margin."""
    n_frames = len(bboxes)
    keep_vote = np.zeros(n_frames, dtype=np.bool_)
    for (start_frame, end_frame), court_info in zip(scene_intervals, provisional_court_info):
        if court_info is None:
            continue
        for frame in range(start_frame, end_frame):
            n_people = int(ndet[frame])
            boxes = np.asarray(bboxes[frame, :n_people], dtype=float)
            finite_scores = np.isfinite(np.asarray(scores[frame, :n_people], dtype=float))
            finite_boxes = np.isfinite(boxes).all(axis=1)
            if not finite_scores.any() or not finite_boxes.any():
                continue
            bottom_centres = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, boxes[:, 3]))
            normalised = project_pixels_to_court(bottom_centres.T, resolution, court_info).T
            inside = (
                finite_scores
                & finite_boxes
                & (normalised[:, 0] >= -PERSON_COURT_MARGIN)
                & (normalised[:, 0] <= 1.0 + PERSON_COURT_MARGIN)
                & (normalised[:, 1] >= -PERSON_COURT_MARGIN)
                & (normalised[:, 1] <= 1.0 + PERSON_COURT_MARGIN)
            )
            keep_vote[frame] = int(inside.sum()) == 2
    return keep_vote


def build_court_present(
    keep_vote: np.ndarray,
    scene_intervals: Sequence[tuple[int, int]],
    scene_valid: Sequence[bool],
) -> np.ndarray:
    """Mark every frame of accepted scenes; use keep_vote for the video length."""
    court_present = np.zeros(len(keep_vote), dtype=np.bool_)
    for scene_index, (start_frame, end_frame) in enumerate(scene_intervals):
        court_present[start_frame:end_frame] = bool(scene_valid[scene_index])
    return court_present


def _scene_fraction(keep_vote: np.ndarray, interval: tuple[int, int]) -> float:
    start_frame, end_frame = interval
    return float(keep_vote[start_frame:end_frame].mean())


# Court projection and landing uncertainty

def detected_court_info(corners_refpx: np.ndarray) -> dict[str, object]:
    """Build the pixel-to-court transform and unit-square bounds from four corners.

    Corners use reference pixels, ordered top-left, top-right, bottom-right,
    bottom-left.
    """
    homography = cv2.getPerspectiveTransform(
        np.asarray(corners_refpx, dtype=np.float32),
        UNIT_COURT_CORNERS,
    )
    return {
        'H': homography,
        'border_L': 0.0,
        'border_R': 1.0,
        'border_U': 0.0,
        'border_D': 1.0,
    }


def build_net_band(
    court_info: dict[str, object], resolution: tuple[float, float],
) -> tuple[float, float]:
    """Find image y bounds (pixels) for a one-metre strip centred on the net.

    Use the court centreline, half a metre on each side of the net.
    """
    points = np.array(
        [
            [0.5, 0.5 - 0.5 / COURT_LENGTH_M],
            [0.5, 0.5 + 0.5 / COURT_LENGTH_M],
        ],
        dtype=float,
    )
    reference_points = _normalised_court_to_reference(points, court_info)
    pose_y = reference_points[:, 1] * float(resolution[1]) / HOMOGRAPHY_RESOLUTION[1]
    finite_y = pose_y[np.isfinite(pose_y)]
    if len(finite_y) != 2:
        raise ValueError('net band projection is not finite')
    ordered = np.sort(finite_y)
    return round(float(ordered[0]), 1), round(float(ordered[1]), 1)


def corner_error_band_from_corners(
    corners_refpx: np.ndarray, court_info: dict, err_px: float,
) -> float:
    """Estimate landing uncertainty from small corner-position errors.

    Move each corner by +/- err_px horizontally and vertically, project through
    the supplied homography, and take the median displacement in court metres.
    Corners use 1280x720 reference pixels, ordered top-left, top-right,
    bottom-right, bottom-left. The homography stays fixed during this check.
    """
    corners_refpx = np.asarray(corners_refpx, dtype=float)
    base = project_pixels_to_court(corners_refpx.T, HOMOGRAPHY_RESOLUTION, court_info)
    displacements: list[float] = []
    for corner in range(4):
        for dx, dy in ((err_px, 0.0), (-err_px, 0.0), (0.0, err_px), (0.0, -err_px)):
            shifted = corners_refpx.copy()
            shifted[corner, 0] += dx
            shifted[corner, 1] += dy
            projected = project_pixels_to_court(shifted.T, HOMOGRAPHY_RESOLUTION, court_info)
            move = projected[:, corner] - base[:, corner]
            displacements.append(float(np.hypot(move[0] * COURT_WIDTH_M, move[1] * COURT_LENGTH_M)))
    return float(np.median(displacements))


def _as_ref_corners(corners: np.ndarray, source_resolution: tuple[float, float]) -> np.ndarray:
    """Convert TL/TR/BR/BL corners to 1280x720 reference pixels."""
    corners = np.asarray(corners, dtype=float)
    scale = np.asarray(HOMOGRAPHY_RESOLUTION, dtype=float) / np.asarray(source_resolution, dtype=float)
    return corners * scale


def _normalised_court_to_reference(
    normalised_xy: np.ndarray,
    court_info: dict[str, object],
) -> np.ndarray:
    """Map unit-square court coordinates back to reference pixels."""
    normalised_xy = np.asarray(normalised_xy, dtype=float)
    court_xy = np.empty_like(normalised_xy)
    court_xy[:, 0] = float(court_info['border_L']) + normalised_xy[:, 0] * (
        float(court_info['border_R']) - float(court_info['border_L'])
    )
    court_xy[:, 1] = float(court_info['border_U']) + normalised_xy[:, 1] * (
        float(court_info['border_D']) - float(court_info['border_U'])
    )
    inverse = np.linalg.inv(np.asarray(court_info['H'], dtype=float))
    homogeneous = np.column_stack((court_xy, np.ones(len(court_xy))))
    projected = (inverse @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


# Saved scene rows

def build_scene_rows(
    video_id: int | str,
    intervals: Sequence[tuple[int, int]],
    corners_refpx: Sequence[np.ndarray],
    resolution: tuple[float, float],
) -> pd.DataFrame:
    """Build per-scene corner rows in the pose video's pixel coordinates."""
    rows = [
        _scene_row(video_id, interval, corners, resolution)
        for interval, corners in zip(intervals, corners_refpx)
    ]
    return pd.DataFrame(rows, columns=SCENE_ROW_COLUMNS)


def _scene_row(
    video_id: int | str,
    interval: tuple[int, int],
    corners_refpx: np.ndarray,
    resolution: tuple[float, float],
) -> dict[str, object]:
    pose_corners = _as_ref_corners(corners_refpx, HOMOGRAPHY_RESOLUTION) * np.asarray(
        [
            float(resolution[0]) / HOMOGRAPHY_RESOLUTION[0],
            float(resolution[1]) / HOMOGRAPHY_RESOLUTION[1],
        ]
    )
    row: dict[str, object] = {
        'video_id': video_id,
        'start_frame': interval[0],
        'end_frame': interval[1],
    }
    for prefix, corner_index in SCENE_COLUMNS_BY_CORNER:
        row[f'{prefix}_x'] = float(pose_corners[corner_index, 0])
        row[f'{prefix}_y'] = float(pose_corners[corner_index, 1])
    return row
