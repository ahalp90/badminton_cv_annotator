"""Prepare detected court geometry and scene coverage for annotation."""
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
UNIT_COURT_CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float32,
)


class SceneStatus(StrEnum):
    """Court detector result for one scene; only COURT scenes have corners."""

    COURT = 'court'
    NO_COURT = 'no_court'  # the detector found none; no_court_reason says why
    SCENE_TOO_SHORT_FOR_FEET = 'scene_too_short_for_feet'  # unanalysed: shorter than the foot window
    DETECTION_FAILED = 'detection_failed'  # the search or fit raised; error records it


@dataclass(frozen=True)
class CourtInputs:
    """Detected scene ranges and their court corners, in video pixels.

    Annotation builds each scene's geometry from ``homography_rows`` using
    the video's width and height in ``resolution``.
    """

    resolution: tuple[float, float]
    homography_rows: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, 'homography_rows', self.homography_rows.copy(deep=True))


@dataclass(frozen=True)
class DetectorScene:
    """Validated detector output for one scene.

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
    """Detector result for one scene, including scenes without a court.

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

    def __post_init__(self) -> None:
        if self.corners_native_px is not None:
            object.__setattr__(self, 'corners_native_px', np.asarray(self.corners_native_px).copy())


@dataclass(frozen=True)
class CourtEvidenceResult:
    """Court inputs, frame coverage and detector records for every scene.

    ``inputs`` is None when the detector found no court. ``court_present``
    marks every frame in scenes with detected geometry.
    """

    inputs: CourtInputs | None
    scene_records: tuple[CourtSceneRecord, ...]
    court_present: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.court_present, dtype=np.bool_)
        object.__setattr__(self, 'court_present', np.ascontiguousarray(values).copy())


class NoDetectedCourtError(ValueError):
    """No detected court; ``result`` retains the scene records."""

    def __init__(self, result: CourtEvidenceResult, message: str) -> None:
        super().__init__(message)
        self.result = result


def build_court_detector_evidence(
    video_id: int | str,
    resolution: tuple[float, float],
    native_size: tuple[float, float],
    scenes: Sequence[DetectorScene],
) -> CourtEvidenceResult:
    """Prepare every detected court and retain records for scenes without one.

    Scenes have been validated by ``read_detector_scenes`` and cover the video.
    Raise NoDetectedCourtError with the scene records if no court was found.

    :param resolution: Annotation-video width and height in pixels.
    :param native_size: Original video size used by the detected corners.
    """
    court_present = np.zeros(scenes[-1].end_frame, dtype=np.bool_)
    intervals = []
    corners_refpx = []
    records = []
    for scene_index, scene in enumerate(scenes):
        if scene.corners_native_px is not None:
            intervals.append((scene.start_frame, scene.end_frame))
            corners_refpx.append(_as_ref_corners(scene.corners_native_px, native_size))
            court_present[scene.start_frame:scene.end_frame] = True
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
        ))
    if not intervals:
        statuses = dict(Counter(scene.status.value for scene in scenes))
        result = CourtEvidenceResult(None, tuple(records), court_present)
        raise NoDetectedCourtError(result, f'no scene has a detected court; scene statuses: {statuses}')

    inputs = CourtInputs(
        resolution=tuple(map(float, resolution)),
        homography_rows=build_scene_rows(video_id, intervals, corners_refpx, resolution),
    )
    return CourtEvidenceResult(inputs, tuple(records), court_present)


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
