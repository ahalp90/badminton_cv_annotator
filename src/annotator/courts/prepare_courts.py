"""Prepare detected court geometry and accepted-frame masks for annotation.

Each detected scene is accepted using player occupancy against its own court
geometry. Static ShuttleSet fixtures are prepared by the historical experiments.
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

DETECTED_PARENT = 'court_detector'
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
# Pixel size of the fixed measurement's downsampled videos. Static scene records
# store their corners in these pixels.
DETECTOR_RESOLUTION = (512.0, 288.0)
PERSON_COURT_MARGIN = 0.10
SCENE_VALID_MIN_FRACTION = 0.5
UNIT_COURT_CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float32,
)


class SceneStatus(StrEnum):
    """The court detector's classification of one scene.

    Only ``court`` scenes carry corners. The historical supplied-calibration
    helpers in ``experiments.annotator.legacy_evaluation.static_courts`` also
    assign that status because their homography rows already provide a court.
    """

    COURT = 'court'
    NO_COURT = 'no_court'  # the detector found none; no_court_reason says why
    SCENE_TOO_SHORT_FOR_FEET = 'scene_too_short_for_feet'  # unanalysed: shorter than the foot window
    DETECTION_FAILED = 'detection_failed'  # the search or fit raised; error records it


@dataclass(frozen=True)
class CourtInputs:
    """Court geometry, frame masks and scene rows supplied to annotation.

    Construction copies mutable arrays and tables; frozen dataclass fields alone
    would only prevent rebinding.
    """

    court_info: dict[str, object]
    net_band: tuple[float, float]
    resolution: tuple[float, float]
    homography_rows: pd.DataFrame
    landing_error_band_m: float
    active_corners_refpx: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, 'court_info', _copy_court_info(self.court_info))
        object.__setattr__(self, 'homography_rows', self.homography_rows.copy(deep=True))
        object.__setattr__(self, 'active_corners_refpx', np.asarray(self.active_corners_refpx).copy())


@dataclass(frozen=True)
class DetectorScene:
    """One validated scene row from a ``court-detector-video/1`` result."""

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
    """Typed evidence for one raw scene, ready for the court-scenes writer.

    A scene outside ``court_present`` keeps its reason here: a status other than
    ``court``, or a ``court`` whose person vote failed (``scene_valid`` False).

    :param parent: court-evidence producer profile used for this scene.
    :param analysed_frame: the frame the detector analysed; None for the static parent.
    :param corners_native_px: (4, 2) TL, TR, BR, BL corners in video pixels, or None.
    """

    video_id: int | str
    case_id: str
    parent: str
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
    """Prepared court inputs, occupancy votes and per-scene records."""

    inputs: CourtInputs | None
    scene_records: tuple[CourtSceneRecord, ...]
    keep_vote: np.ndarray
    court_present: np.ndarray

    def __post_init__(self) -> None:
        for field_name in ('keep_vote', 'court_present'):
            values = np.asarray(getattr(self, field_name), dtype=np.bool_)
            object.__setattr__(self, field_name, np.ascontiguousarray(values).copy())


class NoAcceptedCourtError(ValueError):
    """No scene passed the person vote; ``result`` keeps its records and votes."""

    def __init__(self, result: CourtEvidenceResult, message: str) -> None:
        super().__init__(message)
        self.result = result


def _copy_court_info(court_info: dict[str, object]) -> dict[str, object]:
    """Copy the dictionary and its mutable array values."""
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in court_info.items()
    }


def read_detector_scenes(
    result: Mapping[str, object], *, frame_count: int, native_size: tuple[float, float],
) -> tuple[DetectorScene, ...]:
    """Check one detector video result against its source video and type its scene rows.

    The schema is checked first, so an old or foreign file names its schema. The
    rows must tile ``[0, frame_count)`` in order, and only ``court`` rows carry
    corners.

    :param native_size: the source video's (width, height) in pixels.
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


def _as_ref_corners(corners: np.ndarray, source_resolution: tuple[float, float]) -> np.ndarray:
    """Convert TL/TR/BR/BL corners to 1280x720 reference pixels."""
    corners = np.asarray(corners, dtype=float)
    scale = np.asarray(HOMOGRAPHY_RESOLUTION, dtype=float) / np.asarray(source_resolution, dtype=float)
    return corners * scale


def detected_court_info(corners_refpx: np.ndarray) -> dict[str, object]:
    """Build an annotator court dictionary from a TL/TR/BR/BL reference quad."""
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


def build_net_band(
    court_info: dict[str, object], resolution: tuple[float, float],
) -> tuple[float, float]:
    """Project the one-metre centre net band into the pose-resolution y axis."""
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


def build_keep_vote(
    bboxes: np.ndarray,
    scores: np.ndarray,
    ndet: np.ndarray,
    resolution: tuple[float, float],
    scene_intervals: Sequence[tuple[int, int]],
    provisional_court_info: Sequence[dict[str, object] | None],
) -> np.ndarray:
    """Return the raw frame vote for exactly two in-margin people."""
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
    """Mark frames in scenes whose majority vote identifies exactly two in-court players."""
    court_present = np.zeros(len(keep_vote), dtype=np.bool_)
    for scene_index, (start_frame, end_frame) in enumerate(scene_intervals):
        court_present[start_frame:end_frame] = bool(scene_valid[scene_index])
    return court_present


def corner_error_band_from_corners(
    corners_refpx: np.ndarray, court_info: dict, err_px: float,
) -> float:
    """Return the median metre displacement from sixteen corner perturbations.

    ``corners_refpx`` uses the TL, TR, BR, BL order at
    :data:`HOMOGRAPHY_RESOLUTION`. The supplied ``court_info`` owns the active
    parent homography and court boundaries, so this helper is independent of
    ShuttleSet's homography table.
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


def _scene_fraction(keep_vote: np.ndarray, interval: tuple[int, int]) -> float:
    start_frame, end_frame = interval
    return float(keep_vote[start_frame:end_frame].mean())


def build_court_detector_evidence(
    case_id: str,
    video_id: int | str,
    resolution: tuple[float, float],
    native_size: tuple[float, float],
    scenes: Sequence[DetectorScene],
    bboxes: np.ndarray,
    scores: np.ndarray,
    ndet: np.ndarray,
    *,
    ref_err_px: float = 3.5,
) -> CourtEvidenceResult:
    """Accept scene-specific detected courts using their player-occupancy votes.

    A ``court`` scene is accepted when at least half its frames contain exactly two
    people inside the court margin. Accepted scenes retain their own corner rows.
    Rejected scenes have ``court_present`` False and keep their rejection status.

    :param resolution: width and height of the pose video, in pixels.
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
            case_id=case_id,
            parent=DETECTED_PARENT,
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

    # Static callers retain a representative court. Scene-aware callers use the rows.
    representative = max(accepted_indices, key=lambda index: intervals[index][1] - intervals[index][0])
    representative_corners = corners_refpx[representative]
    active_info = detected_court_info(representative_corners)
    inputs = CourtInputs(
        court_info=active_info,

        net_band=build_net_band(active_info, resolution),
        resolution=tuple(map(float, resolution)),
        homography_rows=build_scene_rows(
            video_id, [intervals[index] for index in accepted_indices],
            [corners_refpx[index] for index in accepted_indices], resolution,
        ),
        landing_error_band_m=corner_error_band_from_corners(
            representative_corners, active_info, ref_err_px,
        ),
        active_corners_refpx=representative_corners,
    )
    return CourtEvidenceResult(inputs, tuple(records), keep_vote, court_present)
