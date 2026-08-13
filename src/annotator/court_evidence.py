"""Court evidence and parent-specific geometry for the annotator chain.

Here, a parent is one alternative court-evidence producer profile for a run,
not process lineage. The adapter keeps the static ShuttleSet homography and
detected CourtKeyNet consensus parents on the same operational interface. The
two parents share only their raw scene intervals; scene geometry and person
votes are built from the active parent.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from courtkeynet.court_corners import ConsensusRepair, CourtQuad, FallbackDiagnostics, pick_scene_corners
from shared.court import HOMOGRAPHY_RESOLUTION, get_corner_camera, get_court_info

from .composition_mask import detect_cuts
from .config import COMPOSITION_CONTENT_THRESHOLD
from .fps_constants import scale_for_fps
from .point_winner import (
    COURT_LENGTH_M,
    SHUTTLESET_TO_COURTKEYNET_CORNER_ORDER,
    corner_error_band_from_corners,
    project_pixels_to_court,
)


SCENE_ROW_COLUMNS = (
    'video_id', 'start_frame', 'end_frame',
    'upleft_x', 'upleft_y', 'upright_x', 'upright_y',
    'downleft_x', 'downleft_y', 'downright_x', 'downright_y',
)
POSE_SCENE_COLUMNS_BY_COURTKEYNET_CORNER = (
    ('upleft', 0),
    ('upright', 1),
    ('downleft', 3),
    ('downright', 2),
)
DETECTOR_RESOLUTION = (512.0, 288.0)
COURT_SCENE_SAMPLE_LIMIT = 10
PERSON_COURT_MARGIN = 0.10
SCENE_VALID_MIN_FRACTION = 0.5
UNIT_COURT_CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class CourtInputs:
    """All court-dependent inputs needed by one ``run_video`` parent.

    Arrays and tables are copied at construction because the dataclass's frozen
    fields prevent rebinding, but do not make mutable NumPy or pandas values
    immutable themselves.
    """

    court_info: dict[str, object]
    gate_court_info: dict[str, dict[str, object]]
    net_band: tuple[float, float]
    resolution: tuple[float, float]
    gate_resolution_table: pd.DataFrame
    homography_rows: pd.DataFrame
    landing_error_band_m: float
    active_corners_refpx: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, 'court_info', _copy_court_info(self.court_info))
        object.__setattr__(
            self,
            'gate_court_info',
            {str(video_id): _copy_court_info(info) for video_id, info in self.gate_court_info.items()},
        )
        gate_resolution_table = self.gate_resolution_table.copy(deep=True)
        gate_resolution_table.index = gate_resolution_table.index.astype(str)
        object.__setattr__(self, 'gate_resolution_table', gate_resolution_table)
        object.__setattr__(self, 'homography_rows', self.homography_rows.copy(deep=True))
        object.__setattr__(self, 'active_corners_refpx', np.asarray(self.active_corners_refpx).copy())


@dataclass(frozen=True)
class SceneEvidence:
    """Raw detector evidence for one half-open scene interval."""

    start_frame: int
    end_frame: int
    sampled_frame_indices: tuple[int, ...]
    quad: CourtQuad | None


@dataclass(frozen=True)
class CourtSceneRecord:
    """Typed evidence for one raw scene, ready for the court-scenes writer.

    :param parent: court-evidence producer profile used for this scene.
    """

    video_id: int | str
    case_id: str
    parent: str
    scene_index: int
    start_frame: int
    end_frame: int
    sampled_frame_indices: tuple[int, ...]
    raw_corners_px: np.ndarray | None
    raw_source: str | None
    raw_peaks: np.ndarray | None
    raw_corner_source: tuple[str, str, str, str] | None
    fallback_diagnostics: FallbackDiagnostics | None
    exactly_two_count: int
    exactly_two_fraction: float
    scene_valid: bool
    consensus_distance_px: float | None
    consensus_flag: bool | None
    active_corners_native_px: np.ndarray | None

    def __post_init__(self) -> None:
        if self.raw_corners_px is not None:
            object.__setattr__(self, 'raw_corners_px', np.asarray(self.raw_corners_px).copy())
        if self.raw_peaks is not None:
            object.__setattr__(self, 'raw_peaks', np.asarray(self.raw_peaks).copy())
        if self.active_corners_native_px is not None:
            object.__setattr__(
                self, 'active_corners_native_px', np.asarray(self.active_corners_native_px).copy(),
            )


@dataclass(frozen=True)
class CourtEvidenceResult:
    """One parent build, including operational arrays and writer evidence."""

    inputs: CourtInputs | None
    scene_records: tuple[CourtSceneRecord, ...]
    keep_vote: np.ndarray
    court_present: np.ndarray
    consensus: ConsensusRepair | None

    def __post_init__(self) -> None:
        for field_name in ('keep_vote', 'court_present'):
            values = np.asarray(getattr(self, field_name), dtype=np.bool_)
            object.__setattr__(self, field_name, np.ascontiguousarray(values).copy())


class CourtConsensusError(ValueError):
    """Detected consensus failed after raw evidence and votes were derived."""

    def __init__(self, result: CourtEvidenceResult, original_error: ValueError) -> None:
        super().__init__(str(original_error))
        self.result = result
        self.original_error = original_error


def _copy_court_info(court_info: dict[str, object]) -> dict[str, object]:
    """Copy the dictionary and its mutable array values."""
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in court_info.items()
    }


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
    """Run the pinned cut detector once and return complete half-open intervals."""
    cut_frames = detect_cuts(
        video_path,
        expected_frames=n_frames,
        threshold=COMPOSITION_CONTENT_THRESHOLD,
        min_scene_len=scale_for_fps(fps).composition_min_scene_len,
    )
    cut_frames = np.asarray(cut_frames, dtype=int)
    boundaries = np.concatenate((np.array([0], dtype=int), cut_frames, np.array([n_frames], dtype=int)))
    intervals = [(int(start), int(end)) for start, end in zip(boundaries[:-1], boundaries[1:])]
    if any(start < 0 or end > n_frames or end <= start for start, end in intervals):
        raise ValueError('detected cuts do not form non-empty in-range intervals')
    if intervals[0][0] != 0 or intervals[-1][1] != n_frames:
        raise ValueError('detected cuts do not cover the video')
    if any(end != next_start for (_, end), (next_start, _) in zip(intervals, intervals[1:])):
        raise ValueError('detected cuts contain a gap or overlap')
    return intervals


def scene_sample_indices(start_frame: int, end_frame: int) -> list[int]:
    """Return deterministic centred-bin samples for one non-empty scene."""
    scene_length = end_frame - start_frame
    if scene_length <= 0:
        raise ValueError('scene interval must be non-empty')
    sample_count = min(COURT_SCENE_SAMPLE_LIMIT, scene_length)
    return [
        start_frame + ((2 * sample_index + 1) * scene_length // (2 * sample_count))
        for sample_index in range(sample_count)
    ]


def detect_scene_evidence(
    video_path: Path,
    raw_cuts: Sequence[tuple[int, int]] | pd.DataFrame,
    detector: object,
) -> list[SceneEvidence]:
    """Sample each raw scene and preserve CourtKeyNet provenance."""
    intervals = _normalise_intervals(raw_cuts)
    samples_by_scene = [scene_sample_indices(start, end) for start, end in intervals]
    corner_floor = float(getattr(detector, 'corner_min_peak_conf'))
    evidence: list[SceneEvidence] = []
    capture = cv2.VideoCapture(str(video_path))
    frame_index = 0
    try:
        if not capture.isOpened():
            raise ValueError(f'could not open video {video_path}')
        for (start_frame, end_frame), sample_indices in zip(intervals, samples_by_scene):
            sampled_frames: list[np.ndarray] = []
            for sample_index in sample_indices:
                while frame_index <= sample_index:
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f'video ended before sampled frame {sample_index}')
                    if frame_index == sample_index:
                        sampled_frames.append(frame)
                    frame_index += 1
            detections = detector.detect_batch(sampled_frames)
            quad = pick_scene_corners(
                sampled_frames,
                detections,
                corner_min_peak_conf=corner_floor,
            )
            evidence.append(SceneEvidence(start_frame, end_frame, tuple(sample_indices), quad))
    finally:
        capture.release()
    return evidence


def _as_ref_corners(corners: np.ndarray, source_resolution: tuple[float, float]) -> np.ndarray:
    """Convert TL/TR/BR/BL corners to 1280x720 reference pixels."""
    corners = np.asarray(corners, dtype=float)
    scale = np.asarray(HOMOGRAPHY_RESOLUTION, dtype=float) / np.asarray(source_resolution, dtype=float)
    return corners * scale


def _as_native_corners(corners_refpx: np.ndarray) -> np.ndarray:
    """Convert reference-space corners to native 512x288 video pixels."""
    scale = np.asarray(DETECTOR_RESOLUTION, dtype=float) / np.asarray(HOMOGRAPHY_RESOLUTION, dtype=float)
    return np.asarray(corners_refpx, dtype=float) * scale


def _static_corners_refpx(homography_row: pd.Series) -> np.ndarray:
    """Return static row corners in the CourtKeyNet TL/TR/BR/BL order."""
    source_order = get_corner_camera(homography_row).T
    return source_order[list(SHUTTLESET_TO_COURTKEYNET_CORNER_ORDER)].copy()


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


def _gate_resolution_table(
    video_id: int | str,
    resolution: tuple[float, float],
    table: pd.DataFrame | None,
) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame(
            {'width': [float(resolution[0])], 'height': [float(resolution[1])]},
            index=pd.Index([str(video_id)], dtype=object),
        )
    copied = table.copy(deep=True)
    copied.index = copied.index.astype(str)
    return copied


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
    for prefix, corner_index in POSE_SCENE_COLUMNS_BY_COURTKEYNET_CORNER:
        row[f'{prefix}_x'] = float(pose_corners[corner_index, 0])
        row[f'{prefix}_y'] = float(pose_corners[corner_index, 1])
    return row


def build_scene_rows(
    video_id: int | str,
    intervals: Sequence[tuple[int, int]],
    corners_refpx: Sequence[np.ndarray],
    resolution: tuple[float, float],
) -> pd.DataFrame:
    """Build the pose-resolution scene table expected by sticky and replay."""
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
    """Expand each scene's exactly-two majority to the sole court vector."""
    court_present = np.zeros(len(keep_vote), dtype=np.bool_)
    for scene_index, (start_frame, end_frame) in enumerate(scene_intervals):
        court_present[start_frame:end_frame] = bool(scene_valid[scene_index])
    return court_present


def build_static_court_inputs(
    video_id: int | str,
    homo_df: pd.DataFrame,
    resolution: tuple[float, float],
    raw_cuts: Sequence[tuple[int, int]] | pd.DataFrame,
    *,
    gate_resolution_table: pd.DataFrame | None = None,
    ref_err_px: float = 3.5,
) -> CourtInputs:
    """Build static inputs from the existing ShuttleSet homography row."""
    intervals = _normalise_intervals(raw_cuts)
    court_info = get_court_info(homo_df, video_id)
    active_corners = _static_corners_refpx(homo_df.loc[video_id])
    homography_rows = build_scene_rows(video_id, intervals, [active_corners] * len(intervals), resolution)
    return CourtInputs(
        court_info=court_info,
        gate_court_info={str(video_id): court_info},
        net_band=build_net_band(court_info, resolution),
        resolution=tuple(map(float, resolution)),
        gate_resolution_table=_gate_resolution_table(video_id, resolution, gate_resolution_table),
        homography_rows=homography_rows,
        landing_error_band_m=corner_error_band_from_corners(active_corners, court_info, ref_err_px),
        active_corners_refpx=active_corners,
    )


def _validate_scene_evidence(
    raw_cuts: Sequence[tuple[int, int]] | pd.DataFrame,
    scene_evidence: Sequence[SceneEvidence],
) -> list[SceneEvidence]:
    """Check that typed detector evidence still matches the raw scene order."""
    intervals = _normalise_intervals(raw_cuts)
    if len(scene_evidence) != len(intervals):
        raise ValueError('scene evidence count must match raw cuts')
    for interval, scene in zip(intervals, scene_evidence):
        if not isinstance(scene, SceneEvidence):
            raise TypeError('scene evidence must contain SceneEvidence records')
        if (scene.start_frame, scene.end_frame) != interval:
            raise ValueError('scene evidence intervals must match raw cuts in scene order')
    return list(scene_evidence)


def _scene_fraction(keep_vote: np.ndarray, interval: tuple[int, int]) -> float:
    start_frame, end_frame = interval
    return float(keep_vote[start_frame:end_frame].mean())


def _scene_record(
    video_id: int | str,
    case_id: str,
    parent: str,
    scene_index: int,
    scene: SceneEvidence,
    keep_vote: np.ndarray,
    scene_valid: bool,
    *,
    active_corners_native_px: np.ndarray | None,
    consensus_distance_px: float | None,
    consensus_flag: bool | None,
    static_corners_px: np.ndarray | None = None,
) -> CourtSceneRecord:
    """Build one immutable writer record without reprojecting later."""
    fraction = _scene_fraction(keep_vote, (scene.start_frame, scene.end_frame))
    count = int(keep_vote[scene.start_frame:scene.end_frame].sum())
    if static_corners_px is not None:
        return CourtSceneRecord(
            video_id=video_id,
            case_id=case_id,
            parent=parent,
            scene_index=scene_index,
            start_frame=scene.start_frame,
            end_frame=scene.end_frame,
            sampled_frame_indices=(),
            raw_corners_px=static_corners_px,
            raw_source=None,
            raw_peaks=None,
            raw_corner_source=None,
            fallback_diagnostics=None,
            exactly_two_count=count,
            exactly_two_fraction=fraction,
            scene_valid=scene_valid,
            consensus_distance_px=None,
            consensus_flag=None,
            active_corners_native_px=static_corners_px,
        )

    quad = scene.quad
    return CourtSceneRecord(
        video_id=video_id,
        case_id=case_id,
        parent=parent,
        scene_index=scene_index,
        start_frame=scene.start_frame,
        end_frame=scene.end_frame,
        sampled_frame_indices=scene.sampled_frame_indices,
        raw_corners_px=None if quad is None else quad.corners_px,
        raw_source=None if quad is None else quad.source,
        raw_peaks=None if quad is None else quad.peak,
        raw_corner_source=None if quad is None else quad.corner_source,
        fallback_diagnostics=None if quad is None else quad.diagnostics,
        exactly_two_count=count,
        exactly_two_fraction=fraction,
        scene_valid=scene_valid,
        consensus_distance_px=consensus_distance_px,
        consensus_flag=consensus_flag,
        active_corners_native_px=active_corners_native_px,
    )


def build_static_court_evidence(
    case_id: str,
    parent: str,
    video_id: int | str,
    homo_df: pd.DataFrame,
    resolution: tuple[float, float],
    raw_cuts: Sequence[tuple[int, int]] | pd.DataFrame,
    bboxes: np.ndarray,
    scores: np.ndarray,
    ndet: np.ndarray,
    *,
    gate_resolution_table: pd.DataFrame | None = None,
    ref_err_px: float = 3.5,
) -> CourtEvidenceResult:
    """Build static inputs, votes and writer records in one parent pass."""
    intervals = _normalise_intervals(raw_cuts)
    inputs = build_static_court_inputs(
        video_id, homo_df, resolution, intervals,
        gate_resolution_table=gate_resolution_table,
        ref_err_px=ref_err_px,
    )
    provisional_infos = [inputs.court_info] * len(intervals)
    keep_vote = build_keep_vote(
        bboxes, scores, ndet, resolution, intervals, provisional_infos,
    )
    scene_valid = [
        _scene_fraction(keep_vote, interval) >= SCENE_VALID_MIN_FRACTION
        for interval in intervals
    ]
    court_present = build_court_present(keep_vote, intervals, scene_valid)
    static_corners_refpx = inputs.active_corners_refpx
    static_corners_px = _as_native_corners(static_corners_refpx)
    records = tuple(
        _scene_record(
            video_id,
            case_id, parent,
            scene_index,
            SceneEvidence(start, end, (), None),
            keep_vote,
            valid,
            active_corners_native_px=static_corners_px,
            consensus_distance_px=None,
            consensus_flag=None,
            static_corners_px=static_corners_px,
        )
        for scene_index, ((start, end), valid) in enumerate(zip(intervals, scene_valid))
    )
    return CourtEvidenceResult(inputs, records, keep_vote, court_present, None)


def build_detected_court_evidence(
    case_id: str,
    parent: str,
    video_id: int | str,
    resolution: tuple[float, float],
    raw_cuts: Sequence[tuple[int, int]] | pd.DataFrame,
    scene_evidence: Sequence[SceneEvidence],
    bboxes: np.ndarray,
    scores: np.ndarray,
    ndet: np.ndarray,
    *,
    detector_resolution: tuple[float, float] = DETECTOR_RESOLUTION,
    gate_resolution_table: pd.DataFrame | None = None,
    ref_err_px: float = 3.5,
) -> CourtEvidenceResult:
    """Build detected geometry, votes, consensus and records in one pass."""
    from courtkeynet.court_corners import consensus_repair

    evidence = _validate_scene_evidence(raw_cuts, scene_evidence)
    intervals = [(scene.start_frame, scene.end_frame) for scene in evidence]
    native_corners = [
        None if scene.quad is None else np.asarray(scene.quad.corners_px, dtype=float)
        for scene in evidence
    ]
    provisional_infos = [
        None if corners is None else detected_court_info(_as_ref_corners(corners, detector_resolution))
        for corners in native_corners
    ]
    keep_vote = build_keep_vote(
        bboxes, scores, ndet, resolution, intervals, provisional_infos,
    )
    scene_valid = [
        corners is not None
        and _scene_fraction(keep_vote, interval) >= SCENE_VALID_MIN_FRACTION
        for corners, interval in zip(native_corners, intervals)
    ]
    court_present = build_court_present(keep_vote, intervals, scene_valid)
    raw_records = tuple(
        _scene_record(
            video_id,
            case_id,
            parent,
            scene_index,
            scene,
            keep_vote,
            scene_valid[scene_index],
            active_corners_native_px=None,
            consensus_distance_px=None,
            consensus_flag=None,
        )
        for scene_index, scene in enumerate(evidence)
    )
    raw_result = CourtEvidenceResult(None, raw_records, keep_vote, court_present, None)
    accepted_scene_indices = [
        index for index, valid in enumerate(scene_valid) if valid
    ]
    if not accepted_scene_indices:
        original_error = ValueError(
            'detected court consensus requires at least one accepted scene quad',
        )
        raise CourtConsensusError(raw_result, original_error) from original_error

    accepted_quads = np.stack([native_corners[index] for index in accepted_scene_indices])
    try:
        consensus = consensus_repair(accepted_quads)
    except ValueError as original_error:
        raise CourtConsensusError(raw_result, original_error) from original_error
    consensus_corners = _as_ref_corners(consensus.consensus_quad, detector_resolution)
    repaired_corners = np.asarray(consensus.repaired_quads, dtype=float)
    repaired_corners_refpx = _as_ref_corners(repaired_corners, detector_resolution)
    active_info = detected_court_info(consensus_corners)
    active_rows = [
        _scene_row(video_id, intervals[index], repaired_corners_refpx[accepted_position], resolution)
        for accepted_position, index in enumerate(accepted_scene_indices)
    ]
    homography_rows = pd.DataFrame(active_rows, columns=SCENE_ROW_COLUMNS)
    inputs = CourtInputs(
        court_info=active_info,
        gate_court_info={str(video_id): active_info},
        net_band=build_net_band(active_info, resolution),
        resolution=tuple(map(float, resolution)),
        gate_resolution_table=_gate_resolution_table(video_id, resolution, gate_resolution_table),
        homography_rows=homography_rows,
        landing_error_band_m=corner_error_band_from_corners(
            consensus_corners, active_info, ref_err_px,
        ),
        active_corners_refpx=consensus_corners,
    )
    accepted_positions = {
        scene_index: accepted_position
        for accepted_position, scene_index in enumerate(accepted_scene_indices)
    }
    records = []
    for scene_index, scene in enumerate(evidence):
        accepted_position = accepted_positions.get(scene_index)
        records.append(
            _scene_record(
                video_id,
                case_id,
                parent,
                scene_index,
                scene,
                keep_vote,
                scene_valid[scene_index],
                active_corners_native_px=(
                    repaired_corners[accepted_position]
                    if accepted_position is not None else None
                ),
                consensus_distance_px=(
                    float(consensus.distances_px[accepted_position])
                    if accepted_position is not None else None
                ),
                consensus_flag=(
                    bool(consensus.flagged[accepted_position])
                    if accepted_position is not None else None
                ),
            )
        )
    return CourtEvidenceResult(inputs, tuple(records), keep_vote, court_present, consensus)
