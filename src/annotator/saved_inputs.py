"""Load the saved vision inputs used by annotation and training."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_builder.shuttle_evidence import GUARD_CODES_FILENAME
from dataset_builder.vision import (
    TRACK_FILENAME,
    CourtVision,
    PoseArrays,
    load_court_vision,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
)
from shared.shuttle_track_validation import validate_shuttle_track
from shared.video_metadata import VideoMetadata


@dataclass(frozen=True)
class SavedVideoInputs:
    """Vision arrays and metadata shared by annotation and training."""

    metadata: VideoMetadata
    track: np.ndarray
    shuttle_grades: np.ndarray
    pose: PoseArrays
    court: CourtVision


def load_saved_video_inputs(run_dir: Path, video_id: str) -> SavedVideoInputs:
    """Read metadata, shuttle, pose and courts from the extraction stages."""
    stages = run_dir / 'stages'
    metadata = VideoMetadata.from_dict(load_json_gz(stages / 'metadata' / video_id / 'video_metadata.json.gz'))
    track = load_npy_xz(stages / 'shuttle' / video_id / TRACK_FILENAME)
    validate_shuttle_track(track, metadata.frame_count)
    shuttle_grades = load_npy_xz(stages / 'shuttle' / video_id / GUARD_CODES_FILENAME)
    pose = load_pose_arrays(stages / 'pose' / video_id, metadata.frame_count)
    court = load_court_vision(
        stages / 'court' / video_id, video_id=video_id, frame_count=metadata.frame_count,
        resolution=(float(metadata.width), float(metadata.height)),
    )
    return SavedVideoInputs(metadata, track, shuttle_grades, pose, court)
