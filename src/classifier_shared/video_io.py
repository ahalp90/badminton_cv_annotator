"""Video metadata used by the classifier pipelines."""

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    """Metadata for a video file."""
    path: Path
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def get_video_info(path: str | Path) -> VideoInfo:
    """Read video metadata without decoding any frames."""
    p = Path(path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {p}')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return VideoInfo(path=p, fps=fps, n_frames=n_frames, width=width, height=height)
