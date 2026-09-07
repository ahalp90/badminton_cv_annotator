"""Frame lookup for the court geometry of each accepted camera scene."""

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from annotator.court_evidence import build_net_band, detected_court_info
from annotator.point_winner import corner_error_band_from_corners
from shared.court import HOMOGRAPHY_RESOLUTION


@dataclass(frozen=True)
class SceneCourt:
    start_frame: int
    end_frame: int
    court_info: dict[str, object]
    net_band: tuple[float, float]
    landing_error_band_m: float


def scene_ref_corners(
    row: Mapping[str, object], resolution: tuple[float, float]
) -> np.ndarray:
    """Scale one scene row's native corner quad to the homography reference frame."""
    width, height = map(float, resolution)
    if not math.isfinite(width) or not math.isfinite(height) or min(width, height) <= 0:
        raise ValueError("resolution must contain positive finite values")
    native = np.array(
        [
            [row["upleft_x"], row["upleft_y"]],
            [row["upright_x"], row["upright_y"]],
            [row["downright_x"], row["downright_y"]],
            [row["downleft_x"], row["downleft_y"]],
        ],
        dtype=float,
    )
    if not np.isfinite(native).all():
        raise ValueError("scene corners must be finite")
    return native * np.array(
        [HOMOGRAPHY_RESOLUTION[0] / width, HOMOGRAPHY_RESOLUTION[1] / height]
    )


def build_scene_courts(
    rows: Iterable[Mapping[str, object]], resolution: tuple[float, float], ref_err_px: float = 3.5,
) -> tuple[SceneCourt, ...]:
    """Derive projections and net bands once from the saved scene corner rows."""
    scenes = []
    for row in rows:
        corners = scene_ref_corners(row, resolution)
        info = detected_court_info(corners)
        scenes.append(SceneCourt(
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            court_info=info,
            net_band=build_net_band(info, resolution),
            landing_error_band_m=corner_error_band_from_corners(corners, info, ref_err_px),
        ))
    scenes.sort(key=lambda scene: scene.start_frame)
    return tuple(scenes)


def court_at_frame(scenes: Sequence[SceneCourt], frame: int) -> SceneCourt:
    """Find the accepted scene covering a frame; gaps have no usable geometry."""
    index = bisect_right(scenes, frame, key=lambda scene: scene.start_frame) - 1
    if index < 0 or frame >= scenes[index].end_frame:
        raise ValueError(f"no accepted court geometry at frame {frame}")
    return scenes[index]
