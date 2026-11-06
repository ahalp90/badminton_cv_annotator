"""Supported settings for synthetic trajectory tests."""

from annotator.config import (
    BaseAnnotatorConfig,
    FpsScaledAnnotatorConfig,
    scale_config_to_fps,
)
from annotator.types import SmoothingMode


def trajectory_config(fps: float = 25.0) -> FpsScaledAnnotatorConfig:
    return scale_config_to_fps(BaseAnnotatorConfig(
        smoothing_mode=SmoothingMode.INCLUDE_INVISIBLE,
    ), fps)
