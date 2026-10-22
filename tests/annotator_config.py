"""Explicit configuration for standalone burst-opening trajectory fixtures."""

from annotator.config import BaseAnnotatorConfig, ResolvedAnnotatorConfig, resolve
from annotator.types import SmoothingMode


def burst_config(fps: float = 25.0) -> ResolvedAnnotatorConfig:
    return resolve(BaseAnnotatorConfig(
        smoothing_mode=SmoothingMode.ZERO_FILL,
        span_open=None,
        gap_state_demotion_bound=None,
        reentry_guard_variant=None,
        reentry_guard_buffer=None,
    ), fps)
