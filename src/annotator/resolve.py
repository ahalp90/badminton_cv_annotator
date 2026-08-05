"""Composition module for a base preset and caller-provided video fps.

Probing fps is the caller's business (``probe_fps``); this function never
defaults it.
"""
from __future__ import annotations

from .config import BaseAnnotatorConfig, ResolvedAnnotatorConfig
from .fps_constants import FPS_CONSTANT_FIELD_NAMES, ScalingKind, scale_for_fps
from .rally_segmentation import scale_thresholds


_OVERRIDABLE_BASE30_ROWS = FPS_CONSTANT_FIELD_NAMES | frozenset({'contact_impulse_multiple'})


def resolve(base: BaseAnnotatorConfig, fps: float) -> ResolvedAnnotatorConfig:
    """Resolve one preset for a probed fps; probing fps is the caller's business."""
    overrides = base.overrides_base30
    if overrides is not None:
        unknown_rows = sorted(set(overrides) - _OVERRIDABLE_BASE30_ROWS)
        if unknown_rows:
            raise ValueError(f'unknown base-30 override rows: {unknown_rows}')
    constants = scale_for_fps(fps, dict(overrides) if overrides is not None else None)
    thresholds = scale_thresholds(base.thresholds, fps, constants=constants, overrides_base30=overrides)
    def frame_count(value: float | None) -> int | None:
        return None if value is None else int(ScalingKind.FRAME_COUNT.scale(value, fps))
    return ResolvedAnnotatorConfig(
        fps=fps,
        constants=constants,
        thresholds=thresholds,
        dead_mask_mode=base.dead_mask_mode,
        smoothing_mode=base.smoothing_mode,
        span_open=base.span_open,
        gap_state_demotion_bound=frame_count(base.gap_state_demotion_bound),
        reentry_guard_variant=base.reentry_guard_variant,
        reentry_guard_buffer=base.reentry_guard_buffer,
        quiet_start_window=frame_count(base.quiet_start_window),
        rejected_grades=base.rejected_grades,
    )
