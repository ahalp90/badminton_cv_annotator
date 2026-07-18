"""Composition module for a base preset and caller-provided video fps.

Probing fps is the caller's business (``probe_fps``); this function never
defaults it.
"""
from __future__ import annotations

from .config import BaseAnnotatorConfig, ResolvedAnnotatorConfig
from .fps_constants import scale_for_fps
from .rally_segmentation import scale_thresholds


def resolve(base: BaseAnnotatorConfig, fps: float) -> ResolvedAnnotatorConfig:
    """Resolve one preset for a probed fps; probing fps is the caller's business."""
    constants = scale_for_fps(fps)
    thresholds = scale_thresholds(base.thresholds, fps)
    return ResolvedAnnotatorConfig(fps=fps, constants=constants, thresholds=thresholds)
