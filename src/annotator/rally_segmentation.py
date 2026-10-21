"""Rally segmentation and contact detection.

See ``docs/scraper_pipeline/scraper_architecture.md`` for the pipeline context.

Trajectory rules operate on a whole-video TrackNetV3 shuttle track shaped
`(t, 3)` as `[x_norm, y_norm, visibility]`. Coordinates are already normalised
to video resolution, while visibility passes through. Speed everywhere below
is per-frame L2 displacement of `(x, y)` on frames where visibility is 1.

Three primitives (`compute_speed`, `true_runs`, `rolling_nanmedian`) are
re-exported from `annotator.types` because replay masking reuses them. Its slow-motion
signal is defined against the same per-frame speed, so re-deriving it there
would be a second source of truth. All per-frame arrays here share one
frame-index space `[0, t)`; that invariant is what lets rally spans, contacts
and masks line up downstream.

`segment_video` takes four off-by-default keyword options that each preserve
today's behaviour exactly when left at their default:
  - `thresholds`: a `RallySegmentationThresholds` preset used instead of the module globals;
    None reads the globals through the low-level opt-out path.
  - `serve_start`: `ServeStartOptions` gating rally openings on a serve-setup
    lookback (the shuttle sitting near a court-scale player through the last
    second before the burst).
  - `span_open`: a `SpanOpen` rule that changes where a span opens (region start
    vs the qualifying burst).
  - `exclusion_mask`: a `(t,)` bool dead-time mask applied at entry (via
    `apply_replay_mask`), freezing excluded frames to rest before speed.

The contact chain keeps raw impulse candidates, applies a body-unit wrist gate,
then applies greedy suppression. The three operations are separate helpers and
`segment_video` composes them in that order.

Run as `python -m annotator.rally_segmentation --shuttle-dir ...` with
PYTHONPATH=src.
"""
from collections.abc import Mapping

import numpy as np

from .config import (
    CONTACT_FRAMES_CSV,
    END_REST_FRAMES,
    PROXIMITY_MAX,
    RALLY_SPANS_CSV,
    REST_SPEED,
    REST_WINDOW,
    SMOOTH_WINDOW,
    START_MIN_FRAMES,
    START_SPEED,
    BaseAnnotatorConfig,
    RallySegmentationThresholds,
)
from .doubles_flag import read_whole_video_flags
from .fps_constants import FpsConstants, scale_for_fps
from .rally.cli import log
from .rally.cli import main as _cli_main
from .rally.contacts import (
    BODY_UNIT_WRIST_THRESHOLD,
    CONTACT_DEDUP_RADIUS_FRAMES,
    CONTACT_IMPULSE_MULTIPLE,
    CONTACT_SUPPRESSION_RADIUS_FRAMES,
    FLOOR_EPS,
    IMPULSE_FLOOR_HALF_WINDOW_FRAMES,
    assemble_contacts,
    contact_proximity_ok,
    detect_contact_flags,
    impulse_cell_candidates,
    rolling_floor,
    span_impulses,
    suppress_contact_flags,
    wrist_contact_near,
)
from .rally.evidence import (
    BODY_UNIT_HALF_WINDOW,
    CourtGeo,
    build_sticky_result,
    court_scale_slots,
    tracker_segments,
)
from .rally.serve import (
    PLAYER_PRESENT_MIN_FRAC,
    ServeSetupInputs,
    ServeStartClose,
    ServeStartMode,
    ServeStartOptions,
    build_serve_setup_inputs,
    series_drift,
    serve_setup_still,
)
from .rally.spans import (
    QUIET_START_REST_FRACTION,
    VISIBILITY_REST_FRAC,
    find_rally_spans,
)
from .rally.trajectory import apply_replay_mask
from .types import (
    ANKLE_L,
    ANKLE_R,
    WRIST_L,
    WRIST_R,
    ContactCandidate,
    ReentryGuardVariant,
    Slot,
    SmoothingMode,
    SpanOpen,
    StickyResult,
    compute_speed,
    rolling_nanmedian,
    true_runs,
)

# Re-exported for callers that import these names from here; see the module docstring.
__all__ = [
    "ANKLE_L",
    "ANKLE_R",
    "BODY_UNIT_HALF_WINDOW",
    "BODY_UNIT_WRIST_THRESHOLD",
    "CONTACT_DEDUP_RADIUS_FRAMES",
    "CONTACT_FRAMES_CSV",
    "CONTACT_IMPULSE_MULTIPLE",
    "CONTACT_SUPPRESSION_RADIUS_FRAMES",
    "END_REST_FRAMES",
    "FLOOR_EPS",
    "IMPULSE_FLOOR_HALF_WINDOW_FRAMES",
    "PLAYER_PRESENT_MIN_FRAC",
    "PROXIMITY_MAX",
    "QUIET_START_REST_FRACTION",
    "RALLY_SPANS_CSV",
    "REST_SPEED",
    "REST_WINDOW",
    "SMOOTH_WINDOW",
    "START_MIN_FRAMES",
    "START_SPEED",
    "VISIBILITY_REST_FRAC",
    "WRIST_L",
    "WRIST_R",
    "BaseAnnotatorConfig",
    "CourtGeo",
    "ServeSetupInputs",
    "ServeStartClose",
    "ServeStartMode",
    "ServeStartOptions",
    "Slot",
    "SpanOpen",
    "StickyResult",
    "apply_replay_mask",
    "build_serve_setup_inputs",
    "build_sticky_result",
    "compute_speed",
    "contact_proximity_ok",
    "court_scale_slots",
    "detect_contact_flags",
    "find_rally_spans",
    "impulse_cell_candidates",
    "log",
    "main",
    "read_whole_video_flags",
    "rolling_floor",
    "rolling_nanmedian",
    "scale_thresholds",
    "segment_video",
    "series_drift",
    "serve_setup_still",
    "span_impulses",
    "suppress_contact_flags",
    "tracker_segments",
    "true_runs",
    "wrist_contact_near",
]

def scale_thresholds(
    thresholds: RallySegmentationThresholds, fps: float, *,
    constants: FpsConstants | None = None, overrides_base30: Mapping[str, float] | None = None,
) -> RallySegmentationThresholds:
    """Replace a preset's fps-dependent fields from the base-30 table; the preset
    contributes only its non-fps fields. Returned fields are final.
    """
    values = scale_for_fps(fps) if constants is None else constants
    overrides = {} if overrides_base30 is None else overrides_base30
    return thresholds._replace(
        rest_speed=values.rest_speed, rest_window=values.rest_window,
        start_speed=values.start_speed, start_min_frames=values.start_min_frames,
        smooth_window=values.smooth_window, end_rest_frames=values.end_rest_frames,
        impulse_floor_half_window_frames=values.impulse_floor_half_window_frames,
        contact_dedup_radius_frames=values.contact_dedup_radius_frames,
        contact_suppression_radius_frames=values.contact_suppression_radius_frames,
        contact_impulse_multiple=overrides.get('contact_impulse_multiple', thresholds.contact_impulse_multiple),
        wrist_distance_body_heights=overrides.get('wrist_distance_body_heights', thresholds.wrist_distance_body_heights),
    )


def segment_video(
    track: np.ndarray, positions: np.ndarray | None = None, *,
    thresholds: RallySegmentationThresholds | None = None,
    serve_start: ServeStartOptions | None = None,
    span_open: SpanOpen | None = None,
    exclusion_mask: np.ndarray | None = None,
    sticky_distances: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
    suppression_radius: int | None = None,
    smoothing_mode: SmoothingMode = SmoothingMode.ZERO_FILL,
    constants: FpsConstants | None = None,
    gap_state_demotion_bound: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None,
    reentry_guard_buffer: float | None = None,
    quiet_start_window: int | None = None,
) -> tuple[list[tuple[int, int]], list[ContactCandidate]]:
    """Full rally-segmentation pass over one video's shuttle track.

    Every keyword option is off by default and each default preserves today's behaviour
    exactly. `thresholds=None` reads the module globals through the low-level opt-out path.

    :param track: `(t, 3)` whole-video track.
    :param positions: optional `(t, 2, 2)` court positions for the proximity guardrail.
    :param thresholds: a `RallySegmentationThresholds` preset used instead of the globals, or None.
    :param serve_start: `ServeStartOptions` gating rally openings on sticky serve-setup evidence,
        or None. Its setup inputs are built from the UNMASKED track by the caller (the committed
        measurement convention); serve-start was only ever measured with masking off, so combining
        it with `exclusion_mask` is unmeasured territory.
    :param span_open: a `SpanOpen` rule (REGION_START / BACK_FILL) changing where a span opens,
        or None (today's burst-open rule). `serve_start` with REGION_START raises ValueError, and
        `serve_start.close` (a split) with BACK_FILL raises too (BACK_FILL is one span per region).
    :param exclusion_mask: `(t,)` bool dead-time mask (True = dead), applied at entry via
        `apply_replay_mask` before speed is computed, or None.
    :param sticky_distances: optional `(t,)` body-unit shuttle-to-nearest-wrist gaps. NaN fails
        closed. Production callers supply the cached sticky distances.
    :param suppression_radius: optional contact suppression radius; None keeps the shipped 9-frame default.
    :param smoothing_mode: span coordinate smoothing policy; ZERO_FILL preserves
        the shipped rule and IGNORE_INVISIBLE drops invisible xy from each mean.
    :return: `(spans, contacts)` where spans is `[(start_frame, end_frame), ...]`
        (rally_id is the list index) and contacts is
        `ContactCandidate(rally_id, contact_frame, proximity_ok, wrist_near, suppressed)`.
        Every detected candidate is a row (the RAW set, kept for recall-first uses).
        `wrist_near` is the pure wrist-gate verdict and `suppressed` records a gate-passing
        candidate that lost the suppression-radius contest. Both are blank when no gate inputs
        are supplied, so every raw candidate stands.
    """
    if serve_start is not None and span_open is SpanOpen.REGION_START:
        raise ValueError(
            'serve_start with span_open=REGION_START is contradictory: REGION_START drops the '
            'qualifying gate serve_start refines. Use span_open=BACK_FILL under serve gating '
            '(the two forms coincide there).'
        )
    if serve_start is not None and serve_start.close is not None and span_open is SpanOpen.BACK_FILL:
        raise ValueError(
            'serve_start.close (split placement) with span_open=BACK_FILL is contradictory: '
            'BACK_FILL emits one span per qualifying region at region_start, so there is nothing '
            'to split. Drop the split close, or use it without BACK_FILL.'
        )
    # Argument validation precedes any span work so bad combinations fail loudly
    # here, never as a numpy error from deep inside span finding.
    if sticky_distances is not None and sticky_distances.shape != (len(track),):
        raise ValueError('sticky_distances must have shape (len(track),)')
    if exclusion_mask is not None:
        track = apply_replay_mask(track, exclusion_mask)

    if spans is None:
        spans = find_rally_spans(
            track, thresholds, serve_start, span_open, constants=constants,
            gap_state_demotion_bound=gap_state_demotion_bound,
            reentry_guard_variant=reentry_guard_variant, reentry_guard_buffer=reentry_guard_buffer,
            quiet_start_window=quiet_start_window,
        )

    return spans, assemble_contacts(
        track, positions, spans, thresholds, sticky_distances, suppression_radius,
        smoothing_mode=smoothing_mode,
    )


def main() -> None:
    """Run the rally-segmentation batch CLI."""
    _cli_main()


if __name__ == '__main__':
    main()
