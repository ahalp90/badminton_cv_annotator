"""Compose rally spans and contact candidates; expose the batch CLI."""

import numpy as np

from .config import ResolvedAnnotatorConfig
from .masks.replay import filter_exclusion_mask
from .rally.contacts import assemble_contacts
from .rally.serve import ServeStartOptions
from .rally.spans import find_rally_spans
from .rally.trajectory import apply_replay_mask
from .types import ContactCandidate, SpanOpen


def segment_video(
    track: np.ndarray, positions: np.ndarray | None = None, *,
    resolved: ResolvedAnnotatorConfig,
    serve_start: ServeStartOptions | None = None,
    exclusion_mask: np.ndarray | None = None,
    sticky_distances: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], list[ContactCandidate]]:
    """Find spans, then detect, gate and suppress their contact candidates.

    :param track: Whole-video normalised shuttle track, shaped (frames, 3).
    :param resolved: Numeric thresholds and strategies resolved once for this video.
    :param serve_start: Optional serve evidence built from the original track.
    :param exclusion_mask: Optional dead-time mask applied before segmentation.
    :param sticky_distances: Optional shuttle-to-wrist distances in body heights.
    :param spans: Supplied spans bypass span finding but retain contact extraction.
    :return: Half-open rally spans and every raw contact candidate with gate verdicts.
    """
    if serve_start is not None and resolved.span_open is SpanOpen.REGION_START:
        raise ValueError(
            'serve_start with span_open=REGION_START is contradictory: REGION_START drops the '
            'qualifying gate serve_start refines. Use span_open=BACK_FILL under serve gating '
            '(the two forms coincide there).'
        )
    if serve_start is not None and serve_start.close is not None and resolved.span_open is SpanOpen.BACK_FILL:
        raise ValueError(
            'serve_start.close (split placement) with span_open=BACK_FILL is contradictory: '
            'BACK_FILL emits one span per qualifying region at region_start, so there is nothing '
            'to split. Drop the split close, or use it without BACK_FILL.'
        )
    if sticky_distances is not None and sticky_distances.shape != (len(track),):
        raise ValueError('sticky_distances must have shape (len(track),)')
    if exclusion_mask is not None:
        track = apply_replay_mask(track, exclusion_mask)
    if spans is None:
        spans = find_rally_spans(track, resolved, serve_start)
    return spans, assemble_contacts(
        track, positions, spans, resolved.thresholds, sticky_distances,
        smoothing_mode=resolved.smoothing_mode,
    )


def segment_shuttle_track(
    track: np.ndarray,
    resolved: ResolvedAnnotatorConfig,
    *,
    positions: np.ndarray | None = None,
    raw_exclusion_mask: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], list[ContactCandidate]]:
    """Segment a shuttle track without player tracking or court evidence.

    The batch CLI uses this operation with ``span_open=None`` by default.
    Its optional mask excludes dead time without adding court-based exclusions.
    """
    if raw_exclusion_mask is None:
        raw_exclusion_mask = np.zeros(len(track), dtype=bool)
    exclusion_mask = filter_exclusion_mask(
        raw_exclusion_mask, len(track), resolved.thresholds.replay_mask_min_frames,
    )
    return segment_video(track, positions=positions, resolved=resolved, exclusion_mask=exclusion_mask)


def main() -> None:
    """Run the rally-segmentation batch CLI."""
    from .rally.cli import main as cli_main

    cli_main()


if __name__ == '__main__':
    main()
