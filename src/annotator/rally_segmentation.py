"""Compose rally spans and contact candidates; expose the batch CLI."""

import numpy as np

from .calc_helpers import true_runs
from .config import ResolvedAnnotatorConfig
from .masks.replay import filter_exclusion_mask
from .rally.contacts import assemble_contacts
from .rally.spans import find_rally_spans
from .types import ContactCandidate


def segment_video(
    track: np.ndarray, positions: np.ndarray | None = None, *,
    resolved: ResolvedAnnotatorConfig,
    exclusion_mask: np.ndarray | None = None,
    sticky_distances: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], list[ContactCandidate]]:
    """Find spans, then detect, gate and suppress their contact candidates.

    :param track: Whole-video normalised shuttle track, shaped (frames, 3).
    :param resolved: Numeric thresholds and strategies resolved once for this video.
    :param exclusion_mask: Optional dead-time mask applied before segmentation.
    :param sticky_distances: Optional shuttle-to-wrist distances in body heights.
    :param spans: Supplied spans bypass span finding but retain contact extraction.
    :return: Half-open rally spans and every raw contact candidate with gate verdicts.
    """
    if sticky_distances is not None and sticky_distances.shape != (len(track),):
        raise ValueError('sticky_distances must have shape (len(track),)')
    if exclusion_mask is not None:
        track = apply_replay_mask(track, exclusion_mask)
    if spans is None:
        spans = find_rally_spans(track, resolved)
    return spans, assemble_contacts(
        track, positions, spans, resolved.thresholds, sticky_distances,
        smoothing_mode=resolved.smoothing_mode,
    )


def apply_replay_mask(track: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Freeze excluded footage so segmentation reads it as rest.

    Return a copy with masked positions fixed to the preceding live frame and
    visibility set to one so NaN speed cannot reopen a rally. A mask starting at
    frame zero uses the first live frame after it. An entirely masked track has
    no anchor and is rejected.
    """
    if len(mask) != len(track):
        raise ValueError(f'mask length {len(mask)} != track length {len(track)}')
    if mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')

    frozen = track.copy()
    for start, end in true_runs(mask):
        anchor = start - 1 if start > 0 else end
        frozen[start:end, :2] = track[anchor, :2]
        frozen[start:end, 2] = 1
    return frozen


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
