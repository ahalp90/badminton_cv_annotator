"""Judge proposed contacts directly against labelled rallies."""

import numpy as np

from annotator.rallies.build_contact_proposals import RallyOptions
from annotator.training.sequences import (
    SequenceTrainingVideo,
    whole_rally_correctness,
)
from shared.frame_timing import frames_at_fps


def option_targets_at_tolerances(
    video: SequenceTrainingVideo, pool: RallyOptions, tolerances_at_30_fps: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    """Judge each proposed contact list at each matching tolerance.

    Use the proposed bounds here. Inference extends bounds after selection.
    """
    fps = video.context.fps
    targets = tuple(np.empty(len(pool.sequences), dtype=np.int8) for _ in tolerances_at_30_fps)
    tolerance_frames = [frames_at_fps(tolerance, fps) for tolerance in tolerances_at_30_fps]
    margin = max(tolerance_frames, default=0)
    for rows in pool.span_rows.values():
        options = tuple(pool.sequences[row] for row in rows)
        start = min(sequence.start_frame for sequence in options) - margin
        end = max(sequence.end_frame for sequence in options) + margin
        # All options for this span use the same nearby labels. Retain each
        # complete labelled rally so partial and merged proposals remain wrong.
        nearby_rallies = tuple(
            rally for rally in video.rallies if any(start <= frame < end for frame in rally.frames)
        )
        for row in rows:
            for values, tolerance in zip(targets, tolerances_at_30_fps, strict=True):
                values[row] = whole_rally_correctness(
                    pool.sequences[row], nearby_rallies, fps, tolerance_at_30_fps=tolerance,
                )
    return targets


def option_targets(
    video: SequenceTrainingVideo, pool: RallyOptions, tolerance_at_30_fps: int = 10,
) -> np.ndarray:
    """Judge every proposal at one frame-rate-scaled matching tolerance."""
    return option_targets_at_tolerances(video, pool, (tolerance_at_30_fps,))[0]


def preferred_targets(pool: RallyOptions, wide: np.ndarray, preferred: np.ndarray) -> np.ndarray:
    """Use the tighter matching tolerance if at least one judged option still passes.

    Otherwise keep the results from the wider tolerance. Options that cannot be
    judged at the wider tolerance remain unjudgeable.
    """
    if wide.shape != (len(pool.sequences),) or preferred.shape != wide.shape:
        raise ValueError('preferred targets must align with every option row')
    targets = wide.copy()
    for rows in pool.span_rows.values():
        included = rows[wide[rows] >= 0]
        if np.any(preferred[included] == 1):
            targets[included] = preferred[included]
    return targets
