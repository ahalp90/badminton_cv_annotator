"""Judge rally options using labelled training videos and initial neighbours."""

import numpy as np

from annotator.sequence.contacts import ContactSequence, apply_choices, scale_frames
from annotator.sequence.rally_model import BOUNDARY_PADDING_AT_30_FPS, extend_boundaries
from annotator.sequence.rally_options import RallyOptions, sequence_key
from annotator.training.sequences import SequenceTrainingVideo, whole_rally_correctness


def finish_in_initial_context(
    video: SequenceTrainingVideo, position: int, after: ContactSequence, stream_frames: np.ndarray,
) -> ContactSequence:
    """Apply canonical boundaries in a three-span window with initial neighbours."""
    before = video.initial[position]
    start_position = max(0, position - 1)
    window = list(video.initial[start_position:position + 2])
    local_position = position - start_position
    window[local_position] = after
    padding = scale_frames(BOUNDARY_PADDING_AT_30_FPS, video.fps)
    left = min(window[0].start_frame, after.start_frame) - padding
    right = max(window[-1].end_frame, after.end_frame) + padding + 1
    first, last = np.searchsorted(stream_frames, (left, right))
    local_events = video.events[first:last]
    _chosen, stream = apply_choices((before,), local_events, {before.span_id: after})
    finished = extend_boundaries(window, stream, video.fps, frame_count=video.frame_count)
    return finished[local_position]


def option_targets(video: SequenceTrainingVideo, pool: RallyOptions, tolerance_at_30_fps: int = 10) -> np.ndarray:
    targets = np.empty(len(pool.sequences), dtype=np.int8)
    stream_frames = np.asarray([event.frame for event in video.events], dtype=np.int64)
    padding = scale_frames(BOUNDARY_PADDING_AT_30_FPS, video.fps)
    for position, before in enumerate(video.initial):
        rows = pool.span_rows[before.span_id]
        ordered = tuple(pool.sequences[row] for row in rows)
        label_start = min(sequence.start_frame for sequence in ordered) - padding
        label_end = max(sequence.end_frame for sequence in ordered) + padding + 1
        rallies = tuple(
            rally for rally in video.rallies if any(label_start <= frame < label_end for frame in rally.frames)
        )
        for row in rows:
            finished = finish_in_initial_context(video, position, pool.sequences[row], stream_frames)
            targets[row] = whole_rally_correctness(
                finished, rallies, video.fps, tolerance_at_30_fps=tolerance_at_30_fps,
            )
    return targets


def five_frame_preferred_targets(pool: RallyOptions, wide: np.ndarray, five: np.ndarray) -> np.ndarray:
    """Prefer five-frame labels only when a judged span retains a positive option."""
    if wide.shape != (len(pool.sequences),) or five.shape != wide.shape:
        raise ValueError('five-frame targets must align with every option row')
    targets = wide.copy()
    for rows in pool.span_rows.values():
        included = rows[wide[rows] >= 0]
        if np.any(five[included] == 1):
            targets[included] = five[included]
    return targets


def original_option_rows(pool: RallyOptions) -> np.ndarray:
    """Locate the original options to anchor training weight across pool sizes."""
    original_keys: dict[int, set[tuple[int, int, tuple[int, ...]]]] = {}
    for sequence in pool.context.initial:
        original_keys.setdefault(sequence.span_id, set()).add(sequence_key(sequence))
    for sequence in pool.context.repairs.sequences:
        original_keys.setdefault(sequence.span_id, set()).add(sequence_key(sequence))
    return np.asarray([
        row for row, sequence in enumerate(pool.sequences)
        if sequence_key(sequence) in original_keys[sequence.span_id]
    ], dtype=np.int64)
