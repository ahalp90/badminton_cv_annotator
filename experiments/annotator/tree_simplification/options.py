"""Fixed, label-independent proposals for the simplified rally-layer pilot.

Targets finish each proposal with neighbouring spans kept at their initial choices.
This is a training approximation: final evaluation must jointly finish all selected
spans because their boundary padding can interact.
"""

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from annotator.contacts.model import NEARBY_CONTACT_DISTANCE_AT_30_FPS
from annotator.sequence.boundaries import BOUNDARY_PADDING_AT_30_FPS, extend_boundaries
from annotator.sequence.confidence import summarise_gap_maxima
from annotator.sequence.contacts import ContactEvent, ContactSequence, scale_frames
from annotator.sequence.edits import apply_choices
from annotator.sequence.features import (
    PHYSICAL_FEATURE_NAMES,
    SIDE_FEATURE_NAMES,
    SUMMARY_FEATURE_NAMES,
    physical_or_nan,
    sequence_summary,
    side_agreement,
)
from annotator.training.sequences import SequenceTrainingVideo
from annotator.training.workflow import whole_rally_correctness

FEATURE_NAMES = (
    *(f'before__{name}' for name in SUMMARY_FEATURE_NAMES),
    *(f'after__{name}' for name in SUMMARY_FEATURE_NAMES),
    *(f'before__{name}' for name in SIDE_FEATURE_NAMES),
    *(f'after__{name}' for name in SIDE_FEATURE_NAMES),
    'added_count', 'deleted_count',
    *(f'{kind}_probability_{stat}' for kind in ('added', 'deleted') for stat in ('min', 'max', 'mean')),
    *(f'{kind}__{name}' for kind in ('original_first', 'new_first', 'strongest_addition', 'weakest_deletion')
      for name in PHYSICAL_FEATURE_NAMES),
    'unused_candidate_count', 'unused_probability_min', 'unused_probability_max', 'unused_probability_mean',
    'usable_candidate_count', 'excluded_candidate_count',
    'gaps_with_candidates', 'positive_gap_count', 'gaps_without_candidates', 'longest_gap_seconds',
    'last_gap_has_candidate',
    'gap_probability_max', 'gap_probability_second', 'gap_probability_mean', 'gap_probability_sum',
    'strongest_unused_left_gap_seconds', 'strongest_unused_right_gap_seconds',
    *(f'strongest_unused__{name}' for name in PHYSICAL_FEATURE_NAMES),
)
feature_names = FEATURE_NAMES


@dataclass(frozen=True)
class ExpandedPool:
    sequences: tuple[ContactSequence, ...]
    matrix: np.ndarray
    targets: np.ndarray
    span_rows: dict[int, np.ndarray]
    original: SequenceTrainingVideo


def sequence_key(sequence: ContactSequence) -> tuple[int, int, tuple[int, ...]]:
    return sequence.start_frame, sequence.end_frame, tuple(event.frame for event in sequence.events)


def probability_summary(events: tuple[ContactEvent, ...]) -> tuple[float, float, float]:
    if not events:
        return np.nan, np.nan, np.nan
    values = [event.probability for event in events]
    return min(values), max(values), float(np.mean(values))


def span_candidates(video: SequenceTrainingVideo, span_id: int) -> tuple[ContactEvent, ...]:
    """Recover the existing serve/later shortlist without running fitted models."""
    candidates = {event.frame: event for event in video.pool.later_candidates.get(span_id, ())}
    for option in video.pool.options:
        if option.span_id != span_id or option.edit.serve_frame is None:
            continue
        frame = option.edit.serve_frame
        event = next(event for event in option.sequence.events if event.frame == frame)
        candidates[frame] = event
    return tuple(candidates[frame] for frame in sorted(candidates))


def toggle_proposals(
    before: ContactSequence, candidates: tuple[ContactEvent, ...], stream: tuple[ContactEvent, ...],
    previous_end: int, distance: int, max_edits: int, deletion_candidates: int,
) -> list[ContactSequence]:
    original_frames = {event.frame for event in before.events}
    additions = tuple(event for event in candidates if event.frame not in original_frames)
    deletions = []
    if before.events and deletion_candidates:
        deletions = [before.events[0]]
        weakest = sorted(before.events[1:], key=lambda event: (event.probability, event.frame))
        deletions.extend(weakest[:deletion_candidates - 1])
    proposals = []
    for added_count in range(min(max_edits, len(additions)) + 1):
        for added in combinations(additions, added_count):
            start = min((before.start_frame, *(event.frame for event in added)))
            if start < previous_end:
                continue
            # An earlier serve absorbs the full stream before removing explicit deletions.
            base = {event.frame: event for event in stream if start <= event.frame < before.end_frame}
            base.update((event.frame, event) for event in added)
            for deleted_count in range(min(max_edits - added_count, len(deletions)) + 1):
                for deleted in combinations(deletions, deleted_count):
                    deleted_frames = {event.frame for event in deleted}
                    events = tuple(base[frame] for frame in sorted(base) if frame not in deleted_frames)
                    if any(not start <= event.frame < before.end_frame for event in added):
                        continue
                    if any(abs(candidate.frame - event.frame) <= distance
                           for candidate in added for event in events if event.frame != candidate.frame):
                        continue
                    proposals.append(replace(before, start_frame=start, events=events))
    return proposals


def unused_evidence(
    after: ContactSequence, candidates: tuple[ContactEvent, ...], fps: float,
) -> tuple[tuple[float, ...], int | None]:
    frames = np.asarray([event.frame for event in after.events], dtype=np.int64)
    chosen = set(frames.tolist())
    unused = tuple(event for event in candidates if event.frame not in chosen)
    durations = np.diff([after.start_frame, *frames.tolist(), after.end_frame])
    positive_gaps = set(np.flatnonzero(durations > 0).tolist())
    distance = scale_frames(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    by_gap: dict[int, list[float]] = {}
    usable = []
    for event in unused:
        inside = after.start_frame <= event.frame < after.end_frame
        spaced = not np.any(np.abs(frames - event.frame) <= distance)
        if inside and spaced:
            gap = int(np.searchsorted(frames, event.frame))
            by_gap.setdefault(gap, []).append(event.probability)
            usable.append(event)
    present = positive_gaps & set(by_gap)
    strongest = min(unused, key=lambda event: (-event.probability, event.frame)) if unused else None
    left_gap = right_gap = np.nan
    if strongest is not None and after.start_frame <= strongest.frame < after.end_frame:
        position = int(np.searchsorted(frames, strongest.frame))
        left = int(frames[position - 1]) if position else after.start_frame
        right = int(frames[position]) if position < len(frames) else after.end_frame
        left_gap, right_gap = (strongest.frame - left) / fps, (right - strongest.frame) / fps
    values = (
        float(len(unused)), *probability_summary(unused), float(len(usable)), float(len(unused) - len(usable)),
        float(len(present)), float(len(positive_gaps)), float(len(positive_gaps - present)),
        float(max(durations) / fps), float(len(durations) - 1 in present),
        *summarise_gap_maxima(list(by_gap.values())), left_gap, right_gap,
    )
    return values, None if strongest is None else strongest.frame


def option_features(
    before: ContactSequence, after: ContactSequence, candidates: tuple[ContactEvent, ...],
    video: SequenceTrainingVideo, before_summary: tuple[float, ...], before_sides: tuple[float, ...],
) -> np.ndarray:
    before_frames = {event.frame for event in before.events}
    after_frames = {event.frame for event in after.events}
    added = tuple(event for event in after.events if event.frame not in before_frames)
    deleted = tuple(event for event in before.events if event.frame not in after_frames)
    strongest = min(added, key=lambda event: (-event.probability, event.frame)) if added else None
    weakest = min(deleted, key=lambda event: (event.probability, event.frame)) if deleted else None
    physical_frames = (
        before.events[0].frame if before.events else None,
        after.events[0].frame if after.events else None,
        None if strongest is None else strongest.frame, None if weakest is None else weakest.frame,
    )
    raw, unused_frame = unused_evidence(after, candidates, video.fps)
    return np.concatenate((
        before_summary, sequence_summary(after, video.fps), before_sides, side_agreement(after),
        (len(added), len(deleted)), probability_summary(added), probability_summary(deleted),
        *(physical_or_nan(video.pool.physical, frame) for frame in physical_frames),
        raw, physical_or_nan(video.pool.physical, unused_frame),
    ))


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


def expand_pool(video: SequenceTrainingVideo, max_edits: int = 3, deletion_candidates: int = 6) -> ExpandedPool:
    """Keep all legacy choices and add fixed-budget combinations of contact toggles.

    Features and proposals use only detector evidence. Targets use full-rally
    correctness after canonical boundary expansion; that helper applies side
    alternation itself. Neighbours stay at their initial choices for targets.
    The proposal budget is independent of the fitted layer count.
    """
    if max_edits < 0 or deletion_candidates < 0:
        raise ValueError('edit and deletion budgets must be non-negative')
    sequences = []
    matrices = []
    targets = []
    span_rows = {}
    stream_frames = np.asarray([event.frame for event in video.events], dtype=np.int64)
    legacy_by_span: dict[int, list[ContactSequence]] = {}
    for option in video.pool.options:
        legacy_by_span.setdefault(option.span_id, []).append(option.sequence)
    distance = scale_frames(NEARBY_CONTACT_DISTANCE_AT_30_FPS, video.fps)
    previous_end = -1
    for position, before in enumerate(video.initial):
        candidates = span_candidates(video, before.span_id)
        earliest = min((before.start_frame, *(event.frame for event in candidates)))
        first, last = np.searchsorted(stream_frames, (earliest, before.end_frame))
        proposals = toggle_proposals(
            before, candidates, video.events[first:last], previous_end, distance, max_edits, deletion_candidates,
        )
        unique = {sequence_key(before): before}
        for sequence in (*legacy_by_span.get(before.span_id, ()), *proposals):
            unique.setdefault(sequence_key(sequence), sequence)
        original_frames = {event.frame for event in before.events}
        ordered = sorted(unique.values(), key=lambda sequence: (
            sequence_key(sequence) != sequence_key(before),
            len(original_frames ^ {event.frame for event in sequence.events}),
            tuple(event.frame for event in sequence.events), sequence.start_frame, sequence.end_frame,
        ))
        evidence = {event.frame: event for sequence in ordered for event in sequence.events}
        evidence.update((event.frame, event) for event in candidates)
        evidence_candidates = tuple(evidence[frame] for frame in sorted(evidence))
        before_summary, before_sides = sequence_summary(before, video.fps), side_agreement(before)
        padding = scale_frames(BOUNDARY_PADDING_AT_30_FPS, video.fps)
        label_start = min(sequence.start_frame for sequence in ordered) - padding
        label_end = max(sequence.end_frame for sequence in ordered) + padding + 1
        rallies = tuple(
            rally for rally in video.rallies if any(label_start <= frame < label_end for frame in rally.frames)
        )
        start_row = len(sequences)
        for after in ordered:
            sequences.append(after)
            matrices.append(option_features(before, after, evidence_candidates, video, before_summary, before_sides))
            finished = finish_in_initial_context(video, position, after, stream_frames)
            targets.append(whole_rally_correctness(finished, rallies, video.fps))
        span_rows[before.span_id] = np.arange(start_row, len(sequences), dtype=np.int64)
        previous_end = before.end_frame
    matrix = np.asarray(matrices, dtype=np.float32).reshape(len(sequences), len(FEATURE_NAMES))
    return ExpandedPool(tuple(sequences), matrix, np.asarray(targets, dtype=np.int8), span_rows, video)
