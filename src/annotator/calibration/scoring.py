"""Pure scoring functions for rally and contact detection."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from annotator.fps_constants import ScalingKind
from annotator.types import ContactCandidate

CANONICAL_CONTACT_TOLERANCE_BASE30 = 5
CONTACT_TOLERANCES_BASE30 = (1, 2, CANONICAL_CONTACT_TOLERANCE_BASE30, 10)

# Replay masking is not applied in sset_01, so replay spans (which
# carry no GT strokes) count as spurious by design. Surfaced in the output so a
# reader does not misread the spurious count as pure false positives.
SPURIOUS_NOTE = (
    'replay masking not applied in sset_01; replays inflate the '
    'spurious-span count by design'
)


class RallyBoundary(StrEnum):
    """How a GT rally's strokes land relative to the detected spans."""

    COVERED = 'covered'  # every stroke inside one and the same span
    SPLIT = 'split'      # strokes across 2+ spans, or partly outside any span
    MISSED = 'missed'    # no stroke inside any span


class GtRally(NamedTuple):
    """One ground-truth rally: its identity and the frames of its strokes.

    :param set_id: ShuttleSet set label, e.g. ``'set1'``.
    :param rally: rally number within the set (restarts per set).
    :param stroke_frames: source-video frames of the strokes, ascending.
    """

    set_id: str
    rally: int
    stroke_frames: tuple[int, ...]

    @property
    def extent(self) -> tuple[int, int]:
        """Inclusive ``(first_stroke_frame, last_stroke_frame)`` of the rally."""
        return self.stroke_frames[0], self.stroke_frames[-1]

    @property
    def n_strokes(self) -> int:
        return len(self.stroke_frames)


def safe_f1(precision: float, recall: float) -> float:
    """Return the harmonic mean, including zero when both inputs are zero."""
    denominator = precision + recall
    return 0.0 if denominator == 0 else 2 * precision * recall / denominator


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def load_gt_rallies(shots_master: pd.DataFrame, vid: int) -> list[GtRally]:
    """Filter shots_master to one video and group strokes into GT rallies.

    :param shots_master: the full shots_master frame (columns per
        `scripts.build_shots_master`; needs ``vid``, ``set_id``, ``rally``,
        ``frame_num``).
    :param vid: ShuttleSet video id to score.
    :return: one `GtRally` per ``(set_id, rally)``, ordered by ``(set_id, rally)``.
    """
    for_vid = shots_master[shots_master['vid'] == vid]
    if for_vid.empty:
        raise ValueError(f'no strokes in shots_master for vid={vid}')

    rallies: list[GtRally] = []
    # groupby sorts on the key by default, so rally order is deterministic.
    for (set_id, rally), group in for_vid.groupby(['set_id', 'rally']):
        frames = tuple(sorted(int(frame) for frame in group['frame_num']))
        rallies.append(GtRally(set_id=str(set_id), rally=int(rally), stroke_frames=frames))
    return rallies


# ---------------------------------------------------------------------------
# Interval helpers (spans are half-open [start, end); extents are inclusive)
# ---------------------------------------------------------------------------
def _spans_containing(frame: int, spans: Sequence[tuple[int, int]]) -> list[int]:
    """Indices of the spans that contain ``frame`` (start <= frame < end)."""
    return [idx for idx, (start, end) in enumerate(spans) if start <= frame < end]


def _spans_overlapping_extent(
    extent: tuple[int, int], spans: Sequence[tuple[int, int]]
) -> list[int]:
    """Indices of the spans whose frame range overlaps the inclusive ``extent``."""
    first, last = extent
    return [idx for idx, (start, end) in enumerate(spans) if start <= last and first < end]


def merged_span_indices(
    spans: Sequence[tuple[int, int]], gt_rallies: Sequence[GtRally]
) -> set[int]:
    """Spans that fully contain 2+ GT rally extents (a merge of rallies).

    A span ``[start, end)`` fully contains an inclusive extent ``[first, last]``
    when ``start <= first`` and ``last < end``.

    :return: set of span indices that swallow two or more whole rally extents.
    """
    extents = [rally.extent for rally in gt_rallies]
    merged: set[int] = set()
    for idx, (start, end) in enumerate(spans):
        contained = sum(1 for first, last in extents if start <= first and last < end)
        if contained >= 2:
            merged.add(idx)
    return merged


def classify_rally_boundary(
    stroke_frames: Sequence[int], spans: Sequence[tuple[int, int]]
) -> tuple[RallyBoundary, int | None]:
    """Classify how one rally's strokes land in the detected spans.

    :param stroke_frames: the rally's stroke frames.
    :param spans: detected rally spans.
    :return: ``(category, mapped_span_index)``. The mapped span index is the
        single span all strokes fall in when COVERED, else None.
    """
    containing_per_stroke = [_spans_containing(frame, spans) for frame in stroke_frames]
    span_indices = {idx for containing in containing_per_stroke for idx in containing}

    if not span_indices:
        return RallyBoundary.MISSED, None

    every_stroke_in_exactly_one = all(len(containing) == 1 for containing in containing_per_stroke)
    all_in_one_span = len(span_indices) == 1
    if every_stroke_in_exactly_one and all_in_one_span:
        return RallyBoundary.COVERED, next(iter(span_indices))
    return RallyBoundary.SPLIT, None


def classify_all(
    spans: Sequence[tuple[int, int]], gt_rallies: Sequence[GtRally]
) -> list[tuple[RallyBoundary, int | None]]:
    """Per-rally boundary classification, index-aligned to ``gt_rallies``."""
    return [classify_rally_boundary(rally.stroke_frames, spans) for rally in gt_rallies]


# ---------------------------------------------------------------------------
# Boundary metrics
# ---------------------------------------------------------------------------
def _offset_stats(offsets: Sequence[int]) -> dict[str, float | int] | None:
    """Mean/median/p10/p90 of an offset sample, or None when the sample is empty."""
    if not offsets:
        return None
    values = np.asarray(offsets, dtype=float)
    p10, p90 = np.percentile(values, [10, 90])
    return {
        'n': len(offsets),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p10': float(p10),
        'p90': float(p90),
    }


def score_boundaries(
    spans: Sequence[tuple[int, int]], gt_rallies: Sequence[GtRally]
) -> dict:
    """Boundary-side metrics: coverage taxonomy, merge/spurious counts, alignment.

    Self-contained: re-derives the per-rally classification and merged-span set
    (both cheap over the small rally/span counts) so the sweep runner and tests
    can call it in isolation.

    :param spans: detected rally spans.
    :param gt_rallies: ground-truth rallies for one video.
    :return: dict of boundary metrics (see keys inline).
    """
    classifications = classify_all(spans, gt_rallies)
    merged = merged_span_indices(spans, gt_rallies)

    covered = sum(1 for category, _ in classifications if category is RallyBoundary.COVERED)
    split = sum(1 for category, _ in classifications if category is RallyBoundary.SPLIT)
    missed = sum(1 for category, _ in classifications if category is RallyBoundary.MISSED)

    # Spurious spans hold no GT stroke at all. One flat array of every stroke
    # frame lets each span test membership with a vectorised range check.
    all_stroke_frames = np.array(
        [frame for rally in gt_rallies for frame in rally.stroke_frames], dtype=int
    )
    spurious = 0
    for start, end in spans:
        in_span = (all_stroke_frames >= start) & (all_stroke_frames < end)
        if not in_span.any():
            spurious += 1

    # Start/end alignment is only meaningful for covered rallies, which map to a
    # single span. Start offset is expected negative (span opens before the
    # first stroke); end offset is structurally positive (span end is rest onset).
    start_offsets: list[int] = []
    end_offsets: list[int] = []
    for rally, (category, span_idx) in zip(gt_rallies, classifications):
        if category is not RallyBoundary.COVERED or span_idx is None:
            continue
        span_start, span_end = spans[span_idx]
        first, last = rally.extent
        start_offsets.append(span_start - first)
        end_offsets.append(span_end - last)

    n_rallies = len(gt_rallies)
    return {
        'n_gt_rallies': n_rallies,
        'covered': covered,
        'covered_fraction': covered / n_rallies if n_rallies else None,
        'split': split,
        'missed': missed,
        'merged_spans': len(merged),
        'spurious_spans': spurious,
        'spurious_spans_note': SPURIOUS_NOTE,
        'start_alignment': _offset_stats(start_offsets),
        'end_alignment': _offset_stats(end_offsets),
    }


# ---------------------------------------------------------------------------
# Contact matching
# ---------------------------------------------------------------------------
def greedy_match(
    gt_frames: Sequence[int], candidate_frames: Sequence[int], tolerance: int
) -> list[tuple[int, int]]:
    """One-to-one match of candidates to GT strokes, closest pairs first.

    Every within-tolerance (gt, candidate) pair is ranked by absolute frame
    distance and claimed greedily: the closest pair binds first, and neither its
    GT stroke nor its candidate can be reused. Ties (equal distance) break by
    lower GT index, then lower candidate index, so the result is deterministic.

    :param gt_frames: GT stroke frames for one rally.
    :param candidate_frames: candidate contact frames for the same rally.
    :param tolerance: max absolute frame distance for a pair to be eligible.
    :return: matched ``(gt_index, candidate_index)`` pairs.
    """
    ranked: list[tuple[int, int, int]] = []  # (distance, gt_index, candidate_index)
    for gt_index, gt_frame in enumerate(gt_frames):
        for candidate_index, candidate_frame in enumerate(candidate_frames):
            distance = abs(gt_frame - candidate_frame)
            if distance <= tolerance:
                ranked.append((distance, gt_index, candidate_index))
    ranked.sort()

    matched: list[tuple[int, int]] = []
    claimed_gt: set[int] = set()
    claimed_candidate: set[int] = set()
    for _distance, gt_index, candidate_index in ranked:
        if gt_index in claimed_gt or candidate_index in claimed_candidate:
            continue
        claimed_gt.add(gt_index)
        claimed_candidate.add(candidate_index)
        matched.append((gt_index, candidate_index))
    return matched


def _scale_base30_frames(base_frames: float, fps: float) -> int:
    """Scale a base-30 frame count with the annotator's half-up rule."""
    return int(ScalingKind.FRAME_COUNT.scale(base_frames, fps))


def strict_contact_rows(
    spans: Sequence[tuple[int, int]],
    filtered_contacts: Sequence[ContactCandidate],
    gt_rallies: Sequence[GtRally],
    fps: float,
    tolerances_base30: Sequence[int] = (5, 10),
) -> list[dict[str, int | str | None]]:
    """Build strict GT contact rows for covered rallies only.

    A covered GT rally receives candidates from its one associated predicted span.
    Split and missed rallies therefore have unmatched GT rows but no candidates.
    ``greedy_match`` supplies the matched-row order and its deterministic ties.
    """
    classifications = classify_all(spans, gt_rallies)
    candidates_by_span: dict[int, list[int]] = defaultdict(list)
    for contact in filtered_contacts:
        candidates_by_span[contact.rally_id].append(contact.contact_frame)

    tolerance_frames = [
        (base30, _scale_base30_frames(base30, fps))
        for base30 in tolerances_base30
    ]
    rows: list[dict[str, int | str | None]] = []
    for gt_index, (rally, (boundary, span_idx)) in enumerate(zip(gt_rallies, classifications)):
        gt_frames = list(rally.stroke_frames)
        candidate_frames = candidates_by_span.get(span_idx, []) if boundary is RallyBoundary.COVERED else []
        for tolerance_base30, tolerance in tolerance_frames:
            matches = greedy_match(gt_frames, candidate_frames, tolerance)
            matched_gt = {gt_idx for gt_idx, _candidate_idx in matches}
            matched_candidates = {candidate_idx for _gt_idx, candidate_idx in matches}
            for gt_idx, candidate_idx in matches:
                gt_frame = gt_frames[gt_idx]
                candidate_frame = candidate_frames[candidate_idx]
                rows.append({
                    'rally_id': gt_index,
                    'tolerance_base30': tolerance_base30,
                    'tolerance_frames': tolerance,
                    'row_kind': 'matched',
                    'gt_frame': gt_frame,
                    'candidate_frame': candidate_frame,
                    'offset_frames': candidate_frame - gt_frame,
                })
            for gt_idx, gt_frame in enumerate(gt_frames):
                if gt_idx not in matched_gt:
                    rows.append({
                        'rally_id': gt_index,
                        'tolerance_base30': tolerance_base30,
                        'tolerance_frames': tolerance,
                        'row_kind': 'unmatched_gt',
                        'gt_frame': gt_frame,
                        'candidate_frame': None,
                        'offset_frames': None,
                    })
            for candidate_idx in sorted(
                (idx for idx in range(len(candidate_frames)) if idx not in matched_candidates),
                key=lambda idx: candidate_frames[idx],
            ):
                candidate_frame = candidate_frames[candidate_idx]
                rows.append({
                    'rally_id': gt_index,
                    'tolerance_base30': tolerance_base30,
                    'tolerance_frames': tolerance,
                    'row_kind': 'unmatched_candidate',
                    'gt_frame': None,
                    'candidate_frame': candidate_frame,
                    'offset_frames': None,
                })
    return rows


def wide_edge_contact_rows(
    gt_rallies: Sequence[GtRally],
    filtered_contacts: Sequence[ContactCandidate],
    fps: float,
    n_frames: int,
) -> list[dict[str, int | str | None]]:
    """Build the diagnostic first/last-GT-contact report with wide edge windows.

    Windows are clipped to the video and split at the deterministic midpoint of
    adjacent targets. Matching is global and one-to-one, even though final windows
    are normally disjoint.
    """
    half_width = _scale_base30_frames(90, fps)
    targets: list[tuple[int, int, str, int, int]] = []
    # (GT frame, source rally order, edge, initial start, initial end)
    for source_order, rally in enumerate(gt_rallies):
        edge_frames = [('first', rally.stroke_frames[0])]
        if len(rally.stroke_frames) > 1:
            edge_frames.append(('last', rally.stroke_frames[-1]))
        for edge, gt_frame in edge_frames:
            start = max(0, gt_frame - half_width)
            end = min(n_frames, gt_frame + half_width + 1)
            targets.append((gt_frame, source_order, edge, start, end))
    targets.sort(key=lambda target: (target[0], target[1], 0 if target[2] == 'first' else 1))

    initial_windows = [(target[3], target[4]) for target in targets]
    windows = [[start, end] for start, end in initial_windows]
    for target_idx in range(len(targets) - 1):
        earlier = targets[target_idx]
        later = targets[target_idx + 1]
        if initial_windows[target_idx][1] <= initial_windows[target_idx + 1][0]:
            continue
        boundary = math.floor((earlier[0] + later[0]) / 2) + 1
        windows[target_idx][1] = min(windows[target_idx][1], boundary)
        windows[target_idx + 1][0] = max(windows[target_idx + 1][0], boundary)

    candidate_frames = sorted({
        contact.contact_frame for contact in filtered_contacts
    })
    ranked_pairs: list[tuple[int, int, int]] = []
    for target_idx, (target, (window_start, window_end)) in enumerate(zip(targets, windows)):
        for candidate_idx, candidate_frame in enumerate(candidate_frames):
            if window_start <= candidate_frame < window_end:
                ranked_pairs.append((abs(candidate_frame - target[0]), target_idx, candidate_idx))
    ranked_pairs.sort(key=lambda pair: (pair[0], pair[1], candidate_frames[pair[2]]))

    claimed_targets: set[int] = set()
    claimed_candidates: set[int] = set()
    matches: dict[int, int] = {}
    for _distance, target_idx, candidate_idx in ranked_pairs:
        if target_idx in claimed_targets or candidate_idx in claimed_candidates:
            continue
        claimed_targets.add(target_idx)
        claimed_candidates.add(candidate_idx)
        matches[target_idx] = candidate_idx

    rows: list[dict[str, int | str | None]] = []
    for target_idx, (target, (window_start, window_end)) in enumerate(zip(targets, windows)):
        _gt_frame, source_order, edge, _start, _end = target
        common = {
            'window_id': target_idx,
            'rally_id': source_order,
            'edge': edge,
            'window_start': window_start,
            'window_end': window_end,
        }
        if target_idx in matches:
            candidate_frame = candidate_frames[matches[target_idx]]
            rows.append({
                **common,
                'row_kind': 'matched',
                'gt_frame': target[0],
                'candidate_frame': candidate_frame,
                'offset_frames': candidate_frame - target[0],
            })
        else:
            rows.append({
                **common,
                'row_kind': 'unmatched_gt',
                'gt_frame': target[0],
                'candidate_frame': None,
                'offset_frames': None,
            })
        for candidate_idx, candidate_frame in enumerate(candidate_frames):
            if candidate_idx in claimed_candidates:
                continue
            if window_start <= candidate_frame < window_end:
                rows.append({
                    **common,
                    'row_kind': 'unmatched_candidate',
                    'gt_frame': None,
                    'candidate_frame': candidate_frame,
                    'offset_frames': None,
                })
    return rows


def _prf(matched: int, n_gt: int, n_candidates: int) -> dict:
    """Recall/precision/F1 from raw counts; None where a denominator is zero."""
    recall = matched / n_gt if n_gt else None
    precision = matched / n_candidates if n_candidates else None
    f1 = None if recall is None or precision is None else safe_f1(precision, recall)
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'gt': n_gt,
        'candidates': n_candidates,
        'matched': matched,
    }


def _tolerance_curve(
    rally_pairs: Sequence[tuple[Sequence[int], Sequence[int]]], tolerances: Sequence[int]
) -> dict[str, dict]:
    """Per-tolerance recall/precision/F1 aggregated over the given rally pairs.

    :param rally_pairs: one ``(gt_frames, candidate_frames)`` per rally.
    :param tolerances: frame tolerances to score at.
    :return: ``{str(tolerance): {recall, precision, f1, gt, candidates, matched}}``.
    """
    curve: dict[str, dict] = {}
    for tolerance in tolerances:
        matched = 0
        total_gt = 0
        total_candidates = 0
        for gt_frames, candidate_frames in rally_pairs:
            matched += len(greedy_match(gt_frames, candidate_frames, tolerance))
            total_gt += len(gt_frames)
            total_candidates += len(candidate_frames)
        curve[str(tolerance)] = _prf(matched, total_gt, total_candidates)
    return curve


def _raw_precision_curve(
    contacts: Sequence[tuple[int, int, bool | None, bool | None]],
    rally_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    tolerances: Sequence[int],
) -> dict[str, dict]:
    """Per-tolerance precision over PHYSICAL candidates, not merge-pooled ones.

    The ``tolerances`` curve above counts a candidate once per GT rally its span
    overlaps, so its precision denominator swells with merge structure (50,164 vs
    9,002 raw at the crown) and precision reads differently for two configs that
    detect the same contacts but merge rallies differently. This curve fixes both
    ends to the physical candidate set:

      - denominator: unique detected contact frames across the whole video, ALL of
        them, including contacts in spurious spans that no rally pools. Spans are
        disjoint in frame range, so distinct contacts carry distinct frames and the
        set size equals ``len(contacts)`` on real input.
      - numerator: unique candidate frames matched in ANY rally's greedy matching
        at that tolerance. A frame pooled into two rallies (its span overlaps both
        extents) and matched in either counts once, so a cross-rally double match
        can't inflate the numerator past the denominator.

    The greedy matching is re-run here rather than threaded out of
    ``_tolerance_curve``: keeping that function's return byte-identical matters
    (the sweep CSV's committed columns depend on it) more than the one extra pass,
    which is over the overall rally pairs only and cheap next to segmentation.

    :param contacts: detected contacts as
        ``(rally_id, contact_frame, proximity_ok, wrist_near)``. The full list
        ensures spurious-span candidates land in the denominator.
    :param rally_pairs: one ``(gt_frames, candidate_frames)`` per rally, the same
        pooled pairs ``_tolerance_curve`` scores.
    :param tolerances: frame tolerances to score at.
    :return: ``{str(tolerance): {precision_raw, matched, candidates}}``; precision_raw
        None only when no candidate was detected at all.
    """
    n_unique_candidates = len({contact_frame for _rally_id, contact_frame, *_ in contacts})
    curve: dict[str, dict] = {}
    for tolerance in tolerances:
        matched_frames: set[int] = set()
        for gt_frames, candidate_frames in rally_pairs:
            for _gt_index, candidate_index in greedy_match(gt_frames, candidate_frames, tolerance):
                matched_frames.add(candidate_frames[candidate_index])
        n_matched = len(matched_frames)
        curve[str(tolerance)] = {
            'precision_raw': n_matched / n_unique_candidates if n_unique_candidates else None,
            'matched': n_matched,
            'candidates': n_unique_candidates,
        }
    return curve


def _count_gate(passes: int, total: int) -> dict:
    """Pass/total/fraction for the per-rally count gate (candidates == strokes)."""
    return {
        'pass': passes,
        'total': total,
        'fraction': passes / total if total else None,
    }


def score_contacts(
    spans: Sequence[tuple[int, int]],
    contacts: Sequence[tuple[int, int, bool | None, bool | None]],
    gt_rallies: Sequence[GtRally],
    tolerances: Sequence[int] = CONTACT_TOLERANCES_BASE30,
) -> dict:
    """Contact-side metrics: count gate and per-tolerance credit, overall + per set.

    Candidates for a GT rally are the contacts from every span overlapping the
    rally extent. The count gate (candidates == strokes) is reported over covered
    rallies and, more strictly, over covered rallies whose span is not a merge of
    two rallies. Per-stroke credit uses greedy one-to-one matching per rally.
    ``precision_5`` is merge-pooled, counted over per-rally pooled copies, while
    ``precision_raw_5`` is per-candidate, counted over unique physical contact
    frames. The ``_raw_precision_curve`` docstring summarises the detail.

    :param spans: detected rally spans.
    :param contacts: detected contacts as
        ``(rally_id, contact_frame, proximity_ok, wrist_near)``. The two
        verdict fields are not used here.
    :param gt_rallies: ground-truth rallies for one video.
    :param tolerances: frame tolerances for the credit curve.
    :return: dict of contact metrics (see keys inline).
    """
    classifications = classify_all(spans, gt_rallies)
    merged = merged_span_indices(spans, gt_rallies)

    contacts_by_span: dict[int, list[int]] = defaultdict(list)
    for rally_id, contact_frame, *_ in contacts:
        contacts_by_span[rally_id].append(contact_frame)

    # Per rally: its candidate frames (from overlapping spans) and count-gate pass.
    rally_candidates: list[list[int]] = []
    count_gate_pass: list[bool] = []
    for rally in gt_rallies:
        overlapping = _spans_overlapping_extent(rally.extent, spans)
        candidates = [frame for idx in overlapping for frame in contacts_by_span.get(idx, [])]
        rally_candidates.append(candidates)
        count_gate_pass.append(len(candidates) == rally.n_strokes)

    covered_flags = [category is RallyBoundary.COVERED for category, _ in classifications]
    # "unmerged" tightens covered to rallies whose single span holds only them:
    # a merged span pools two rallies' contacts, so its count gate is unfair.
    unmerged_flags = [
        category is RallyBoundary.COVERED and span_idx not in merged
        for category, span_idx in classifications
    ]

    count_gate = {
        'covered': _count_gate(
            sum(1 for passed, covered in zip(count_gate_pass, covered_flags) if covered and passed),
            sum(covered_flags),
        ),
        'unmerged': _count_gate(
            sum(1 for passed, clean in zip(count_gate_pass, unmerged_flags) if clean and passed),
            sum(unmerged_flags),
        ),
    }

    all_pairs = [
        (rally.stroke_frames, candidates)
        for rally, candidates in zip(gt_rallies, rally_candidates)
    ]

    # Per-set breakdown: same metrics restricted to each set's rallies.
    per_set_indices: dict[str, list[int]] = defaultdict(list)
    for idx, rally in enumerate(gt_rallies):
        per_set_indices[rally.set_id].append(idx)
    per_set: dict[str, dict] = {}
    for set_id, indices in per_set_indices.items():
        set_pairs = [all_pairs[idx] for idx in indices]
        set_covered = sum(1 for idx in indices if covered_flags[idx])
        set_covered_pass = sum(1 for idx in indices if covered_flags[idx] and count_gate_pass[idx])
        per_set[set_id] = {
            'tolerances': _tolerance_curve(set_pairs, tolerances),
            'count_gate_covered': _count_gate(set_covered_pass, set_covered),
        }

    return {
        'count_gate': count_gate,
        'tolerances': _tolerance_curve(all_pairs, tolerances),
        'precision_raw': _raw_precision_curve(contacts, all_pairs, tolerances),
        'per_set': per_set,
    }
