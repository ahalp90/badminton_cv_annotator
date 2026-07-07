"""Score stage-8 rally segmentation and contact detection against ShuttleSet GT.

Scores the one-video pilot (stage A of the stage-8 sweep plan, section 4) and is
imported by the later threshold-sweep runner. The core is pure functions over
in-memory inputs, matching what `scraper.stage8_rally_segmentation.segment_video`
returns:

  - spans:    ``list[(start_frame, end_frame)]`` half-open detected rally spans,
              rally_id is the list index
  - contacts: ``list[(rally_id, contact_frame, proximity_ok)]``

Ground truth is `shots_master.csv` filtered to one ``vid``; strokes group by
``(set_id, rally)`` into one GT rally whose extent is
``[min frame_num, max frame_num]``. All frames are source-video frames; the
pilot track starts at source frame 0, so no offset is applied.

The metrics core (`score_stage8`, `load_gt_rallies`, `score_boundaries`,
`score_contacts`, `greedy_match`, ...) is what the sweep runner imports. The CLI
is a thin wrapper that reads the stage-8 CSVs plus shots_master, prints a
summary, and writes the metrics dict as JSON.

Usage:
    python -m scripts.stage8_score \
        --rally-spans-csv data/scrape_output/rally_spans.csv \
        --contact-frames-csv data/scrape_output/contact_frames.csv \
        --vid 1 --out-json data/scrape_output/stage8_score.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHOTS_MASTER = (
    REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
)
DEFAULT_TOLERANCES = (1, 2, 5, 10)

# Stage 9 replay masking is not applied in the pilot, so replay spans (which
# carry no GT strokes) count as spurious by design. Surfaced in the output so a
# reader does not misread the spurious count as pure false positives.
SPURIOUS_NOTE = (
    'stage 9 replay masking not applied in the pilot; replays inflate the '
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


def _prf(matched: int, n_gt: int, n_candidates: int) -> dict:
    """Recall/precision/F1 from raw counts; None where a denominator is zero."""
    recall = matched / n_gt if n_gt else None
    precision = matched / n_candidates if n_candidates else None
    if recall is None or precision is None:
        f1 = None
    elif recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
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


def _count_gate(passes: int, total: int) -> dict:
    """Pass/total/fraction for the per-rally count gate (candidates == strokes)."""
    return {
        'pass': passes,
        'total': total,
        'fraction': passes / total if total else None,
    }


def score_contacts(
    spans: Sequence[tuple[int, int]],
    contacts: Sequence[tuple[int, int, bool | None]],
    gt_rallies: Sequence[GtRally],
    tolerances: Sequence[int] = DEFAULT_TOLERANCES,
) -> dict:
    """Contact-side metrics: count gate and per-tolerance credit, overall + per set.

    Candidates for a GT rally are the contacts from every span overlapping the
    rally extent. The count gate (candidates == strokes) is reported over covered
    rallies and, more strictly, over covered rallies whose span is not a merge of
    two rallies. Per-stroke credit uses greedy one-to-one matching per rally.

    :param spans: detected rally spans.
    :param contacts: detected contacts as ``(rally_id, contact_frame, proximity_ok)``;
        ``proximity_ok`` is not used here.
    :param gt_rallies: ground-truth rallies for one video.
    :param tolerances: frame tolerances for the credit curve.
    :return: dict of contact metrics (see keys inline).
    """
    classifications = classify_all(spans, gt_rallies)
    merged = merged_span_indices(spans, gt_rallies)

    contacts_by_span: dict[int, list[int]] = defaultdict(list)
    for rally_id, contact_frame, _proximity in contacts:
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
        'per_set': per_set,
    }


# ---------------------------------------------------------------------------
# Top-level score
# ---------------------------------------------------------------------------
def score_stage8(
    spans: Sequence[tuple[int, int]],
    contacts: Sequence[tuple[int, int, bool | None]],
    gt_rallies: Sequence[GtRally],
    tolerances: Sequence[int] = DEFAULT_TOLERANCES,
) -> dict:
    """Full stage-8 score for one video: boundary and contact metrics in one dict.

    :param spans: detected rally spans, ``[(start_frame, end_frame), ...]``.
    :param contacts: detected contacts, ``[(rally_id, contact_frame, proximity_ok), ...]``.
    :param gt_rallies: ground-truth rallies for the same video.
    :param tolerances: frame tolerances for the contact credit curve.
    :return: ``{'n_gt_rallies', 'tolerances', 'boundaries', 'contacts'}``.
    """
    return {
        'n_gt_rallies': len(gt_rallies),
        'tolerances': list(tolerances),
        'boundaries': score_boundaries(spans, gt_rallies),
        'contacts': score_contacts(spans, contacts, gt_rallies, tolerances),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _spans_from_df(spans_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Build the ordered spans list from a rally_spans frame (one video).

    rally_id must be contiguous ``0..n-1`` so the list index equals rally_id,
    which is the contract `segment_video` writes and `score_contacts` relies on.
    """
    ordered = spans_df.sort_values('rally_id')
    rally_ids = [int(rally_id) for rally_id in ordered['rally_id']]
    if rally_ids != list(range(len(ordered))):
        raise ValueError(f'rally_id not contiguous 0..n-1: {rally_ids}')
    return [(int(start), int(end)) for start, end in zip(ordered['start_frame'], ordered['end_frame'])]


def _parse_proximity(value: object) -> bool | None:
    """Parse a proximity_ok cell: blank/NaN -> None, else ``value == 'True'``."""
    if pd.isna(value):
        return None
    return str(value) == 'True'


def _contacts_from_df(contacts_df: pd.DataFrame) -> list[tuple[int, int, bool | None]]:
    """Build the contacts list from a contact_frames frame (one video)."""
    contacts: list[tuple[int, int, bool | None]] = []
    for row in contacts_df.itertuples(index=False):
        contacts.append((int(row.rally_id), int(row.contact_frame), _parse_proximity(row.proximity_ok)))
    return contacts


def _parse_tolerances(text: str) -> list[int]:
    """Parse a comma-separated tolerance list like ``'1,2,5,10'``."""
    return [int(part) for part in text.split(',') if part.strip()]


def _print_summary(results: dict) -> None:
    """Print a human-readable digest of the metrics dict."""
    boundaries = results['boundaries']
    contacts = results['contacts']
    n_rallies = results['n_gt_rallies']
    covered_frac = boundaries['covered_fraction']
    covered_pct = f'{covered_frac:.1%}' if covered_frac is not None else 'n/a'

    print(f'GT rallies: {n_rallies}')
    print(f'  covered {boundaries["covered"]} ({covered_pct}), '
          f'split {boundaries["split"]}, missed {boundaries["missed"]}')
    print(f'  merged spans {boundaries["merged_spans"]}, spurious spans {boundaries["spurious_spans"]}')

    start_alignment = boundaries['start_alignment']
    if start_alignment is not None:
        print(f'  start offset (span_start - first_stroke): '
              f'mean {start_alignment["mean"]:.1f}, median {start_alignment["median"]:.1f}, '
              f'p10 {start_alignment["p10"]:.1f}, p90 {start_alignment["p90"]:.1f}')

    covered_gate = contacts['count_gate']['covered']
    gate_frac = covered_gate['fraction']
    gate_pct = f'{gate_frac:.1%}' if gate_frac is not None else 'n/a'
    print(f'Count gate (covered): {covered_gate["pass"]}/{covered_gate["total"]} ({gate_pct})')

    print('Contact credit:')
    for tolerance, metric in contacts['tolerances'].items():
        recall = metric['recall']
        precision = metric['precision']
        f1 = metric['f1']
        recall_s = f'{recall:.3f}' if recall is not None else 'n/a'
        precision_s = f'{precision:.3f}' if precision is not None else 'n/a'
        f1_s = f'{f1:.3f}' if f1 is not None else 'n/a'
        print(f'  +/-{tolerance}: recall {recall_s}, precision {precision_s}, f1 {f1_s}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Score stage-8 rally spans and contacts against ShuttleSet GT (one video).'
    )
    parser.add_argument('--rally-spans-csv', type=Path, required=True,
                        help='Stage-8 rally_spans.csv (video_id, rally_id, start_frame, end_frame)')
    parser.add_argument('--contact-frames-csv', type=Path, required=True,
                        help='Stage-8 contact_frames.csv (video_id, rally_id, contact_frame, proximity_ok)')
    parser.add_argument('--shots-master', type=Path, default=DEFAULT_SHOTS_MASTER,
                        help='ShuttleSet shots_master.csv (default: the in-repo build output)')
    parser.add_argument('--vid', type=int, required=True,
                        help='ShuttleSet video id to score against (filters shots_master)')
    parser.add_argument('--tolerances', type=_parse_tolerances, default=list(DEFAULT_TOLERANCES),
                        help='Comma-separated frame tolerances for contact credit (default 1,2,5,10)')
    parser.add_argument('--out-json', type=Path, default=None,
                        help='Optional path to write the metrics dict as JSON')
    args = parser.parse_args()

    spans_df = pd.read_csv(args.rally_spans_csv)
    contacts_df = pd.read_csv(args.contact_frames_csv)
    shots_master = pd.read_csv(args.shots_master)

    # The pilot scores one video; the stage-8 CSVs must carry exactly one, else
    # rally_id (per-video list index) collides across videos. Fail loud.
    span_video_ids = set(spans_df['video_id'].astype(str).unique())
    if len(span_video_ids) != 1:
        raise ValueError(
            f'rally_spans has {len(span_video_ids)} video_ids {sorted(span_video_ids)}; '
            'the scorer expects one-video pilot input. Filter the CSV first.'
        )
    video_id = next(iter(span_video_ids))
    if len(contacts_df):
        stray = set(contacts_df['video_id'].astype(str).unique()) - span_video_ids
        if stray:
            raise ValueError(f'contact_frames carries video_ids absent from rally_spans: {sorted(stray)}')

    spans = _spans_from_df(spans_df)
    contacts = _contacts_from_df(contacts_df)
    gt_rallies = load_gt_rallies(shots_master, args.vid)

    results = score_stage8(spans, contacts, gt_rallies, args.tolerances)
    results['video_id'] = video_id
    results['vid'] = args.vid

    _print_summary(results)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(f'wrote {args.out_json}')


if __name__ == '__main__':
    main()
