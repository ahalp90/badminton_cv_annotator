"""Score stage-8 rally segmentation and contact detection against ShuttleSet GT.

Scores the one-video pilot (stage A of the stage-8 sweep plan, section 4) and is
imported by the later threshold-sweep runner. The core is pure functions over
in-memory inputs, matching what ``scraper.stage8_rally_segmentation.segment_video``
returns:

  - spans:    ``list[(start_frame, end_frame)]`` half-open detected rally spans,
              rally_id is the list index
  - contacts: ``list[(rally_id, contact_frame, proximity_ok)]``

Ground truth is ``shots_master.csv`` filtered to one ``vid``; strokes group by
``(set_id, rally)`` into one GT rally whose extent is
``[min frame_num, max frame_num]``. All frames are source-video frames; the
pilot track starts at source frame 0, so no offset is applied.

The metrics core is imported from the calibration package. The CLI is a thin
wrapper that reads the stage-8 CSVs plus shots_master, prints a summary, and
writes the metrics dict as JSON.

Usage:
    python -m scripts.stage8_score \
        --rally-spans-csv data/scrape_output/rally_spans.csv \
        --contact-frames-csv data/scrape_output/contact_frames.csv \
        --vid 1 --out-json data/scrape_output/stage8_score.json
"""
from __future__ import annotations

# These imports are the compatibility surface for script consumers.
# ruff: noqa: F401
import argparse
import json
from pathlib import Path

import pandas as pd

from annotator.calibration.scoring import (
    DEFAULT_TOLERANCES,
    SPURIOUS_NOTE,
    GtRally,
    RallyBoundary,
    _count_gate,
    _offset_stats,
    _prf,
    _raw_precision_curve,
    _spans_containing,
    _spans_overlapping_extent,
    _tolerance_curve,
    classify_all,
    classify_rally_boundary,
    greedy_match,
    load_gt_rallies,
    merged_span_indices,
    score_boundaries,
    score_contacts,
    score_stage8,
)  # noqa: F401 — this module preserves the script import surface

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHOTS_MASTER = (
    REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
)


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


def _contacts_from_df(contacts_df: pd.DataFrame) -> list[tuple[int, int, bool | None, bool | None]]:
    """Build the contacts list from a contact_frames frame (one video).

    wrist_near (the contact wrist check) rides in the fourth field when the column is present;
    an older CSV without it reads None, so scoring a pre-wrist-check file still works. Both bool
    columns share `_parse_proximity`'s blank/True/False encoding.
    """
    has_wrist = 'wrist_near' in contacts_df.columns
    contacts: list[tuple[int, int, bool | None, bool | None]] = []
    for row in contacts_df.itertuples(index=False):
        wrist_near = _parse_proximity(row.wrist_near) if has_wrist else None
        contacts.append((int(row.rally_id), int(row.contact_frame),
                         _parse_proximity(row.proximity_ok), wrist_near))
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
