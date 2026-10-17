"""Extend inference context while keeping the original benchmark score windows."""

from __future__ import annotations

import argparse
import math
import shutil
from bisect import bisect_left, bisect_right
from dataclasses import replace
from pathlib import Path
from typing import Any

from dataset_builder.vision import (
    CourtVision,
    load_court_vision,
    persist_court_vision,
    save_json_gz,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    load_manifest,
)

COURT_FOLDERS = ('ours', 'monotrack_for_ours')
COUNT_FIELDS = (
    'windows', 'changed_starts', 'changed_ends', 'clipped_starts', 'clipped_ends',
    'added_before_frames', 'added_after_frames',
)


def padded_spans(
    item: dict[str, Any], raw_cuts: tuple[tuple[int, int], ...], before_seconds: float, after_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Add context up to gap midpoints and scene cuts beyond the original windows."""
    spans = item['spans']
    cuts = [0, *(end for _start, end in raw_cuts)]
    before_frames = round(before_seconds * item['fps'])
    after_frames = round(after_seconds * item['fps'])
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    expanded = []
    for index, span in enumerate(spans):
        start, end = span['start_frame'], span['end_frame']
        previous_end = 0 if index == 0 else spans[index - 1]['end_frame']
        next_start = item['frame_count'] if index == len(spans) - 1 else spans[index + 1]['start_frame']
        left_gap = 0 if index == 0 else (previous_end + start) // 2
        right_gap = item['frame_count'] if index == len(spans) - 1 else (end + next_start) // 2
        left_cut = cuts[bisect_right(cuts, start) - 1]
        right_cut = cuts[bisect_left(cuts, end)]
        requested_start = min(start, span['label_frames'][0] - before_frames)
        requested_end = max(end, span['label_frames'][-1] + after_frames + 1)
        new_start = max(requested_start, left_gap, left_cut)
        new_end = min(requested_end, right_gap, right_cut)
        expanded.append({**span, 'start_frame': new_start, 'end_frame': new_end})
        counts['windows'] += 1
        counts['changed_starts'] += new_start != start
        counts['changed_ends'] += new_end != end
        counts['clipped_starts'] += new_start > requested_start
        counts['clipped_ends'] += new_end < requested_end
        counts['added_before_frames'] += start - new_start
        counts['added_after_frames'] += new_end - end
    return expanded, counts


def extend_courts(
    court: CourtVision, original_spans: list[dict[str, Any]], expanded_spans: list[dict[str, Any]],
) -> CourtVision:
    """Extend each saved court's coverage without fitting geometry or voting again."""
    original = [(span['start_frame'], span['end_frame']) for span in original_spans]
    expanded = [(span['start_frame'], span['end_frame']) for span in expanded_spans]
    if original == expanded:
        return court
    windows = dict(zip(original, expanded, strict=True))
    boundaries = {}
    present = court.evidence.court_present.copy()
    records_by_bounds = {(record.start_frame, record.end_frame): record for record in court.evidence.scene_records}
    for (start, end), (new_start, new_end) in windows.items():
        boundaries[start], boundaries[end] = new_start, new_end
        valid = records_by_bounds[start, end].scene_valid
        present[new_start:start] = valid
        present[end:new_end] = valid

    records = []
    for record in court.evidence.scene_records:
        start = boundaries.get(record.start_frame, record.start_frame)
        end = boundaries.get(record.end_frame, record.end_frame)
        if start == end:
            continue
        analysed_frame = record.analysed_frame
        if (record.start_frame, record.end_frame) not in windows and analysed_frame is not None:
            analysed_frame = (start + end) // 2
        records.append(replace(
            record, scene_index=len(records), start_frame=start, end_frame=end, analysed_frame=analysed_frame,
            exactly_two_fraction=record.exactly_two_count / (end - start),
        ))
    inputs = court.evidence.inputs
    assert inputs is not None
    rows = inputs.homography_rows.copy(deep=True)
    for column in ('start_frame', 'end_frame'):
        rows[column] = rows[column].replace(boundaries)
    extended_inputs = replace(inputs, homography_rows=rows)
    evidence = replace(court.evidence, inputs=extended_inputs, scene_records=tuple(records), court_present=present)
    return CourtVision(tuple((record.start_frame, record.end_frame) for record in records), evidence)


def prepare(inputs: Path, output: Path, before_seconds: float = 1.0, after_seconds: float = 1.0) -> None:
    """Write two court folders, an inference manifest and the original score manifest."""
    if any(not math.isfinite(value) or value < 1.0 for value in (before_seconds, after_seconds)):
        raise ValueError('padding must be finite and at least one second on each side')
    manifest = load_manifest(inputs)
    output.mkdir(parents=True, exist_ok=False)
    score_inputs = output / 'score_inputs'
    score_inputs.mkdir()
    shutil.copyfile(inputs / 'manifest.json.gz', score_inputs / 'manifest.json.gz')
    items = []
    counts_by_video = {}
    totals = dict.fromkeys(COUNT_FIELDS, 0)
    for item in manifest['items']:
        options = {
            'video_id': item['item_id'], 'frame_count': item['frame_count'],
            'resolution': tuple(map(float, item['resolution'])),
        }
        discovery = load_court_vision(inputs / 'discovery' / item['item_id'], **options)
        expanded, counts = padded_spans(item, discovery.raw_cuts, before_seconds, after_seconds)
        items.append({**item, 'spans': expanded})
        counts_by_video[item['item_id']] = counts
        for key in COUNT_FIELDS:
            totals[key] += counts[key]
        for name in COURT_FOLDERS:
            original = load_court_vision(inputs / name / item['item_id'], **options)
            padded = extend_courts(original, item['spans'], expanded)
            persist_court_vision(output / name / item['item_id'], court=padded, **options)
    save_json_gz(output / 'manifest.json.gz', {**manifest, 'items': items})
    save_json_gz(output / 'padding.json.gz', {
        'before_seconds': before_seconds, 'after_seconds': after_seconds, 'totals': totals, 'by_video': counts_by_video,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before-seconds', type=float, default=1.0)
    parser.add_argument('--after-seconds', type=float, default=1.0)
    args = parser.parse_args()
    prepare(args.inputs, args.output, args.before_seconds, args.after_seconds)


if __name__ == '__main__':
    main()
