"""Export pooled search results and original-dataset frame-rate groups."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.annotator.heuristic_only_tune.data import (
    COUNT_FIELDS,
    TRACKED_SPLIT,
    add_rates,
)
from experiments.annotator.heuristic_only_tune.run import read_json, write_json

TABLE_SCHEMA = 'heuristic-only-tune-result-table/1'


def _pool(row: dict[str, Any], videos: list[str], tolerance: str) -> dict[str, int]:
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    for video in videos:
        values = row['by_video'][video][tolerance]
        for name in COUNT_FIELDS:
            counts[name] += values[name]
    return counts


def result_rows(summaries: list[Path], selected: Path | None = None) -> list[dict[str, Any]]:
    """Check stored totals and return rows pooled from per-video integer counts."""
    selected_ids = None
    if selected is not None:
        objectives = read_json(selected)['objectives']
        selected_ids = {identity for identities in objectives.values() for identity in identities}
    frozen = read_json(TRACKED_SPLIT)['videos']
    original = {video['fixture']: video for video in frozen}
    rows = []
    for path in summaries:
        summary = read_json(path)
        if summary['schema'] != 'heuristic-only-tune-summary/1':
            raise ValueError(f'{path.parent.name}: unsupported summary schema')
        split, mode = summary['split'], summary['mode']
        videos = summary['videos']
        groups = {'all': videos}
        if split in {'train', 'validation'}:
            for video in videos:
                if original[video]['role'] != split:
                    raise ValueError(f'{path.parent.name}: {video} belongs to a different frozen split')
            for fps in (25, 30):
                own = [video for video in videos if original[video]['fps'] == fps]
                if own:
                    groups[str(fps)] = own
        for candidate in summary['candidates']:
            identity = candidate['candidate_id']
            if set(candidate['by_video']) != set(videos):
                raise ValueError(f'{path.parent.name}: {identity} has a different video population')
            for tolerance in ('5', '10'):
                totals = _pool(candidate, videos, tolerance)
                if add_rates(totals) != candidate['by_tolerance'][tolerance]:
                    raise ValueError(f'{path.parent.name}: {identity} has inconsistent totals at ±{tolerance}')
                if mode == 'heuristic' and selected_ids is not None and identity not in selected_ids:
                    continue
                for fps, own in groups.items():
                    counts = totals if fps == 'all' else _pool(candidate, own, tolerance)
                    rows.append({
                        'stage': path.parent.name,
                        'split': split,
                        'mode': mode,
                        'candidate_id': identity,
                        'tolerance_base30': int(tolerance),
                        'fps_group': fps,
                        'n_videos': len(own),
                        **add_rates(counts),
                    })
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries', type=Path, nargs='+', required=True)
    parser.add_argument('--selected', type=Path, help='keep heuristic candidate IDs named by selected objectives')
    parser.add_argument('--output', type=Path, required=True, help='write the result table as .json.gz')
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {'schema': TABLE_SCHEMA, 'rows': result_rows(args.summaries, args.selected)})


if __name__ == '__main__':
    main()
