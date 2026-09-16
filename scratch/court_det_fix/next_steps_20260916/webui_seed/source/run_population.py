"""Audit automatic direction and rectangle populations on frozen input packs."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from vp_pruning import Settings, estimate, select

from experiments.annotator.independent_court import detector


def prepare(source: dict) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    """Reproduce detector scaling and original proposal families."""
    width, height = source['dimensions']['width'], source['dimensions']['height']
    settings = detector.Settings(wide_families=True, min_supported_lines=3)
    resize = min(1.0, settings.max_dimension / max(width, height))
    size = (round(width * resize), round(height * resize))
    native_scale = np.asarray([width, height], dtype=float) / size
    segments = np.asarray(source['segments_px'], dtype=float) / np.tile(native_scale, 2)
    raw_families = detector._wide_line_families(segments)
    families = (detector._merge_lines(raw_families[0], settings), detector._merge_lines(raw_families[1], settings))
    return segments, families, size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--pencil-selection', choices=('ranked', 'coverage'), default='ranked')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    settings = Settings(pencil_selection=args.pencil_selection)
    args.output.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for input_path in args.inputs:
        with gzip.open(input_path, 'rt') as stream:
            packed = json.load(stream)
        for source in packed['cases']:
            case_id = source['id']
            if case_id in seen or (args.ids and case_id not in args.ids):
                continue
            seen.add(case_id)
            started = perf_counter()
            segments, families, size = prepare(source)
            points, estimator = estimate(segments, size, settings)
            estimated = perf_counter()
            _, selection = select(families, points, size,
                                  detector.Settings(wide_families=True, min_supported_lines=3), settings)
            selected = perf_counter()
            # Known seed identities assess the completed automatic selection only.
            targets = {'gxBQ_window_00_frame_5': 85795, 'am2_window_00_frame_150': 170556}
            target = targets.get(case_id)
            record = {
                'schema': 'vp-pruning-population/1', 'case_id': case_id,
                'settings': asdict(settings), 'working_size': size,
                'estimator': estimator, 'seed_selection': selection,
                'timing': {'estimate_s': estimated - started, 'select_s': selected - estimated},
                'known_seed': target,
                'known_seed_in_union': target in selection.get('union_pair_product_ids', []) if target else None,
                'known_seed_selected': target in selection['retained_pair_product_ids'] if target else None,
            }
            with gzip.open(args.output / f'{case_id}.json.gz', 'wt', compresslevel=9) as stream:
                json.dump(record, stream, separators=(',', ':'))
            print(case_id, 'pencils', len(points), 'union', selection.get('union_convex_quads'),
                  'rectangles', len(selection['retained_pair_product_ids']),
                  'known seed', record['known_seed_in_union'], record['known_seed_selected'],
                  'seconds', round(selected - started, 2), flush=True)
    if args.ids and seen != set(args.ids):
        raise ValueError(f'Missing requested cases: {set(args.ids) - seen}')


if __name__ == '__main__':
    main()
