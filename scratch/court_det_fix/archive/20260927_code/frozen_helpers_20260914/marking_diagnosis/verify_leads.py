"""Verify direction-family and template-spacing review leads on Carmack."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from run_diagnosis import read, write
from run_population import prepare
from vp_pruning import rectangle_population

from experiments.annotator.independent_court import detector


def verify(source: dict, reference: dict, saved: dict) -> dict:
    segments, merged, size = prepare(source)
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M,
                                            (np.asarray(reference['corners_px']) / scale).astype(np.float32))
    projected, _ = detector.project(homography[None], detector.SEGMENTS_M)
    intervals = projected.reshape(12, 2, 2)
    vectors = intervals[:, 1] - intervals[:, 0]
    angles = (np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) + 90) % 180 - 90
    families = detector._wide_line_families(segments)
    maps = detector._distance_maps(families, size)
    samples, visible = detector._visible_samples(intervals[None], size, 24)
    pixels = np.clip(samples[0], 0, np.asarray(size) - 1).astype(int)
    support = np.asarray([(distance_map[pixels[..., 1], pixels[..., 0]] <= 4).mean(axis=1) * visible[0]
                          for distance_map in maps])
    generation = saved['diagnostic_rankings']['geometry_before_players']['max_corner_native_px'][0]['generation_id']
    template = detector.TEMPLATE_TRANSFORMS[generation % len(detector.TEMPLATE_TRANSFORMS)]
    coordinates = detector.project(np.linalg.inv(template)[None], detector.UNIT_CORNERS)[0][0]
    top_seeds = []
    if saved['final']['candidates']:
        target = np.asarray(saved['final']['candidates'][0]['corners_px']) / scale
        quads, ranks, pair_ids = rectangle_population(merged, size)
        selected_ids = np.asarray(saved['seed_selection']['retained_pair_product_ids'])
        indices = np.searchsorted(pair_ids, selected_ids)
        np.testing.assert_array_equal(pair_ids[indices], selected_ids)
        rectangles = np.asarray([cv2.getPerspectiveTransform(detector.UNIT_CORNERS, quad.astype(np.float32))
                                 for quad in quads[indices]])
        for offset in range(0, len(rectangles), 8):
            transforms = (rectangles[offset:offset + 8, None] @ detector.TEMPLATE_TRANSFORMS).reshape(-1, 3, 3)
            corners, _ = detector.project(transforms, detector.CORNER_COURT_M)
            matches = np.flatnonzero(np.max(np.abs(corners - target), axis=(1, 2)) <= 1e-5)
            for index in matches:
                rectangle_index, template_index = divmod(offset * len(detector.TEMPLATE_TRANSFORMS) + int(index),
                                                         len(detector.TEMPLATE_TRANSFORMS))
                rectangle = int(indices[rectangle_index])
                top_seeds.append({'generation_id': offset * len(detector.TEMPLATE_TRANSFORMS) + int(index),
                                  'pair_product_id': int(pair_ids[rectangle]), 'template_index': template_index,
                                  'line_ranks': ranks[rectangle].tolist()})
        assert top_seeds, 'Retained top court must reproduce from the saved population'
    return {'case_id': source['id'], 'reference_interval_angle_deg': angles.tolist(),
            'reference_support_by_map': support.tolist(), 'template_index': generation % 150,
            'template_span_m': np.ptp(coordinates, axis=0).tolist(), 'top_retained_seeds': top_seeds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--saved', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    records = []
    for pack_path in args.inputs:
        packed = read(pack_path)
        for source in packed['cases']:
            saved_path = args.saved / f"{source['id']}.json.gz"
            if saved_path.exists():
                records.append(verify(source, packed['references'][source['id']], read(saved_path)))
    write(args.output, {'records': records, 'label_guided': True})
    print('Verified', len(records), 'cases', flush=True)


if __name__ == '__main__':
    main()
