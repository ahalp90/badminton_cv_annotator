"""Measure whether each fragment sits on white paint, before any filter rule is chosen.

For every fragment in a view's input pack this samples the native frame along the fragment and
across it, at perpendicular offsets of up to PROFILE_HALF_WIDTH working pixels, and records two
numbers: the ridge contrast (how much brighter the brightest pixel within RIDGE_SEARCH working
pixels of the fragment is than the brighter of the two flank means on each side; two flank
windows are tried, FLANK_WINDOWS_WORKING_PX, and the larger contrast is kept) and the saturation
at that brightest pixel. The near window suits thin far lines whose surroundings change within
a few pixels; the far window suits the wide near-court stripes a fragment sits at the edge of. Court paint is a bright, unsaturated
ridge that falls away on both sides; floor-plank seams, wall cladding seams, the edge of a tan
strip (a step) and skin (a ramp) are not. Fragments often sit on one edge of a stripe rather
than its centre, which is why the peak is searched near the fragment and the flanks start
beyond the widest stripe seen (about 6 working pixels on the near court at 1080p).

Fragments are labelled for the study only: "marking" when both endpoints lie within EDGE_PX
working pixels of any painted court marking projected through the control (outline, service
lines, centre line and the inner sidelines), "row_31", "row_51" or "row_61" for GX0's three rows
the earlier check identified as non-court structures, "other" for the rest. The labels come from
the control and are never used by any filter. Near the far end of a court, parallel markings a
few pixels apart can put a fragment between two markings within the band; the label is a study
aid, not ground truth. Every raw pack fragment is measured, including the few the matcher's
observation step later clips away; samples that fall outside the frame read the nearest pixel.
The broadcast frames are single-channel PNGs that cv2 replicates to three channels, so their
saturation is zero everywhere and only the contrast term can act there. Writes
runs/<out>/profiles.csv.gz (one row per fragment per view) and prints quantiles per label.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import cv2
import numpy as np

from shared import (
    CASE_IDS,
    LABELS,
    add_helper_paths,
    control_corners,
    frame_path,
    load_source,
)

add_helper_paths()

from run_population import prepare

from experiments.annotator.independent_court import detector
from scratch.court_det_fix.court_detector.search import features, profiles

EDGE_PX = 6.0
# Raw fragment IDs of GX0's three merged rows that the earlier check judged non-court (wall seam,
# floor strip, spectator's face); person_mask_replay.py carries the same lists.
GX0_OFFENDING = {'gxBQ_window_00_frame_0': {584: 'row_31', 683: 'row_31', 417: 'row_31', 583: 'row_31',
                                             469: 'row_51', 499: 'row_61'}}
COLUMNS = ['case_id', 'label', 'fragment_id', 'length_working_px', 'study_label', 'ridge_contrast', 'peak_saturation',
           'peak_offset_working_px', 'centre_brightness', 'flank_brightness']


def point_to_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    fraction = np.clip(((points - start) @ vector) / (vector @ vector), 0., 1.)
    nearest = start + fraction[:, None] * vector
    return np.linalg.norm(points - nearest, axis=1)


def edge_distance(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest of the four court edges."""
    best = np.full(len(points), np.inf)
    for index in range(4):
        best = np.minimum(best, point_to_segment_distance(points, corners[index], corners[(index + 1) % 4]))
    return best


def marking_distance(points: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest painted court marking projected through the control."""
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M.astype(np.float32), control.astype(np.float32))
    segments, _ = detector.project(np.asarray(homography, dtype=float)[None], detector.SEGMENTS_M.reshape(-1, 2))
    segments = segments[0].reshape(-1, 2, 2)
    best = np.full(len(points), np.inf)
    for start, end in segments:
        best = np.minimum(best, point_to_segment_distance(points, start, end))
    return best


def study_case(case_id: str) -> list[dict]:
    source = load_source(case_id)
    frame = cv2.imread(str(frame_path(source)))
    assert frame is not None, frame_path(source)
    assert frame.shape[:2] == (source['dimensions']['height'], source['dimensions']['width'])
    segments_working, _, size = prepare(source)
    scale = np.asarray([source['dimensions']['width'], source['dimensions']['height']], dtype=float) / size
    fragments_native = np.asarray(source['segments_px'], dtype=float)
    control_working, _ = control_corners(case_id)
    lengths = np.linalg.norm(segments_working[:, 2:] - segments_working[:, :2], axis=1)
    start_distance = marking_distance(segments_working[:, :2], control_working)
    end_distance = marking_distance(segments_working[:, 2:], control_working)
    on_marking = (start_distance <= EDGE_PX) & (end_distance <= EDGE_PX)
    offending = GX0_OFFENDING.get(case_id, {})
    measured = features(profiles(frame, fragments_native, scale))
    rows = []
    for fragment_id, (contrast, peak_saturation, peak_offset, centre, flank) in enumerate(measured):
        study_label = offending.get(fragment_id, 'marking' if on_marking[fragment_id] else 'other')
        rows.append({'case_id': case_id, 'label': LABELS[case_id], 'fragment_id': fragment_id,
                     'length_working_px': round(float(lengths[fragment_id]), 2), 'study_label': study_label,
                     'ridge_contrast': round(float(contrast), 2), 'peak_saturation': round(float(peak_saturation), 2),
                     'peak_offset_working_px': float(peak_offset), 'centre_brightness': round(float(centre), 2),
                     'flank_brightness': round(float(flank), 2)})
    return rows


def quantiles(values: list[float]) -> str:
    if not values:
        return 'none'
    points = np.percentile(values, [5, 25, 50, 75, 95])
    return f'n={len(values)} p5 {points[0]:.0f} p25 {points[1]:.0f} p50 {points[2]:.0f} p75 {points[3]:.0f} p95 {points[4]:.0f}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in CASE_IDS:
        case_rows = study_case(case_id)
        rows.extend(case_rows)
        print(f'{LABELS[case_id]}: {len(case_rows)} fragments')
        for study_label in ('marking', 'other'):
            subset = [row for row in case_rows if row['study_label'] == study_label]
            print(f'  {study_label:7s} ridge contrast {quantiles([row["ridge_contrast"] for row in subset])}')
            print(f'  {study_label:7s} peak saturation {quantiles([row["peak_saturation"] for row in subset])}')
        for row in case_rows:
            if row['study_label'].startswith('row_'):
                print(f'  {row["study_label"]} fragment {row["fragment_id"]}: ridge contrast {row["ridge_contrast"]}, '
                      f'peak saturation {row["peak_saturation"]}, centre {row["centre_brightness"]}, flank {row["flank_brightness"]}')
    with gzip.open(args.output / 'profiles.csv.gz', 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {args.output / "profiles.csv.gz"} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
