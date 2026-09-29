"""Compare the existing paint-ridge cue on fixed court candidates and controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from run_diagnosis import read, write

from experiments.annotator.independent_court import assignment, detector


def ridge_mask(frame: np.ndarray, segments: np.ndarray) -> np.ndarray:
    accepted = detector._filter_painted_stripes(frame, segments)
    identities = {tuple(segment) for segment in accepted}
    return np.asarray([tuple(segment) in identities for segment in segments], dtype=bool)


def profiles(frame: np.ndarray, homographies: np.ndarray) -> list[dict]:
    """Measure finite visible intervals, retaining missing profiles separately."""
    height, width = frame.shape[:2]
    projected, _ = detector.project(homographies, detector.SEGMENTS_M)
    endpoints, visible = detector._visible_samples(projected.reshape(-1, 12, 2, 2), (width, height), 2)
    passed = np.zeros(visible.shape, dtype=bool)
    passed[visible] = ridge_mask(frame, endpoints[visible].reshape(-1, 4))
    results = []
    for interval_pass, interval_visible in zip(passed, visible, strict=True):
        markings = []
        for intervals in assignment.MARKING_INTERVALS:
            available = interval_visible[list(intervals)]
            markings.append(float(interval_pass[list(intervals)][available].mean()) if available.any() else None)
        available_scores = [value for value in markings if value is not None]
        results.append({'score': float(np.mean(available_scores)) if available_scores else None,
                        'marking_ridge': markings, 'interval_visible': interval_visible.tolist(),
                        'interval_ridge': interval_pass.tolist()})
    return results


def frame_path(source: dict, root: Path) -> Path:
    if source['id'].startswith('gxBQ'):
        return root / 'gx_extension/people' / source['image']
    if source['id'].startswith('shuttleset'):
        return root / 'original' / source['image']
    video = source['id'].split('_')[0]
    frame = int(source['id'].rsplit('_', 1)[1])
    return root / 'axis_matching_20260914/images' / video / f'frame_{frame:08d}.png'


def inspect(source: dict, given: dict, marking: dict, root: Path) -> dict:
    path = frame_path(source, root)
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(path)
    assert frame.shape[:2] == (source['dimensions']['height'], source['dimensions']['width'])
    size = tuple(given['working_size'])
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    previous = next(row for row in marking['records'] if row['case_id'] == source['id'])['fit']
    controls = [given['control_corners_px'], previous['corners_px']]
    transforms = [np.asarray(entry['homography_working']) for entry in given['entries']]
    for corners in controls:
        transforms.append(cv2.getPerspectiveTransform(detector.CORNER_COURT_M,
                                                       (np.asarray(corners) / scale).astype(np.float32)))
    measured = profiles(frame, np.asarray(transforms))
    entries = []
    for entry, profile in zip(given['entries'], measured[:-2], strict=True):
        entries.append({'candidate_id': entry['candidate_id'], 'profile': profile,
                        'stripe_score': entry['stripe']['exclusive']['score'], 'gates': entry['gates'],
                        'reference_max_corner_1280_px': entry['reference_max_corner_1280_px'],
                        'corners_px': entry['corners_px']})
    eligible = [entry for entry in entries if entry['gates']['camera_error'] is not None
                and entry['gates']['camera_error'] <= .1 and entry['profile']['score'] is not None]
    stripe = max(eligible, key=lambda entry: entry['stripe_score'], default=None)
    ridge = max(eligible, key=lambda entry: (entry['profile']['score'], entry['stripe_score']), default=None)
    return {'case_id': source['id'], 'image_path': str(path), 'given_direction_source': given['given_direction_source'],
            'automatic_detection': False, 'entries': entries, 'control_profile': measured[-2],
            'previous_inspected_profile': measured[-1],
            'stripe_winner_id': None if stripe is None else stripe['candidate_id'],
            'ridge_winner_id': None if ridge is None else ridge['candidate_id']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--given', type=Path, required=True)
    parser.add_argument('--marking-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    marking = read(args.marking_summary)
    records = []
    for pack_path in args.inputs:
        for source in read(pack_path)['cases']:
            path = args.given / f"{source['id']}.json.gz"
            if path.exists():
                result = inspect(source, read(path), marking, args.root)
                records.append(result)
                print(source['id'], 'stripe/ridge IDs', result['stripe_winner_id'], result['ridge_winner_id'], flush=True)
    write(args.output, {'schema': 'fixed-court-ridge-diagnostic/1', 'records': records,
                        'settings': {'samples': detector.RIDGE_SAMPLES,
                                     'centre_shifts_px': detector.RIDGE_CENTRE_SHIFTS.tolist(),
                                     'side_distance_px': detector.RIDGE_SIDE_DISTANCE,
                                     'minimum_contrast': detector.RIDGE_MIN_CONTRAST,
                                     'minimum_fraction': detector.RIDGE_MIN_FRACTION}})


if __name__ == '__main__':
    main()
