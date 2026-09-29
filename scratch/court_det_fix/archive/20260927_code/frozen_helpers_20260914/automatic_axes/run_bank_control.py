"""Test the unchanged matcher with reference-selected directions from the observed bank."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
from inspect_appearance import frame_path, profiles
from run_diagnosis import read, write
from run_given import run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy.resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    pack, bank = read(args.inputs), read(args.bank)
    case_id = bank['case_id']
    source = next(source for source in pack['cases'] if source['id'] == case_id)
    selected = bank['best_measured_fit']
    control = np.asarray(selected['homography_working'])
    result = run_case(source, control, 'reference_selected_observed_bank_directions', zone,
                      pack['references'][case_id], True, 512, 'finite')
    result['label_selected_bank_directions'] = True
    result['bank_candidate_ids'] = [selected['x_candidate_id'], selected['y_candidate_id']]
    entries = result['entries']
    frame = cv2.imread(str(frame_path(source, Path('.'))))
    if frame is None:
        raise FileNotFoundError(source['image'])
    frame = cv2.resize(frame, result['working_size'], interpolation=cv2.INTER_AREA)
    if entries:
        paint = profiles(frame, np.asarray([entry['homography_working'] for entry in entries]))
        for entry, profile in zip(entries, paint, strict=True):
            entry['profile'] = profile
    eligible = [entry for entry in entries if entry['gates']['camera_error'] is not None
                and entry['gates']['camera_error'] <= .1 and entry['profile']['score'] is not None]
    line = max(eligible, key=lambda entry: entry['stripe']['exclusive']['score'], default=None)
    paint = max(eligible, key=lambda entry: (entry['profile']['score'], entry['stripe']['exclusive']['score']), default=None)
    result['camera_eligible'] = len(eligible)
    result['line_winner_id'] = None if line is None else line['candidate_id']
    result['paint_winner_id'] = None if paint is None else paint['candidate_id']
    result['winner_reference_errors'] = {
        'line': None if line is None else line['reference_max_corner_1280_px'],
        'paint': None if paint is None else paint['reference_max_corner_1280_px'],
    }
    write(args.output, result)
    print(case_id, 'label-selected observed directions', result['bank_candidate_ids'],
          'control diagnostics', result['control_diagnostics'], 'camera eligible', len(eligible),
          'winner reference errors', result['winner_reference_errors'], flush=True)


if __name__ == '__main__':
    main()
