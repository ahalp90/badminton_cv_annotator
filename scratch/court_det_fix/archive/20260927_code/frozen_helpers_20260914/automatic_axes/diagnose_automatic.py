"""Attach reference and inspected-control diagnostics after automatic generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from projective_seed import corner_errors
from run_diagnosis import read, write


def nearest(entries: list[dict], target: np.ndarray, scale: np.ndarray) -> dict | None:
    if not entries:
        return None
    corners = np.asarray([entry['corners_px'] for entry in entries]) * scale
    errors = corner_errors(corners, target * scale)
    index = int(np.argmin(errors))
    return {'candidate_id': entries[index]['candidate_id'], 'max_corner_px': float(errors[index])}


def diagnose(source: dict, reference: dict, automatic: dict, given: dict) -> dict:
    native_size = np.array([source['dimensions']['width'], source['dimensions']['height']])
    working_scale = np.asarray(automatic['working_size']) / native_size
    display_scale = np.array([1280, 720]) / native_size
    control = np.asarray(given['control_corners_px'])
    manual = np.asarray(reference['corners_px'])
    final_ids = {entry['candidate_id'] for entry in automatic['entries']}
    pairs = []
    pooled = []
    for pair in automatic['pairs']:
        if pair['status'] != 'matched':
            continue
        entries = pair['shortlist']
        pooled.extend(entries)
        pairs.append({'pair_id': pair['pair_id'], 'pencils': pair['pencils'],
                      'shortlist_count': len(entries),
                      'global_survivors': sum(entry['candidate_id'] in final_ids for entry in entries),
                      'group_counts': [len(axis['diagnostics']['retained_group_ids'])
                                       for axis in pair['role'].get('axes', [])],
                      'control_working': nearest(entries, control, working_scale),
                      'reference_display': nearest(entries, manual, display_scale)})
    before_camera = None
    if 'camera_prefilter' in automatic:
        camera_ids = {entry['candidate_id'] for entry in automatic['camera_prefilter']
                      if entry['camera_error'] is not None and entry['camera_error'] <= automatic['camera_error_limit']}
        eligible_pooled = [entry for entry in pooled if entry['candidate_id'] in camera_ids]
        before_camera = nearest(eligible_pooled, control, working_scale)
    entries = automatic['entries']
    camera_eligible = [entry for entry in entries if entry['gates']['camera_error'] is not None
                       and entry['gates']['camera_error'] <= automatic['camera_error_limit']]
    winner_ids = [automatic['line_winner_id'], automatic['paint_winner_id']]
    winners = {}
    for name, candidate_id in zip(('line', 'paint'), winner_ids, strict=True):
        entry = next((entry for entry in entries if entry['candidate_id'] == candidate_id), None)
        if entry is None:
            winners[name] = None
            continue
        winners[name] = {'candidate_id': candidate_id, 'pair_id': entry['pair_id'],
                         'reference_display': nearest([entry], manual, display_scale)['max_corner_px'],
                         'control_working': nearest([entry], control, working_scale)['max_corner_px'],
                         'stripe_score': entry['stripe']['exclusive']['score'],
                         'profile_score': entry['profile']['score'], 'gates': entry['gates'],
                         'corners_px': entry['corners_px']}
    return {'case_id': source['id'], 'control_source': given['given_direction_source'],
            'label_guided_diagnostics': True, 'control_corners_px': control.tolist(),
            'pairs': pairs, 'before_global_cap': nearest(pooled, control, working_scale),
            'before_global_camera_eligible': before_camera,
            'after_global_cap': nearest(entries, control, working_scale),
            'after_camera_check': nearest(camera_eligible, control, working_scale), 'winners': winners,
            'counts': {'ordered_pairs': len(automatic['pairs']),
                       'camera_direction_rejected': sum(pair['status'] == 'camera_direction_bound'
                                                        for pair in automatic['pairs']),
                       'pooled': len(pooled), 'final': len(entries), 'final_camera_eligible': len(camera_eligible)}, 'elapsed_s': automatic['elapsed_s']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--given', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for pack_path in args.inputs:
        pack = read(pack_path)
        for source in pack['cases']:
            path = args.results / f"{source['id']}.json.gz"
            if path.exists():
                record = diagnose(source, pack['references'][source['id']], read(path), read(args.given / path.name))
                records.append(record)
                print(source['id'], record['counts'], 'control', record['before_global_cap'],
                      record['after_global_cap'], 'winner errors',
                      {key: None if value is None else value['reference_display']
                       for key, value in record['winners'].items()}, flush=True)
    write(args.output, {'schema': 'automatic-axis-diagnosis/1', 'records': records})


if __name__ == '__main__':
    main()
