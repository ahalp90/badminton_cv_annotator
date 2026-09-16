"""Gate 1: the instrumented generation record must equal the saved one for the same case-arm.

Compared fields, as the brief lists them: every pair's status, every shortlist's candidate_id
list in order, every shortlist's corners_px, the final entries' candidate IDs, line_winner_id and
paint_winner_id. Prints one line per check and exits 1 if anything differs.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASE_ARMS = (
    ('gxBQ_window_00_frame_0', 'M', 'M_gxBQ_window_00_frame_0.json.gz'),
    ('am3_window_00_frame_0', 'R', 'R_am3_window_00_frame_0.json.gz'),
    ('am2_window_01_frame_28019', 'B', 'baseline_am2_window_01_frame_28019.json.gz'),
)


def read(path: Path) -> dict:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def compare(case_id: str, arm: str, new: dict, saved: dict) -> bool:
    ok = True

    def check(name: str, passed: bool, detail: str = '') -> None:
        nonlocal ok
        ok = ok and passed
        print(f'  {"PASS" if passed else "FAIL"} {name} {detail}'.rstrip())

    check('pair count', len(new['pairs']) == len(saved['pairs']), f'{len(new["pairs"])} vs {len(saved["pairs"])}')
    status_equal = [pair_new['status'] == pair_saved['status'] for pair_new, pair_saved in zip(new['pairs'], saved['pairs'], strict=True)]
    check('every pair status', all(status_equal), f'{sum(status_equal)} of {len(status_equal)} equal')
    ids_equal, corners_equal, largest_gap, matched = [], [], 0., 0
    for pair_new, pair_saved in zip(new['pairs'], saved['pairs'], strict=True):
        if pair_saved['status'] != 'matched':
            continue
        matched += 1
        new_ids = [entry['candidate_id'] for entry in pair_new['shortlist']]
        saved_ids = [entry['candidate_id'] for entry in pair_saved['shortlist']]
        ids_equal.append(new_ids == saved_ids)
        new_corners = np.asarray([entry['corners_px'] for entry in pair_new['shortlist']], dtype=float)
        saved_corners = np.asarray([entry['corners_px'] for entry in pair_saved['shortlist']], dtype=float)
        same_shape = new_corners.shape == saved_corners.shape
        corners_equal.append(same_shape and np.array_equal(new_corners, saved_corners))
        if same_shape and new_corners.size:
            largest_gap = max(largest_gap, float(np.abs(new_corners - saved_corners).max()))
    check('every shortlist candidate_id list in order', all(ids_equal), f'{sum(ids_equal)} of {matched} matched pairs equal')
    check('every shortlist corners_px', all(corners_equal),
          f'{sum(corners_equal)} of {matched} matched pairs exactly equal; largest absolute gap {largest_gap:.3e} px')
    new_entries = [entry['candidate_id'] for entry in new['entries']]
    saved_entries = [entry['candidate_id'] for entry in saved['entries']]
    check('final entries candidate IDs', new_entries == saved_entries, f'{len(new_entries)} vs {len(saved_entries)} entries')
    check('line_winner_id', new['line_winner_id'] == saved['line_winner_id'], f'{new["line_winner_id"]} vs {saved["line_winner_id"]}')
    check('paint_winner_id', new['paint_winner_id'] == saved['paint_winner_id'], f'{new["paint_winner_id"]} vs {saved["paint_winner_id"]}')
    check('pooled_candidates', new['pooled_candidates'] == saved['pooled_candidates'], f'{new["pooled_candidates"]} vs {saved["pooled_candidates"]}')
    return ok


def main() -> None:
    all_ok = True
    for case_id, arm, saved_name in CASE_ARMS:
        new = read(HERE / 'records' / 'new' / arm / 'results' / f'{case_id}.json.gz')
        saved = read(HERE / 'records' / 'saved' / saved_name)
        assert new['case_id'] == case_id == saved['case_id'] and new['arm'] == arm, (new['case_id'], new.get('arm'))
        print(f'{case_id} arm {arm}: new run {new["run"]} vs saved {saved.get("run", "automatic_axes_20260914 baseline")}')
        all_ok = compare(case_id, arm, new, saved) and all_ok
    print('GATE 1', 'PASS' if all_ok else 'FAIL')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
