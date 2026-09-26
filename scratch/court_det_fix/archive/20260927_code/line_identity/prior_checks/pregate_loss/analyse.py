"""Gate 2 and the pre-gate analysis: where the close courts sit before the geometry and player masks.

For each case-arm this loads the frozen control (working pixels, from the E3 record), the new
generation record and the side file. The side file holds, per matched pair, every combined court
before the masks (pair_<id>_combined_corners, float32 working pixels, in transforms order), the
geometry mask (pair_<id>_valid), the player mask (pair_<id>_usable), the two player fractions the
player mask reads (pair_<id>_player_any, pair_<id>_player_both_halves), and the arrays the cap_loss
run already wrote (pair_<id>_corners, the post-gate courts, and pair_<id>_retained_index).

Gate 2: for every pair the post-gate corners must equal the combined corners at the usable
indices, exactly. The record's combined, geometry_valid and geometry_players counts must equal
the array lengths and mask sums, and every usable court must also be valid.

Distances are maximum corner distance in working pixels with the 180-degree relabelling allowed,
through corner_errors() from cap_loss/analyse.py. "Nearest usable" per pair should equal the
cap_loss table's nearest proposed court, since the post-gate corners are the usable ones; the
script prints the largest gap between the two as a cross-check. The cap_loss table indexes the
post-gate array, so its index is compared with the nearest usable court's position among the usable
courts, not with its position in the combined array.
"""

from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAP_LOSS = HERE.parent / 'cap_loss'
# Loaded by path under its own name: this file shares the basename analyse.py with the cap_loss script.
_spec = importlib.util.spec_from_file_location('cap_loss_analyse', CAP_LOSS / 'analyse.py')
cap_loss_analyse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap_loss_analyse)
E3_DIR, corner_errors = cap_loss_analyse.E3_DIR, cap_loss_analyse.corner_errors

CASE_ARMS = (('gxBQ_window_00_frame_0', 'M'), ('am3_window_00_frame_0', 'R'))
# The direction fit per case-arm from the direction experiment, for context in the printout.
DIRECTION_FIT_PX = {('gxBQ_window_00_frame_0', 'M'): 19.7, ('am3_window_00_frame_0', 'R'): 4.3}
NEAR_PX = (5., 20.)
MIN_VISIBLE_SPAN_FRACTION = 0.15  # detector.DEFAULT_SETTINGS.min_visible_span_fraction
COLUMNS = ['case_id', 'arm', 'pair_id', 'pencils', 'combined_count', 'valid_count', 'usable_count',
           'gate2_usable_corners_equal',
           'nearest_combined_px', 'nearest_combined_index', 'nearest_combined_valid', 'nearest_combined_usable',
           'nearest_combined_player_any', 'nearest_combined_player_both_halves',
           'nearest_valid_px', 'nearest_valid_index', 'nearest_valid_usable',
           'nearest_valid_player_any', 'nearest_valid_player_both_halves',
           'nearest_usable_px', 'nearest_usable_index', 'nearest_usable_post_gate_index',
           'combined_within_5px', 'valid_within_5px', 'usable_within_5px',
           'combined_within_20px', 'valid_within_20px', 'usable_within_20px']


def read(path: Path) -> dict:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def load_case(case_id: str, arm: str) -> tuple[dict, np.ndarray, dict, np.lib.npyio.NpzFile]:
    e3_control = read(E3_DIR / f'{case_id}.json.gz')['control']
    control = np.asarray(e3_control['corners_working_px'], dtype=float)
    record = read(HERE / 'records' / 'new' / arm / 'results' / f'{case_id}.json.gz')
    assert record['working_size'] == e3_control['working_size'], (record['working_size'], e3_control['working_size'])
    pool = np.load(HERE / 'records' / 'new' / arm / 'results' / f'{case_id}_pool.npz')
    return e3_control, control, record, pool


def geometry_from_corners(corners: np.ndarray, size: tuple[int, int]) -> dict:
    """Convexity and visible span recomputed from one court's corners; positive depth needs the homography."""
    edges = np.roll(corners, -1, axis=0) - corners
    turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    span = np.minimum(corners.max(axis=0), np.asarray(size) - 1) - np.maximum(corners.min(axis=0), 0)
    fraction = span / np.asarray(size)
    return {'finite': bool(np.isfinite(corners).all()), 'convex': bool(np.all(turns > 0)),
            'span_fraction_xy': [round(float(value), 4) for value in fraction],
            'span_ok': bool(np.all(fraction >= MIN_VISIBLE_SPAN_FRACTION))}


def gate_pair(pair: dict, arrays: dict) -> tuple[bool, list[str]]:
    """Exact consistency between the record counts, the masks and the post-gate corners."""
    failures = []
    role = pair['role']
    combined = len(arrays['valid'])
    if arrays['combined_corners'].shape != (combined, 4, 2):
        failures.append(f'combined_corners shape {arrays["combined_corners"].shape} vs {combined}')
    if combined != role['combined']:
        failures.append(f'combined {combined} vs record {role["combined"]}')
    if int(arrays['valid'].sum()) != role['geometry_valid']:
        failures.append(f'valid sum {int(arrays["valid"].sum())} vs record {role["geometry_valid"]}')
    if int(arrays['usable'].sum()) != role['geometry_players']:
        failures.append(f'usable sum {int(arrays["usable"].sum())} vs record {role["geometry_players"]}')
    if np.any(arrays['usable'] & ~arrays['valid']):
        failures.append('usable court that is not valid')
    post_gate = arrays['corners']
    at_usable = arrays['combined_corners'][arrays['usable']]
    if post_gate.shape != at_usable.shape or not np.array_equal(post_gate, at_usable):
        failures.append(f'post-gate corners {post_gate.shape} differ from combined[usable] {at_usable.shape}')
    recomputed = arrays['valid'] & (arrays['player_any'] == 1) & (arrays['player_both_halves'] >= .5)
    if not np.array_equal(recomputed, arrays['usable']):
        failures.append('player mask recomputed from the stored fractions differs from usable')
    return not failures, failures


def nearest(errors: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, int] | tuple[None, None]:
    ids = np.arange(len(errors)) if mask is None else np.flatnonzero(mask)
    if not len(ids):
        return None, None
    index = int(ids[np.argmin(errors[ids])])
    return float(errors[index]), index


def analyse(case_id: str, arm: str) -> tuple[list[dict], dict, bool]:
    e3_control, control, record, pool = load_case(case_id, arm)
    size = tuple(record['working_size'])
    matched = [pair for pair in record['pairs'] if pair['status'] == 'matched']
    assert all(f'pair_{pair["pair_id"]}_combined_corners' not in pool.files
               for pair in record['pairs'] if pair['status'] != 'matched')
    rows, gate_ok = [], True
    best = {'combined': (np.inf, None, None), 'valid': (np.inf, None, None), 'usable': (np.inf, None, None)}
    near_courts = []
    totals = {f'{name}_within_{limit:g}px': 0 for limit in NEAR_PX for name in ('combined', 'valid', 'usable')}
    for pair in matched:
        pair_id = pair['pair_id']
        arrays = {name: pool[f'pair_{pair_id}_{name}'] for name in
                  ('corners', 'retained_index', 'combined_corners', 'valid', 'usable', 'player_any', 'player_both_halves')}
        ok, failures = gate_pair(pair, arrays)
        gate_ok = gate_ok and ok
        for failure in failures:
            print(f'  GATE 2 FAIL pair {pair_id}: {failure}')
        valid, usable = arrays['valid'], arrays['usable']
        row = {'case_id': case_id, 'arm': arm, 'pair_id': pair_id, 'pencils': ' '.join(map(str, pair['pencils'])),
               'combined_count': len(valid), 'valid_count': int(valid.sum()), 'usable_count': int(usable.sum()),
               'gate2_usable_corners_equal': ok}
        row.update(dict.fromkeys(COLUMNS[8:], None))
        if len(valid):
            errors = corner_errors(arrays['combined_corners'].astype(float), control)
            for name, mask in (('combined', None), ('valid', valid), ('usable', usable)):
                error, index = nearest(errors, mask)
                if index is None:
                    continue
                row[f'nearest_{name}_px'], row[f'nearest_{name}_index'] = error, index
                if name == 'usable':
                    # Position within the post-gate array, which the cap_loss table indexes.
                    row['nearest_usable_post_gate_index'] = int(usable[:index].sum())
                if name != 'usable':
                    row[f'nearest_{name}_player_any'] = float(arrays['player_any'][index])
                    row[f'nearest_{name}_player_both_halves'] = float(arrays['player_both_halves'][index])
                    row[f'nearest_{name}_usable'] = bool(usable[index])
                if name == 'combined':
                    row['nearest_combined_valid'] = bool(valid[index])
                if error < best[name][0]:
                    best[name] = (error, pair_id, index)
            for limit in NEAR_PX:
                close = errors <= limit
                for name, mask in (('combined', close), ('valid', close & valid), ('usable', close & usable)):
                    row[f'{name}_within_{limit:g}px'] = int(mask.sum())
                    totals[f'{name}_within_{limit:g}px'] += int(mask.sum())
            for index in np.flatnonzero(errors <= NEAR_PX[0]):
                near_courts.append((float(errors[index]), pair_id, int(index), bool(valid[index]), bool(usable[index]),
                                    float(arrays['player_any'][index]), float(arrays['player_both_halves'][index])))
        rows.append(row)
    # Details of the three overall nearest courts, read back from the side file.
    details = {}
    for name, (error, pair_id, index) in best.items():
        if pair_id is None:
            details[name] = None
            continue
        corners = pool[f'pair_{pair_id}_combined_corners'][index]
        details[name] = {'px': error, 'candidate': f'{pair_id}:{index}', 'pair_id': pair_id, 'index': index,
                         'pencils': next(pair['pencils'] for pair in matched if pair['pair_id'] == pair_id),
                         'valid': bool(pool[f'pair_{pair_id}_valid'][index]),
                         'usable': bool(pool[f'pair_{pair_id}_usable'][index]),
                         'player_any': float(pool[f'pair_{pair_id}_player_any'][index]),
                         'player_both_halves': float(pool[f'pair_{pair_id}_player_both_halves'][index]),
                         'geometry_from_corners': geometry_from_corners(corners.astype(float), size),
                         'corners_working_px': np.round(corners.astype(float), 2).tolist()}
    summary = {'case_id': case_id, 'arm': arm, 'control_source': e3_control['control_source'],
               'matched_pairs': len(rows), 'combined_total': sum(row['combined_count'] for row in rows),
               'valid_total': sum(row['valid_count'] for row in rows),
               'usable_total': sum(row['usable_count'] for row in rows),
               'nearest': details, 'totals': totals, 'near_courts': sorted(near_courts)}
    return rows, summary, gate_ok


def cross_check(rows: list[dict]) -> None:
    """The nearest usable court per pair must equal the cap_loss table's nearest proposed court."""
    with open(CAP_LOSS / 'table.csv', newline='') as stream:
        earlier = {(row['case_id'], row['arm'], int(row['pair_id'])): row for row in csv.DictReader(stream)}
    gaps = []
    for row in rows:
        key = (row['case_id'], row['arm'], row['pair_id'])
        assert key in earlier, key
        assert int(earlier[key]['proposed_count']) == row['usable_count'], (key, earlier[key]['proposed_count'], row['usable_count'])
        if row['nearest_usable_px'] is not None:
            gaps.append(abs(float(earlier[key]['nearest_proposed_px']) - row['nearest_usable_px']))
            assert int(earlier[key]['nearest_proposed_index']) == row['nearest_usable_post_gate_index'], key
    print(f'cross-check against cap_loss/table.csv: {len(gaps)} pairs with a nearest usable court; '
          f'largest gap to the earlier nearest proposed court {max(gaps):.2e} px; every index equal')


def report(summary: dict) -> None:
    print(f'\n{summary["case_id"]} arm {summary["arm"]}: control {summary["control_source"]}; '
          f'{summary["matched_pairs"]} matched pairs, {summary["combined_total"]} combined, '
          f'{summary["valid_total"]} passed geometry, {summary["usable_total"]} passed players; '
          f'direction fit {DIRECTION_FIT_PX[(summary["case_id"], summary["arm"])]} px')
    for name, label in (('combined', 'nearest combined court (before both masks)'),
                        ('valid', 'nearest court passing geometry'), ('usable', 'nearest usable court')):
        court = summary['nearest'][name]
        if court is None:
            print(f'  {label}: none')
            continue
        masks = f'geometry {"pass" if court["valid"] else "FAIL"}, players {"pass" if court["usable"] else "FAIL"}'
        print(f'  {label}: {court["px"]:.3f} px  candidate {court["candidate"]}  pair {court["pair_id"]} '
              f'(pencils {court["pencils"]}); {masks}; player fractions any {court["player_any"]:.4f}, '
              f'both halves {court["player_both_halves"]:.4f}')
        if not court['valid']:
            print(f'    geometry from the corners: {court["geometry_from_corners"]}')
        print(f'    corners (working px): {court["corners_working_px"]}')
    totals = summary['totals']
    for limit in NEAR_PX:
        print(f'  courts within {limit:g} px of the control: combined {totals[f"combined_within_{limit:g}px"]}, '
              f'passing geometry {totals[f"valid_within_{limit:g}px"]}, passing players {totals[f"usable_within_{limit:g}px"]}')
    for error, pair_id, index, valid, usable, any_fraction, both in summary['near_courts'][:12]:
        print(f'    candidate {pair_id}:{index}  {error:.3f} px  geometry {"pass" if valid else "FAIL"}  '
              f'players {"pass" if usable else "FAIL"}  fractions any {any_fraction:.4f} both {both:.4f}')


def main() -> None:
    all_rows, gate_ok = [], True
    for case_id, arm in CASE_ARMS:
        print(f'\n== {case_id} arm {arm}: gate 2 ==')
        rows, summary, ok = analyse(case_id, arm)
        print(f'  GATE 2 {"PASS" if ok else "FAIL"}: {sum(bool(row["gate2_usable_corners_equal"]) for row in rows)} of '
              f'{len(rows)} matched pairs have post-gate corners equal to the combined corners at the usable indices, '
              f'and record counts equal to the array lengths and mask sums')
        gate_ok = gate_ok and ok
        cross_check(rows)
        report(summary)
        all_rows.extend(rows)
    with open(HERE / 'table.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nwrote {HERE / "table.csv"} with {len(all_rows)} rows')
    print('GATE 2', 'PASS' if gate_ok else 'FAIL')
    sys.exit(0 if gate_ok else 1)


if __name__ == '__main__':
    main()
