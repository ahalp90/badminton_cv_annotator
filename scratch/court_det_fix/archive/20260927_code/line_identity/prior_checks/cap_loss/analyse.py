"""Where the close courts vanish: per-pair nearest proposed court versus nearest retained court.

For each case-arm this loads the frozen control (working pixels, from the E3 record), the pool
recording (every court that passed the player checks, per direction pair, plus the indices the
per-pair cap of 256 kept) and the new generation record. It writes table.csv with one row per
matched pair and prints the overall nearest proposed and nearest retained courts.

Gate 2: the overall nearest retained court must equal the accounting table's "nearest pooled"
value. The accounting measured it in float64 from the record shortlists (native corners scaled
to working pixels); the recording holds float32 working corners. Both are checked: the float64
recomputation from the shortlists must match the accounting exactly (value and candidate ID),
and the float32 recording must match the value within FLOAT32_TOLERANCE_PX. When two courts tie
on their worst corner, the candidate ID depends only on ordering, so it is reported, not gated.

A discarded court counts as "closer" only when it beats the retained court by more than
ROUNDING_MARGIN_PX. Float32 rounding of working-pixel corners is below 1e-4 px, and the per-pair
distinctness rule drops near-duplicates that differ by far less than that.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]  # the repository root; this copy lives at scratch/court_det_fix/line_identity/prior_checks/cap_loss
E3_DIR = REPO / 'scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/e3'
# The accounting table's nearest pooled court per case-arm (summary.md and e4/diagnosis.json.gz).
ACCOUNTING = {
    ('gxBQ_window_00_frame_0', 'M'): ('23:10987', 33.822484574184806),
    ('am3_window_00_frame_0', 'R'): ('30:16137', 44.20775329754375),
    ('am2_window_01_frame_28019', 'B'): ('16:381', 15.32145906187604),
}
FLOAT32_TOLERANCE_PX = 1e-3
ROUNDING_MARGIN_PX = 1e-3
COLUMNS = ['case_id', 'arm', 'pair_id', 'pencils', 'proposed_count', 'retained_count', 'nearest_proposed_px',
           'nearest_proposed_index', 'nearest_proposed_retained', 'nearest_retained_px', 'nearest_retained_index',
           'nearest_retained_px_float64_from_record', 'cap_gain_px', 'cap_discarded_closer_court',
           'courts_closer_than_pool_nearest']


def corner_errors(corners: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Copy of projective_seed.corner_errors: maximum corner distance allowing the 180-degree relabelling."""
    direct = np.linalg.norm(corners - reference, axis=-1).max(axis=-1)
    rotated = np.linalg.norm(corners - reference[[2, 3, 0, 1]], axis=-1).max(axis=-1)
    return np.minimum(direct, rotated)


def read(path: Path) -> dict:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def load_case(case_id: str, arm: str) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    e3_control = read(E3_DIR / f'{case_id}.json.gz')['control']
    control = np.asarray(e3_control['corners_working_px'], dtype=float)
    # Native-to-working scale, for recomputing the accounting's float64 errors from the record shortlists.
    scale = np.asarray(e3_control['working_size'], dtype=float) / np.asarray(e3_control['native_size'], dtype=float)
    record = read(HERE / 'records' / 'new' / arm / 'results' / f'{case_id}.json.gz')
    assert record['working_size'] == e3_control['working_size'], (record['working_size'], e3_control['working_size'])
    pool = np.load(HERE / 'records' / 'new' / arm / 'results' / f'{case_id}_pool.npz')
    return control, scale, record, pool


def pair_errors(pair: dict, pool: dict, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Errors of every proposed court in the pair and the indices the cap retained, after consistency checks."""
    pair_id = pair['pair_id']
    corners = pool[f'pair_{pair_id}_corners'].astype(float)
    retained_index = pool[f'pair_{pair_id}_retained_index']
    # The recording must describe exactly the courts the record's shortlist kept, in order.
    shortlist_positions = [int(entry['candidate_id'].split(':')[1]) for entry in pair['shortlist']]
    assert all(entry['pair_id'] == pair_id for entry in pair['shortlist']), pair_id
    assert retained_index.tolist() == shortlist_positions, (pair_id, retained_index[:5], shortlist_positions[:5])
    assert corners.shape == (pair['role']['geometry_players'], 4, 2), (pair_id, corners.shape, pair['role'])
    errors = corner_errors(corners, control) if len(corners) else np.empty(0)
    return errors, retained_index


def analyse(case_id: str, arm: str) -> tuple[list[dict], dict]:
    control, scale, record, pool = load_case(case_id, arm)
    matched = [pair for pair in record['pairs'] if pair['status'] == 'matched']
    assert all(f'pair_{pair["pair_id"]}_corners' not in pool.files for pair in record['pairs'] if pair['status'] != 'matched')
    per_pair = {pair['pair_id']: pair_errors(pair, pool, control) for pair in matched}
    # Nearest retained court across the pool, in float64 from the record shortlists (the accounting's method)
    # and in float32 from the recording.
    best64 = (np.inf, None)
    best32 = (np.inf, None, None)
    for pair in matched:
        errors, retained_index = per_pair[pair['pair_id']]
        if not len(retained_index):
            continue
        native = np.asarray([entry['corners_px'] for entry in pair['shortlist']], dtype=float)
        shortlist_errors = corner_errors(native * scale, control)
        position = int(np.argmin(shortlist_errors))
        if shortlist_errors[position] < best64[0]:
            best64 = (float(shortlist_errors[position]), pair['shortlist'][position]['candidate_id'])
        retained_errors = errors[retained_index]
        position = int(np.argmin(retained_errors))
        if retained_errors[position] < best32[0]:
            best32 = (float(retained_errors[position]), pair['pair_id'], int(retained_index[position]))
    pool_nearest = best64[0]
    rows, closer_courts, best_proposed = [], [], (np.inf, None, None)
    for pair in matched:
        pair_id = pair['pair_id']
        errors, retained_index = per_pair[pair_id]
        retained_set = set(retained_index.tolist())
        row = {'case_id': case_id, 'arm': arm, 'pair_id': pair_id, 'pencils': ' '.join(map(str, pair['pencils'])),
               'proposed_count': len(errors), 'retained_count': len(retained_index)}
        row.update(dict.fromkeys(COLUMNS[6:], None))
        if len(errors):
            nearest = int(np.argmin(errors))
            row.update({'nearest_proposed_px': float(errors[nearest]), 'nearest_proposed_index': nearest,
                        'nearest_proposed_retained': nearest in retained_set})
            if errors[nearest] < best_proposed[0]:
                best_proposed = (float(errors[nearest]), pair_id, nearest)
            beating = np.flatnonzero(errors < pool_nearest - ROUNDING_MARGIN_PX)
            row['courts_closer_than_pool_nearest'] = len(beating)
            closer_courts.extend((pair_id, int(index), float(errors[index]), index in retained_set) for index in beating)
        if len(retained_index):
            retained_errors = errors[retained_index]
            position = int(np.argmin(retained_errors))
            row.update({'nearest_retained_px': float(retained_errors[position]),
                        'nearest_retained_index': int(retained_index[position])})
            native = np.asarray([entry['corners_px'] for entry in pair['shortlist']], dtype=float)
            row['nearest_retained_px_float64_from_record'] = float(corner_errors(native * scale, control).min())
            row['cap_gain_px'] = row['nearest_retained_px'] - row['nearest_proposed_px']
            row['cap_discarded_closer_court'] = row['cap_gain_px'] > ROUNDING_MARGIN_PX
        rows.append(row)
    summary = {'case_id': case_id, 'arm': arm, 'matched_pairs': len(rows),
               'proposed_total': sum(row['proposed_count'] for row in rows),
               'retained_total': sum(row['retained_count'] for row in rows),
               'nearest_proposed': best_proposed, 'nearest_retained_float32': best32, 'nearest_retained_float64': best64,
               'pairs_with_cap_gain': sum(bool(row['cap_discarded_closer_court']) for row in rows),
               'closer_courts': sorted(closer_courts, key=lambda court: court[2])}
    return rows, summary


def report(summary: dict, expected_id: str, expected_px: float) -> bool:
    proposed_px, proposed_pair, proposed_index = summary['nearest_proposed']
    retained_px, retained_pair, retained_index = summary['nearest_retained_float32']
    retained64_px, retained64_id = summary['nearest_retained_float64']
    print(f'\n{summary["case_id"]} arm {summary["arm"]}: {summary["matched_pairs"]} matched pairs, '
          f'{summary["proposed_total"]} proposed, {summary["retained_total"]} retained')
    print(f'  nearest proposed court   {proposed_px:.6f} px  candidate {proposed_pair}:{proposed_index}  (pair {proposed_pair})')
    print(f'  nearest retained court   {retained_px:.6f} px  candidate {retained_pair}:{retained_index}  (pair {retained_pair}); '
          f'float64 from the record shortlists {retained64_px:.9f} px candidate {retained64_id}')
    gain = retained64_px - proposed_px
    print(f'  cap gain (nearest retained minus nearest proposed): {gain:.6f} px; '
          f'{"real" if gain > ROUNDING_MARGIN_PX else "within the rounding margin, so no loss at the cap"}')
    courts = summary['closer_courts']
    pairs = sorted({court[0] for court in courts})
    print(f'  proposed courts more than {ROUNDING_MARGIN_PX} px closer than the pool\'s nearest retained court: '
          f'{len(courts)} in {len(pairs)} pairs {pairs}')
    for pair_id, index, error, retained in courts[:20]:
        print(f'    candidate {pair_id}:{index}  {error:.6f} px  {"retained" if retained else "discarded by the cap"}')
    print(f'  pairs where the cap discarded a court closer than that pair\'s own nearest retained court: '
          f'{summary["pairs_with_cap_gain"]} of {summary["matched_pairs"]}')
    exact = retained64_id == expected_id and abs(retained64_px - expected_px) < 1e-9
    close = abs(retained_px - expected_px) < FLOAT32_TOLERANCE_PX
    same_id = f'{retained_pair}:{retained_index}' == expected_id
    print(f'  GATE 2 accounting nearest pooled {expected_px:.3f} px candidate {expected_id}: '
          f'float64 from record {"PASS" if exact else "FAIL"}; float32 recording value {"PASS" if close else "FAIL"} '
          f'(gap {abs(retained_px - expected_px):.2e} px); float32 argmin candidate '
          f'{"same" if same_id else f"{retained_pair}:{retained_index}, a tie on the worst corner broken by ordering"}')
    return exact and close


def main() -> None:
    all_rows, gate_ok = [], True
    for (case_id, arm), (expected_id, expected_px) in ACCOUNTING.items():
        rows, summary = analyse(case_id, arm)
        all_rows.extend(rows)
        gate_ok = report(summary, expected_id, expected_px) and gate_ok
    with open(HERE / 'table.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nwrote {HERE / "table.csv"} with {len(all_rows)} rows')
    print('GATE 2', 'PASS' if gate_ok else 'FAIL')
    sys.exit(0 if gate_ok else 1)


if __name__ == '__main__':
    main()
