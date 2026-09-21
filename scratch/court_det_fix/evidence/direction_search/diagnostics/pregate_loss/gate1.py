"""Gate 1 for the pregate run: the new generation records must equal the saved E4 records.

Reuses compare() from cap_loss/gate_records.py, which checks every pair status, every shortlist
candidate_id list in order, every shortlist corners_px, the final entry IDs, line_winner_id,
paint_winner_id and pooled_candidates. The saved records are the ones cap_loss pulled from the
direction-agreement run. Exits 1 if anything differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP_LOSS = HERE.parent / 'cap_loss'
sys.path.insert(0, str(CAP_LOSS))
from gate_records import compare, read

CASE_ARMS = (
    ('gxBQ_window_00_frame_0', 'M', 'M_gxBQ_window_00_frame_0.json.gz'),
    ('am3_window_00_frame_0', 'R', 'R_am3_window_00_frame_0.json.gz'),
)


def main() -> None:
    run = (HERE / 'run_name.txt').read_text().strip()
    all_ok = True
    for case_id, arm, saved_name in CASE_ARMS:
        new = read(HERE / 'records' / 'new' / arm / 'results' / f'{case_id}.json.gz')
        saved = read(CAP_LOSS / 'records' / 'saved' / saved_name)
        assert new['case_id'] == case_id == saved['case_id'] and new['arm'] == arm == saved['arm'], (new['case_id'], new.get('arm'))
        assert new['run'] == run, (new['run'], run)
        print(f'{case_id} arm {arm}: new run {new["run"]} vs saved {saved["run"]}')
        all_ok = compare(case_id, arm, new, saved) and all_ok
    print('GATE 1', 'PASS' if all_ok else 'FAIL')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
