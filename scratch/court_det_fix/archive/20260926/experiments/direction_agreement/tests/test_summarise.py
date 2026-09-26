"""The loss decomposition reshapes accounting rows without recomputing anything."""

import pytest
from diagnose_matrix import COLUMNS
from summarise import decomposition_rows, markdown

CASE = 'gxBQ_window_00_frame_0'
STAGES = ('results', 'camera_first', 'all_camera')
FITS = [{'case_id': CASE, 'B': 6.8, 'B_svd': 4.0, 'M': 19.6, 'M_svd': 19.9, 'R': 6.2, 'R_svd': 4.2, 'MR': 19.7,
         'MR_svd': 19.8}]


def accounting_row(arm: str, stage: str, **values) -> dict:
    """A diagnosed accounting row with every column present, as diagnose_matrix writes it."""
    row = dict.fromkeys(COLUMNS)
    row.update({'case_id': CASE, 'arm': arm, 'stage': stage, 'status': 'diagnosed', 'pooled': 100,
                'nearest_pre_global_id': '1:1', 'nearest_pre_global_px': 7.0, 'nearest_final_px': 8.0,
                'line_control_working_px': 9.0, 'paint_control_working_px': 10.0})
    row.update(values)
    return row


def fit_row() -> dict:
    row = dict(FITS[0], case='GX0', visually_approved=True)
    for name in FITS[0]:
        if name != 'case_id':
            row[f'{name}_best_finite_converged'] = True
    return row


def test_decomposition_reads_each_quantity_from_its_defining_stage() -> None:
    matcher = [
        accounting_row('B', 'results'),
        accounting_row('B', 'camera_first', nearest_pre_global_camera_eligible_id='2:2',
                       nearest_pre_global_camera_eligible_px=7.5, nearest_final_px=8.5, line_control_working_px=9.5),
        accounting_row('B', 'all_camera', nearest_pre_global_camera_eligible_px=7.5, nearest_final_px=7.5,
                       paint_control_working_px=10.5),
    ]
    rows = decomposition_rows(FITS, matcher)
    assert len(rows) == 1
    row = rows[0]
    assert (row['case'], row['arm'], row['fit_px'], row['fit_svd_px']) == ('GX0', 'B', 6.8, 4.0)
    assert (row['pooled'], row['pooled_nearest_id'], row['pooled_nearest_px']) == (100, '1:1', 7.0)
    assert (row['pooled_camera_nearest_id'], row['pooled_camera_nearest_px']) == ('2:2', 7.5)
    assert [row[f'final_nearest_{stage}_px'] for stage in STAGES] == [8.0, 8.5, 7.5]
    assert [row[f'line_{stage}_px'] for stage in STAGES] == [9.0, 9.5, 9.0]
    assert [row[f'paint_{stage}_px'] for stage in STAGES] == [10.0, 10.0, 10.5]
    assert row['status'] == dict.fromkeys(STAGES, 'diagnosed')


def test_missing_stages_leave_cells_empty_and_render() -> None:
    matcher = []
    for stage in STAGES:
        row = dict.fromkeys(COLUMNS)
        row.update({'case_id': CASE, 'arm': 'R', 'stage': stage, 'status': 'missing'})
        matcher.append(row)
    row = decomposition_rows(FITS, matcher)[0]
    assert row['fit_px'] == 6.2
    assert row['pooled'] is None and row['pooled_nearest_px'] is None and row['paint_all_camera_px'] is None
    text = markdown([], [], [fit_row()], matcher)
    assert '## E4 loss decomposition' in text
    assert '| GX0 | R | 6.200 | 4.200 |  |  |  |  |  |  |  |  |  |  |  |  |' in text


def test_pool_must_agree_across_diagnosed_stages() -> None:
    matcher = [accounting_row('M', 'results'), accounting_row('M', 'camera_first', nearest_pre_global_px=7.1)]
    with pytest.raises(AssertionError):
        decomposition_rows(FITS, matcher)
