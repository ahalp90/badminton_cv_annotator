"""Compare frozen GX0 directions with the visually approved supplied-direction court."""

from pathlib import Path

import numpy as np
from diagnose_automatic import diagnose as diagnose_pool
from diagnose_direction_bank import diagnose as diagnose_bank
from diagnose_directions import control_fit
from run_diagnosis import read, write


def main() -> None:
    base = Path('automatic_axes_20260914')
    output = base / 'gx0_control'
    output.mkdir(exist_ok=True)
    case_id = 'gxBQ_window_00_frame_0'
    given = read(Path('axis_matching_20260914/given_finite') / f'{case_id}.json.gz')
    automatic = read(base / 'all_camera' / f'{case_id}.json.gz')
    approved = next(entry for entry in given['entries'] if entry['candidate_id'] == 89)
    target = {'control_corners_px': approved['corners_px'],
              'given_direction_source': 'visually_approved_GX0_supplied_direction_line_winner_89'}
    write(output / 'approved_control.json.gz', target)
    inputs = read(Path('gx_extension/inputs.json.gz'))
    source = next(source for source in inputs['cases'] if source['id'] == case_id)
    native_size = (source['dimensions']['width'], source['dimensions']['height'])
    scale = np.asarray(native_size) / automatic['working_size']
    corners = np.asarray(approved['corners_px']) / scale
    points = np.asarray(automatic['estimator']['points_working'])
    fixed_fits = []
    for pair in automatic['pairs']:
        fit = control_fit(points[pair['pencils']], corners)
        fixed_fits.append({'pair_id': pair['pair_id'], 'pencils': pair['pencils'],
                           'generation_status': pair['status'], **fit})
    bank = diagnose_bank(automatic, target, native_size)
    write(output / 'bank_diagnosis.json.gz', bank)
    pool = diagnose_pool(source, inputs['references'][case_id], automatic, target)
    result = {'case_id': case_id, 'label_guided_diagnostics': True,
              'approved_source': target['given_direction_source'], 'approved_entry': approved,
              'frozen_direction_fits': fixed_fits, 'pool_diagnosis': pool}
    write(output / 'diagnosis.json.gz', result)
    best_fixed = min(fixed_fits, key=lambda record: record['max_corner_working_px'])
    print('fixed best', best_fixed['pair_id'], best_fixed['max_corner_working_px'], flush=True)
    print('bank best', bank['best_measured_fit'], flush=True)
    print('nearest pool', pool['before_global_cap'], pool['after_camera_check'], flush=True)
    print('winner control distances', {key: value['control_working']
                                       for key, value in pool['winners'].items()}, flush=True)


if __name__ == '__main__':
    main()
