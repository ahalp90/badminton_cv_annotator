#!/usr/bin/env python3
"""Independent arithmetic on the uploaded W2 table; no pixels, models or fitted weights.

Python 3.10+, standard library. Strict input checksum and row/invariant checks.
This verifies a derived CSV, not the unavailable local run or original pixel trace.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

CSV_SHA256 = '41fd27c0387d5dc9864bde9cd0031a04b8669c37698fca7e5173aaaa15644e61'
REF = 'bb6787adeca9a19b5f1d3c3324d54cfcb2ea6e68'
WITNESS_BLOB = '739af3d1dacc7f3fedb551eab01a2b4bad94fd06'
N = 24
MARKINGS = ['left_doubles','left_singles','centre','centre','right_singles','right_doubles',
            'far_baseline','far_long_service','far_short_service','near_short_service',
            'near_long_service','near_baseline']
PANEL = {
    ('am2_window_01_frame_28019', '184:4123'): 'Am2-28019 false',
    ('am2_window_00_frame_150', '30:33'): 'Am2-150 approved',
    ('shuttleset_03_scene_0019', '165:6702'): 'SS03-19 false',
    ('shuttleset_03_scene_0019', '1:60'): 'SS03-19 usable',
}
INTS = ['interval','observed_pass_stations','stations_with_any_measured_offset',
        'unknown_offset_tests','run_lower_stations','run_upper_stations']
FLOATS = ['legacy_pass_station_fraction','run_lower_fraction','run_upper_fraction',
          'clipped_span_working_px','station_spacing_working_px',
          'lower_path_endpoint_span_working_px']


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def read_rows(path: Path) -> list[dict]:
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == CSV_SHA256, 'Input CSV checksum differs')
    rows = list(csv.DictReader(data.decode('utf-8').splitlines()))
    require(len(rows) == 48, 'Expected 48 interval rows')
    seen = set()
    for r in rows:
        for k in INTS:
            r[k] = int(r[k]) if r[k] else None
        for k in FLOATS:
            r[k] = float(r[k]) if r[k] else None
        flag = r['legacy_interval_pass']
        require(flag in ('', 'True', 'False'), 'Invalid Boolean')
        r['legacy_interval_pass'] = None if not flag else flag == 'True'
        key = (r['case_id'], r['candidate_id'])
        ident = (*key, r['interval'])
        require(key in PANEL and ident not in seen and 0 <= r['interval'] < 12, 'Unexpected identity')
        seen.add(ident)
        require(r['population'] == 'ORIGINAL automatic_all_camera', 'Population changed')
        require(r['marking'] == MARKINGS[r['interval']], 'Interval-to-marking mapping changed')
        if r['status'] == 'geometrically_unavailable':
            require(key == ('shuttleset_03_scene_0019', '165:6702') and r['interval'] in [2,3,6,7,9,10,11],
                    'Unexpected unavailable interval')
            require(all(r[k] is None for k in INTS[1:]+FLOATS+['legacy_interval_pass']),
                    'Missing interval is carrying a numerical observation')
            continue
        require(r['status'] == 'measured', 'Unexpected status')
        hits, known = r['observed_pass_stations'], r['stations_with_any_measured_offset']
        lo, hi, unknown = r['run_lower_stations'], r['run_upper_stations'], r['unknown_offset_tests']
        require(0 <= lo <= hi <= N and lo <= hits <= known <= N, 'Invalid count bounds')
        require(5*(N-known) <= unknown <= 5*N-known, 'Unavailable-test count inconsistent')
        require((unknown != 0 or lo == hi), 'Unknown completion enlarged a fully known interval')
        require(r['legacy_interval_pass'] == (hits >= 10), 'Legacy .4 predicate does not reproduce')
        for field, target in [('legacy_pass_station_fraction',hits/N),('run_lower_fraction',lo/N),
                              ('run_upper_fraction',hi/N),
                              ('station_spacing_working_px',r['clipped_span_working_px']/23),
                              ('lower_path_endpoint_span_working_px',max(0,lo-1)*r['clipped_span_working_px']/23)]:
            require(math.isclose(r[field],target,rel_tol=1e-12,abs_tol=1e-9), f'Inconsistent {field}')
    require(len(seen) == 48, 'Repeated rows')
    return rows


def balanced(rows: list[dict], function) -> Fraction:
    groups = defaultdict(list)
    for r in rows:
        if r['status'] == 'measured':
            groups[r['marking']].append(Fraction(function(r)))
    return sum((sum(v)/len(v) for v in groups.values()), Fraction())/len(groups)


def frac_record(value: Fraction) -> dict:
    return {'exact': str(value), 'value': float(value)}


def analyse(rows: list[dict]) -> dict:
    summaries = []
    for key, label in PANEL.items():
        all_rows = [r for r in rows if (r['case_id'],r['candidate_id']) == key]
        visible = [r for r in all_rows if r['status'] == 'measured']
        summary = {'case_id':key[0], 'candidate_id':key[1], 'label':label,
                   'measured_intervals':len(visible), 'unavailable_intervals':12-len(visible),
                   'observed_pass_stations':sum(r['observed_pass_stations'] for r in visible),
                   'stations_with_any_measured_offset':sum(r['stations_with_any_measured_offset'] for r in visible),
                   'unknown_offset_tests':sum(r['unknown_offset_tests'] for r in visible),
                   'legacy_passing_intervals':sum(r['legacy_interval_pass'] for r in visible),
                   'run_ge_10_lower_intervals':sum(r['run_lower_stations'] >= 10 for r in visible),
                   'run_ge_10_upper_intervals':sum(r['run_upper_stations'] >= 10 for r in visible)}
        metrics = {
            'legacy_profile':lambda r: int(r['legacy_interval_pass']),
            'mean_hit_fraction':lambda r: Fraction(r['observed_pass_stations'],N),
            'mean_run_lower_fraction':lambda r: Fraction(r['run_lower_stations'],N),
            'mean_run_upper_fraction':lambda r: Fraction(r['run_upper_stations'],N),
            'dropin_run_profile_lower':lambda r: int(r['run_lower_stations'] >= 10),
            'dropin_run_profile_upper':lambda r: int(r['run_upper_stations'] >= 10),
        }
        for name, function in metrics.items():
            summary[name] = frac_record(balanced(visible,function))
        summary['diagnostic_not_acceptance'] = True
        summaries.append(summary)
    scene = []
    shared = [0,1,4,5,8]
    for candidate in ['165:6702','1:60']:
        rr = [r for r in rows if r['case_id']=='shuttleset_03_scene_0019'
              and r['candidate_id']==candidate and r['interval'] in shared]
        scene.append({'candidate_id':candidate,'intervals':shared,
                      'pass_stations':[r['observed_pass_stations'] for r in rr],
                      'run_lower_stations':[r['run_lower_stations'] for r in rr],
                      'run_upper_stations':[r['run_upper_stations'] for r in rr],
                      'hit_fraction':frac_record(balanced(rr,lambda r: Fraction(r['observed_pass_stations'],N))),
                      'run_lower_fraction':frac_record(balanced(rr,lambda r: Fraction(r['run_lower_stations'],N))),
                      'run_upper_fraction':frac_record(balanced(rr,lambda r: Fraction(r['run_upper_stations'],N)))})
    selected = [r for r in rows if (r['candidate_id']=='184:4123' and r['interval'] in [10,11])
                or (r['candidate_id']=='30:33' and r['interval'] in [6,7])]
    visible = [r for r in rows if r['status']=='measured']
    ambiguous = [r for r in visible if r['run_lower_stations'] != r['run_upper_stations']]
    spacing = [r['station_spacing_working_px'] for r in visible]
    return {
        'kind':'independent real-data calculation on a user-supplied derived CSV; not a pixel replay',
        'input_sha256':CSV_SHA256,'expected_upstream_witness_blob_if_unmodified_probe':WITNESS_BLOB,
        'upstream_packaging_revision':REF,
        'local_producing_commit_and_dirty_state':'not supplied; CSV does not contain them',
        'candidates':summaries,'same_five_scene19_markings':scene,'paired_intervals':selected,
        'validation':{'rows':len(rows),'measured_rows':len(visible),'unavailable_rows':len(rows)-len(visible),
                      'legacy_pass_rows':sum(r['legacy_interval_pass'] for r in visible),
                      'measured_failure_rows':sum(not r['legacy_interval_pass'] for r in visible),
                      'unknown_offset_tests':sum(r['unknown_offset_tests'] for r in visible),
                      'rows_with_unknown_offsets':sum(r['unknown_offset_tests']>0 for r in visible),
                      'rows_with_run_bound_gap':len(ambiguous),
                      'max_run_bound_gap':max(r['run_upper_stations']-r['run_lower_stations'] for r in visible),
                      'csv_arithmetic_checks':'passed', 'pixel_or_mask_replay':'not performed'},
        'sampling':{'spacing_min_working_px':min(spacing),'spacing_max_working_px':max(spacing),
                    'allowed_offset_step_px':2,
                    'equivalent_max_step_angle_min_deg':math.degrees(math.atan2(2,max(spacing))),
                    'equivalent_max_step_angle_max_deg':math.degrees(math.atan2(2,min(spacing))),
                    'caution':'Different station spacing changes lattice connectivity strictness; no continuity between samples is proved.'},
        'not_inferred':['physical pixel ownership','raw fragment IDs','new visual approvals',
                        'calibrated acceptance threshold','full-pool ranking improvement'],
    }


def write_results(result: dict, output: Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    (output/'fresh_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    flat = []
    for row in result['candidates']:
        flat.append({k:(v['value'] if isinstance(v,dict) and 'value' in v else v) for k,v in row.items()})
    with (output/'candidate_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    with (output/'paired_intervals.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['paired_intervals'][0])); writer.writeheader();writer.writerows(result['paired_intervals'])


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).parent/'inputs/real_interval_probe.csv')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args()
    result=analyse(read_rows(args.input)); write_results(result,args.output)
    print(json.dumps({'validation':result['validation'],'sampling':result['sampling'],
                      'summary':[{'candidate_id':r['candidate_id'],
                                  'connected_run_dropin':r['dropin_run_profile_lower'],
                                  'continuous_run_mean':r['mean_run_lower_fraction']} for r in result['candidates']]},indent=2))

if __name__=='__main__':
    main()
