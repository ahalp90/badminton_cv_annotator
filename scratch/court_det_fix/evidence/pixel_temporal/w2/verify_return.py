#!/usr/bin/env python3
"""Independent audit of the returned two-pair W2 atlas; no repository writes/imports.

Run: python verify_return.py --evidence evidence --csv inputs/real_interval_probe.csv --out results

This replays FOUR intervals from the two returned working PNGs. It does not claim
access to the full pinned source frames or the 48-interval L2 JSON. Its direct
bilinear interpolation uses the saved float32 map coordinates, rather than silently
substituting the interpolation behaviour of the installed cv2.remap implementation.
No court scoring, extraction, new geometry, threshold fitting or acceptance occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REF = 'bb6787adeca9a19b5f1d3c3324d54cfcb2ea6e68'
OFFSETS = [-4, -2, 0, 2, 4]
TARGETS = [('184_4123', 'am2_window_01_frame_28019', '184:4123', [10, 11]),
           ('30_33', 'am2_window_00_frame_150', '30:33', [6, 7])]
KEYS = ['tested_sample_coordinates_working_px', 'minus_side_coordinates_working_px',
        'plus_side_coordinates_working_px']
CONTRAST_KEYS = ['contrast_centre_minus_minus_side', 'contrast_centre_minus_plus_side']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def bilinear(gray: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """Direct bilinear interpolation at float32 map coordinates; replicate border.

    Accumulate in float64. This differs slightly in the last bits from the recorded
    float32 runtime; 1e-4 is a REPLAY tolerance, not an image/court acceptance rule.
    """
    xy = np.asarray(coordinates, dtype=np.float32).astype(np.float64)
    lower = np.floor(xy).astype(np.int64)
    f = xy - lower
    h, w = gray.shape
    x0, y0 = np.clip(lower[..., 0], 0, w - 1), np.clip(lower[..., 1], 0, h - 1)
    x1, y1 = np.clip(lower[..., 0] + 1, 0, w - 1), np.clip(lower[..., 1] + 1, 0, h - 1)
    fx, fy = f[..., 0], f[..., 1]
    return ((1-fx)*(1-fy)*gray[y0, x0] + fx*(1-fy)*gray[y0, x1]
            + (1-fx)*fy*gray[y1, x0] + fx*fy*gray[y1, x1])


def inside(xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)


def longest_run(mask: np.ndarray) -> int:
    """Adjacent stations, at most one offset-grid step; unknowns handled by caller."""
    current = np.zeros(mask.shape[1], dtype=int)
    best = 0
    for row in np.asarray(mask, dtype=bool):
        previous = np.pad(current, (1, 1))
        current = np.where(row, 1 + np.maximum.reduce([previous[:-2], previous[1:-1], previous[2:]]), 0)
        best = max(best, int(current.max(initial=0)))
    return best


def footprint(gray: np.ndarray, point: np.ndarray) -> list[dict]:
    """Contributing integer pixels and weights for direct bilinear interpolation."""
    xy = np.asarray(point, dtype=np.float32).astype(float)
    low = np.floor(xy).astype(int)
    fx, fy = xy - low
    h, w = gray.shape
    weights: dict[tuple[int, int], float] = {}
    for dx, dy, weight in [(0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)), (0,1,(1-fx)*fy), (1,1,fx*fy)]:
        x = int(np.clip(low[0] + dx, 0, w - 1))
        y = int(np.clip(low[1] + dy, 0, h - 1))
        if weight > 0:
            weights[(x, y)] = weights.get((x, y), 0.) + float(weight)
    return [{'x': x, 'y': y, 'weight': weights[(x,y)], 'gray': float(gray[y,x])}
            for x, y in sorted(weights)]


def footprint_key(record: dict) -> tuple:
    return tuple(tuple((p['x'], p['y']) for p in fp) for fp in record['footprints_centre_minus_plus'])


def image(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError(f'Could not decode {path}')
    return value


def verify_crop(work: np.ndarray, folder: Path, trace: dict) -> dict:
    crop = image(folder/'native_pair_raw.png')
    lo = np.array(trace['native_crop_origin_xy'], dtype=int)
    hi = np.array(trace['native_crop_end_exclusive_xy'], dtype=int)
    if list(trace['native_per_working_xy']) != [2., 2.]:
        raise ValueError('This bounded audit expects the recorded 2x native scale')
    if crop.shape[:2] != (hi[1]-lo[1], hi[0]-lo[0]):
        raise ValueError('Native crop extent/size mismatch')
    # Align on GLOBAL even native coordinates: crop origins are odd in this packet.
    start, end = ((lo+1)//2)*2, (hi//2)*2
    native_part = crop[start[1]-lo[1]:end[1]-lo[1], start[0]-lo[0]:end[0]-lo[0]]
    reduced = cv2.resize(native_part, tuple(((end-start)//2).tolist()), interpolation=cv2.INTER_AREA)
    reference = work[start[1]//2:end[1]//2, start[0]//2:end[0]//2]
    residual = int(np.abs(reduced.astype(int)-reference.astype(int)).max(initial=0))
    return {'compared_working_rgb_pixels': int(reference.shape[0]*reference.shape[1]),
            'max_abs_channel_difference': residual, 'identical': residual == 0}


def audit(evidence: Path, csv_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((evidence/'manifest.json').read_text())
    if manifest['pinned_revision'] != REF:
        raise ValueError('Unexpected source revision')
    if digest(csv_path) != manifest['csv_trace_reproduction']['uploaded_csv_sha256']:
        raise ValueError('CSV bytes differ from the returned manifest')
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows_by_key = {(r['case_id'], r['candidate_id'], int(r['interval'])): r for r in rows}
    checks, sample_rows, runtime_changes, matching_footprints, selected_pixels = [], [], [], [], []
    coords_cache: dict[tuple[str, int, int, int], dict] = {}
    interval_count = 0
    for name, case, candidate, ids in TARGETS:
        folder = evidence/name
        trace = json.loads((folder/'pair_trace.json').read_text())
        if trace['case_id'] != case or trace['candidate_id'] != candidate:
            raise ValueError('Candidate identity mismatch')
        pair = trace['original_intervals']
        if [r['interval'] for r in pair] != ids:
            raise ValueError('Unexpected target intervals')
        work = image(folder/'context_raw.png')
        if work.shape[:2] != (540,960):
            raise ValueError('Unexpected working image')
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float64)
        crop_check = verify_crop(work, folder, trace)
        side_entries = []
        interval_checks = []
        for row in pair:
            interval_count += 1
            coordinates = [np.asarray(row[k], dtype=float) for k in KEYS]
            availability = np.logical_and.reduce([inside(xy, gray.shape) for xy in coordinates])
            values = [bilinear(gray, xy) for xy in coordinates]
            contrasts = [values[0]-values[1], values[0]-values[2]]
            minimum = np.minimum(*contrasts)
            passing = availability & (minimum >= 10.)
            stored_pass = np.asarray(row['pass_by_shift'], dtype=bool)
            old_values = [cv2.remap(gray.astype(np.float32), xy[...,0].astype(np.float32),
                                   xy[...,1].astype(np.float32), cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE) for xy in coordinates]
            old_contrasts = [old_values[0]-old_values[1], old_values[0]-old_values[2]]
            old_minimum = np.minimum(*old_contrasts)
            old_passing = availability & (old_minimum >= 10.)
            for s, o in np.argwhere(old_passing != stored_pass):
                runtime_changes.append({'case_id': case, 'candidate_id': candidate, 'interval': row['interval'],
                                        'station_zero_based': int(s), 'offset_working_px': OFFSETS[o],
                                        'recorded_minimum_contrast': float(row['minimum_contrast'][s][o]),
                                        'installed_cv2_minimum_contrast': float(old_minimum[s,o]),
                                        'recorded_pass': bool(stored_pass[s,o]),
                                        'installed_cv2_pass': bool(old_passing[s,o])})
            error = max(float(np.abs(v-np.array(row[k])).max()) for v,k in zip(contrasts, CONTRAST_KEYS))
            old_error = max(float(np.abs(v-np.array(row[k])).max()) for v,k in zip(old_contrasts, CONTRAST_KEYS))
            measured_stations = int(passing.any(axis=1).sum())
            run = longest_run(stored_pass)
            expected = rows_by_key[(case, candidate, row['interval'])]
            csv_equal = (measured_stations == int(expected['observed_pass_stations'])
                         and run == int(expected['run_lower_stations'])
                         and longest_run(stored_pass | ~availability) == int(expected['run_upper_stations']))
            ch = {'interval': row['interval'], 'available_offset_tests': int(availability.sum()),
                  'passing_offset_tests': int(passing.sum()), 'passing_stations': measured_stations,
                  'longest_run': run, 'direct_bilinear_max_abs_contrast_error': error,
                  'direct_bilinear_pass_bits_equal': bool(np.array_equal(passing, stored_pass)),
                  'availability_equal': bool(np.array_equal(availability, row['available'])),
                  'uploaded_csv_four_row_check_equal': csv_equal,
                  'installed_cv2_max_abs_contrast_error': old_error,
                  'installed_cv2_pass_bits_equal': bool(np.array_equal(old_passing, stored_pass))}
            if error > 1e-4 or not ch['direct_bilinear_pass_bits_equal'] or not ch['availability_equal'] or not csv_equal:
                raise AssertionError(ch)
            interval_checks.append(ch)
            entries=[]
            for s in range(24):
                for o in range(5):
                    sample_rows.append({'case_id':case,'candidate_id':candidate,'interval':row['interval'],
                                        'station_zero_based':s,'offset_working_px':OFFSETS[o],
                                        'x':float(coordinates[0][s,o,0]),'y':float(coordinates[0][s,o,1]),
                                        'available':bool(availability[s,o]),'passing':bool(stored_pass[s,o]),
                                        'recorded_minimum_contrast':float(row['minimum_contrast'][s][o]),
                                        'direct_bilinear_minimum_contrast':float(minimum[s,o]),
                                        'installed_cv2_minimum_contrast':float(old_minimum[s,o])})
                    if not stored_pass[s,o]:
                        continue
                    obj={'case_id':case,'candidate_id':candidate,'interval':row['interval'],
                         'station_zero_based':s,'offset_working_px':OFFSETS[o],
                         'centre_xy_working':coordinates[0][s,o].tolist(),
                         'coordinates_centre_minus_plus':np.asarray([xy[s,o] for xy in coordinates]).tolist(),
                         'recorded_minimum_contrast':float(row['minimum_contrast'][s][o]),
                         'footprints_centre_minus_plus':[footprint(gray,xy[s,o]) for xy in coordinates]}
                    entries.append(obj)
                    coords_cache[(candidate,row['interval'],s,OFFSETS[o])]=obj
            side_entries.append(entries)
        for a in side_entries[0]:
            for b in side_entries[1]:
                if footprint_key(a)==footprint_key(b):
                    matching_footprints.append({'candidate_id':candidate,
                                               'note':'Same contributing pixels; different interpolation weights. Not proof of ownership.',
                                               'first':a,'second':b})
        a=np.array(pair[0][KEYS[0]],dtype=float)[[0,-1],2]
        b=np.array(pair[1][KEYS[0]],dtype=float)[[0,-1],2]
        length=np.linalg.norm(a[1]-a[0]); u=(a[1]-a[0])/length; n=np.array([-u[1],u[0]])
        tb=(b-a[0])@u; zb=(b-a[0])@n
        order=np.argsort(tb); tb=tb[order];zb=zb[order]
        common=np.array([max(0.,tb[0]),min(float(length),tb[-1])]); sep=np.interp(common,tb,zb)
        with np.load(folder/'shared_strip.npz', allow_pickle=False) as strip:
            strip_values=bilinear(gray,strip['coordinates_working_px'])
            av=strip['available']
            strip_error=float(np.abs(strip_values[av]-strip['gray_intensity'][av]).max(initial=0))
            chart={'common_span_working_px':common.tolist(),'common_span_length_working_px':float(np.diff(common)[0]),
                   'nominal_separation_in_first_normal_working_px':sep.tolist(),
                   'strip_direct_bilinear_max_abs_intensity_error':strip_error,
                   'strip_unknown_locations':int((~av).sum()),
                   'warning':'Interpolated display samples, not independent observations or additional resolution'}
            if strip_error>1e-4:
                raise AssertionError(chart)
        checks.append({'case_id':case,'candidate_id':candidate,'native_working_consistency':crop_check,
                       'interval_checks':interval_checks,'shared_chart':chart})
    # Selected pixels were identified by inspecting raw images, not classified by this code.
    selected_keys=[('184:4123',10,14,-4),('184:4123',11,13,-4),
                   ('30:33',6,3,4),('30:33',7,4,2),
                   ('30:33',6,20,4),('30:33',7,20,4)]
    selected_pixels=[coords_cache[k] for k in selected_keys]
    result={'schema':'w2-return-pixel-review/1','source_revision':REF,
            'producer_reported_manifest':manifest,
            'review_environment':{'python':sys.version,'numpy':np.__version__,'opencv':cv2.__version__,
                                  'platform':platform.platform()},
            'independently_replayed_intervals':interval_count,'independently_compared_contrast_values':interval_count*24*5*2,
            'pixel_thresholds_changed':False,'new_court_or_ranking_run':False,
            'producer_full_48_row_check':'Reported by manifest; only four rows have full traces in this return packet.',
            'checks':checks,'installed_cv2_borderline_changes':runtime_changes,
            'same_centre_and_flank_footprint_examples':matching_footprints,
            'selected_pixels_for_visual_readout':selected_pixels}
    write_json(output/'verification.json',result)
    with (output/'all_offset_checks.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(sample_rows[0]));writer.writeheader();writer.writerows(sample_rows)
    write_json(output/'evidence_sha256.json',{str(p.relative_to(evidence)):digest(p)
                                             for p in sorted(evidence.rglob('*')) if p.is_file()})
    print(json.dumps({'intervals':interval_count,'contrast_values':result['independently_compared_contrast_values'],
                      'checks_passed':True,'installed_cv2_offset_bit_changes':len(runtime_changes),
                      'same_full_footprint_examples':len(matching_footprints)},indent=2))
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,default=Path(__file__).parent/'evidence')
    parser.add_argument('--csv',type=Path,default=Path(__file__).parent/'inputs/real_interval_probe.csv')
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    args=parser.parse_args()
    cv2.setNumThreads(1)
    audit(args.evidence,args.csv,args.out)


if __name__=='__main__':
    main()
