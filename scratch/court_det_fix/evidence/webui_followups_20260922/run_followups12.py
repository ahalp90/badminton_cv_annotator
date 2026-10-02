#!/usr/bin/env python3
"""Read-only WEBUI follow-ups 1 then 2. Python 3.10+, NumPy, Pillow.

python run_followups12.py --inputs /path/to/followups12_inputs --output results
Use --task 1 or --task 2 to replay separately; --figures adds exact pixel overlays.
Inputs can be the extracted collector archive or a checkout at the pinned revision.
No detector imports, image scoring, proposal fitting, or network access.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import gzip
import io
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REVISION = 'b47eee0ff43ec7cbebcc2d84eb9e5e98075d4eea'
ROOT = Path('scratch/court_det_fix')
EVAL = ROOT / 'evidence/pixel_temporal/evaluation_20260922'
GX_CSV = Path('data/amateur_court_corners/2026-09-09/hand_corners_landmarks.csv')
CORNERS_CSV = GX_CSV.with_name('hand_corners.csv')
AM_CSV = Path('data/amateur_court_corners/hand_corners_landmarks.csv')
WIDTH, LENGTH, FAR_SERVICE, HALF = 6.1, 13.4, 0.76, 6.7
WORKING = np.array([960., 540.])
CORNERS = np.array([[0., 0.], [WIDTH, 0.], [WIDTH, LENGTH], [0., LENGTH]])
STRIP = np.array([[0., 0.], [WIDTH, 0.], [WIDTH, FAR_SERVICE], [0., FAR_SERVICE]])
SYMMETRIES = {
    'identity': np.eye(3),
    'rotate_180': np.array([[-1., 0., WIDTH], [0., -1., LENGTH], [0., 0., 1.]]),
    'reflect_x': np.array([[-1., 0., WIDTH], [0., 1., 0.], [0., 0., 1.]]),
    'reflect_y': np.array([[1., 0., 0.], [0., -1., LENGTH], [0., 0., 1.]])}
REGIONS = ('all', 'far_backcourt', 'remaining_far_half', 'near_half')
REGION_DEFINITIONS = {'far_backcourt': '0 <= court_y_m <= 0.76',
    'remaining_far_half': '0.76 < court_y_m < 6.7', 'near_half': '6.7 <= court_y_m <= 13.4'}
DISPLAY_SEPARATION_WORK_PX = 1.0  # Diversity for the three diagnostic figures only.


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(obj, allow_nan=False, separators=(',', ':')).encode(), mtime=0))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=keys)
    w.writeheader()
    for row in rows:
        w.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(s.getvalue().encode(), mtime=0))


def frame_number(case: str) -> int:
    return int(case.rsplit('_', 1)[1])


def project(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    z = np.column_stack((points, np.ones(len(points)))) @ h.T
    require(np.isfinite(z).all() and np.all(np.abs(z[:, 2]) > 1e-10), 'Nonfinite/at-horizon projection')
    return z[:, :2] / z[:, 2:3]


def working_alignment(h_native: np.ndarray, target_size: np.ndarray, anchor_size: np.ndarray) -> np.ndarray:
    st = np.diag([*(WORKING / target_size), 1.])
    sa = np.diag([*(WORKING / anchor_size), 1.])
    a = sa @ h_native @ np.linalg.inv(st)
    return a / a[2, 2]


class Cohort:
    def __init__(self, inputs: Path, name: str):
        self.inputs, self.name = inputs, name
        base = inputs / EVAL / 'results' / name
        self.manifest = read_json(base / 'manifest.json')
        self.saved = read_json(base / 'selection.json')
        payload = read_json(base / 'candidates.json.gz')
        self.cases = tuple(self.manifest['scoring_cases'])
        self.anchor = payload['anchor_case']
        self.candidates = payload['candidates']
        self.by_id = {c['court_id']: c for c in self.candidates}
        self.ids = list(self.by_id)
        require(self.ids == sorted(self.ids) and len(self.ids) == payload['candidate_count'], 'Candidate order/count')
        require(payload['coordinate_system'] == 'anchor working pixels at 960x540', 'Unknown candidate space')
        require(len(self.ids) == len(self.cases) * 512, 'Unexpected population size')
        require(Counter((c['origin_case'], c['origin_arm']) for c in self.candidates) ==
                Counter({(f, a): 256 for f in self.cases for a in ('G0', 'G1')}), 'Origin population mismatch')
        self.scores = {}
        for f in self.cases:
            record = read_json(base / 'scores' / f'{f}.json.gz')
            require(record['target_case'] == f and record['observation_arm'] == 'S0', 'Wrong score target/arm')
            require(record['candidate_ids'] == self.ids and list(record['scores']) == self.ids, 'Score ID mismatch')
            require(record['population_identity'] == payload['population_identity'], 'Wrong score population')
            self.scores[f] = record['scores']
            for cid, s in self.scores[f].items():
                c = self.by_id[cid]
                require(cid == '::'.join((c['origin_case'], c['origin_arm'], c['origin_candidate_id'])), 'Qualified ID')
                require(all(s[k] == c[k] for k in ('origin_case', 'origin_arm', 'origin_candidate_id')), 'Score origin')
                if s['status'] == 'ok':
                    require(all(np.isfinite(s[k]) for k in ('line_score', 'paint_profile_score')), 'Nonfinite ok score')
        self.sizes, self.images = {}, {}
        if name == 'gx':
            pack = read_json(inputs / ROOT / 'frozen_views/packs/gx_extension_inputs.json.gz')
            for c in pack['cases']:
                self.sizes[c['id']] = np.array([c['dimensions']['width'], c['dimensions']['height']], float)
                self.images[c['id']] = inputs / ROOT / 'frozen_views/frames/gx' / c['image']
        else:
            for f in self.cases:
                p = inputs / ROOT / 'frozen_views/frames/amateur/am3' / f'frame_{frame_number(f):08d}.png'
                self.images[f] = p
                with Image.open(p) as im:
                    self.sizes[f] = np.array(im.size, float)
        for f, p in self.images.items():
            if p.exists():
                with Image.open(p) as im:
                    require(np.array_equal(im.size, self.sizes[f]), 'Raw image dimension mismatch')
        self.alignments, self.hnative = {}, {}
        for f in self.cases:
            rec = self.manifest['alignment']['records'][f]
            require(rec['status'] in ('anchor', 'ok'), 'Registration not usable')
            self.hnative[f] = np.asarray(rec['homography_target_to_anchor'], float)
            self.alignments[f] = working_alignment(self.hnative[f], self.sizes[f], self.sizes[self.anchor])
        for c in self.candidates:
            h = np.asarray(c['homography_anchor_working'], float)
            require(h.shape == (3, 3) and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-12, 'Invalid H')

    def h(self, cid: str, case: str, native: bool = True) -> np.ndarray:
        h = np.linalg.inv(self.alignments[case]) @ np.asarray(self.by_id[cid]['homography_anchor_working'])
        return np.diag([*(self.sizes[case] / WORKING), 1.]) @ h if native else h

    def eligible(self, cases: tuple[str, ...], origins: set[str] | None = None,
                 arms: set[str] | None = None) -> list[str]:
        return [cid for cid in self.ids
                if (origins is None or self.by_id[cid]['origin_case'] in origins)
                and (arms is None or self.by_id[cid]['origin_arm'] in arms)
                and all(self.scores[f][cid]['status'] == 'ok' for f in cases)]


def choose(ids: list[str], observed_scores: dict, cases: tuple[str, ...]) -> dict:
    """Selection boundary: this function receives observed scores only, never labels."""
    if not ids:
        return {'per_frame_paint': {f: None for f in cases}, 'shared_paint': None, 'shared_line': None}
    paint = np.array([[observed_scores[f][cid]['paint_profile_score'] for f in cases] for cid in ids])
    line = np.array([[observed_scores[f][cid]['line_score'] for f in cases] for cid in ids])
    mp, ml = np.median(paint, axis=1), np.median(line, axis=1)
    return {'per_frame_paint': {f: min(ids, key=lambda cid: (
                -observed_scores[f][cid]['paint_profile_score'], -observed_scores[f][cid]['line_score'], cid))
            for f in cases},
            'shared_paint': ids[min(range(len(ids)), key=lambda j: (-mp[j], -ml[j], ids[j]))],
            'shared_line': ids[min(range(len(ids)), key=lambda j: (-ml[j], ids[j]))]}


def reproduce(c: Cohort) -> dict:
    """Check published counts, both native winners, common winners, arm splits and shared ties."""
    common = c.eligible(c.cases)
    require(len(common) == c.saved['common_union_count'], 'Common count regression')
    tests = 1
    for label, origins in [('native_per_frame', True), ('common_union_per_frame', False)]:
        for f in c.cases:
            ids = c.eligible((f,), {f}) if origins else common
            got = choose(ids, {f: c.scores[f]}, (f,))
            expected = c.saved[label][f]
            require(len(ids) == expected['eligible_count'], f'{label} count {f}')
            require(got['per_frame_paint'][f] == expected['paint']['court_id'], f'{label} paint {f}')
            require(got['shared_line'] == expected['line']['court_id'], f'{label} line {f}')
            tests += 3
    result = choose(common, c.scores, c.cases)
    for policy in ('shared_paint', 'shared_line'):
        require(result[policy] == c.saved['common_union_temporal'][policy]['court_id'], f'{policy} mismatch')
        tests += 1
    for name, arms in [('G0', {'G0'}), ('G1', {'G1'}), ('union', {'G0', 'G1'})]:
        ids = c.eligible(c.cases, arms=arms)
        ex = c.saved['access_arms'][name]
        require(len(ids) == ex['common_union_count'], 'Arm common count')
        got = choose(ids, c.scores, c.cases)
        for policy in ('shared_line', 'shared_paint'):
            require(got[policy] == ex['temporal'][policy]['court_id'], 'Arm shared ID')
        tests += 3
    return {'cohort': c.name, 'candidate_count': len(c.ids), 'common_eligible': len(common),
            'scalar_records': len(c.ids)*len(c.cases), 'exact_count_id_checks': tests,
            'shared_paint': result['shared_paint'], 'shared_line': result['shared_line']}


class Measurements:
    def __init__(self, c: Cohort):
        self.c = c
        path = c.inputs / (GX_CSV if c.name == 'gx' else AM_CSV)
        rows = list(csv.DictReader(path.open()))
        needed = {frame_number(f) for f in c.cases}
        videos = {r['video'] for r in rows}
        videos = [v for v in videos if needed <= {int(r['frame']) for r in rows if r['video'] == v}]
        require(len(videos) == 1, 'Annotation video identity is ambiguous')
        self.video = videos[0]
        self.points, self.labels, self.names = {}, {}, {}
        for f in c.cases:
            rr = [r for r in rows if r['video'] == self.video and int(r['frame']) == frame_number(f)]
            rr.sort(key=lambda r: (float(r['court_y_m']), float(r['court_x_m'])))
            x = np.array([[float(r['court_x_m']), float(r['court_y_m'])] for r in rr])
            y = np.array([[float(r['x_px']), float(r['y_px'])] for r in rr])
            require(len(x) >= 4 and len({tuple(p) for p in x}) == len(x), 'Missing/duplicate court coordinate')
            require(np.isfinite(y).all() and np.all(y >= 0) and np.all(y <= c.sizes[f]), 'Click outside native frame')
            self.points[f], self.labels[f] = x, y
            self.names[f] = [r['landmark'] for r in rr]
        self.symmetry, self.cache = {}, {}
        self.corner_check = {}
        if c.name == 'gx':
            rr = list(csv.DictReader((c.inputs / CORNERS_CSV).open()))
            clicked = [r for r in rr if r['source'] == 'click']
            discrepancies = []
            for r in clicked:
                f = next(f for f in c.cases if frame_number(f) == int(r['frame']))
                distance = float(np.min(np.linalg.norm(
                    self.labels[f] - [float(r['x_px']), float(r['y_px'])], axis=1)))
                if distance > 1e-9:
                    discrepancies.append({'case_id': f, 'corner_idx': r['corner_idx'],
                        'nearest_landmark_distance_native_px': distance})
            self.corner_check = {'corner_table_clicked_rows': len(clicked),
                'extrapolated_corners_excluded': sum(r['source'] == 'extrapolated' for r in rr),
                'corner_table_not_merged': True, 'corner_click_discrepancies': discrepancies,
                'measurement_source': 'landmark CSV only; corner table has small coordinate differences and no court-metre columns'}


    def orientation(self, cid: str) -> str:
        if cid not in self.symmetry:
            f = self.c.by_id[cid]['origin_case']
            h = self.c.h(cid, f)
            # One symmetry for the WHOLE candidate, selected on its origin's clicks AFTER selection.
            self.symmetry[cid] = min(SYMMETRIES, key=lambda s: float(np.sum(
                (project(h @ SYMMETRIES[s], self.points[f]) - self.labels[f])**2)))
        return self.symmetry[cid]

    def h(self, cid: str, f: str) -> np.ndarray:
        return self.c.h(cid, f) @ SYMMETRIES[self.orientation(cid)]

    def measure(self, cid: str, f: str) -> dict:
        key = (cid, f)
        if key in self.cache:
            return self.cache[key]
        x, y = self.points[f], self.labels[f]
        row = {'case_id': f, 'court_id': cid, 'symmetry': self.orientation(cid), 'projection_status': 'ok'}
        h = self.h(cid, f)
        try:
            denominator = np.column_stack((CORNERS, np.ones(4))) @ h[2]
            require(np.all(denominator > 0) or np.all(denominator < 0), 'Court crosses projective horizon')
            delta = project(h, x) - y
            errors = {'native': np.linalg.norm(delta, axis=1),
                      'working': np.linalg.norm(delta * (WORKING / self.c.sizes[f]), axis=1)}
        except ValueError as exc:
            row['projection_status'] = str(exc)
            errors = None
        yy = x[:, 1]
        masks = {'all': np.ones(len(x), bool), 'far_backcourt': yy <= FAR_SERVICE + 1e-9,
                 'remaining_far_half': (yy > FAR_SERVICE + 1e-9) & (yy < HALF), 'near_half': yy >= HALF}
        for region, mask in masks.items():
            row[region+'_n'] = int(mask.sum())
            for unit in ('native', 'working'):
                for statistic in ('max', 'median'):
                    value = None
                    if errors is not None and mask.any():
                        value = float(getattr(np, statistic)(errors[unit][mask]))
                    row[f'{region}_{statistic}_{unit}_px'] = value
        self.cache[key] = row
        return row


def normalise_points(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = p.mean(axis=0)
    scale = np.sqrt(2) / np.mean(np.linalg.norm(p-centre, axis=1))
    t = np.array([[scale, 0., -scale*centre[0]], [0., scale, -scale*centre[1]], [0., 0., 1.]])
    return project(t, p), t


def plane_fit(world: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalized DLT of annotations ONLY; none of the detector courts is fitted or modified."""
    x, tx = normalise_points(world)
    y, ty = normalise_points(pixels)
    rows = []
    for (a, b), (u, v) in zip(x, y):
        rows.extend([[-a, -b, -1., 0., 0., 0., u*a, u*b, u],
                     [0., 0., 0., -a, -b, -1., v*a, v*b, v]])
    _, s, vt = np.linalg.svd(rows, full_matrices=True)
    require(len(s) >= 8 and s[7] > 1e-10*s[0], 'Degenerate annotation plane')
    h = np.linalg.inv(ty) @ vt[-1].reshape(3, 3) @ tx
    h /= h[2, 2]
    den = np.column_stack((CORNERS, np.ones(4))) @ h[2]
    require(np.all(den > 0) or np.all(den < 0), 'Reference court crosses horizon')
    err = np.linalg.norm(project(h, world)-pixels, axis=1)
    return h, {'method': 'Hartley-normalized DLT, all clicked correspondences, equal weights',
               'point_count': len(world), 'rms_native_px': float(np.sqrt(np.mean(err**2))),
               'median_native_px': float(np.median(err)), 'max_native_px': float(max(err)),
               'non_null_condition': float(s[0]/s[7]), 'null_gap_ratio': float(s[7]/s[8]) if len(s)>8 else None,
               'homography_court_to_native': h.tolist()}


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0]*b[1]-a[1]*b[0])


def area(p: np.ndarray) -> float:
    return abs(float(np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1))))/2 if len(p)>2 else 0.


def ccw(p: np.ndarray) -> np.ndarray:
    signed = np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1))
    return p if signed >= 0 else p[::-1]


def clip(subject: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Convex polygon intersection, with no raster resolution dependence."""
    if len(subject) < 3 or len(boundary) < 3:
        return np.empty((0, 2))
    output = list(subject)
    boundary = ccw(boundary)
    for a, b in zip(boundary, np.roll(boundary, -1, axis=0)):
        points, output = output, []
        if not points:
            break
        previous = points[-1]
        dp = cross(b-a, previous-a)
        for current in points:
            dc = cross(b-a, current-a)
            if (dc >= -1e-9) != (dp >= -1e-9):
                output.append(previous+(current-previous)*(dp/(dp-dc)))
            if dc >= -1e-9:
                output.append(current)
            previous, dp = current, dc
    return np.asarray(output).reshape(-1, 2)


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, points)))
    def half(seq):
        result = []
        for p in seq:
            while len(result) >= 2 and cross(np.array(result[-1])-result[-2], np.array(p)-result[-1]) <= 0:
                result.pop()
            result.append(p)
        return result[:-1]
    return np.array(half(pts)+half(pts[::-1]))


def strip_measure(reference: np.ndarray, candidate: np.ndarray, clicks_world: np.ndarray, size: np.ndarray) -> dict:
    w, h = size
    viewport = np.array([[0., 0.], [w, 0.], [w, h], [0., h]])
    ref_strip = clip(project(reference, STRIP), viewport)
    footprint = clip(project(candidate, CORNERS), viewport)
    pred_strip = clip(project(candidate, STRIP), viewport)
    regions = {'viewport': ref_strip,
               'click_hull': clip(ref_strip, project(reference, convex_hull(clicks_world)))}
    result = {}
    for name, region in regions.items():
        overlap = clip(region, footprint)
        band_overlap = clip(region, pred_strip)
        denom = area(region)
        require(denom > 1e-6, 'Reference strip area is degenerate')
        plane_area = area(project(np.linalg.inv(reference), region))
        plane_overlap = area(project(np.linalg.inv(reference), overlap)) if len(overlap) else 0.
        result[name] = {'reference_image_area_native_px2': denom,
            'reference_plane_area_m2': plane_area,
            'missing_from_court_image_fraction': max(0., min(1., 1-area(overlap)/denom)),
            'missing_from_court_plane_fraction': max(0., min(1., 1-plane_overlap/plane_area)),
            'not_in_predicted_strip_image_fraction': max(0., min(1., 1-area(band_overlap)/denom)),
            'reference_polygon_native': region.tolist(), 'court_overlap_polygon_native': overlap.tolist()}
    return result


def projection_checks(c: Cohort, measures: Measurements, ids: list[str]) -> dict:
    max_error, naive_error = 0., 0.
    for f in c.cases:
        for cid in ids:
            h = np.asarray(c.by_id[cid]['homography_anchor_working'])
            reference = np.linalg.inv(c.hnative[f]) @ np.diag([*(c.sizes[c.anchor]/WORKING), 1.]) @ h
            actual = c.h(cid, f)
            pts = measures.points[f]
            max_error = max(max_error, float(np.max(np.abs(project(reference, pts)-project(actual, pts)))))
            wrong = np.diag([*(c.sizes[f]/WORKING), 1.]) @ np.linalg.inv(c.hnative[f]) @ h
            naive_error = max(naive_error, float(np.max(np.linalg.norm(project(wrong, pts)-project(actual, pts), axis=1))))
    require(max_error < 1e-7, 'Native/working algebra disagrees')
    return {'equivalent_transform_max_abs_native_px': max_error,
            'wrong_unconjugated_transform_max_landmark_shift_native_px': naive_error,
            'raw_dimensions': {f: c.sizes[f].tolist() for f in c.cases}, **measures.corner_check}


def task1(c: Cohort, output: Path) -> dict:
    ms = Measurements(c)
    common = c.eligible(c.cases)
    require(len(common) == 278, 'Task 1 must scan exactly 278 occurrences')
    rows = []
    for f in c.cases:
        for policy in ('native_per_frame', 'common_union_per_frame', 'shared_paint'):
            cid = (c.saved['common_union_temporal']['shared_paint']['court_id'] if policy == 'shared_paint'
                   else c.saved[policy][f]['paint']['court_id'])
            rows.append({'policy': policy, **ms.measure(cid, f)})
    write_csv(output / 'task1_per_frame.csv.gz', rows)
    target = next(f for f in c.cases if frame_number(f) == 86088)
    named = [c.anchor+'::G1::143:158', target+'::G1::16:44']
    scan = [dict(ms.measure(cid, target)) for cid in common]
    scan.sort(key=lambda r: (r['all_max_working_px'] if r['all_max_working_px'] is not None else float('inf'),
                            r['all_median_working_px'] or float('inf'), r['court_id']))
    alternatives, all_projections = [], {}
    for rank, row in enumerate(scan, 1):
        row['diagnostic_rank'] = rank
        cid = row['court_id']
        all_projections[cid] = project(ms.h(cid, target), ms.points[target]) * WORKING/c.sizes[target]
        if len(alternatives) < 3 and cid not in named:
            separations = [float(np.max(np.linalg.norm(all_projections[cid]-all_projections[old], axis=1)))
                           for old in alternatives]
            if not separations or min(separations) > DISPLAY_SEPARATION_WORK_PX:
                alternatives.append(cid)
    write_csv(output / 'task1_oracle_scan.csv.gz', scan)
    h_ref, fit = plane_fit(ms.points[target], ms.labels[target])
    strips = {cid: strip_measure(h_ref, ms.h(cid, target), ms.points[target], c.sizes[target])
              for cid in named+alternatives}
    sensitivity = {cid: defaultdict(list) for cid in named+alternatives}
    for dropped in range(len(ms.points[target])):
        keep = np.arange(len(ms.points[target])) != dropped
        h_loo, _ = plane_fit(ms.points[target][keep], ms.labels[target][keep])
        for cid in strips:
            # Fix the support polygon across leave-one-out fits; vary only reference-plane fitting.
            value = strip_measure(h_loo, ms.h(cid, target), ms.points[target], c.sizes[target])
            for scope in ('viewport', 'click_hull'):
                sensitivity[cid][scope].append(value[scope]['missing_from_court_plane_fraction'])
    for cid in strips:
        strips[cid]['loo_missing_plane_ranges'] = {k: [min(v), max(v)] for k, v in sensitivity[cid].items()}
    result = {'evidence_revision': REVISION, 'region_definitions': REGION_DEFINITIONS,
        'symmetry_convention': 'one of identity, rotate_180, reflect_x, reflect_y; minimum total squared origin-click residual after selection; fixed across targets; ties in listed order',
        'coordinate_equation': 'H_target_native = inv(S_target) @ inv(S_anchor @ A_native @ inv(S_target)) @ H_anchor_working',
        'annotation_video': ms.video, 'checks': projection_checks(c, ms, named),
        'plane_fit': fit, 'far_strip_definition': 'court rectangle x in [0,6.1], y in [0,0.76]; missing area is reference strip outside candidate FULL court, divided by reference strip area',
        'visibility_limits': 'viewport = image rectangle only, not occlusion mask; click_hull = also clip to convex hull of clicked court coordinates; neither claims exact unobscured pixels',
        'area_sensitivity': 'leave-one-click-out reference DLT refits, fixed support hull; diagnostic range, not statistical confidence interval',
        'named': {cid: ms.measure(cid, target) for cid in named},
        'alternatives': alternatives, 'alternative_metrics': [ms.measure(cid, target) for cid in alternatives],
        'display_separation_work_px': DISPLAY_SEPARATION_WORK_PX, 'strips': strips,
        'alternative_pairwise_max_landmark_separation_work_px': [
            {'first': a, 'second': b, 'distance': float(np.max(np.linalg.norm(
                all_projections[a]-all_projections[b], axis=1)))}
            for a, b in itertools.combinations(alternatives, 2)],
        'oracle_sort': 'all clicked maximum working-pixel error, then median, then court_id',
        'symmetries': {cid: ms.orientation(cid) for cid in common}}
    write_json(output / 'task1_details.json.gz', result)
    compact = []
    for cid in named+alternatives:
        row = dict(ms.measure(cid, target))
        row['role'] = 'named_paint_winner' if cid in named else 'reference_guided_alternative'
        for scope in ('viewport', 'click_hull'):
            row[scope+'_missing_plane_fraction'] = strips[cid][scope]['missing_from_court_plane_fraction']
            row[scope+'_missing_image_fraction'] = strips[cid][scope]['missing_from_court_image_fraction']
        compact.append(row)
    write_csv(output/'task1_gx86088.csv.gz', compact)
    return result


def lock_subsets(c: Cohort) -> list[dict]:
    """Selection uses only origins and statuses/scores of each sampled subset."""
    locks = []
    max_k = 3 if c.name == 'gx' else 2
    for k in range(1, max_k+1):
        for observed in itertools.combinations(c.cases, k):
            admitted = [cid for cid in c.ids if c.by_id[cid]['origin_case'] in observed]
            obs = {f: {cid: c.scores[f][cid] for cid in admitted} for f in observed}
            eligible = [cid for cid in admitted if all(obs[f][cid]['status'] == 'ok' for f in observed)]
            chosen = choose(eligible, obs, observed)
            numbers = [frame_number(f) for f in observed]
            category = ('adjacent_0_5' if c.name == 'gx' and numbers == [0, 5] else
                        'contains_0_5_plus_other' if c.name == 'gx' and 0 in numbers and 5 in numbers else
                        'other_multi' if k > 1 else 'singleton')
            locks.append({'cohort': c.name, 'subset_id': '+'.join(map(str, numbers)), 'sample_count': k,
                'observed': list(observed), 'unobserved': [f for f in c.cases if f not in observed],
                'temporal_category': category, 'frame_span': max(numbers)-min(numbers),
                'proposal_occurrences': len(admitted), 'observed_status_score_cells': len(admitted)*k,
                'eligible_count': len(eligible), 'eligible_by_origin': dict(Counter(c.by_id[cid]['origin_case'] for cid in eligible)),
                'eligible_ids': eligible, 'paint_id_disagreement': len(set(chosen['per_frame_paint'].values())) > 1, **chosen})
    return locks


def choices(lock: dict):
    for f, cid in lock['per_frame_paint'].items():
        yield 'per_frame_paint', f, cid
    for policy in ('shared_paint', 'shared_line'):
        yield policy, 'shared', lock[policy]


def evaluate_locks(c: Cohort, locks: list[dict]) -> tuple[list[dict], list[dict], Measurements]:
    # Labels are loaded only after all choices were locked and written.
    ms = Measurements(c)
    rows, subsets = [], []
    for lock in locks:
        for policy, decision_frame, cid in choices(lock):
            evaluation = []
            for f in c.cases:
                if cid is None:
                    value = {'status': 'empty_pool', 'projection_status': 'not_attempted', 'case_id': f, 'court_id': None}
                else:
                    value = {'status': c.scores[f][cid]['status'], **ms.measure(cid, f)}
                row = {'cohort': c.name, 'subset_id': lock['subset_id'], 'sample_count': lock['sample_count'],
                    'policy': policy, 'decision_frame': decision_frame, 'temporal_category': lock['temporal_category'],
                    'target_role': 'observed' if f in lock['observed'] else 'unobserved', **value}
                rows.append(row)
                evaluation.append(row)
            summary = {k: lock[k] for k in ('cohort', 'subset_id', 'sample_count', 'observed', 'unobserved',
                'temporal_category', 'frame_span', 'proposal_occurrences', 'observed_status_score_cells',
                'eligible_count', 'eligible_by_origin', 'paint_id_disagreement')}
            summary.update(policy=policy, decision_frame=decision_frame, court_id=cid,
                           empty_pool=cid is None, symmetry=ms.orientation(cid) if cid else None)
            for role in ('observed', 'unobserved'):
                rr = [r for r in evaluation if r['target_role'] == role]
                summary[role+'_target_count'] = len(rr)
                summary[role+'_eligibility_failures'] = sum(r['status'] not in ('ok', 'empty_pool') for r in rr)
                summary[role+'_projection_failures'] = sum(r['projection_status'] not in ('ok', 'not_attempted') for r in rr)
                for region in ('all', 'far_backcourt'):
                    key = region+'_max_working_px'
                    vals = [r[key] for r in rr if r['status'] == 'ok' and r.get(key) is not None]
                    summary[f'{role}_{region}_max_working_px_median'] = float(np.median(vals)) if vals else None
                    summary[f'{role}_{region}_max_working_px_worst'] = float(max(vals)) if vals else None
            subsets.append(summary)
    return rows, subsets, ms


def aggregate(subsets: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in subsets:
        groups[r['cohort'], r['sample_count'], r['policy'], 'all_subsets'].append(r)
        groups[r['cohort'], r['sample_count'], r['policy'], r['temporal_category']].append(r)
    output = []
    for (cohort, k, policy, category), rr in sorted(groups.items()):
        out = {'cohort': cohort, 'sample_count': k, 'policy': policy, 'category': category,
               'subsets': len({r['subset_id'] for r in rr}), 'locked_decisions': len(rr),
               'empty_decisions': sum(r['empty_pool'] for r in rr),
               'eligible_pool_min': min(r['eligible_count'] for r in rr),
               'eligible_pool_median': float(np.median([r['eligible_count'] for r in rr])),
               'eligible_pool_max': max(r['eligible_count'] for r in rr)}
        for role in ('observed', 'unobserved'):
            out[role+'_targets'] = sum(r[role+'_target_count'] for r in rr)
            out[role+'_eligibility_failures'] = sum(r[role+'_eligibility_failures'] for r in rr)
            out[role+'_projection_failures'] = sum(r[role+'_projection_failures'] for r in rr)
            for region in ('all', 'far_backcourt'):
                key = f'{role}_{region}_max_working_px_worst'
                values = [r[key] for r in rr if r[key] is not None]
                for q, label in [(0, 'min'), (.5, 'median'), (.9, 'p90'), (1, 'max')]:
                    out[f'{key}_{label}'] = float(np.quantile(values, q)) if values else None
        output.append(out)
    return output


def self_checks() -> dict:
    # Exact ties, paint's secondary line tie, shared-line's ID-only tie.
    fake = {'f1': {'a': {'paint_profile_score': 1., 'line_score': 2.}, 'b': {'paint_profile_score': 1., 'line_score': 3.}},
            'f2': {'a': {'paint_profile_score': 1., 'line_score': 4.}, 'b': {'paint_profile_score': 1., 'line_score': 3.}}}
    result = choose(['b', 'a'], fake, ('f1', 'f2'))
    require(result == {'per_frame_paint': {'f1': 'b', 'f2': 'a'}, 'shared_paint': 'a', 'shared_line': 'a'}, 'Tie rule')
    require(choose([], {}, ())['shared_paint'] is None, 'Empty pool')
    h = np.array([[60., 3., 200.], [1., 20., 100.], [.001, -.005, 1.]])
    grid = np.array(list(itertools.product([0., 1., 3., 6.1], [0., .76, 6.7, 13.4])))
    got, _ = plane_fit(grid, project(h, grid))
    require(np.max(np.abs(project(got, grid)-project(h, grid))) < 1e-8, 'Reference DLT')
    rect = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    require(abs(area(clip(rect, rect+[5., 0.]))-50.) < 1e-9, 'Polygon clipping')
    require(len(clip(rect, np.empty((0, 2)))) == 0, 'Empty polygon clipping')
    for f in SYMMETRIES.values():
        require(np.max(np.abs(f@f-np.eye(3))) < 1e-12, 'Symmetry involution')
    a = np.array([[1., .02, 22.], [.01, 1., -11.], [1e-5, -2e-5, 1.]])
    st, sa = np.array([1280., 720.]), np.array([1920., 1080.])
    wa = working_alignment(a, st, sa)
    p = np.array([[320., 180.], [600., 300.]])
    expected = project(a, p*st/WORKING)*WORKING/sa
    require(np.max(np.abs(project(wa, p)-expected)) < 1e-10, 'Unequal native size conversion')
    return {'checks': ['paint and line exact ties', 'empty pool', 'exact synthetic DLT',
                       'polygon half overlap and empty intersection', 'four whole-court symmetries', 'unequal native dimensions'], 'passed': True}


def observed_access_check(c: Cohort, locks: list[dict]) -> dict:
    # Reproduce each lock through a mapping with no hidden targets or unobserved-origin entries.
    for lock in locks:
        observed = tuple(lock['observed'])
        admitted = [cid for cid in c.ids if c.by_id[cid]['origin_case'] in observed]
        restricted = {f: {cid: c.scores[f][cid] for cid in admitted} for f in observed}
        eligible = [cid for cid in admitted if all(restricted[f][cid]['status']=='ok' for f in observed)]
        require(eligible == lock['eligible_ids'], 'Admission inspected unseen status')
        result = choose(eligible, restricted, observed)
        require(all(result[k] == lock[k] for k in result), 'Selection lock not reproducible with observed access')
    return {'cohort': c.name, 'restricted_access_replays': len(locks), 'passed': True}


def font(size: int):
    for name in ('DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def template_segments() -> list[np.ndarray]:
    lines = [np.array([[x, 0.], [x, LENGTH]]) for x in (0., .46, 5.64, WIDTH)]
    lines += [np.array([[0., y], [WIDTH, y]]) for y in (0., FAR_SERVICE, 4.72, 8.68, 12.64, LENGTH)]
    lines += [np.array([[WIDTH/2, a], [WIDTH/2, b]]) for a, b in ((0., 4.72), (8.68, LENGTH))]
    return lines


def annotated_image(c: Cohort, ms: Measurements, case: str, cid: str,
                    model_strip: np.ndarray | None = None) -> Image.Image:
    image = Image.open(c.images[case]).convert('RGB')
    draw = ImageDraw.Draw(image)
    if model_strip is not None:
        layer = Image.new('RGBA', image.size)
        ld = ImageDraw.Draw(layer)
        ld.polygon([tuple(p) for p in model_strip], fill=(255, 40, 180, 115), outline=(255, 40, 180, 255), width=2)
        image = Image.alpha_composite(image.convert('RGBA'), layer).convert('RGB')
        draw = ImageDraw.Draw(image)
    h = ms.h(cid, case)
    for line in template_segments():
        p = project(h, line)
        draw.line([tuple(x) for x in p], fill=(0, 215, 255), width=2)
    predicted = project(h, ms.points[case])
    for (x, y), (u, v) in zip(ms.labels[case], predicted):
        draw.line([(x, y), (u, v)], fill=(255, 190, 50), width=1)
        draw.ellipse((x-4, y-4, x+4, y+4), outline=(10, 10, 10), width=2)
        draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 190, 50))
        draw.line([(u-4, v), (u+4, v)], fill=(0, 215, 255), width=2)
        draw.line([(u, v-4), (u, v+4)], fill=(0, 215, 255), width=2)
    return image


def comparison_sheet(c: Cohort, ms: Measurements, case: str, ids: list[str], title: str,
                     filename: Path, reference: np.ndarray | None = None) -> None:
    row_height, top = 625, 96
    sheet = Image.new('RGB', (1920, top+row_height*len(ids)), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 10), title, fill='black', font=font(30))
    draw.text((20, 53), 'Cyan: fixed candidate template / predicted crosses. Amber: clicked landmarks. Pink: model strip (not ground truth).', fill='black', font=font(21))
    strip = project(reference, STRIP) if reference is not None else None
    for index, cid in enumerate(ids):
        y = top+row_height*index
        draw.text((20, y), cid, fill='black', font=font(25))
        overlay = annotated_image(c, ms, case, cid, strip)
        sheet.paste(overlay.resize((960, 540), Image.Resampling.LANCZOS), (0, y+40))
        # Fixed crop coordinates across GX candidates, no cherry-picked crop per geometry.
        crop = (90, 520, 890, 730) if c.name == 'gx' else (0, 430, 1000, 670)
        sheet.paste(overlay.crop(crop).resize((960, 252), Image.Resampling.LANCZOS), (960, y+40))
        draw.text((980, y+305), 'Far-end crop; same raw-image coordinates for every candidate', fill='black', font=font(22))
        m = ms.measure(cid, case)
        label = (f"Maximum / median clicked error (working px): {m['all_max_working_px']:.2f} / {m['all_median_working_px']:.2f}\n"
                 f"Far backcourt: {m['far_backcourt_max_working_px']:.2f} / {m['far_backcourt_median_working_px']:.2f}  (n={m['far_backcourt_n']})\n"
                 f"Near half: {m['near_half_max_working_px']:.2f} / {m['near_half_median_working_px']:.2f}  (n={m['near_half_n']})\n"
                 'Native-pixel errors are 2x these values. Candidate geometry unchanged.')
        draw.multiline_text((980, y+350), label, fill='black', font=font(23), spacing=12)
    filename.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(filename)


def make_figures(inputs: Path, output: Path) -> None:
    c = Cohort(inputs, 'gx')
    ms = Measurements(c)
    detail = read_json(output/'task1_details.json.gz')
    target = next(f for f in c.cases if frame_number(f)==86088)
    href = np.asarray(detail['plane_fit']['homography_court_to_native'])
    figures = output/'figures'
    comparison_sheet(c, ms, target, list(detail['named']), 'GX86088 | named paint candidates, far-strip coverage',
                     figures/'01_named_gx86088.png', href)
    comparison_sheet(c, ms, target, detail['alternatives'], 'GX86088 | three reference-guided alternatives; not visual approvals',
                     figures/'02_alternatives_gx86088.png', href)
    comparison_sheet(c, ms, c.anchor, [c.saved['native_per_frame'][c.anchor]['paint']['court_id'],
                     c.saved['common_union_temporal']['shared_paint']['court_id']],
                     'GX0 | native and shared paint; compare with archived saved-selection sheet',
                     figures/'03_projection_gx0.png')
    print('Created three task-1 comparison figures.')


def sparse_control_figure(inputs: Path, output: Path) -> None:
    cohorts = {name: Cohort(inputs, name) for name in ('gx', 'am3')}
    metrics = {name: Measurements(c) for name, c in cohorts.items()}
    picks = [
        ('gx', 'gxBQ_window_00_frame_0', 'gxBQ_window_00_frame_5::G0::181:973', 'Singleton {5}; evaluation on unobserved GX0'),
        ('gx', 'gxBQ_window_00_frame_0', 'gxBQ_window_01_frame_5111::G0::136:541', 'Shared paint {5111,5766}; unobserved GX0'),
        ('gx', 'gxBQ_window_00_frame_0', 'gxBQ_window_01_frame_5111::G0::0:17320', 'Shared paint {5,5111,5766}; unobserved GX0'),
        ('am3', 'am3_window_00_frame_0', 'am3_window_00_frame_0::G1::43:159', 'Am3 singleton {0}; observed frame 0'),
        ('am3', 'am3_window_01_frame_10514', 'am3_window_01_frame_10514::G1::30:325', 'Am3 singleton {10514}; observed frame 10514'),
        ('am3', 'am3_window_00_frame_0', 'am3_window_00_frame_0::G1::43:451', 'Am3 shared two-frame paint; observed frame 0')]
    locks = read_json(output/'task2_locks.json.gz')
    selected = {cid for group in locks.values() for lock in group for _, _, cid in choices(lock)}
    require(all(cid in selected for _, _, cid, _ in picks), 'Figure ID was not locked')
    sheet = Image.new('RGB', (1920, 80+3*670), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 10), 'Sparse-access controls | six locked geometries, no visual approval from scalar scores', fill='black', font=font(29))
    draw.text((20, 47), 'Cyan: fixed candidate. Amber: clicked landmarks. Three GX failures and the Am3 paint contrast.', fill='black', font=font(22))
    for index, (name, case, cid, caption) in enumerate(picks):
        x, y = (index % 2)*960, 80+(index//2)*670
        draw.text((x+12, y), cid, fill='black', font=font(23))
        draw.text((x+12, y+33), caption, fill='black', font=font(23))
        im = annotated_image(cohorts[name], metrics[name], case, cid)
        sheet.paste(im.resize((960, 540), Image.Resampling.LANCZOS), (x, y+68))
        value = metrics[name].measure(cid, case)
        draw.text((x+12, y+618), f"Clicked error on this frame: max {value['all_max_working_px']:.2f}, median {value['all_median_working_px']:.2f} working px", fill='black', font=font(22))
    dest = output/'figures'/'04_sparse_controls.png'
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    print('Created task-2 six-geometry control figure.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--task', choices=('1', '2', 'all'), default='all')
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = {'synthetic': self_checks(), 'evidence_revision': REVISION}
    bundle = args.inputs/'BUNDLE_INFO.json'
    if bundle.exists():
        info = read_json(bundle)
        require(info['evidence_revision'] == REVISION, 'Wrong supplied snapshot')
        require(all((args.inputs/p).stat().st_size == n for p, n in info['files'].items()), 'Collector file sizes differ')
        checks['bundle_files'] = len(info['files'])
    gx = Cohort(args.inputs, 'gx')
    checks['gx_reproduction'] = reproduce(gx)
    if args.task in ('1', 'all'):
        result1 = task1(gx, args.output)
        print('Task 1 complete; alternatives:', result1['alternatives'])
    if args.task in ('2', 'all'):
        am3 = Cohort(args.inputs, 'am3')
        checks['am3_reproduction'] = reproduce(am3)
        # All locks for both cohorts exist on disk before the task-2 evaluation loads annotations.
        locks = {c.name: lock_subsets(c) for c in (gx, am3)}
        write_json(args.output/'task2_locks.json.gz', locks)
        all_evaluations, all_subsets = [], []
        checks['access'] = []
        for c in (gx, am3):
            checks['access'].append(observed_access_check(c, locks[c.name]))
            evaluation, subsets, _ = evaluate_locks(c, locks[c.name])
            all_evaluations.extend(evaluation)
            all_subsets.extend(subsets)
        write_csv(args.output/'task2_evaluations.csv.gz', all_evaluations)
        write_csv(args.output/'task2_subsets.csv.gz', all_subsets)
        write_csv(args.output/'task2_distributions.csv.gz', aggregate(all_subsets))
        print('Task 2 complete:', len(locks['gx']), 'GX subsets;', len(locks['am3']), 'Am3 subsets;',
              len(all_subsets), 'locked decisions;', len(all_evaluations), 'decision/target rows')
    write_json(args.output/'checks.json.gz', checks)
    if args.figures:
        make_figures(args.inputs, args.output)
        if (args.output/'task2_locks.json.gz').exists():
            sparse_control_figure(args.inputs, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
