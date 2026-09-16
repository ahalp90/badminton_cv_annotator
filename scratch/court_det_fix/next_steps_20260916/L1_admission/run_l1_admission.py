"""Run the fixed-budget score-plus-diversity admission probe on six pairs.

This is an exploratory replay. It preserves the frozen direction pair, the original axis
matching rules, and the 512-row budget. Arm A keeps the original score-ordered distinct rows;
arm B admits the same distinct rows with the fixed quantile-cell selector from the follow-up
pack. The control court is read only after both arms have locked their axis IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
COURT_DET_FIX = HERE.parents[1]
REPO = HERE.parents[3]
LINE_IDENTITY = COURT_DET_FIX / 'line_identity'
FOLLOW_UPS = COURT_DET_FIX / 'worklog/webui_further_followups_16092026'
CAMERA_HELPERS = COURT_DET_FIX / 'worklog/checks/independent/player_guided/20260908'
WEBUI_SEED = COURT_DET_FIX / 'next_steps_20260916/webui_seed'

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LINE_IDENTITY))
sys.path.insert(0, str(FOLLOW_UPS / 'helpers'))

from shared import add_helper_paths

add_helper_paths()

from projective_seed import AxisMatches, Settings, basis_for, combine, match_axis
from retention_probe_reference import select_score_plus_strata
from run_given import canonicalise, finite_scores
from run_population import prepare
from scan_population import geometry, retain

from experiments.annotator.independent_court import assignment, detector
from shared import (
    BASELINE_GENERATION,
    DIRECTION_RUN,
    LABELS,
    arm_points,
    control_corners,
    corner_errors,
    feet_working,
    frame_path,
    load_source,
    read,
)

KEEP_AXES = 512
CORE_AXES = 256
STRATA_BINS = 16
SHORTLIST_CAP = 256
# Keep the existing player-mask calculation below its large transient array size.
HORIZONTAL_CHUNK = 8
AXIS_NAMES = ('horizontal', 'vertical')
AXIS_COORDINATES = (detector.X_COORDS, detector.Y_COORDS)
EXPERIMENT_ARMS = ('A', 'B')
STAGES = ('combined', 'geometry_player', 'per_pair_shortlist')
MATCHER_SETTINGS = Settings(keep_axes=KEEP_AXES)
UNCAPPED_SETTINGS = replace(MATCHER_SETTINGS, keep_axes=10**9)
SHORTLIST_SETTINGS = replace(
    detector.DEFAULT_SETTINGS,
    keep_candidates=SHORTLIST_CAP,
    distinct_corner_distance=2.,
)


@dataclass(frozen=True)
class PairSpec:
    case_id: str
    direction_arm: str
    pair_id: int


PAIR_SPECS = (
    PairSpec('am3_window_00_frame_0', 'R', 43),
    PairSpec('am3_window_00_frame_0', 'B', 43),
    PairSpec('gxBQ_window_00_frame_0', 'B', 143),
    PairSpec('gxBQ_window_00_frame_0', 'B', 22),
    PairSpec('am2_window_01_frame_28019', 'B', 15),
    PairSpec('shuttleset_03_scene_0019', 'B', 1),
)


CSV_COLUMNS = [
    'case_id', 'label', 'direction_arm', 'pair_id', 'pencils', 'experimental_arm',
    'direction_fit_px', 'horizontal_distinct', 'vertical_distinct', 'selected_horizontal',
    'selected_vertical', 'horizontal_core_preserved', 'vertical_core_preserved',
    'combined_count', 'geometry_valid_count', 'geometry_player_count', 'per_pair_shortlist_count',
    'nearest_combined_px', 'nearest_geometry_player_px', 'nearest_per_pair_shortlist_px',
    'combined_axis_ids', 'geometry_player_axis_ids', 'per_pair_shortlist_axis_ids',
    'original_axis_check', 'original_shortlist_check', 'pair_elapsed_s',
]


def current_commit() -> str:
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def load_axis_table() -> dict[tuple[str, str, int, str], dict[str, str]]:
    path = LINE_IDENTITY / 'runs/axis_replay/table.csv'
    if not path.exists():
        return {}
    with open(path, newline='') as stream:
        return {
            (row['case_id'], row['arm'], int(row['pair_id']), row['axis']): row
            for row in csv.DictReader(stream)
        }


def saved_record_path(case_id: str, direction_arm: str) -> Path | None:
    if direction_arm == 'B':
        baseline = BASELINE_GENERATION / f'{case_id}.json.gz'
        if baseline.exists():
            return baseline
    direction_record = DIRECTION_RUN / 'e4' / direction_arm / 'results' / f'{case_id}.json.gz'
    return direction_record if direction_record.exists() else None


def saved_pair(record: dict[str, Any], pair_id: int) -> dict[str, Any]:
    for pair in record.get('pairs', []):
        if pair.get('pair_id') == pair_id:
            if pair.get('status') != 'matched':
                raise ValueError(f'pair {pair_id} is not matched: {pair.get("status")}')
            return pair
    raise KeyError(f'pair {pair_id} not found in saved record')


def axis_diagnostics_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    # The replay uses an uncapped Settings object so the distinct population stays in memory.
    # Its `axis_cap_excluded` diagnostic is therefore zero by construction; compare the shared
    # population counts and the retained IDs instead.
    fields = ('enumerated', 'pattern_supported', 'pattern_and_players', 'distinct_assignments')
    return all(actual.get(field) == expected.get(field) for field in fields)


def verify_original_axes(
    spec: PairSpec,
    matches: tuple[AxisMatches, AxisMatches],
    axis_table: dict[tuple[str, str, int, str], dict[str, str]],
    record_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    """Check the unchanged score-ordered 512 against saved evidence where it exists."""
    path = saved_record_path(spec.case_id, spec.direction_arm)
    saved = None
    pair = None
    if path is not None:
        if path not in record_cache:
            record_cache[path] = read(path)
        saved = record_cache[path]
        pair = saved_pair(saved, spec.pair_id)

    axis_checks = {}
    for axis, name in enumerate(AXIS_NAMES):
        matching = matches[axis]
        original = matching.distinct[:KEEP_AXES]
        table_row = axis_table.get((spec.case_id, spec.direction_arm, spec.pair_id, name))
        table_ok = None
        if table_row is not None:
            table_ok = (
                int(table_row['enumerated']) == matching.diagnostics['enumerated']
                and int(table_row['distinct']) == matching.diagnostics['distinct_assignments']
                and int(table_row['kept']) == len(original)
            )
            if not table_ok:
                raise AssertionError(f'{spec} {name}: saved axis table differs from replay')

        record_ok = None
        parameter_ok = None
        diagnostics_ok = None
        if pair is not None:
            expected_axis = pair['role']['axes'][axis]
            expected_entries = expected_axis['entries']
            expected_ids = np.asarray([entry['axis_id'] for entry in expected_entries], dtype=int)
            expected_parameters = np.asarray([entry['parameters'] for entry in expected_entries], dtype=float)
            record_ok = np.array_equal(original, expected_ids)
            parameter_ok = np.allclose(
                matching.parameters[original], expected_parameters, rtol=0., atol=1e-12,
            )
            diagnostics_ok = axis_diagnostics_match(matching.diagnostics, expected_axis['diagnostics'])
            if not (record_ok and parameter_ok and diagnostics_ok):
                raise AssertionError(f'{spec} {name}: original replay differs from saved record')

        if record_ok is True and table_ok is True:
            status = 'matched_record_and_table'
        elif record_ok is True:
            status = 'matched_record'
        elif table_ok is True:
            status = 'matched_table_counts'
        else:
            status = 'not_available'
        axis_checks[name] = {
            'status': status,
            'saved_record': None if path is None else relative_path(path),
            'saved_axis_ids_match': record_ok,
            'saved_parameters_match': parameter_ok,
            'saved_diagnostics_match': diagnostics_ok,
            'saved_axis_table_counts_match': table_ok,
        }
    return axis_checks


def prepare_pair(
    spec: PairSpec,
    axis_table: dict[tuple[str, str, int, str], dict[str, str]],
    record_cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    """Replay both axis selectors before reading the control used for evaluation."""
    source = load_source(spec.case_id)
    segments, _, size = prepare(source)
    observations = assignment.prepare_observations(segments, size)
    feet = feet_working(source, size)
    points = arm_points(spec.case_id, spec.direction_arm)
    e3 = read(DIRECTION_RUN / 'e3' / f'{spec.case_id}.json.gz')
    fit = e3['sets'][spec.direction_arm]['fits']['records'][spec.pair_id]
    pencils = np.asarray(fit['groups'], dtype=int)
    basis, basis_details = basis_for(points[pencils], size, MATCHER_SETTINGS)
    if basis is None or basis_details['status'] != 'valid':
        raise ValueError(f'{spec}: invalid basis {basis_details}')

    matches = []
    selections = {}
    for axis, coordinates in enumerate(AXIS_COORDINATES):
        matching = match_axis(basis, axis, coordinates, observations, size, UNCAPPED_SETTINGS, feet)
        original = matching.distinct[:KEEP_AXES].copy()
        diverse = select_score_plus_strata(
            matching.parameters, matching.scores, matching.distinct, coordinates,
            keep=KEEP_AXES, core=CORE_AXES, bins=STRATA_BINS,
        )
        matches.append(matching)
        selections['horizontal' if axis == 0 else 'vertical'] = {
            'A': original,
            'B': diverse,
        }

    axis_checks = verify_original_axes(spec, (matches[0], matches[1]), axis_table, record_cache)
    locked_ids = {
        arm: [selections[name][arm].copy() for name in AXIS_NAMES]
        for arm in EXPERIMENT_ARMS
    }
    # The control is intentionally loaded only after both arms have fixed their axis IDs.
    control, control_record = control_corners(spec.case_id)
    return {
        'spec': spec,
        'source': source,
        'size': size,
        'observations': observations,
        'feet': feet,
        'basis': basis,
        'basis_details': basis_details,
        'matches': (matches[0], matches[1]),
        'locked_ids': locked_ids,
        'control': control,
        'control_record': control_record,
        'fit': fit,
        'pencils': pencils,
        'axis_checks': axis_checks,
    }


def selection_summary(matches: AxisMatches, selected: np.ndarray) -> dict[str, Any]:
    core = matches.distinct[:CORE_AXES]
    return {
        'distinct_count': len(matches.distinct),
        'selected_count': len(selected),
        'core_count': len(core),
        'core_preserved': bool(set(core.tolist()).issubset(set(selected.tolist()))),
        'negative_scale_selected': int(np.count_nonzero(matches.parameters[selected, 0] < 0)),
        'positive_scale_selected': int(np.count_nonzero(matches.parameters[selected, 0] > 0)),
        'score_first': None if not len(selected) else float(matches.scores[selected[0]]),
        'score_last': None if not len(selected) else float(matches.scores[selected[-1]]),
    }


def minimum_witness(
    stage: str,
    local_index: int,
    product_position: int,
    corners: np.ndarray,
    errors: np.ndarray,
    axis_pairs: np.ndarray,
    rotated: np.ndarray,
    matches: tuple[AxisMatches, AxisMatches],
    candidate_details: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    horizontal_id, vertical_id = (int(value) for value in axis_pairs[local_index])
    detail = candidate_details.get(product_position, {})
    return {
        'stage': stage,
        'corner_error_px': float(errors[local_index]),
        'axis_ids': [horizontal_id, vertical_id],
        'parameters': {
            'horizontal': matches[0].parameters[horizontal_id].tolist(),
            'vertical': matches[1].parameters[vertical_id].tolist(),
        },
        'scores': {
            'horizontal': float(matches[0].scores[horizontal_id]),
            'vertical': float(matches[1].scores[vertical_id]),
            'mean': float((matches[0].scores[horizontal_id] + matches[1].scores[vertical_id]) / 2.),
        },
        'corners_working_px': corners[local_index].tolist(),
        'rotated_180': bool(rotated[local_index]),
        'product_position': int(product_position),
        'candidate_id': detail.get('candidate_id'),
        'shortlist_score': detail.get('shortlist_score'),
    }


def consider_minimum(
    best: dict[str, dict[str, Any] | None],
    stage: str,
    indexes: np.ndarray,
    product_offset: int,
    corners: np.ndarray,
    errors: np.ndarray,
    axis_pairs: np.ndarray,
    rotated: np.ndarray,
    matches: tuple[AxisMatches, AxisMatches],
    candidate_details: dict[int, dict[str, Any]],
) -> None:
    if not len(indexes):
        return
    local_index = int(indexes[np.argmin(errors[indexes])])
    witness = minimum_witness(
        stage, local_index, product_offset + local_index, corners, errors, axis_pairs, rotated,
        matches, candidate_details,
    )
    current = best[stage]
    if current is None or witness['corner_error_px'] < current['corner_error_px']:
        best[stage] = witness


def evaluate_selection(
    context: dict[str, Any], experimental_arm: str, zone: object,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the existing geometry, player and greedy shortlist stages to one arm."""
    started = perf_counter()
    spec: PairSpec = context['spec']
    matches: tuple[AxisMatches, AxisMatches] = context['matches']
    selected_h, selected_v = context['locked_ids'][experimental_arm]
    control = context['control']
    best: dict[str, dict[str, Any] | None] = {stage: None for stage in STAGES}
    candidate_details: dict[int, dict[str, Any]] = {}
    candidate_object_details: dict[int, dict[str, Any]] = {}
    candidates: list[detector.Candidate] = []
    geometry_valid_count = 0
    geometry_player_count = 0

    for horizontal_start in range(0, len(selected_h), HORIZONTAL_CHUNK):
        horizontal_ids = selected_h[horizontal_start:horizontal_start + HORIZONTAL_CHUNK]
        horizontal = replace(matches[0], retained=horizontal_ids)
        vertical = replace(matches[1], retained=selected_v)
        homographies, axis_pairs = combine(context['basis'], horizontal, vertical)
        homographies, rotated = canonicalise(homographies)
        valid, corners = geometry(homographies, context['size'])
        one, two = zone.player_fractions(homographies, context['feet'])
        geometry_player = valid & (one == 1) & (two >= .5)
        errors = corner_errors(corners, control)
        geometry_valid_count += int(valid.sum())
        geometry_player_count += int(geometry_player.sum())
        product_offset = horizontal_start * len(selected_v)
        all_indexes = np.arange(len(axis_pairs), dtype=int)
        masked_indexes = np.flatnonzero(geometry_player)
        consider_minimum(
            best, 'combined', all_indexes, product_offset, corners, errors, axis_pairs, rotated,
            matches, candidate_details,
        )
        consider_minimum(
            best, 'geometry_player', masked_indexes, product_offset, corners, errors, axis_pairs, rotated,
            matches, candidate_details,
        )
        combined_scores = finite_scores(
            homographies[masked_indexes],
            context['observations'],
            (horizontal, vertical),
            context['size'],
        ) if len(masked_indexes) else np.asarray([], dtype=float)
        for score_index, local_index in enumerate(masked_indexes):
            product_position = product_offset + int(local_index)
            horizontal_id, vertical_id = (int(value) for value in axis_pairs[local_index])
            candidate = detector.Candidate(
                corners[local_index],
                float(combined_scores[score_index]),
                (0., 0.),
                (0, 0),
            )
            candidate_id = f'{spec.pair_id}:{len(candidates)}'
            candidate_details[product_position] = {
                'candidate_id': candidate_id,
                'axis_ids': [horizontal_id, vertical_id],
                'rotated_180': bool(rotated[local_index]),
                'shortlist_score': float(combined_scores[score_index]),
            }
            candidate_object_details[id(candidate)] = candidate_details[product_position]
            candidates.append(candidate)

    retained = retain(candidates, SHORTLIST_SETTINGS)
    if retained:
        shortlist_corners = np.asarray([candidate.corners_px for candidate in retained], dtype=float)
        shortlist_errors = corner_errors(shortlist_corners, control)
        shortlist_index = int(np.argmin(shortlist_errors))
        shortlist_candidate = retained[shortlist_index]
        shortlist_detail = candidate_object_details[id(shortlist_candidate)]
        shortlist_axis_pairs = np.asarray([shortlist_detail['axis_ids']], dtype=int)
        shortlist_rotated = np.asarray([shortlist_detail['rotated_180']], dtype=bool)
        shortlist_meta = {
            int(shortlist_index): shortlist_detail,
        }
        best['per_pair_shortlist'] = minimum_witness(
            'per_pair_shortlist',
            0,
            0,
            shortlist_corners[[shortlist_index]],
            shortlist_errors[[shortlist_index]],
            shortlist_axis_pairs,
            shortlist_rotated,
            matches,
            shortlist_meta,
        )
        best['per_pair_shortlist']['shortlist_position'] = shortlist_index
    else:
        shortlist_errors = np.asarray([], dtype=float)

    result = {
        'experimental_arm': experimental_arm,
        'selection': {
            'horizontal': selection_summary(matches[0], selected_h),
            'vertical': selection_summary(matches[1], selected_v),
        },
        'counts': {
            'combined': int(len(selected_h) * len(selected_v)),
            'geometry_valid': geometry_valid_count,
            'geometry_player': geometry_player_count,
            'per_pair_shortlist': len(retained),
        },
        'minima': best,
        'elapsed_s': perf_counter() - started,
    }
    internal_shortlist = [
        {
            'candidate_id': candidate_object_details[id(candidate)]['candidate_id'],
            'corners_working_px': candidate.corners_px.tolist(),
        }
        for candidate in retained
    ]
    result['_shortlist_internal'] = internal_shortlist
    return result, internal_shortlist


def verify_original_shortlist(
    spec: PairSpec,
    arm_result: dict[str, Any],
    context: dict[str, Any],
    record_cache: dict[Path, dict[str, Any]],
) -> str:
    if arm_result['experimental_arm'] != 'A':
        return 'not_applicable'
    path = saved_record_path(spec.case_id, spec.direction_arm)
    if path is None:
        return 'not_available'
    if path not in record_cache:
        record_cache[path] = read(path)
    pair = saved_pair(record_cache[path], spec.pair_id)
    expected = pair.get('shortlist', [])
    actual = arm_result['_shortlist_internal']
    ids_match = [row['candidate_id'] for row in expected] == [row['candidate_id'] for row in actual]
    source = context['source']
    size = context['size']
    scale = np.asarray([source['dimensions']['width'], source['dimensions']['height']], dtype=float) / size
    expected_corners = np.asarray([row['corners_px'] for row in expected], dtype=float) / scale
    actual_corners = np.asarray([row['corners_working_px'] for row in actual], dtype=float)
    corners_match = np.allclose(expected_corners, actual_corners, rtol=0., atol=1e-5)
    if not (ids_match and corners_match):
        raise AssertionError(f'{spec}: original per-pair shortlist differs from saved record')
    return 'matched'


def csv_row(context: dict[str, Any], result: dict[str, Any], shortlist_check: str) -> dict[str, Any]:
    spec: PairSpec = context['spec']
    minima = result['minima']
    axis_checks = context['axis_checks']
    axis_statuses = {check['status'] for check in axis_checks.values()}
    if 'matched_record_and_table' in axis_statuses:
        axis_check = 'matched_record_and_table'
    elif 'matched_record' in axis_statuses:
        axis_check = 'matched_record'
    elif 'matched_table_counts' in axis_statuses:
        axis_check = 'matched_table_counts'
    else:
        axis_check = 'not_available'

    def axis_ids(stage: str) -> str:
        witness = minima[stage]
        return '' if witness is None else ':'.join(str(value) for value in witness['axis_ids'])

    return {
        'case_id': spec.case_id,
        'label': LABELS[spec.case_id],
        'direction_arm': spec.direction_arm,
        'pair_id': spec.pair_id,
        'pencils': ' '.join(str(value) for value in context['pencils']),
        'experimental_arm': result['experimental_arm'],
        'direction_fit_px': round(float(context['fit']['max_corner_working_px']), 6),
        'horizontal_distinct': result['selection']['horizontal']['distinct_count'],
        'vertical_distinct': result['selection']['vertical']['distinct_count'],
        'selected_horizontal': result['selection']['horizontal']['selected_count'],
        'selected_vertical': result['selection']['vertical']['selected_count'],
        'horizontal_core_preserved': result['selection']['horizontal']['core_preserved'],
        'vertical_core_preserved': result['selection']['vertical']['core_preserved'],
        'combined_count': result['counts']['combined'],
        'geometry_valid_count': result['counts']['geometry_valid'],
        'geometry_player_count': result['counts']['geometry_player'],
        'per_pair_shortlist_count': result['counts']['per_pair_shortlist'],
        'nearest_combined_px': None if minima['combined'] is None else round(minima['combined']['corner_error_px'], 6),
        'nearest_geometry_player_px': (
            None if minima['geometry_player'] is None else round(minima['geometry_player']['corner_error_px'], 6)
        ),
        'nearest_per_pair_shortlist_px': (
            None if minima['per_pair_shortlist'] is None else round(minima['per_pair_shortlist']['corner_error_px'], 6)
        ),
        'combined_axis_ids': axis_ids('combined'),
        'geometry_player_axis_ids': axis_ids('geometry_player'),
        'per_pair_shortlist_axis_ids': axis_ids('per_pair_shortlist'),
        'original_axis_check': axis_check,
        'original_shortlist_check': shortlist_check,
        'pair_elapsed_s': round(float(result['elapsed_s']), 6),
    }


def draw_overlay(
    output: Path,
    context: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> str | None:
    source = context['source']
    image_path = frame_path(source)
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    frame = cv2.resize(frame, context['size'], interpolation=cv2.INTER_AREA)
    control = np.rint(context['control']).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [control], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'control', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
    colours = {'A': (210, 90, 30), 'B': (180, 60, 180)}
    for arm in EXPERIMENT_ARMS:
        witness = results[arm]['minima']['per_pair_shortlist']
        if witness is None:
            continue
        corners = np.rint(np.asarray(witness['corners_working_px'])).astype(np.int32).reshape(-1, 1, 2)
        colour = colours[arm]
        cv2.polylines(frame, [corners], True, colour, 2, cv2.LINE_AA)
        anchor = tuple(int(value) for value in corners[0, 0])
        cv2.putText(frame, f'{arm} shortlist', anchor, cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2, cv2.LINE_AA)
    overlay_path = output / 'overlays' / f'{context["spec"].case_id}_{context["spec"].direction_arm}_{context["spec"].pair_id}.png'
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(overlay_path), frame):
        raise OSError(f'could not write {overlay_path}')
    return relative_path(overlay_path)


def clean_result(result: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    result.pop('_shortlist_internal', None)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--webui-seed', type=Path, default=WEBUI_SEED)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    started_at = datetime.now(UTC).isoformat()
    axis_table = load_axis_table()
    record_cache: dict[Path, dict[str, Any]] = {}
    sys.path.insert(0, str(CAMERA_HELPERS))
    sys.path.insert(0, str(COURT_DET_FIX / 'frozen_helpers_20260914/legacy'))
    import zone_net

    cv2.setNumThreads(1)
    rows = []
    comparisons = []
    for spec in PAIR_SPECS:
        context = prepare_pair(spec, axis_table, record_cache)
        results = {}
        pair_rows = []
        for experimental_arm in EXPERIMENT_ARMS:
            result, _ = evaluate_selection(context, experimental_arm, zone_net)
            shortlist_check = verify_original_shortlist(spec, result, context, record_cache)
            result['original_shortlist_check'] = shortlist_check
            results[experimental_arm] = result
            pair_rows.append(csv_row(context, result, shortlist_check))
        overlay = draw_overlay(args.output, context, results)
        comparisons.append({
            'case_id': spec.case_id,
            'label': LABELS[spec.case_id],
            'direction_arm': spec.direction_arm,
            'pair_id': spec.pair_id,
            'pencils': context['pencils'].tolist(),
            'direction_fit_px': float(context['fit']['max_corner_working_px']),
            'working_size': list(context['size']),
            'axis_checks': context['axis_checks'],
            'arms': {arm: clean_result(results[arm]) for arm in EXPERIMENT_ARMS},
            'overlay': overlay,
        })
        rows.extend(pair_rows)
        for row in pair_rows:
            print(
                row['case_id'], row['direction_arm'], 'pair', row['pair_id'], row['experimental_arm'],
                'combined', row['combined_count'], 'geometry/player', row['geometry_player_count'],
                'shortlist', row['per_pair_shortlist_count'], 'nearest',
                row['nearest_combined_px'], row['nearest_geometry_player_px'],
                row['nearest_per_pair_shortlist_px'], 'seconds', row['pair_elapsed_s'],
                flush=True,
            )

    elapsed = perf_counter() - started
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
    runtime = {
        'started_utc': started_at,
        'elapsed_s': elapsed,
        'peak_rss_mib': peak_rss_mib,
        'processes': 1,
        'opencv_threads': 1,
    }
    witness = {
        'schema': 'l1-fixed-budget-admission-probe/1',
        'git_commit_at_run': current_commit(),
        'prompt': relative_path(FOLLOW_UPS / 'prompts/L1_fixed_budget_admission_probe.md'),
        'reference_files': {
            'axis_replay': relative_path(LINE_IDENTITY / 'axis_replay.py'),
            'axis_table': relative_path(LINE_IDENTITY / 'runs/axis_replay/table.csv'),
            'retention_helper': relative_path(FOLLOW_UPS / 'helpers/retention_probe_reference.py'),
            'c2_witnesses': relative_path(COURT_DET_FIX / 'next_steps_20260916/C2_traces/witnesses.json'),
        },
        'matcher_settings': asdict(MATCHER_SETTINGS),
        'admission_settings': {'keep': KEEP_AXES, 'core': CORE_AXES, 'bins': STRATA_BINS},
        'stages': {
            'combined': 'all admitted horizontal-by-vertical products',
            'geometry_player': 'existing geometry() and zone_net.player_fractions() masks',
            'per_pair_shortlist': 'existing retain() with keep_candidates=256 and distinct_corner_distance=2',
        },
        'comparisons': comparisons,
        'runtime': runtime,
    }
    table_path = args.output / 'comparison.csv'
    witness_path = args.output / 'witnesses.json'
    write_csv(table_path, rows)
    witness_path.write_text(json.dumps(witness, indent=2) + '\n')
    args.webui_seed.joinpath('tables').mkdir(parents=True, exist_ok=True)
    args.webui_seed.joinpath('tables/l1_admission.csv').write_bytes(table_path.read_bytes())
    args.webui_seed.joinpath('l1_admission_witnesses.json').write_bytes(witness_path.read_bytes())
    print('wrote', relative_path(table_path), 'and', relative_path(witness_path), flush=True)
    print('runtime_s', round(elapsed, 3), 'peak_rss_mib', round(peak_rss_mib, 1), flush=True)


if __name__ == '__main__':
    main()
