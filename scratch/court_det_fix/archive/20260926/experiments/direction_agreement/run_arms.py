"""E0-E2: recover merge membership, prove baseline replay, compare anchors and freeze four arms.

One process handles the nine cases in the packet's order. No control geometry is read
here; the arm records are frozen before run_fits.py loads any control.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import vp_pruning
from common import (
    ARMS,
    CASE_IDS,
    SAVED_ESTIMATORS,
    code_md5,
    load_sources,
    md5,
    read,
    run_dir,
    save_array,
    write,
    write_csv,
)
from diagnose_direction_bank import reconstruct_bank
from membership import (
    line_feet,
    longest_member,
    merge_lines_with_membership,
    project_onto_lines,
    residual_matrix,
    to_normalised,
    to_working,
)
from run_population import prepare
from selection import arm_record, retain_pencils_with_buckets, support_arrays

from experiments.annotator.independent_court import assignment, detector

EXPECTED_SETTINGS = {'angle_deg': 1.5, 'direction_lines': 128, 'pencils': 16, 'overlap': 0.8,
                     'pencil_selection': 'coverage'}
ESTIMATE_MERGE_CAP = 300  # vp_pruning.estimate merges with max_family_lines=300 before the 128 cap.
TOP_CHANGES = 20
FLOAT_ATOL = 1e-12
EXACT_ESTIMATOR_FIELDS = (
    'raw_fragments', 'visible_fragments', 'visible_raw_ids', 'preparation_groups', 'merge_input_cap',
    'merge_input_excluded', 'merged_direction_count', 'direction_cap_excluded', 'compatible_raw_ids',
    'pair_candidates', 'infinity_candidates', 'degenerate_candidates', 'candidate_ids', 'support_counts',
    'candidate_status', 'retained_candidate_ids', 'retained_support_masks',
)
FLOAT_ESTIMATOR_FIELDS = ('direction_lines', 'points_working', 'normalised_to_working')


def compare_estimator(replayed: dict, saved: dict) -> None:
    """Require the replayed estimator record to match the saved one field by field."""
    for field in EXACT_ESTIMATOR_FIELDS:
        assert replayed[field] == saved[field], f'estimator field differs: {field}'
    for field in FLOAT_ESTIMATOR_FIELDS:
        np.testing.assert_allclose(replayed[field], saved[field], rtol=0, atol=FLOAT_ATOL, err_msg=field)


def statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float).ravel()
    if not len(values):
        return {'count': 0}
    return {'count': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'p90': float(np.percentile(values, 90)), 'p99': float(np.percentile(values, 99)),
            'max': float(values.max()), 'min': float(values.min())}


def largest_changes(
    residuals_foot: np.ndarray, residuals_midpoint: np.ndarray, masks_foot: np.ndarray, masks_midpoint: np.ndarray,
    candidate_ids: np.ndarray, generators: list[list[int]], finite_distances: np.ndarray | None,
    finite_row_positions: np.ndarray, anchors: list[dict],
) -> list[dict]:
    """Twenty largest absolute residual changes; ties by candidate ID, then line ID."""
    delta = residuals_midpoint - residuals_foot
    magnitude = np.abs(delta)
    candidate_grid = np.broadcast_to(candidate_ids[:, None], delta.shape)
    line_grid = np.broadcast_to(np.arange(delta.shape[1])[None], delta.shape)
    order = np.lexsort((line_grid.ravel(), candidate_grid.ravel(), -magnitude.ravel()))[:TOP_CHANGES]
    rows, columns = np.unravel_index(order, delta.shape)
    records = []
    for row, column in zip(rows, columns, strict=True):
        finite_position = finite_row_positions[row]
        distance = None
        if finite_distances is not None and finite_position >= 0:
            distance = float(finite_distances[finite_position, column])
        records.append({
            'candidate_id': int(candidate_ids[row]),
            'candidate_generators': generators[row],
            'candidate_finite': bool(finite_position >= 0),
            'line_id': int(column),
            'residual_foot_deg': float(residuals_foot[row, column]),
            'residual_midpoint_deg': float(residuals_midpoint[row, column]),
            'change_deg': float(delta[row, column]),
            'member_foot': bool(masks_foot[row, column]),
            'member_midpoint': bool(masks_midpoint[row, column]),
            'anchor_fragment_length_px': anchors[column]['fragment_length_px'],
            'anchor_displacement_px': anchors[column]['displacement_px'],
            'distance_to_midpoint_anchor_px': distance,
        })
    return records


def build_case(case_id: str, source: dict, saved: dict, output: Path, run: str, code: dict[str, str]) -> dict:
    """Run E0, E1 and E2 for one case and write its records atomically."""
    started = perf_counter()
    settings = vp_pruning.Settings(**saved['settings'])
    settings_flags = {name: saved['settings'][name] != value for name, value in EXPECTED_SETTINGS.items()}
    segments, _, size = prepare(source)
    assert list(size) == saved['working_size'], (size, saved['working_size'])
    observations = assignment.prepare_observations(segments, size)
    prepared = perf_counter()

    # E0: membership through the unchanged merge, then a full estimator replay.
    merge_settings = replace(detector.DEFAULT_SETTINGS, max_family_lines=ESTIMATE_MERGE_CAP)
    fragments = observations.segments.reshape(-1, 4)  # (visible fragments, 4) clipped working XYXY
    reference_lines = detector._merge_lines(fragments, merge_settings)
    all_lines, sidecar = merge_lines_with_membership(fragments, merge_settings)
    assert np.array_equal(reference_lines, all_lines), 'instrumented merge changed coefficients or order'
    lines = all_lines[:settings.direction_lines]
    estimator = saved['estimator']
    np.testing.assert_allclose(lines, estimator['direction_lines'], rtol=0, atol=FLOAT_ATOL)
    merged = perf_counter()
    _, replayed = vp_pruning.estimate(segments, size, settings)
    compare_estimator(replayed, estimator)
    replayed_at = perf_counter()

    candidates, transform, generators = reconstruct_bank(estimator)
    candidate_ids = np.asarray(estimator['candidate_ids'], dtype=int)
    normalised_lines = lines @ transform
    feet = line_feet(normalised_lines)
    foot_started = perf_counter()
    residuals_foot = residual_matrix(normalised_lines, candidates, feet, settings.candidate_batch)
    foot_elapsed = perf_counter() - foot_started
    original_blocks = []
    for offset in range(0, len(candidates), settings.candidate_batch):
        original_blocks.append(vp_pruning.angular_residuals(normalised_lines, candidates[offset:offset + settings.candidate_batch]))
    assert np.array_equal(np.concatenate(original_blocks), residuals_foot), 'foot anchor does not reproduce the estimator'
    masks_foot, counts_foot, supports_foot = support_arrays(residuals_foot, settings.angle_deg)
    assert counts_foot.tolist() == estimator['support_counts']
    reference_retained, reference_status = vp_pruning.retain_pencils(masks_foot, counts_foot, supports_foot, candidate_ids, settings)
    selection_started = perf_counter()
    retained_foot, status_foot, buckets_foot = retain_pencils_with_buckets(
        masks_foot, counts_foot, supports_foot, candidate_ids, settings)
    selection_foot_elapsed = perf_counter() - selection_started
    assert retained_foot == reference_retained and np.array_equal(status_foot, reference_status), 'bucket instrumentation changed selection'
    assert candidate_ids[retained_foot].tolist() == estimator['retained_candidate_ids']
    assert masks_foot[retained_foot].tolist() == estimator['retained_support_masks']
    assert status_foot.tolist() == estimator['candidate_status']
    np.testing.assert_allclose(candidates[retained_foot] @ transform.T, estimator['points_working'], rtol=0, atol=FLOAT_ATOL)

    # E1: the longest clipped contributing fragment's midpoint, projected onto the merged line.
    lengths = np.linalg.norm(fragments[:, 2:] - fragments[:, :2], axis=1)
    anchor_rows = np.asarray([longest_member(entry['member_observation_ids'], lengths) for entry in sidecar[:len(lines)]])
    midpoints = fragments[anchor_rows].reshape(-1, 2, 2).mean(axis=1)  # (lines, 2) working px
    anchors_working = project_onto_lines(midpoints, lines)
    anchors_normalised = to_normalised(anchors_working, transform)
    feet_working = to_working(feet, transform)
    displacement = np.linalg.norm(anchors_working - feet_working, axis=1)
    on_line = np.abs(np.einsum('li,li->l', lines[:, :2], anchors_working) + lines[:, 2])
    assert np.all(on_line <= 1e-6 * np.linalg.norm(lines[:, :2], axis=1) * (1 + np.abs(anchors_working).max())), 'anchor left the merged line'
    midpoint_started = perf_counter()
    residuals_midpoint = residual_matrix(normalised_lines, candidates, anchors_normalised, settings.candidate_batch)
    midpoint_elapsed = perf_counter() - midpoint_started
    masks_midpoint, counts_midpoint, supports_midpoint = support_arrays(residuals_midpoint, settings.angle_deg)
    infinite = candidates[:, 2] == 0
    assert np.array_equal(residuals_foot[infinite], residuals_midpoint[infinite]), 'infinity residuals depend on the anchor'
    finite_rows = np.flatnonzero(~infinite)
    finite_row_positions = np.full(len(candidates), -1)
    finite_row_positions[finite_rows] = np.arange(len(finite_rows))
    working_points = candidates[finite_rows] @ transform.T
    cartesian = working_points[:, :2] / working_points[:, 2:]
    finite_distances = np.linalg.norm(cartesian[:, None, :] - anchors_working[None], axis=2)  # (finite candidates, lines)
    anchors = []
    for line_id, row in enumerate(anchor_rows):
        anchors.append({
            'line_id': line_id, 'observation_id': int(row), 'raw_fragment_id': int(observations.fragment_ids[row]),
            'fragment_length_px': float(lengths[row]), 'member_count': len(sidecar[line_id]['member_observation_ids']),
            'midpoint_working_px': midpoints[line_id].tolist(), 'anchor_working_px': anchors_working[line_id].tolist(),
            'anchor_normalised': anchors_normalised[line_id].tolist(), 'foot_working_px': feet_working[line_id].tolist(),
            'displacement_px': float(displacement[line_id]),
            'supporters_foot': int(masks_foot[:, line_id].sum()), 'supporters_midpoint': int(masks_midpoint[:, line_id].sum()),
        })
    gained = (masks_midpoint & ~masks_foot).sum(axis=1)
    lost = (masks_foot & ~masks_midpoint).sum(axis=1)
    change = residuals_midpoint - residuals_foot
    # Most entries are far outside the 1.5-degree membership band under both anchors; the
    # band-relevant statistics restrict to entries that are members under at least one anchor.
    band_relevant = masks_foot | masks_midpoint
    e1_summary = {
        'residual_change_abs_deg': statistics(np.abs(change)),
        'residual_change_abs_deg_finite_only': statistics(np.abs(change[finite_rows])),
        'residual_change_abs_deg_member_under_either_anchor': statistics(np.abs(change[band_relevant])),
        'entries_member_under_either_anchor': int(band_relevant.sum()),
        'membership_entries_changed': int((masks_midpoint != masks_foot).sum()),
        'membership_entries_total': int(masks_foot.size),
        'candidates_gaining_support': int((gained > 0).sum()),
        'candidates_losing_support': int((lost > 0).sum()),
        'candidates_count_changed': int((counts_midpoint != counts_foot).sum()),
        'eligible_foot': int((counts_foot >= 2).sum()),
        'eligible_midpoint': int((counts_midpoint >= 2).sum()),
        'anchor_displacement_px': statistics(displacement),
        'anchor_fragment_length_px': statistics(lengths[anchor_rows]),
        'anchor_fragment_is_first_member': int(sum(row == entry['member_observation_ids'][0]
                                                   for row, entry in zip(anchor_rows, sidecar[:len(lines)], strict=True))),
    }
    top_changes = largest_changes(residuals_foot, residuals_midpoint, masks_foot, masks_midpoint, candidate_ids,
                                  generators, finite_distances, finite_row_positions, anchors)

    # E2: allocation per anchor with the unchanged coverage rule, then representatives.
    selection_started = perf_counter()
    retained_midpoint, status_midpoint, buckets_midpoint = retain_pencils_with_buckets(
        masks_midpoint, counts_midpoint, supports_midpoint, candidate_ids, settings)
    selection_midpoint_elapsed = perf_counter() - selection_started
    arms = {}
    representative_timing = {}
    for arm in ARMS:
        arm_started = perf_counter()
        if arm in ('B', 'R'):
            arms[arm] = arm_record(arm, buckets_foot, residuals_foot, masks_foot, counts_foot, status_foot,
                                   candidate_ids, candidates, generators, transform, settings.angle_deg)
        else:
            arms[arm] = arm_record(arm, buckets_midpoint, residuals_midpoint, masks_midpoint, counts_midpoint,
                                   status_midpoint, candidate_ids, candidates, generators, transform, settings.angle_deg)
        representative_timing[arm] = perf_counter() - arm_started
    assert arms['B']['representative_candidate_ids'] == estimator['retained_candidate_ids']
    assert retained_midpoint == [group['leader_row'] for group in arms['M']['groups']]
    # Candidate IDs understate geometric overlap between arms, so record each direction's
    # signed-invariant angle to the nearest baseline direction as well.
    baseline_unit = np.asarray(arms['B']['points_normalised'])
    baseline_unit /= np.linalg.norm(baseline_unit, axis=1)[:, None]
    for arm in ('M', 'R', 'MR'):
        unit = np.asarray(arms[arm]['points_normalised'])
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        cosine = np.clip(np.abs(unit @ baseline_unit.T), 0., 1.)
        arms[arm]['nearest_baseline_direction_deg'] = np.degrees(np.arccos(cosine)).min(axis=1).tolist()

    provenance = {'run': run, 'case_id': case_id, 'code_md5': code, 'working_size': list(size),
                  'native_size': [source['dimensions']['width'], source['dimensions']['height']]}
    timing = {'prepare_s': prepared - started, 'merge_replay_s': merged - prepared,
              'estimate_replay_s': replayed_at - merged, 'residual_foot_s': foot_elapsed,
              'residual_midpoint_s': midpoint_elapsed, 'selection_foot_s': selection_foot_elapsed,
              'selection_midpoint_s': selection_midpoint_elapsed, 'representatives_s': representative_timing}
    write(output / 'e0' / f'{case_id}.json.gz', {
        **provenance, 'schema': 'direction-agreement-e0/1',
        'settings': saved['settings'], 'settings_differ_from_expected': settings_flags,
        'checks': ['instrumented_merge_identical', 'direction_lines_match_saved', 'estimator_replay_matches_saved',
                   'foot_anchor_reproduces_angular_residuals', 'support_counts_match_saved',
                   'bucket_instrumentation_preserves_selection', 'retained_ids_masks_status_points_match_saved'],
        'merge_input_fragments': int(min(len(fragments), ESTIMATE_MERGE_CAP)), 'merged_lines_total': len(all_lines),
        'direction_lines_used': len(lines),
        'membership': [{**entry, 'member_raw_fragment_ids': observations.fragment_ids[entry['member_observation_ids']].tolist()}
                       for entry in sidecar],
        'membership_note': 'member_observation_ids index observations.segments; membership is the merge group '
                           'assignment, distinct from compatible_raw_ids and from preparation groups',
        'foot_buckets': [{'leader_candidate_id': int(candidate_ids[bucket['leader_row']]),
                          'member_candidate_ids': candidate_ids[bucket['member_rows']].tolist()} for bucket in buckets_foot],
    })
    save_array(output / 'e1' / f'{case_id}_residuals_foot.npy.xz', residuals_foot)
    save_array(output / 'e1' / f'{case_id}_residuals_midpoint.npy.xz', residuals_midpoint)
    save_array(output / 'e1' / f'{case_id}_finite_distance_to_midpoint_anchor_px.npy.xz', finite_distances)
    write_csv(output / 'e1' / f'{case_id}_candidates.csv.gz',
              ['candidate_id', 'finite', 'status_foot', 'status_midpoint', 'count_foot', 'count_midpoint', 'gained', 'lost',
               'soft_support_foot', 'soft_support_midpoint'],
              [[int(candidate_ids[row]), int(not infinite[row]), str(status_foot[row]), str(status_midpoint[row]),
                int(counts_foot[row]), int(counts_midpoint[row]), int(gained[row]), int(lost[row]),
                float(supports_foot[row]), float(supports_midpoint[row])] for row in range(len(candidate_ids))])
    write(output / 'e1' / f'{case_id}.json.gz', {
        **provenance, 'schema': 'direction-agreement-e1/1', 'angle_deg': settings.angle_deg,
        'candidate_count': len(candidate_ids), 'finite_candidate_ids': candidate_ids[finite_rows].tolist(),
        'matrix_rows': 'candidate_ids order', 'matrix_columns': f'direction line IDs 0..{len(lines) - 1}',
        'masks_note': 'membership masks are residuals <= angle_deg on the saved matrices; they are not stored separately',
        'anchors': anchors, 'summary': e1_summary, 'largest_residual_changes': top_changes,
        'membership_is_not_ground_truth': True,
    })
    write(output / 'e2' / f'{case_id}.json.gz', {
        **provenance, 'schema': 'direction-agreement-e2/1', 'angle_deg': settings.angle_deg,
        'overlap': settings.overlap, 'pencils': settings.pencils, 'controls_loaded': False,
        'arms': arms, 'timing': timing,
    })
    return {'case_id': case_id, 'timing': timing, 'e1': e1_summary,
            'arms': {arm: {'directions': arms[arm]['direction_count'],
                           'representatives_differ': arms[arm]['representatives_differ_from_leaders'],
                           'ids': arms[arm]['representative_candidate_ids']} for arm in ARMS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--ids', nargs='+', default=list(CASE_IDS))
    parser.add_argument('--resume', action='store_true', help='skip cases whose e2 record carries the current code MD5')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    sources, _ = load_sources(args.root)
    output = run_dir(args.root, args.run)
    code = code_md5(args.root)
    for case_id in args.ids:
        saved_path = args.root / SAVED_ESTIMATORS / f'{case_id}.json.gz'
        identity = {**code, 'saved_estimator': md5(saved_path)}
        existing = output / 'e2' / f'{case_id}.json.gz'
        if args.resume and existing.exists() and read(existing)['code_md5'] == identity:
            print(case_id, 'complete under identical code and input; skipped', flush=True)
            continue
        saved = read(saved_path)
        assert saved['case_id'] == case_id
        summary = build_case(case_id, sources[case_id], saved, output, args.run, identity)
        print(case_id, 'arms', summary['arms'], 'timing', {name: round(value, 3) for name, value in summary['timing'].items()
                                                            if isinstance(value, float)}, flush=True)


if __name__ == '__main__':
    main()
