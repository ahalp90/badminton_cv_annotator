"""Diagnose the axis matcher with supplied directions before automatic VP search.

pregate copy of axis_matching_20260914/run_given.py: propose_role also hands back every
combined court before the geometry and player masks, with both masks and the two player
fractions the player mask reads. Candidate construction is untouched.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from projective_seed import (
    AxisMatches,
    Settings,
    basis_for,
    combine,
    corner_errors,
    joint_player_fractions,
    match_axis,
)
from run_diagnosis import gate_evidence, read, write
from run_population import prepare
from scan_population import continuous_support, geometry, marking_score, retain

from experiments.annotator.independent_court import assignment, detector
from experiments.annotator.independent_court import stripe_observations as stripes

KEEP_COMBINED = 256
SCORING_BATCH = 256
# continuous_support reads 64 samples along each marking interval; the bound reads every eighth.
SAMPLES_PER_INTERVAL = 64
BOUND_SAMPLES = np.unique(np.r_[np.arange(0, SAMPLES_PER_INTERVAL, 8), SAMPLES_PER_INTERVAL - 1])
# Truncating a sample to its pixel moves it by less than one pixel on each axis.
PIXEL_TRUNCATION_PX = float(np.sqrt(2))
# Cover float32 map values and float32 responses, which differ from exact values by about 1e-7.
DISTANCE_SLACK_PX = 1e-3
SCORE_SLACK = 1e-6


def finite_scores(
    homographies: np.ndarray, observations: assignment.Observations,
    axes: tuple[AxisMatches, AxisMatches], size: tuple[int, int],
    shortlist: detector.Settings | None = None, corners: np.ndarray | None = None,
) -> np.ndarray:
    """Score canonically oriented courts using families from the current directions.

    Given the pair's shortlist settings and the courts' corners, courts that cannot enter the
    shortlist keep -inf instead of a score (see shortlist_scores).
    """
    families = []
    for matched in axes:
        groups = matched.diagnostics['retained_group_ids']
        members = np.concatenate([observations.groups[index] for index in groups])
        families.append(observations.segments[members].reshape(-1, 4))
    maps = detector._distance_maps((families[0], families[1]), size)
    if shortlist is not None:
        assert corners is not None
        return shortlist_scores(homographies, maps, size, corners, shortlist)
    scores = np.empty(len(homographies))
    for start in range(0, len(homographies), SCORING_BATCH):
        scores[start:start + SCORING_BATCH] = continuous_support(homographies[start:start + SCORING_BATCH], maps, size)
    return scores


def score_upper_bounds(homographies: np.ndarray, maps: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """An upper bound on each court's continuous_support score, from every eighth sample.

    A distance map is an exact Euclidean distance transform, so its value changes by at most
    the pixel distance between two samples. Between measured samples a and b, every sample is
    therefore at least (d_a + d_b - (b - a) * spacing) / 2 - sqrt(2) px from a line, where
    sqrt(2) covers truncation to whole pixels. The score only grows with each response, so
    using the largest response each sample could have bounds it.
    """
    projected, _ = detector.project(homographies, detector.SEGMENTS_M)
    endpoints = projected.reshape(-1, 12, 2, 2)
    starts = endpoints[:, :, 0]
    vectors = endpoints[:, :, 1] - starts
    lower, upper, visible = detector._visible_fractions(endpoints, size)
    # The same arithmetic as detector._visible_samples, so measured samples land on the same pixels.
    sample_fractions = np.linspace(0, 1, SAMPLES_PER_INTERVAL)[BOUND_SAMPLES]
    fractions = lower[..., None] + (upper - lower)[..., None] * sample_fractions
    samples = starts[..., None, :] + fractions[..., None] * vectors[..., None, :]
    pixel_x = np.clip(samples[..., 0], 0, size[0] - 1).astype(int)
    pixel_y = np.clip(samples[..., 1], 0, size[1] - 1).astype(int)
    family = np.repeat([0, 1], 6)[None, :, None]
    distance = maps[family, pixel_y, pixel_x].astype(float)
    spacing = (upper - lower) * np.linalg.norm(vectors, axis=-1) / (SAMPLES_PER_INTERVAL - 1)
    gaps = np.diff(BOUND_SAMPLES)
    between = (distance[..., :-1] + distance[..., 1:] - gaps * spacing[..., None]) / 2 - PIXEL_TRUNCATION_PX
    response_sum = (largest_response(distance).sum(axis=2)
                    + ((gaps - 1) * largest_response(between)).sum(axis=2))
    response = np.where(visible, response_sum / SAMPLES_PER_INTERVAL, 0.)
    return marking_score(response, visible) + SCORE_SLACK


def largest_response(least_distance_px: np.ndarray) -> np.ndarray:
    """continuous_support's sample response at the smallest distance a sample could have."""
    distance = np.maximum(least_distance_px - DISTANCE_SLACK_PX, 0)
    return np.exp(-.5 * np.square(distance / assignment.DISTANCE_SIGMA_PX))


def shortlist_scores(
    homographies: np.ndarray, maps: np.ndarray, size: tuple[int, int], corners: np.ndarray,
    shortlist: detector.Settings,
) -> np.ndarray:
    """Score courts in descending order of their bounds until the unscored ones cannot matter.

    retain keeps the shortlist's best mutually distinct courts, greedily in score order. Once
    the courts scored above every unscored court's bound fill the shortlist, the unscored
    courts sort after the point where retain stops, so they cannot change the shortlist. They
    keep -inf, which also sorts them last. Scored courts get exactly continuous_support's score.
    """
    bounds = np.concatenate([score_upper_bounds(homographies[start:start + SCORING_BATCH], maps, size)
                             for start in range(0, len(homographies), SCORING_BATCH)])
    order = np.argsort(-bounds, kind='stable')
    scores = np.full(len(homographies), -np.inf)
    scored = 0
    while scored < len(order):
        # Each round adds a quarter of what is scored so far, which keeps the retain checks few.
        end = min(len(order), scored + max(SCORING_BATCH, scored // 4))
        for start in range(scored, end, SCORING_BATCH):
            rows = order[start:min(start + SCORING_BATCH, end)]
            scores[rows] = continuous_support(homographies[rows], maps, size)
        scored = end
        if scored < len(order) and shortlist_filled(scores, corners, bounds[order[scored]], shortlist):
            break
    return scores


def shortlist_filled(scores: np.ndarray, corners: np.ndarray, remaining_bound: float,
                     shortlist: detector.Settings) -> bool:
    """Whether the courts scored above every unscored court's bound already fill the shortlist.

    Every court scoring above remaining_bound has been scored, so these courts are exactly the
    top of the full score order, and retain on them matches retain on all courts up to its stop.
    """
    ahead = np.flatnonzero(scores > remaining_bound)
    if len(ahead) < shortlist.keep_candidates:
        return False
    candidates = [detector.Candidate(corners[index], float(scores[index]), (0., 0.), (0, 0)) for index in ahead]
    return len(retain(candidates, shortlist)) == shortlist.keep_candidates


def control_homography(source: dict, reference: dict, vp_saved: dict, marking_summary: dict) -> tuple[np.ndarray, str]:
    case_id = source['id']
    if case_id == 'gxBQ_window_00_frame_5':
        corners = vp_saved['known_target']['corners_px']
        origin = 'visually_approved_GX5_generated_court'
    elif case_id in ('shuttleset_03_scene_0019', 'shuttleset_03_scene_0016', 'shuttleset_21_scene_0020'):
        record = next(record for record in marking_summary['records'] if record['case_id'] == case_id)
        corners = record['fit']['corners_px']
        origin = 'visually_approved_automatic_marking_refit'
    elif case_id == 'am2_window_00_frame_150':
        selected = vp_saved['diagnostic_rankings']['geometry_before_players']['max_corner_native_px'][0]['generation_id']
        corners = next(row['corners_px'] for row in vp_saved['examples'] if row['generation_id'] == selected)
        origin = 'visually_approved_VP_diagnostic_court'
    else:
        corners = reference['corners_px']
        origin = 'manual_reference_directions_only'
    _, _, size = prepare(source)
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    transform = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, (np.asarray(corners) / scale).astype(np.float32))
    return transform, origin


def canonicalise(homographies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners, _ = detector.project(homographies, detector.CORNER_COURT_M)
    rotate = corners[:, :2, 1].mean(axis=1) > corners[:, 2:, 1].mean(axis=1)
    symmetry = np.array([[-1., 0., detector.CORNER_COURT_M[:, 0].max()],
                         [0., -1., detector.CORNER_COURT_M[:, 1].max()], [0., 0., 1.]])
    homographies = homographies.copy()
    homographies[rotate] = homographies[rotate] @ symmetry
    return homographies, rotate


def axis_diagnostic(matches: AxisMatches, basis: np.ndarray, axis: int, control: np.ndarray) -> dict:
    """Assess axis loss with the other control axis held fixed; never select proposals."""
    rectified = np.linalg.solve(basis, control)
    rectified /= rectified[2, 2]
    maps = np.tile(rectified, (len(matches.parameters), 1, 1))
    maps[:, axis, axis], maps[:, axis, 2] = matches.parameters.T
    projected, _ = detector.project(basis @ maps, detector.CORNER_COURT_M)
    truth, _ = detector.project(control[None], detector.CORNER_COURT_M)
    errors = np.linalg.norm(projected - truth, axis=2).max(axis=1)
    # Under player pruning, player-incompatible rows are never scored (NaN score, zero support),
    # so the pattern-supported stage is unmeasured.
    pattern = matches.supported >= Settings().minimum_matches
    pruned = matches.diagnostics['necessary_player_pruning']
    result = {}
    for name, ids in [('enumerated', np.arange(len(errors))),
                      ('pattern_supported', None if pruned else np.flatnonzero(pattern)),
                      ('pattern_and_players', np.flatnonzero(pattern & matches.player_compatible)),
                      ('distinct', matches.distinct), ('retained', matches.retained)]:
        if ids is None or not len(ids):
            result[name] = None
            continue
        index = int(ids[np.argmin(errors[ids])])
        result[name] = {'axis_id': index, 'max_corner_working_px': float(errors[index]),
                        'score': float(matches.scores[index]), 'support_count': int(matches.supported[index]),
                        'matched_groups': matches.matches[index].tolist(), 'anchors': matches.anchors[index].tolist()}
    return result


def pack_axis(matches: AxisMatches) -> dict:
    entries = []
    for index in matches.retained:
        entries.append({'axis_id': int(index), 'parameters': matches.parameters[index].tolist(),
                        'score': float(matches.scores[index]), 'matched_groups': matches.matches[index].tolist(),
                        'anchors': matches.anchors[index].tolist(), 'support_count': int(matches.supported[index])})
    return {'diagnostics': matches.diagnostics, 'entries': entries}


@dataclass
class RoleProposals:
    record: dict
    basis: np.ndarray | None
    axes: tuple[AxisMatches, AxisMatches] | None
    candidates: list[detector.Candidate]
    details: list[dict]
    # pregate copy: every combined court in transforms order (working px, float32), the geometry
    # mask, the player mask and the two player fractions. Empty when the basis fails.
    combined_corners: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 2), dtype=np.float32))
    valid: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    usable: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    player_any: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    player_both_halves: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


def propose_role(
    points: np.ndarray, observations: assignment.Observations, feet: np.ndarray,
    size: tuple[int, int], settings: Settings,
    player_pruning: bool = True, combined_ranking: str = 'finite',
    shortlist: detector.Settings | None = None,
) -> RoleProposals:
    """Generate one ordered direction role without reference geometry or labels.

    :param shortlist: the retention settings the caller will apply to these candidates. When
        given, candidates that provably fall outside that shortlist keep a score of -inf.
    """
    basis, details = basis_for(points, size, settings)
    record = {'basis_status': details}
    if basis is None:
        return RoleProposals(record, None, None, [], [])
    axis_feet = feet if player_pruning else None
    horizontal = match_axis(basis, 0, detector.X_COORDS, observations, size, settings, axis_feet)
    vertical = match_axis(basis, 1, detector.Y_COORDS, observations, size, settings, axis_feet)
    transforms, axis_pairs = combine(basis, horizontal, vertical)
    transforms, rotated = canonicalise(transforms)
    valid, corners = geometry(transforms, size)
    one, two = joint_player_fractions(basis, horizontal, vertical, feet)
    usable = valid & (one == 1) & (two >= .5)
    record.update({'basis_working': basis.tolist(), 'axes': [pack_axis(horizontal), pack_axis(vertical)],
                   'combined': len(transforms), 'geometry_valid': int(valid.sum()),
                   'geometry_players': int(usable.sum())})
    usable_ids = np.flatnonzero(usable)
    finite = finite_scores(transforms[usable], observations, (horizontal, vertical), size,
                           shortlist, corners[usable]) if (combined_ranking == 'finite' and len(usable_ids)) else None
    if finite is not None:
        record['finite_scored'] = int(np.isfinite(finite).sum())
    candidates, provenance = [], []
    for position, index in enumerate(usable_ids):
        first, second = axis_pairs[index]
        score = float((horizontal.scores[first] + vertical.scores[second]) / 2)
        shortlist_score = score if finite is None else float(finite[position])
        candidates.append(detector.Candidate(corners[index], shortlist_score, (0., 0.), (0, 0)))
        provenance.append({'axis_ids': [int(first), int(second)], 'rotated_180': bool(rotated[index]),
                           'axis_score': score, 'homography_working': transforms[index].tolist()})
    return RoleProposals(record, basis, (horizontal, vertical), candidates, provenance,
                         np.asarray(corners, dtype=np.float32).reshape(-1, 4, 2),
                         np.asarray(valid, dtype=bool), np.asarray(usable, dtype=bool),
                         np.asarray(one, dtype=np.float32), np.asarray(two, dtype=np.float32))


def run_case(
    source: dict, control: np.ndarray, origin: str, zone: object, reference: dict, player_pruning: bool = False,
    keep_axes: int = 64,
    combined_ranking: str = 'axis',
) -> dict:
    started = perf_counter()
    settings = Settings(keep_axes=keep_axes)
    segments, families, size = prepare(source)
    scale = np.array([source['dimensions']['width'], source['dimensions']['height']]) / size
    observations = assignment.prepare_observations(segments, size)
    weights = stripes.fragment_weights(observations)
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in source['all_feet_px']], dtype=float) / scale
    role_records, candidates, provenance = [], [], {}
    counter = 0
    true_corners, _ = detector.project(control[None], detector.CORNER_COURT_M)
    for role, points in enumerate((control[:, :2].T, control[:, [1, 0]].T)):
        proposed = propose_role(points, observations, feet, size, settings, player_pruning, combined_ranking)
        record = {'role': role, **proposed.record}
        for candidate, details in zip(proposed.candidates, proposed.details, strict=True):
            candidates.append(candidate)
            provenance[id(candidate)] = {'candidate_id': counter, 'role': role, **details}
            counter += 1
        # Given directions are diagnostic; cap-loss measurements happen after generation.
        if role == 0 and proposed.axes is not None:
            horizontal, vertical = proposed.axes
            record['axis_control_diagnostics'] = [axis_diagnostic(horizontal, proposed.basis, 0, control),
                                                  axis_diagnostic(vertical, proposed.basis, 1, control)]
        role_records.append(record)
    retention = replace(detector.DEFAULT_SETTINGS, keep_candidates=KEEP_COMBINED, distinct_corner_distance=2.)
    retained = retain(candidates, retention)
    maps = detector._distance_maps(detector._wide_line_families(segments), size)
    entries = []
    for candidate in retained:
        details = provenance[id(candidate)]
        homography = np.asarray(details['homography_working'])
        stripe = stripes.score_model(stripes.measure(homography, observations, size), weights, 3)
        corners = candidate.corners_px * scale
        gates = gate_evidence(corners, source, scale, size, families, maps, zone)
        entries.append({**details, 'corners_px': corners.tolist(), 'shortlist_score': candidate.score,
                        'stripe': stripe, 'gates': gates})
    rankings = {}
    for name, pool in [('geometry_players_before_cap', candidates), ('after_combined_cap', retained)]:
        if not pool:
            rankings[name] = None
            continue
        errors = corner_errors(np.asarray([candidate.corners_px for candidate in pool]), true_corners[0])
        best = int(np.argmin(errors))
        rankings[name] = {'candidate_id': provenance[id(pool[best])]['candidate_id'],
                          'control_max_corner_working_px': float(errors[best])}
    for entry in entries:
        display_scale = np.array([1280, 720]) / np.array([source['dimensions']['width'], source['dimensions']['height']])
        entry['reference_max_corner_1280_px'] = float(corner_errors(
            np.asarray(entry['corners_px']) * display_scale, np.asarray(reference['corners_px']) * display_scale))
    return {'schema': 'given-directions-axis-matching/1', 'case_id': source['id'], 'given_direction_source': origin,
            'automatic_detection': False, 'settings': asdict(settings), 'working_size': size,
            'necessary_player_pruning': player_pruning,
            'combined_ranking': combined_ranking,
            'control_corners_px': (true_corners[0] * scale).tolist(), 'roles': role_records, 'entries': entries,
            'combined_shortlist': len(retained), 'combined_cap': KEEP_COMBINED, 'control_diagnostics': rankings,
            'raw_groups': [observations.fragment_ids[group].tolist() for group in observations.groups],
            'elapsed_s': perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--vp-saved', type=Path, required=True)
    parser.add_argument('--marking-summary', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--axis-player-pruning', action='store_true')
    parser.add_argument('--keep-axes', type=int, default=64)
    parser.add_argument('--combined-ranking', choices=('axis', 'finite'), default='axis')
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy.resolve()))
    zone = importlib.import_module('zone_net')
    cv2.setNumThreads(1)
    packed = read(args.inputs)
    cases = {source['id']: source for source in packed['cases']}
    marking = read(args.marking_summary)
    for case_id in args.ids:
        source, reference = cases[case_id], packed['references'][case_id]
        control, origin = control_homography(source, reference, read(args.vp_saved / f'{case_id}.json.gz'), marking)
        result = run_case(source, control, origin, zone, reference, args.axis_player_pruning, args.keep_axes,
                          args.combined_ranking)
        write(args.output / f'{case_id}.json.gz', result)
        print(case_id, 'roles', [{key: role.get(key) for key in ('combined', 'geometry_valid', 'geometry_players')}
                                for role in result['roles']], 'control', result['control_diagnostics'],
              'seconds', result['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
