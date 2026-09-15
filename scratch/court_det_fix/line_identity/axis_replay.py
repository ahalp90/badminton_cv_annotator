"""Replay the axis matching for chosen direction pairs and find where the control court is lost.

The matcher turns one pair of selected directions into courts in four steps per direction:
enumerate a scale and offset from every pair of direction-compatible observed line groups
against every pair of court markings; keep the enumerations with at least three supported
markings that the necessary player rule accepts; drop duplicates by their marking assignment;
keep the 512 best by score (the per-direction cap). Courts are every kept horizontal matching
combined with every kept vertical matching.

For one view, one direction selection and one pair this script reruns those steps with no cap,
gates the replay against the generation record (the 512 kept matchings per direction must agree
by ID and parameters), and then measures, per direction and per step, the best court reachable
when that direction takes its best surviving matching and the other direction takes the ideal
one. The ideal per-direction scale and offset come from the direction-fit homography (stage E3
of the direction experiment), which shares the pair's two vanishing points by construction.

Read-only on every record. Writes table.csv and a JSON summary into --output.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from common import (
    DIRECTION_RUN,
    LABELS,
    add_helper_paths,
    arm_points,
    control_corners,
    corner_errors,
    feet_working,
    load_source,
    read,
)

add_helper_paths()

from projective_seed import (
    Settings,
    basis_for,
    match_axis,
    necessary_players,
    offsets,
    score_axes,
)
from run_population import prepare

from experiments.annotator.independent_court import assignment, detector

MATCHER_SETTINGS = Settings(keep_axes=512)  # run_automatic.generate's settings
UNCAPPED = replace(MATCHER_SETTINGS, keep_axes=10 ** 9)
AXIS_NAMES = ('horizontal', 'vertical')
AXIS_COORDINATES = (detector.X_COORDS, detector.Y_COORDS)
# The canonical 180-degree relabelling of the court (run_given.canonicalise).
SYMMETRY = np.array([[-1., 0., detector.CORNER_COURT_M[:, 0].max()],
                     [0., -1., detector.CORNER_COURT_M[:, 1].max()], [0., 0., 1.]])
STEPS = ('enumerated', 'supported', 'supported_and_players', 'distinct', 'kept')
COLUMNS = ['case_id', 'label', 'arm', 'pair_id', 'pencils', 'direction_fit_px', 'nearest_combined_px_record',
           'nearest_kept_by_kept_px_replay', 'nearest_distinct_by_distinct_px', 'axis', 'coordinates', 'enumerated', 'supported',
           'supported_and_players', 'distinct', 'kept', 'ideal_markings_supported', 'ideal_markings_matched',
           'ideal_player_compatible',
           'cap_threshold_score', 'kept_support_counts', 'ideal_neighbourhood_supported_share',
           *[f'best_{step}_px' for step in STEPS], *[f'best_{step}_rank' for step in STEPS],
           *[f'best_{step}_support' for step in STEPS], *[f'best_{step}_score' for step in STEPS]]


def load_record(arm: str, case_id: str, record_dir: Path) -> dict:
    if arm == 'B':
        record = read(record_dir / f'{case_id}.json.gz')
    else:
        record = read(record_dir / arm / 'results' / f'{case_id}.json.gz')
    assert record['case_id'] == case_id, (record['case_id'], case_id)
    return record


def pair_record(record: dict, pair_id: int) -> dict:
    pair = next(pair for pair in record['pairs'] if pair['pair_id'] == pair_id)
    assert pair['status'] == 'matched', (pair_id, pair['status'])
    return pair


def nearest_in_record(record: dict, control: np.ndarray, scale: np.ndarray) -> tuple[float, int, str]:
    """Nearest shortlisted court in the record, over every matched pair (native corners scaled to working)."""
    best = (np.inf, -1, '')
    for pair in record['pairs']:
        if pair['status'] != 'matched' or not pair['shortlist']:
            continue
        corners = np.asarray([entry['corners_px'] for entry in pair['shortlist']], dtype=float) / scale
        errors = corner_errors(corners, control)
        index = int(np.argmin(errors))
        if errors[index] < best[0]:
            best = (float(errors[index]), pair['pair_id'], pair['shortlist'][index]['candidate_id'])
    return best


def ideal_parameters(basis: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-direction (scale, offset) that reproduce the homography in the basis chart, and the off-diagonal leak."""
    chart = np.linalg.solve(basis, homography)
    chart = chart / chart[2, 2]
    leak = float(np.abs([chart[0, 1], chart[1, 0], chart[2, 0], chart[2, 1]]).max())
    return np.array([[chart[0, 0], chart[0, 2]], [chart[1, 1], chart[1, 2]]]), leak


def courts_from(basis: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    """Corners of every court from paired (scale, offset) rows; mirrors projective_seed.combine."""
    maps = np.tile(np.eye(3), (len(horizontal), 1, 1))
    maps[:, 0, 0], maps[:, 0, 2] = horizontal.T
    maps[:, 1, 1], maps[:, 1, 2] = vertical.T
    homographies = basis @ maps
    corners, _ = detector.project(homographies, detector.CORNER_COURT_M)
    return corners


def sweep(basis: np.ndarray, axis: int, parameters: np.ndarray, ideal_other: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Corner error of every matching on one direction when the other direction is ideal."""
    other = np.tile(ideal_other, (len(parameters), 1))
    horizontal, vertical = (parameters, other) if axis == 0 else (other, parameters)
    errors = np.empty(len(parameters))
    for start in range(0, len(parameters), 4096):
        stop = start + 4096
        errors[start:stop] = corner_errors(courts_from(basis, horizontal[start:stop], vertical[start:stop]), control)
    return errors


def step_masks(matches, cap: int) -> dict[str, np.ndarray]:
    """Boolean survival masks over the enumerated matchings for each step of the matching."""
    count = len(matches.parameters)
    supported = matches.supported >= MATCHER_SETTINGS.minimum_matches
    distinct = np.zeros(count, dtype=bool)
    distinct[matches.distinct] = True
    kept = np.zeros(count, dtype=bool)
    kept[matches.distinct[:cap]] = True
    return {'enumerated': np.ones(count, dtype=bool), 'supported': supported,
            'supported_and_players': supported & matches.player_compatible, 'distinct': distinct, 'kept': kept}


def nearest_over_product(basis: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray, control: np.ndarray) -> float:
    """Nearest court over every horizontal matching combined with every vertical matching."""
    best = np.inf
    block = max(1, 4_000_000 // max(1, len(vertical)))
    for start in range(0, len(horizontal), block):
        rows = horizontal[start:start + block]
        block_h = np.repeat(rows, len(vertical), axis=0)
        block_v = np.tile(vertical, (len(rows), 1))
        best = min(best, float(corner_errors(courts_from(basis, block_h, block_v), control).min()))
    return best


def replay_pair(case_id: str, arm: str, pair_id: int, record: dict, e3: dict, uncapped: bool) -> tuple[list[dict], dict]:
    source = load_source(case_id)
    segments, _, size = prepare(source)
    observations = assignment.prepare_observations(segments, size)
    feet = feet_working(source, size)
    control, _ = control_corners(case_id)
    points = arm_points(case_id, arm)
    pair = pair_record(record, pair_id)
    pair_points = points[pair['pencils']]
    fit = e3['sets'][arm]['fits']['records'][pair_id]
    assert fit['pair_id'] == pair_id and fit['groups'] == pair['pencils'], (fit['pair_id'], fit['groups'], pair['pencils'])
    homography = np.asarray(fit['homography_working'], dtype=float)
    fit_corners, _ = detector.project(homography[None], detector.CORNER_COURT_M)
    fit_error = float(corner_errors(fit_corners, control)[0])
    assert abs(fit_error - fit['max_corner_working_px']) < 1e-6, (fit_error, fit['max_corner_working_px'])

    basis, details = basis_for(pair_points, size, MATCHER_SETTINGS)
    assert basis is not None and details['status'] == 'valid', details
    np.testing.assert_allclose(basis, pair['role']['basis_working'], rtol=0, atol=1e-9)

    # Both court orientations reproduce the same physical court; keep the one whose chart leaks least.
    candidates = [ideal_parameters(basis, homography), ideal_parameters(basis, homography @ SYMMETRY)]
    ideal, leak = min(candidates, key=lambda item: item[1])
    ideal_corners = courts_from(basis, ideal[:1], ideal[1:])
    ideal_error = float(corner_errors(ideal_corners, control)[0])
    assert abs(ideal_error - fit_error) < 1e-6, (ideal_error, fit_error)

    rows, matched = [], {}
    for axis, (name, coordinates) in enumerate(zip(AXIS_NAMES, AXIS_COORDINATES)):
        matches = match_axis(basis, axis, coordinates, observations, size, UNCAPPED, feet)
        matched[axis] = matches
        recorded = pair['role']['axes'][axis]
        kept_ids = matches.distinct[:MATCHER_SETTINGS.keep_axes]
        assert kept_ids.tolist() == [entry['axis_id'] for entry in recorded['entries']], 'kept matching IDs differ from the record'
        np.testing.assert_allclose(matches.parameters[kept_ids], [entry['parameters'] for entry in recorded['entries']], rtol=0, atol=1e-9)
        assert matches.diagnostics['distinct_assignments'] == recorded['diagnostics']['distinct_assignments']
        assert matches.diagnostics['enumerated'] == recorded['diagnostics']['enumerated']

        # Which court markings the ideal matching finds among the direction-compatible observed groups.
        ids, _, endpoints, _ = offsets(basis, axis, observations, size, MATCHER_SETTINGS)
        _, group_indexes, counts = score_axes(ideal[axis][None], coordinates, endpoints, basis, axis)
        homogeneous_feet = np.concatenate((feet, np.ones((*feet.shape[:-1], 1))), axis=2)
        mapped = homogeneous_feet @ np.linalg.inv(basis).T
        with np.errstate(divide='ignore', invalid='ignore'):
            rectified_feet = mapped[..., axis] / mapped[..., 2]
        ideal_players = bool(necessary_players(ideal[axis][None], rectified_feet, axis, float(coordinates.max()))[0])

        errors = sweep(basis, axis, matches.parameters, ideal[1 - axis], control)
        rank = np.full(len(matches.parameters), -1)
        rank[matches.distinct] = np.arange(len(matches.distinct))
        masks = step_masks(matches, MATCHER_SETTINGS.keep_axes)
        row = {'case_id': case_id, 'label': LABELS[case_id], 'arm': arm, 'pair_id': pair_id,
               'pencils': ' '.join(map(str, pair['pencils'])), 'direction_fit_px': round(fit_error, 4),
               'axis': name, 'coordinates': len(coordinates),
               **{step: int(mask.sum()) for step, mask in masks.items()},
               'ideal_markings_supported': int(counts[0]),
               'ideal_markings_matched': ' '.join('-' if group < 0 else str(int(ids[group])) for group in group_indexes[0]),
               'ideal_player_compatible': ideal_players}
        kept_ids = matches.distinct[:MATCHER_SETTINGS.keep_axes]
        # The score the 512th kept matching had, and how support counts spread among the kept ones.
        row['cap_threshold_score'] = round(float(matches.scores[kept_ids[-1]]), 4) if len(kept_ids) else None
        support_counts = np.bincount(matches.supported[kept_ids], minlength=len(coordinates) + 1)
        row['kept_support_counts'] = ' '.join(f'{count}:{int(total)}' for count, total in enumerate(support_counts) if total)
        # Among matchings within 10 px of the fit (other direction ideal), the share that pass the support rule.
        neighbourhood = errors <= 10.
        row['ideal_neighbourhood_supported_share'] = (round(float(masks['supported'][neighbourhood].mean()), 4)
                                                     if neighbourhood.any() else None)
        for step, mask in masks.items():
            if not mask.any():
                row[f'best_{step}_px'], row[f'best_{step}_rank'] = None, None
                row[f'best_{step}_support'], row[f'best_{step}_score'] = None, None
                continue
            index = int(np.flatnonzero(mask)[np.argmin(errors[mask])])
            row[f'best_{step}_px'] = round(float(errors[index]), 4)
            row[f'best_{step}_rank'] = int(rank[index])
            row[f'best_{step}_support'] = int(matches.supported[index])
            row[f'best_{step}_score'] = round(float(matches.scores[index]), 4)
        rows.append(row)

    # Second gate: every kept horizontal with every kept vertical must reproduce the record's nearest court.
    kept_h = matched[0].parameters[matched[0].distinct[:MATCHER_SETTINGS.keep_axes]]
    kept_v = matched[1].parameters[matched[1].distinct[:MATCHER_SETTINGS.keep_axes]]
    best_combined = nearest_over_product(basis, kept_h, kept_v, control)
    # With no per-direction cap: every distinct horizontal matching with every distinct vertical one.
    best_uncapped = (nearest_over_product(basis, matched[0].parameters[matched[0].distinct],
                                          matched[1].parameters[matched[1].distinct], control) if uncapped else None)
    summary = {'case_id': case_id, 'arm': arm, 'pair_id': pair_id, 'pencils': pair['pencils'],
               'direction_fit_px': fit_error, 'chart_leak': leak, 'ideal_parameters': ideal.tolist(),
               'nearest_kept_by_kept_px_replay': best_combined, 'nearest_distinct_by_distinct_px': best_uncapped}
    for row in rows:
        row['nearest_kept_by_kept_px_replay'] = round(best_combined, 4)
        row['nearest_distinct_by_distinct_px'] = None if best_uncapped is None else round(best_uncapped, 4)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pregate-records', type=Path, required=True,
                        help='folder holding <arm>/results/<case>.json.gz from the courts-before-the-gate run')
    parser.add_argument('--baseline-records', type=Path, required=True,
                        help='folder holding the baseline generation records <case>.json.gz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--uncapped-combination', action='store_true',
                        help='also combine every distinct matching with every distinct one (millions of courts per pair)')
    parser.add_argument('--pairs', nargs='+', required=True,
                        help='case:arm:pair_id, or case:arm:best for the pair with the best direction fit, '
                             'or case:arm:nearest for the pair holding the nearest shortlisted court')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    for spec in args.pairs:
        case_id, arm, which = spec.split(':')
        record = load_record(arm, case_id, args.baseline_records if arm == 'B' else args.pregate_records)
        e3 = read(DIRECTION_RUN / 'e3' / f'{case_id}.json.gz')
        source = load_source(case_id)
        control, _ = control_corners(case_id)
        scale = np.asarray([source['dimensions']['width'], source['dimensions']['height']], dtype=float) / np.asarray(record['working_size'])
        nearest_px, nearest_pair, nearest_candidate = nearest_in_record(record, control, scale)
        if which == 'best':
            pair_id = e3['sets'][arm]['summary']['best_finite']['pair_id']
        elif which == 'nearest':
            pair_id = nearest_pair
        else:
            pair_id = int(which)
        print(f'{LABELS[case_id]} arm {arm} pair {pair_id}: record nearest shortlisted court {nearest_px:.3f} px '
              f'(pair {nearest_pair}, {nearest_candidate}); replaying', flush=True)
        pair_rows, summary = replay_pair(case_id, arm, pair_id, record, e3, args.uncapped_combination)
        summary.update({'record_nearest_px': nearest_px, 'record_nearest_pair': nearest_pair,
                        'record_nearest_candidate': nearest_candidate, 'selector': which})
        for row in pair_rows:
            row['nearest_combined_px_record'] = round(nearest_px, 4)
        rows.extend(pair_rows)
        summaries.append(summary)
        for row in pair_rows:
            ladder = ', '.join(f'{step} {row[f"best_{step}_px"]} (rank {row[f"best_{step}_rank"]}, support {row[f"best_{step}_support"]}, '
                               f'score {row[f"best_{step}_score"]}, {row[step]} rows)' for step in STEPS)
            print(f'  {row["axis"]}: ideal matching supports {row["ideal_markings_supported"]} of {row["coordinates"]} markings '
                  f'[{row["ideal_markings_matched"]}], players {"pass" if row["ideal_player_compatible"] else "FAIL"}; '
                  f'cap threshold score {row["cap_threshold_score"]}, kept support counts {row["kept_support_counts"]}; '
                  f'best court with the other direction ideal: {ladder}', flush=True)
        uncapped = summary['nearest_distinct_by_distinct_px']
        print(f'  kept x kept replay nearest {summary["nearest_kept_by_kept_px_replay"]:.3f} px; '
              f'distinct x distinct nearest {"not computed" if uncapped is None else f"{uncapped:.3f} px"}; '
              f'direction fit {summary["direction_fit_px"]:.3f}; chart leak {summary["chart_leak"]:.2e}', flush=True)
    with open(args.output / 'table.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with open(args.output / 'summary.json', 'w') as stream:
        json.dump(summaries, stream, indent=1)
    print(f'wrote {args.output / "table.csv"} ({len(rows)} rows) and summary.json')


if __name__ == '__main__':
    main()
