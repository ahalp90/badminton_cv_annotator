"""Filter fragments before the merge, then replay direction selection and axis matching on every view.

Three fragment filters, fixed before any result was read (thresholds from the paint-profile
study, `runs/paint_profiles/`):

- person: drop a fragment whose midpoint lies inside any person box the input pack carries.
- paint: keep a fragment whose ridge contrast is at least PAINT_CONTRAST and whose peak
  saturation is at most PAINT_SATURATION (see paint_profiles.py for the two measures).
- paint_person: both.

Two sensitivity arms move the contrast threshold to 15 and 25.

Stage 1, direction selection: the unfiltered replay must reproduce the saved baseline direction
record exactly (retained candidate IDs, working points, support masks). Each arm then feeds its
filtered fragments through the unchanged merge and coverage selection (`vp_pruning.estimate`)
and is measured against the frozen control: the incidence bound over the selected directions,
the best direction fit, the smallest angle to each of the control's two vanishing points, and
how many fragments on the control's painted markings the filter removed.

Stage 2, axis matching: for the arm's best-fit direction pair, the axis matching is rerun on the
filtered observations with no per-direction cap, and the ladder of axis_replay.py is reported
(best court per direction after each rule, the rank the cap truncates at, the kept-by-kept
nearest court). Two observation-only arms, paint_observations and person_observations, keep
the baseline directions and the baseline's best-fit pair and filter only the observations the
axis matching sees; their matcher inputs carry the saved baseline direction record with the
filtered fragments.

Writes runs/<out>/table.csv (stage 1), axis_table.csv (stage 2), run.log via the caller, and
--inputs-dir/<arm>/ (default: inputs/ beside this script) with the filtered pack entries and
direction records the matcher run on the compute host reads.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from axis_replay import (
    AXIS_COORDINATES,
    AXIS_NAMES,
    MATCHER_SETTINGS,
    STEPS,
    SYMMETRY,
    UNCAPPED,
    courts_from,
    ideal_parameters,
    nearest_over_product,
    step_masks,
    sweep,
)
from paint_profiles import EDGE_PX, features, marking_distance, profiles

from shared import (
    CASE_IDS,
    LABELS,
    PACK_OF,
    PACKS,
    add_helper_paths,
    control_corners,
    corner_errors,
    feet_working,
    frame_path,
    load_direction_record,
    load_estimator,
    read,
    write,
)

add_helper_paths()

import vp_pruning
from geometry_certificate import set_bound
from projective_seed import basis_for, match_axis
from run_population import prepare
from run_svd_fixed import fit_pairs

from experiments.annotator.independent_court import assignment, detector

PAINT_CONTRAST = 20.
PAINT_SATURATION = 90.
SENSITIVITY_CONTRASTS = (15., 25.)
ARMS = ('baseline', 'person', 'paint', 'paint_person', 'paint15', 'paint25')
FLOAT_ATOL = 1e-12
STAGE1_COLUMNS = ['case_id', 'label', 'arm', 'fragments', 'dropped', 'marking_fragments', 'marking_dropped', 'merged_lines',
                  'directions', 'angle_to_control_x_deg', 'angle_to_control_y_deg', 'bound_px', 'best_fit_px',
                  'best_fit_pair', 'best_fit_groups', 'selection_identical_to_baseline']
OBSERVATION_ARMS = {'paint_observations': 'paint', 'person_observations': 'person'}
DEFAULT_AXIS_ARMS = ['baseline', 'person', 'paint', 'paint_person', 'paint_observations', 'person_observations']
STAGE2_COLUMNS = ['case_id', 'label', 'arm', 'pair_id', 'groups', 'direction_fit_px', 'axis', 'coordinates',
                  'enumerated', 'supported', 'supported_and_players', 'distinct', 'kept', 'cap_threshold_score',
                  'nearest_kept_by_kept_px', 'nearest_distinct_by_distinct_px',
                  *[f'best_{step}_px' for step in STEPS], *[f'best_{step}_rank' for step in STEPS]]


def fragment_masks(source: dict, frame: np.ndarray, scale: np.ndarray) -> dict[str, np.ndarray]:
    """Keep masks over the pack's fragments for every arm (True keeps the fragment)."""
    fragments = np.asarray(source['segments_px'], dtype=float)
    contrast, saturation = features(profiles(frame, fragments, scale))[:, :2].T
    midpoints = (fragments[:, :2] + fragments[:, 2:]) / 2
    boxes = np.asarray(source['bbox_px'], dtype=float).reshape(-1, 4)
    inside_x = (midpoints[:, None, 0] >= np.minimum(boxes[:, 0], boxes[:, 2])) & (midpoints[:, None, 0] <= np.maximum(boxes[:, 0], boxes[:, 2]))
    inside_y = (midpoints[:, None, 1] >= np.minimum(boxes[:, 1], boxes[:, 3])) & (midpoints[:, None, 1] <= np.maximum(boxes[:, 1], boxes[:, 3]))
    outside_people = ~(inside_x & inside_y).any(axis=1)
    paint = (contrast >= PAINT_CONTRAST) & (saturation <= PAINT_SATURATION)
    masks = {'baseline': np.ones(len(fragments), dtype=bool), 'person': outside_people, 'paint': paint,
             'paint_person': paint & outside_people}
    for threshold in SENSITIVITY_CONTRASTS:
        masks[f'paint{threshold:.0f}'] = (contrast >= threshold) & (saturation <= PAINT_SATURATION)
    return masks


def filtered_source(source: dict, keep: np.ndarray) -> dict:
    return {**source, 'segments_px': [segment for segment, kept in zip(source['segments_px'], keep, strict=True) if kept]}


def control_vanishing_points(control: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The control court's two vanishing points as unit vectors in the selection's normalised chart."""
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M.astype(np.float32), control.astype(np.float32))
    normaliser = vp_pruning.normalisation(size)
    directions = np.linalg.solve(normaliser, np.asarray(homography, dtype=float)[:, :2])
    return (directions / np.linalg.norm(directions, axis=0)).T


def angles_to_control(points_working: np.ndarray, control_directions: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    normaliser = vp_pruning.normalisation(size)
    normalised = np.linalg.solve(normaliser, points_working.T).T
    normalised /= np.linalg.norm(normalised, axis=1)[:, None]
    cosines = np.clip(np.abs(normalised @ control_directions.T), 0., 1.)
    return np.degrees(np.arccos(cosines)).min(axis=0)


def gate_baseline(segments: np.ndarray, size: tuple[int, int], saved: dict) -> vp_pruning.Settings:
    """The unfiltered replay must reproduce the saved direction record exactly."""
    settings = vp_pruning.Settings(**saved['settings'])
    assert list(size) == saved['working_size'], (size, saved['working_size'])
    _, replayed = vp_pruning.estimate(segments, size, settings)
    estimator = saved['estimator']
    assert replayed['retained_candidate_ids'] == estimator['retained_candidate_ids'], 'retained candidate IDs differ'
    np.testing.assert_allclose(replayed['points_working'], estimator['points_working'], rtol=0, atol=FLOAT_ATOL)
    assert replayed['retained_support_masks'] == estimator['retained_support_masks'], 'support masks differ'
    return settings


def load_case_inputs(case_id: str) -> tuple[dict, np.ndarray, np.ndarray, tuple[int, int], dict, vp_pruning.Settings]:
    """Load one pack entry and its saved baseline coverage inputs."""
    source = read(PACKS[PACK_OF[case_id]])
    source = next(case for case in source['cases'] if case['id'] == case_id)
    image_path = frame_path(source)
    frame = cv2.imread(str(image_path))
    assert frame is not None, image_path
    segments, _, size = prepare(source)
    saved = load_estimator(case_id)
    settings = gate_baseline(segments, size, saved)
    return source, frame, segments, size, saved, settings


def write_observation_inputs_only(case_ids: list[str], inputs_dir: Path) -> None:
    """Write paint-observation matcher inputs without reading control or diagnostic records."""
    for case_id in case_ids:
        source, frame, _, size, saved, settings = load_case_inputs(case_id)
        scale = np.asarray([source['dimensions']['width'], source['dimensions']['height']], dtype=float) / size
        masks = fragment_masks(source, frame, scale)
        write_inputs(inputs_dir, 'paint_observations', case_id, source, masks['paint'], saved['estimator'], settings, size)


def stage_one(case_id: str, arm: str, source: dict, keep: np.ndarray, settings: vp_pruning.Settings, control: np.ndarray,
              control_directions: np.ndarray, baseline_ids: list[int]) -> tuple[dict, dict, np.ndarray, dict]:
    """Filtered fragments through the unchanged merge and selection; measures against the control."""
    filtered = filtered_source(source, keep)
    segments, _, size = prepare(filtered)
    all_segments, _, _ = prepare(source)
    _, estimator = vp_pruning.estimate(segments, size, settings)
    points = np.asarray(estimator['points_working'], dtype=float)
    on_marking = (marking_distance(all_segments[:, :2], control) <= EDGE_PX) & (marking_distance(all_segments[:, 2:], control) <= EDGE_PX)
    fits = fit_pairs(points, control) if len(points) >= 2 else None
    best = fits['best_finite'] if fits else None
    bound = set_bound(control, points)[0] if len(points) >= 2 else None
    angles = angles_to_control(points, control_directions, size) if len(points) else [None, None]
    row = {'case_id': case_id, 'label': LABELS[case_id], 'arm': arm, 'fragments': len(keep), 'dropped': int((~keep).sum()),
           'marking_fragments': int(on_marking.sum()), 'marking_dropped': int((on_marking & ~keep).sum()),
           'merged_lines': len(estimator['direction_lines']), 'directions': len(points),
           'angle_to_control_x_deg': None if angles[0] is None else round(float(angles[0]), 4),
           'angle_to_control_y_deg': None if angles[1] is None else round(float(angles[1]), 4),
           'bound_px': None if bound is None else round(float(bound), 4),
           'best_fit_px': None if best is None else round(best['max_corner_working_px'], 4),
           'best_fit_pair': None if best is None else best['pair_id'],
           'best_fit_groups': None if best is None else ' '.join(map(str, best['groups'])),
           'selection_identical_to_baseline': estimator['retained_candidate_ids'] == baseline_ids}
    return row, estimator, points, best


def axis_stage(case_id: str, arm: str, source: dict, keep: np.ndarray, pair_points: np.ndarray, fit: dict,
               control: np.ndarray, uncapped: bool) -> list[dict]:
    """The axis-matching ladder for one direction pair on the filtered observations (axis_replay.py's measure)."""
    filtered = filtered_source(source, keep)
    segments, _, size = prepare(filtered)
    observations = assignment.prepare_observations(segments, size)
    feet = feet_working(source, size)
    basis, _details = basis_for(pair_points, size, MATCHER_SETTINGS)
    if basis is None:
        return [{'case_id': case_id, 'label': LABELS[case_id], 'arm': arm, 'pair_id': fit['pair_id'],
                 'groups': ' '.join(map(str, fit['groups'])), 'direction_fit_px': round(fit['max_corner_working_px'], 4),
                 'axis': 'none', 'coordinates': None, **dict.fromkeys(STAGE2_COLUMNS[8:], None)}]
    homography = np.asarray(fit['homography_working'], dtype=float)
    ideal, _ = min((ideal_parameters(basis, homography), ideal_parameters(basis, homography @ SYMMETRY)), key=lambda item: item[1])
    ideal_error = float(corner_errors(courts_from(basis, ideal[:1], ideal[1:]), control)[0])
    assert abs(ideal_error - fit['max_corner_working_px']) < 1e-6, (ideal_error, fit['max_corner_working_px'])
    rows, matched = [], {}
    for axis, (name, coordinates) in enumerate(zip(AXIS_NAMES, AXIS_COORDINATES)):
        matches = match_axis(basis, axis, coordinates, observations, size, UNCAPPED, feet)
        matched[axis] = matches
        errors = sweep(basis, axis, matches.parameters, ideal[1 - axis], control)
        rank = np.full(len(matches.parameters), -1)
        rank[matches.distinct] = np.arange(len(matches.distinct))
        masks = step_masks(matches, MATCHER_SETTINGS.keep_axes)
        kept_ids = matches.distinct[:MATCHER_SETTINGS.keep_axes]
        row = {'case_id': case_id, 'label': LABELS[case_id], 'arm': arm, 'pair_id': fit['pair_id'],
               'groups': ' '.join(map(str, fit['groups'])), 'direction_fit_px': round(fit['max_corner_working_px'], 4),
               'axis': name, 'coordinates': len(coordinates), **{step: int(mask.sum()) for step, mask in masks.items()},
               'cap_threshold_score': round(float(matches.scores[kept_ids[-1]]), 4) if len(kept_ids) else None}
        for step, mask in masks.items():
            if not mask.any():
                row[f'best_{step}_px'], row[f'best_{step}_rank'] = None, None
                continue
            index = int(np.flatnonzero(mask)[np.argmin(errors[mask])])
            row[f'best_{step}_px'], row[f'best_{step}_rank'] = round(float(errors[index]), 4), int(rank[index])
        rows.append(row)
    kept = [matched[axis].parameters[matched[axis].distinct[:MATCHER_SETTINGS.keep_axes]] for axis in (0, 1)]
    nearest_kept = nearest_over_product(basis, kept[0], kept[1], control) if all(len(side) for side in kept) else None
    nearest_uncapped = None
    if uncapped:
        distinct = [matched[axis].parameters[matched[axis].distinct] for axis in (0, 1)]
        nearest_uncapped = nearest_over_product(basis, distinct[0], distinct[1], control) if all(len(side) for side in distinct) else None
    for row in rows:
        row['nearest_kept_by_kept_px'] = None if nearest_kept is None else round(nearest_kept, 4)
        row['nearest_distinct_by_distinct_px'] = None if nearest_uncapped is None else round(nearest_uncapped, 4)
    return rows


def write_inputs(inputs_dir: Path, arm: str, case_id: str, source: dict, keep: np.ndarray, estimator: dict,
                 settings: vp_pruning.Settings, size: tuple[int, int]) -> None:
    """The filtered pack entry and direction record the matcher run reads for this arm and view."""
    folder = inputs_dir / arm
    write(folder / 'cases' / f'{case_id}.json.gz', filtered_source(source, keep))
    write(folder / 'estimators' / f'{case_id}.json.gz', {
        'case_id': case_id, 'working_size': list(size), 'settings': asdict(settings), 'estimator': estimator,
        'arm': arm, 'fragments_kept': int(keep.sum()), 'fragments_total': len(keep),
        # Indices into the pack's segments_px, so a reader can map the filtered list back to raw fragment IDs.
        'kept_fragment_ids': np.flatnonzero(keep).tolist(),
        'note': (f'Baseline directions (the saved estimator) with {OBSERVATION_ARMS[arm]}-filtered fragments; see filter_replay.py.'
                 if arm in OBSERVATION_ARMS else
                 'Directions selected by the unchanged coverage rule from the filtered fragments; see filter_replay.py.')})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        help='directory for the full replay tables; not used by input-only mode')
    parser.add_argument('--cases', nargs='+',
                        help='case IDs; input-only mode requires this option explicitly')
    parser.add_argument('--uncapped-combination', action='store_true')
    parser.add_argument('--inputs-dir', type=Path, default=Path(__file__).resolve().parent / 'inputs')
    parser.add_argument('--axis-arms', nargs='+',
                        help='arms that also run the axis-matching stage')
    parser.add_argument('--write-observation-inputs-only', action='store_true',
                        help='write paint_observations inputs for explicitly named cases and stop')
    args = parser.parse_args(argv)
    if args.write_observation_inputs_only:
        if args.cases is None:
            parser.error('--write-observation-inputs-only requires explicit --cases')
        if args.axis_arms is not None and args.axis_arms != ['paint_observations']:
            parser.error('--write-observation-inputs-only only accepts --axis-arms paint_observations')
        if args.uncapped_combination:
            parser.error('--uncapped-combination is only valid for the full replay')
    elif args.output is None:
        parser.error('--output is required for the full replay')
    args.cases = list(CASE_IDS) if args.cases is None else args.cases
    if args.write_observation_inputs_only:
        args.axis_arms = ['paint_observations'] if args.axis_arms is None else args.axis_arms
    else:
        args.axis_arms = list(DEFAULT_AXIS_ARMS) if args.axis_arms is None else args.axis_arms
    return args


def main() -> None:
    args = parse_args()
    if args.write_observation_inputs_only:
        write_observation_inputs_only(args.cases, args.inputs_dir)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    stage1_rows, stage2_rows = [], []
    for case_id in args.cases:
        source, frame, segments, size, saved, settings = load_case_inputs(case_id)
        control, _ = control_corners(case_id)
        control_directions = control_vanishing_points(control, size)
        baseline_ids = saved['estimator']['retained_candidate_ids']
        scale = np.asarray([source['dimensions']['width'], source['dimensions']['height']], dtype=float) / size
        masks = fragment_masks(source, frame, scale)
        print(f'{LABELS[case_id]}: gate passed (unfiltered replay equals the saved direction record); '
              f'{len(segments)} fragments, {len(source["bbox_px"])} person boxes', flush=True)
        best_by_arm, points_by_arm = {}, {}
        for arm in ARMS:
            row, estimator, points, best = stage_one(case_id, arm, source, masks[arm], settings, control, control_directions, baseline_ids)
            stage1_rows.append(row)
            best_by_arm[arm], points_by_arm[arm] = best, points
            if arm == 'baseline':
                # The observation-only arms: the saved baseline directions with the filtered fragments.
                for observation_arm, filter_arm in OBSERVATION_ARMS.items():
                    write_inputs(args.inputs_dir, observation_arm, case_id, source, masks[filter_arm], saved['estimator'], settings, size)
            else:
                write_inputs(args.inputs_dir, arm, case_id, source, masks[arm], estimator, settings, size)
            print(f'  {arm:13s} dropped {row["dropped"]:4d} of {row["fragments"]} (on markings {row["marking_dropped"]} of {row["marking_fragments"]}); '
                  f'lines {row["merged_lines"]:3d}; angles to control x {row["angle_to_control_x_deg"]} y {row["angle_to_control_y_deg"]}; '
                  f'bound {row["bound_px"]}; best fit {row["best_fit_px"]} (pair {row["best_fit_pair"]})'
                  f'{"; selection identical to baseline" if row["selection_identical_to_baseline"] else ""}', flush=True)
        for arm in args.axis_arms:
            if arm in OBSERVATION_ARMS:
                fit = load_direction_record('e3', case_id)['sets']['B']['summary']['best_finite']
                fit = load_direction_record('e3', case_id)['sets']['B']['fits']['records'][fit['pair_id']]
                pair_points, keep = points_by_arm['baseline'][fit['groups']], masks[OBSERVATION_ARMS[arm]]
            else:
                fit = best_by_arm[arm]
                if fit is None:
                    continue
                pair_points, keep = points_by_arm[arm][fit['groups']], masks[arm]
            rows = axis_stage(case_id, arm, source, keep, pair_points, fit, control, args.uncapped_combination)
            stage2_rows.extend(rows)
            for row in rows:
                if row['axis'] == 'none':
                    print(f'  axis {arm}: degenerate basis for pair {row["pair_id"]}', flush=True)
                    continue
                uncapped = row['nearest_distinct_by_distinct_px']
                print(f'  axis {arm:23s} pair {row["pair_id"]:3d} fit {row["direction_fit_px"]:7.3f} {row["axis"]:10s} '
                      f'distinct {row["best_distinct_px"]} (rank {row["best_distinct_rank"]} of {row["distinct"]}) '
                      f'kept {row["best_kept_px"]} (rank {row["best_kept_rank"]}); kept x kept {row["nearest_kept_by_kept_px"]}'
                      + ('' if uncapped is None else f'; uncapped {uncapped}'), flush=True)
    with open(args.output / 'table.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=STAGE1_COLUMNS)
        writer.writeheader()
        writer.writerows(stage1_rows)
    with open(args.output / 'axis_table.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=STAGE2_COLUMNS)
        writer.writeheader()
        writer.writerows(stage2_rows)
    with open(args.output / 'settings.json', 'w') as stream:
        json.dump({'paint_contrast': PAINT_CONTRAST, 'paint_saturation': PAINT_SATURATION,
                   'sensitivity_contrasts': SENSITIVITY_CONTRASTS, 'arms': ARMS, 'edge_px': EDGE_PX}, stream, indent=1)
    print(f'wrote {args.output / "table.csv"} ({len(stage1_rows)} rows) and axis_table.csv ({len(stage2_rows)} rows)')


if __name__ == '__main__':
    main()
