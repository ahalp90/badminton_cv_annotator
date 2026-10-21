"""Condense the dropped search arms from the archived run records into ``screening-evidence.json.gz``.

Appendix A of the report describes search arms whose outputs are not tracked: the backend pilot,
the first grid, the label and option-list comparison, the smaller-tree, specialist, keep-margin
and 28-video contact checks, and the option-list ceiling. Appendix B adds the frame-rate contact
diagnostics. This script reads their records from the local archive and keeps the counts behind
every number those appendices quote. The six-setting screen and the five-seed repeats are in
``five-seed-evidence.json.gz`` already.

``--archive`` is the unpacked ``raw-data.tar.zst``. Its ``remote/`` folder holds the run records,
``fps-k5/`` and ``fps-*-v1/`` hold the frame-rate runs, and ``pool-diagnostics.json.gz`` and
``baseline-noise.json.gz`` sit at its top level. Source paths in the output are members of that
archive, so they keep the folder names the runs were written with. The tables below map each
folder to a plain label.

Every arm's counts are summed from its per-video records, and the sums must equal the totals the
run recorded. All arms are single seed-0 fits on the eight validation videos (668 rallies).
``remote/tightest-legacy`` and its keep-margin replay are left out: they are an earlier copy of the
first selector in ``remote/tight-targets/legacy-15``.

Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any, NamedTuple

from experiments.annotator.tree_simplification.analysis.common import (
    RESULTS,
    check,
    pooled_both,
    read_json,
    write_json_gz,
)

TOLERANCES = ('10', '5')
VALIDATION_RALLIES = 668
# Every selector arm in the screen used 200 rounds and learning rate 0.05.
SCREEN_ROUNDS = 200
SCREEN_LEARNING_RATE = 0.05
# Record keys and values as the archived runs wrote them, under the names used here and in the current code.
OLD_RECORD_NAMES = {
    'original_options': 'legacy_pool',
    'five-frame-preferred': 'tightest-available',
    'original': 'legacy',
    'larger': 'expanded',
    'rally_sections': 'total_spans',
    'with_correct_option': 'unique_gt_positive_in_initial_context',
    'frame_union_impossible': 'bounds_possible_frames_impossible',
    'correct_option_with_initial_neighbours': 'positive_in_initial_context',
    'bounds_possible': 'optimistic_bounds_possible',
    'frame_union_possible': 'optimistic_frame_union_possible',
    'scaled': 'normalised',
}


class Arm(NamedTuple):
    """One selector run and the settings its label claims, checked against its record."""

    folder: str
    backend: str
    option_list: str
    training_labels: str
    leaves: int
    min_child_weight: float
    rally_selectors: int
    specialists: tuple[str, ...] = ()


TEN = '±10 frames'
FIVE = '±5 frames where the option list allows, else ±10'
# Section: plain label: the run folder under remote/ and its settings.
SELECTOR_ARMS = {
    'first_grid_xgboost': {
        'original-options-15-leaves': Arm('search-xgb/legacy-pool', 'xgboost', 'original', TEN, 15, 5.0, 3),
        'larger-options-7-leaves': Arm('search-xgb/leaves-7', 'xgboost', 'larger', TEN, 7, 5.0, 3),
        'larger-options-15-leaves': Arm('search-xgb/leaves-15', 'xgboost', 'larger', TEN, 15, 5.0, 3),
        'larger-options-31-leaves': Arm('search-xgb/leaves-31', 'xgboost', 'larger', TEN, 31, 5.0, 3),
        'larger-options-15-leaves-child-weight-10': Arm('search-xgb/child-10', 'xgboost', 'larger', TEN, 15, 10.0, 3),
    },
    'first_grid_hgb': {
        'original-options-15-leaves': Arm('search-hgb/legacy-pool', 'hist_gradient_boosting', 'original', TEN, 15, 5.0, 3),
        'larger-options-7-leaves': Arm('search-hgb/leaves-7', 'hist_gradient_boosting', 'larger', TEN, 7, 5.0, 3),
        'larger-options-15-leaves': Arm('search-hgb/leaves-15', 'hist_gradient_boosting', 'larger', TEN, 15, 5.0, 3),
        'larger-options-31-leaves': Arm('search-hgb/leaves-31', 'hist_gradient_boosting', 'larger', TEN, 31, 5.0, 3),
    },
    'labels_and_option_lists': {
        'original-options-15-leaves': Arm('tight-targets/legacy-15', 'xgboost', 'original', FIVE, 15, 5.0, 2),
        'larger-options-15-leaves': Arm('tight-targets/expanded-15', 'xgboost', 'larger', FIVE, 15, 5.0, 2),
    },
    'smaller_trees': {
        'larger-options-7-leaves': Arm('tight-targets/expanded-7', 'xgboost', 'larger', FIVE, 7, 5.0, 2),
        'larger-options-15-leaves-third-selector': Arm(
            'tight-targets/expanded-15-depth3', 'xgboost', 'larger', FIVE, 15, 5.0, 3),
    },
    'contact_fit_28_videos': {
        'larger-options-15-leaves': Arm('tight-oof28/expanded-15', 'xgboost', 'larger', FIVE, 15, 5.0, 2),
    },
    'specialists': {
        'serve': Arm('structures-xgb/serve', 'xgboost', 'original', TEN, 15, 5.0, 2, ('serve',)),
        'insertion': Arm('structures-xgb/insertion', 'xgboost', 'original', TEN, 15, 5.0, 2, ('insertion',)),
        'serve-and-insertion': Arm(
            'structures-xgb/both', 'xgboost', 'original', TEN, 15, 5.0, 2, ('insertion', 'serve')),
    },
}
# Keep-margin replay folder under remote/margins/: the section and label of the run it replays.
MARGIN_REPLAYS = {
    'legacy-pool': ('first_grid_xgboost', 'original-options-15-leaves'),
    'leaves-7': ('first_grid_xgboost', 'larger-options-7-leaves'),
    'leaves-15': ('first_grid_xgboost', 'larger-options-15-leaves'),
    'leaves-31': ('first_grid_xgboost', 'larger-options-31-leaves'),
    'tight-legacy-15': ('labels_and_option_lists', 'original-options-15-leaves'),
    'tight-expanded-15': ('labels_and_option_lists', 'larger-options-15-leaves'),
    'tight-expanded-7': ('smaller_trees', 'larger-options-7-leaves'),
    'tight-expanded-15-depth3': ('smaller_trees', 'larger-options-15-leaves-third-selector'),
    'oof28-expanded-15': ('contact_fit_28_videos', 'larger-options-15-leaves'),
    'specialist-serve': ('specialists', 'serve'),
    'specialist-insertion': ('specialists', 'insertion'),
    'specialist-both': ('specialists', 'serve-and-insertion'),
}
KEEP_MARGIN = 0.05
# The old eight-model stack with each backend: the run folder under remote/.
PILOT_RUNS = {
    'old-eight-model-stack-hgb': 'hgb-2core',
    'old-eight-model-stack-xgboost': 'xgb-5',
    'old-eight-model-stack-xgboost-child-weight-10': 'xgb-10',
}
# Frame-rate variant: its folder name. Offset 0 ran in the top-level folder, the rest under fps-k5/.
FRAME_RATE_VARIANTS = {'all-motion-scaled': 'fps-motion-v1', 'velocity-only-scaled': 'fps-velocity-v1'}
# Offset 0's diagnostics come from the last retry of each variant, the first that finished.
SEED_ZERO_DIAGNOSTICS = {
    'all-motion-scaled': 'contact-diagnostics-retry2.json.gz',
    'velocity-only-scaled': 'contact-diagnostics-retry4.json.gz',
}
FRAME_RATE_SEEDS = (0, 100, 200, 300, 400)
# The report's four-seed summary of the contact diagnostics, made before offset 0's retries finished.
DIAGNOSTIC_SUMMARY_SEEDS = (100, 200, 300, 400)


def totals_from_videos(per_video: dict[str, Any], recorded: dict[str, Any], where: str) -> dict[str, Any]:
    """Pool the per-video counts and check them against the totals the run recorded."""
    totals = pooled_both(per_video)
    check(totals, {tolerance: {name: recorded[tolerance][name] for name in counts}
                   for tolerance, counts in totals.items()}, where)
    labelled = {totals[tolerance]['labelled_rallies'] for tolerance in TOLERANCES}
    if labelled != {VALIDATION_RALLIES}:
        raise ValueError(f'{where}: expected {VALIDATION_RALLIES} labelled rallies, found {labelled}')
    return totals


def arm_settings(settings: dict[str, Any], arm: Arm, where: str) -> dict[str, Any]:
    """The run's settings in plain terms; they must match what its label claims."""
    tree = settings['tree']
    policy = settings.get('target_policy')
    if policy is not None and policy['name'] != OLD_RECORD_NAMES['five-frame-preferred']:
        raise ValueError(f'{where}: unexpected label policy {policy["name"]}')
    plain = {
        'backend': tree['backend'],
        'option_list': 'original' if settings[OLD_RECORD_NAMES['original_options']] else 'larger',
        'training_labels': TEN if policy is None else FIVE,
        'leaves': tree['max_leaf_nodes'],
        'min_child_weight': tree['min_child_weight'],
        'rally_selectors': settings['layers'],
        'specialists': tuple(settings.get('specialists', {}).get('roles', ())),
    }
    claimed = {name: getattr(arm, name) for name in plain}
    if plain != claimed or (tree['max_iter'], tree['learning_rate']) != (SCREEN_ROUNDS, SCREEN_LEARNING_RATE):
        raise ValueError(f'{where}: record settings {plain} do not match the label {claimed}')
    return {
        **plain,
        'specialists': list(plain['specialists']),
        'rounds': tree['max_iter'],
        'learning_rate': tree['learning_rate'],
        'training_weight_total': settings['span_weight'],
    }


def selector_runs(remote: Path, folder: str, selectors: int) -> dict[str, Any]:
    """Per-video counts and pooled totals for each selector count of one run."""
    result = read_json(remote / folder / 'result.json.gz')
    if not result['complete'] or sorted(result['layers'], key=int) != [str(depth) for depth in range(1, selectors + 1)]:
        raise ValueError(f'{folder}: expected a complete run with {selectors} selectors')
    by_selectors = {}
    for depth in result['layers']:
        validation = read_json(remote / folder / f'layers-{depth}' / 'validation.json.gz')
        where = f'{folder} selector {depth}'
        check(validation['totals'], result['layers'][depth], f'{where} validation record equals the run result')
        by_selectors[depth] = {
            'per_video': validation['per_video'],
            'totals': totals_from_videos(validation['per_video'], validation['totals'], where),
        }
    return {'result': result, 'selectors': by_selectors}


def selector_sections(remote: Path) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    for section, arms in SELECTOR_ARMS.items():
        sections[section] = {}
        for label, arm in arms.items():
            run = selector_runs(remote, arm.folder, arm.rally_selectors)
            entry = {
                'source': f'remote/{arm.folder}',
                'settings': arm_settings(run['result']['settings'], arm, arm.folder),
                'selectors': run['selectors'],
            }
            if section == 'contact_fit_28_videos':
                entry['contact_folds'] = contact_folds(run['result']['settings'], arm.folder)
            sections[section][label] = entry
    # The third-selector run continues the two-selector run, so its first two selectors must agree.
    third = sections['smaller_trees']['larger-options-15-leaves-third-selector']['selectors']
    two = sections['labels_and_option_lists']['larger-options-15-leaves']['selectors']
    check({depth: third[depth] for depth in two}, two, 'third-selector run repeats the first two selectors')
    return sections


def contact_folds(settings: dict[str, Any], where: str) -> dict[str, Any]:
    """Each training video's contact predictions came from a model fitted without its fold."""
    folds = settings['target_policy']['manifest']['pools_manifest']['contact_folds']
    videos = [video for fold in folds.values() for video in fold]
    if len(folds) != 8 or {len(fold) for fold in folds.values()} != {4} or len(set(videos)) != 32:
        raise ValueError(f'{where}: expected eight folds of four training videos')
    return {'folds': len(folds), 'videos_per_fold': 4, 'videos_per_contact_fit': len(videos) - 4}


def keep_margin(remote: Path, sections: dict[str, Any]) -> dict[str, Any]:
    """Margin 0.05 replays. Margin 0 reproduced each source run, so its counts are the source run's."""
    replays: dict[str, Any] = {}
    for folder, (section, label) in MARGIN_REPLAYS.items():
        source = sections[section][label]
        replay = read_json(remote / 'margins' / folder / f'margin-{KEEP_MARGIN}' / 'result.json.gz')
        source_result = read_json(remote / source['source'].removeprefix('remote/') / 'result.json.gz')
        where = f'margins/{folder}'
        if not (replay['complete'] and replay['zero_margin_parity'] and replay['unchanged_intermediate_history']):
            raise ValueError(f'{where}: the replay is incomplete or margin 0 did not reproduce its source run')
        if replay['margin'] != KEEP_MARGIN or not replay['source_run'].endswith('/' + source['source'].removeprefix('remote/')):
            raise ValueError(f'{where}: replays {replay["source_run"]} at margin {replay["margin"]}')
        check(replay['source_settings'], source_result['settings'], f'{where} source settings')
        by_selectors = {}
        for depth in replay['layers']:
            validation = read_json(remote / 'margins' / folder / f'margin-{KEEP_MARGIN}' / f'layers-{depth}'
                                   / 'validation.json.gz')
            totals = totals_from_videos(validation['per_video'], replay['layers'][depth], f'{where} selector {depth}')
            by_selectors[depth] = {'per_video': validation['per_video'], 'totals': totals}
        replays.setdefault(section, {})[label] = {
            'source': f'remote/margins/{folder}/margin-{KEEP_MARGIN}',
            'replays': {'section': section, 'label': label},
            'margin': KEEP_MARGIN,
            'margin_0_reproduced_source_run': True,
            'selectors': by_selectors,
        }
    return replays


def backend_pilot(remote: Path, archive: Path) -> dict[str, Any]:
    """The old eight-model stack fitted with HGB and with XGBoost, with end-to-end times."""
    pilot: dict[str, Any] = {}
    for label, folder in PILOT_RUNS.items():
        result = read_json(remote / folder / 'result.json.gz')
        pilot[label] = {
            'source': f'remote/{folder}',
            'contact_model_settings': result['settings']['contact'],
            'threads': result['threads'],
            'training_device': result['training_device'],
            'prediction_device': result['prediction_device'],
            'fit_and_validation_seconds': result['total_wall_s'],
            'validation_seconds': result['wall_s'],
            'per_video': result['per_video'],
            'totals': totals_from_videos(result['per_video'], result['totals'], folder),
        }
    pilot['hgb_time_over_xgboost_time'] = (pilot['old-eight-model-stack-hgb']['fit_and_validation_seconds']
                                           / pilot['old-eight-model-stack-xgboost']['fit_and_validation_seconds'])

    # Five earlier HGB fits of the old stack. The first must be the HGB pilot itself.
    earlier = read_json(archive / 'baseline-noise.json.gz')
    seeds = {tolerance: {name: earlier[tolerance][name]['values'] for name in ('attributed_f1', 'fully_correct_rallies')}
             for tolerance in TOLERANCES}
    pilot_totals = pilot['old-eight-model-stack-hgb']['totals']
    check({tolerance: {name: values[0] for name, values in seeds[tolerance].items()} for tolerance in TOLERANCES},
          {tolerance: {name: pilot_totals[tolerance][name] for name in ('attributed_f1', 'fully_correct_rallies')}
           for tolerance in TOLERANCES}, 'first earlier HGB seed equals the HGB pilot')
    pilot['earlier_hgb_seeds'] = {
        'source': 'baseline-noise.json.gz',
        'about': 'The old eight-model stack fitted with HGB at five seeds before this experiment. The first is the HGB pilot.',
        **seeds,
    }
    return pilot


def option_list_ceiling(remote: Path, archive: Path) -> dict[str, Any]:
    """Option-list sizes and the probe of which validation rallies any choice could get right."""
    diagnostics = read_json(archive / 'pool-diagnostics.json.gz')
    option_lists = {
        split: {
            'videos': values['videos'],
            'options': values['options'],
            'rally_sections': values['coverage'][OLD_RECORD_NAMES['rally_sections']],
            'sections_with_a_correct_option': {
                name: values['coverage'][OLD_RECORD_NAMES[name]] for name in ('original', 'larger')
            },
        }
        for split, values in diagnostics.items()
    }
    probe = read_json(remote / 'candidate-pool-coverage.json.gz')
    per_video = {}
    for video in probe['per_video']:
        rallies = [
            {
                'rally': row['rally'],
                'all_sides_known': row['all_sides_known'],
                **{name: row[OLD_RECORD_NAMES[name]]
                   for name in ('correct_option_with_initial_neighbours', 'bounds_possible', 'frame_union_possible')},
            }
            for row in video['rallies']
        ]
        counts = {
            'labelled_rallies': len(rallies),
            'with_correct_option': sum(row['correct_option_with_initial_neighbours'] for row in rallies),
            'bounds_impossible': sum(not row['bounds_possible'] for row in rallies),
            'frame_union_impossible': sum(row['bounds_possible'] and not row['frame_union_possible'] for row in rallies),
        }
        recorded = {
            'labelled_rallies': video['labelled_rallies'],
            'with_correct_option': video[OLD_RECORD_NAMES['with_correct_option']],
            'bounds_impossible': video['bounds_impossible'],
            'frame_union_impossible': video[OLD_RECORD_NAMES['frame_union_impossible']],
        }
        check(counts, recorded, f'option-list probe {video["video"]}')
        per_video[video['video']] = {**counts, 'rallies': rallies}
    totals = {name: sum(video[name] for video in per_video.values())
              for name in ('labelled_rallies', 'with_correct_option', 'bounds_impossible', 'frame_union_impossible')}
    validation = option_lists['validation']['sections_with_a_correct_option']['larger']
    if totals['labelled_rallies'] != VALIDATION_RALLIES or totals['with_correct_option'] != validation:
        raise ValueError(f'The probe totals {totals} disagree with the option-list diagnostics')
    return {
        'option_lists': {'source': 'pool-diagnostics.json.gz', **option_lists},
        'probe': {
            'source': 'remote/candidate-pool-coverage.json.gz',
            'about': ('Each rally is checked against the larger option lists from the seed-0 XGBoost contact model. '
                      'A correct option assumes neighbouring sections stay as they started. Bounds and frame-union '
                      'checks are generous necessary conditions, so failing either makes the rally unreachable.'),
            'per_video': per_video,
            'totals': totals,
            'most_rallies_any_choice_can_get_right': (totals['labelled_rallies'] - totals['bounds_impossible']
                                                      - totals['frame_union_impossible']),
        },
    }


def diagnostics_path(archive: Path, variant: str, seed: int) -> Path:
    folder = FRAME_RATE_VARIANTS[variant]
    if seed == 0:
        return archive / folder / 'seed-0' / SEED_ZERO_DIAGNOSTICS[variant]
    return archive / 'fps-k5' / folder / f'seed-{seed}' / 'contact-diagnostics.json.gz'


def pooled_by_split_and_rate(videos: dict[str, Any]) -> dict[str, Any]:
    """Sum per-video counts within each split and frame rate, then add correct-player F1."""
    pooled: dict[str, Any] = {}
    for video in videos.values():
        split = 'validation' if video['group'] == 'V' else 'train'
        rate = f'{video["fps"]:g}'
        for tolerance in TOLERANCES:
            counts = pooled.setdefault(split, {}).setdefault(rate, {}).setdefault(tolerance, {})
            for name, value in video['scores'][tolerance]['counts'].items():
                counts[name] = counts.get(name, 0) + value
    for split in pooled.values():
        for rate in split.values():
            for counts in rate.values():
                counts['attributed_f1'] = (2 * counts['correct_side_contacts']
                                           / (counts['predicted_contacts'] + counts['labelled_contacts']))
    return pooled


def frame_rate_diagnostics(archive: Path) -> dict[str, Any]:
    """Raw and scaled contact models scored on their own initial sections, by split and frame rate."""
    variants: dict[str, Any] = {}
    for variant in FRAME_RATE_VARIANTS:
        per_seed = {}
        for seed in FRAME_RATE_SEEDS:
            path = diagnostics_path(archive, variant, seed)
            record = read_json(path)
            arms = {'raw': record['arms']['raw'], 'scaled': record['arms'][OLD_RECORD_NAMES['scaled']]}
            pooled = {}
            for arm, values in arms.items():
                pooled[arm] = pooled_by_split_and_rate(values['videos'])
                recorded = {
                    split: {rate: {tolerance: {**scores['counts'], 'attributed_f1': scores['metrics']['attributed_f1']}
                                   for tolerance, scores in by_tolerance.items()}
                            for rate, by_tolerance in split_values['by_fps'].items()}
                    for split, split_values in values['pooled']['by_split'].items()
                }
                check(pooled[arm], recorded, f'{variant} seed {seed} {arm} pooled diagnostics')
            difference = {
                split: {rate: {tolerance: 100 * (pooled['scaled'][split][rate][tolerance]['attributed_f1']
                                                 - pooled['raw'][split][rate][tolerance]['attributed_f1'])
                               for tolerance in TOLERANCES}
                        for rate in pooled['raw'][split]}
                for split in pooled['raw']
            }
            per_seed[str(seed)] = {
                'source': str(path.relative_to(archive)),
                **pooled,
                'scaled_minus_raw_f1_points': difference,
            }
        first = per_seed[str(DIAGNOSTIC_SUMMARY_SEEDS[0])]['scaled_minus_raw_f1_points']
        summary = {
            split: {rate: {tolerance: mean(per_seed[str(seed)]['scaled_minus_raw_f1_points'][split][rate][tolerance]
                                           for seed in DIAGNOSTIC_SUMMARY_SEEDS)
                           for tolerance in TOLERANCES}
                    for rate in first[split]}
            for split in first
        }
        variants[variant] = {'per_seed': per_seed, 'mean_scaled_minus_raw_f1_points_offsets_100_to_400': summary}
    return {
        'about': ('Training videos are scored with the cross-fitted contact models and validation videos with the '
                  'final contact model, so training scores are not unbiased. Offset 0 is kept per seed but left '
                  'out of the four-seed mean, which the report quotes.'),
        'variants': variants,
    }


def build(archive: Path) -> dict[str, Any]:
    remote = archive / 'remote'
    sections = selector_sections(remote)
    return {
        'schema': 'tree-simplification-screening-evidence/1',
        'about': __doc__.split('Run from the repository root')[0].strip(),
        'validation_rallies': VALIDATION_RALLIES,
        'backend_pilot': backend_pilot(remote, archive),
        **sections,
        'keep_margin': keep_margin(remote, sections),
        'option_list_ceiling': option_list_ceiling(remote, archive),
        'frame_rate_contact_diagnostics': frame_rate_diagnostics(archive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive', type=Path, required=True, help='the unpacked raw-data archive')
    parser.add_argument('--results', type=Path, default=RESULTS, help='tracked results folder')
    parser.add_argument('--output', type=Path, required=True, help='gzip JSON file to write')
    args = parser.parse_args()
    document = build(args.archive)
    write_json_gz(args.output, document)
    tracked = args.results / 'screening-evidence.json.gz'
    if tracked.exists():
        check(document, read_json(tracked), 'screening-evidence.json.gz')


if __name__ == '__main__':
    main()
