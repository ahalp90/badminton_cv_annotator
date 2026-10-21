"""Re-score the tracked validation predictions and check the selector, frame-rate and ranking evidence.

Subcommands:

- ``selectors``: re-scores ``streams/selection/`` and rebuilds the per-seed results, means and
  paired differences in ``five-seed-evidence.json.gz`` and ``selector-settings-evidence.json.gz``.
- ``frame-rate``: re-scores ``streams/fps/`` and rebuilds ``fps-normalisation-evidence.json.gz``.
- ``ranking``: rebuilds the six review-ranking groups in ``selected-production.json.gz`` from
  ``selected-review-rows.json.gz``.

Run from the repository root with ``PYTHONPATH=src:.``. Each subcommand exits non-zero on the
first difference from the committed evidence.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from experiments.annotator.development import pooled_scores
from experiments.annotator.tree_simplification.analysis.common import (
    RESULTS,
    SEED_OFFSETS,
    SHOTS_MASTER,
    TOLERANCES,
    attributed_f1,
    check,
    pooled_both,
    pooled_counts,
    read_json,
    rescore_folders,
    validation_videos,
)
from experiments.annotator.tree_simplification.integrate import ranking_metrics

OLD_STACK = 'old-eight-model-stack-xgboost'
SELECTED = '31-leaves-400-rounds-1-selector'
SELECTION_LABELS = (
    OLD_STACK,
    '7-leaves-200-rounds-1-selector', '7-leaves-200-rounds-2-selectors',
    '15-leaves-200-rounds-1-selector', '15-leaves-200-rounds-2-selectors',
    '31-leaves-200-rounds-1-selector', '31-leaves-200-rounds-2-selectors',
    SELECTED, '31-leaves-400-rounds-2-selectors',
)
# Six-setting screen rows whose selectors are also tracked prediction files at seed 0.
SCREEN_STREAMS = {
    '31-leaves-200-rounds': ('31-leaves-200-rounds-1-selector', '31-leaves-200-rounds-2-selectors'),
    '31-leaves-400-rounds': (SELECTED, '31-leaves-400-rounds-2-selectors'),
}
SCREEN_REFERENCE = ('15-leaves-200-rounds-1-selector', '15-leaves-200-rounds-2-selectors')
FRAME_RATE_ARMS = ('all-motion-scaled', 'velocity-only-scaled')
# Fields that record how a run was made. They are copied facts, not results to rebuild.
COPIED_FACTS = (
    'source_files', 'source_validation', 'reported_totals_verified', 'result_summary_verified',
    'complete', 'same_pool_and_target_contract', 'tree', 'description', 'sources',
)


def seed_label(seed: int) -> str:
    return f'seed-{seed}'


def per_video_f1(counts: dict[str, Any]) -> dict[str, Any]:
    return {tolerance: {**counts[tolerance], 'attributed_f1': attributed_f1(counts[tolerance])} for tolerance in TOLERANCES}


def selection_scores(results: Path, shots_master: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-video counts keyed by selector label, then seed offset, then video."""
    root = results / 'streams/selection'
    found = sorted(path.name for path in root.iterdir())
    if found != sorted(SELECTION_LABELS):
        raise ValueError(f'{root}: expected the nine selection labels, found {found}')
    folders = {
        f'{label}/{seed}': root / label / seed_label(seed) for label in SELECTION_LABELS for seed in SEED_OFFSETS
    }
    scored = rescore_folders(folders, validation_videos(shots_master))
    return {label: {str(seed): scored[f'{label}/{seed}'] for seed in SEED_OFFSETS} for label in SELECTION_LABELS}


def five_seed_paired(candidate: dict[str, Any], old: dict[str, Any], tolerance: str) -> dict[str, Any]:
    """The five-seed file's paired summary of one candidate against the old stack."""
    per_seed = []
    for seed in SEED_OFFSETS:
        old_pooled = pooled_counts(old[str(seed)], tolerance)
        candidate_pooled = pooled_counts(candidate[str(seed)], tolerance)
        per_seed.append({
            'seed': seed,
            'old_correct_rallies': old_pooled['fully_correct_rallies'],
            'candidate_correct_rallies': candidate_pooled['fully_correct_rallies'],
            'delta_correct_rallies': candidate_pooled['fully_correct_rallies'] - old_pooled['fully_correct_rallies'],
            'old_attributed_f1_percent': 100 * old_pooled['attributed_f1'],
            'candidate_attributed_f1_percent': 100 * candidate_pooled['attributed_f1'],
            'delta_attributed_f1_percentage_points': 100 * (candidate_pooled['attributed_f1'] - old_pooled['attributed_f1']),
        })
    rally_deltas = [row['delta_correct_rallies'] for row in per_seed]
    f1_deltas = [row['delta_attributed_f1_percentage_points'] for row in per_seed]
    per_video = {}
    for video in old['0']:
        video_rows = []
        for seed in SEED_OFFSETS:
            old_counts = old[str(seed)][video][tolerance]
            candidate_counts = candidate[str(seed)][video][tolerance]
            video_rows.append({
                'seed': seed,
                'delta_correct_rallies': candidate_counts['fully_correct_rallies'] - old_counts['fully_correct_rallies'],
                'delta_attributed_f1_percentage_points': 100 * (attributed_f1(candidate_counts) - attributed_f1(old_counts)),
            })
        per_video[video] = {
            'mean_delta_correct_rallies': mean(row['delta_correct_rallies'] for row in video_rows),
            'mean_delta_attributed_f1_percentage_points': mean(
                row['delta_attributed_f1_percentage_points'] for row in video_rows
            ),
            'per_seed': video_rows,
        }
    return {
        'mean_old_correct_rallies': mean(row['old_correct_rallies'] for row in per_seed),
        'mean_candidate_correct_rallies': mean(row['candidate_correct_rallies'] for row in per_seed),
        'mean_delta_correct_rallies': mean(rally_deltas),
        'range_delta_correct_rallies': [min(rally_deltas), max(rally_deltas)],
        'mean_old_attributed_f1_percent': mean(row['old_attributed_f1_percent'] for row in per_seed),
        'mean_candidate_attributed_f1_percent': mean(row['candidate_attributed_f1_percent'] for row in per_seed),
        'mean_delta_attributed_f1_percentage_points': mean(f1_deltas),
        'range_delta_attributed_f1_percentage_points': [min(f1_deltas), max(f1_deltas)],
        'per_seed': per_seed,
        'per_video_mean_deltas': per_video,
    }


def six_setting_row(per_video: dict[str, Any], reference: dict[str, Any], tolerance: str) -> dict[str, Any]:
    candidate = pooled_counts(per_video, tolerance)
    baseline = pooled_counts(reference, tolerance)
    return {
        'per_video': per_video,
        'candidate_recomputed': candidate,
        'baseline_same_pool_recomputed': baseline,
        'delta_correct_rallies': candidate['fully_correct_rallies'] - baseline['fully_correct_rallies'],
        'delta_attributed_f1_percentage_points': 100 * (candidate['attributed_f1'] - baseline['attributed_f1']),
    }


def check_five_seed(scores: dict[str, Any], evidence: dict[str, Any]) -> None:
    arms = evidence['protocol']['arms']
    seed_results = {
        str(seed): {
            'arms': {
                arm: {'per_video': scores[arm][str(seed)], 'pooled_recomputed': pooled_both(scores[arm][str(seed)])}
                for arm in arms
            },
        }
        for seed in SEED_OFFSETS
    }
    check(seed_results, evidence['seed_results'], 'five-seed seed_results', skip=COPIED_FACTS)
    paired = {
        arm: {tolerance: five_seed_paired(scores[arm], scores[OLD_STACK], tolerance) for tolerance in TOLERANCES}
        for arm in arms if arm != OLD_STACK
    }
    check(paired, evidence['paired_comparisons'], 'five-seed paired_comparisons')
    baseline = {
        str(depth): {'pooled_recomputed': pooled_both(scores[label]['0'])}
        for depth, label in enumerate(SCREEN_REFERENCE, start=1)
    }
    check(baseline, evidence['same_pool_seed_zero_baseline']['layers'], 'five-seed same_pool_seed_zero_baseline',
          skip=COPIED_FACTS)

    screen = evidence['six_setting_screen']
    rebuilt_screen: dict[str, Any] = {}
    for setting, entry in screen.items():
        rebuilt_screen[setting] = {'layers': {}}
        for depth, reference_label in enumerate(SCREEN_REFERENCE, start=1):
            if setting in SCREEN_STREAMS:
                per_video = scores[SCREEN_STREAMS[setting][depth - 1]]['0']
            else:
                # No prediction files are tracked for these four settings, so their saved per-video
                # counts are taken as given; the pooled scores and differences are still rebuilt.
                per_video = entry['layers'][str(depth)]['10']['per_video']
            rebuilt_screen[setting]['layers'][str(depth)] = {
                tolerance: six_setting_row(per_video, scores[reference_label]['0'], tolerance) for tolerance in TOLERANCES
            }
    check(rebuilt_screen, screen, 'five-seed six_setting_screen', skip=COPIED_FACTS)


def selector_paired(candidate: dict[str, Any], reference: dict[str, Any], tolerance: str) -> dict[str, Any]:
    """The selector-settings file's paired summary of one setting against a reference."""
    seed_deltas = []
    for seed in SEED_OFFSETS:
        candidate_pooled = pooled_counts(candidate[str(seed)], tolerance)
        reference_pooled = pooled_counts(reference[str(seed)], tolerance)
        seed_deltas.append({
            'seed': seed,
            'candidate_correct_rallies': candidate_pooled['fully_correct_rallies'],
            'reference_correct_rallies': reference_pooled['fully_correct_rallies'],
            'delta_correct_rallies': candidate_pooled['fully_correct_rallies'] - reference_pooled['fully_correct_rallies'],
            'candidate_attributed_f1': candidate_pooled['attributed_f1'],
            'reference_attributed_f1': reference_pooled['attributed_f1'],
            'delta_attributed_f1_percentage_points': 100 * (
                candidate_pooled['attributed_f1'] - reference_pooled['attributed_f1']
            ),
        })
    rally_deltas = [row['delta_correct_rallies'] for row in seed_deltas]
    f1_deltas = [row['delta_attributed_f1_percentage_points'] for row in seed_deltas]
    per_video = {}
    for video in reference['0']:
        video_rows = []
        for seed in SEED_OFFSETS:
            candidate_counts = candidate[str(seed)][video][tolerance]
            reference_counts = reference[str(seed)][video][tolerance]
            video_rows.append({
                'seed': seed,
                'delta_correct_rallies': (
                    candidate_counts['fully_correct_rallies'] - reference_counts['fully_correct_rallies']
                ),
                'delta_attributed_f1_percentage_points': 100 * (
                    attributed_f1(candidate_counts) - attributed_f1(reference_counts)
                ),
            })
        per_video[video] = {
            'mean_delta_correct_rallies': mean(row['delta_correct_rallies'] for row in video_rows),
            'mean_delta_attributed_f1_percentage_points': mean(
                row['delta_attributed_f1_percentage_points'] for row in video_rows
            ),
            'seed_deltas': video_rows,
        }
    return {
        'seed_deltas': seed_deltas,
        'summary': {
            'mean_candidate_correct_rallies': mean(row['candidate_correct_rallies'] for row in seed_deltas),
            'mean_reference_correct_rallies': mean(row['reference_correct_rallies'] for row in seed_deltas),
            'mean_delta_correct_rallies': mean(rally_deltas),
            'range_delta_correct_rallies': [min(rally_deltas), max(rally_deltas)],
            'mean_candidate_attributed_f1': mean(row['candidate_attributed_f1'] for row in seed_deltas),
            'mean_reference_attributed_f1': mean(row['reference_attributed_f1'] for row in seed_deltas),
            'mean_delta_attributed_f1_percentage_points': mean(f1_deltas),
            'range_delta_attributed_f1_percentage_points': [min(f1_deltas), max(f1_deltas)],
        },
        'per_video_mean_deltas': per_video,
    }


def check_selector_settings(
    scores: dict[str, Any], evidence: dict[str, Any], previous_hgb: dict[str, Any], shots_master: Path,
) -> None:
    videos = validation_videos(shots_master)
    denominators = {
        identity: {
            'labelled_rallies': len(video.rallies),
            'labelled_contacts': sum(len(rally.frames) for rally in video.rallies),
        }
        for identity, video in videos.items()
    }
    check(denominators, evidence['per_video_label_denominators'], 'selector settings per_video_label_denominators')
    unknown_sides = sum(side is None for video in videos.values() for rally in video.rallies for side in rally.sides)
    check(unknown_sides, evidence['unknown_side_labels'], 'selector settings unknown_side_labels')
    seed_arms = {
        str(seed): {
            arm: {
                'per_video': {video: per_video_f1(counts) for video, counts in scores[arm][str(seed)].items()},
                'pooled_recomputed': pooled_both(scores[arm][str(seed)]),
            }
            for arm in evidence['protocol']['arms']
        }
        for seed in SEED_OFFSETS
    }
    check(seed_arms, evidence['seed_arms'], 'selector settings seed_arms')
    paired = {
        name: {
            'candidate': entry['candidate'],
            'reference': entry['reference'],
            'tolerances': {
                tolerance: selector_paired(scores[entry['candidate']], scores[entry['reference']], tolerance)
                for tolerance in TOLERANCES
            },
        }
        for name, entry in evidence['paired_comparisons'].items()
    }
    check(paired, evidence['paired_comparisons'], 'selector settings paired_comparisons', skip=COPIED_FACTS)
    check(
        previous_hgb['mean_validation'], evidence['descriptive_hgb_grid017']['mean_validation'],
        'selector settings descriptive_hgb_grid017 (equals previous-hgb-validation.json.gz)',
    )


def check_selected_production(scores: dict[str, Any], production: dict[str, Any]) -> None:
    """The chosen seed-0 bundle's production replay equals its tracked validation predictions."""
    per_video = scores[SELECTED]['0']
    check({'per_video': per_video, 'totals': pooled_scores(per_video)},
          {'per_video': production['per_video'], 'totals': production['totals']},
          'selected-production per_video and totals')


def run_selectors(results: Path, shots_master: Path) -> None:
    scores = selection_scores(results, shots_master)
    check_five_seed(scores, read_json(results / 'five-seed-evidence.json.gz'))
    check_selector_settings(
        scores, read_json(results / 'selector-settings-evidence.json.gz'),
        read_json(results / 'previous-hgb-validation.json.gz'), shots_master,
    )
    check_selected_production(scores, read_json(results / 'selected-production.json.gz'))


def rate_differences(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Same-seed differences in fully correct rallies and correct-player F1 (percentage points)."""
    result = {}
    for tolerance in TOLERANCES:
        left, right = first[tolerance], second[tolerance]
        for key in ('labelled_contacts', 'labelled_rallies'):
            if left[key] != right[key]:
                raise ValueError(f'Paired runs have different {key}')
        result[tolerance] = {
            'correct_rallies': left['fully_correct_rallies'] - right['fully_correct_rallies'],
            'attributed_f1_pp': 100 * (attributed_f1(left) - attributed_f1(right)),
        }
    return result


def spread(values: list[float]) -> dict[str, Any]:
    return {
        'n': len(values), 'mean': mean(values), 'min': min(values), 'max': max(values),
        'sample_sd': statistics.stdev(values),
    }


def summarise_pairs(pairs: dict[str, Any]) -> dict[str, Any]:
    return {
        tolerance: {
            metric: spread([pair[tolerance][metric] for pair in pairs.values()])
            for metric in ('correct_rallies', 'attributed_f1_pp')
        }
        for tolerance in TOLERANCES
    }


def run_frame_rate(results: Path, shots_master: Path) -> None:
    root = results / 'streams/fps'
    found = sorted(path.name for path in root.iterdir())
    if found != sorted(FRAME_RATE_ARMS):
        raise ValueError(f'{root}: expected the two frame-rate arms, found {found}')
    videos = validation_videos(shots_master)
    folders = {f'{arm}/{seed}': root / arm / seed_label(seed) for arm in FRAME_RATE_ARMS for seed in SEED_OFFSETS}
    scored = rescore_folders(folders, videos)
    selector_evidence = read_json(results / 'selector-settings-evidence.json.gz')
    raw = {}
    for seed in SEED_OFFSETS:
        saved = selector_evidence['seed_arms'][str(seed)][SELECTED]
        raw[str(seed)] = {'per_video': saved['per_video'], 'totals': pooled_both(saved['per_video'])}
    arms = {}
    for arm in FRAME_RATE_ARMS:
        runs = {}
        for seed in SEED_OFFSETS:
            per_video = scored[f'{arm}/{seed}']
            runs[str(seed)] = {'per_video': per_video, 'totals': pooled_both(per_video)}
        pairs = {seed: rate_differences(run['totals'], raw[seed]['totals']) for seed, run in runs.items()}
        arms[arm] = {
            'runs': runs,
            'paired_minus_raw': pairs,
            'paired_summary': summarise_pairs(pairs),
            'per_video_paired_summary': {
                video: summarise_pairs({
                    seed: rate_differences(run['per_video'][video], raw[seed]['per_video'][video])
                    for seed, run in runs.items()
                })
                for video in sorted(videos)
            },
        }
    motion, velocity = (arms[arm]['runs'] for arm in FRAME_RATE_ARMS)
    cross = {str(seed): rate_differences(motion[str(seed)]['totals'], velocity[str(seed)]['totals']) for seed in SEED_OFFSETS}
    rebuilt = {
        'complete': True, 'missing': [], 'seeds': list(SEED_OFFSETS), 'raw': raw, 'arms': arms,
        'all_motion_minus_velocity_only': cross,
        'all_motion_minus_velocity_only_summary': summarise_pairs(cross),
    }
    check(rebuilt, read_json(results / 'fps-normalisation-evidence.json.gz'), 'fps-normalisation-evidence',
          skip=('baseline_source', 'interpretation', 'verification', 'sources'))


def run_ranking(results: Path) -> None:
    rows = read_json(results / 'selected-review-rows.json.gz')
    production = read_json(results / 'selected-production.json.gz')
    groups = sorted({(row['ranking'], row['tolerance']) for row in rows})
    pooled = {}
    per_video: dict[str, dict[str, Any]] = {}
    for ranking, tolerance in groups:
        group_rows = [row for row in rows if row['ranking'] == ranking and row['tolerance'] == tolerance]
        name = f'{ranking}-{tolerance}'
        pooled[name] = ranking_metrics(
            np.asarray([row['correct'] for row in group_rows]), np.asarray([row['score'] for row in group_rows]),
        )
        for video in sorted({row['video'] for row in group_rows}):
            video_rows = [row for row in group_rows if row['video'] == video]
            per_video.setdefault(video, {})[name] = ranking_metrics(
                np.asarray([row['correct'] for row in video_rows], dtype=np.int8),
                np.asarray([row['score'] for row in video_rows], dtype=np.float64),
            )
    check(pooled, production['review'], 'selected-production review (six ranking groups)')
    check(per_video, production['review_per_video'], 'selected-production review_per_video')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('selectors', 'frame-rate', 'ranking'))
    parser.add_argument('--results', type=Path, default=RESULTS, help='tracked results folder')
    parser.add_argument('--shots-master', type=Path, default=SHOTS_MASTER, help='ShuttleSet labels')
    args = parser.parse_args()
    if args.command == 'selectors':
        run_selectors(args.results, args.shots_master)
    elif args.command == 'frame-rate':
        run_frame_rate(args.results, args.shots_master)
    else:
        run_ranking(args.results)


if __name__ == '__main__':
    main()
