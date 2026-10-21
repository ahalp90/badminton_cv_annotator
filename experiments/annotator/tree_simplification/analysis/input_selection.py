"""Rebuild how validation chose the two preprocessing configurations for the test run.

Subcommands:

- ``rank``: re-scores ``streams/input-validation/`` and rebuilds the seed-0 screen of all 20
  configurations, the shortlist, and the five-seed ranking of the shortlist. It checks them
  against ``input-preprocessing-evidence.json.gz``, and checks that the rebuilt rankings hash to
  the ``analysis_sha256`` values the shortlist and the test selection recorded. Writes the
  rebuilt rankings to ``--output``.
- ``choose``: reads that output and checks ``test-run/selection.json.gz``: the configuration
  chosen by each objective, their export records and the md5 of every tracked test bundle.
- ``rows``: rebuilds both review-row files from the per-run rows in the local archive.

Run from the repository root with ``PYTHONPATH=src:.``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from annotator.models import model_files
from experiments.annotator.tree_simplification.analysis.common import (
    METRICS,
    RESULTS,
    SEED_OFFSETS,
    SHOTS_MASTER,
    check,
    md5,
    read_json,
    rescore_folders,
    seed_metrics,
    validation_videos,
    write_json_gz,
)

BASELINE = 'raw_baseline'
OBJECTIVES = ('f1', 'fully_correct_rallies')
# The archived rankings were hashed under their original key names; these map old names to new.
OLD_KEYS = {
    'recovery': 'fully_correct_rallies',
    'recovery_10': 'fully_correct_rallies_10',
    'recovery_5': 'fully_correct_rallies_5',
}
NEW_TO_OLD = {new: old for old, new in OLD_KEYS.items()}


def rank(rows: dict[str, Any], objective: str) -> list[str]:
    """Order by the objective's metrics, then prefer the baseline, fewer changes and the name."""
    if objective == 'f1':
        order = ('f1_10', 'fully_correct_rallies_10', 'f1_5')
    else:
        order = ('fully_correct_rallies_10', 'f1_10', 'fully_correct_rallies_5')

    def key(candidate: str) -> tuple:
        row = rows[candidate]
        return (*(-row['mean'][metric] for metric in order), candidate != BASELINE, row['changes'], candidate)

    return sorted(rows, key=key)


def shortlist_of(rankings: dict[str, list[str]]) -> list[str]:
    """The rule fixed before the sweep: the baseline plus the top two under each objective."""
    return sorted({BASELINE, *rankings['f1'][:2], *rankings['fully_correct_rallies'][:2]})


def analyse(
    scores: dict[str, dict[str, Any]], specs: dict[str, Any], candidates: list[str], offsets: list[int],
) -> dict[str, Any]:
    """Rank configurations by mean metrics over seeds, with paired and leave-one-video-out checks.

    ``candidates`` sets the order of ``rows``, which the recorded hashes depend on.
    """
    rows = {}
    for candidate in candidates:
        spec = specs[candidate]
        per_seed = {str(offset): seed_metrics(scores[candidate][str(offset)]) for offset in offsets}
        rows[candidate] = {
            'changes': len(spec['overrides_base30']) + len(spec['strategies']),
            'per_seed': per_seed,
            'mean': {metric: mean(seed[metric] for seed in per_seed.values()) for metric in METRICS},
        }
    videos = sorted(scores[BASELINE]['0'])
    baseline = rows[BASELINE]
    for candidate, row in rows.items():
        row['paired_delta'] = {
            offset: {metric: values[metric] - baseline['per_seed'][offset][metric] for metric in METRICS}
            for offset, values in row['per_seed'].items()
        }
        row['mean_delta'] = {metric: row['mean'][metric] - baseline['mean'][metric] for metric in METRICS}
        row['per_video_mean_delta'] = {}
        for video in videos:
            deltas = []
            for offset in map(str, offsets):
                actual = seed_metrics({video: scores[candidate][offset][video]})
                control = seed_metrics({video: scores[BASELINE][offset][video]})
                deltas.append({metric: actual[metric] - control[metric] for metric in METRICS})
            row['per_video_mean_delta'][video] = {metric: mean(delta[metric] for delta in deltas) for metric in METRICS}
    rankings = {objective: rank(rows, objective) for objective in OBJECTIVES}
    leave_one_out = {}
    for excluded in videos:
        reduced = {}
        for candidate, row in rows.items():
            per_seed = [
                seed_metrics({video: counts for video, counts in scores[candidate][str(offset)].items() if video != excluded})
                for offset in offsets
            ]
            reduced[candidate] = {
                'changes': row['changes'],
                'mean': {metric: mean(seed[metric] for seed in per_seed) for metric in METRICS},
            }
        leave_one_out[excluded] = {objective: rank(reduced, objective) for objective in OBJECTIVES}
    return {
        'offsets': offsets, 'videos': videos, 'rows': rows, 'rankings': rankings,
        'shortlist': shortlist_of(rankings), 'leave_one_video_out_rankings': leave_one_out,
    }


def with_old_keys(value: Any) -> Any:
    """Rename keys back to the names the recorded hashes were taken under, keeping key order."""
    if isinstance(value, dict):
        return {NEW_TO_OLD.get(key, key): with_old_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [with_old_keys(item) for item in value]
    return value


def recorded_hash(analysis: dict[str, Any]) -> str:
    """The hash of the analysis file as the sweep wrote it: indented JSON and a trailing newline."""
    return hashlib.sha256((json.dumps(with_old_keys(analysis), indent=2) + '\n').encode()).hexdigest()


def run_rank(results: Path, shots_master: Path, output: Path) -> None:
    evidence = read_json(results / 'input-preprocessing-evidence.json.gz')
    specs = evidence['candidate_specs']
    root = results / 'streams/input-validation'
    configurations = sorted(path.name for path in root.iterdir())
    if configurations != sorted(specs):
        raise ValueError(f'{root}: prediction folders differ from the 20 configurations in candidate_specs')
    folders = {}
    for configuration in configurations:
        for seed_folder in sorted((root / configuration).iterdir()):
            folders[f'{configuration}/{seed_folder.name.removeprefix("seed-")}'] = seed_folder
    scored = rescore_folders(folders, validation_videos(shots_master))
    scores: dict[str, dict[str, Any]] = {configuration: {} for configuration in configurations}
    for key, per_video in scored.items():
        configuration, offset = key.split('/')
        scores[configuration][offset] = per_video

    # The screen ran every configuration at seed 0, in name order.
    screen = analyse(scores, specs, configurations, [0])
    check(screen, evidence['screen'], 'screen')
    recorded = evidence['shortlist']
    check({
        'shortlist': screen['shortlist'],
        'f1_top_two': screen['rankings']['f1'][:2],
        'fully_correct_rallies_top_two': screen['rankings']['fully_correct_rallies'][:2],
        'analysis_sha256': recorded_hash(screen),
    }, {name: recorded[name] for name in ('shortlist', 'f1_top_two', 'fully_correct_rallies_top_two', 'analysis_sha256')},
        'shortlist')
    # The five-seed repeats listed the baseline first, then the other shortlisted configurations.
    shortlisted = [BASELINE, *(name for name in screen['shortlist'] if name != BASELINE)]
    matched = analyse(scores, specs, shortlisted, list(SEED_OFFSETS))
    check(matched, evidence['matched_validation'], 'matched_validation')
    selection = read_json(results / 'test-run/selection.json.gz')
    check(recorded_hash(matched), selection['analysis_sha256'], 'test selection analysis_sha256')
    write_json_gz(output, {'screen': screen, 'matched_validation': matched})
    print(f'Wrote the rebuilt rankings to {output}', flush=True)


def winners(matched: dict[str, Any]) -> dict[str, list[str]]:
    """Each objective's top configuration, in objective order."""
    chosen: dict[str, list[str]] = {}
    for objective in OBJECTIVES:
        chosen.setdefault(matched['rankings'][objective][0], []).append(objective)
    return chosen


def check_seed_export(
    results: Path, candidate: str, seed: dict[str, Any], expected: dict[str, Any],
) -> dict[str, Any]:
    """Check one chosen export against the tracked records and bundle, and return its fit source."""
    offset = seed['offset']
    folder = results / 'test-run/candidates' / candidate / f'seed-{offset}'
    label = f'{candidate} seed {offset}'
    check(seed['export'], f'SWEEP/candidates/{candidate}/seed-{offset}/export', f'{label} export folder')
    settings = read_json(folder / 'export/settings.json.gz')
    result = read_json(folder / 'export/result.json.gz')
    check(seed['settings'], settings, f'{label} settings equal export/settings.json.gz')
    check([result['complete'], result['replay_streams_match']], [True, True], f'{label} export complete and replayed')
    check(seed_metrics(result['per_video']), expected, f'{label} export validation counts equal the ranking')
    check([result['source'], result['preprocessing']], [settings['source'], settings['preprocessing']],
          f'{label} result and settings agree')
    bundle = results / 'models' / f'{candidate}-seed-{offset}'
    actual_md5 = {name: md5(path) for name, path in model_files(bundle).items()}
    check(actual_md5, seed['bundle_md5'], f'{label} tracked bundle md5')
    return settings['source']


def run_choose(results: Path, ranking: Path) -> None:
    matched = read_json(ranking)['matched_validation']
    selection = read_json(results / 'test-run/selection.json.gz')
    check({
        'schema': 'annotator-input-finalists/1', 'fit_policy': 'contact-fit-32-videos', 'feature_units': 'raw',
        'primary_tolerance': 10, 'seed_offsets': list(SEED_OFFSETS),
    }, {name: selection[name] for name in ('schema', 'fit_policy', 'feature_units', 'primary_tolerance', 'seed_offsets')},
        'test selection settings')
    chosen = winners(matched)
    check([[candidate, objectives] for candidate, objectives in chosen.items()],
          [[finalist['candidate'], finalist['objectives']] for finalist in selection['finalists']],
          'configurations chosen on validation and their objectives')
    for finalist in selection['finalists']:
        candidate = finalist['candidate']
        if [seed['offset'] for seed in finalist['seeds']] != list(SEED_OFFSETS):
            raise ValueError(f'{candidate}: the selection must list the five seed offsets in order')
        sources = [
            check_seed_export(results, candidate, seed, matched['rows'][candidate]['per_seed'][str(seed['offset'])])
            for seed in finalist['seeds']
        ]
        if any(source != sources[0] for source in sources) or sources[0]['dirty'] is not False:
            raise ValueError(f'{candidate}: the five exports must come from one clean checkout')
    print('The test selection matches the validation ranking and the tracked bundles', flush=True)


def assemble_rows(sweep: Path, selection: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Combine one split's per-run review rows, after checking each run against the selection."""
    expected_videos = 8 if split == 'validation' else 46
    combined = []
    reference_videos = None
    for finalist in selection['finalists']:
        candidate = finalist['candidate']
        for seed in finalist['seeds']:
            offset = seed['offset']
            if split == 'validation':
                folder = sweep / 'remote/candidates' / candidate / f'seed-{offset}' / 'export'
            else:
                folder = sweep / 'remote/test' / candidate / f'seed-{offset}' / 'evaluation'
            result = read_json(folder / 'result.json.gz')
            settings = read_json(folder / 'settings.json.gz')
            videos = set(result['per_video'])
            if result['complete'] is not True or len(videos) != expected_videos:
                raise ValueError(f'Incomplete {split} run: {candidate}/{offset}')
            if reference_videos is not None and videos != reference_videos:
                raise ValueError(f'Inconsistent {split} videos: {candidate}/{offset}')
            reference_videos = videos
            if split == 'validation':
                if settings != seed['settings'] or result['replay_streams_match'] is not True:
                    raise ValueError(f'Validation export differs from the selection: {candidate}/{offset}')
            else:
                encoded = json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()
                if result['evaluation_sha256'] != hashlib.sha256(encoded).hexdigest():
                    raise ValueError(f'Test evaluation identity mismatch: {candidate}/{offset}')
                selected = settings['selected_seed']
                if (
                    result['candidate'] != candidate or result['seed_offset'] != offset
                    or result['bundle_md5'] != seed['bundle_md5'] or settings['finalist'] != finalist
                    or selected['settings'] != seed['settings'] or selected['bundle_md5'] != seed['bundle_md5']
                ):
                    raise ValueError(f'Test run differs from the selection: {candidate}/{offset}')
            rows = read_json(folder / 'review-rows.json.gz')
            if not rows or any(row['video'] not in videos for row in rows):
                raise ValueError(f'Missing or unknown review rows: {candidate}/{offset}')
            combined.extend({**row, 'candidate': candidate, 'seed_offset': offset, 'split': split} for row in rows)
    return combined


def run_rows(results: Path, archive: Path, output: Path) -> None:
    sweep = archive / 'input-sweep'
    # The archived selection record keeps its original paths and key names, like the runs it names.
    selection = read_json(sweep / 'finalists.json.gz')
    for split in ('validation', 'test'):
        rows = assemble_rows(sweep, selection, split)
        name = f'input-{split}-review-rows.json.gz'
        write_json_gz(output / name, rows)
        check(rows, read_json(results / name), f'{name} rebuilt from {len(rows)} per-run rows')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    rank_parser = commands.add_parser('rank', help='rebuild the screen, shortlist and five-seed ranking')
    rank_parser.add_argument('--output', type=Path, required=True, help='gzip JSON file for the rebuilt rankings')
    rank_parser.add_argument('--shots-master', type=Path, default=SHOTS_MASTER, help='ShuttleSet labels')
    choose_parser = commands.add_parser('choose', help='check the test selection against the ranking')
    choose_parser.add_argument('--ranking', type=Path, required=True, help='output of rank')
    rows_parser = commands.add_parser('rows', help='rebuild both review-row files from the local archive')
    rows_parser.add_argument('--archive', type=Path, required=True, help='unpacked raw-data archive')
    rows_parser.add_argument('--output', type=Path, required=True, help='folder for the rebuilt row files')
    for subparser in (rank_parser, choose_parser, rows_parser):
        subparser.add_argument('--results', type=Path, default=RESULTS, help='tracked results folder')
    args = parser.parse_args()
    if args.command == 'rank':
        run_rank(args.results, args.shots_master, args.output)
    elif args.command == 'choose':
        run_choose(args.results, args.ranking)
    else:
        run_rows(args.results, args.archive, args.output)


if __name__ == '__main__':
    main()
