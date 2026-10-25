"""Select discovery training on complete validation rallies, with a separate flaw view."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path.insert(0, str(REPO := Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(REPO / 'src'))

import numpy as np

from annotator.training.sequences import RallyBoundaryPolicy
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_original_videos,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    read_json,
    write_json,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import SEEDS
from experiments.annotator.monotrack_benchmark.writeup_scripts.validation_label_sensitivity import (
    flaw_labels,
)

POLICY = RallyBoundaryPolicy(None, None)
PROTOCOL = {
    'selection_split': 'validation', 'tolerance_base30': 10,
    'boundary_policy': {'before_seconds': None, 'after_seconds': None},
    'primary': 'five-seed mean fully correct rally percentage across all 668 validation rallies',
    'tie_breaks': ['correct-player F1', 'contact F1'],
    'flaw_view': 'exclude marked rallies after judgement; retain all contacts and labels during scoring',
    'boundary_survey_stage': 'discovery output before dataset export',
}


@dataclass(frozen=True)
class ValidationJob:
    name: str
    seed: int
    video: SearchVideo
    stream: Path
    flagged: frozenset[str]
    output: Path


def score_validation(job: ValidationJob) -> dict:
    """Score all labels once, then filter only the rally reporting view."""
    output = job.output / 'videos' / job.name / f'seed-{job.seed}' / f'{job.video.identity}.json.gz'
    if output.exists():
        saved = read_json(output)
        if saved['stream'] != str(job.stream.resolve()) or saved['protocol'] != PROTOCOL:
            raise ValueError(f'{output}: validation inputs changed; use a fresh output directory')
        return saved
    video = job.video
    _proposals, contacts, predictions, rallies = score_video(
        video.identity, video.split, read_json(job.stream), video.rallies, video.fps, 10, boundary_policy=POLICY,
    )
    unflagged = [rally for rally in rallies if rally['rally_id'] not in job.flagged]
    counts = {
        'labelled_contacts': len(contacts), 'predicted_contacts': len(predictions),
        'matched_contacts': sum(bool(contact['matched']) for contact in contacts),
        'correct_player_contacts': sum(bool(contact['player_correct']) for contact in contacts),
        'rallies': len(rallies), 'fully_correct': sum(bool(rally['fully_correct']) for rally in rallies),
        'unflagged_rallies': len(unflagged),
        'unflagged_fully_correct': sum(bool(rally['fully_correct']) for rally in unflagged),
    }
    record = {'name': job.name, 'seed': job.seed, 'video': video.identity, 'protocol': PROTOCOL,
              'stream': str(job.stream.resolve()), 'counts': counts}
    write_json(output, record)
    return record


def summarise(records: list[dict]) -> dict:
    per_seed = []
    for seed in SEEDS:
        selected = [record['counts'] for record in records if record['seed'] == seed]
        totals = {name: sum(counts[name] for counts in selected) for name in selected[0]}
        denominator = totals['labelled_contacts'] + totals['predicted_contacts']
        per_seed.append({
            'seed': seed, **totals,
            'contact_f1': 200 * totals['matched_contacts'] / denominator,
            'correct_player_f1': 200 * totals['correct_player_contacts'] / denominator,
            'fully_correct_percent': 100 * totals['fully_correct'] / totals['rallies'],
            'unflagged_fully_correct_percent': 100 * totals['unflagged_fully_correct'] / totals['unflagged_rallies'],
        })
    measures = ('contact_f1', 'correct_player_f1', 'fully_correct', 'fully_correct_percent',
                'unflagged_fully_correct', 'unflagged_fully_correct_percent')
    return {'per_seed': per_seed,
            'mean': {measure: float(np.mean([row[measure] for row in per_seed])) for measure in measures}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs=2, action='append', required=True, metavar=('NAME', 'VALIDATION_DIRECTORY'))
    parser.add_argument('--shots-master', type=Path,
                        default=REPO / 'training/data/shuttleset/annotations/shots_master.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--score-only', action='store_true', help='Cache one arm before the second is available')
    args = parser.parse_args()
    runs = {name: Path(directory) for name, directory in args.run}
    if len(runs) != len(args.run):
        parser.error('run names must be unique')
    if not args.score_only and len(runs) != 2:
        parser.error('selection requires both comparison arms')
    videos = [video for video in load_original_videos(args.shots_master) if video.split == 'validation']
    flagged, _windows, mapping = flaw_labels(videos)
    population = sum(len(video.rallies) for video in videos)
    flagged_count = mapping['rallies_with_flaw_eq_1']
    if (population, flagged_count, population - flagged_count) != (668, 314, 354):
        raise ValueError(f'Unexpected validation population: {population} total, {flagged_count} marked')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'selection_protocol.json.gz', PROTOCOL)
    jobs = []
    for name, directory in runs.items():
        for seed in SEEDS:
            for video in videos:
                jobs.append(ValidationJob(name, seed, video, directory / f'seed-{seed}' / f'{video.identity}.json.gz',
                                          frozenset(flagged[video.identity]), args.output))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(score_validation, jobs))
    comparison = {name: summarise([record for record in records if record['name'] == name]) for name in runs}
    result = {'protocol': PROTOCOL, 'rallies': population, 'flagged_rallies': flagged_count,
              'unflagged_rallies': population - flagged_count, 'comparison': comparison}
    if not args.score_only:
        winner = max(comparison, key=lambda name: (
            comparison[name]['mean']['fully_correct_percent'],
            comparison[name]['mean']['correct_player_f1'], comparison[name]['mean']['contact_f1'],
        ))
        result['selected'] = winner
    write_json(args.output / ('validation_scores.json.gz' if args.score_only else 'policy_selection.json.gz'), result)
    for name, summary in comparison.items():
        print(name, summary['mean'], flush=True)
    if not args.score_only:
        print('Selected:', result['selected'], flush=True)


if __name__ == '__main__':
    main()
