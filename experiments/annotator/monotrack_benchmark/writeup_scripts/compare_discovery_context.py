"""Compare discovery configurations on judgeable validation rallies."""

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

from annotator.sequence.contacts import scale_frames, stream_from_payload
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
from experiments.annotator.monotrack_benchmark.writeup_scripts.metrics import (
    DISCOVERY_BOUNDARY_BASIS,
    SEEDS,
    contact_endpoint_stream,
    judge_found_rallies,
)

PROTOCOL = {
    'selection_split': 'validation', 'tolerance_base30': 10,
    'boundary_basis': DISCOVERY_BOUNDARY_BASIS,
    'boundary_policy': {'before_seconds': None, 'after_seconds': None},
    'primary': 'five-seed mean fully correct rally percentage across judgeable validation rallies',
    'tie_breaks': ['correct-player F1', 'contact F1'],
    'exclusions': 'original flaws, missing master rows and repeated players; merge check before exclusion',
    'boundary_survey_stage': 'discovery output before dataset export',
}


@dataclass(frozen=True)
class ValidationJob:
    name: str
    seed: int
    video: SearchVideo
    stream: Path
    output: Path
    reuse_contact_counts: Path | None = None


def score_validation(job: ValidationJob) -> dict:
    """Retain all contact labels and exclude unreliable whole-rally judgements."""
    output = job.output / 'videos' / job.name / f'seed-{job.seed}' / f'{job.video.identity}.json.gz'
    if output.exists():
        saved = read_json(output)
        if saved['stream'] != str(job.stream.resolve()) or saved['protocol'] != PROTOCOL:
            raise ValueError(f'{output}: validation inputs changed; use a fresh output directory')
        return saved
    video = job.video
    stream = contact_endpoint_stream(read_json(job.stream))
    if job.reuse_contact_counts is not None:
        previous = read_json(job.reuse_contact_counts / 'videos' / job.name / f'seed-{job.seed}' / f'{video.identity}.json.gz')
        if previous['stream'] != str(job.stream.resolve()) or previous['protocol']['tolerance_base30'] != 10:
            raise ValueError('Contact-count reuse requires the same saved stream and timing tolerance')
        contact_fields = ('labelled_contacts', 'predicted_contacts', 'matched_contacts', 'correct_player_contacts')
        counts = {field: previous['counts'][field] for field in contact_fields}
    else:
        _proposals, contacts, predictions, _rallies = score_video(
            video.identity, video.split, stream, video.rallies, video.fps, 10,
        )
        counts = {
            'labelled_contacts': len(contacts), 'predicted_contacts': len(predictions),
            'matched_contacts': sum(bool(contact['matched']) for contact in contacts),
            'correct_player_contacts': sum(bool(contact['player_correct']) for contact in contacts),
        }
    sequences, _events = stream_from_payload(stream)
    judged = judge_found_rallies(sequences, video.rallies, scale_frames(10, video.fps), fps=video.fps)
    correct = {row['rally_id'] for row in judged if row['outcome'] == 'fully_correct'}
    counts |= {'rallies': sum(rally.judgeable for rally in video.rallies), 'fully_correct': len(correct),
               'excluded_rallies': sum(not rally.judgeable for rally in video.rallies)}
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
        })
    measures = ('contact_f1', 'correct_player_f1', 'fully_correct', 'fully_correct_percent')
    return {'per_seed': per_seed,
            'mean': {measure: float(np.mean([row[measure] for row in per_seed])) for measure in measures}}


def selection_key(summary: dict) -> tuple[int, float, float]:
    """Use exact rally totals so rounding cannot override the F1 tie-breaks."""
    return (sum(row['fully_correct'] for row in summary['per_seed']),
            summary['mean']['correct_player_f1'], summary['mean']['contact_f1'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', nargs=2, action='append', required=True, metavar=('NAME', 'VALIDATION_DIRECTORY'))
    parser.add_argument('--shots-master', type=Path,
                        default=REPO / 'training/data/shuttleset/annotations/shots_master.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--score-only', action='store_true', help='Cache one arm before the second is available')
    parser.add_argument('--reuse-contact-counts', type=Path,
                        help='Previous comparison scores for these immutable saved prediction files')
    args = parser.parse_args()
    runs = {name: Path(directory) for name, directory in args.run}
    if len(runs) != len(args.run):
        parser.error('run names must be unique')
    if not args.score_only and len(runs) != 2:
        parser.error('selection requires both comparison arms')
    videos = [video for video in load_original_videos(args.shots_master) if video.split == 'validation']
    population = sum(rally.judgeable for video in videos for rally in video.rallies)
    flagged_count = sum(not rally.judgeable for video in videos for rally in video.rallies)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'selection_protocol.json.gz', PROTOCOL)
    jobs = []
    for name, directory in runs.items():
        for seed in SEEDS:
            for video in videos:
                jobs.append(ValidationJob(name, seed, video, directory / f'seed-{seed}' / f'{video.identity}.json.gz',
                                          args.output, args.reuse_contact_counts))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(score_validation, jobs))
    comparison = {name: summarise([record for record in records if record['name'] == name]) for name in runs}
    result = {'protocol': PROTOCOL, 'rallies': population, 'excluded_rallies': flagged_count, 'comparison': comparison}
    if not args.score_only:
        winner = max(comparison, key=lambda name: selection_key(comparison[name]))
        result['best_validation_coverage'] = winner
    write_json(args.output / ('validation_scores.json.gz' if args.score_only else 'policy_selection.json.gz'), result)
    for name, summary in comparison.items():
        print(name, summary['mean'], flush=True)
    if not args.score_only:
        print('Best validation coverage:', result['best_validation_coverage'], flush=True)


if __name__ == '__main__':
    main()
