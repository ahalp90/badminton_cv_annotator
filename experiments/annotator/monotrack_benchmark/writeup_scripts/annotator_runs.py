"""Run the annotator's three-model bundles for the MonoTrack write-up.

Every run covers five seeds. Each video is one single-threaded job.

Test, on the 46 ShuttleSet22 videos:

- ``discovery``: whole videos with per-scene courts. Each found rally keeps its review
  score, which is the confidence score in the report.
- ``our_courts``: the supplied rally bounds with our per-rally courts.
- ``monotrack_courts``: the supplied rally bounds with MonoTrack's courts.

Validation, on the 8 original ShuttleSet validation videos: discovery with review scores,
for choosing the confidence cut-offs.

Model families:

- ``32_videos``: the five test bundles, fitted on the 32 training videos.
- ``40_videos``: the same bundles with the contact model refitted on all 40 original
  videos by ``fit_contact_40.py``. Test only, because validation is in their training set.
- ``body_window_36``: the preprocessing setting with the most fully correct validation rallies.
  Test with our courts only, for the retuning comparison.

Models come from the supplied fresh training root.

Usage: python annotator_runs.py --paths PATHS.toml --tree-results DIR --models-40 DIR --out DIR
"""
from __future__ import annotations

import os

# Spawned workers inherit these before numerical libraries load.
os.environ.update(
    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', BLIS_NUM_THREADS='2',
    NUMEXPR_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2',
)

import argparse
import multiprocessing
import resource
import sys
import time
import tomllib
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from annotator.training.sequences import RallyBoundaryPolicy

REPO = Path(__file__).resolve().parents[4]
BENCHMARK_SCRIPTS = REPO / 'experiments/annotator/monotrack_benchmark/benchmark_scripts'
INPUTS = REPO / 'experiments/annotator/monotrack_benchmark/benchmark_results/inputs'
SEEDS = (0, 100, 200, 300, 400)
SUPPLIED_CONDITIONS = {'our_courts': 'ours_our_courts', 'monotrack_courts': 'ours_mono_courts'}
TEST_PATH_KEYS = ('test_courts', 'inpaint_root', 'pose_root')
VALIDATION_PATH_KEYS = ('stages', 'development_courts')


@dataclass(frozen=True)
class Run:
    split: str
    condition: str
    family: str
    seed: int

    def directory(self, output: Path) -> Path:
        if self.split == 'validation':
            return output / 'validation' / self.family / f'seed-{self.seed}'
        return output / 'test' / self.condition / self.family / f'seed-{self.seed}'


def planned_runs() -> list[Run]:
    runs = []
    for seed in SEEDS:
        runs.append(Run('validation', 'discovery', '32_videos', seed))
        for family in ('32_videos', '40_videos'):
            runs.extend(Run('test', condition, family, seed)
                        for condition in ('discovery', 'our_courts', 'monotrack_courts'))
        runs.append(Run('test', 'our_courts', 'body_window_36', seed))
    return runs


def bundle_path(run: Run, models_40: Path, tree_results: Path) -> Path:
    if run.family == '40_videos':
        return models_40 / f'seed-{run.seed}'
    label = 'body_window_36' if run.family == 'body_window_36' else 'raw_baseline'
    return tree_results / f'models/{label}-seed-{run.seed}'


def add_paths() -> None:
    sys.path[:0] = [str(REPO), str(REPO / 'src'), str(BENCHMARK_SCRIPTS)]


def discovery_video(run: Run, bundle: Path, paths: dict[str, Path], video_id: str, output: Path,
                    boundary_policy: RallyBoundaryPolicy) -> dict:
    """Find rallies in one whole video and keep each found rally's review score."""
    from threadpoolctl import threadpool_limits

    from annotator.models import load_models
    from annotator.run_video import RunCapture, run_video
    from annotator.sequence.contacts import stream_payload
    from experiments.annotator.heuristic_only_tune import data
    from experiments.annotator.heuristic_only_tune.run import read_json, write_json

    if run.split == 'test':
        video = next(video for video in data.load_test_videos(INPUTS) if video.identity == video_id)
        inputs = data.load_test_inputs(video, paths['inpaint_root'], paths['pose_root'], paths['test_courts'])
    else:
        video = next(video for video in data.load_original_videos(REPO / 'training/data/shuttleset/annotations/shots_master.csv')
                     if video.identity == video_id)
        inputs = data.load_original_inputs(video, paths['stages'], paths['development_courts'])
    with threadpool_limits(limits=2):
        models = load_models(bundle)
        capture = RunCapture()
        pose = inputs.pose
        run_video(inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
                  base=models.preprocessing, heuristic_only=False, models=models, serve_start=None,
                  capture=capture, **data.run_video_kwargs(inputs))
    prediction = capture.hybrid
    if prediction is None:
        raise ValueError(f'{video_id}: the annotator returned no contact stream')
    sequences = prediction.refined.sequences
    review = prediction.confidence
    scores = {int(span): float(score) for span, score in zip(review.features.span_ids, review.scores, strict=True)}
    missing = [sequence.span_id for sequence in sequences if sequence.span_id not in scores]
    if missing:
        raise ValueError(f'{video_id}: found rallies {missing} have no review score')
    path = output / f'{video_id}.json.gz'
    write_json(path, stream_payload(sequences, prediction.refined.events) | {
        'confidence': {str(sequence.span_id): scores[sequence.span_id] for sequence in sequences},
    })
    return {'score': data.score_stream(video, read_json(path), boundary_policy=boundary_policy)}


def supplied_video(run: Run, bundle: Path, paths: dict[str, Path], video_id: str, output: Path) -> dict:
    """Annotate one video inside the supplied rally bounds, through the benchmark's own runner."""
    from annotator_runner import run_item
    from common import load_manifest, read_json

    item = next(item for item in load_manifest(INPUTS)['items'] if item['item_id'] == video_id)
    config = {'paths': {'bundle': bundle, 'inputs': INPUTS, 'inpaint_root': paths['inpaint_root'],
                        'pose_root': paths['pose_root']}}
    run_item((config, item, SUPPLIED_CONDITIONS[run.condition], output))
    if read_json(output / f'{video_id}.json.gz')['status'] != 'completed':
        raise ValueError(f'{video_id}: annotation failed')
    return {'score': None}


def run_video_job(run: Run, bundle: str, paths: dict[str, str], video_id: str, output: str,
                  boundary_policy: RallyBoundaryPolicy) -> dict:
    add_paths()
    started = time.monotonic()
    resolved = {name: Path(value) for name, value in paths.items()}
    if run.condition == 'discovery':
        record = discovery_video(run, Path(bundle), resolved, video_id, Path(output), boundary_policy)
    else:
        record = supplied_video(run, Path(bundle), resolved, video_id, Path(output))
    return record | {'video': video_id, 'seconds': time.monotonic() - started,
                     'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def video_ids(split: str) -> list[str]:
    from experiments.annotator.heuristic_only_tune import data

    if split == 'test':
        return sorted((video.identity for video in data.load_test_videos(INPUTS)), key=int)
    ids = sorted(video.identity for video in
                 data.load_original_videos(REPO / 'training/data/shuttleset/annotations/shots_master.csv')
                 if video.split == 'validation')
    if len(ids) != 8:
        raise ValueError('expected the 8 fixed validation videos')
    return ids


def finish_run(run: Run, directory: Path, records: list[dict]) -> None:
    """Score a finished run and save how it went."""
    from score import score_run

    from experiments.annotator.development import pooled_scores
    from experiments.annotator.heuristic_only_tune.run import write_json

    if run.condition == 'discovery':
        per_video = {record['video']: record['score'] for record in records}
        write_json(directory / 'scores.json.gz', {'by_video': per_video, 'by_tolerance': pooled_scores(per_video)})
    else:
        summary = score_run(INPUTS, directory)
        if summary['failed_videos'] or summary['failed_rallies']:
            raise ValueError(f'{directory}: annotation failures')
    write_json(directory / 'execution.json.gz', {
        'records': sorted(({key: value for key, value in record.items() if key != 'score'} for record in records),
                          key=lambda record: record['video']),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True,
                        help='TOML [paths] with test_courts, inpaint_root, pose_root, stages and development_courts')
    parser.add_argument('--models-40', type=Path, required=True, help='output folder of fit_contact_40.py')
    parser.add_argument('--tree-results', type=Path, required=True, help='fresh training root containing models/')
    parser.add_argument('--boundary-before-seconds', type=float, default=2.0)
    parser.add_argument('--boundary-after-seconds', type=float, default=3.0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--only', nargs='*', default=None,
                        help='limit to runs whose folder (relative to --out) starts with one of these, e.g. test/discovery')
    parser.add_argument('--videos', nargs='*', default=None, help='limit to these videos, for a smoke test')
    args = parser.parse_args()
    if not 1 <= args.workers <= 14:
        parser.error('--workers must be between 1 and 14 for two-core models')
    add_paths()
    from annotator.training.sequences import RallyBoundaryPolicy
    from experiments.annotator.heuristic_only_tune.run import write_json

    policy = RallyBoundaryPolicy(args.boundary_before_seconds, args.boundary_after_seconds)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'run-policy.json.gz', {
        'tree_results': str(args.tree_results.resolve()), 'models_40': str(args.models_40.resolve()),
        'boundary_policy': asdict(policy),
    })
    with args.paths.open('rb') as handle:
        paths = {name: str(Path(value).resolve()) for name, value in tomllib.load(handle)['paths'].items()}
    for name in (*TEST_PATH_KEYS, *VALIDATION_PATH_KEYS):
        if not Path(paths[name]).is_dir():
            raise NotADirectoryError(f'{name}: {paths[name]}')
    output = args.out.resolve()
    runs = planned_runs()
    if args.only is not None:
        runs = [run for run in runs
                if any(str(run.directory(output).relative_to(output)).startswith(prefix) for prefix in args.only)]
    ids = {split: video_ids(split) for split in ('test', 'validation')}

    pending, records, expected = [], {run: [] for run in runs}, {}
    for run in runs:
        directory = run.directory(output)
        # A rerun into used folders could mix old and new predictions under old scores.
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'{directory}: run folders must start empty')
        directory.mkdir(parents=True, exist_ok=True)
        bundle = bundle_path(run, args.models_40.resolve(), args.tree_results.resolve())
        if not (bundle / 'models.joblib').is_file():
            raise FileNotFoundError(bundle)
        wanted = [video for video in ids[run.split] if args.videos is None or video in args.videos]
        expected[run] = len(wanted)
        pending.extend((run, str(bundle), video, str(directory)) for video in wanted)

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=1) as executor:
        jobs = iter(pending)
        active = {}
        while True:
            while len(active) < args.workers:
                job = next(jobs, None)
                if job is None:
                    break
                run, bundle, video, directory = job
                active[executor.submit(run_video_job, run, bundle, paths, video, directory, policy)] = job
            if not active:
                break
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                run, _bundle, video, directory = active.pop(future)
                label = f'{Path(directory).relative_to(output)}/{video}'
                try:
                    record = future.result()
                except Exception:  # noqa: BLE001 — finish the other jobs, then report every failure.
                    failures.append({'job': label, 'traceback': traceback.format_exc()})
                    print(f'{label}: failed', flush=True)
                    continue
                records[run].append(record)
                print(f"{label}: {record['seconds']:.0f}s {record['peak_rss_mib']:.0f} MiB", flush=True)
                if len(records[run]) == expected[run] and args.videos is None:
                    finish_run(run, run.directory(output), records[run])

    print(f'{sum(map(len, records.values()))} videos complete, {len(failures)} failed', flush=True)
    if failures:
        write_json(output / 'failures.json.gz', {'failures': failures})
        raise SystemExit(1)


if __name__ == '__main__':
    main()
