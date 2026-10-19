"""Rerun the 32-video tree retune bundles on the 8 validation videos and keep rally confidence.

The committed discovery streams for these bundles hold contacts and spans only.
This rerun repeats the same inference, adds each found rally's confidence score
and checks that contacts and spans match the committed streams exactly.
"""

from __future__ import annotations

import os

# Spawned workers inherit these before numerical libraries load.
os.environ.update(
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
)

import argparse
import multiprocessing
import resource
import sys
import time
import tomllib
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True,
                        help='TOML with shots_master, stages and development_courts')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--only', nargs='*', default=None, help='seed-offset/video pairs, e.g. 0/10')
    parser.add_argument('--label', default='grid-017', choices=('grid-017', 'shipped'),
                        help='tree retune configuration; the report uses grid-017, the validation winner')
    return parser.parse_args()


def add_repository(repository: Path) -> None:
    sys.path[:0] = [str(repository), str(repository / 'src')]


def run_one(repository: str, paths: dict[str, str], output: str, label: str, seed_offset: int,
            video_id: str) -> dict:
    add_repository(Path(repository))
    from threadpoolctl import threadpool_limits

    from annotator.models import load_models
    from annotator.run_video import RunCapture, run_video
    from experiments.annotator.heuristic_only_tune.data import (
        load_original_inputs,
        load_original_videos,
        run_video_kwargs,
    )
    from experiments.annotator.heuristic_only_tune.run import read_json, write_json
    from experiments.annotator.old_court_regression.retrain import stream_payload
    from experiments.annotator.tree_hp_tune import test_rerun

    started = time.monotonic()
    bundles = [bundle for bundle in test_rerun.discover_bundles()
               if bundle.stage == 'v32' and bundle.label == label and bundle.seed_offset == seed_offset]
    (bundle,) = bundles
    videos = {video.identity: video for video in load_original_videos(Path(paths['shots_master']))
              if video.split == 'validation'}
    video = videos[video_id]
    with threadpool_limits(limits=1):
        models = load_models(bundle.path)
        inputs = load_original_inputs(video, Path(paths['stages']), Path(paths['development_courts']))
        capture = RunCapture()
        pose = inputs.pose
        run_video(
            inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
            base=models.preprocessing, heuristic_only=False, models=models, serve_start=None,
            capture=capture, **run_video_kwargs(inputs),
        )
    prediction = capture.hybrid
    if prediction is None:
        raise ValueError(f'{video_id}: fitted annotation did not return its contact stream')
    sequences = prediction.refined.sequences
    scores = prediction.confidence.scores
    if len(scores) != len(sequences):
        raise ValueError(f'{video_id}: {len(scores)} confidence scores for {len(sequences)} rallies')
    stream = stream_payload(sequences, prediction.refined.events)
    # The validation streams sit beside each 32-video bundle.
    reference_path = bundle.path.parent / 'validation_streams' / f'{video_id}.json.gz'
    reference = read_json(reference_path)
    # Compare through a JSON round trip so tuples and lists compare equal.
    write_json(Path(output) / f'seed-{seed_offset}' / f'{video_id}.json.gz', stream | {
        'confidence': {str(sequence.span_id): float(score) for sequence, score in zip(sequences, scores, strict=True)},
    })
    saved = read_json(Path(output) / f'seed-{seed_offset}' / f'{video_id}.json.gz')
    matches = saved['sequences'] == reference['sequences'] and saved['events'] == reference['events']
    return {
        'seed_offset': seed_offset, 'video': video_id, 'matches_committed': matches,
        'rallies': len(sequences), 'seconds': time.monotonic() - started,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    add_repository(repository)
    from experiments.annotator.heuristic_only_tune.data import load_original_videos
    from experiments.annotator.heuristic_only_tune.run import write_json
    from experiments.annotator.tree_hp_tune.repeats import SEED_OFFSETS

    with args.paths.open('rb') as handle:
        paths = {name: str(Path(value).resolve()) for name, value in tomllib.load(handle)['paths'].items()}
    video_ids = sorted(video.identity for video in load_original_videos(Path(paths['shots_master']))
                       if video.split == 'validation')
    if len(video_ids) != 8:
        raise ValueError('expected the 8 fixed validation videos')
    jobs = [(offset, video_id) for offset in SEED_OFFSETS for video_id in video_ids]
    if args.only is not None:
        wanted = {tuple(item.split('/')) for item in args.only}
        jobs = [job for job in jobs if (str(job[0]), job[1]) in wanted]
    output = args.output.resolve()
    for offset in SEED_OFFSETS:
        (output / f'seed-{offset}').mkdir(parents=True, exist_ok=True)

    records = []
    failures = []
    pending = iter(jobs)
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1,
    ) as executor:
        active = {}
        while True:
            while len(active) < args.workers:
                job = next(pending, None)
                if job is None:
                    break
                future = executor.submit(run_one, str(repository), paths, str(output), args.label, *job)
                active[future] = job
            if not active:
                break
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                job = active.pop(future)
                try:
                    record = future.result()
                except Exception:  # noqa: BLE001 — keep the batch going and report every failure.
                    failures.append({'job': list(job), 'traceback': traceback.format_exc()})
                    print(f'seed-{job[0]}/{job[1]}: failed', flush=True)
                    continue
                records.append(record)
                print(f"seed-{job[0]}/{job[1]}: {'match' if record['matches_committed'] else 'DIFFERS'} "
                      f"{record['seconds']:.0f}s {record['peak_rss_mib']:.0f} MiB", flush=True)
    write_json(output / 'execution.json.gz', {
        'records': sorted(records, key=lambda row: (row['seed_offset'], row['video'])),
        'failures': failures,
    })
    mismatches = [record for record in records if not record['matches_committed']]
    print(f'{len(records)} complete, {len(failures)} failed, {len(mismatches)} differ from committed streams')
    if failures or mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
