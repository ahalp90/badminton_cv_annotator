"""Prepare the fixed vision evidence once for the tree-stack comparison."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.heuristic_only_tune.run import (
    load_paths,
    read_json,
    source_record,
    write_json,
)

# isort: split

import joblib
from threadpoolctl import threadpool_limits

from annotator.config import BaseAnnotatorConfig
from annotator.hybrid import features_from_evidence
from annotator.run_video import RunCapture, run_video
from experiments.annotator.heuristic_only_tune.cache import cache_sticky_result
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_original_inputs,
    load_original_videos,
    run_video_kwargs,
)
from experiments.annotator.tree_hp_tune.run import development_for
from experiments.annotator.tree_simplification.data import PreparedVideo


def prepare_video(video: SearchVideo, group: str, paths: dict[str, Path], output: Path) -> dict[str, object]:
    started = time.perf_counter()
    destination = output / 'videos' / f'{video.identity}.joblib'
    record_path = output / 'videos' / f'{video.identity}.json.gz'
    if destination.exists() and record_path.exists():
        return read_json(record_path)
    with threadpool_limits(limits=1):
        inputs = load_original_inputs(video, paths['stages'], paths['development_courts'])
        capture = RunCapture()
        pose = inputs.pose
        with cache_sticky_result(output / 'sticky', video.identity):
            run_video(
                inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
                base=BaseAnnotatorConfig(), heuristic_only=True, capture=capture, **run_video_kwargs(inputs),
            )
        evidence = capture.contact_evidence
        if evidence is None:
            raise ValueError(f'{video.identity}: annotation did not capture contact evidence')
        features = features_from_evidence(evidence)
        prepared = PreparedVideo(video, group, evidence, features)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.tmp')
        joblib.dump(prepared, temporary, compress=('gzip', 1))
        temporary.replace(destination)
        record = {
            'video': video.identity, 'split': video.split, 'group': group,
            'feature_rows': len(features.rows), 'wall_s': time.perf_counter() - started,
        }
        write_json(record_path, record)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--only', help='Prepare one video as a pilot')
    args = parser.parse_args()
    if not 1 <= args.workers <= 28:
        parser.error('--workers must be between 1 and 28')
    paths = load_paths(args.paths)
    videos = load_original_videos(paths['shots_master'])
    development = development_for(videos)
    identity = {
        'paths': {name: str(path) for name, path in paths.items()},
        'preprocessing': asdict(BaseAnnotatorConfig()),
        'groups': development.group_by_video,
    }
    identity = json.loads(json.dumps(identity))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'preparation.json.gz'
    if manifest.exists() and read_json(manifest)['inputs'] != identity:
        raise ValueError('Prepared inputs or preprocessing changed; choose a fresh output directory')
    if not manifest.exists():
        write_json(manifest, {'inputs': identity, 'source': source_record()})
    selected = [video for video in videos if args.only is None or video.identity == args.only]
    if not selected:
        parser.error('--only does not name a development video')
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1,
    ) as executor:
        futures = [executor.submit(
            prepare_video, video, development.group_by_video[video.identity], paths, args.output,
        ) for video in selected]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == '__main__':
    main()
