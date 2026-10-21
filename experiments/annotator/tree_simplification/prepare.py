"""Prepare fixed vision evidence for rally-model experiments."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.heuristic_only_tune.run import (
    load_candidates,
    load_paths,
    read_json,
    source_record,
    write_json,
)

# isort: split

import joblib
from threadpoolctl import threadpool_limits

from annotator.config import BaseAnnotatorConfig, RallySegmentationThresholds
from annotator.hybrid import features_from_evidence
from annotator.run_video import RunCapture, run_video
from annotator.types import DeadMaskMode, ReentryGuardVariant, SmoothingMode, SpanOpen
from experiments.annotator.development import development_for, preprocessing
from experiments.annotator.heuristic_only_tune.cache import cache_sticky_result
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_original_inputs,
    load_original_videos,
    run_video_kwargs,
)
from experiments.annotator.tree_simplification.data import PreparedVideo


def preprocessing_record(config: BaseAnnotatorConfig) -> dict[str, Any]:
    """Return the JSON-safe preprocessing value used by manifests and settings."""
    return json.loads(json.dumps(asdict(config), default=sorted))


def preprocessing_from_record(record: dict[str, Any]) -> BaseAnnotatorConfig:
    """Restore a saved preprocessing config, failing if its schema has drifted."""
    expected_fields = set(asdict(BaseAnnotatorConfig()))
    if set(record) != expected_fields:
        raise ValueError('Saved preprocessing fields differ from BaseAnnotatorConfig')
    thresholds = record['thresholds']
    if not isinstance(thresholds, (list, tuple)):
        raise TypeError('Saved preprocessing thresholds must be a sequence')
    overrides = record['overrides_base30']
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError('Saved preprocessing overrides must be an object or null')
    rejected_grades = record['rejected_grades']
    if not isinstance(rejected_grades, list):
        raise TypeError('Saved rejected grades must be a list')
    config = BaseAnnotatorConfig(
        thresholds=RallySegmentationThresholds(*thresholds),
        dead_mask_mode=DeadMaskMode(record['dead_mask_mode']),
        smoothing_mode=SmoothingMode(record['smoothing_mode']),
        overrides_base30=overrides,
        span_open=SpanOpen(record['span_open']) if record['span_open'] is not None else None,
        gap_state_demotion_bound=record['gap_state_demotion_bound'],
        reentry_guard_variant=(
            ReentryGuardVariant(record['reentry_guard_variant'])
            if record['reentry_guard_variant'] is not None else None
        ),
        reentry_guard_buffer=record['reentry_guard_buffer'],
        quiet_start_window=record['quiet_start_window'],
        rejected_grades=frozenset(rejected_grades),
    )
    if preprocessing_record(config) != record:
        raise ValueError('Saved preprocessing does not round-trip through BaseAnnotatorConfig')
    return config


def candidate_preprocessing(path: Path) -> tuple[BaseAnnotatorConfig, dict[str, object]]:
    """Load one canonical CandidateSpec from the experiment's gzip JSON format."""
    document, specs = load_candidates(path)
    if not isinstance(document, dict):
        raise TypeError('Candidate document must be an object')
    if 'schema' not in document or 'stage' not in document:
        raise ValueError('Candidate document must name its schema and stage')
    if not isinstance(document['schema'], str) or not isinstance(document['stage'], str):
        raise TypeError('Candidate schema and stage must be strings')
    candidates = document.get('candidates')
    if not isinstance(candidates, list) or len(candidates) != 1 or len(specs) != 1:
        raise ValueError('Candidate file must contain exactly one candidate in a candidates list')
    spec = specs[0]
    config = preprocessing(spec)
    return config, document


def prepare_video(
    video: SearchVideo,
    group: str,
    paths: dict[str, Path],
    output: Path,
    base: BaseAnnotatorConfig | None = None,
) -> dict[str, object]:
    if base is None:
        base = BaseAnnotatorConfig()
    started = time.perf_counter()
    destination = output / 'videos' / f'{video.identity}.joblib'
    record_path = output / 'videos' / f'{video.identity}.json.gz'
    base_record = preprocessing_record(base)
    if destination.exists() and record_path.exists():
        record = read_json(record_path)
        if record.get('preprocessing', preprocessing_record(BaseAnnotatorConfig())) != base_record:
            raise ValueError(f'{video.identity}: cached preprocessing differs; choose a fresh prepared directory')
        return record
    with threadpool_limits(limits=1):
        inputs = load_original_inputs(video, paths['stages'], paths['development_courts'])
        capture = RunCapture()
        pose = inputs.pose
        with cache_sticky_result(output / 'sticky', video.identity):
            run_video(
                inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
                base=base, heuristic_only=True, capture=capture, **run_video_kwargs(inputs),
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
            'feature_rows': len(features.rows), 'preprocessing': base_record,
            'wall_s': time.perf_counter() - started,
        }
        write_json(record_path, record)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, help='gzip JSON with one canonical CandidateSpec')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--only', help='Prepare one video as a pilot')
    args = parser.parse_args()
    if not 1 <= args.workers <= 28:
        parser.error('--workers must be between 1 and 28')
    paths = load_paths(args.paths)
    videos = load_original_videos(paths['shots_master'])
    development = development_for(videos)
    base, candidate = candidate_preprocessing(args.candidate) if args.candidate else (BaseAnnotatorConfig(), None)
    identity = {
        'paths': {name: str(path) for name, path in paths.items()},
        'preprocessing': preprocessing_record(base),
        'groups': development.group_by_video,
    }
    identity = json.loads(json.dumps(identity, default=sorted))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'preparation.json.gz'
    manifest_record = {'inputs': identity, 'source': source_record()}
    if candidate is not None:
        manifest_record['candidate'] = candidate
    if manifest.exists():
        saved = read_json(manifest)
        if (
            saved['inputs'] != identity
            or saved.get('candidate') != candidate
            or saved['source'] != manifest_record['source']
        ):
            raise ValueError('Prepared inputs, preprocessing or source changed; choose a fresh output directory')
    if not manifest.exists():
        write_json(manifest, manifest_record)
    selected = [video for video in videos if args.only is None or video.identity == args.only]
    if not selected:
        parser.error('--only does not name a development video')
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1,
    ) as executor:
        futures = [executor.submit(
            prepare_video, video, development.group_by_video[video.identity], paths, args.output, base,
        ) for video in selected]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == '__main__':
    main()
