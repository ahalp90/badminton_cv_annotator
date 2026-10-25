"""Evaluate raw-production bundles chosen on validation on the fixed ShuttleSet22 test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

from experiments.annotator.heuristic_only_tune.run import (
    file_manifest,
    input_paths,
    load_paths,
    read_json,
    source_record,
    write_json,
)

# isort: split

import numpy as np
from threadpoolctl import threadpool_limits

from annotator.models import AnnotatorModels, fitted_trees, load_models, model_files
from annotator.run_video import RunCapture, run_video
from annotator.sequence.contacts import stream_payload
from annotator.sequence.rally_model import RallyLayerModels
from annotator.training.sequences import RallyBoundaryPolicy
from annotator.training.workflow import DEFAULT_TRAINING_SETTINGS
from experiments.annotator.development import pooled_scores
from experiments.annotator.heuristic_only_tune.cache import cache_sticky_result
from experiments.annotator.heuristic_only_tune.data import (
    load_test_inputs,
    load_test_videos,
    run_video_kwargs,
    score_stream,
)
from experiments.annotator.tree_simplification import input_stages, integrate
from experiments.annotator.tree_simplification.prepare import preprocessing_record

SCHEMA = 'annotator-input-finalists/1'
SEED_OFFSETS = (0, 100, 200, 300, 400)
MODEL_HASH_SIZE = 32


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_hash(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_selection(document: dict[str, Any]) -> list[dict[str, Any]]:
    if document.get('schema') != SCHEMA:
        raise ValueError(f'Expected test selection schema {SCHEMA!r}')
    evaluation_source = document.get('evaluation_source')
    if not isinstance(evaluation_source, dict) or evaluation_source.get('dirty') is not False:
        raise ValueError('The test selection must record a clean evaluator checkout')
    if document.get('fit_policy') != 'contact-fit-32-videos' or document.get('feature_units') != 'raw':
        raise ValueError('The test selection must use contact models fitted on all 32 training videos and raw feature units')
    if document.get('seed_offsets') != list(SEED_OFFSETS):
        raise ValueError('The test selection must name the five required seed offsets in order')
    if document.get('primary_tolerance') != 10:
        raise ValueError("The test selection's primary tolerance must be 10 frames")
    finalists = document.get('finalists')
    if not isinstance(finalists, list) or not finalists:
        raise ValueError('The test selection must contain at least one finalist')

    candidates = []
    for finalist in finalists:
        if not isinstance(finalist, dict):
            raise TypeError('Finalist entries must be objects')
        candidate = finalist.get('candidate')
        objectives = finalist.get('objectives')
        seeds = finalist.get('seeds')
        if not isinstance(candidate, str) or not candidate:
            raise ValueError('Each finalist must have a candidate string')
        if not isinstance(objectives, list) or not isinstance(seeds, list) or len(seeds) != len(SEED_OFFSETS):
            raise ValueError(f'{candidate}: finalist must save objectives and exactly five seed exports')
        if any(not isinstance(seed, dict) for seed in seeds):
            raise TypeError(f'{candidate}: seed export entries must be objects')
        offsets = [seed.get('offset') for seed in seeds]
        if offsets != list(SEED_OFFSETS):
            raise ValueError(f'{candidate}: seed exports must match the five required offsets in order')
        for seed in seeds:
            export = seed.get('export')
            settings = seed.get('settings')
            hashes = seed.get('bundle_md5')
            if not isinstance(export, str) or not Path(export).is_absolute():
                raise ValueError(f'{candidate}: every seed export must be an absolute path')
            expected_keys = set(model_files(Path(export) / 'bundle'))
            if not isinstance(settings, dict) or not isinstance(hashes, dict) or set(hashes) != expected_keys:
                raise ValueError(f'{candidate}: each seed must record export settings and all model-file hashes')
            if any(
                not isinstance(value, str) or len(value) != MODEL_HASH_SIZE
                or any(character not in '0123456789abcdef' for character in value)
                for value in hashes.values()
            ):
                raise ValueError(f'{candidate}: bundle hashes must be lowercase MD5 digests')
        candidates.append(candidate)
    if len(set(candidates)) != len(candidates):
        raise ValueError('Test selection candidate strings must be distinct')
    return finalists


def _load_test_bundle(
    selection_path: Path, candidate: str, seed_offset: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], AnnotatorModels]:
    document = read_json(selection_path)
    if not isinstance(document, dict):
        raise TypeError('The test selection must be a JSON object')
    finalists = _validate_selection(document)
    matches = [finalist for finalist in finalists if finalist['candidate'] == candidate]
    if len(matches) != 1:
        raise ValueError(f'Candidate {candidate!r} is not in the test selection')
    finalist = matches[0]
    seed = next(item for item in finalist['seeds'] if item['offset'] == seed_offset)
    export = Path(seed['export'])
    export_settings = read_json(export / 'settings.json.gz')
    if seed['settings'] != export_settings:
        raise ValueError(f'{candidate}, seed {seed_offset}: export settings differ from the test selection')
    if export_settings.get('schema') != 'tree-simplification-raw-export/1':
        raise ValueError(f'{candidate}, seed {seed_offset}: export is not a raw production export')
    result = read_json(export / 'result.json.gz')
    if result.get('complete') is not True or result.get('replay_streams_match') is not True:
        raise ValueError(f'{candidate}, seed {seed_offset}: export replay is incomplete or mismatched')
    if len(result.get('per_video', {})) != 8:
        raise ValueError(f'{candidate}, seed {seed_offset}: export does not contain all eight validation videos')

    run_settings = export_settings['settings']
    pool_settings = run_settings['pool_settings']
    contact_stage_root = Path(pool_settings['contact_run'])
    contact_stage = read_json(contact_stage_root / 'contact-stage.json.gz')
    required_fits = {'held_A', 'held_B', 'held_C', 'held_D', 'v_32'}
    if contact_stage.get('schema') != 'tree-simplification-contacts/1':
        raise ValueError(f'{candidate}, seed {seed_offset}: contact stage has an unsupported schema')
    if set(contact_stage.get('fits', {})) != required_fits:
        raise ValueError(f'{candidate}, seed {seed_offset}: contact stage must contain only held-group and v_32 fits')
    if contact_stage.get('settings') != {
        'prepared': pool_settings['prepared'],
        'settings': pool_settings['settings'],
    }:
        raise ValueError(f'{candidate}, seed {seed_offset}: contact stage settings differ from the fitted run')
    if contact_stage.get('seed_offset') != seed_offset:
        raise ValueError(f'{candidate}, seed {seed_offset}: contact stage used a different seed offset')
    fit_settings = pool_settings['settings']
    expected_seeds = {
        'contact': DEFAULT_TRAINING_SETTINGS.contact.random_seed,
        'confidence': DEFAULT_TRAINING_SETTINGS.confidence.random_seed,
    }
    for fit_name, base_seed in expected_seeds.items():
        section, *subsection = fit_name.split('.')
        config = fit_settings[section]
        if subsection:
            config = config[subsection[0]]
        if config['random_seed'] != base_seed + seed_offset:
            raise ValueError(f'{candidate}, seed {seed_offset}: {fit_name} seed does not match the selected offset')
    if run_settings['tree']['random_seed'] != (
        DEFAULT_TRAINING_SETTINGS.rally_layers.tree.random_seed + seed_offset
    ):
        raise ValueError(f'{candidate}, seed {seed_offset}: selector seed does not match the selected offset')

    bundle = export / 'bundle'
    files = model_files(bundle)
    if set(files) != set(seed['bundle_md5']):
        raise ValueError(f'{candidate}, seed {seed_offset}: bundle file set differs from its recorded checksums')
    actual_hashes = {name: _md5(path) for name, path in files.items()}
    if actual_hashes != seed['bundle_md5']:
        raise ValueError(f'{candidate}, seed {seed_offset}: bundle checksum differs from the test selection')

    models = load_models(bundle)
    if not isinstance(models.sequences, RallyLayerModels) or models.confidence is None:
        raise TypeError('Test bundle must contain fitted rally selectors and review ranking')
    if models.side_geometry.value != 'video':
        raise ValueError('Test bundle must use video-level side geometry')
    if preprocessing_record(models.preprocessing) != export_settings.get('preprocessing'):
        raise ValueError(f'{candidate}, seed {seed_offset}: bundle preprocessing differs from its export record')
    for tree in fitted_trees(models):
        if type(tree).__module__.startswith('xgboost'):
            tree.set_params(device='cpu')
    selected = {
        'schema': document['schema'],
        'fit_policy': document['fit_policy'],
        'feature_units': document['feature_units'],
        'primary_tolerance': document['primary_tolerance'],
        'candidate': candidate,
        'objectives': finalist['objectives'],
        'seed_offset': seed_offset,
        'export': str(export),
        'settings': export_settings,
        'bundle_md5': actual_hashes,
    }
    return document, finalist, selected, models


def _video_outputs(output: Path, identity: str) -> tuple[Path, Path]:
    return output / 'streams' / f'{identity}.json.gz', output / 'videos' / f'{identity}.json.gz'


def _cached_video(
    identity: str, evaluation_hash: str, stream_path: Path, metadata_path: Path,
) -> dict[str, Any] | None:
    if stream_path.exists() != metadata_path.exists():
        raise ValueError(f'{identity}: cached stream and review record must exist together')
    if not stream_path.exists():
        return None
    metadata = read_json(metadata_path)
    if metadata.get('complete') is not True or metadata.get('video') != identity:
        raise ValueError(f'{identity}: cached test result is incomplete or names another video')
    if metadata.get('evaluation_sha256') != evaluation_hash or metadata.get('stream_md5') != _md5(stream_path):
        raise ValueError(f'{identity}: cached test result differs from the recorded evaluation identity')
    if 'review_rows' not in metadata:
        raise ValueError(f'{identity}: cached review rows are missing')
    if not isinstance(metadata['review_rows'], list):
        raise TypeError(f'{identity}: cached review rows must be a list')
    return metadata


def evaluate(
    selection: Path,
    candidate: str,
    seed_offset: int,
    paths_file: Path,
    output: Path,
    *,
    threads: int = 2,
) -> dict[str, Any]:
    """Verify one chosen bundle, then evaluate it on all 46 test fixtures."""
    if not candidate:
        raise ValueError('candidate must be a non-empty candidate string from the test selection')
    if seed_offset not in SEED_OFFSETS:
        raise ValueError(f'seed offset must be one of {SEED_OFFSETS}')
    if not 1 <= threads <= 28:
        raise ValueError('threads must be between 1 and 28')

    selection = selection.resolve()
    evaluation_source = source_record()
    if evaluation_source.get('dirty') is not False:
        raise ValueError('Evaluator checkout must be clean before test evaluation')
    document, finalist, selected, models = _load_test_bundle(selection, candidate, seed_offset)
    boundary_policy = RallyBoundaryPolicy(**selected['settings']['boundary_policy'])
    if document.get('evaluation_source') != evaluation_source:
        raise ValueError('The test selection names another evaluator source; record the selection again from this checkout')
    paths = load_paths(paths_file)
    videos = load_test_videos(paths['test_inputs'])
    identities = [video.identity for video in videos]
    if len(videos) != 46 or len(set(identities)) != 46:
        raise ValueError('ShuttleSet22 test evaluation requires 46 distinct fixtures')
    if any(video.dataset != 'ShuttleSet22' or video.split != 'test' for video in videos):
        raise ValueError('Test manifest contains a fixture outside ShuttleSet22/test')

    fixed_inputs = file_manifest(input_paths(paths, videos, full=False))
    test_manifest_path = paths['test_inputs'] / 'manifest.json.gz'
    source = evaluation_source
    code_hashes = {
        'input_test_md5': _md5(Path(__file__)),
        'input_stages_md5': _md5(Path(input_stages.__file__)),
    }
    evaluation_identity = {
        'schema': 'annotator-input-test/1',
        'source': source,
        'code_md5': code_hashes,
        'selection': str(selection),
        'finalist': finalist,
        'selected_seed': selected,
        'paths': {name: str(path) for name, path in paths.items()},
        'input_files': fixed_inputs,
        'test_manifest_md5': _md5(test_manifest_path),
        'fixtures': [
            {
                'identity': video.identity,
                'dataset': video.dataset,
                'split': video.split,
                'fps': video.fps,
                'resolution': list(video.resolution),
                'frame_count': video.frame_count,
            }
            for video in videos
        ],
    }
    evaluation_hash = _evaluation_hash(evaluation_identity)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'settings.json.gz'
    if output.exists() and any(output.iterdir()) and not receipt_path.exists():
        raise ValueError(f'{output}: non-empty output has no evaluation identity')
    if receipt_path.exists() and read_json(receipt_path) != evaluation_identity:
        raise ValueError(f'{output}: bundle or test inputs changed; use a fresh output directory')
    write_json(receipt_path, evaluation_identity)
    result_path = output / 'result.json.gz'
    write_json(result_path, {'complete': False, 'evaluation_sha256': evaluation_hash})

    per_video: dict[str, Any] = {}
    review_per_video: dict[str, Any] = {}
    review_rows: list[dict[str, Any]] = []
    (output / 'streams').mkdir(exist_ok=True)
    (output / 'videos').mkdir(exist_ok=True)
    with threadpool_limits(limits=threads):
        for index, video in enumerate(videos, start=1):
            stream_path, metadata_path = _video_outputs(output, video.identity)
            cached = _cached_video(video.identity, evaluation_hash, stream_path, metadata_path)
            if cached is not None:
                per_video[video.identity] = cached['score']
                review_per_video[video.identity] = cached['review']
                review_rows.extend(cached['review_rows'])
                continue

            inputs = load_test_inputs(video, paths['inpaint_root'], paths['pose_root'], paths['test_courts'])
            capture = RunCapture()
            with cache_sticky_result(output / 'sticky', video.identity):
                run_video(
                    inputs.track,
                    inputs.pose.bboxes,
                    inputs.pose.scores,
                    inputs.pose.kps,
                    inputs.pose.ndet,
                    base=models.preprocessing,
                    models=models,
                    heuristic_only=False,
                    capture=capture,
                    **run_video_kwargs(inputs),
                )
            prediction = capture.hybrid
            if prediction is None:
                raise ValueError(f'{video.identity}: fitted annotation did not return its hybrid prediction')
            stream = stream_payload(prediction.refined.sequences, prediction.refined.events)
            score = score_stream(video, stream, boundary_policy=boundary_policy)
            review, rows = input_stages._review_rows(
                video, video.identity, prediction, boundary_policy=boundary_policy,
            )
            write_json(stream_path, stream)
            metadata = {
                'complete': True,
                'video': video.identity,
                'evaluation_sha256': evaluation_hash,
                'stream_md5': _md5(stream_path),
                'score': score,
                'review': review,
                'review_rows': rows,
            }
            write_json(metadata_path, metadata)
            per_video[video.identity] = score
            review_per_video[video.identity] = review
            review_rows.extend(rows)
            write_json(output / 'progress.json.gz', {
                'stage': 'test-video-complete', 'video': video.identity,
                'complete': index, 'planned': len(videos),
            })

    if len(per_video) != 46 or set(per_video) != set(identities):
        raise ValueError('Test evaluation requires one complete result for every fixed fixture')
    review_totals = {}
    for ranking in ('selector', 'fitted'):
        for tolerance in (5, 10):
            selected_rows = [
                row for row in review_rows
                if row['ranking'] == ranking and row['tolerance'] == tolerance
            ]
            review_totals[f'{ranking}-{tolerance}'] = integrate.ranking_metrics(
                np.asarray([row['correct'] for row in selected_rows], dtype=np.int8),
                np.asarray([row['score'] for row in selected_rows], dtype=np.float64),
            )
    write_json(output / 'review-rows.json.gz', review_rows)
    result = {
        'complete': True,
        'evaluation_sha256': evaluation_hash,
        'source': source,
        'code_md5': code_hashes,
        'fit_policy': document['fit_policy'],
        'feature_units': document['feature_units'],
        'candidate': candidate,
        'objectives': finalist['objectives'],
        'seed_offset': seed_offset,
        'export': selected['export'],
        'bundle_md5': selected['bundle_md5'],
        'config': selected['settings'],
        'primary_tolerance': document['primary_tolerance'],
        'per_video': per_video,
        'totals': pooled_scores(per_video),
        'review_per_video': review_per_video,
        'review': review_totals,
    }
    write_json(result_path, result)
    write_json(output / 'progress.json.gz', {'stage': 'complete', 'videos': len(per_video)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--selection', type=Path, required=True,
        help='Gzip JSON record of the configurations chosen on validation',
    )
    parser.add_argument('--candidate', required=True, help='Candidate string from the test selection')
    parser.add_argument('--seed-offset', type=int, choices=SEED_OFFSETS, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    evaluate(
        args.selection, args.candidate, args.seed_offset, args.paths, args.output, threads=args.threads,
    )


if __name__ == '__main__':
    main()
