"""Evaluate complete annotator bundles on ShuttleSet22."""

from __future__ import annotations

import json
import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from annotator.config import scale_config_to_fps
from annotator.inference.learned import (
    LearnedModelPrediction,
    contact_features_from_inputs,
    predict_from_contact_features,
    predict_from_rally_options,
)
from annotator.inference.prepare_annotation_inputs import prepare_annotation_inputs
from annotator.models import AnnotatorModels, fitted_trees, load_models
from experiments.annotator.contact_records import stream_payload
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_test_inputs,
    load_test_videos,
)
from experiments.annotator.heuristic_only_tune.run import (
    load_paths,
    read_json,
    write_json,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    load_manifest,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts import metrics
from experiments.annotator.tree_simplification.input_test import (
    EXPECTED_CONTACTS,
    EXPECTED_LEFT_OUT,
    EXPECTED_RALLIES,
    EXPECTED_VIDEOS,
    TOLERANCES,
    UNKNOWN_REVIEW_OUTCOMES,
)
from experiments.annotator.tree_simplification.prepare import preprocessing_record
from experiments.annotator.tree_simplification.review_metrics import ranking_metrics
from experiments.annotator.tree_simplification.selector_review import (
    score_selector_review,
)


@lru_cache(maxsize=32)
def _models(path: str, device: str = 'cpu') -> AnnotatorModels:
    models = load_models(Path(path), device=device)
    for tree in fitted_trees(models):
        tree.set_params(n_jobs=1)
    return models


def _score_prediction(
    item: dict[str, Any], windows: list[tuple[int, int]], prediction: LearnedModelPrediction,
) -> dict[str, Any]:
    stream = stream_payload(prediction.sequences, prediction.events)
    reviews = {'selector': score_selector_review(prediction.review.features), 'fitted': prediction.review}
    scores = {}
    span_ids = tuple(sequence.span_id for sequence in prediction.sequences)
    for name, review in reviews.items():
        scores[name] = dict(zip(span_ids, prediction.scores_for_rallies(review), strict=True))
    counts, judgements, review_rows = {}, {}, []
    for tolerance in TOLERANCES:
        key = str(tolerance)
        counts[key], judgements[key] = metrics.score_stream_for_item(item, stream, tolerance, windows, supplied=False)
        for judgement in judgements[key]:
            outcome = judgement['outcome']
            correct = -1 if outcome in UNKNOWN_REVIEW_OUTCOMES else int(outcome == 'fully_correct')
            for name, ranking in scores.items():
                review_rows.append({
                    'video': item['item_id'], 'span': judgement['span_id'], 'tolerance': tolerance,
                    'ranking': name, 'outcome': outcome, 'correct': correct, 'score': ranking[judgement['span_id']],
                })
    stream['confidence'] = {str(span): score for span, score in scores['fitted'].items()}
    return {'stream': stream, 'counts': counts, 'judgements': judgements, 'review_rows': review_rows}


def _test_video(
    video: SearchVideo, item: dict[str, Any], windows: list[tuple[int, int]],
    paths: dict[str, Path], bundles: list[dict[str, Any]], output: Path, deadline: float,
) -> str:
    pending = []
    for record in bundles:
        destination = output / record['label'] / f"seed-{record['seed']}" / 'videos' / f'{video.identity}.json.gz'
        if not destination.exists():
            pending.append((record, destination))
    if not pending:
        return video.identity
    inputs = load_test_inputs(video, paths['inpaint_root'], paths['pose_root'], paths['test_courts'])
    prepared = {}
    predictions = {}
    with threadpool_limits(limits=1):
        for record, destination in pending:
            if time.time() >= deadline:
                raise TimeoutError('Test evaluation reached the search deadline')
            models = _models(str(record['path']))
            preparation_key = json.dumps(preprocessing_record(models.preprocessing), sort_keys=True)
            if preparation_key not in prepared:
                pose, court = inputs.pose, inputs.court
                evidence = prepare_annotation_inputs(
                    inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
                    video_config=scale_config_to_fps(models.preprocessing, inputs.fps), resolution=court.resolution,
                    court_present=inputs.court_present, homography_rows=court.homography_rows,
                    shuttle_grades=inputs.shuttle_grades,
                )
                prepared[preparation_key] = evidence, contact_features_from_inputs(evidence)
            evidence, features = prepared[preparation_key]
            # Equal contact trees, cutoffs and proposal budgets produce the same pool.
            pool_key = (
                preparation_key, bytes(models.contact_model.get_booster().save_raw()),
                models.contact_settings.score_cutoff, models.rally_selector.max_edits,
                models.rally_selector.deletion_candidates,
            )
            if pool_key not in predictions:
                prediction = predict_from_contact_features(evidence, features, models)
                predictions[pool_key] = prediction
            else:
                first = predictions[pool_key]
                prediction = predict_from_rally_options(
                    evidence, features, first.scored_candidates, first.refinement.pool, models,
                )
            result = _score_prediction(item, windows, prediction)
            stream = result.pop('stream')
            write_json(destination.parent.parent / 'streams' / f'{video.identity}.json.gz', stream)
            write_json(destination, result)
    return video.identity


def _test_summary(record: dict[str, Any], videos: tuple[SearchVideo, ...], output: Path) -> dict[str, Any]:
    directory = output / record['label'] / f"seed-{record['seed']}"
    per_video, review_rows = {}, []
    for video in videos:
        result = read_json(directory / 'videos' / f'{video.identity}.json.gz')
        per_video[video.identity] = result['counts']
        review_rows.extend(result['review_rows'])
    totals, rates, reviews = {}, {}, {}
    for tolerance in TOLERANCES:
        key = str(tolerance)
        first = next(iter(per_video.values()))[key]
        totals[key] = {name: sum(counts[key][name] for counts in per_video.values()) for name in first}
        rates[key] = metrics.rates(totals[key])
        for ranking in ('selector', 'fitted'):
            rows = [row for row in review_rows if row['ranking'] == ranking and row['tolerance'] == tolerance]
            reviews[f'{ranking}-{tolerance}'] = ranking_metrics(
                np.asarray([row['correct'] for row in rows], dtype=np.int8),
                np.asarray([row['score'] for row in rows], dtype=np.float64),
            )
    if (totals['10']['labelled_rallies'], totals['10']['labelled_contacts']) != (EXPECTED_RALLIES, EXPECTED_CONTACTS):
        raise ValueError('The fixed test requires 3327 scored rallies and 37184 labelled contacts')
    result = {
        'label': record['label'], 'seed': record['seed'], 'complete': True,
        'per_video': per_video, 'totals': totals, 'rates': rates, 'review': reviews,
    }
    write_json(directory / 'result.json.gz', result)
    return result


def evaluate_test(
    paths_file: Path, bundles: list[dict[str, Any]], output: Path, workers: int,
    deadline: float, progress: Callable[[int, int, str], None],
) -> dict[str, Any]:
    """Compare frozen bundles, sharing each test video's preparation across fits.

    :param deadline: Absolute Unix time after which no further work is started.
    :param progress: Coordinator callback receiving completed videos, total and ID.
    """
    output.mkdir(parents=True, exist_ok=True)
    paths = load_paths(paths_file)
    videos = load_test_videos(paths['test_inputs'])
    if len(videos) != EXPECTED_VIDEOS or len({video.identity for video in videos}) != EXPECTED_VIDEOS:
        raise ValueError('Test evaluation requires 46 distinct ShuttleSet22 fixtures')
    items = {item['item_id']: item for item in load_manifest(paths['test_inputs'])['items']}
    metrics.INPUTS = paths['test_inputs']
    windows = {identity: metrics.left_out_windows(rallies) for identity, rallies in metrics.test_left_out().items()}
    if sum(map(len, windows.values())) != EXPECTED_LEFT_OUT:
        raise ValueError('The fixed test requires 524 excluded rally windows')
    settings = {'bundles': [{**record, 'path': str(record['path'])} for record in bundles], 'paths': str(paths_file)}
    settings_path = output / 'test-settings.json.gz'
    if settings_path.exists() and read_json(settings_path) != settings:
        raise ValueError('Test bundles or input paths changed; use a fresh output directory')
    write_json(settings_path, settings)
    for record in bundles:
        directory = output / record['label'] / f"seed-{record['seed']}"
        (directory / 'streams').mkdir(parents=True, exist_ok=True)
        (directory / 'videos').mkdir(parents=True, exist_ok=True)
    completed = 0
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
    try:
        remaining = iter(videos)
        active = {}
        while True:
            if time.time() >= deadline:
                raise TimeoutError('Test evaluation reached the search deadline')
            while len(active) < workers:
                video = next(remaining, None)
                if video is None:
                    break
                future = executor.submit(_test_video, video, items[video.identity], windows[video.identity],
                                         paths, bundles, output, deadline)
                active[future] = video.identity
            if not active:
                break
            finished, _ = wait(active, timeout=max(0.0, deadline - time.time()), return_when=FIRST_COMPLETED)
            for future in finished:
                identity = future.result()
                del active[future]
                completed += 1
                progress(completed, len(videos), identity)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    runs = [_test_summary(record, videos, output) for record in bundles]
    means = {}
    for label in dict.fromkeys(record['label'] for record in bundles):
        selected = [run for run in runs if run['label'] == label]
        means[label] = {
            'contact_f1': float(np.mean([run['rates']['10']['f1'] for run in selected])),
            'correct_player_f1': float(np.mean([run['rates']['10']['player_f1'] for run in selected])),
            'fully_correct_rallies': float(np.mean([run['rates']['10']['fully_correct_count'] for run in selected])),
        }
    result = {
        'complete': True, 'videos': len(videos), 'left_out_rallies': EXPECTED_LEFT_OUT, 'runs': runs, 'means': means,
    }
    write_json(output / 'test-summary.json.gz', result)
    return result
