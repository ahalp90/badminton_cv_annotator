"""Evaluate a fitted annotator on the fixed 46-video ShuttleSet22 test set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(REPOSITORY := Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPOSITORY / 'src'))

import numpy as np
from threadpoolctl import threadpool_limits

from annotator.annotate_video import annotate_video
from annotator.models import AnnotatorModels, fitted_trees, load_models
from experiments.annotator.contact_records import stream_payload
from experiments.annotator.heuristic_only_tune.data import (
    SearchVideo,
    load_test_inputs,
    load_test_videos,
    run_video_kwargs,
)
from experiments.annotator.heuristic_only_tune.run import load_paths, write_json
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    load_manifest,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts import metrics
from experiments.annotator.tree_simplification.integrate import ranking_metrics
from experiments.annotator.tree_simplification.selector_review import (
    score_selector_review,
)

EXPECTED_VIDEOS = 46
EXPECTED_LEFT_OUT = 524
EXPECTED_RALLIES = 3327
EXPECTED_CONTACTS = 37184
TOLERANCES = (5, 10)
UNKNOWN_REVIEW_OUTCOMES = frozenset(('no_labelled_rally', 'unlabelled_player'))


def evaluate_video(
    video: SearchVideo, item: dict[str, Any], windows: list[tuple[int, int]],
    models: AnnotatorModels, paths: dict[str, Path], output: Path,
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    """Annotate one video, then score contacts and review ranking with the same exclusions."""
    inputs = load_test_inputs(video, paths['inpaint_root'], paths['pose_root'], paths['test_courts'])
    pose = inputs.pose
    annotation = annotate_video(
        inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
        base=models.preprocessing, models=models, keep_working_data=True, **run_video_kwargs(inputs),
    )
    assert annotation.working_data is not None
    prediction = annotation.working_data.learned_model_prediction
    if prediction is None:
        raise ValueError(f'{video.identity}: fitted annotation did not return its contact stream')
    stream = stream_payload(prediction.rally_prediction.sequences, prediction.rally_prediction.events)
    selector_review = score_selector_review(prediction.rally_prediction)
    scores_by_ranking = {}
    for name, review in (('selector', selector_review), ('fitted', prediction.rally_review)):
        scores_by_ranking[name] = dict(zip(review.features.span_ids, review.scores.tolist(), strict=True))

    counts_by_tolerance, judgements_by_tolerance = {}, {}
    review_rows = []
    for tolerance in TOLERANCES:
        counts, judgements = metrics.score_stream_for_item(item, stream, tolerance, windows, supplied=False)
        counts_by_tolerance[str(tolerance)] = counts
        judgements_by_tolerance[str(tolerance)] = judgements
        for judgement in judgements:
            outcome = judgement['outcome']
            correct = -1 if outcome in UNKNOWN_REVIEW_OUTCOMES else int(outcome == 'fully_correct')
            span_id = judgement['span_id']
            for name, scores in scores_by_ranking.items():
                review_rows.append({
                    'video': video.identity, 'span': span_id, 'tolerance': tolerance,
                    'ranking': name, 'outcome': outcome, 'correct': correct, 'score': scores[span_id],
                })
    write_json(output / 'streams' / f'{video.identity}.json.gz', stream)
    write_json(output / 'videos' / f'{video.identity}.json.gz', {
        'counts': counts_by_tolerance, 'judgements': judgements_by_tolerance,
    })
    return counts_by_tolerance, review_rows


def evaluate(models_path: Path, paths_file: Path, output: Path, *, threads: int = 2) -> dict[str, Any]:
    """Evaluate current models in a new or empty output directory."""
    if threads < 1:
        raise ValueError('threads must be positive')
    models = load_models(models_path)
    for tree in fitted_trees(models):
        if type(tree).__module__.startswith('xgboost'):
            tree.set_params(device='cpu')
    paths = load_paths(paths_file)
    videos = load_test_videos(paths['test_inputs'])
    if len(videos) != EXPECTED_VIDEOS or len({video.identity for video in videos}) != EXPECTED_VIDEOS:
        raise ValueError('Test evaluation requires 46 distinct ShuttleSet22 fixtures')
    manifest = load_manifest(paths['test_inputs'])
    items = {item['item_id']: item for item in manifest['items']}
    metrics.INPUTS = paths['test_inputs']
    windows = {identity: metrics.left_out_windows(rallies) for identity, rallies in metrics.test_left_out().items()}
    if sum(map(len, windows.values())) != EXPECTED_LEFT_OUT:
        raise ValueError('The fixed test requires 524 excluded rally windows')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output}: choose a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'streams').mkdir()
    (output / 'videos').mkdir()
    write_json(output / 'settings.json.gz', {
        'models': str(models_path.resolve()), 'paths': str(paths_file.resolve()), 'threads': threads,
    })
    per_video, review_rows = {}, []
    with threadpool_limits(limits=threads):
        for video in videos:
            counts, rows = evaluate_video(video, items[video.identity], windows[video.identity], models, paths, output)
            per_video[video.identity] = counts
            review_rows.extend(rows)
            print(f'Completed {video.identity} ({len(per_video)}/{len(videos)})', flush=True)

    totals, rates, review = {}, {}, {}
    for tolerance in TOLERANCES:
        key = str(tolerance)
        first = next(iter(per_video.values()))[key]
        totals[key] = {name: sum(counts[key][name] for counts in per_video.values()) for name in first}
        rates[key] = metrics.rates(totals[key])
        for ranking in ('selector', 'fitted'):
            rows = [row for row in review_rows if row['ranking'] == ranking and row['tolerance'] == tolerance]
            review[f'{ranking}-{tolerance}'] = ranking_metrics(
                np.asarray([row['correct'] for row in rows], dtype=np.int8),
                np.asarray([row['score'] for row in rows], dtype=np.float64),
            )
    if (totals['10']['labelled_rallies'], totals['10']['labelled_contacts']) != (EXPECTED_RALLIES, EXPECTED_CONTACTS):
        raise ValueError('The fixed test requires 3327 scored rallies and 37184 labelled contacts')
    result = {
        'complete': True, 'models': str(models_path.resolve()), 'paths': str(paths_file.resolve()),
        'videos': len(videos), 'left_out_rallies': EXPECTED_LEFT_OUT, 'primary_tolerance': 10,
        'per_video': per_video, 'totals': totals, 'rates': rates, 'review': review,
    }
    write_json(output / 'review-rows.json.gz', review_rows)
    write_json(output / 'result.json.gz', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    evaluate(args.models, args.paths, args.output, threads=args.threads)


if __name__ == '__main__':
    main()
