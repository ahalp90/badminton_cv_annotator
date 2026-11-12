"""The fixed test scorer uses current models, exclusions and review evidence."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from annotator.inference.learned import LearnedModelPrediction
from annotator.rallies.refine_rallies import (
    RallyPrediction,
    RallyReview,
    RallyReviewFeatures,
)
from annotator.types import ContactSequence, FinishedRally
from experiments.annotator.heuristic_only_tune.run import read_json
from experiments.annotator.tree_simplification import input_test


def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[list, list[str]]:
    videos = [SimpleNamespace(identity=f'test-{index:02d}') for index in range(46)]
    paths = {name: tmp_path / name for name in ('test_inputs', 'inpaint_root', 'pose_root', 'test_courts')}
    models = SimpleNamespace(preprocessing=object())
    calls = []
    monkeypatch.setattr(input_test, 'load_models', lambda path: models)
    monkeypatch.setattr(input_test, 'fitted_trees', lambda model: ())
    monkeypatch.setattr(input_test, 'load_paths', lambda path: paths)
    monkeypatch.setattr(input_test, 'load_test_videos', lambda path: videos)
    monkeypatch.setattr(input_test, 'load_manifest', lambda path: {'items': [{'item_id': video.identity} for video in videos]})
    monkeypatch.setattr(input_test.metrics, 'test_left_out', lambda: {
        video.identity: [(0, 1)] * (524 if index == 0 else 0) for index, video in enumerate(videos)
    })
    monkeypatch.setattr(input_test.metrics, 'left_out_windows', lambda rallies: rallies)
    monkeypatch.setattr(input_test.metrics, 'rates', lambda counts: {'fixture': True})

    def evaluate_video(video, item, windows, fitted, configured_paths, output):
        assert fitted is models and configured_paths is paths
        calls.append(video.identity)
        counts = {'labelled_rallies': 3327 if video.identity == 'test-00' else 0,
                  'labelled_contacts': 37184 if video.identity == 'test-00' else 0}
        rows = []
        for tolerance in (5, 10):
            for ranking in ('selector', 'fitted'):
                rows.append({'video': video.identity, 'span': 0, 'tolerance': tolerance, 'ranking': ranking,
                             'correct': int(video.identity == 'test-00'), 'score': .8})
        return {'5': counts, '10': counts}, rows

    monkeypatch.setattr(input_test, 'evaluate_video', evaluate_video)
    return videos, calls


def test_current_bundle_runs_the_fixed_population_without_selection_records(tmp_path, monkeypatch):
    videos, calls = runtime(monkeypatch, tmp_path)
    models, paths, output = tmp_path / 'models', tmp_path / 'paths.toml', tmp_path / 'output'
    result = input_test.evaluate(models, paths, output)
    assert calls == [video.identity for video in videos]
    assert result['complete'] is True and result['videos'] == 46
    assert result['totals']['10']['labelled_rallies'] == 3327
    assert result['totals']['10']['labelled_contacts'] == 37184
    assert set(result['review']) == {'selector-5', 'selector-10', 'fitted-5', 'fitted-10'}
    assert read_json(output / 'settings.json.gz') == {
        'models': str(models.resolve()), 'paths': str(paths.resolve()), 'threads': 2,
    }


@pytest.mark.parametrize('population', ('missing', 'duplicate'))
def test_wrong_video_population_fails_before_inference(tmp_path, monkeypatch, population):
    videos, calls = runtime(monkeypatch, tmp_path)
    if population == 'missing':
        videos.pop()
    else:
        videos[-1] = videos[0]
    with pytest.raises(ValueError, match='46 distinct'):
        input_test.evaluate(tmp_path / 'models', tmp_path / 'paths.toml', tmp_path / 'output')
    assert not calls


def test_changed_exclusion_population_fails_before_inference(tmp_path, monkeypatch):
    _, calls = runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(input_test.metrics, 'test_left_out', lambda: {'test-00': [(0, 1)] * 523})
    with pytest.raises(ValueError, match='524 excluded'):
        input_test.evaluate(tmp_path / 'models', tmp_path / 'paths.toml', tmp_path / 'output')
    assert not calls


def test_failed_video_cannot_leave_a_complete_result(tmp_path, monkeypatch):
    runtime(monkeypatch, tmp_path)

    def fail(*args):
        raise RuntimeError('inference failed')

    monkeypatch.setattr(input_test, 'evaluate_video', fail)
    output = tmp_path / 'output'
    with pytest.raises(RuntimeError, match='inference failed'):
        input_test.evaluate(tmp_path / 'models', tmp_path / 'paths.toml', output)
    assert not (output / 'result.json.gz').exists()


def test_nonempty_output_is_not_reused(tmp_path, monkeypatch):
    runtime(monkeypatch, tmp_path)
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'previous.txt').write_text('previous result')
    with pytest.raises(FileExistsError, match='new or empty'):
        input_test.evaluate(tmp_path / 'models', tmp_path / 'paths.toml', output)


def test_video_scoring_and_review_use_the_same_excluded_window_judgements(tmp_path, monkeypatch):
    video = SimpleNamespace(identity='fixture')
    pose = SimpleNamespace(bboxes=None, scores=None, kps=None, ndet=None)
    inputs = SimpleNamespace(track=None, pose=pose)
    models = SimpleNamespace(preprocessing=object())
    review_features = RallyReviewFeatures((0, 3, 7), np.empty((3, 0)))
    fitted = RallyReview(review_features, np.array([.9, .9, .2]))
    selector = RallyReview(review_features, np.array([.8, .8, .1]))
    source_rallies = tuple(FinishedRally(ContactSequence(span, 0, 1, ()), None) for span in (10, 20))
    refinement = RallyPrediction(source_rallies, (), None, np.array([.8, .1]), {10: 0, 20: 1})
    final_rallies = tuple(FinishedRally(ContactSequence(span, 0, 1, ()), None) for span in (0, 3, 7))
    prediction = LearnedModelPrediction(None, None, refinement, fitted, final_rallies, (), (10, 10, 20))
    stream = {'events': [], 'sequences': []}
    windows = [(10, 20)]
    seen = []
    monkeypatch.setattr(input_test, 'load_test_inputs', lambda *args: inputs)
    monkeypatch.setattr(input_test, 'run_video_kwargs', lambda value: {})
    monkeypatch.setattr(input_test, 'stream_payload', lambda *args: stream)
    monkeypatch.setattr(input_test, 'score_selector_review', lambda value: selector)

    def run_video(*args, **kwargs):
        assert kwargs['models'] is models and kwargs['base'] is models.preprocessing
        assert kwargs['keep_working_data'] is True
        return SimpleNamespace(working_data=SimpleNamespace(learned_model_prediction=prediction))

    def score(item, actual, tolerance, excluded, *, supplied):
        assert actual is stream and excluded is windows and not supplied
        seen.append(tolerance)
        return {'labelled_rallies': 1}, [
            {'span_id': 3, 'outcome': 'fully_correct'},
            {'span_id': 7, 'outcome': 'unlabelled_player'},
        ]

    monkeypatch.setattr(input_test, 'annotate_video', run_video)
    monkeypatch.setattr(input_test.metrics, 'score_stream_for_item', score)
    paths = {name: tmp_path for name in ('inpaint_root', 'pose_root', 'test_courts')}
    (tmp_path / 'streams').mkdir()
    (tmp_path / 'videos').mkdir()
    counts, rows = input_test.evaluate_video(video, {}, windows, models, paths, tmp_path)
    assert seen == [5, 10] and set(counts) == {'5', '10'}
    assert len(rows) == 8 and {row['span'] for row in rows} == {3, 7}
    assert {row['correct'] for row in rows} == {-1, 1}
    assert next(row['score'] for row in rows if row['span'] == 3 and row['ranking'] == 'fitted') == .9


def test_changed_scored_population_cannot_complete(tmp_path, monkeypatch):
    runtime(monkeypatch, tmp_path)
    original = input_test.evaluate_video

    def changed(*args):
        counts, rows = original(*args)
        counts['10'] = counts['10'] | {'labelled_rallies': 0}
        return counts, rows

    monkeypatch.setattr(input_test, 'evaluate_video', changed)
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='3327 scored rallies'):
        input_test.evaluate(tmp_path / 'models', tmp_path / 'paths.toml', output)
    assert not (output / 'result.json.gz').exists()
