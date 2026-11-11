"""Keep frozen heuristic settings within the benchmark's supplied boundaries."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import annotator.annotate_video as video_module
import annotator.models as model_module
from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.rally_results.landing_and_winner import SHIPPED_LANDING_FILTER_OPTIONS
from annotator.types import ContactEvent, ContactSequence, CourtHalf, FinishedRally
from dataset_builder import vision
from experiments.annotator.contact_records import stream_payload
from experiments.annotator.heuristic_tuning.corpus_sweep import candidate_id
from experiments.annotator.heuristic_tuning.sweep import CandidateSpec, serialise_spec
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import read_json
from tests.annotator_result import empty_annotation


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts = Path(__file__).resolve().parents[1] / 'experiments/annotator/monotrack_benchmark/benchmark_scripts'
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module('experiments.annotator.monotrack_benchmark.benchmark_scripts.annotator_runner')


@pytest.fixture
def inputs(runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    frame_count = 80
    paths = {key: tmp_path / key for key in ('bundle', 'inputs', 'inpaint_root', 'pose_root')}
    config = {'paths': paths, 'execution': {'annotator_workers': 2}}
    item = {
        'item_id': 'ss22_08', 'video_id': 8, 'directory': '08 example', 'frame_count': frame_count,
        'fps': 25.0, 'resolution': [1280, 720],
        'spans': [{'span_id': 0, 'rally_id': 'rally_1', 'start_frame': 0, 'end_frame': 45,
                   'label_frames': [10, 20], 'label_sides': ['Top', 'Bot']}],
    }
    track = np.zeros((frame_count, 3))
    guards = np.zeros(frame_count, dtype=np.uint8)
    pose = SimpleNamespace(
        bboxes=np.zeros((frame_count, 1, 4)), scores=np.ones((frame_count, 1)),
        kps=np.zeros((frame_count, 1, 17, 2)), ndet=np.ones(frame_count, dtype=int),
    )
    court = SimpleNamespace(
        net_band=(0.4, 0.6), resolution=(1280.0, 720.0), court_info={},
        homography_rows=object(), landing_error_band_m=0.05,
    )
    court_vision = SimpleNamespace(
        evidence=SimpleNamespace(inputs=court, court_present=np.ones(frame_count, dtype=bool)),
        raw_cuts=((0, 30), (30, frame_count)),
    )
    array_paths = []
    pose_paths = []
    court_paths = []
    model_paths = []
    models = SimpleNamespace(
        preprocessing=BaseAnnotatorConfig(),
        contact_settings=SimpleNamespace(score_cutoff=0.9),
    )

    def load_array(path: Path) -> np.ndarray:
        array_paths.append(path)
        return track if path.name == 'shuttle_track_inpainted.npy.xz' else guards

    def load_pose(path: Path, frames: int) -> Any:
        assert frames == frame_count
        pose_paths.append(path)
        return pose

    def load_courts(path: Path, **kwargs: Any) -> Any:
        assert kwargs == {'video_id': item['item_id'], 'frame_count': frame_count, 'resolution': (1280.0, 720.0)}
        court_paths.append(path)
        return court_vision

    def load_models(path: Path) -> Any:
        model_paths.append(path)
        return models

    monkeypatch.setattr(runner, 'read_array', load_array)
    monkeypatch.setattr(vision, 'load_pose_arrays', load_pose)
    monkeypatch.setattr(vision, 'load_court_vision', load_courts)
    monkeypatch.setattr(model_module, 'load_models', load_models)
    return SimpleNamespace(
        config=config, item=item, track=track, guards=guards, pose=pose, court=court, vision=court_vision,
        models=models, array_paths=array_paths, pose_paths=pose_paths, court_paths=court_paths, model_paths=model_paths,
    )


def _selected_spec() -> CandidateSpec:
    return CandidateSpec(
        'selected', {'wrist_distance_body_heights': 1.8, 'smooth_window': 6.0}, {'require_fast_burst': False},
    )


@pytest.mark.parametrize('condition', ['ours_our_courts', 'ours_mono_courts'])
def test_heuristic_keeps_benchmark_inputs_spans_and_court_flags(
    runner: ModuleType, inputs: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str,
) -> None:
    spec = _selected_spec()

    def infer(*args: Any, **kwargs: Any) -> Any:
        for actual, expected in zip(args, (inputs.track, inputs.pose.bboxes, inputs.pose.scores,
                                          inputs.pose.kps, inputs.pose.ndet), strict=True):
            assert actual is expected
        assert kwargs['models'] is None
        assert kwargs['heuristic_only'] is True
        assert kwargs['spans'] == [(0, 45)]
        assert kwargs['fps'] == 25.0
        video_config = scale_config_to_fps(kwargs['base'], kwargs['fps'])
        assert video_config.thresholds.wrist_distance_body_heights == 1.8
        assert video_config.thresholds.smooth_window == 5
        assert kwargs['landing_options'] is SHIPPED_LANDING_FILTER_OPTIONS
        assert kwargs['shuttle_grades'] is inputs.guards
        assert kwargs['court_present'] is inputs.vision.evidence.court_present
        assert kwargs['homography_rows'] is inputs.court.homography_rows
        result = empty_annotation(kwargs['spans'], [])._replace(rallies=(FinishedRally(
            ContactSequence(0, *kwargs['spans'][0], (
                ContactEvent(10, np.nan, CourtHalf.TOP), ContactEvent(20, np.nan, CourtHalf.BOT),
            )),
            CourtHalf.BOT,
        ),))
        return SimpleNamespace(result=result)

    monkeypatch.setattr(video_module, 'annotate_video', infer)
    runner.run_item((inputs.config, inputs.item, condition, tmp_path), heuristic_spec=spec)
    result = read_json(tmp_path / 'ss22_08.json.gz')
    assert result['status'] == 'completed'
    assert result['mode'] == 'heuristic'
    assert result['boundaries'] == 'provided'
    assert result['candidate_id'] == candidate_id(spec)
    assert result['heuristic_settings'] == serialise_spec(spec)
    assert result['stream']['events'] == [[10, None, 'Top'], [20, None, 'Bot']]
    assert not inputs.model_paths
    assert inputs.pose_paths == [inputs.config['paths']['pose_root'] / inputs.item['directory']]
    assert inputs.array_paths == [
        inputs.config['paths']['inpaint_root'] / inputs.item['directory'] / filename
        for filename in ('shuttle_track_inpainted.npy.xz', 'shuttle_guard_codes_inpainted.npy.xz')
    ]
    assert inputs.court_paths == [inputs.config['paths']['inputs'] / runner.COURTS[condition] / inputs.item['item_id']]


@pytest.mark.parametrize('condition', ['ours_our_courts', 'ours_mono_courts', 'ours_discovery'])
def test_full_default_keeps_the_selected_models_and_contact_stream(
    runner: ModuleType, inputs: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str,
) -> None:
    events = (ContactEvent(10, 0.8, CourtHalf.TOP), ContactEvent(20, 0.9, CourtHalf.BOT))
    sequences = (ContactSequence(0, 0, 45, events),)

    def infer(*args: Any, **kwargs: Any) -> Any:
        assert kwargs['models'] is inputs.models
        assert kwargs['base'] is inputs.models.preprocessing
        assert kwargs['heuristic_only'] is False
        assert kwargs['spans'] == (None if condition == 'ours_discovery' else [(0, 45)])
        assert kwargs['keep_working_data'] is True
        return SimpleNamespace(
            result=empty_annotation([], []),
            working_data=SimpleNamespace(learned_model_prediction=SimpleNamespace(
                sequences=sequences, events=events,
            )),
        )

    monkeypatch.setattr(video_module, 'annotate_video', infer)
    runner.run_item((inputs.config, inputs.item, condition, tmp_path))
    result = read_json(tmp_path / 'ss22_08.json.gz')
    assert result['mode'] == 'full'
    assert result['status'] == 'completed'
    assert result['stream'] == stream_payload(sequences, events)
    assert 'heuristic_settings' not in result
    assert inputs.model_paths == [inputs.config['paths']['bundle']]


def test_full_mode_keeps_existing_model_checks(
    runner: ModuleType, inputs: SimpleNamespace, tmp_path: Path,
) -> None:
    inputs.models.preprocessing = BaseAnnotatorConfig(rejected_grades=frozenset())
    with pytest.raises(ValueError, match='settings differ'):
        runner.run_item((inputs.config, inputs.item, 'ours_our_courts', tmp_path))


def test_failed_heuristic_keeps_contacts_and_rallies_in_the_scored_population(
    runner: ModuleType, inputs: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('failed inference')

    monkeypatch.setattr(video_module, 'annotate_video', fail)
    runner.run_item((inputs.config, inputs.item, 'ours_our_courts', tmp_path), _selected_spec())
    result = read_json(tmp_path / 'ss22_08.json.gz')
    assert result['status'] == 'failed'
    assert result['failure_type'] == 'RuntimeError'
    assert result['stream'] == {'events': [], 'sequences': []}
    scorer = importlib.import_module('experiments.annotator.monotrack_benchmark.benchmark_scripts.score')
    for metrics in scorer.score_item(inputs.item, result).values():
        assert metrics['labelled_contacts'] == 2
        assert metrics['labelled_rallies'] == 1
        assert metrics['predicted_contacts'] == metrics['correct_side_contacts'] == 0


@pytest.mark.parametrize(('stage', 'count'), [('shortlist', 1), ('selected', 2)])
def test_heuristic_settings_require_one_selected_candidate(
    runner: ModuleType, tmp_path: Path, stage: str, count: int,
) -> None:
    specs = [_selected_spec(), CandidateSpec('other', {}, {})][:count]
    path = tmp_path / 'selected.json.gz'
    runner.write_json(path, {'stage': stage, 'candidates': [serialise_spec(spec) for spec in specs]})
    with pytest.raises(ValueError, match='exactly one selected configuration'):
        runner.load_heuristic_settings(path)


def test_cli_passes_frozen_settings_to_workers(
    runner: ModuleType, inputs: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _selected_spec()
    settings = tmp_path / 'selected.json.gz'
    runner.write_json(settings, {'stage': 'selected', 'candidates': [{'label': spec.label, **serialise_spec(spec)}]})
    output = tmp_path / 'output'
    tasks = []

    def run(task: Any, heuristic_spec: Any = None) -> None:
        tasks.append((task, heuristic_spec))

    @contextmanager
    def pool(*, max_workers: int) -> Any:
        assert max_workers == 2
        yield SimpleNamespace(map=map)

    monkeypatch.setattr(runner, 'ProcessPoolExecutor', pool)
    monkeypatch.setattr(runner, 'run_item', run)
    monkeypatch.setattr(runner, 'load_config', lambda _path: inputs.config)
    monkeypatch.setattr(runner, 'load_manifest', lambda _path: {'items': [inputs.item]})
    monkeypatch.setattr('sys.argv', [
        'annotator_runner', '--config', 'config.toml', '--condition', 'ours_our_courts',
        '--output', str(output), '--heuristic-settings', str(settings), '--only', 'ss22_08',
    ])
    runner.main()
    assert tasks == [((inputs.config, inputs.item, 'ours_our_courts', output), spec)]


def test_cli_rejects_unknown_smoke_video_clearly(
    runner: ModuleType, inputs: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, 'load_config', lambda _path: inputs.config)
    monkeypatch.setattr(runner, 'load_manifest', lambda _path: {'items': [inputs.item]})
    monkeypatch.setattr('sys.argv', ['annotator_runner', '--config', 'config.toml', '--condition', 'ours_our_courts',
                                    '--output', str(tmp_path / 'output'), '--only', 'unknown'])
    with pytest.raises(ValueError, match='outside the fixed benchmark population'):
        runner.main()
