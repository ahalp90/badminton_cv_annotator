"""Staged selector targets and review labels retain their saved endpoint limits."""

import lzma
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.features import PHYSICAL_FEATURE_NAMES
from annotator.sequence.rally_model import RallyLayerPrediction, score_rally_review
from annotator.sequence.repairs import RallyRepairs
from annotator.training.sequences import (
    LabelledRally,
    RallyBoundaryPolicy,
    SequenceTrainingVideo,
)
from experiments.annotator import development as experiment
from experiments.annotator.tree_simplification import input_stages, pools, targets
from experiments.annotator.tree_simplification.options import expand_pool


def training_video() -> SequenceTrainingVideo:
    events = (ContactEvent(300, .9, Half.TOP), ContactEvent(400, .9, Half.BOT))
    sequence = ContactSequence(0, 180, 521, events)
    # Five-frame contact errors can put trimmed bounds just outside the label limits.
    rally = LabelledRally('rally', (305, 395), (Half.TOP, Half.BOT))
    physical = {event.frame: np.zeros(len(PHYSICAL_FEATURE_NAMES)) for event in events}
    return SequenceTrainingVideo('fixture', 'A', 30., (sequence,), events, RallyRepairs((), {}, physical), (rally,), 600)


@pytest.mark.parametrize(('policy', 'expected'), [(RallyBoundaryPolicy(2, 3), 0), (RallyBoundaryPolicy(5, 5), 1)])
def test_saved_policy_controls_wide_five_and_review_labels(
    tmp_path: Path, policy: RallyBoundaryPolicy, expected: int,
) -> None:
    pool = expand_pool(training_video(), max_edits=0, deletion_candidates=0, boundary_policy=policy)
    np.testing.assert_array_equal(pool.targets, [expected])
    assert pools.original_options(pool).boundary_policy == policy
    assert pools.pool_summary(pool, 'train')['boundary_policy'] == asdict(policy)

    path = tmp_path / 'pools' / 'videos' / 'fixture.joblib'
    path.parent.mkdir(parents=True)
    joblib.dump(pool, path)
    saved = joblib.load(path)
    assert saved.boundary_policy == policy
    manifest = {'boundary_policy': asdict(policy)}
    output = tmp_path / 'targets'
    summary = targets._target_job(targets.TargetJob('fixture', 'A', 'train', path, path.parents[1], manifest, output))
    assert summary is not None
    assert summary['pools_manifest']['boundary_policy'] == asdict(policy)
    with lzma.open(output / 'videos' / 'fixture.npy.xz', 'rb') as stream:
        five = np.load(stream, allow_pickle=False)
    np.testing.assert_array_equal(five, [expected])
    np.testing.assert_array_equal(targets.five_frame_preferred_targets(saved, five), [expected])

    refined = RallyLayerPrediction(saved.sequences, saved.video.events, saved.runtime, (np.asarray([.8]),), {0: 0})
    prediction = SimpleNamespace(refined=refined, confidence=score_rally_review(refined, None))
    _rankings, rows = input_stages._review_rows(saved.video, 'fixture', prediction, boundary_policy=policy)
    assert len(rows) == 4
    assert {row['correct'] for row in rows} == {expected}


def test_target_stage_rejects_pool_policy_that_differs_from_manifest(tmp_path: Path) -> None:
    pool = expand_pool(training_video(), max_edits=0, boundary_policy=RallyBoundaryPolicy(5, 5))
    path = tmp_path / 'fixture.joblib'
    joblib.dump(pool, path)
    job = targets.TargetJob('fixture', 'A', 'train', path, tmp_path, {
        'boundary_policy': asdict(RallyBoundaryPolicy(2, 3)),
    }, tmp_path / 'targets')
    with pytest.raises(ValueError, match='boundary policy differs'):
        targets._target_job(job)
    assert not (tmp_path / 'targets' / 'videos' / 'fixture.npy.xz').exists()


@pytest.mark.parametrize('policy', [RallyBoundaryPolicy(2, 3), RallyBoundaryPolicy(5, 5)])
def test_pool_cli_passes_boundary_policy(monkeypatch: pytest.MonkeyPatch, policy: RallyBoundaryPolicy) -> None:
    captured: dict[str, Any] = {}

    def prepare(*_args: Any, **kwargs: Any) -> list[dict[str, object]]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(pools, 'prepare_pools', prepare)
    assert pools.main([
        '--prepared', 'prepared', '--contact-run', 'contacts', '--output', 'output',
        '--boundary-before-seconds', str(policy.before_seconds),
        '--boundary-after-seconds', str(policy.after_seconds),
    ]) == 0
    assert captured['boundary_policy'] == policy


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_export_passes_device_to_confidence_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str,
) -> None:
    policy = asdict(RallyBoundaryPolicy(5, 5))
    settings = {'boundary_policy': policy}
    inputs = {'preprocessing': {'fixture': True}}
    monkeypatch.setattr(input_stages, '_export_inputs', lambda *_args: (settings, inputs, tmp_path, tmp_path, 1))
    monkeypatch.setattr(input_stages, 'source_record', lambda: {'revision': 'fixture'})

    class StopAfterFit(Exception):
        pass

    captured: dict[str, Any] = {}

    def fit_review(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        raise StopAfterFit

    monkeypatch.setattr(input_stages.integrate, 'fit_review', fit_review)
    output = tmp_path / 'export'
    with pytest.raises(StopAfterFit):
        input_stages.export_run(tmp_path / 'run', tmp_path / 'targets', output, device=device)
    assert captured['device'] == device
    receipt = experiment.read_json(output / 'settings.json.gz')
    assert receipt['device'] == device
    assert receipt['boundary_policy'] == policy
