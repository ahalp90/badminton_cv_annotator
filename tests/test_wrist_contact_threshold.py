"""Check the configurable wrist gate and compatibility with saved settings."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from annotator.config import BaseAnnotatorConfig, resolve
from annotator.rally.contacts import wrist_contact_near
from annotator.rally_segmentation import segment_video
from experiments.annotator.heuristic_only_tune.run import load_candidates, write_json
from experiments.annotator.heuristic_tuning.corpus_sweep import candidate_id
from experiments.annotator.heuristic_tuning.sweep import (
    CandidateSpec,
    _base_and_serve,
    serialise_spec,
)
from tests.annotator_config import burst_config


@pytest.mark.parametrize('fps', [25.0, 30.0, 60.0])
def test_wrist_distance_stays_in_body_heights_at_each_fps(fps: float) -> None:
    base = BaseAnnotatorConfig(overrides_base30={'wrist_distance_body_heights': 1.8, 'smooth_window': 6.0})
    resolved = resolve(base, fps)
    assert resolved.thresholds.wrist_distance_body_heights == 1.8
    assert resolved.thresholds.smooth_window == round(6.0 * fps / 30.0)
    assert resolve(BaseAnnotatorConfig(), fps).thresholds.wrist_distance_body_heights == 1.4


def test_resolved_wrist_distance_supports_direct_numeric_changes() -> None:
    resolved = resolve(BaseAnnotatorConfig(), 60.0)
    custom = replace(resolved, thresholds=replace(resolved.thresholds, wrist_distance_body_heights=1.1))
    assert custom.thresholds.wrist_distance_body_heights == 1.1
    assert custom.thresholds.smooth_window == resolved.thresholds.smooth_window == 6


def _turning_track() -> np.ndarray:
    frames = np.arange(80)
    horizontal = np.where(frames < 40, frames, 80 - frames) * 0.005
    return np.column_stack((horizontal, np.full(80, 0.5), np.ones(80)))


def test_resolved_wrist_distance_changes_contact_acceptance() -> None:
    track = _turning_track()
    distances = np.full(len(track), 1.6)
    candidates = []
    for threshold in (1.4, 2.0):
        base = BaseAnnotatorConfig(overrides_base30={'wrist_distance_body_heights': threshold, 'smooth_window': 1.0})
        _spans, contacts = segment_video(
            track,
            spans=[(0, len(track))],
            sticky_distances=distances,
            resolved=resolve(base, 30.0),
        )
        candidates.append(contacts)
    strict, loose = candidates
    assert [contact.contact_frame for contact in strict] == [40]
    assert [contact.contact_frame for contact in loose] == [40]
    assert strict[0].wrist_near is False
    assert loose[0].wrist_near is True
    assert loose[0].suppressed is False


def test_default_wrist_gate_keeps_existing_verdicts_and_contacts() -> None:
    assert burst_config().thresholds.wrist_distance_body_heights == 1.4
    distances = np.array([1.39, 1.4, 1.41, np.nan])
    assert [wrist_contact_near(distances, frame, threshold=1.4) for frame in range(4)] == [True, True, False, False]
    assert wrist_contact_near(None, 0, threshold=2.0) is None
    assert wrist_contact_near(distances, 2, threshold=2.0) is True
    assert wrist_contact_near(distances, 3, threshold=2.0) is False

    track = _turning_track()
    kwargs = {'spans': [(0, len(track))], 'sticky_distances': np.full(len(track), 1.4)}
    _, contacts = segment_video(track, resolved=burst_config(), **kwargs)
    assert [contact.contact_frame for contact in contacts] == [1, 40, 78]


def test_current_preprocessing_pickle_contains_only_the_clean_schema() -> None:
    config = BaseAnnotatorConfig(overrides_base30={'wrist_distance_body_heights': 1.8})
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config
    assert 'thresholds' not in vars(restored)
    assert resolve(restored, 60.0).thresholds.wrist_distance_body_heights == 1.8


def test_sweep_routes_and_saves_wrist_distance_without_changing_its_identity(tmp_path: Path) -> None:
    spec = CandidateSpec('wider wrist gate', {'wrist_distance_body_heights': 2.0}, {})
    identity = candidate_id(spec)
    payload = serialise_spec(spec)
    base, serve = _base_and_serve(spec)
    assert serve is None
    assert resolve(base, 60.0).thresholds.wrist_distance_body_heights == 2.0
    assert payload['overrides_base30']['wrist_distance_body_heights'] == 2.0
    path = tmp_path / 'candidates.json.gz'
    write_json(path, {'candidates': [{'label': spec.label, **payload}]})
    _document, restored = load_candidates(path)
    assert restored == [spec]
    assert candidate_id(restored[0]) == identity
    assert spec.overrides_base30 == {'wrist_distance_body_heights': 2.0}
