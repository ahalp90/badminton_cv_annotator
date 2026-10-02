"""Scene attribution permits serve candidates before tracked court evidence begins."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from annotator import hybrid
from annotator.contacts.features import ContactFeatures, contact_feature_dtype
from annotator.contacts.model import (
    CONTACT_FEATURE_NAMES,
    ContactModelConfig,
    ScoredContacts,
)
from annotator.courts.scenes import SceneCourt
from annotator.hybrid import ContactEvidence, sequence_inputs
from annotator.models import SideGeometry
from annotator.sequence.contacts import initial_sequences
from annotator.sequence.refine import build_option_pool
from annotator.training.workflow import Split, VideoInput, sequence_training_video
from annotator.types import StickyResult


def test_scene_serve_lookback_without_tracked_players_keeps_unknown_sides():
    frame_count = 400
    scenes = (SceneCourt(0, 100, {}, (500.0, 520.0), 0.1), SceneCourt(200, 400, {}, (500.0, 520.0), 0.1))
    search_intervals = [(0, 100), (155, 400)]

    track = np.zeros((frame_count, 3))
    track[:, 2] = 1
    in_scene = np.zeros(frame_count, dtype=bool)
    in_scene[0:100] = True
    in_scene[200:400] = True
    # The sticky tracker leaves +inf distances outside tracker segments (scene rows).
    distances_per_slot = np.where(in_scene[:, None], 0.5, np.inf) * np.ones((frame_count, 2))
    distances_per_slot[in_scene, 1] = 0.8
    picks = np.where(in_scene[:, None], [[0, 1]], -1)
    sticky = StickyResult(
        np.where(in_scene, 0.5, np.inf), picks, np.zeros(frame_count, dtype=int),
        np.full((frame_count, 2, 2), np.nan), np.full((frame_count, 2), np.nan),
        distances_per_slot, distances_per_slot.copy(), in_scene.copy(),
    )
    bboxes = np.zeros((frame_count, 2, 4))
    bboxes[:, 0, 3] = 300.0
    bboxes[:, 1, 3] = 800.0

    rows = np.zeros(sum(end - start for start, end in search_intervals), dtype=contact_feature_dtype())
    position = 0
    for interval_id, (start, end) in enumerate(search_intervals):
        rows['frame'][position:position + end - start] = np.arange(start, end)
        rows['interval_id'][position:position + end - start] = interval_id
        position += end - start
    candidate_frames = np.array([160, 170, 220, 260])
    candidates = rows[np.isin(rows['frame'], candidate_frames)]
    contacts = ScoredContacts(candidates, np.array([0.5, 0.4, 0.95, 0.93]), np.array([2, 3]))

    evidence = ContactEvidence(
        identity='scene-gap', fps=30.0, resolution=(1280.0, 720.0), track=track, pose_kps=np.zeros((frame_count, 2, 17, 2)),
        bboxes=bboxes, sticky=sticky, tracker_intervals=[(0, 100), (200, 400)],
        exclusion_mask=~in_scene, heuristic_spans=[(200, 300)], raw_contact_frames=[220, 260],
        scene_courts=scenes, net_band=(500.0, 520.0),
    )

    pools = []
    for geometry in (SideGeometry.VIDEO, SideGeometry.SCENE):
        scores, events, side_for_frame = sequence_inputs(evidence, contacts, geometry)
        assert side_for_frame(160) is None
        assert side_for_frame(170) is None
        assert side_for_frame(220) is not None
        initial = initial_sequences(evidence.heuristic_spans, events)
        pools.append(build_option_pool(
            initial, events, scores, rows, search_intervals, fps=30.0, side_for_frame=side_for_frame,
        ))
    video_pool, scene_pool = pools
    assert scene_pool.serve_inputs.rows_by_key
    assert scene_pool.options == video_pool.options
    np.testing.assert_array_equal(scene_pool.sequence_features, video_pool.sequence_features)


@pytest.mark.parametrize('enabled', [False, True])
def test_crossfit_and_inference_share_rejection_before_repair_shortlists(monkeypatch, annotator_models, enabled):
    frame_count = 100
    rows = np.zeros(frame_count, dtype=contact_feature_dtype())
    rows['frame'] = np.arange(frame_count)
    rows['region_current_raw'][[1, 10, 16, 40, 60, 80]] = 1
    rows['shuttle_speed_t+0'][[1, 10, 16, 40, 60, 80]] = [.8, .99, .95, .8, .7, .95]
    rows['pose_valid_top_t+0'][16] = 1
    features = ContactFeatures(rows, [(0, frame_count)], [(0, frame_count)])
    mask = np.zeros(frame_count, dtype=bool)
    mask[[1, 10, 40, 60]] = True
    sticky = StickyResult(
        np.full(frame_count, np.inf), np.full((frame_count, 2), -1), np.zeros(frame_count),
        np.full((frame_count, 2, 2), np.nan), np.full((frame_count, 2), np.nan),
        np.full((frame_count, 2), np.inf), np.full((frame_count, 2), np.inf), np.ones(frame_count, dtype=bool),
    )
    track = np.zeros((frame_count, 3))
    track[:, 2] = 1
    evidence = ContactEvidence(
        'masked', 30.0, (1280.0, 720.0), track, np.zeros((frame_count, 2, 17, 2)),
        np.zeros((frame_count, 2, 4)), sticky, [(0, frame_count)], np.zeros(frame_count, dtype=bool),
        [(0, 90)], [16, 80], (), (500.0, 520.0), shuttle_hallucination_mask=mask,
    )
    probability_column = CONTACT_FEATURE_NAMES.index('shuttle_speed_t+0')

    def predict(matrix: np.ndarray) -> np.ndarray:
        probabilities = matrix[:, probability_column]
        return np.column_stack((1.0 - probabilities, probabilities))

    config = ContactModelConfig(reject_masked_without_player=enabled)
    models = replace(annotator_models, contact=SimpleNamespace(predict_proba=predict), contact_settings=config)
    training = sequence_training_video(
        VideoInput('masked', 'group', Split.TRAIN, Path('unused.csv')), evidence, features, (),
        models.contact, config, models.side_geometry,
    )

    def refine(spans, events, scores, feature_rows, search_intervals, *, fps, side_for_frame, **_kwargs):
        initial = initial_sequences(spans, events)
        pool = build_option_pool(
            initial, events, scores, feature_rows, search_intervals, fps=fps, side_for_frame=side_for_frame,
        )
        return SimpleNamespace(pool=pool)

    monkeypatch.setattr(hybrid, 'features_from_evidence', lambda _evidence: features)
    monkeypatch.setattr(hybrid, 'refine_contact_sequences', refine)
    monkeypatch.setattr(hybrid, 'score_rally_confidence', lambda *_args: SimpleNamespace())
    inference = hybrid.predict_contacts(evidence, models)
    expected_candidates = [16, 80] if enabled else [1, 10, 16, 40, 60, 80]
    np.testing.assert_array_equal(inference.contacts.candidates['frame'], expected_candidates)
    assert [event.frame for event in training.events] == list(inference.contacts.frames)
    assert list(inference.contacts.frames) == ([16, 80] if enabled else [10, 80])
    for pool in (training.pool, inference.refined.pool):
        later = [event.frame for event in pool.later_candidates[0]]
        assert later == ([] if enabled else [40, 60])
        if enabled:
            assert set(pool.physical).isdisjoint({1, 10, 40, 60})
            assert not pool.serve_inputs.rows_by_key
        else:
            assert 1 in pool.physical
            assert pool.serve_inputs.rows_by_key
