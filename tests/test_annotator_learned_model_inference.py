"""Scene attribution permits serve candidates before tracked court evidence begins."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from sklearn.dummy import DummyClassifier

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.contacts.build_features import ContactFeatures, contact_feature_dtype
from annotator.contacts.score_candidates import (
    CONTACT_FEATURE_NAMES,
    ContactModelConfig,
    ScoredContacts,
    contact_feature_matrix,
)
from annotator.courts.frame_geometry import SceneCourt
from annotator.inference import learned as learned_model_inference
from annotator.inference.heuristic import select_heuristic_contacts
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.rallies.build_contact_proposals import FEATURE_NAMES, build_rally_context
from annotator.rallies.refine_rallies import (
    REVIEW_FEATURE_NAMES,
    RallyPrediction,
    RallySelector,
)
from annotator.training import followup
from annotator.training.rally import RallyFitConfig
from annotator.types import ContactCandidate, CourtHalf, TrackedPlayers, WristGaps


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
    distances_per_slot[260] = (0.1, 2.0)
    picks = np.where(in_scene[:, None], [[0, 1]], -1)
    players = TrackedPlayers(
        picks, np.full((frame_count, 2, 2), np.nan), np.full((frame_count, 2), np.nan), np.full(frame_count, np.nan),
    )
    wrist_gaps = WristGaps(np.where(in_scene, 0.5, np.inf), distances_per_slot)
    bboxes = np.zeros((frame_count, 2, 4))
    bboxes[:, 0, 3] = 300.0
    bboxes[:, 1, 3] = 800.0

    rows = np.zeros(sum(end - start for start, end in search_intervals), dtype=contact_feature_dtype())
    position = 0
    for interval_id, (start, end) in enumerate(search_intervals):
        rows['frame'][position:position + end - start] = np.arange(start, end)
        rows['interval_id'][position:position + end - start] = interval_id
        position += end - start
    candidate_frames = np.array([160, 170, 220, 240, 260])
    candidates = rows[np.isin(rows['frame'], candidate_frames)]
    contacts = ScoredContacts(
        candidates, contact_feature_matrix(candidates), np.array([0.5, 0.4, 0.95, 0.6, 0.93]), np.array([2, 4]),
    )

    evidence = PreparedAnnotationInputs(
         video_config=scale_config_to_fps(BaseAnnotatorConfig(), 30.0), resolution=(1280.0, 720.0), track=track, pose_kps=np.zeros((frame_count, 2, 17, 2)),
        bboxes=bboxes, players=players, wrist_gaps=wrist_gaps, tracker_intervals=[(0, 100), (200, 400)],
        exclusion_mask=~in_scene, heuristic_spans=[(200, 300)],
        heuristic_contacts=[ContactCandidate(0, frame, None, None) for frame in [220, 260]],
        raw_exclusion_mask=~in_scene, rejected_shuttle_frames=np.zeros(frame_count, dtype=bool),
        scene_courts=scenes,
    )

    context = build_rally_context(evidence, contacts)
    selected = {event.frame: event for event in context.events}
    assert selected[220].side is not None
    assert selected[220].wrist_distances == (0.5, 0.8)
    earlier = {event.frame: event for event in context.shortlisted_by_span[0]}
    assert earlier[160].side is None
    assert earlier[170].side is None
    assert earlier[160].wrist_distances is None
    assert earlier[240].wrist_distances == (0.5, 0.8)

    heuristic = select_heuristic_contacts(evidence)[0]
    assert [event.wrist_distances for event in heuristic.sequence.events] == [(0.5, 0.8), (0.1, 2.0)]
    assert [event.side for event in heuristic.sequence.events] == [CourtHalf.BOT, CourtHalf.TOP]


def test_training_and_inference_share_candidates_before_repair_shortlists(monkeypatch, annotator_models, tmp_path):
    frame_count = 100
    rows = np.zeros(frame_count, dtype=contact_feature_dtype())
    rows['frame'] = np.arange(frame_count)
    rows['shuttle_speed_t+0'][[1, 10, 16, 40, 60, 80]] = [.8, .99, .95, .8, .7, .95]
    rows['pose_valid_top_t+0'][16] = 1
    rows = rows[[1, 10, 16, 40, 60, 80]]
    features = ContactFeatures(rows, [(0, frame_count)], [(0, frame_count)])
    mask = np.zeros(frame_count, dtype=bool)
    mask[[1, 10, 40, 60]] = True
    players = TrackedPlayers(
        np.full((frame_count, 2), -1),
        np.full((frame_count, 2, 2), np.nan),
        np.full((frame_count, 2), np.nan),
        np.full(frame_count, np.nan),
    )
    wrist_gaps = WristGaps(np.full(frame_count, np.inf), np.full((frame_count, 2), np.inf))
    track = np.zeros((frame_count, 3))
    track[:, 2] = 1
    evidence = PreparedAnnotationInputs(
         video_config=scale_config_to_fps(BaseAnnotatorConfig(), 30.0), resolution=(1280.0, 720.0),
        track=track, pose_kps=np.zeros((frame_count, 2, 17, 2)), bboxes=np.zeros((frame_count, 2, 4)),
        players=players, wrist_gaps=wrist_gaps, tracker_intervals=[(0, frame_count)],
        exclusion_mask=np.zeros(frame_count, dtype=bool), raw_exclusion_mask=np.zeros(frame_count, dtype=bool),
        heuristic_spans=[(0, 90)], heuristic_contacts=[ContactCandidate(0, frame, None, None) for frame in [16, 80]],
        scene_courts=(), rejected_shuttle_frames=mask,
    )
    probability_column = CONTACT_FEATURE_NAMES.index('shuttle_speed_t+0')
    scored_matrices = []

    def predict(matrix: np.ndarray) -> np.ndarray:
        scored_matrices.append(matrix)
        probabilities = matrix[:, probability_column]
        return np.column_stack((1.0 - probabilities, probabilities))

    config = ContactModelConfig()
    contact_model = SimpleNamespace(predict_proba=predict, set_params=lambda **_kwargs: None)
    models = replace(annotator_models, contact_model=contact_model, contact_settings=config)
    video = followup.save_training_inputs('masked', 'group', evidence, features, (), tmp_path / 'inputs.joblib')
    model_path = tmp_path / 'contact.joblib'
    original_load = followup.joblib.load
    monkeypatch.setattr(followup.joblib, 'load', lambda path: contact_model if path == model_path else original_load(path))
    training_path = followup.prepare_training_data(
        video, model_path, config, RallyFitConfig(), tmp_path / 'training.joblib',
    )
    expected_pool = followup.joblib.load(training_path).pool.options
    training_context = expected_pool.context

    monkeypatch.setattr(learned_model_inference, 'contact_features_from_inputs', lambda _evidence: features)
    inference = learned_model_inference.predict_rallies_and_contacts(evidence, models)
    expected_candidates = [1, 10, 16, 40, 60, 80]
    np.testing.assert_array_equal(inference.scored_candidates.candidates['frame'], expected_candidates)
    assert [event.frame for event in training_context.events] == list(inference.scored_candidates.frames)
    assert list(inference.scored_candidates.frames) == [10, 80]
    for context, matrix in zip((training_context, inference.refinement.pool.context), scored_matrices, strict=True):
        candidate_frames = [event.frame for event in context.shortlisted_by_span[0]]
        assert candidate_frames == [1, 40, 60]
        assert set(context.contact_features_by_frame) == {1, 10, 40, 60, 80}
        for physical in context.contact_features_by_frame.values():
            assert physical.dtype == np.float32
            assert not np.shares_memory(physical, matrix)
        # Frame numbers differ from row indices; keep each contact's own feature values.
        np.testing.assert_allclose(
            [context.contact_features_by_frame[frame][probability_column] for frame in [1, 10, 40, 60, 80]],
            [.8, .99, .8, .7, .95],
        )

    selector = DummyClassifier(strategy='prior').fit(np.zeros((2, len(FEATURE_NAMES))), [0, 1])
    review = DummyClassifier(strategy='prior').fit(np.zeros((2, len(REVIEW_FEATURE_NAMES))), [0, 1])
    fitted_layers = replace(models, rally_selector=RallySelector(selector), review_model=review)
    layer_prediction = learned_model_inference.predict_from_contact_features(evidence, features, fitted_layers)
    assert isinstance(layer_prediction.refinement, RallyPrediction)
    assert layer_prediction.refinement.pool.sequences == expected_pool.sequences
    np.testing.assert_array_equal(layer_prediction.refinement.pool.matrix, expected_pool.matrix)
    assert layer_prediction.review.features.span_ids == tuple(sequence.span_id for sequence in layer_prediction.sequences)
    assert len(layer_prediction.rally_confidence) == len(layer_prediction.sequences)
