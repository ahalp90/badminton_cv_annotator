"""Exercise the complete four-model fitter with real trees and spawned workers."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.contacts.build_features import ContactFeatures, contact_feature_dtype
from annotator.courts.frame_geometry import SceneCourt
from annotator.inference.learned import predict_from_contact_features
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.models import fitted_trees, load_models
from annotator.training import workflow
from annotator.training.sequences import LabelledRally
from annotator.training.trees import TreeFitConfig
from annotator.types import ContactCandidate, CourtHalf, TrackedPlayers, WristGaps


def synthetic_video(seed: int) -> tuple[PreparedAnnotationInputs, ContactFeatures, tuple[LabelledRally, ...]]:
    random = np.random.default_rng(seed)
    labels, spans = [], []
    strengths, hitters = {}, {}
    next_start = 60
    for rally_index in range(16):
        start = next_start
        frame = start + 20
        frames, pieces = [], [[]]
        top = bool(random.integers(0, 2))
        for contact_index in range(int(random.integers(4, 9))):
            frames.append(frame)
            pieces[-1].append(frame)
            hitters[frame] = CourtHalf.TOP if top else CourtHalf.BOT
            top = not top
            strengths[frame] = random.uniform(.75, 1) if random.random() < .85 else random.uniform(.2, .5)
            if random.random() < .12 and contact_index >= 1:
                frame += int(random.integers(100, 150))
                pieces.append([])
                top = bool(random.integers(0, 2))
            else:
                frame += int(random.integers(25, 45))
        end = frames[-1] + 25
        spans.append((start, end))
        pieces = [piece for piece in pieces if piece]
        # Some long gaps remain inside a labelled rally, giving confidence both outcomes.
        if random.random() < .3:
            pieces = [frames]
        for piece_index, piece in enumerate(pieces):
            labels.append(LabelledRally(f'{rally_index}-{piece_index}', tuple(piece), tuple(hitters[hit] for hit in piece)))
        for _ in range(int(random.integers(4, 12))):
            candidate = int(random.integers(start, end))
            if all(abs(candidate - hit) > 8 for hit in strengths):
                strengths[candidate] = random.uniform(.75, 1) if random.random() < .06 else random.uniform(0, .5)
        next_start = end + int(random.integers(40, 90))

    frame_count = next_start + 100
    candidates = np.asarray(sorted(strengths))
    rows = np.zeros(len(candidates), dtype=contact_feature_dtype())
    rows['frame'] = candidates
    rows['shuttle_speed_t+0'] = [strengths[frame] for frame in candidates]
    features = ContactFeatures(rows, [(0, frame_count)], [(0, frame_count)])
    track = np.zeros((frame_count, 3))
    track[:, 2] = 1
    per_player = np.tile([.5, .8], (frame_count, 1))
    for frame, side in hitters.items():
        per_player[frame] = (.1, 2.) if side == CourtHalf.TOP else (2., .1)
    players = TrackedPlayers(
        np.tile([0, 1], (frame_count, 1)), np.full((frame_count, 2, 2), np.nan),
        np.full((frame_count, 2), np.nan), np.full(frame_count, np.nan),
    )
    bboxes = np.zeros((frame_count, 2, 4))
    bboxes[:, 0, 3], bboxes[:, 1, 3] = 300., 800.
    evidence = PreparedAnnotationInputs(
        video_config=scale_config_to_fps(BaseAnnotatorConfig(), 30.), resolution=(1280., 720.), track=track,
        pose_kps=np.zeros((frame_count, 2, 17, 2)), bboxes=bboxes, players=players,
        wrist_gaps=WristGaps(per_player.min(axis=1), per_player), tracker_intervals=[(0, frame_count)],
        exclusion_mask=np.zeros(frame_count, dtype=bool), raw_exclusion_mask=np.zeros(frame_count, dtype=bool),
        heuristic_spans=spans, heuristic_contacts=[ContactCandidate(0, int(frame), None, None) for frame in candidates[:2]],
        scene_courts=(SceneCourt(0, frame_count, {}, (500., 520.), .1),),
        rejected_shuttle_frames=np.zeros(frame_count, dtype=bool),
    )
    return evidence, features, tuple(labels)


def test_real_training_with_three_groups_saves_and_runs_four_models(tmp_path, monkeypatch):
    data = {f'video-{index}': synthetic_video(index) for index in range(6)}
    features_by_evidence = {id(evidence): features for evidence, features, _ in data.values()}
    manifest = workflow.TrainingManifest(tmp_path, tuple(
        workflow.VideoInput(identity, group, workflow.Split.TRAIN, Path(identity))
        for identity, group in zip(data, 'AABBCC', strict=True)
    ))
    monkeypatch.setattr(workflow, 'prepare_saved_video_inputs', lambda _run, identity, _settings: data[identity][0])
    monkeypatch.setattr(workflow, 'contact_features_from_inputs', lambda evidence: features_by_evidence[id(evidence)])
    monkeypatch.setattr(workflow, 'load_labels', lambda path, _count: data[path.name][2])
    defaults = workflow.DEFAULT_TRAINING_SETTINGS
    settings = replace(
        defaults, contact=replace(defaults.contact, max_iter=40, max_leaf_nodes=7, device='cpu'),
        rally=replace(defaults.rally, tree=TreeFitConfig(max_iter=30, max_leaf_nodes=7, device='cpu')),
        confidence=replace(defaults.confidence, max_iter=20, device='cpu'),
    )
    bundle = tmp_path / 'bundle'
    with threadpool_limits(limits=1):
        workflow.fit_from_manifest(manifest, bundle, settings, workers=2, gpu_fits=2)
        restored = load_models(bundle, device='cpu')
        evidence, features, labels = synthetic_video(100)
        prediction = predict_from_contact_features(evidence, features, restored)
    assert len(fitted_trees(restored)) == 4
    assert restored.rally_selector.selector is not restored.followup_selector.selector
    assert all(tree.classes_.tolist() == [0, 1] for tree in fitted_trees(restored))
    assert any(count > 1 for count in Counter(prediction.source_span_ids).values())
    assert len(prediction.rally_confidence) == len(prediction.rallies)
    assert np.isfinite(prediction.rally_confidence).all()
    assert workflow.evaluation_counts(prediction.events, prediction.sequences, labels, 30.)['whole_rally_correct'] > 0
    assert workflow.load_training_settings(bundle / 'fit-settings.json.gz') == settings
    assert not list(bundle.glob('.training-*'))
