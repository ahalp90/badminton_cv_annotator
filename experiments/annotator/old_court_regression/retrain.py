"""Work items for the old-court retrain: one function per resumable item.

Each item writes its outputs atomically and finishes with a small summary
file, so a rerun skips completed items. Fitting and prediction call the
current ``annotator`` functions; this module only supplies the original video
populations, orders and old input files.
"""

import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from annotator.contacts.features import REGION_FIELDS, ContactFeatures
from annotator.contacts.model import CONTACT_FEATURE_NAMES, contact_feature_matrix
from annotator.hybrid import ContactEvidence, features_from_evidence
from annotator.models import AnnotatorModels, SideGeometry, load_models, save_models
from annotator.outcomes.point_winner import Half
from annotator.run_video import RunCapture
from annotator.sequence.confidence import ConfidenceFeatures, build_confidence_features
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.features import positive_probabilities
from annotator.training.confidence import fit_confidence_model
from annotator.training.contact import ContactTrainingVideo, fit_contact_model
from annotator.training.sequences import (
    LabelledRally,
    SequenceModelFit,
    SequenceTrainingVideo,
    fit_sequence_models,
    overlapping_rallies,
)
from annotator.training.workflow import (
    DEFAULT_TRAINING_SETTINGS,
    Split,
    VideoInput,
    evaluation_counts,
    load_labels,
    sequence_training_video,
    whole_rally_correctness,
)
from dataset_builder.vision import (
    annotation_result_payload,
    load_json_gz,
    load_npy_xz,
    save_json_gz,
)
from experiments.annotator.old_court_regression.old_inputs import (
    OldVideoInputs,
    all_source_rallies,
    development_label_rows,
    load_development_inputs,
    load_test_inputs,
    run_old_video,
    stream_from_output,
    trusted_test_label_rows,
    write_label_csv,
)

TRAINING_GROUPS = ('A', 'B', 'C', 'D')
VALIDATION_GROUP = 'V'
TEST_VIDEO_IDS = (
    8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57,
)
SELECTION_COUNT_NAMES = ('positive', 'nearby_negative', 'sampled_other_negative', 'selected')
# Each contact fit's saved selection record, relative to --contact-records.
CONTACT_RECORDS = {
    'final_40': 'final_contact_model/final_contact_model_result.json',
    'held_A': 'training_video_scores/group_A/training_video_score_result.json',
    'held_B': 'training_video_scores/group_B/training_video_score_result.json',
    'held_C': 'training_video_scores/group_C/training_video_score_result.json',
    'held_D': 'training_video_scores/group_D/training_video_score_result.json',
    'v_32': 'final_contact_scores/group_V/group_score_result.json',
    'frozen_final_40': 'final_contact_model/final_contact_model_result.json',
}
# PR149 files, relative to --pr149-root (scratch/contact_det_closing_pass).
PR149_TEST_STREAM = 'results/followups/local_boundary_broader_predictions_fixed_membership.json.gz'
PR149_TEST_CONFIDENCE = 'results/serve_followups/chosen_acceptance_broader_predictions.json.gz'
PR149_DEVELOPMENT_STREAM = 'raw/followups/development_predictions/local_predictions.json.gz'
PR149_DEVELOPMENT_BOUNDS = 'results/followups/local_boundary_result_fixed_membership.json.gz'
PR149_DEVELOPMENT_CORRECT = 1209  # sections fully correct under the PR149 scorer; checks the file choice


@dataclass(frozen=True)
class RunSettings:
    """Input and output locations shared by every work item."""

    output: Path
    dev_stages: Path
    test_inputs: Path
    shots_master: Path
    groups_file: Path
    split_file: Path
    clean_labels: Path
    pr149_root: Path
    contact_records: Path | None
    frozen_dev_features: Path | None
    frozen_test_features: Path | None
    shuttleset22_annotations: Path | None


@dataclass(frozen=True)
class Development:
    """The 40 original development videos: groups A-D for training and V for validation."""

    group_by_video: dict[str, str]  # original group order, then original order within each group
    fps_by_video: dict[str, float]
    id_by_video: dict[str, int]

    def videos(self, groups: Sequence[str]) -> tuple[str, ...]:
        """Return the named groups' videos in original group order."""
        return tuple(video for group in groups for video, owner in self.group_by_video.items() if owner == group)

    def id_order(self, videos: Sequence[str]) -> tuple[str, ...]:
        """Return videos in ShuttleSet ID order, as the 40- and 32-video trees used."""
        return tuple(sorted(videos, key=self.id_by_video.__getitem__))


def load_development(settings: RunSettings) -> Development:
    """Read the fixed groups and per-video frame rates from the original records."""
    groups = load_json(settings.groups_file)
    split = load_json(settings.split_file)
    group_by_video = {}
    for group in groups['groups']:
        for video in group['videos']:
            group_by_video[video['fixture']] = group['group']
    for video in groups['fixed_validation_videos']:
        group_by_video[video] = VALIDATION_GROUP
    by_fixture = {video['fixture']: video for video in split['videos']}
    if set(group_by_video) != set(by_fixture) or len(group_by_video) != 40:
        raise ValueError('groups and development split must name the same 40 videos')
    for group in (*TRAINING_GROUPS, VALIDATION_GROUP):
        if list(group_by_video.values()).count(group) != 8:
            raise ValueError(f'group {group} must contain eight videos')
    return Development(
        group_by_video,
        {fixture: float(video['fps']) for fixture, video in by_fixture.items()},
        {fixture: int(video['video_id']) for fixture, video in by_fixture.items()},
    )


def contact_fit_videos(development: Development) -> dict[str, tuple[str, ...]]:
    """Name each contact fit and its training videos in sampling order.

    The fold trees used group order; the 40-video final tree and the 32-video
    tree that scored V used ID order (``final_contact_scores/group_V``).
    """
    training = development.videos(TRAINING_GROUPS)
    fits = {'final_40': development.id_order(development.videos((*TRAINING_GROUPS, VALIDATION_GROUP)))}
    for held_out in TRAINING_GROUPS:
        fits[f'held_{held_out}'] = tuple(
            video for video in training if development.group_by_video[video] != held_out
        )
    fits['v_32'] = development.id_order(training)
    return fits


def load_json(path: Path) -> dict[str, Any]:
    """Read one plain or gzip-compressed JSON object."""
    if path.name.endswith('.gz'):
        return load_json_gz(path)
    return json.loads(path.read_text())


def dump_atomically(value: object, path: Path) -> None:
    """Write a joblib file under a temporary name, then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def label_path(settings: RunSettings, split: str, identity: str) -> Path:
    return settings.output / 'labels' / split / f'{identity}.csv'


def summary_path(settings: RunSettings, *parts: str) -> Path:
    """Locate one item's summary file; its presence marks the item complete."""
    return settings.output.joinpath(*parts[:-1], f'{parts[-1]}.json.gz')


def write_labels_job(settings: RunSettings) -> dict[str, Any]:
    """Write the supported label CSVs and the in-memory all-source read."""
    development = load_development(settings)
    shots = pd.read_csv(settings.shots_master, usecols=['vid', 'set_id', 'rally', 'frame_num', 'player_side'])
    for identity, video_id in development.id_by_video.items():
        rallies = development_label_rows(shots, video_id)
        write_label_csv(label_path(settings, 'development', identity), rallies)
    trusted_rows = trusted_test_label_rows(settings.clean_labels)
    if tuple(sorted(trusted_rows)) != TEST_VIDEO_IDS:
        raise ValueError('trusted test labels must cover the 47 ShuttleSet22 test videos, including video 15')
    for video_id, rallies in trusted_rows.items():
        write_label_csv(label_path(settings, 'test', str(video_id)), rallies)
    summary = {
        'development_videos': len(development.id_by_video), 'test_videos': len(trusted_rows),
        'trusted_test_rallies': sum(map(len, trusted_rows.values())),
        'all_source': settings.shuttleset22_annotations is not None,
    }
    if settings.shuttleset22_annotations is not None:
        trusted = {
            video_id: tuple(
                LabelledRally(rally_id, tuple(frame for frame, _side in contacts),
                              tuple(side for _frame, side in contacts))
                for rally_id, contacts in rallies
            )
            for video_id, rallies in trusted_rows.items()
        }
        all_source = all_source_rallies(settings.shuttleset22_annotations, TEST_VIDEO_IDS, trusted)
        save_json_gz(settings.output / 'labels' / 'test_all_source.json.gz', {
            str(video_id): [rally_payload(rally) for rally in rallies] for video_id, rallies in all_source.items()
        })
        summary['all_source_rallies'] = sum(map(len, all_source.values()))
    save_json_gz(summary_path(settings, 'labels', 'summary'), summary)
    return summary


def rally_payload(rally: LabelledRally) -> dict[str, Any]:
    return {'id': rally.identity, 'frames': list(rally.frames),
            'sides': [None if side is None else side.value for side in rally.sides]}


def rally_from_payload(payload: dict[str, Any]) -> LabelledRally:
    return LabelledRally(payload['id'], tuple(payload['frames']),
                         tuple(None if side is None else Half(side) for side in payload['sides']))


def heuristic_evidence(inputs: OldVideoInputs) -> ContactEvidence:
    """Run the heuristic pass on old inputs and keep the evidence the trees use."""
    capture = RunCapture()
    run_old_video(inputs, None, capture)
    if capture.contact_evidence is None:
        raise ValueError(f'{inputs.identity}: heuristic annotation did not capture contact evidence')
    return capture.contact_evidence


def frozen_feature_path(root: Path | None, split: str, identity: str) -> Path | None:
    """Locate the research-time feature rows for one video, if supplied."""
    if root is None:
        return None
    if split == 'test':
        return root / 'videos' / f'ss22_{int(identity):02d}' / 'contact_features.npy.xz'
    return root / 'videos' / identity / 'contact_features.npy.xz'


def feature_parity(rows: np.ndarray, frozen_path: Path | None) -> dict[str, Any] | None:
    """Compare regenerated feature rows with the frozen research rows.

    Equal row identities with a few differing values show heuristic or
    feature-code drift; different identities show a changed search region.
    """
    if frozen_path is None:
        return None
    frozen = load_npy_xz(frozen_path)
    same_rows = (
        len(frozen) == len(rows)
        and np.array_equal(frozen['frame'], rows['frame'])
        and np.array_equal(frozen['interval_id'], rows['interval_id'])
    )
    parity: dict[str, Any] = {'rows': len(rows), 'frozen_rows': len(frozen), 'same_row_identities': bool(same_rows)}
    if not same_rows:
        parity['shared_frames'] = len(np.intersect1d(frozen['frame'], rows['frame']))
        return parity
    differing_rows = np.zeros(len(rows), dtype=bool)
    differing_columns = {}
    for name in (*CONTACT_FEATURE_NAMES, *REGION_FIELDS):
        regenerated = rows[name].astype(np.float64)
        saved = frozen[name].astype(np.float64)
        both_missing = np.isnan(regenerated) & np.isnan(saved)
        differs = ~((regenerated == saved) | both_missing)
        if differs.any():
            differing_columns[name] = int(differs.sum())
            differing_rows |= differs
    parity['rows_with_any_difference'] = int(differing_rows.sum())
    parity['differing_columns'] = differing_columns
    return parity


def development_features_job(settings: RunSettings, identity: str) -> dict[str, Any]:
    """Build one development video's contact features from its old inputs."""
    started = time.perf_counter()
    development = load_development(settings)
    inputs = load_development_inputs(settings.dev_stages, identity, development.fps_by_video[identity])
    evidence = heuristic_evidence(inputs)
    features = features_from_evidence(evidence)
    rallies = load_labels(label_path(settings, 'development', identity), len(evidence.track))
    dump_atomically(features, settings.output / 'features' / f'{identity}.joblib')
    summary = {
        'video': identity, 'group': development.group_by_video[identity], 'frame_count': len(evidence.track),
        'heuristic_spans': len(evidence.heuristic_spans), 'feature_rows': len(features.rows),
        'labelled_rallies': len(rallies), 'labelled_contacts': sum(len(rally.frames) for rally in rallies),
        'parity_with_frozen_features': feature_parity(
            features.rows, frozen_feature_path(settings.frozen_dev_features, 'development', identity)),
        'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(summary_path(settings, 'features', identity), summary)
    return summary


def contact_training_video(
    settings: RunSettings, development: Development, identity: str, frozen: bool,
) -> ContactTrainingVideo:
    """Pair one video's feature rows with its labelled contact frames."""
    frame_count = load_json_gz(summary_path(settings, 'features', identity))['frame_count']
    rallies = load_labels(label_path(settings, 'development', identity), frame_count)
    frames = np.asarray(sorted(frame for rally in rallies for frame in rally.frames), dtype=np.int32)
    if frozen:
        rows = load_npy_xz(frozen_feature_path(settings.frozen_dev_features, 'development', identity))
    else:
        rows = joblib.load(settings.output / 'features' / f'{identity}.joblib').rows
    return ContactTrainingVideo(identity, rows, frames, development.fps_by_video[identity])


def selection_comparison(record_path: Path, counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Compare per-video selected example counts with the original fit's record."""
    saved = load_json(record_path)['training_selection']['videos']
    if set(saved) != set(counts):
        return {'record': str(record_path), 'same_videos': False}
    comparison: dict[str, Any] = {'record': str(record_path), 'same_videos': True}
    for name in SELECTION_COUNT_NAMES:
        comparison[f'{name}_equal_videos'] = sum(saved[video][name] == counts[video][name] for video in counts)
        comparison[f'{name}_total_original'] = sum(saved[video][name] for video in counts)
        comparison[f'{name}_total_new'] = sum(counts[video][name] for video in counts)
    return comparison


def check_row_comparison(record_path: Path, model: Any, rows_by_video: dict[str, np.ndarray]) -> dict[str, Any]:
    """Score the 80 saved check rows of the original 40-video tree with a new tree."""
    differences = []
    missing = 0
    for check in load_json(record_path)['model_check_rows']:
        rows = rows_by_video[check['fixture']]
        position = np.flatnonzero((rows['frame'] == check['frame']) & (rows['interval_id'] == check['interval_id']))
        if len(position) != 1:
            missing += 1
            continue
        score = positive_probabilities(model, contact_feature_matrix(rows[position]))[0]
        differences.append(abs(float(score) - check['contact_score']))
    return {
        'check_rows_scored': len(differences), 'check_rows_missing': missing,
        'largest_score_difference': max(differences) if differences else None,
        'mean_score_difference': float(np.mean(differences)) if differences else None,
    }


def contact_fit_job(settings: RunSettings, name: str) -> dict[str, Any]:
    """Fit one contact tree; ``frozen_final_40`` uses the research-time features."""
    started = time.perf_counter()
    development = load_development(settings)
    frozen = name == 'frozen_final_40'
    videos = contact_fit_videos(development)['final_40' if frozen else name]
    training = [contact_training_video(settings, development, identity, frozen) for identity in videos]
    fit = fit_contact_model(training, DEFAULT_TRAINING_SETTINGS.contact)
    dump_atomically(fit.model, settings.output / 'contact' / f'{name}.joblib')
    counts = {video: asdict(value) for video, value in fit.selection.video_counts.items()}
    summary: dict[str, Any] = {
        'fit': name, 'videos': list(videos), 'feature_source': 'frozen research rows' if frozen else 'regenerated',
        'selection_by_video': counts,
        'positive_rows': int(fit.selection.labels.sum()), 'selected_rows': len(fit.selection.labels),
        'iterations': int(fit.model.n_iter_), 'seconds': round(time.perf_counter() - started, 1),
    }
    record = None if settings.contact_records is None else settings.contact_records / CONTACT_RECORDS[name]
    if record is not None and record.is_file():
        summary['selection_vs_original'] = selection_comparison(record, counts)
        if name in ('final_40', 'frozen_final_40'):
            rows_by_video = {video.identity: video.features for video in training}
            summary['check_rows_vs_original'] = check_row_comparison(record, fit.model, rows_by_video)
    save_json_gz(summary_path(settings, 'contact', name), summary)
    return {key: summary[key] for key in ('fit', 'positive_rows', 'selected_rows', 'iterations', 'seconds')}


def pool_job(settings: RunSettings, identity: str) -> dict[str, Any]:
    """Score one A-D video with its held-group tree and build its option pool."""
    started = time.perf_counter()
    development = load_development(settings)
    group = development.group_by_video[identity]
    inputs = load_development_inputs(settings.dev_stages, identity, development.fps_by_video[identity])
    evidence = heuristic_evidence(inputs)
    features: ContactFeatures = joblib.load(settings.output / 'features' / f'{identity}.joblib')
    labels = label_path(settings, 'development', identity)
    rallies = load_labels(labels, len(evidence.track))
    contact_model = joblib.load(settings.output / 'contact' / f'held_{group}.joblib')
    video = sequence_training_video(
        VideoInput(identity, group, Split.TRAIN, labels), evidence, features, rallies, contact_model,
        DEFAULT_TRAINING_SETTINGS.contact_prediction, SideGeometry.VIDEO,
    )
    dump_atomically(video, settings.output / 'pools' / f'{identity}.joblib')
    summary = {
        'video': identity, 'group': group, 'initial_sequences': len(video.initial), 'events': len(video.events),
        'options': len(video.pool.options), 'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(summary_path(settings, 'pools', identity), summary)
    return summary


def stream_payload(sequences: Sequence[ContactSequence], events: Sequence[ContactEvent]) -> dict[str, Any]:
    """Store final sections and the full stream with each contact's side."""
    def event_row(event: ContactEvent) -> list:
        return [event.frame, event.probability, None if event.side is None else event.side.value]

    return {
        'sequences': [
            {'span_id': sequence.span_id, 'start_frame': sequence.start_frame, 'end_frame': sequence.end_frame,
             'events': [event_row(event) for event in sequence.events]}
            for sequence in sequences
        ],
        'events': [event_row(event) for event in events],
    }


def stream_from_payload(payload: dict[str, Any]) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    def event_from_row(row: list) -> ContactEvent:
        frame, probability, side = row
        return ContactEvent(frame, probability, None if side is None else Half(side))

    sequences = tuple(
        ContactSequence(sequence['span_id'], sequence['start_frame'], sequence['end_frame'],
                        tuple(event_from_row(row) for row in sequence['events']))
        for sequence in payload['sequences']
    )
    return sequences, tuple(event_from_row(row) for row in payload['events'])


def fit_downstream(videos: Sequence[SequenceTrainingVideo]) -> tuple[SequenceModelFit, Any, np.ndarray]:
    """Fit the sequence stack, then confidence on its held-group output.

    Confidence rows follow ``fit_from_manifest``: held-group predictions, their
    matching held-group insertion model, and whole-rally correctness labels.
    """
    settings = DEFAULT_TRAINING_SETTINGS
    sequence_fit = fit_sequence_models(videos, settings.sequences)
    base_rows, gap_rows, correctness = [], [], []
    for video in videos:
        prediction = sequence_fit.cross_fitted[video.identity]
        insertion_model = sequence_fit.cross_fitted_insertion_models[video.identity]
        features = build_confidence_features(prediction, insertion_model, video.fps)
        sequences = {sequence.span_id: sequence for sequence in prediction.sequences}
        correctness.extend(
            whole_rally_correctness(sequences[span_id], video.rallies, video.fps) for span_id in features.span_ids
        )
        base_rows.append(features.base)
        gap_rows.append(features.gap)
    base, gap = np.concatenate(base_rows), np.concatenate(gap_rows)
    labels = np.asarray(correctness, dtype=np.int8)
    confidence = fit_confidence_model(
        ConfidenceFeatures(tuple(range(len(base))), base, gap), labels, settings.confidence,
    )
    return sequence_fit, confidence, labels


def save_bundle(contact_model: Any, sequence_fit: SequenceModelFit, confidence: Any, directory: Path) -> None:
    """Save one complete bundle with the supported settings and video-wide net position."""
    settings = DEFAULT_TRAINING_SETTINGS
    save_models(AnnotatorModels(
        contact_model, sequence_fit.models, confidence, settings.contact_prediction, settings.preprocessing,
        SideGeometry.VIDEO,
    ), directory)


def sequence_fit_job(settings: RunSettings) -> dict[str, Any]:
    """Fit the downstream and confidence trees on A-D, then save both complete bundles."""
    started = time.perf_counter()
    development = load_development(settings)
    videos = [joblib.load(settings.output / 'pools' / f'{identity}.joblib')
              for identity in development.videos(TRAINING_GROUPS)]
    sequence_fit, confidence, labels = fit_downstream(videos)
    for video in videos:
        prediction = sequence_fit.cross_fitted[video.identity]
        save_json_gz(settings.output / 'dev_heldout' / f'{video.identity}.json.gz',
                     stream_payload(prediction.sequences, prediction.events))
    for directory, contact_name in (('bundle', 'final_40'), ('bundle_v32', 'v_32')):
        contact_model = joblib.load(settings.output / 'contact' / f'{contact_name}.joblib')
        save_bundle(contact_model, sequence_fit, confidence, settings.output / directory)
    summary = {
        'training_videos': [video.identity for video in videos],
        'confidence_rows': len(labels), 'confidence_correct': int((labels == 1).sum()),
        'confidence_wrong': int((labels == 0).sum()), 'confidence_unjudgeable': int((labels == -1).sum()),
        'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(summary_path(settings, 'sequence', 'summary'), summary)
    return summary


def evaluate_job(settings: RunSettings, split: str, identity: str) -> dict[str, Any]:
    """Run full annotation with a saved bundle on one held-out video's old inputs.

    V uses the 32-video contact tree, which never saw V. Test videos use the
    40-video tree, as PR149 did.
    """
    started = time.perf_counter()
    if split == 'validation':
        development = load_development(settings)
        models = load_models(settings.output / 'bundle_v32')
        inputs = load_development_inputs(settings.dev_stages, identity, development.fps_by_video[identity])
    else:
        models = load_models(settings.output / 'bundle')
        inputs = load_test_inputs(settings.test_inputs, int(identity))
    capture = RunCapture()
    result = run_old_video(inputs, models, capture)
    prediction = capture.hybrid
    if prediction is None:
        raise ValueError(f'{identity}: full annotation did not capture the model prediction')
    directory = settings.output / 'eval' / split / identity
    save_json_gz(directory / 'annotator_result.json.gz', annotation_result_payload(identity, result))
    save_json_gz(directory / 'stream.json.gz', {
        **stream_payload(prediction.refined.sequences, prediction.refined.events),
        'confidence': dict(zip(map(str, prediction.confidence.features.span_ids),
                               map(float, prediction.confidence.scores), strict=True)),
    })
    summary = {
        'video': identity, 'split': split, 'frame_count': len(inputs.track), 'fps': inputs.fps,
        'sections': len(prediction.refined.sequences), 'contacts': len(prediction.refined.events),
        'rallies_with_verdict': len(result.verdict_rows),
        'rallies_with_landing': sum(landing is not None for landing in result.landings.values()),
        'hit_heights': len(result.hit_height_by_frame), 'hit_height_failures': len(result.hit_height_failures),
        'parity_with_frozen_features': feature_parity(prediction.features.rows, frozen_feature_path(
            settings.frozen_dev_features if split == 'validation' else settings.frozen_test_features,
            split, identity)),
        'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(summary_path(settings, 'eval', split, identity, 'summary'), summary)
    return {key: summary[key] for key in ('video', 'sections', 'contacts', 'seconds')}


def old_streams_job(settings: RunSettings) -> dict[str, Any]:
    """Split the saved PR149 outputs into one small stream file per video.

    Test: the final 47-video stream and its confidence scores. Development:
    the held-group A-D stream, whose boundary-corrected bounds are stored in
    a separate result file.
    """
    root = settings.pr149_root
    test = load_json(root / PR149_TEST_STREAM)
    confidence = load_json(root / PR149_TEST_CONFIDENCE)
    policy = confidence['frozen_policies']['gap']['comparison']
    scores_by_video = {
        video['fixture']: {str(row['span_id']): row['gap_score'] for row in video['rows']}
        for video in confidence['videos']
    }
    for video in test['videos']:
        fixture = video['fixture']
        sequences, events = stream_from_output(video['output'], fixture)
        save_json_gz(settings.output / 'old' / 'test' / f'{fixture}.json.gz',
                     {**stream_payload(sequences, events), 'confidence': scores_by_video[fixture]})
    bounds_result = load_json(root / PR149_DEVELOPMENT_BOUNDS)['comparison_to_input_detector']['10']
    if bounds_result['paired']['correct_after'] != PR149_DEVELOPMENT_CORRECT:
        raise ValueError('development boundary result is not the selected PR149 file')
    bounds: dict[str, dict[int, tuple[int, int]]] = {}
    for section in bounds_result['sections']:
        bounds.setdefault(section['fixture'], {})[section['span_id']] = (section['start_frame'], section['end_frame'])
    development_output = load_json(root / PR149_DEVELOPMENT_STREAM)['outputs']
    padded = 0
    for fixture, video_bounds in bounds.items():
        sequences, events = stream_from_output(development_output, fixture, video_bounds)
        unpadded, _events = stream_from_output(development_output, fixture)
        padded += sum((old.start_frame, old.end_frame) != (new.start_frame, new.end_frame)
                      for old, new in zip(unpadded, sequences, strict=True))
        save_json_gz(settings.output / 'old' / 'development' / f'{fixture}.json.gz', stream_payload(sequences, events))
    summary = {
        'test_videos': len(test['videos']), 'development_videos': len(bounds),
        'development_sections_with_corrected_bounds': padded, 'confidence_threshold': policy['threshold'],
    }
    save_json_gz(summary_path(settings, 'old', 'summary'), summary)
    return summary


def score_stream(
    sequences: Sequence[ContactSequence], events: Sequence[ContactEvent], rallies: Sequence[LabelledRally],
    fps: float, confidence: dict[str, float] | None,
) -> dict[str, Any]:
    """Score one stream with the supported evaluator and keep per-section detail."""
    counts = evaluation_counts(events, sequences, rallies, fps)
    sections = []
    correct_rallies = []
    for sequence in sequences:
        correctness = whole_rally_correctness(sequence, rallies, fps)
        if correctness == 1:
            correct_rallies.append(overlapping_rallies(sequence, rallies)[0].identity)
        score = None if confidence is None else confidence[str(sequence.span_id)]
        sections.append([sequence.span_id, correctness, score])
    return {**counts, 'labelled_rallies': len(rallies), 'correct_rallies': correct_rallies, 'sections': sections}


def compare_job(settings: RunSettings, split: str, identity: str) -> dict[str, Any]:
    """Score old and new outputs for one video with the same labels and scorer."""
    started = time.perf_counter()
    if split == 'development':
        new = load_json_gz(settings.output / 'dev_heldout' / f'{identity}.json.gz')
        old_path = settings.output / 'old' / 'development' / f'{identity}.json.gz'
        frame_count = load_json_gz(summary_path(settings, 'features', identity))['frame_count']
        fps = load_development(settings).fps_by_video[identity]
        label_split = 'development'
    else:
        new = load_json_gz(settings.output / 'eval' / split / identity / 'stream.json.gz')
        old_path = settings.output / 'old' / 'test' / f'{identity}.json.gz' if split == 'test' else None
        evaluated = load_json_gz(summary_path(settings, 'eval', split, identity, 'summary'))
        frame_count, fps = evaluated['frame_count'], evaluated['fps']
        label_split = 'development' if split == 'validation' else 'test'
    label_sets = {'trusted': load_labels(label_path(settings, label_split, identity), frame_count)}
    all_source_path = settings.output / 'labels' / 'test_all_source.json.gz'
    if split == 'test' and all_source_path.is_file():
        rallies = tuple(rally_from_payload(rally) for rally in load_json_gz(all_source_path)[identity])
        if any(not 0 <= frame < frame_count for rally in rallies for frame in rally.frames):
            raise ValueError(f'{identity}: an all-source label frame lies outside the video')
        label_sets['all_source'] = rallies
    streams = {'new': new}
    if old_path is not None:
        streams['old'] = load_json_gz(old_path)
    scores: dict[str, Any] = {}
    for side, payload in streams.items():
        sequences, events = stream_from_payload(payload)
        scores[side] = {
            name: score_stream(sequences, events, rallies, fps, payload.get('confidence'))
            for name, rallies in label_sets.items()
        }
    result = {'video': identity, 'split': split, 'scores': scores, 'seconds': round(time.perf_counter() - started, 1)}
    save_json_gz(summary_path(settings, 'compare', split, identity), result)
    return {'video': identity, 'seconds': result['seconds']}


COUNT_NAMES = ('labelled_contacts', 'predicted_contacts', 'matched_contacts', 'labelled_rallies',
               'whole_rally_judged', 'whole_rally_correct', 'whole_rally_unjudgeable')


def totals(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-video counts and derive contact precision/recall and rally recall."""
    summed: dict[str, Any] = {name: sum(row[name] for row in rows) for name in COUNT_NAMES}
    summed['fully_correct_rallies'] = sum(len(set(row['correct_rallies'])) for row in rows)
    summed['contact_precision'] = summed['matched_contacts'] / summed['predicted_contacts']
    summed['contact_recall'] = summed['matched_contacts'] / summed['labelled_contacts']
    summed['rally_recall'] = summed['fully_correct_rallies'] / summed['labelled_rallies']
    return summed


def confidence_metrics(rows: Sequence[dict[str, Any]], threshold: float, kept_count: int | None) -> dict[str, Any]:
    """Rank sections by confidence; judge them by whole-rally correctness.

    Reports ranking quality on judged sections, the fixed PR149 threshold and
    the same number of top-ranked sections that PR149 kept.
    """
    sections = [section for row in rows for section in row['sections']]
    rally_ids = []
    for video_index, row in enumerate(rows):
        recovered = iter(row['correct_rallies'])
        for section in row['sections']:
            rally_ids.append((video_index, next(recovered)) if section[1] == 1 else None)
    labelled = sum(row['labelled_rallies'] for row in rows)
    correctness = np.asarray([section[1] for section in sections])
    scores = np.asarray([section[2] for section in sections], dtype=np.float64)
    judged = correctness != -1

    def kept_counts(kept: np.ndarray) -> dict[str, Any]:
        kept_judged = int((kept & judged).sum())
        kept_correct = int((kept & (correctness == 1)).sum())
        recovered_rallies = {rally_ids[index] for index in np.flatnonzero(kept & (correctness == 1))}
        return {'kept': int(kept.sum()), 'kept_judged': kept_judged, 'kept_correct': kept_correct,
                'kept_correct_rallies': len(recovered_rallies),
                'precision': kept_correct / kept_judged if kept_judged else None,
                'recall': len(recovered_rallies) / labelled}

    order = np.argsort(-scores, kind='stable')
    top = np.zeros(len(scores), dtype=bool)
    top[order[:kept_count if kept_count is not None else int((scores >= threshold).sum())]] = True
    return {
        'sections': len(sections), 'judged': int(judged.sum()),
        'roc_auc': float(roc_auc_score(correctness[judged] == 1, scores[judged])),
        'average_precision': float(average_precision_score(correctness[judged] == 1, scores[judged])),
        'at_pr149_threshold': {'threshold': threshold, **kept_counts(scores >= threshold)},
        'top_ranked_matching_pr149_count': kept_counts(top),
    }


def build_report(rows_by_split: dict[str, list[dict[str, Any]]], threshold: float) -> dict[str, Any]:
    """Combine per-video comparisons into totals, per-video changes and confidence metrics."""
    report: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        sides = rows[0]['scores']
        report[split] = {}
        for label_set in sides['new']:
            entry: dict[str, Any] = {}
            for side in sides:
                entry[side] = totals([row['scores'][side][label_set] for row in rows])
            if 'old' in sides:
                entry['per_video'] = [per_video_change(row, label_set) for row in rows]
            # Held-group development streams carry no confidence scores.
            if split != 'development':
                old_kept = None
                if 'old' in sides:
                    old_metrics = confidence_metrics([row['scores']['old'][label_set] for row in rows], threshold, None)
                    entry['old']['confidence'] = old_metrics
                    old_kept = old_metrics['at_pr149_threshold']['kept']
                entry['new']['confidence'] = confidence_metrics(
                    [row['scores']['new'][label_set] for row in rows], threshold, old_kept)
            report[split][label_set] = entry
    return report


def report_job(settings: RunSettings) -> dict[str, Any]:
    """Write comparison.json.gz and comparison.md from every per-video comparison."""
    threshold = load_json_gz(summary_path(settings, 'old', 'summary'))['confidence_threshold']
    development = load_development(settings)
    populations = {
        'test': [str(video_id) for video_id in TEST_VIDEO_IDS],
        'development': list(development.videos(TRAINING_GROUPS)),
        'validation': list(development.videos((VALIDATION_GROUP,))),
    }
    rows_by_split = {
        split: [load_json_gz(summary_path(settings, 'compare', split, video)) for video in videos]
        for split, videos in populations.items()
    }
    report = build_report(rows_by_split, threshold)
    save_json_gz(settings.output / 'comparison.json.gz', report)
    (settings.output / 'comparison.md').write_text(markdown_report(report))
    return {split: {label_set: {side: entry[side]['whole_rally_correct'] for side in ('old', 'new') if side in entry}
                    for label_set, entry in by_label.items()} for split, by_label in report.items()}


def per_video_change(row: dict[str, Any], label_set: str) -> dict[str, Any]:
    old = row['scores']['old'][label_set]
    new = row['scores']['new'][label_set]
    old_correct, new_correct = set(old['correct_rallies']), set(new['correct_rallies'])
    return {
        'video': row['video'],
        **{f'{name}_old': old[name] for name in ('matched_contacts', 'predicted_contacts', 'whole_rally_correct')},
        **{f'{name}_new': new[name] for name in ('matched_contacts', 'predicted_contacts', 'whole_rally_correct')},
        'labelled_contacts': new['labelled_contacts'], 'labelled_rallies': new['labelled_rallies'],
        'rallies_gained': sorted(new_correct - old_correct), 'rallies_lost': sorted(old_correct - new_correct),
    }


def markdown_report(report: dict[str, Any]) -> str:
    """Write the headline numbers as short tables; the JSON keeps every detail."""
    lines = ['# Old-court retrain: old (PR149) and new outputs, one scorer', '',
             'Contacts match within 10 frames at 30 FPS. Rally recall counts each correctly recovered rally once.',
             '']
    titles = {'test': 'ShuttleSet22 test, 47 videos (video 15 included)',
              'development': 'Development A-D held-group output, 32 videos',
              'validation': 'Validation V, 8 videos (no integrated PR149 output exists)'}
    for split, by_label in report.items():
        for label_set, entry in by_label.items():
            lines += [f'## {titles[split]}: {label_set} labels', '',
                      '| Side | Contact P | Contact R | Fully correct | Rally recall | Judged | Unjudgeable |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for side in ('old', 'new'):
                if side not in entry:
                    continue
                values = entry[side]
                lines.append(
                    f'| {side} | {values["contact_precision"]:.1%} | {values["contact_recall"]:.1%} | '
                    f'{values["fully_correct_rallies"]:,} / {values["labelled_rallies"]:,} | {values["rally_recall"]:.1%} | '
                    f'{values["whole_rally_judged"]:,} | {values["whole_rally_unjudgeable"]:,} |'
                )
            if 'per_video' in entry:
                gained = sum(len(change['rallies_gained']) for change in entry['per_video'])
                lost = sum(len(change['rallies_lost']) for change in entry['per_video'])
                lines += ['', f'Fully correct rallies gained {gained}, lost {lost}.']
            confidence_lines = []
            for side in ('old', 'new'):
                if side in entry and 'confidence' in entry[side]:
                    metrics = entry[side]['confidence']
                    fixed, matched = metrics['at_pr149_threshold'], metrics['top_ranked_matching_pr149_count']
                    confidence_lines.append(
                        f'| {side} | {metrics["roc_auc"]:.3f} | {metrics["average_precision"]:.3f} | '
                        f'{fixed["kept"]} | {fixed["kept_correct"]} / {fixed["kept_judged"]} | '
                        f'{matched["kept"]} | {matched["kept_correct"]} / {matched["kept_judged"]} |'
                    )
            if confidence_lines:
                header = ('| Confidence | ROC AUC | Avg precision | Kept at PR149 threshold | Correct / judged '
                          '| Top-ranked kept | Correct / judged |')
                lines += ['', header, '|---|---:|---:|---:|---:|---:|---:|', *confidence_lines]
            lines.append('')
    return '\n'.join(lines)
