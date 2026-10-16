"""Load frozen search videos and score the annotator's discovered rallies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from annotator.config import BaseAnnotatorConfig
from annotator.models import AnnotatorModels
from annotator.outcomes.point_winner import (
    SHIPPED_LANDING_FILTER_OPTIONS,
    alternating_player_sides,
)
from annotator.run_video import AnnotatorResult, RunCapture, run_video
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.shuttle_track import validate_shuttle_track
from annotator.training.sequences import LabelledRally
from annotator.types import ServeStartConfig
from dataset_builder.shuttle_evidence import GUARD_CODES_FILENAME
from dataset_builder.vision import (
    TRACK_FILENAME,
    load_court_vision,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
)
from experiments.annotator.annotator_closeout_new_courts.scripts.evaluate_saved import (
    score_video,
)
from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    labels_for_item,
    load_manifest,
)
from experiments.annotator.old_court_regression.old_inputs import (
    TEST_GUARD_CODES_FILENAME,
    TEST_TRACK_FILENAME,
    OldVideoInputs,
    development_label_rows,
)
from experiments.annotator.old_court_regression.retrain import stream_payload

REPOSITORY = Path(__file__).resolve().parents[3]
TRACKED_SPLIT = (
    REPOSITORY / 'experiments/annotator/monotrack_benchmark/hitnet_retrain/shuttleset_development_split.json.gz'
)
COUNT_FIELDS = (
    'labelled_rallies', 'labelled_contacts', 'predicted_contacts', 'matched_contacts',
    'correct_side_contacts', 'fully_correct_rallies',
)


@dataclass(frozen=True)
class SearchVideo:
    """One video's fixed metadata and human contact labels."""

    identity: str
    dataset: str
    split: str
    fps: float
    resolution: tuple[float, float]
    rallies: tuple[LabelledRally, ...]
    directory: str | None = None
    frame_count: int | None = None


def load_original_videos(shots_master: Path, split_path: Path = TRACKED_SPLIT) -> tuple[SearchVideo, ...]:
    """Read the original 32/8 split and group its labels by set and rally."""
    split = load_json_gz(split_path)
    if split['schema'] != 'contact-full-dataset-development-split/1':
        raise ValueError(f'{split_path}: unexpected original split schema')
    metadata = split['videos']
    counts = {role: sum(row['role'] == role for row in metadata) for role in ('train', 'validation')}
    if counts != {'train': 32, 'validation': 8} or len({row['fixture'] for row in metadata}) != 40:
        raise ValueError('original split must contain 32 training and eight validation videos')
    shots = pd.read_csv(shots_master, usecols=['vid', 'set_id', 'rally', 'frame_num', 'player_side'])
    videos = []
    for row in metadata:
        rallies = []
        for identity, contacts in development_label_rows(shots, int(row['video_id'])):
            frames, sides = zip(*contacts, strict=True)
            rallies.append(LabelledRally(identity, tuple(frames), tuple(sides)))
        videos.append(SearchVideo(
            row['fixture'], 'ShuttleSet', row['role'], float(row['fps']),
            (float(row['width']), float(row['height'])), tuple(rallies),
        ))
    return tuple(videos)


def load_test_videos(inputs_root: Path) -> tuple[SearchVideo, ...]:
    """Read the benchmark's fixed 46-video test population and cleaned labels."""
    videos = []
    for item in load_manifest(inputs_root)['items']:
        videos.append(SearchVideo(
            item['item_id'], 'ShuttleSet22', 'test', float(item['fps']),
            tuple(map(float, item['resolution'])), labels_for_item(item),
            item['directory'], int(item['frame_count']),
        ))
    return tuple(videos)


def _load_inputs(
    video: SearchVideo, track_path: Path, guard_path: Path, pose_dir: Path, court_dir: Path,
) -> OldVideoInputs:
    track = load_npy_xz(track_path)
    frame_count = len(track)
    validate_shuttle_track(track, frame_count)
    if video.frame_count is not None and video.frame_count != frame_count:
        raise ValueError(f'{video.identity}: track frame count differs from the frozen manifest')
    for rally in video.rallies:
        if any(not 0 <= frame < frame_count for frame in rally.frames):
            raise ValueError(f'{video.identity}: rally {rally.identity} has a label outside the video')
    guards = load_npy_xz(guard_path)
    if guards.shape != (frame_count,):
        raise ValueError(f'{video.identity}: guard codes do not cover the video')
    pose = load_pose_arrays(pose_dir, frame_count)
    vision = load_court_vision(
        court_dir, video_id=video.identity, frame_count=frame_count, resolution=video.resolution,
    )
    court = vision.evidence.inputs
    if court is None:
        raise ValueError(f'{video.identity}: saved court evidence has no operational inputs')
    return OldVideoInputs(
        video.identity, video.fps, track, guards, pose, vision.raw_cuts, court,
        vision.evidence.keep_vote, vision.evidence.court_present,
    )


def load_original_inputs(video: SearchVideo, stages_root: Path, court_root: Path) -> OldVideoInputs:
    """Read original shuttle/pose stages and the separately saved current courts."""
    shuttle = stages_root / 'shuttle' / video.identity
    return _load_inputs(
        video, shuttle / TRACK_FILENAME, shuttle / GUARD_CODES_FILENAME,
        stages_root / 'pose' / video.identity, court_root / video.identity,
    )


def load_test_inputs(
    video: SearchVideo, inpaint_root: Path, pose_root: Path, court_root: Path,
) -> OldVideoInputs:
    """Read the test tracks, poses and scene courts named by the benchmark."""
    if video.directory is None:
        raise ValueError(f'{video.identity}: test video has no source directory')
    track_dir = inpaint_root / video.directory
    return _load_inputs(
        video, track_dir / TEST_TRACK_FILENAME, track_dir / TEST_GUARD_CODES_FILENAME,
        pose_root / video.directory, court_root / video.identity,
    )


def run_video_kwargs(inputs: OldVideoInputs) -> dict[str, Any]:
    """Expose the fixed court and track arguments shared by all candidates."""
    court = inputs.court
    return {
        'fps': inputs.fps,
        'landing_options': SHIPPED_LANDING_FILTER_OPTIONS,
        'net_band': court.net_band,
        'resolution': court.resolution,
        'video_id': inputs.identity,
        'court_info': court.court_info,
        'gate_court_info': court.gate_court_info,
        'gate_resolution_table': court.gate_resolution_table,
        'court_present': inputs.court_present,
        'homography_rows': court.homography_rows,
        'cut_frames': [end for _start, end in inputs.raw_cuts[:-1]],
        'keep_vote': inputs.keep_vote,
        'inpaint_codes': inputs.guard_codes,
        'court_invalid_is_excluded': True,
        'landing_error_band_m': court.landing_error_band_m,
    }


def heuristic_stream(result: AnnotatorResult) -> dict[str, Any]:
    """Keep discovered spans and expand each rally's fitted final striker phase."""
    sequences = []
    events = []
    for rally_id, (start, end) in enumerate(result.spans):
        frames = result.filtered_by_rally.get(rally_id, [])
        final_half = result.striker_halves[rally_id]
        sides = [None] * len(frames) if final_half is None else alternating_player_sides(final_half, len(frames))
        own_events = tuple(ContactEvent(frame, 1.0, side) for frame, side in zip(frames, sides, strict=True))
        sequences.append(ContactSequence(rally_id, start, end, own_events))
        events.extend(own_events)
    return stream_payload(sequences, events)


def run_configuration(
    inputs: OldVideoInputs, base: BaseAnnotatorConfig, *,
    serve_start: ServeStartConfig | None = None, models: AnnotatorModels | None = None,
) -> dict[str, Any]:
    """Run the complete annotator with discovered boundaries and optional fitted models."""
    capture = RunCapture()
    pose = inputs.pose
    result = run_video(
        inputs.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
        base=base, heuristic_only=models is None, models=models, serve_start=serve_start,
        capture=capture, **run_video_kwargs(inputs),
    )
    if models is None:
        return heuristic_stream(result)
    prediction = capture.hybrid
    if prediction is None:
        raise ValueError(f'{inputs.identity}: fitted annotation did not return its contact stream')
    return stream_payload(prediction.refined.sequences, prediction.refined.events)


def score_stream(video: SearchVideo, stream: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Count full-video contact matches and whole rallies using discovered spans."""
    scores = {}
    for tolerance in (5, 10):
        _proposals, contacts, predictions, rallies = score_video(
            video.identity, video.split, stream, video.rallies, video.fps, tolerance,
        )
        scores[str(tolerance)] = {
            'labelled_rallies': len(rallies),
            'labelled_contacts': len(contacts),
            'predicted_contacts': len(predictions),
            'matched_contacts': sum(bool(row['matched']) for row in contacts),
            'correct_side_contacts': sum(bool(row['player_correct']) for row in contacts),
            'fully_correct_rallies': sum(bool(row['fully_correct']) for row in rallies),
        }
    return scores


def add_rates(counts: dict[str, int]) -> dict[str, int | float]:
    """Calculate pooled rates after callers sum the integer counts."""
    matched = counts['matched_contacts']
    predicted = counts['predicted_contacts']
    labelled = counts['labelled_contacts']
    attributed = counts['correct_side_contacts']
    rallies = counts['labelled_rallies']
    return {
        **counts,
        'precision': matched / predicted if predicted else 0.0,
        'recall': matched / labelled if labelled else 0.0,
        'f1': 2 * matched / (predicted + labelled) if predicted + labelled else 0.0,
        'attributed_precision': attributed / predicted if predicted else 0.0,
        'attributed_recall': attributed / labelled if labelled else 0.0,
        'attributed_f1': 2 * attributed / (predicted + labelled) if predicted + labelled else 0.0,
        'fully_correct_rallies_percent': 100 * counts['fully_correct_rallies'] / rallies if rallies else 0.0,
    }
