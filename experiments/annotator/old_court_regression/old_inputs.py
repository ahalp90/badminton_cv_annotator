"""Read the saved old-court vision files, labels and PR149 outputs for the regression run.

The current court loaders accept only court-evidence/0.2. These videos have
court-evidence/0.1, whose operational fields match 0.2 but whose per-scene
provenance records do not. This adapter reads the operational fields with the
existing parsers and passes them to ``run_video`` exactly as
``load_contact_evidence`` and ``run_full_annotation_stage`` do. The old
provenance records are ignored rather than converted.
"""

import csv
import gzip
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.courts.evidence import CourtInputs
from annotator.models import AnnotatorModels
from annotator.outcomes.point_winner import SHIPPED_LANDING_FILTER_OPTIONS, Half
from annotator.run_video import AnnotatorResult, RunCapture, run_video
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.shuttle_track import validate_shuttle_track
from annotator.training.sequences import LabelledRally
from annotator.training.workflow import DEFAULT_TRAINING_SETTINGS
from dataset_builder.shuttle_evidence import GUARD_CODES_FILENAME
from dataset_builder.vision import (
    COURT_EVIDENCE_FILENAME,
    COURT_KEEP_VOTE_FILENAME,
    COURT_PRESENT_FILENAME,
    TRACK_FILENAME,
    PoseArrays,
    _court_inputs_from_payload,
    _raw_cuts_from_payload,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
)

OLD_COURT_SCHEMA = 'court-evidence/0.1'
TEST_TRACK_FILENAME = 'shuttle_track_inpainted.npy.xz'
TEST_GUARD_CODES_FILENAME = 'shuttle_guard_codes_inpainted.npy.xz'
TEST_FPS = 30.0
# The original ShuttleSet22 run used one fixed clock and frame size for all 47 videos.
TEST_RESOLUTION = (1920.0, 1080.0)
SIDE_BY_LABEL = {'Top': Half.TOP, 'Bot': Half.BOT, 'Bottom': Half.BOT}
ALL_SOURCE_RALLY_COUNT = 3965
ALL_SOURCE_CONTACT_COUNT = 43159


@dataclass(frozen=True)
class OldVideoInputs:
    """One video's saved shuttle, pose and operational court inputs."""

    identity: str
    fps: float
    track: np.ndarray
    guard_codes: np.ndarray
    pose: PoseArrays
    raw_cuts: tuple[tuple[int, int], ...]
    court: CourtInputs
    keep_vote: np.ndarray
    court_present: np.ndarray


def frame_mask(path: Path, frame_count: int) -> np.ndarray:
    """Load one boolean array with one value per video frame."""
    values = load_npy_xz(path)
    if values.shape != (frame_count,) or values.dtype != np.bool_:
        raise ValueError(f'{path}: expected {frame_count} boolean values, got {values.dtype} {values.shape}')
    return values


def load_old_inputs(
    identity: str, fps: float, track_path: Path, guard_codes_path: Path, pose_dir: Path, court_dir: Path,
) -> OldVideoInputs:
    """Load and check one video's old files without reading any labels."""
    track = load_npy_xz(track_path)
    frame_count = len(track)
    validate_shuttle_track(track, frame_count)
    guard_codes = load_npy_xz(guard_codes_path)
    if guard_codes.shape != (frame_count,):
        raise ValueError(f'{identity}: guard codes cover {guard_codes.shape}, track has {frame_count} frames')
    pose = load_pose_arrays(pose_dir, frame_count)
    payload = load_json_gz(court_dir / COURT_EVIDENCE_FILENAME)
    if payload.get('schema') != OLD_COURT_SCHEMA:
        raise ValueError(f'{identity}: expected {OLD_COURT_SCHEMA}, found {payload.get("schema")!r}')
    if payload.get('video_id') != identity:
        raise ValueError(f'{identity}: court evidence names video {payload.get("video_id")!r}')
    raw_cuts = _raw_cuts_from_payload(payload['raw_cuts'])
    # run_video derives scene cut frames from these intervals, so they must tile the video.
    starts = [start for start, _end in raw_cuts]
    ends = [end for _start, end in raw_cuts]
    if starts[0] != 0 or ends[-1] != frame_count or starts[1:] != ends[:-1]:
        raise ValueError(f'{identity}: raw court cuts do not tile the {frame_count}-frame video')
    court = _court_inputs_from_payload(payload['inputs'])
    return OldVideoInputs(
        identity=identity, fps=fps, track=track, guard_codes=guard_codes, pose=pose, raw_cuts=raw_cuts,
        court=court, keep_vote=frame_mask(court_dir / COURT_KEEP_VOTE_FILENAME, frame_count),
        court_present=frame_mask(court_dir / COURT_PRESENT_FILENAME, frame_count),
    )


def load_development_inputs(stage_root: Path, identity: str, fps: float) -> OldVideoInputs:
    """Read one development video from the old dataset-builder stages layout."""
    stages = stage_root / 'stages'
    shuttle = stages / 'shuttle' / identity
    return load_old_inputs(
        identity, fps, shuttle / TRACK_FILENAME, shuttle / GUARD_CODES_FILENAME,
        stages / 'pose' / identity, stages / 'court' / identity,
    )


def test_video_directory(inpaint_root: Path, video_id: int) -> Path:
    """Find the one ``NN <video name>`` directory written by the inpaint run."""
    matches = [path for path in inpaint_root.glob(f'{video_id:02d} *') if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(f'expected one directory for ShuttleSet22 video {video_id} in {inpaint_root}, found {matches}')
    return matches[0]


def load_test_inputs(inpaint_root: Path, video_id: int) -> OldVideoInputs:
    """Read one ShuttleSet22 video with the inpainted track used by PR149."""
    directory = test_video_directory(inpaint_root, video_id)
    inputs = load_old_inputs(
        str(video_id), TEST_FPS, directory / TEST_TRACK_FILENAME, directory / TEST_GUARD_CODES_FILENAME,
        directory, directory,
    )
    if tuple(inputs.court.resolution) != TEST_RESOLUTION:
        raise ValueError(f'{video_id}: court resolution {inputs.court.resolution} differs from {TEST_RESOLUTION}')
    return inputs


def run_old_video(
    inputs: OldVideoInputs, models: AnnotatorModels | None, capture: RunCapture,
) -> AnnotatorResult:
    """Run the heuristic pass (no models) or full annotation on old inputs.

    Arguments mirror ``load_contact_evidence`` for the heuristic pass and
    ``run_full_annotation_stage`` for full annotation.
    """
    court = inputs.court
    return run_video(
        inputs.track, inputs.pose.bboxes, inputs.pose.scores, inputs.pose.kps, inputs.pose.ndet,
        fps=inputs.fps,
        base=DEFAULT_TRAINING_SETTINGS.preprocessing if models is None else models.preprocessing,
        models=models, heuristic_only=models is None,
        landing_options=SHIPPED_LANDING_FILTER_OPTIONS,
        net_band=court.net_band, resolution=court.resolution, video_id=inputs.identity,
        court_info=court.court_info, homo_df=None,
        gate_court_info=court.gate_court_info, gate_resolution_table=court.gate_resolution_table,
        court_present=inputs.court_present, homography_rows=court.homography_rows,
        cut_frames=[end for _start, end in inputs.raw_cuts[:-1]], keep_vote=inputs.keep_vote,
        inpaint_codes=inputs.guard_codes, court_invalid_is_excluded=True,
        landing_error_band_m=court.landing_error_band_m, capture=capture,
    )


def write_label_csv(path: Path, rallies: Sequence[tuple[str, Sequence[tuple[int, Half | None]]]]) -> None:
    """Write the supported rally_id,frame,side contract; empty side means unknown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('rally_id', 'frame', 'side'))
        for rally_id, contacts in rallies:
            for frame, side in contacts:
                writer.writerow((rally_id, frame, '' if side is None else side.value))
    temporary.replace(path)


def development_label_rows(shots: pd.DataFrame, video_id: int) -> list[tuple[str, list[tuple[int, Half | None]]]]:
    """Read one ShuttleSet video's contacts from shots_master.csv as the original fits did.

    Rallies are keyed by set and rally number; frames are sorted within each
    rally. Every development contact has a Top or Bottom player side.
    """
    rows = shots[shots['vid'] == video_id]
    if rows.empty:
        raise ValueError(f'shots_master.csv has no contacts for video {video_id}')
    rallies = []
    for (set_id, rally_number), group in sorted(rows.groupby(['set_id', 'rally']), key=lambda item: item[0]):
        contacts = sorted(
            ((int(frame), SIDE_BY_LABEL[side])
             for frame, side in zip(group['frame_num'], group['player_side'], strict=True)),
            key=lambda contact: contact[0],
        )
        rallies.append((f'{set_id}:{int(rally_number)}', contacts))
    return rallies


def trusted_test_label_rows(clean_labels: Path) -> dict[int, list[tuple[str, list[tuple[int, Half | None]]]]]:
    """Read the 3,422 trusted ShuttleSet22 rallies saved by the PR149 scorer."""
    payload = json.loads(gzip.decompress(clean_labels.read_bytes()))
    if payload.get('schema') != 'shuttleset22-clean-contact-labels/1' or payload.get('status') != 'complete':
        raise ValueError(f'{clean_labels}: unexpected clean-label schema or status')
    by_video = {}
    for video in payload['videos']:
        by_video[int(video['video_id'])] = [
            (f'{rally["set_id"]}:{rally["rally"]}',
             [(int(contact['frame']), None if contact['side'] is None else SIDE_BY_LABEL[contact['side']])
              for contact in rally['contacts']])
            for rally in video['rallies']
        ]
    return by_video


def source_player_side(player_y: object, opponent_y: object) -> Half | None:
    """Apply the PR149 rule: the player nearer the top of the image is Top."""
    player = pd.to_numeric(pd.Series([player_y]), errors='coerce').iloc[0]
    opponent = pd.to_numeric(pd.Series([opponent_y]), errors='coerce').iloc[0]
    if not np.isfinite(player) or not np.isfinite(opponent) or player == opponent:
        return None
    return Half.TOP if player < opponent else Half.BOT


def all_source_rallies(
    annotation_root: Path, video_ids: Sequence[int], trusted: dict[int, tuple[LabelledRally, ...]],
) -> dict[int, tuple[LabelledRally, ...]]:
    """Read every source rally, including those removed by the trusted cleaning rule.

    This mirrors PR149's all-GT read. Duplicate timestamps stay. The supported
    CSV contract rejects them, so these rallies stay in memory. A rally with
    out-of-order source frames is sorted by frame, each frame keeping its side.
    """
    names = pd.read_csv(annotation_root / 'set' / 'match.csv').set_index('id')['video'].to_dict()
    by_video = {}
    for video_id in video_ids:
        rallies = []
        for path in sorted((annotation_root / 'set' / names[video_id]).glob('set*.csv')):
            for rally_number, group in pd.read_csv(path).groupby('rally', sort=True):
                ordered = group.sort_values(['ball_round', 'frame_num'])
                contacts = sorted(
                    ((int(row['frame_num']), source_player_side(row['player_location_y'], row['opponent_location_y']))
                     for _, row in ordered.iterrows()),
                    key=lambda contact: contact[0],
                )
                rallies.append(LabelledRally(
                    f'{path.stem}:{int(rally_number)}',
                    tuple(frame for frame, _side in contacts), tuple(side for _frame, side in contacts),
                ))
        by_id = {rally.identity: rally for rally in rallies}
        for rally in trusted[video_id]:
            if by_id[rally.identity] != rally:
                raise ValueError(f'{video_id}/{rally.identity}: all-source rally differs from its trusted copy')
        by_video[video_id] = tuple(rallies)
    rally_count = sum(map(len, by_video.values()))
    contact_count = sum(len(rally.frames) for rallies in by_video.values() for rally in rallies)
    if (rally_count, contact_count) != (ALL_SOURCE_RALLY_COUNT, ALL_SOURCE_CONTACT_COUNT):
        raise ValueError(f'all-source labels have {rally_count} rallies and {contact_count} contacts; '
                         f'PR149 read {ALL_SOURCE_RALLY_COUNT} and {ALL_SOURCE_CONTACT_COUNT}')
    return by_video


def stream_from_output(
    output: dict, fixture: str, bounds: dict[int, tuple[int, int]] | None = None,
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Restore one video's saved PR149 sections and full contact stream.

    Saved sides are the raw per-contact guesses before whole-rally alternation;
    the scorer applies the alternation itself. ``bounds`` replaces section
    start/end frames when the saved boundary correction is stored separately.
    """
    events = tuple(
        ContactEvent(int(frame), float(score), None if side is None else Half(side))
        for frame, score, side in output['contacts'][fixture]
    )
    by_frame = {event.frame: event for event in events}
    if len(by_frame) != len(events):
        raise ValueError(f'{fixture}: saved contact stream repeats a frame')
    sequences = []
    for span in output['spans']:
        if span['fixture'] != fixture:
            continue
        start, end = (span['start_frame'], span['end_frame']) if bounds is None else bounds[span['span_id']]
        members = tuple(by_frame[int(frame)] for frame in span['frames'])
        if not all(start <= event.frame < end for event in members):
            raise ValueError(f'{fixture}/{span["span_id"]}: saved contacts lie outside their section')
        sequences.append(ContactSequence(int(span['span_id']), int(start), int(end), members))
    return tuple(sequences), events
