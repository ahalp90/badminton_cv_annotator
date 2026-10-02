"""Detect courts from one video or a batch of videos, using live or saved lines and people.

A batch loads DeepLSD, RTMLib and the detector once for every video in its manifest.
README.md owns the options and the output format.
"""

from __future__ import annotations

# ruff: noqa: E402 -- Set worker thread limits before importing NumPy.

import argparse
import gzip
import json
import logging
import lzma
import os
import sys
import traceback
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

# Process workers inherit these settings. Set them before importing NumPy.
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from . import feet
from .detect import CourtDetector, CourtFitError, SceneCourts, Switches
from .inputs import FrameReader, PeopleSource, ViewInputs, same_frame_provenance
from .line_sources import DeepLSDLines, LineSource, SavedLines
from .scene_sources import PySceneDetectSource, SavedScenes, SceneInfo, SceneSource
from .template_arrays import TEMPLATE_DEVICES
from .video_inputs import PoseArrays, RtmlibPeople, VideoFrames
from .view_pool import CourtMode, VideoPool

if TYPE_CHECKING:
    from shared.rtmlib_pose import RtmlibPoseExtractor

VIDEO_RESULT_SCHEMA = 'court-detector-video/1'
BATCH_SUMMARY_SCHEMA = 'court-detector-batch/1'
MANIFEST_KEYS = {'id', 'video', 'people', 'scenes'}


class PosePrerunError(RuntimeError):
    """One video's isolated pose process or metadata probe failed."""


# A batch video that fails with these had bad files of its own, such as an unreadable
# video or scenes that miss frames. The shared models and workers are unaffected.
VIDEO_INPUT_ERRORS = (OSError, ValueError, EOFError, lzma.LZMAError, PosePrerunError)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CourtTools:
    """Models loaded once and shared by every video in a run.

    None of them holds per-video state. Each video opens its own frames, people
    source and list of known courts.
    """

    lines: LineSource
    saved_lines: bool  # lines come from a saved extract, so the timings exclude DeepLSD
    detector: CourtDetector
    pose_extractor: RtmlibPoseExtractor | None  # None when every video has saved people, or people are optional
    load_seconds: float


@dataclass(frozen=True)
class BatchVideo:
    """One manifest entry. Relative paths resolve from the working directory, as on the command line."""

    video_id: str  # names the result file and starts each scene's view_id
    video: Path
    people: Path | None  # saved pose directory; None means live RTMLib, or no people when they are optional
    scenes: Path | None  # saved scene ranges; None means --pyscenedetect or the whole video


@dataclass(frozen=True)
class PosePrerun:
    """Where to save full-video poses and which environment runs their extraction."""

    output_dir: Path
    interpreter: str
    device: str


def prerun_people(video: Path, video_id: str, settings: PosePrerun) -> Path:
    """Use the dataset builder's eight-shard, ten-person pose settings."""
    from annotator.video_metadata import probe_video_metadata
    from dataset_builder.pose_sharding import extract_sharded_rtmlib_pose_stage

    output = settings.output_dir / video_id
    # An existing folder may belong to different footage. Reuse it explicitly via --people.
    output.mkdir(parents=True, exist_ok=False)
    try:
        extract_sharded_rtmlib_pose_stage(
            metadata=probe_video_metadata(video), output_dir=output, interpreter=settings.interpreter,
            shards=8, n_max=10, device=settings.device,
        )
    except RuntimeError as error:
        # These subprocesses cannot leave partial pose state in the court models.
        raise PosePrerunError(f'{video_id}: pose prerun failed: {error}') from error
    return output


def frame_view(view_id: str, frame: np.ndarray, frame_index: int, scene: SceneInfo, lines: LineSource,
               people: PeopleSource | None, alignment_image: np.ndarray | None = None) -> ViewInputs:
    """One scene frame with its own lines and person boxes."""
    segments = lines.segments(frame, frame_index)
    if people is None:
        boxes = np.empty((0, 4), dtype=float)
    else:
        samples = people.samples([frame_index])
        if len(samples) != 1 or samples[0].frame_index != frame_index:
            raise ValueError(f'{view_id}: people source did not return the requested frame {frame_index}')
        boxes = samples[0].boxes_px
    return ViewInputs(view_id, frame, frame_index, (scene.start_frame, scene.end_frame), segments,
                      boxes, same_frame_provenance(view_id, frame_index), alignment_image)


def endpoint_views(decoded: dict[int, np.ndarray], endpoints: Sequence[int], scene_id: str, scene: SceneInfo,
                   lines: LineSource, people: PeopleSource | None) -> list[ViewInputs]:
    """The endpoint frames' views, in order. The detector asks for them only when it composes."""
    return [frame_view(f'{scene_id}_frame_{index}', decoded[index], index, scene, lines, people)
            for index in endpoints]


def scene_courts(
    detector: CourtDetector, frames: FrameReader, people: PeopleSource | None, lines: LineSource,
    scenes: Sequence[SceneInfo], *, video_id: str, reuse_courts: bool = False, compose_scenes: bool = True,
    on_court: Callable[[dict[str, Any], SceneCourts], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Detect the middle frame of each scene without crossing a cut for foot samples.

    When people are required, a scene too short for the foot window is
    reported as unanalysed, not as evidence that no court is present. A scene
    whose search or fit fails is reported as `detection_failed` with its error,
    and the next scene runs. Each row repeats its scene's `[start_frame, end_frame)`
    bounds.

    When a fresh search finds the middle frame's court, the detector also searches
    the first and last frames of the foot window and may report a composite court
    in the middle frame's pixels. A scene too short for the window, or one that
    reuses an earlier court, keeps the middle frame alone. compose_scenes=False
    (fast-robust mode) skips the endpoint frames for every scene; the foot window
    still runs.

    on_court receives each row with a court and its scene's finished courts before the
    row is yielded. Video-robust and fast-robust modes pass VideoPool.add.
    """
    if people is None and detector.switches.require_people:
        raise ValueError('A people source is required when require_people is enabled')
    # Keep only fully searched courts: chaining refits would accumulate movement.
    known_views = []
    for scene_index, scene in enumerate(scenes):
        started = perf_counter()
        anchor = scene.middle_frame
        scene_id = f'{video_id}_scene_{scene_index:04d}'
        view_id = f'{scene_id}_frame_{anchor}'
        row: dict[str, Any] = {'view_id': view_id, 'start_frame': scene.start_frame, 'end_frame': scene.end_frame,
                               'frame_index': anchor}
        # Validate this input boundary before spending time on line/pose inference.
        try:
            window = feet.window_frames(anchor, frames.fps, scene.start_frame, scene.end_frame)
        except ValueError:
            if detector.switches.require_people:
                row.update(status='scene_too_short_for_feet', corners_native_px=None, seconds=perf_counter() - started)
                yield row
                continue
            window = None
        endpoints = [window[0], window[-1]] if window is not None and compose_scenes else []
        # The reuse median's frames: the window's ends when it fits, else the scene's.
        # The end is exclusive, so the scene's last frame is end_frame - 1.
        if window is None:
            alignment_frames = (scene.start_frame, anchor, scene.end_frame - 1)
        else:
            alignment_frames = (window[0], anchor, window[-1])
        if window is not None and people is not None and detector.switches.enforce_scene_consistency:
            frame_indices = window
        elif endpoints or reuse_courts:
            frame_indices = list(alignment_frames)
        else:
            frame_indices = [anchor]
        # Decode in order. The feet check can then use the cached window without
        # seeking backwards after the median's last sample.
        decoded = dict(zip(frame_indices, frames.read(frame_indices), strict=True))
        frame = decoded[anchor]
        alignment_image = None
        known_courts = []
        if reuse_courts:
            from .reuse import view_image

            # Moving players occupy different pixels across these samples. A median
            # retains the static court for alignment with returning camera views.
            images = [view_image(decoded[index]) for index in alignment_frames]
            alignment_image = np.median(images, axis=0).astype(np.uint8)
            alignment_image.flags.writeable = False
            # Histograms only order the attempts. Image alignment and court checks
            # decide reuse. Missing histograms leave the most recent views first.
            ordered = known_views
            if scene.histogram is not None:
                ordered = sorted(known_views, key=lambda known: float('inf') if known[1] is None
                                 else float(abs(scene.histogram - known[1]).sum()))
            known_courts = [known[0] for known in ordered[:3]]
        view = frame_view(view_id, frame, anchor, scene, lines, people, alignment_image)
        endpoint_inputs = None
        if endpoints:
            endpoint_inputs = partial(endpoint_views, decoded, endpoints, scene_id, scene, lines, people)
        try:
            result = detector.detect(view, people, frames, known_courts=known_courts, endpoint_views=endpoint_inputs)
        except CourtFitError as error:
            # Long-video rule: record this scene's failure and keep going. A failed
            # scene skips the reuse store below, so it never becomes a template.
            logger.exception('%s: court detection failed', view_id)
            row.update(status='detection_failed', corners_native_px=None, error=repr(error),
                       traceback=traceback.format_exc(), seconds=perf_counter() - started)
            yield row
            continue
        # A composite court is stored like a searched one: in the middle frame's pixels,
        # with its paint support measured there.
        if (reuse_courts and result.corners_native_px is not None and result.reused_from is None
                and result.paint_score is not None and result.paint_score > 0):
            from .reuse import make_known_court

            known = make_known_court(view_id, frame, result.corners_native_px, result.paint_score,
                                     alignment_image=alignment_image)
            known_views.insert(0, (known, scene.histogram))
            del known_views[8:]
        row.update(status='court' if result.corners_native_px is not None else 'no_court',
                   corners_native_px=None if result.corners_native_px is None else result.corners_native_px.tolist(),
                   chosen_key=result.chosen_key, no_court_reason=result.no_court_reason, reused_from=result.reused_from,
                   composition=result.composition, stage_seconds=result.stage_seconds,
                   seconds=perf_counter() - started)
        if on_court is not None and result.scene is not None:
            on_court(row, result.scene)
        yield row


def validate_scenes(scenes: Sequence[SceneInfo], frame_count: int) -> None:
    """External scene inputs must partition the source timeline `[0, frame_count)` exactly.

    Each scene must start where the previous one ended, hold at least one frame
    and end by `frame_count`.
    """
    next_frame = 0
    for scene in scenes:
        if scene.start_frame != next_frame or not scene.start_frame < scene.end_frame <= frame_count:
            raise ValueError(f'Scene [{scene.start_frame}, {scene.end_frame}) must start at frame {next_frame} '
                             f'and end after it, by frame {frame_count}')
        next_frame = scene.end_frame
    if next_frame != frame_count:
        raise ValueError(f'Scenes end at frame {next_frame}, expected {frame_count}')


def read_json(path: Path) -> Any:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    with gzip.open(path, 'wt') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_court_tools(switches: Switches, *, saved_lines: Path | None = None, deeplsd_source: Path | None = None,
                     deeplsd_weights: Path | None = None, device: str = 'cuda', live_pose: bool = False) -> CourtTools:
    """Load the line source, detector and optional live pose models once for a run.

    :param saved_lines: one video's saved line extracts, keyed by its frame numbers.
        Without them, DeepLSD loads from `deeplsd_source` and `deeplsd_weights`.
    :param device: torch or ONNX device for DeepLSD and RTMLib.
    :param live_pose: load RTMLib for videos without saved people.
    """
    started = perf_counter()
    lines: LineSource
    if saved_lines is not None:
        lines = SavedLines({int(index): segments for index, segments in read_json(saved_lines).items()})
    elif deeplsd_source is not None and deeplsd_weights is not None:
        lines = DeepLSDLines(deeplsd_source, deeplsd_weights, device=device)
    else:
        raise ValueError('Provide saved lines or both the DeepLSD source and weights')
    pose_extractor = None
    if live_pose:
        from shared.rtmlib_pose import RtmlibPoseExtractor

        pose_extractor = RtmlibPoseExtractor(device=device)
    return CourtTools(lines, saved_lines is not None, CourtDetector(switches), pose_extractor,
                      perf_counter() - started)


def scene_source_for(scenes: Path | None, pyscenedetect: bool) -> SceneSource | None:
    """A saved scene file, a PySceneDetect pass with histograms, or None for the whole video."""
    if scenes is not None:
        return SavedScenes(scenes)
    if pyscenedetect:
        return PySceneDetectSource(histograms=True)
    return None


def detect_video(video: Path, tools: CourtTools, *, video_id: str, people_dir: Path | None,
                 scene_source: SceneSource | None, reuse_courts: bool,
                 pose_prerun: PosePrerun | None = None,
                 court_mode: CourtMode = CourtMode.VIDEO_ROBUST) -> dict[str, Any]:
    """Detect one court per scene of one video with the shared tools; return the video result.

    Open `tools.detector` as a context manager around one or many calls, so every
    scene shares one worker pool. The frames, people source and known courts
    belong to this call alone.

    :param video_id: starts each scene's view_id.
    :param people_dir: saved native-pixel pose arrays. Without them, people come from
        `tools.pose_extractor` when it is loaded, and are absent otherwise.
    :param scene_source: where the cuts come from; None analyses the whole video as one scene.
    :param reuse_courts: try this video's earlier fully searched courts before a full search.
    :param court_mode: video-robust pools scene courts across returning camera views
        (view_pool.py) after the last scene, then prints the rows. fast-robust does the
        same with one fresh middle-frame search per scene, so it cannot reuse courts.
    :return: the `VIDEO_RESULT_SCHEMA` result that README.md describes.
    """
    if court_mode == CourtMode.FAST_ROBUST and reuse_courts:
        raise ValueError('fast-robust fits every scene afresh, so it cannot reuse courts')
    started = perf_counter()
    switches = tools.detector.switches
    pose_prerun_seconds = 0.0
    if pose_prerun is not None and people_dir is None:
        people_dir = prerun_people(video, video_id, pose_prerun)
        pose_prerun_seconds = perf_counter() - started
    with VideoFrames(video) as frames:
        people: PeopleSource | None = None
        if people_dir is not None:
            poses = PoseArrays.from_directory(people_dir)
            if poses.frame_count < frames.frame_count:
                raise ValueError(f'{video_id}: saved poses do not cover the source video')
            people = poses
        elif tools.pose_extractor is not None:
            people = RtmlibPeople(frames, tools.pose_extractor)
        setup_seconds = perf_counter() - started
        scene_started = perf_counter()
        if scene_source is None:
            scenes = [SceneInfo(0, frames.frame_count)]
        else:
            scenes = scene_source.scenes(video, frames.frame_count, frames.fps)
        validate_scenes(scenes, frames.frame_count)
        scene_seconds = perf_counter() - scene_started
        processing_started = perf_counter()
        rows = []
        pool = None if court_mode == CourtMode.SCENE_ROBUST else VideoPool(tools.detector.live, switches, court_mode)
        for row in scene_courts(tools.detector, frames, people, tools.lines, scenes, video_id=video_id,
                                reuse_courts=reuse_courts, compose_scenes=court_mode != CourtMode.FAST_ROBUST,
                                on_court=None if pool is None else pool.add):
            rows.append(row)
            logger.info('%s: scene %d/%d %s', video_id, len(rows), len(scenes), row['status'])
            if pool is None:
                print(json.dumps(row), flush=True)
        extra: dict[str, Any] = {}
        if pool is not None:
            # Pooling can change earlier rows, so they print only once every court is final.
            logger.info('%s: pooling courts across scenes', video_id)
            extra['view_groups'] = pool.apply()
            logger.info('%s: pooling finished with %d view groups', video_id, len(extra['view_groups']))
            for row in rows:
                print(json.dumps(row), flush=True)
        return {'schema': VIDEO_RESULT_SCHEMA, 'video_id': video_id, 'video': video.name, 'fps': frames.fps,
                'frame_count': frames.frame_count, 'native_size': frames.size, 'tools_seconds': tools.load_seconds,
                'setup_seconds': setup_seconds, 'scene_seconds': scene_seconds,
                'pose_prerun_seconds': pose_prerun_seconds,
                'processing_seconds': perf_counter() - processing_started, 'total_seconds': perf_counter() - started,
                'saved_people': people_dir is not None, 'require_people': switches.require_people,
                'reuse_courts': reuse_courts, 'court_mode': str(court_mode),
                'template_device': switches.template_device, 'saved_lines': tools.saved_lines, 'scenes': rows,
                **extra}


def read_manifest(path: Path) -> list[BatchVideo]:
    """The batch's videos from a gzipped JSON list of objects; README.md describes the keys."""
    videos = []
    for entry in read_json(path):
        unknown_keys = entry.keys() - MANIFEST_KEYS
        if unknown_keys or 'id' not in entry or 'video' not in entry:
            raise ValueError(f'Manifest entry {entry} needs "id" and "video", and allows only {sorted(MANIFEST_KEYS)}')
        video_id = entry['id']
        # The ID becomes a file name in the output directory.
        if not isinstance(video_id, str) or not video_id or Path(video_id).name != video_id:
            raise ValueError(f'Manifest ID {video_id!r} must be a plain file name')
        people = Path(entry['people']) if 'people' in entry else None
        scenes = Path(entry['scenes']) if 'scenes' in entry else None
        videos.append(BatchVideo(video_id, Path(entry['video']), people, scenes))
    if not videos:
        raise ValueError(f'{path} lists no videos')
    video_ids = [video.video_id for video in videos]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError('Manifest IDs must be unique')
    return videos


def run_batch(videos: Sequence[BatchVideo], tools: CourtTools, output_dir: Path, *,
              pyscenedetect: bool, reuse_courts: bool,
              pose_prerun: PosePrerun | None = None,
              court_mode: CourtMode = CourtMode.VIDEO_ROBUST) -> dict[str, Any]:
    """Detect every manifest video with one set of models; return the batch summary.

    Write each finished video's result to `output_dir/videos/<id>.json.gz`, and the
    summary to `output_dir/summary.json.gz` after every video. A video that fails
    on its own files fails alone. A broken worker pool is replaced before the next
    video. Any other failure stops the batch, because the shared models, GPU or
    workers may be left in an unknown state; the later videos stay `not_run`.
    """
    started = perf_counter()
    results_dir = output_dir / 'videos'
    results_dir.mkdir(parents=True)
    outcomes = [{'id': video.video_id, 'video': str(video.video), 'status': 'not_run'} for video in videos]
    summary = {'schema': BATCH_SUMMARY_SCHEMA, 'finished': False, 'stopped_after': None,
               'tools_seconds': tools.load_seconds, 'require_people': tools.detector.switches.require_people,
               'reuse_courts': reuse_courts, 'court_mode': str(court_mode), 'videos': outcomes}
    with ExitStack() as workers:
        workers.enter_context(tools.detector)
        for video, outcome in zip(videos, outcomes, strict=True):
            video_started = perf_counter()
            try:
                result = detect_video(video.video, tools, video_id=video.video_id, people_dir=video.people,
                                      scene_source=scene_source_for(video.scenes, pyscenedetect),
                                      reuse_courts=reuse_courts, pose_prerun=pose_prerun, court_mode=court_mode)
                write_json(results_dir / f'{video.video_id}.json.gz', result)
            except Exception as error:
                logger.exception('%s: video failed', video.video_id)
                outcome.update(status='failed', error=repr(error), traceback=traceback.format_exc(),
                               seconds=perf_counter() - video_started)
                if isinstance(error, BrokenProcessPool):
                    # A dead worker makes its whole pool unusable for later videos.
                    workers.close()
                    workers.enter_context(tools.detector)
                elif not isinstance(error, VIDEO_INPUT_ERRORS):
                    summary['stopped_after'] = video.video_id
                    break
            else:
                outcome.update(status='complete', output=f'videos/{video.video_id}.json.gz',
                               seconds=perf_counter() - video_started,
                               scene_statuses=dict(Counter(row['status'] for row in result['scenes'])))
            summary['seconds'] = perf_counter() - started
            write_json(output_dir / 'summary.json.gz', summary)
    summary['finished'] = summary['stopped_after'] is None
    summary['seconds'] = perf_counter() - started
    write_json(output_dir / 'summary.json.gz', summary)
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path, help='one video; its result goes to --output')
    source.add_argument('--manifest', type=Path,
                        help='.json.gz list of {"id", "video", optional "people", optional "scenes"} objects; '
                             'results go to --output-dir')
    parser.add_argument('--output', type=Path, help='output .json.gz file for --video')
    parser.add_argument('--output-dir', type=Path, help='new directory for --manifest results')
    people_options = parser.add_mutually_exclusive_group()
    people_options.add_argument('--people', type=Path,
                                help='native-pixel pose_{bboxes,kps,ndet}.npy.xz directory; otherwise run RTMLib')
    people_options.add_argument('--pose-prerun', type=Path,
                                help='extract full-video poses into DIR/VIDEO_ID before detection; '
                                     'uses eight shards and keeps ten people per frame')
    parser.add_argument('--pose-python', default=sys.executable,
                        help='Python executable for --pose-prerun (default: this interpreter)')
    parser.add_argument('--require-people', action=argparse.BooleanOptionalAction, default=True,
                        help='require person boxes and keypoints for court detection (default: on)')
    parser.add_argument('--saved-lines', type=Path, help='.json.gz object mapping source frame numbers to line arrays')
    parser.add_argument('--deeplsd-source', type=Path)
    parser.add_argument('--deeplsd-weights', type=Path)
    parser.add_argument('--device', default='cuda', help='device for live DeepLSD and RTMLib')
    parser.add_argument('--template-device', choices=TEMPLATE_DEVICES, default='cpu',
                        help='device for line-template scoring; cuda needs CuPy and a GPU (default: cpu)')
    scene_options = parser.add_mutually_exclusive_group()
    scene_options.add_argument('--scenes', type=Path,
                               help='.json.gz list of [start_frame, end_frame] scene ranges; end_frame is exclusive')
    scene_options.add_argument('--pyscenedetect', action='store_true', help='detect cuts and representative scene histograms')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--full-score-limit', type=int, help='optional cheap-score trial limit; omit for exhaustive scoring')
    parser.add_argument('--reuse-courts', action='store_true', help='trial checked reuse of earlier camera views')
    parser.add_argument('--court-mode', type=CourtMode, choices=list(CourtMode), default=CourtMode.VIDEO_ROBUST,
                        help='scene-robust keeps each scene\'s court; video-robust may share one pooled court '
                             'across scenes of the same camera view; fast-robust pools like video-robust but '
                             'searches only each scene\'s middle frame (default: video-robust)')
    args = parser.parse_args()
    if args.pose_prerun is not None and not args.require_people:
        parser.error('--pose-prerun requires --require-people')
    if args.court_mode == CourtMode.FAST_ROBUST and args.reuse_courts:
        parser.error('--court-mode fast-robust fits every scene afresh; it cannot combine with --reuse-courts')
    if args.video is not None and (args.output is None or args.output_dir is not None):
        parser.error('--video writes one file: give --output, not --output-dir')
    if args.manifest is not None:
        if args.output_dir is None or args.output is not None:
            parser.error('--manifest writes a directory: give --output-dir, not --output')
        # These inputs belong to one video's frame numbers; a manifest names people and scenes per video.
        for option in ('people', 'scenes', 'saved_lines'):
            if getattr(args, option) is not None:
                parser.error(f'--{option.replace("_", "-")} applies to --video only')
    if args.saved_lines is None and (args.deeplsd_source is None or args.deeplsd_weights is None):
        parser.error('provide --saved-lines or both --deeplsd-source and --deeplsd-weights')
    return args


def main() -> int:
    # Send scene and pooling progress to stderr alongside any existing diagnostics.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    args = parse_arguments()
    videos = [] if args.manifest is None else read_manifest(args.manifest)
    if args.pyscenedetect and any(video.scenes is not None for video in videos):
        raise ValueError('--pyscenedetect cannot combine with manifest "scenes" files')
    if args.manifest is not None:
        # Fail before loading any model, rather than mix results with an earlier batch.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        people_dirs = [video.people for video in videos]
    else:
        people_dirs = [args.people]
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    switches = Switches(workers=args.workers, timing=True, full_score_limit=args.full_score_limit,
                        require_people=args.require_people, template_device=args.template_device)
    pose_prerun = (PosePrerun(args.pose_prerun, args.pose_python, args.device)
                   if args.pose_prerun is not None else None)
    live_pose = args.require_people and pose_prerun is None and any(people is None for people in people_dirs)
    tools = load_court_tools(switches, saved_lines=args.saved_lines, deeplsd_source=args.deeplsd_source,
                             deeplsd_weights=args.deeplsd_weights, device=args.device, live_pose=live_pose)
    if args.manifest is not None:
        summary = run_batch(videos, tools, args.output_dir, pyscenedetect=args.pyscenedetect,
                            reuse_courts=args.reuse_courts, pose_prerun=pose_prerun, court_mode=args.court_mode)
        return 0 if all(outcome['status'] == 'complete' for outcome in summary['videos']) else 1
    # Every scene's search and scoring share one set of worker processes.
    with tools.detector:
        result = detect_video(args.video, tools, video_id=args.video.stem, people_dir=args.people,
                              scene_source=scene_source_for(args.scenes, args.pyscenedetect),
                              reuse_courts=args.reuse_courts, pose_prerun=pose_prerun,
                              court_mode=args.court_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
