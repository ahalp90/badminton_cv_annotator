"""Detect courts from a video, using live or saved lines and people."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

# Process workers inherit these settings. Set them before importing NumPy.
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from . import feet
from .detect import CourtDetector, Switches
from .inputs import FrameReader, PeopleSource, ViewInputs, same_frame_provenance
from .line_sources import DeepLSDLines, LineSource, SavedLines
from .scene_sources import PySceneDetectSource, SceneInfo
from .template_arrays import TEMPLATE_DEVICES
from .video_inputs import PoseArrays, RtmlibPeople, VideoFrames


def scene_courts(
    detector: CourtDetector, frames: FrameReader, people: PeopleSource | None, lines: LineSource,
    scenes: Sequence[SceneInfo], *, video_id: str, reuse_courts: bool = False,
) -> Iterator[dict[str, Any]]:
    """Detect the middle frame of each scene without crossing a cut for foot samples.

    When people are required, a scene too short for the foot window is
    reported as unanalysed, not as evidence that no court is present. Each row
    repeats its scene's `[start_frame, end_frame)` bounds.
    """
    if people is None and detector.switches.require_people:
        raise ValueError('A people source is required when require_people is enabled')
    # Keep only fully searched courts: chaining refits would accumulate movement.
    known_views = []
    for scene_index, scene in enumerate(scenes):
        started = perf_counter()
        anchor = scene.middle_frame
        view_id = f'{video_id}_scene_{scene_index:04d}_frame_{anchor}'
        row: dict[str, Any] = {'view_id': view_id, 'start_frame': scene.start_frame, 'end_frame': scene.end_frame,
                               'frame_index': anchor}
        # The scene's first, middle and last frames. The end is exclusive, so the last is end_frame - 1.
        alignment_frames = (scene.start_frame, anchor, scene.end_frame - 1)
        frame_indices = list(alignment_frames) if reuse_courts else [anchor]
        if detector.switches.require_people or reuse_courts:
            # Validate this input boundary before spending time on line/pose inference.
            try:
                window = feet.window_frames(anchor, frames.fps, scene.start_frame, scene.end_frame)
                alignment_frames = (window[0], anchor, window[-1])
                if reuse_courts:
                    if people is not None and detector.switches.enforce_scene_consistency:
                        frame_indices = window
                    else:
                        frame_indices = list(alignment_frames)
            except ValueError:
                if detector.switches.require_people:
                    row.update(status='scene_too_short_for_feet', corners_native_px=None, seconds=perf_counter() - started)
                    yield row
                    continue
        # Decode in order. The feet check can then use the cached window without
        # seeking backwards after the median's last sample.
        decoded = frames.read(frame_indices)
        frame = decoded[frame_indices.index(anchor)]
        segments = lines.segments(frame, anchor)
        if people is None:
            boxes = np.empty((0, 4), dtype=float)
        else:
            anchor_people = people.samples([anchor])
            if len(anchor_people) != 1 or anchor_people[0].frame_index != anchor:
                raise ValueError(f'{view_id}: people source did not return the requested anchor')
            boxes = anchor_people[0].boxes_px
        alignment_image = None
        if reuse_courts:
            from .reuse import make_known_court, view_image

            # Moving players occupy different pixels across these samples. A median
            # retains the static court for alignment with returning camera views.
            images = [view_image(decoded[frame_indices.index(index)]) for index in alignment_frames]
            alignment_image = np.median(images, axis=0).astype(np.uint8)
            alignment_image.flags.writeable = False
        view = ViewInputs(view_id, frame, anchor, (scene.start_frame, scene.end_frame), segments,
                          boxes, same_frame_provenance(view_id, anchor), alignment_image)
        if reuse_courts:
            # Histograms only order the attempts. Image alignment and court checks
            # decide reuse. Missing histograms leave the most recent views first.
            ordered = known_views
            if scene.histogram is not None:
                ordered = sorted(known_views, key=lambda known: float('inf') if known[1] is None
                                 else float(abs(scene.histogram - known[1]).sum()))
            result = detector.detect(view, people, frames, known_courts=[known[0] for known in ordered[:3]])
            if (result.corners_native_px is not None and result.reused_from is None
                    and result.paint_score is not None and result.paint_score > 0):
                known = make_known_court(view_id, frame, result.corners_native_px, result.paint_score,
                                        alignment_image=alignment_image)
                known_views.insert(0, (known, scene.histogram))
                del known_views[8:]
        else:
            result = detector.detect(view, people, frames)
        row.update(status='court' if result.corners_native_px is not None else 'no_court',
                   corners_native_px=None if result.corners_native_px is None else result.corners_native_px.tolist(),
                   chosen_key=result.chosen_key, no_court_reason=result.no_court_reason, reused_from=result.reused_from,
                   stage_seconds=result.stage_seconds, seconds=perf_counter() - started)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='output .json.gz file')
    parser.add_argument('--people', type=Path, help='native-pixel pose_{bboxes,kps,ndet}.npy.xz directory; otherwise run RTMLib')
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
    args = parser.parse_args()
    if args.saved_lines is None and (args.deeplsd_source is None or args.deeplsd_weights is None):
        parser.error('provide --saved-lines or both --deeplsd-source and --deeplsd-weights')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    started = perf_counter()
    with VideoFrames(args.video) as frames:
        lines: LineSource
        if args.saved_lines is not None:
            lines = SavedLines({int(index): segments for index, segments in read_json(args.saved_lines).items()})
        else:
            lines = DeepLSDLines(args.deeplsd_source, args.deeplsd_weights, device=args.device)
        if args.people is not None:
            people: PeopleSource | None = PoseArrays.from_directory(args.people)
        elif args.require_people:
            from shared.rtmlib_pose import RtmlibPoseExtractor

            people = RtmlibPeople(frames, RtmlibPoseExtractor(device=args.device))
        else:
            people = None
        if isinstance(people, PoseArrays) and people.frame_count < frames.frame_count:
            raise ValueError('Saved poses do not cover the source video')
        detector = CourtDetector(Switches(workers=args.workers, timing=True,
                                          full_score_limit=args.full_score_limit,
                                          require_people=args.require_people,
                                          template_device=args.template_device))
        setup_seconds = perf_counter() - started
        scene_started = perf_counter()
        if args.scenes is not None:
            scenes = [SceneInfo(*span) for span in read_json(args.scenes)]
        elif args.pyscenedetect:
            scenes = PySceneDetectSource(histograms=True).scenes(args.video, frames.frame_count, frames.fps)
        else:
            scenes = [SceneInfo(0, frames.frame_count)]
        validate_scenes(scenes, frames.frame_count)
        scene_seconds = perf_counter() - scene_started
        processing_started = perf_counter()
        rows = []
        # Every scene's search and scoring share one set of worker processes.
        with detector:
            for row in scene_courts(detector, frames, people, lines, scenes, video_id=args.video.stem,
                                    reuse_courts=args.reuse_courts):
                rows.append(row)
                print(json.dumps(row), flush=True)
        result = {'video': args.video.name, 'fps': frames.fps, 'frame_count': frames.frame_count,
                  'native_size': frames.size, 'setup_seconds': setup_seconds, 'scene_seconds': scene_seconds,
                  'processing_seconds': perf_counter() - processing_started,
                  'total_seconds': perf_counter() - started, 'saved_people': args.people is not None,
                  'require_people': args.require_people, 'template_device': args.template_device,
                  'saved_lines': args.saved_lines is not None, 'scenes': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'wt') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
