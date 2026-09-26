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

from . import feet
from .detect import CourtDetector, Switches
from .inputs import FrameReader, PeopleSource, ViewInputs, same_frame_provenance
from .line_sources import DeepLSDLines, LineSource, SavedLines
from .scene_sources import PySceneDetectSource, SceneInfo
from .video_inputs import PoseArrays, RtmlibPeople, VideoFrames


def scene_courts(
    detector: CourtDetector, frames: FrameReader, people: PeopleSource, lines: LineSource,
    scenes: Sequence[SceneInfo], *, video_id: str,
) -> Iterator[dict[str, Any]]:
    """Detect the middle frame of each scene without crossing a cut for foot samples.

    A scene too short for the existing foot window is reported as unanalysed.
    It is not counted as evidence that no court is present.
    """
    for scene_index, scene in enumerate(scenes):
        started = perf_counter()
        anchor = (scene.first_frame + scene.last_frame) // 2
        view_id = f'{video_id}_scene_{scene_index:04d}_frame_{anchor}'
        row: dict[str, Any] = {'view_id': view_id, 'first_frame': scene.first_frame, 'last_frame': scene.last_frame,
                               'frame_index': anchor}
        # Validate this input boundary before spending time on line/pose inference.
        try:
            feet.window_frames(anchor, frames.fps, scene.first_frame, scene.last_frame)
        except ValueError:
            row.update(status='scene_too_short_for_feet', corners_native_px=None, seconds=perf_counter() - started)
            yield row
            continue
        frame = frames.read([anchor])[0]
        segments = lines.segments(frame, anchor)
        anchor_people = people.samples([anchor])
        if len(anchor_people) != 1 or anchor_people[0].frame_index != anchor:
            raise ValueError(f'{view_id}: people source did not return the requested anchor')
        view = ViewInputs(view_id, frame, anchor, (scene.first_frame, scene.last_frame), segments,
                          anchor_people[0].boxes_px, same_frame_provenance(view_id, anchor))
        result = detector.detect(view, people, frames)
        row.update(status='court' if result.corners_native_px is not None else 'no_court',
                   corners_native_px=None if result.corners_native_px is None else result.corners_native_px.tolist(),
                   chosen_key=result.chosen_key, no_court_reason=result.no_court_reason,
                   stage_seconds=result.stage_seconds, seconds=perf_counter() - started)
        yield row


def validate_scenes(scenes: Sequence[SceneInfo], frame_count: int) -> None:
    """External scene inputs must partition the source timeline exactly."""
    next_frame = 0
    for scene in scenes:
        if scene.first_frame != next_frame or not scene.first_frame <= scene.last_frame < frame_count:
            raise ValueError(f'Scene {scene.first_frame}..{scene.last_frame} does not continue at frame {next_frame}')
        next_frame = scene.last_frame + 1
    if next_frame != frame_count:
        raise ValueError(f'Scenes end at {next_frame - 1}, expected {frame_count - 1}')


def read_json(path: Path) -> Any:
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='output .json.gz file')
    parser.add_argument('--people', type=Path, help='native-pixel pose_{bboxes,kps,ndet}.npy.xz directory; otherwise run RTMLib')
    parser.add_argument('--saved-lines', type=Path, help='.json.gz object mapping source frame numbers to line arrays')
    parser.add_argument('--deeplsd-source', type=Path)
    parser.add_argument('--deeplsd-weights', type=Path)
    parser.add_argument('--device', default='cuda', help='device for live DeepLSD and RTMLib')
    scene_options = parser.add_mutually_exclusive_group()
    scene_options.add_argument('--scenes', type=Path, help='.json.gz list of inclusive [first,last] scene ranges')
    scene_options.add_argument('--pyscenedetect', action='store_true', help='detect cuts and representative scene histograms')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--full-score-limit', type=int, help='optional cheap-score trial limit; omit for exhaustive scoring')
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
        people = PoseArrays.from_directory(args.people) if args.people else RtmlibPeople(frames, args.device)
        if isinstance(people, PoseArrays) and people.frame_count < frames.frame_count:
            raise ValueError('Saved poses do not cover the source video')
        detector = CourtDetector(Switches(workers=args.workers, timing=True, full_score_limit=args.full_score_limit))
        setup_seconds = perf_counter() - started
        scene_started = perf_counter()
        if args.scenes is not None:
            scenes = [SceneInfo(*span) for span in read_json(args.scenes)]
        elif args.pyscenedetect:
            scenes = PySceneDetectSource(histograms=True).scenes(args.video, frames.frame_count, frames.fps)
        else:
            scenes = [SceneInfo(0, frames.frame_count - 1)]
        validate_scenes(scenes, frames.frame_count)
        scene_seconds = perf_counter() - scene_started
        processing_started = perf_counter()
        rows = []
        for row in scene_courts(detector, frames, people, lines, scenes, video_id=args.video.stem):
            rows.append(row)
            print(json.dumps(row), flush=True)
        result = {'video': args.video.name, 'fps': frames.fps, 'frame_count': frames.frame_count,
                  'native_size': frames.size, 'setup_seconds': setup_seconds, 'scene_seconds': scene_seconds,
                  'processing_seconds': perf_counter() - processing_started,
                  'total_seconds': perf_counter() - started, 'saved_people': args.people is not None,
                  'saved_lines': args.saved_lines is not None, 'scenes': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, 'wt') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
