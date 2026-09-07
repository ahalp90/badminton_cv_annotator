"""Rerun scene courts and retain raw neural predictions for later geometric replay."""
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from courtkeynet.court_corners import pick_scene_corners
from courtkeynet.wrapper import CourtKeyNetDetector


def json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: object) -> None:
    with gzip.open(path, 'xt', encoding='utf-8') as handle:
        json.dump(value, handle, default=json_value)


def read_json(path: Path) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def run(args: argparse.Namespace) -> None:
    if args.scene_indices is not None and len(args.video_ids) != 1:
        raise ValueError('--scene-indices requires a single video')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for video_id in args.video_ids:
        folders = list(args.prepared_root.glob(f'{video_id:02d} *'))
        if len(folders) != 1:
            raise ValueError(f'expected one prepared folder for video {video_id}')
        folder = folders[0]
        evidence = read_json(folder / 'court_evidence.json.gz')
        receipt = read_json(folder / 'court_receipt.json.gz')
        settings = receipt['configuration']
        detector_options = {key: settings[key] for key in (
            'resize_mode', 'corner_min_peak_conf', 'max_entropy', 'area_bounds',
        ) if key in settings}
        detector = CourtKeyNetDetector(device=args.device, **detector_options)
        records = evidence['scene_records']
        if args.scene_indices is not None:
            records = [records[index] for index in sorted(args.scene_indices)]
        write_json(args.output / f'video_{video_id:02d}_metadata.json.gz', {
            'video_id': video_id, 'source': folder.name, 'device': args.device,
            'detector_options': detector_options, 'scene_count': len(records),
            'scene_indices': [record['scene_index'] for record in records],
        })
        capture = cv2.VideoCapture(str(args.sources / f'{folder.name}.mp4'))
        if not capture.isOpened():
            raise OSError(f'could not open source video {folder.name}')
        next_frame = 0
        started = perf_counter()
        recovered = 0
        try:
            for position, record in enumerate(records):
                scene_started = perf_counter()
                frames = []
                samples = record['sampled_frame_indices']
                for target in samples:
                    # Seek across long gaps; decode nearby samples sequentially.
                    if target - next_frame > 300 or target < next_frame:
                        if not capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                            raise OSError(f'could not seek source frame {target}')
                        next_frame = target
                    while next_frame <= target:
                        if not capture.grab():
                            raise OSError(f'could not decode source frame {next_frame}')
                        next_frame += 1
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise OSError(f'could not retrieve source frame {target}')
                    frames.append(frame)
                detections = detector.detect_batch(frames)
                quad = pick_scene_corners(frames, detections, corner_min_peak_conf=detector.corner_min_peak_conf)
                recovered += quad is not None
                write_json(args.output / f"video_{video_id:02d}_scene_{record['scene_index']:04d}.json.gz", {
                    'index': record['scene_index'], 'interval': [record['start_frame'], record['end_frame']],
                    'sampled_frame_indices': samples, 'frame_wh': list(frames[0].shape[1::-1]),
                    'raw_nn': [asdict(detection) for detection in detections], 'court_quad': quad,
                    'elapsed_seconds': perf_counter() - scene_started,
                })
                if (position + 1) % 25 == 0:
                    print(f'video {video_id}: {position + 1}/{len(records)} scenes, {recovered} courts', flush=True)
        finally:
            capture.release()
        summary.append({'video_id': video_id, 'scenes': len(records), 'courts': recovered,
                        'seconds': perf_counter() - started})
        print(summary[-1], flush=True)
    write_json(args.output / 'summary.json.gz', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video-ids', type=int, nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--scene-indices', type=int, nargs='+')
    run(parser.parse_args())
