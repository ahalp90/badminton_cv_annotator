"""Run released MonoTrack selection and HitNet with the training-order corner adapter."""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from common import (
    HERE,
    court_records,
    fixed_stream,
    load_config,
    load_manifest,
    read_array,
    read_json,
    write_json,
)

sys.path.insert(0, str(HERE.parent / 'hitnet_retrain'))
sys.path.insert(0, str(HERE.parent / 'monotrack/python/ai-badminton/src'))

import numpy as np
import pandas as pd


def trajectory(shuttle: np.ndarray, interpolate: bool) -> Any:
    from ai_badminton.trajectory import Trajectory  # pyrefly: ignore[missing-import]

    # Use the released CSV reader and interpolation, including its zero handling.
    csv = io.StringIO(pd.DataFrame({'X': shuttle[:, 0], 'Y': shuttle[:, 1]}).to_csv(index=False))
    return Trajectory(csv, interp=interpolate)


def detect_rally(detector: Any, corners: np.ndarray, players: np.ndarray,
                 shuttle: np.ndarray, fps: float, interpolate: bool) -> tuple[list[int], list[int]]:
    # detect_hits applies [1,2,0,3]. This input makes its final corner slots
    # match training's BL, BR, TL, TR without changing the vendored code.
    detector.court = SimpleNamespace(corners=corners[[0, 3, 2, 1]])
    detector.poses = [pd.DataFrame(players[:, slot].reshape(len(players), 34)) for slot in range(2)]
    detector.trajectory = trajectory(shuttle, interpolate)
    detector.fps = fps
    return detector.detect_hits()


def run_item(config: dict[str, Any], item: dict[str, Any], detector: Any,
             court_set: str, arm: str, interpolate: bool, output: Path) -> None:
    from inputs import Court, fill_players, select_players, shuttle_variants

    paths = config['paths']
    pose_root = paths['pose_root'] / item['directory']
    kps = read_array(pose_root / 'pose_kps.npy.xz')
    scores = read_array(pose_root / 'pose_kp_scores.npy.xz')
    ndet = read_array(pose_root / 'pose_ndet.npy.xz')
    track_root = paths['inpaint_root'] / item['directory']
    sidecars = list(track_root.glob('*_stride8_inpaint_mask.json.gz'))
    if len(sidecars) != 1:
        raise ValueError(f'{item["item_id"]}: expected one inpainting mask')
    track = read_array(track_root / 'shuttle_track_inpainted.npy.xz')
    if len(track) != item['frame_count'] or len(kps) != len(track) or len(ndet) != len(track):
        raise ValueError(f'{item["item_id"]}: input timelines differ')
    shuttle = shuttle_variants(track, read_json(sidecars[0]), item['resolution'])[arm]
    courts = court_records(paths['inputs'] / court_set, item)
    started = time.perf_counter()
    events, failures = [], []
    for span, record in zip(item['spans'], courts, strict=True):
        if not record['scene_valid']:
            failures.append({'span_id': span['span_id'], 'reason': 'court_detection'})
            continue
        start, end = span['start_frame'], span['end_frame']
        corners = np.asarray(record['corners_native_px'], dtype=float)
        stage = 'player_selection'
        try:
            court = Court(corners[[3, 2, 0, 1]])
            players = select_players(kps[start:end], scores[start:end], ndet[start:end], court)
            stage = 'pose_filling'
            players = fill_players(players)
            stage = 'hit_detection'
            frames, sides = detect_rally(detector, corners, players, shuttle[start:end], item['fps'], interpolate)
            events.extend([[start + int(frame), 1.0, 'Bot' if side == 1 else 'Top']
                           for frame, side in zip(frames, sides, strict=True)])
        except Exception as error:  # noqa: BLE001 — each failed rally remains in the evaluation denominator.
            traceback.print_exc()
            failures.append({'span_id': span['span_id'], 'reason': type(error).__name__, 'stage': stage})
    events.sort(key=lambda event: event[0])
    result = {'item_id': item['item_id'], 'status': 'completed', 'boundaries': 'provided',
              'court_set': court_set, 'shuttle': arm, 'interpolation': interpolate,
              'stream': fixed_stream(events, item['spans']), 'failed_rallies': failures,
              'seconds': time.perf_counter() - started}
    write_json(output / f'{item["item_id"]}.json.gz', result)
    print(f'{item["item_id"]}: {len(events)} contacts, {len(failures)} failed rallies', flush=True)


def main() -> None:
    import tensorflow as tf  # pyrefly: ignore[missing-import]
    from ai_badminton.hit_detector import (  # pyrefly: ignore[missing-import]
        MLHitDetector,
    )

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--arm', choices=('kept', 'zeroed'), required=True)
    parser.add_argument('--courts', choices=('monotrack', 'ours'), default='monotrack')
    parser.add_argument('--interpolation', choices=('on', 'off'), default='on')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', help='one video for a smoke test')
    args = parser.parse_args()
    config = load_config(args.config)
    items = load_manifest(config['paths']['inputs'])['items']
    if args.only:
        items = [next(item for item in items if item['item_id'] == args.only)]
    detector = MLHitDetector(None, None, None, config['paths']['weights'] / f'hitnet_{args.arm}.h5', debug=False)
    args.output.mkdir(parents=True, exist_ok=False)
    for item in items:
        run_item(config, item, detector, args.courts, args.arm, args.interpolation == 'on', args.output)


if __name__ == '__main__':
    main()
