"""Run the current annotator on our courts, MonoTrack courts, or full-video discovery."""
from __future__ import annotations

import argparse
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from common import load_config, load_manifest, read_array, write_json

COURTS = {'ours_our_courts': 'ours', 'ours_mono_courts': 'monotrack_for_ours', 'ours_discovery': 'discovery'}


def run_item(task: tuple[dict[str, Any], dict[str, Any], str, Path]) -> None:
    from annotator.models import load_models
    from annotator.outcomes.point_winner import SHIPPED_LANDING_FILTER_OPTIONS
    from annotator.run_video import RunCapture, run_video
    from annotator.shuttle_track import validate_shuttle_track
    from annotator.types import DeadMaskMode
    from dataset_builder.vision import load_court_vision, load_pose_arrays
    from experiments.annotator.old_court_regression.retrain import stream_payload

    config, item, condition, output = task
    paths = config['paths']
    models = load_models(paths['bundle'])
    if models.preprocessing.dead_mask_mode is not DeadMaskMode.REPLAY:
        raise ValueError('This comparison requires the selected REPLAY mask, without a person-vote mask')
    if (models.preprocessing.rejected_grades != frozenset({1, 2, 3})
            or models.contact_settings.score_cutoff != 0.9
            or models.contact_settings.reject_masked_without_player
            or models.side_geometry.value != 'video'):
        raise ValueError('Model bundle settings differ from the selected annotator')
    frame_count = item['frame_count']
    track_root = paths['inpaint_root'] / item['directory']
    track = read_array(track_root / 'shuttle_track_inpainted.npy.xz')
    validate_shuttle_track(track, frame_count)
    guards = read_array(track_root / 'shuttle_guard_codes_inpainted.npy.xz')
    pose = load_pose_arrays(paths['pose_root'] / item['directory'], frame_count)
    vision = load_court_vision(
        paths['inputs'] / COURTS[condition] / item['item_id'], video_id=item['item_id'],
        frame_count=frame_count, resolution=tuple(map(float, item['resolution'])),
    )
    court = vision.evidence.inputs
    assert court is not None
    discovery = condition == 'ours_discovery'
    spans = None if discovery else [(span['start_frame'], span['end_frame']) for span in item['spans']]
    record = {'item_id': item['item_id'], 'condition': condition, 'status': 'completed',
              'boundaries': 'discovered' if discovery else 'provided', 'stream': {'events': [], 'sequences': []}}
    started = time.perf_counter()
    capture = RunCapture()
    try:
        run_video(
            track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
            fps=item['fps'], base=models.preprocessing, models=models,
            landing_options=SHIPPED_LANDING_FILTER_OPTIONS, net_band=court.net_band,
            resolution=court.resolution, video_id=item['item_id'], court_info=court.court_info,
            gate_court_info=court.gate_court_info, gate_resolution_table=court.gate_resolution_table,
            court_present=vision.evidence.court_present, homography_rows=court.homography_rows,
            cut_frames=[end for _, end in vision.raw_cuts[:-1]], keep_vote=vision.evidence.keep_vote,
            inpaint_codes=guards, court_invalid_is_excluded=True,
            landing_error_band_m=court.landing_error_band_m, spans=spans, capture=capture,
        )
        prediction = capture.hybrid
        if prediction is None:
            raise ValueError('Annotator did not return its final contact stream')
        record['stream'] = stream_payload(prediction.refined.sequences, prediction.refined.events)
    except Exception as error:  # noqa: BLE001 — inference failures remain in the scored population.
        traceback.print_exc()
        record.update(status='failed', failure_type=type(error).__name__)
    record['seconds'] = time.perf_counter() - started
    write_json(output / f'{item["item_id"]}.json.gz', record)
    print(f'{condition} {item["item_id"]}: {record["status"]}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--condition', choices=tuple(COURTS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', help='one video for a smoke test; cannot produce a full-population score')
    args = parser.parse_args()
    config = load_config(args.config)
    items = load_manifest(config['paths']['inputs'])['items']
    if args.only:
        items = [next(item for item in items if item['item_id'] == args.only)]
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(config, item, args.condition, args.output) for item in items]
    with ProcessPoolExecutor(max_workers=config['execution']['annotator_workers']) as pool:
        list(pool.map(run_item, tasks))


if __name__ == '__main__':
    main()
