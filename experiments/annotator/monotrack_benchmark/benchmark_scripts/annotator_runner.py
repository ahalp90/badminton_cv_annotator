"""Run the current annotator on our courts, MonoTrack courts, or full-video discovery."""
from __future__ import annotations

import argparse
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from experiments.annotator.heuristic_tuning.sweep import CandidateSpec

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

from common import load_config, load_manifest, read_array, write_json

COURTS = {'ours_our_courts': 'ours', 'ours_mono_courts': 'monotrack_for_ours', 'ours_discovery': 'discovery'}


def load_heuristic_settings(path: Path) -> CandidateSpec:
    from experiments.annotator.heuristic_only_tune.run import load_candidates

    document, specs = load_candidates(path)
    if document['stage'] != 'selected' or len(specs) != 1:
        raise ValueError('heuristic settings must contain exactly one selected configuration')
    return specs[0]


def run_item(
    task: tuple[dict[str, Any], dict[str, Any], str, Path], heuristic_spec: CandidateSpec | None = None,
) -> None:
    from annotator.models import load_models
    from annotator.outcomes.point_winner import SHIPPED_LANDING_FILTER_OPTIONS
    from annotator.run_video import RunCapture, run_video
    from annotator.shuttle_track import validate_shuttle_track
    from annotator.types import DeadMaskMode
    from dataset_builder.vision import load_court_vision, load_pose_arrays
    from experiments.annotator.old_court_regression.retrain import stream_payload

    config, item, condition, output = task
    paths = config['paths']
    if heuristic_spec is None:
        models = load_models(paths['bundle'])
        base, serve_start = models.preprocessing, None
        if base.dead_mask_mode is not DeadMaskMode.REPLAY:
            raise ValueError('This comparison requires the selected REPLAY mask, without a person-vote mask')
        if (base.rejected_grades != frozenset({1, 2, 3})
                or models.contact_settings.score_cutoff != 0.9
                or models.contact_settings.reject_masked_without_player
                or models.side_geometry.value != 'video'):
            raise ValueError('Model bundle settings differ from the selected annotator')
    else:
        from experiments.annotator.heuristic_tuning.sweep import _base_and_serve

        models = None
        base, serve_start = _base_and_serve(heuristic_spec)
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
              'mode': 'full' if heuristic_spec is None else 'heuristic',
              'boundaries': 'discovered' if discovery else 'provided', 'stream': {'events': [], 'sequences': []}}
    if heuristic_spec is not None:
        from experiments.annotator.heuristic_tuning.corpus_sweep import candidate_id
        from experiments.annotator.heuristic_tuning.sweep import serialise_spec

        record.update(candidate_id=candidate_id(heuristic_spec), heuristic_settings=serialise_spec(heuristic_spec))
    started = time.perf_counter()
    capture = RunCapture()
    try:
        result = run_video(
            track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
            fps=item['fps'], base=base, models=models, heuristic_only=heuristic_spec is not None,
            serve_start=serve_start,
            landing_options=SHIPPED_LANDING_FILTER_OPTIONS, net_band=court.net_band,
            resolution=court.resolution, video_id=item['item_id'], court_info=court.court_info,
            gate_court_info=court.gate_court_info, gate_resolution_table=court.gate_resolution_table,
            court_present=vision.evidence.court_present, homography_rows=court.homography_rows,
            cut_frames=[end for _, end in vision.raw_cuts[:-1]], keep_vote=vision.evidence.keep_vote,
            inpaint_codes=guards, court_invalid_is_excluded=True,
            landing_error_band_m=court.landing_error_band_m, spans=spans, capture=capture,
        )
        if heuristic_spec is None:
            prediction = capture.hybrid
            if prediction is None:
                raise ValueError('Annotator did not return its final contact stream')
            record['stream'] = stream_payload(prediction.refined.sequences, prediction.refined.events)
        else:
            from experiments.annotator.heuristic_only_tune.data import heuristic_stream

            record['stream'] = heuristic_stream(result)
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
    parser.add_argument(
        '--heuristic-settings', type=Path, help='one frozen selected configuration; disable learned trees',
    )
    parser.add_argument('--only', help='one video for a smoke test; cannot produce a full-population score')
    args = parser.parse_args()
    config = load_config(args.config)
    heuristic_spec = None if args.heuristic_settings is None else load_heuristic_settings(args.heuristic_settings)
    items = load_manifest(config['paths']['inputs'])['items']
    if args.only:
        items = [item for item in items if item['item_id'] == args.only]
        if not items:
            raise ValueError(f'{args.only}: video is outside the fixed benchmark population')
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = [(config, item, args.condition, args.output) for item in items]
    with ProcessPoolExecutor(max_workers=config['execution']['annotator_workers']) as pool:
        list(pool.map(partial(run_item, heuristic_spec=heuristic_spec), tasks))


if __name__ == '__main__':
    main()
