"""Copy the evaluation windows and court inputs into a portable benchmark dataset."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from common import load_manifest, read_json, write_json


def image_order(corners: Any) -> Any:
    """Put MonoTrack's four points in image TL, TR, BR, BL order for our annotator."""
    import numpy as np

    points = np.asarray(corners, dtype=float)
    top, bottom = np.split(points[np.argsort(points[:, 1])], 2)
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]])


def vote_free_courts(item: dict[str, Any], source: Path, output: Path) -> None:
    """Accept every successful MonoTrack fit without applying our detector's person vote."""
    import numpy as np
    import pandas as pd

    from annotator.courts.evidence import (
        CourtEvidenceResult,
        CourtInputs,
        CourtSceneRecord,
        SceneStatus,
        build_net_band,
        build_scene_rows,
        detected_court_info,
    )
    from annotator.outcomes.point_winner import corner_error_band_from_corners
    from dataset_builder.vision import CourtVision, persist_court_vision
    from shared.court import HOMOGRAPHY_RESOLUTION

    payload = read_json(source / 'court_evidence.json.gz')
    resolution = tuple(map(float, item['resolution']))
    records, intervals, accepted_intervals, accepted_corners = [], [], [], []
    present = np.zeros(item['frame_count'], dtype=bool)
    for index, scene in enumerate(payload['scene_records']):
        start, end = scene['start_frame'], scene['end_frame']
        accepted = scene['scene_valid']
        corners = image_order(scene['corners_native_px']) if accepted else None
        intervals.append((start, end))
        if accepted:
            accepted_intervals.append((start, end))
            accepted_corners.append(corners * np.asarray(HOMOGRAPHY_RESOLUTION) / np.asarray(resolution))
            present[start:end] = True
        records.append(CourtSceneRecord(
            video_id=item['item_id'], case_id=item['item_id'], parent='monotrack', scene_index=index,
            start_frame=start, end_frame=end, status=SceneStatus.COURT if accepted else SceneStatus.NO_COURT,
            analysed_frame=(start + end) // 2, corners_native_px=corners,
            no_court_reason=None if accepted else scene['status'], reused_from=None, error=None,
            exactly_two_count=0, exactly_two_fraction=0.0, scene_valid=accepted,
        ))
    representative = max(range(len(accepted_intervals)),
                         key=lambda index: accepted_intervals[index][1] - accepted_intervals[index][0])
    corners = accepted_corners[representative]
    info = detected_court_info(corners)
    inputs = CourtInputs(
        court_info=info, gate_court_info={item['item_id']: info}, net_band=build_net_band(info, resolution),
        resolution=resolution,
        gate_resolution_table=pd.DataFrame({'width': [resolution[0]], 'height': [resolution[1]]},
                                           index=[item['item_id']]),
        homography_rows=build_scene_rows(item['item_id'], accepted_intervals, accepted_corners, resolution),
        landing_error_band_m=corner_error_band_from_corners(corners, info, 3.5), active_corners_refpx=corners,
    )
    # The selected model uses REPLAY, which does not consume person-vote masks.
    # Zero means no vote was measured; scene_valid/present carry detector success.
    evidence = CourtEvidenceResult(inputs, tuple(records), np.zeros_like(present), present)
    persist_court_vision(output, video_id=item['item_id'], court=CourtVision(tuple(intervals), evidence),
                        frame_count=item['frame_count'], resolution=resolution)


def prepare(monotrack: Path, ours: Path, discovery: Path, output: Path) -> None:
    manifests = {name: read_json(path) for name, path in
                 (('monotrack', monotrack), ('ours', ours), ('discovery', discovery))}
    references = {item['item_id']: item for item in manifests['discovery']['items']}
    output.mkdir(parents=True, exist_ok=False)
    items = []
    for item in references.values():
        items.append({**{key: item[key] for key in ('item_id', 'video_id', 'fps', 'frame_count', 'resolution', 'spans')},
                      'directory': Path(item['pose_dir']).name, 'source_filename': Path(item['source_video']).name})
    write_json(output / 'manifest.json.gz', {'population': manifests['discovery']['population'], 'items': items})
    load_manifest(output)
    for name, manifest in manifests.items():
        if {item['item_id'] for item in manifest['items']} != set(references):
            raise ValueError(f'{name}: video population differs')
        for item in manifest['items']:
            reference = references[item['item_id']]
            if {key: value for key, value in item.items() if key != 'court_dir'} != {
                key: value for key, value in reference.items() if key != 'court_dir'
            }:
                raise ValueError(f'{name}/{item["item_id"]}: non-court inputs differ')
            source = Path(item['court_dir'])
            destination = output / name / item['item_id']
            payload = read_json(source / 'court_evidence.json.gz')
            payload.pop('provenance', None)
            for scene in payload['scene_records']:
                scene.pop('source_court_result', None)
                if scene.get('error') is not None:
                    scene['error'] = 'Saved court detector failure'
            write_json(destination / 'court_evidence.json.gz', payload)
            for filename in ('court_present.npy.xz', 'court_keep_vote.npy.xz'):
                if name != 'monotrack' or filename == 'court_present.npy.xz':
                    shutil.copyfile(source / filename, destination / filename)
            if name == 'monotrack':
                vote_free_courts(item, destination, output / 'monotrack_for_ours' / item['item_id'])
    print('Prepared 46 videos, 3327 rally windows and 37184 contact labels', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--monotrack-manifest', type=Path, required=True)
    parser.add_argument('--ours-manifest', type=Path, required=True)
    parser.add_argument('--discovery-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.monotrack_manifest, args.ours_manifest, args.discovery_manifest, args.output)


if __name__ == '__main__':
    main()
