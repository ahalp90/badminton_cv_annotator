"""Build and validate a provenance-safe ``person_observations`` input pack.

The historical person-observation inputs used boxes from a different image for
five cases.  This module makes the repaired route explicit.  It never edits the
historical ``inputs/`` directory.

The exact-frame detector file uses the shared ``person-detections/1`` shape
(gzip JSON as produced by ``detect_exact_people.py``)::

    {
      "schema": "person-detections/1",
      "model": {
        "basename": "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip",
        "score_cutoff": 0.2,
        "score_rule": "strict_gt"
      },
      "coordinate_order": "xyxy",
      "cases": {
        "gxBQ_window_00_frame_5": {
          "frame_index": 5,
          "image": "gxBQ_window_00_frame_00000005.png",
          "image_md5": "<md5 of the frozen image>",
          "dimensions": {"width": 1920, "height": 1080},
          "bboxes": [[x1, y1, x2, y2], ...],
          "scores": [0.91, ...]
        }
      }
    }

The packet has exactly the six pinned exact-frame cases used by the detector
batch.  The builder currently needs the GX5 record; the four broadcast cases
use their three recorded pose samples.  The other exact-frame records remain
in the packet so the shared detector result cannot silently change membership.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np

from shared import (
    PACK_OF,
    PACKS,
    add_helper_paths,
    case_provenance,
    frame_path,
    load_estimator,
    load_source,
    read,
    write,
)

add_helper_paths()

import vp_pruning
from run_population import prepare

SCHEMA = 'line-identity-person-observation-repair/1'
DETECTION_SCHEMA = 'person-detections/1'
ARM = 'person_observations'
SCORE_CUTOFF = 0.2
FOUR_BROADCAST_CASES = (
    'shuttleset_03_scene_0016',
    'shuttleset_03_scene_0017',
    'shuttleset_03_scene_0019',
    'shuttleset_21_scene_0020',
)
GX5_CASE_ID = 'gxBQ_window_00_frame_5'
REPAIR_CASE_IDS = FOUR_BROADCAST_CASES + (GX5_CASE_ID,)
DETECTION_CASE_IDS = (
    GX5_CASE_ID,
    'yellow_short_frame_14',
    'letterboxed_short_frame_58',
    'centre_short_frame_64',
    'centre_short_frame_71',
    'am4_window_00_frame_319',
)
RTMDET_M_BASENAME = 'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip'
# These fields are the compact, stable metadata contract for the shared detector packet.
# The marking consumer checks the frozen image hashes for its own five records.
DETECTION_METADATA = {
    GX5_CASE_ID: (5, 'gxBQ_window_00_frame_00000005.png', 1920, 1080),
    'yellow_short_frame_14': (14, 'yellow_short_frame_00000014.png', 1280, 720),
    'letterboxed_short_frame_58': (58, 'letterboxed_short_frame_00000058.png', 960, 720),
    'centre_short_frame_64': (64, 'centre_short_frame_00000064.png', 1280, 720),
    'centre_short_frame_71': (71, 'centre_short_frame_00000071.png', 1280, 720),
    'am4_window_00_frame_319': (319, 'am4_window_00_frame_00000319.png', 1920, 1080),
}
HERE = Path(__file__).resolve().parent
HISTORICAL_INPUTS = HERE / 'inputs'
FLOAT_ATOL = 1e-12


def _error(context: str, message: str) -> ValueError:
    return ValueError(f'{context}: {message}')


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(context, 'must be an object')
    return value


def _non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(context, 'must be a non-negative integer')
    return value


def file_md5(path: Path) -> str:
    """Return one compact digest for an input artefact."""
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key!r}')
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    """Read either ordinary JSON or gzip-compressed JSON."""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        value = json.load(stream, object_pairs_hook=_unique_object)
    return dict(_object(value, str(path)))


def _as_boxes(value: Any, context: str) -> np.ndarray:
    boxes = np.asarray(value, dtype=float)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise _error(context, 'must be an array of [x1, y1, x2, y2] boxes')
    if not np.isfinite(boxes).all():
        raise _error(context, 'contains non-finite coordinates')
    left, top, right, bottom = boxes.T
    if np.any(right < left) or np.any(bottom < top):
        raise _error(context, 'contains reversed box bounds')
    return boxes


def _validate_box_scale(boxes: np.ndarray, dimensions: Mapping[str, Any], context: str) -> None:
    width = float(dimensions['width'])
    height = float(dimensions['height'])
    if width <= 0 or height <= 0:
        raise _error(context, 'image dimensions must be positive')
    # The frozen detector exports contain tiny edge overshoots from float32.
    tolerance = 1.0
    if len(boxes) and (
        np.any(boxes[:, 0] < -tolerance)
        or np.any(boxes[:, 2] > width + tolerance)
        or np.any(boxes[:, 1] < -tolerance)
        or np.any(boxes[:, 3] > height + tolerance)
    ):
        raise _error(context, f'boxes do not use the {width:g}x{height:g} image coordinates')


def eligible_boxes(bboxes: Any, scores: Any, context: str = 'detections') -> np.ndarray:
    """Select boxes with the strict historical score rule ``score > 0.2``."""
    boxes = _as_boxes(bboxes, f'{context}.bboxes')
    score_array = np.asarray(scores, dtype=float)
    if score_array.ndim != 1 or len(score_array) != len(boxes):
        raise _error(context, 'scores and bboxes must have the same length')
    if not np.isfinite(score_array).all():
        raise _error(f'{context}.scores', 'contains non-finite values')
    return boxes[score_array > SCORE_CUTOFF]


def inclusive_pairwise_intersections(frame_boxes: Sequence[np.ndarray]) -> np.ndarray:
    """Return unique boxes covering a point in at least two frames.

    Rectangle contact counts as coverage.  The cross-product across boxes is
    intentional: detector slot order is not a player identity guarantee.
    """
    if len(frame_boxes) != 3:
        raise ValueError('two-of-three person masks require exactly three pose frames')
    intersections: list[np.ndarray] = []
    for first, second in combinations(frame_boxes, 2):
        first = _as_boxes(first, 'first frame boxes')
        second = _as_boxes(second, 'second frame boxes')
        if len(first) == 0 or len(second) == 0:
            continue
        left = np.maximum(first[:, None, 0], second[None, :, 0])
        top = np.maximum(first[:, None, 1], second[None, :, 1])
        right = np.minimum(first[:, None, 2], second[None, :, 2])
        bottom = np.minimum(first[:, None, 3], second[None, :, 3])
        overlap = (right >= left) & (bottom >= top)
        if overlap.any():
            pair = np.stack((left, top, right, bottom), axis=-1)
            intersections.append(pair[overlap])
    if not intersections:
        return np.empty((0, 4), dtype=float)
    return np.unique(np.concatenate(intersections), axis=0)


def two_of_three_mask(source: Mapping[str, Any]) -> tuple[np.ndarray, list[int]]:
    """Build a broadcast mask and return its box evidence and frame IDs."""
    samples = source.get('pose_samples')
    if not isinstance(samples, list) or len(samples) != 3:
        raise _error(str(source.get('id', 'case')), 'requires three recorded pose samples')
    frame_indices = [_non_negative_int(sample.get('frame_index'), 'pose frame index') for sample in samples]
    if len(set(frame_indices)) != 3:
        raise _error(str(source.get('id', 'case')), 'pose samples must use distinct frames')
    cutoff = source.get('provenance', {}).get('person_score_cutoff', SCORE_CUTOFF)
    if float(cutoff) != SCORE_CUTOFF:
        raise _error(str(source.get('id', 'case')), 'frozen score cutoff is not 0.2')
    frame_boxes = [
        eligible_boxes(sample.get('bboxes'), sample.get('scores'), f'pose_samples[{index}]')
        for index, sample in enumerate(samples)
    ]
    for index, boxes in enumerate(frame_boxes):
        _validate_box_scale(boxes, cast(Mapping[str, Any], source['dimensions']), f'pose_samples[{index}]')
    return inclusive_pairwise_intersections(frame_boxes), frame_indices


def midpoint_kept_ids(segments: Any, boxes: np.ndarray) -> list[int]:
    """Return source fragment IDs whose inclusive midpoint is outside all boxes."""
    fragments = np.asarray(segments, dtype=float)
    if fragments.size == 0:
        fragments = np.empty((0, 4), dtype=float)
    if fragments.ndim != 2 or fragments.shape[1] != 4:
        raise ValueError('segments must be an array of [x1, y1, x2, y2] fragments')
    if not np.isfinite(fragments).all():
        raise ValueError('segments contain non-finite coordinates')
    if len(boxes) == 0:
        return list(range(len(fragments)))
    midpoints = (fragments[:, :2] + fragments[:, 2:]) / 2.0
    left, top, right, bottom = boxes.T
    inside_x = (midpoints[:, None, 0] >= left) & (midpoints[:, None, 0] <= right)
    inside_y = (midpoints[:, None, 1] >= top) & (midpoints[:, None, 1] <= bottom)
    return np.flatnonzero(~(inside_x & inside_y).any(axis=1)).tolist()


def _frozen_image_md5(source: Mapping[str, Any]) -> tuple[Path, str]:
    path = frame_path(dict(source))
    actual = file_md5(path)
    expected = source.get('provenance', {}).get('line_image_file_md5')
    if not isinstance(expected, str) or actual != expected:
        raise _error(str(source['id']), 'frozen image MD5 does not match its provenance')
    return path, actual


def _case_detection(raw: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    if case_id not in DETECTION_METADATA:
        raise _error('detections.cases', f'unsupported case {case_id!r}')
    required = {'schema', 'model', 'coordinate_order', 'cases'}
    if not required.issubset(raw):
        raise _error('detections', f'missing required fields: {sorted(required - set(raw))}')
    if raw['schema'] != DETECTION_SCHEMA:
        raise _error('detections.schema', f'expected {DETECTION_SCHEMA!r}')
    model = _object(raw['model'], 'detections.model')
    if model.get('basename') != RTMDET_M_BASENAME:
        raise _error('detections.model.basename', f'expected {RTMDET_M_BASENAME!r}')
    if model.get('score_cutoff') != SCORE_CUTOFF:
        raise _error('detections.model.score_cutoff', 'must be exactly 0.2')
    if model.get('score_rule') != 'strict_gt':
        raise _error('detections.model.score_rule', 'must be strict_gt')
    if raw['coordinate_order'] != 'xyxy':
        raise _error('detections.coordinate_order', 'must be xyxy')
    cases = raw['cases']
    if not isinstance(cases, Mapping):
        raise _error('detections.cases', 'must be a keyed object')
    if set(cases) != set(DETECTION_CASE_IDS):
        raise _error('detections.cases', f'must contain exactly {list(DETECTION_CASE_IDS)!r}')
    required_record = {'frame_index', 'image', 'image_md5', 'dimensions', 'bboxes', 'scores'}
    for key, value in cases.items():
        record = _object(value, f'detections.cases[{key!r}]')
        if not required_record.issubset(record):
            raise _error(f'detections.cases[{key!r}]',
                         f'missing required fields: {sorted(required_record - set(record))}')
        expected_frame, expected_image, expected_width, expected_height = DETECTION_METADATA[key]
        frame_index = _non_negative_int(record['frame_index'], f'detections.cases[{key!r}].frame_index')
        if frame_index != expected_frame:
            raise _error(f'detections.cases[{key!r}].frame_index', f'expected {expected_frame}')
        image = record['image']
        if image != expected_image:
            raise _error(f'detections.cases[{key!r}].image', f'expected {expected_image!r}')
        image_md5 = record['image_md5']
        if (not isinstance(image_md5, str) or len(image_md5) != 32
                or any(character not in '0123456789abcdefABCDEF' for character in image_md5)):
            raise _error(f'detections.cases[{key!r}].image_md5', 'must be a 32-character hexadecimal MD5 digest')
        dimensions = _object(record['dimensions'], f'detections.cases[{key!r}].dimensions')
        if set(dimensions) != {'width', 'height'}:
            raise _error(f'detections.cases[{key!r}].dimensions', 'requires width and height')
        if any(isinstance(dimensions[name], bool) or not isinstance(dimensions[name], int)
               or dimensions[name] <= 0 for name in ('width', 'height')):
            raise _error(f'detections.cases[{key!r}].dimensions', 'width and height must be positive integers')
        if dimensions['width'] != expected_width or dimensions['height'] != expected_height:
            raise _error(f'detections.cases[{key!r}].dimensions',
                         f'expected {expected_width}x{expected_height}')
        boxes = _as_boxes(record['bboxes'], f'detections.cases[{key!r}].bboxes')
        scores = np.asarray(record['scores'], dtype=float)
        if scores.ndim != 1 or len(scores) != len(boxes) or not np.isfinite(scores).all():
            raise _error(f'detections.cases[{key!r}]', 'scores must be finite and match bboxes')
        if np.any(scores < 0) or np.any(scores > 1):
            raise _error(f'detections.cases[{key!r}].scores', 'must lie between zero and one')
        _validate_box_scale(boxes, dimensions, f'detections.cases[{key!r}]')
    value = cases.get(case_id)
    if value is None:
        raise _error('detections.cases', f'missing {case_id}')
    return _object(value, f'detections.cases[{case_id!r}]')


def exact_frame_evidence(source: Mapping[str, Any], detections_path: Path) -> dict[str, Any]:
    """Validate and extract one native-frame detector record for a frozen case."""
    raw = read_json(detections_path)
    record = _case_detection(raw, str(source['id']))
    required = {'frame_index', 'image', 'image_md5', 'dimensions', 'bboxes', 'scores'}
    if not required.issubset(record):
        raise _error(str(source['id']), f'missing required detector fields: {sorted(required - set(record))}')
    expected_frame = source.get('provenance', {}).get('anchor_frame_index')
    if expected_frame is None:
        expected_frame = source.get('provenance', {}).get('image_frame_indices', [None])[0]
    frame_index = _non_negative_int(record.get('frame_index'), f'{source["id"]}.frame_index')
    if frame_index != expected_frame:
        raise _error(str(source['id']), f'detection frame {frame_index} is not frozen frame {expected_frame}')
    image_path, image_md5 = _frozen_image_md5(source)
    if record.get('image') != image_path.name:
        raise _error(str(source['id']), 'detection image does not name the frozen image')
    if record.get('image_md5') != image_md5:
        raise _error(str(source['id']), 'detection image_md5 does not match the frozen image')
    dimensions = record['dimensions']
    if dimensions != source['dimensions']:
        raise _error(str(source['id']), 'detection dimensions differ from the frozen image')
    boxes = _as_boxes(record.get('bboxes'), f'{source["id"]}.bboxes')
    _validate_box_scale(boxes, cast(Mapping[str, Any], source['dimensions']), f'{source["id"]}.bboxes')
    scores = np.asarray(record.get('scores'), dtype=float)
    if scores.ndim != 1 or len(scores) != len(boxes):
        raise _error(str(source['id']), 'detection scores and boxes differ in length')
    if not np.isfinite(scores).all():
        raise _error(str(source['id']), 'detection scores contain non-finite values')
    return {
        'frame_index': frame_index,
        'image': image_path.name,
        'image_md5': image_md5,
        'dimensions': dict(dimensions),
        'bboxes': boxes.tolist(),
        'scores': scores.tolist(),
    }


def _write_case_inputs(inputs_dir: Path, source: Mapping[str, Any], kept_ids: list[int], saved: Mapping[str, Any]) -> None:
    """Write only the fields consumed by the matcher and its input gate."""
    case_id = str(source['id'])
    folder = inputs_dir / ARM
    filtered = {
        **source,
        'segments_px': [source['segments_px'][index] for index in kept_ids],
    }
    write(folder / 'cases' / f'{case_id}.json.gz', filtered)
    write(folder / 'estimators' / f'{case_id}.json.gz', {
        'case_id': case_id,
        'working_size': saved['working_size'],
        'settings': saved['settings'],
        'estimator': saved['estimator'],
        'arm': ARM,
        'fragments_kept': len(kept_ids),
        'fragments_total': len(source['segments_px']),
        'kept_fragment_ids': kept_ids,
        'note': 'Saved baseline directions with a provenance-repaired person mask.',
    })


def _gate_saved_estimator(source: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    segments, _, size = prepare(dict(source))
    if list(size) != saved['working_size']:
        raise _error(str(source['id']), 'saved working size differs from the frozen source')
    settings = vp_pruning.Settings(**saved['settings'])
    _, replayed = vp_pruning.estimate(segments, size, settings)
    estimator = saved['estimator']
    if replayed['retained_candidate_ids'] != estimator['retained_candidate_ids']:
        raise _error(str(source['id']), 'unfiltered replay does not reproduce saved directions')
    np.testing.assert_allclose(replayed['points_working'], estimator['points_working'], rtol=0, atol=FLOAT_ATOL)
    if replayed['retained_support_masks'] != estimator['retained_support_masks']:
        raise _error(str(source['id']), 'unfiltered replay does not reproduce saved support masks')


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _check_fresh_target(inputs_dir: Path, manifest_path: Path) -> None:
    historical = HISTORICAL_INPUTS.resolve()
    if _inside(inputs_dir, historical):
        raise ValueError('repair inputs must be outside historical line_identity/inputs')
    if _inside(manifest_path, historical):
        raise ValueError('repair manifest must be outside historical line_identity/inputs')
    repair_dir = inputs_dir / ARM
    if repair_dir.exists():
        raise FileExistsError(f'repair arm directory already exists: {repair_dir}')
    if manifest_path.exists():
        raise FileExistsError(f'repair manifest already exists: {manifest_path}')


def _check_repair_allowlist(case_ids: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(sorted(case_ids))
    if len(selected) != len(set(selected)):
        raise ValueError('repair case IDs must be unique')
    if frozenset(selected) != frozenset(REPAIR_CASE_IDS):
        raise ValueError('repair construction requires exactly the five-case repair allowlist')
    return selected


def build_repair_inputs(
    inputs_dir: Path,
    manifest_path: Path,
    case_ids: Sequence[str] = REPAIR_CASE_IDS,
    exact_detections: Path | None = None,
) -> dict[str, Any]:
    """Write a new repaired input directory and its deterministic manifest."""
    inputs_dir = inputs_dir.resolve()
    manifest_path = manifest_path.resolve()
    _check_fresh_target(inputs_dir, manifest_path)
    selected = _check_repair_allowlist(case_ids)
    if GX5_CASE_ID in selected and exact_detections is None:
        raise ValueError(f'{GX5_CASE_ID} requires --exact-detections')

    entries = []
    prepared: list[tuple[Mapping[str, Any], list[int], Mapping[str, Any]]] = []
    for case_id in selected:
        source = load_source(case_id)
        case_provenance(case_id)  # Validate the pinned frozen pack before deriving evidence.
        saved = load_estimator(case_id)
        _gate_saved_estimator(source, saved)
        if case_id == GX5_CASE_ID:
            if exact_detections is None:
                raise ValueError(f'{GX5_CASE_ID} requires --exact-detections')
            detection = exact_frame_evidence(source, exact_detections)
            mask_boxes = eligible_boxes(detection['bboxes'], detection['scores'], f'{case_id}.detections')
            mode = 'exact_frame'
            evidence = {'detection': detection}
            frame_indices = [detection['frame_index']]
        else:
            mask_boxes, frame_indices = two_of_three_mask(source)
            mode = 'two_of_three'
            evidence = {
                'pose_frame_indices': frame_indices,
            }
        kept_ids = midpoint_kept_ids(source['segments_px'], mask_boxes)
        prepared.append((source, kept_ids, saved))
        entries.append({
            'case_id': case_id,
            'source_pack': PACKS[PACK_OF[case_id]].name,
            'source_image': source['image'],
            'mode': mode,
            'score_cutoff': SCORE_CUTOFF,
            'mask_boxes': mask_boxes.tolist(),
            'kept_fragment_ids': kept_ids,
            'fragments_total': len(source['segments_px']),
            'evidence': evidence,
        })
    for source, kept_ids, saved in prepared:
        _write_case_inputs(inputs_dir, source, kept_ids, saved)
    manifest = {
        'schema': SCHEMA,
        'arm': ARM,
        'mask_rule': 'inclusive midpoint exclusion from repair-mode boxes',
        'broadcast_support_rule': 'pairwise box intersection across at least two of three pose frames',
        'cases': entries,
    }
    if exact_detections is not None:
        manifest['detector_packet'] = {
            'basename': exact_detections.name,
            'md5': file_md5(exact_detections),
        }
    write(manifest_path, manifest)
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the manifest and reject a manifest meant for another arm."""
    manifest = read(path)
    if manifest.get('schema') != SCHEMA:
        raise _error(str(path), 'unsupported repair manifest schema')
    if manifest.get('arm') != ARM:
        raise _error(str(path), f'repair manifest is only valid for {ARM}')
    if manifest.get('mask_rule') != 'inclusive midpoint exclusion from repair-mode boxes':
        raise _error(str(path), 'unexpected mask rule')
    if manifest.get('broadcast_support_rule') != 'pairwise box intersection across at least two of three pose frames':
        raise _error(str(path), 'unexpected broadcast support rule')
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not cases:
        raise _error(str(path), 'cases must be a non-empty list')
    case_ids = [entry.get('case_id') for entry in cases if isinstance(entry, Mapping)]
    if len(case_ids) != len(cases) or len(set(case_ids)) != len(cases):
        raise _error(str(path), 'cases must have unique IDs')
    if frozenset(case_ids) != frozenset(REPAIR_CASE_IDS):
        raise _error(str(path), 'cases do not match the exact five-case repair allowlist')
    packet = manifest.get('detector_packet')
    needs_packet = GX5_CASE_ID in case_ids
    if needs_packet != (packet is not None):
        raise _error(str(path), 'detector packet provenance must appear exactly when GX5 is present')
    if packet is not None:
        packet = _object(packet, 'repair manifest.detector_packet')
        if set(packet) != {'basename', 'md5'}:
            raise _error('repair manifest.detector_packet', 'requires basename and md5')
        if (not isinstance(packet['basename'], str) or not packet['basename']
                or Path(packet['basename']).name != packet['basename']):
            raise _error('repair manifest.detector_packet.basename', 'must be a non-empty basename')
        if (not isinstance(packet['md5'], str) or len(packet['md5']) != 32
                or any(character not in '0123456789abcdefABCDEF' for character in packet['md5'])):
            raise _error('repair manifest.detector_packet.md5', 'must be a 32-character hexadecimal MD5 digest')
    return manifest


def _manifest_entry(manifest: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    for entry in manifest['cases']:
        if isinstance(entry, Mapping) and entry.get('case_id') == case_id:
            return entry
    raise _error('repair manifest', f'missing case {case_id}')


def _validate_input_files(inputs_dir: Path, source: Mapping[str, Any], entry: Mapping[str, Any]) -> None:
    case_id = str(source['id'])
    case_path = inputs_dir / ARM / 'cases' / f'{case_id}.json.gz'
    estimator_path = inputs_dir / ARM / 'estimators' / f'{case_id}.json.gz'
    filtered = read(case_path)
    written = read(estimator_path)
    if filtered.get('id') != case_id or written.get('case_id') != case_id or written.get('arm') != ARM:
        raise _error(case_id, 'repair input identity or arm does not match')
    if set(filtered) != set(source):
        raise _error(case_id, 'repair input keys differ from the frozen source')
    if any(filtered[key] != source[key] for key in source if key != 'segments_px'):
        raise _error(case_id, 'repair input changed a frozen source field')
    kept = entry.get('kept_fragment_ids')
    if not isinstance(kept, list) or any(isinstance(index, bool) or not isinstance(index, int) for index in kept):
        raise _error(case_id, 'manifest kept_fragment_ids must be integer IDs')
    expected_segments = [source['segments_px'][index] for index in kept if 0 <= index < len(source['segments_px'])]
    if len(expected_segments) != len(kept) or filtered.get('segments_px') != expected_segments:
        raise _error(case_id, 'filtered fragments do not match manifest kept IDs')
    if written.get('kept_fragment_ids') != kept:
        raise _error(case_id, 'estimator kept IDs do not match manifest')
    if written.get('fragments_kept') != len(kept) or written.get('fragments_total') != len(source['segments_px']):
        raise _error(case_id, 'estimator fragment counts do not match manifest')
    saved = load_estimator(case_id)
    if written.get('working_size') != saved['working_size'] or written.get('settings') != saved['settings']:
        raise _error(case_id, 'repair estimator settings differ from the saved baseline')
    if written.get('estimator') != saved['estimator']:
        raise _error(case_id, 'repair estimator directions differ from the saved baseline')


def validate_repair_case(case_id: str, inputs_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Recompute one mask from frozen evidence and validate its complete input route."""
    entry = _manifest_entry(manifest, case_id)
    source = load_source(case_id)
    case_provenance(case_id)
    if entry.get('source_pack') != PACKS[PACK_OF[case_id]].name or entry.get('source_image') != source['image']:
        raise _error(case_id, 'manifest source identity differs from the frozen pack')
    if entry.get('score_cutoff') != SCORE_CUTOFF:
        raise _error(case_id, 'manifest score cutoff is not exactly 0.2')
    mode = entry.get('mode')
    if mode == 'two_of_three':
        expected_boxes, frame_indices = two_of_three_mask(source)
        evidence = _object(entry.get('evidence'), f'{case_id}.evidence')
        if evidence.get('pose_frame_indices') != frame_indices:
            raise _error(case_id, 'manifest pose-frame evidence differs from the frozen source')
    elif mode == 'exact_frame':
        evidence = _object(entry.get('evidence'), f'{case_id}.evidence')
        detection = _object(evidence.get('detection'), f'{case_id}.evidence.detection')
        frame_index = _non_negative_int(detection.get('frame_index'), f'{case_id}.frame_index')
        if frame_index != 5 or case_id != GX5_CASE_ID:
            raise _error(case_id, 'exact-frame repair evidence is only valid for native GX5 frame 5')
        image_path, image_md5 = _frozen_image_md5(source)
        if detection.get('image') != image_path.name or detection.get('image_md5') != image_md5:
            raise _error(case_id, 'exact-frame evidence does not identify the frozen image')
        if detection.get('dimensions') != source['dimensions']:
            raise _error(case_id, 'exact-frame evidence dimensions differ from the frozen image')
        boxes = _as_boxes(detection.get('bboxes'), f'{case_id}.evidence.detection.bboxes')
        scores = np.asarray(detection.get('scores'), dtype=float)
        if scores.ndim != 1 or len(scores) != len(boxes):
            raise _error(case_id, 'exact-frame evidence scores and boxes differ in length')
        expected_boxes = eligible_boxes(boxes, scores, f'{case_id}.evidence.detection')
        _validate_box_scale(boxes, cast(Mapping[str, Any], source['dimensions']), f'{case_id}.evidence.detection')
    else:
        raise _error(case_id, f'unsupported repair mode {mode!r}')
    manifest_boxes = _as_boxes(entry.get('mask_boxes'), f'{case_id}.mask_boxes')
    if not np.array_equal(manifest_boxes, expected_boxes):
        raise _error(case_id, 'manifest mask boxes differ from frozen evidence')
    expected_kept = midpoint_kept_ids(source['segments_px'], expected_boxes)
    if entry.get('kept_fragment_ids') != expected_kept:
        raise _error(case_id, 'manifest kept IDs do not follow its frozen mask evidence')
    _validate_input_files(inputs_dir, source, entry)


def validate_repair_inputs(case_ids: Sequence[str], inputs_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Preflight every selected case before importing the matcher zone."""
    for case_id in case_ids:
        validate_repair_case(case_id, inputs_dir, manifest)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir', type=Path, required=True)
    parser.add_argument('--repair-manifest', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=list(REPAIR_CASE_IDS))
    parser.add_argument('--exact-detections', type=Path)
    args = parser.parse_args()
    build_repair_inputs(args.inputs_dir, args.repair_manifest, args.cases, args.exact_detections)


if __name__ == '__main__':
    main()
