"""Write the CSV-side provenance for TrackNetV3 inpainting.

The sidecar is written immediately before the paired CSV by ``predict_video``.
When ``out_csv_file`` does not already exist, a sidecar write failure therefore
prevents that CSV from being written. Batch mode supplies that precondition via
its existing CSV skip. Standalone callers, including the BRIC wrapper, may
write into a populated directory and must treat the CSV absence precondition
as their responsibility because a rerun can otherwise leave a new sidecar
beside an older CSV.

The sidecar is beside the CSV and is keyed by the video stem and stride. Two
strides for one video therefore require separate ``save_dir`` values: one CSV
can otherwise sit beside two sidecars, with one necessarily describing stale
CSV contents.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any


PredictionDict = Mapping[str, Sequence[Any]]


def _build_spans(
    inpaint_mask: Sequence[Any],
    frame_ids: Sequence[Any],
    *,
    inpaint_applied: bool,
) -> list[list[int]]:
    """Return sorted half-open spans for the raw applied-mask switch."""
    if not inpaint_applied:
        return []

    if len(inpaint_mask) != len(frame_ids):
        raise ValueError(
            'The applied inpaint mask and final prediction frame list must have equal lengths'
        )

    frame_ids = list(frame_ids)
    if any(frame_id < 0 for frame_id in frame_ids):
        raise ValueError('Final prediction frame ids must be non-negative')
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError('Final prediction frame ids must be unique')

    ordered_rows = sorted(zip(frame_ids, inpaint_mask), key=lambda row: row[0])
    spans: list[list[int]] = []
    for frame_id, selected in ordered_rows:
        if not selected:
            continue

        start = int(frame_id)
        end = start + 1
        if spans and start == spans[-1][1]:
            spans[-1][1] = end
        else:
            spans.append([start, end])

    return spans


def _read_source_provenance(video_file: str) -> dict[str, Any]:
    """Read validated provenance for ``video_file`` from its own directory."""
    manifest_path = os.path.join(os.path.dirname(video_file), 'sources.toml')
    try:
        with open(manifest_path, 'rb') as manifest_file:
            manifest = tomllib.load(manifest_file)
    except FileNotFoundError:
        return {}

    videos = manifest.get('videos')
    if not isinstance(videos, dict):
        raise TypeError("sources.toml 'videos' must be a table")

    video_basename = os.path.basename(video_file)
    if video_basename not in videos:
        return {}

    entry = videos[video_basename]
    if not isinstance(entry, dict):
        raise TypeError(f"sources.toml entry for {video_basename!r} must be a table")

    dataset = manifest.get('dataset')
    if not isinstance(dataset, str):
        raise TypeError("sources.toml 'dataset' must be a string")
    provenance: dict[str, Any] = {'dataset': dataset}

    if 'video_id' in entry:
        video_id = entry['video_id']
        if isinstance(video_id, bool) or not isinstance(video_id, (int, str)):
            raise TypeError("sources.toml 'video_id' must be an integer or string")
        provenance['video_id'] = video_id

    for key in ('title', 'url'):
        if key not in entry:
            continue
        value = entry[key]
        if not isinstance(value, str):
            raise TypeError(f"sources.toml '{key}' must be a string")
        provenance[key] = value

    if 'fps' in entry:
        fps = entry['fps']
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or fps <= 0
        ):
            raise TypeError("sources.toml 'fps' must be a positive number")
        provenance['fps'] = fps

    return provenance


def _serialise_metadata(metadata: Mapping[str, Any], spans: list[list[int]]) -> str:
    """Serialise the header with one compact span per line."""
    header = json.dumps(metadata, ensure_ascii=False, indent=4)
    if spans:
        span_lines = ',\n'.join(f'        {json.dumps(span)}' for span in spans)
        selected = f'[\n{span_lines}\n    ]'
    else:
        selected = '[]'
    return f'{header[:-1]},\n    "inpaint_selected": {selected}\n}}\n'


def write_inpaint_metadata(
    out_csv_file: str | os.PathLike[str],
    *,
    tracknet_pred_dict: PredictionDict,
    pred_dict: PredictionDict,
    video_file: str | os.PathLike[str],
    eval_mode: str,
    tracknet_seq_len: int,
    h: float,
    inpaintnet: object | None,
    tracknet_ckpt: str | None = None,
    inpaintnet_ckpt: str | None = None,
    input_video_identity: str | os.PathLike[str] | None = None,
) -> None:
    """Write the inpaint fill-mask sidecar beside ``out_csv_file``.

    :param out_csv_file: Destination CSV path. The sidecar is placed beside it.
    :param tracknet_pred_dict: Detector predictions containing the raw mask.
    :param pred_dict: Final predictions whose frame ids index the saved CSV.
    :param video_file: Input video path used for the input basename and manifest.
    :param eval_mode: TrackNet temporal ensemble mode.
    :param tracknet_seq_len: Sequence length from the TrackNet checkpoint.
    :param h: Input video height in pixels.
    :param inpaintnet: Loaded InpaintNet, or ``None`` when disabled.
    :param tracknet_ckpt: TrackNet checkpoint basename, if known.
    :param inpaintnet_ckpt: InpaintNet checkpoint basename, if known.
    :param input_video_identity: Canonical source path recorded in provenance.
    """
    csv_path = os.fspath(out_csv_file)
    video_path = os.fspath(video_file)
    identity_path = video_path if input_video_identity is None else os.fspath(input_video_identity)
    inpaint_applied = inpaintnet is not None
    inpaint_mask = tracknet_pred_dict['Inpaint_Mask']
    frame_ids = pred_dict['Frame']
    spans = _build_spans(inpaint_mask, frame_ids, inpaint_applied=inpaint_applied)
    stride = tracknet_seq_len if eval_mode == 'nonoverlap' else 1
    video_basename = os.path.basename(video_path)
    identity_basename = os.path.basename(identity_path)
    tracknet_ckpt = os.path.basename(tracknet_ckpt) if tracknet_ckpt else None
    inpaintnet_ckpt = os.path.basename(inpaintnet_ckpt) if inpaintnet_ckpt else None

    metadata: dict[str, Any] = {
        'schema': 'inpaint_fill_mask/1',
        'index_space': 'frame',
        'inpaint_status': 'applied' if inpaint_applied else 'disabled',
        'n_rows': len(frame_ids),
        'eval_mode': eval_mode,
        'stride': stride,
        'th_h_px': h * 0.05,
        'tracknet_ckpt': tracknet_ckpt,
        'inpaintnet_ckpt': inpaintnet_ckpt if inpaint_applied else None,
        'input_video': identity_basename,
    }
    metadata.update(_read_source_provenance(identity_path))

    video_name = os.path.splitext(video_basename)[0]
    sidecar_name = f'{video_name}_stride{stride}_inpaint_mask.json.gz'
    sidecar_path = os.path.join(os.path.dirname(csv_path), sidecar_name)
    metadata['extracted_utc'] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    )
    with gzip.open(sidecar_path, 'wt', encoding='utf-8') as sidecar_file:
        sidecar_file.write(_serialise_metadata(metadata, spans))
