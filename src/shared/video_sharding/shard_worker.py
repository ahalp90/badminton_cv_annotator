"""One shard worker: decode ``[start, end)``, extract, persist shard artefacts.

Each worker process owns its own extractor (and therefore its own onnxruntime
session for the real specs) and its own ``cv2.VideoCapture``. Per-frame array
assembly reuses ``raw_extract.extract_raw_frame`` so NaN padding, top-``n_max``
truncation and dtypes stay owned by production code.

Persistence per shard, all writes atomic (``.tmp`` then ``os.replace``):

- five ``shard_{start:08d}_{end:08d}_raw_*.npy.xz`` arrays (lzma, preset 9);
- ``shard_{start:08d}_{end:08d}_manifest.json.gz`` written **last**, so a
  manifest's presence means the whole shard landed. A worker that dies leaves
  no manifest and the stitcher refuses the run.

Failure policy: a short read (stream ended before ``end``) raises — a failed
or truncated decode must never look like a small-but-valid shard. The last
shard additionally probes one frame past ``end`` and raises if it exists, so a
plan built from an undercounting container header cannot silently drop the
tail of the video.

CPU determinism note: as with production extraction, bit-reproducibility on
CPU requires pinning ``OMP_NUM_THREADS`` in the environment before python
starts; workers inherit the invoker's environment.
"""

from __future__ import annotations

import gzip
import json
import lzma
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from preparing_data.raw_extract import extract_raw_frame

from shared.video_sharding.range_decode import iter_frame_range

# Ordered to match heuristics.base.RAW_SUFFIXES / RawClip field order.
ARRAY_KINDS = ("raw_kps", "raw_bboxes", "raw_scores", "raw_kp_scores", "raw_ndet")

EXTRACTOR_SPECS = ("fake", "cpu", "cuda")


def build_extractor(spec: str):
    """Return a detect_frame-capable extractor for ``spec``.

    ``fake`` needs no rtmlib; ``cpu``/``cuda`` lazily import the production
    adapter (matching the lazy-import convention in raw_extract).
    """
    if spec == "fake":
        from shared.video_sharding.fake_pose import (
            DeterministicFakeExtractor,
        )
        return DeterministicFakeExtractor()
    if spec in ("cpu", "cuda"):
        from preparing_data.rtmlib_pose import RtmlibPoseExtractor
        return RtmlibPoseExtractor(device=spec)
    raise ValueError(f"unknown extractor spec {spec!r}; expected one of {EXTRACTOR_SPECS}")


def shard_stem(start: int, end: int) -> str:
    return f"shard_{start:08d}_{end:08d}"


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def save_npy_xz(path: Path, array: np.ndarray) -> None:
    """Stream one NumPy array through an atomic XZ temporary file."""
    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with lzma.open(temporary, "wb", format=lzma.FORMAT_XZ, preset=9) as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_npy_xz(path: Path) -> np.ndarray:
    with lzma.open(path, "rb", format=lzma.FORMAT_XZ) as fh:
        return np.load(fh, allow_pickle=False)


def save_gz_json(path: Path, payload: dict) -> None:
    atomic_write_bytes(path, gzip.compress(json.dumps(payload, indent=1).encode()))


def load_gz_json(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def run_shard(
    *,
    video_path: str,
    start: int,
    end: int,
    n_max: int,
    run_dir: str,
    run_id: str,
    source_md5: str,
    extractor_spec: str,
    decode_mode: str = "seek",
    probe_past_end: bool = False,
) -> None:
    """Extract ``[start, end)`` and persist one shard under ``run_dir``."""
    extractor = build_extractor(extractor_spec)
    stem = shard_stem(start, end)
    span = end - start

    kps_ls, bboxes_ls, scores_ls, kp_scores_ls, ndet_ls = [], [], [], [], []
    over_det_warned: set[str] = set()
    probe_end = end + 1 if probe_past_end else end
    for offset, frame in enumerate(iter_frame_range(video_path, start, probe_end, decode_mode)):
        if offset == span:
            raise RuntimeError(
                f"{stem}: source still has frames past planned end {end}; the "
                f"shard plan undercounts the video, refusing to publish a truncated run"
            )
        kps, bboxes, scores, kp_scores, n = extract_raw_frame(
            extractor.detect_frame(frame), n_max, stem, start + offset, over_det_warned,
        )
        kps_ls.append(kps)
        bboxes_ls.append(bboxes)
        scores_ls.append(scores)
        kp_scores_ls.append(kp_scores)
        ndet_ls.append(n)

    if len(ndet_ls) != span:
        raise RuntimeError(
            f"{stem}: decoded {len(ndet_ls)} frames, expected {span}; refusing to "
            f"write a short shard"
        )

    run_path = Path(run_dir)
    arrays = {
        "raw_kps": np.stack(kps_ls),
        "raw_bboxes": np.stack(bboxes_ls),
        "raw_scores": np.stack(scores_ls),
        "raw_kp_scores": np.stack(kp_scores_ls),
        "raw_ndet": np.asarray(ndet_ls, dtype=np.int8),
    }
    for kind in ARRAY_KINDS:
        save_npy_xz(run_path / f"{stem}_{kind}.npy.xz", arrays[kind])

    manifest = {
        "run_id": run_id,
        "source_md5": source_md5,
        "start": start,
        "end": end,
        "frames_read": len(ndet_ls),
        "n_max": n_max,
        "extractor": extractor_spec,
        "decode_mode": decode_mode,
        "arrays": {
            kind: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            for kind, arr in arrays.items()
        },
    }
    # Manifest last: its presence is the shard's "complete" marker.
    save_gz_json(run_path / f"{stem}_manifest.json.gz", manifest)


def worker_entry(kwargs: dict) -> None:
    """multiprocessing spawn target; a raise here exits the process nonzero."""
    run_shard(**kwargs)
