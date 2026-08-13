"""Validate a completed shard run and publish the canonical five raw arrays.

The stitcher trusts nothing about the run directory: every check below guards
a plausible silent-corruption route (missing/short/stale/mixed shards, gap or
overlap, schema or ``n_max`` drift, source mismatch, partial writes).

Publication mirrors production semantics: the five arrays land as plain
``{stem}_raw_*.npy`` (the existing downstream contract needs the canonical
uncompressed form), written atomically with ``_raw_ndet.npy`` **last** — the
same "ndet present means all five landed" resume marker raw_extract uses. A
stitch that fails at any point leaves no ndet file, so the output never looks
complete.

The published stem must begin with the numeric video id (``21_full``, not
``sset_21``): ``apply_heuristic._vid_from_stem`` parses the id from the first
``_``-separated token and silently skips stems that don't parse.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
from pipeline.config import COCO_N_JOINTS
from preparing_data.heuristics.base import RAW_SUFFIXES

from shared.video_sharding.shard_worker import (
    ARRAY_KINDS,
    load_gz_json,
    load_npy_xz,
    save_gz_json,
    shard_stem,
)

RUN_MANIFEST_NAME = "run_manifest.json.gz"


class StitchError(RuntimeError):
    """A shard run failed validation; nothing was published."""


def write_run_manifest(run_dir: Path, manifest: dict) -> None:
    save_gz_json(run_dir / RUN_MANIFEST_NAME, manifest)


def expected_array_specs(span: int, n_max: int) -> dict[str, tuple[tuple[int, ...], str]]:
    """Shape/dtype every shard (or the stitched whole) must satisfy."""
    return {
        "raw_kps": ((span, n_max, COCO_N_JOINTS, 2), "float32"),
        "raw_bboxes": ((span, n_max, 4), "float32"),
        "raw_scores": ((span, n_max), "float32"),
        "raw_kp_scores": ((span, n_max, COCO_N_JOINTS), "float32"),
        "raw_ndet": ((span,), "int8"),
    }


def validate_plan(plan: list[tuple[int, int]], n_frames: int) -> None:
    """The plan must tile [0, n_frames) contiguously in canonical frame order."""
    if not plan:
        raise StitchError("empty shard plan")
    for start, end in plan:
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise StitchError(f"plan has an invalid range [{start}, {end})")
    ordered = sorted(plan)
    if plan != ordered:
        raise StitchError("plan ranges are not in ascending frame order")
    if ordered[0][0] != 0:
        raise StitchError(f"plan does not start at frame 0: {ordered[0]}")
    for (_, prev_end), (start, end) in itertools.pairwise(ordered):
        if start != prev_end:
            kind = "overlap" if start < prev_end else "gap"
            raise StitchError(f"plan has a {kind} at frame {min(start, prev_end)}")
    if ordered[-1][1] != n_frames:
        raise StitchError(
            f"plan ends at {ordered[-1][1]}, expected n_frames={n_frames}"
        )


def stitch_and_publish(run_dir: Path, publish_dir: Path, stem: str) -> Path:
    """Validate every shard against the run manifest, concatenate, publish.

    :return: ``publish_dir``, now containing the five ``{stem}_raw_*.npy``.
    :raises StitchError: on any inconsistency; nothing is published.
    """
    run_manifest_path = run_dir / RUN_MANIFEST_NAME
    if not run_manifest_path.exists():
        raise StitchError(f"missing {RUN_MANIFEST_NAME} in {run_dir}")
    run = load_gz_json(run_manifest_path)
    n_frames, n_max, run_id = run["n_frames"], run["n_max"], run["run_id"]
    plan = [tuple(r) for r in run["plan"]]
    validate_plan(plan, n_frames)
    if run["n_shards"] != len(plan):
        raise StitchError(
            f"run records n_shards={run['n_shards']}, but its plan has {len(plan)} ranges"
        )

    # Refuse unplanned shard artefacts: a stale shard from an earlier layout
    # of the same directory must fail loudly, not sit ignored next to a
    # "clean" publication.
    planned_manifests = {f"{shard_stem(s, e)}_manifest.json.gz" for s, e in plan}
    present_manifests = {p.name for p in run_dir.glob("shard_*_manifest.json.gz")}
    unplanned = sorted(present_manifests - planned_manifests)
    if unplanned:
        raise StitchError(f"unplanned shard manifests in {run_dir}: {unplanned}")

    shard_arrays: list[dict[str, np.ndarray]] = []
    for start, end in plan:
        stem_se = shard_stem(start, end)
        manifest_path = run_dir / f"{stem_se}_manifest.json.gz"
        if not manifest_path.exists():
            raise StitchError(
                f"shard [{start}, {end}) incomplete: missing {manifest_path.name} "
                f"(worker failed or never ran)"
            )
        manifest = load_gz_json(manifest_path)
        if manifest["run_id"] != run_id:
            raise StitchError(
                f"shard [{start}, {end}) belongs to run {manifest['run_id']}, "
                f"expected {run_id}: stale or mixed run"
            )
        if manifest["source_md5"] != run["source_md5"]:
            raise StitchError(f"shard [{start}, {end}) extracted from a different source video")
        if manifest["n_max"] != n_max:
            raise StitchError(
                f"shard [{start}, {end}) has n_max={manifest['n_max']}, run expects {n_max}"
            )
        for field in ("extractor", "decode_mode"):
            if manifest[field] != run[field]:
                raise StitchError(
                    f"shard [{start}, {end}) {field}={manifest[field]!r}, "
                    f"run expects {run[field]!r}"
                )
        if (manifest["start"], manifest["end"]) != (start, end):
            raise StitchError(f"shard manifest {manifest_path.name} disagrees with its filename range")
        if manifest["frames_read"] != end - start:
            raise StitchError(
                f"shard [{start}, {end}) read {manifest['frames_read']} frames, expected {end - start}"
            )

        expected = expected_array_specs(end - start, n_max)
        arrays: dict[str, np.ndarray] = {}
        for kind in ARRAY_KINDS:
            array_path = run_dir / f"{stem_se}_{kind}.npy.xz"
            if not array_path.exists():
                raise StitchError(f"shard [{start}, {end}) missing array file {array_path.name}")
            array = load_npy_xz(array_path)
            want_shape, want_dtype = expected[kind]
            if array.shape != want_shape or str(array.dtype) != want_dtype:
                raise StitchError(
                    f"{array_path.name}: shape {array.shape} dtype {array.dtype}, "
                    f"expected {want_shape} {want_dtype}"
                )
            arrays[kind] = array
        shard_arrays.append(arrays)

    stitched = {
        kind: np.concatenate([arrays[kind] for arrays in shard_arrays], axis=0)
        for kind in ARRAY_KINDS
    }
    for kind, (want_shape, _) in expected_array_specs(n_frames, n_max).items():
        if stitched[kind].shape != want_shape:
            raise StitchError(
                f"stitched {kind} shape {stitched[kind].shape}, expected {want_shape}"
            )

    publish_dir.mkdir(parents=True, exist_ok=True)
    # RAW_SUFFIXES order matches ARRAY_KINDS; write ndet (the marker) last.
    for kind, suffix in zip(ARRAY_KINDS, RAW_SUFFIXES):
        final = publish_dir / f"{stem}{suffix}"
        tmp = Path(str(final) + ".tmp")
        with tmp.open("wb") as fh:
            np.save(fh, stitched[kind])  # handle, not path: np.save appends .npy to paths
        os.replace(tmp, final)
    return publish_dir
