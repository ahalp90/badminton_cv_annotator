"""Gate: does sharded extraction reproduce the sequential production path?

The sequential control is the *unmodified* production driver
(``raw_extract.extract_one_clip`` -> plain ``{stem}_raw_*.npy``); the sharded
side is ``run_sharded.extract_sharded``. Both consume the same video file.

Comparison separates structural disagreement from numeric drift, per array:

- exact:      byte-equal (NaNs equal); the only acceptable result for
              deterministic extractors (fake, CPU at pinned threads)
- ndet:       frames whose detection count differs
- reorder:    frames where the multisets of rows match but order differs
- numeric:    max |diff| over real-detection rows on frames with equal ndet

``--self-variance`` runs the sequential control twice and compares the two
controls instead — the run-to-run noise floor that any CUDA sharded-vs-
sequential difference must be read against.

For real extractors bound the cost with ``--limit-frames`` (the video is first
cut to that many frames with ffmpeg stream copy into the workdir, so the
production driver, which always reads to EOF, sees the same bounded input).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from preparing_data.heuristics.base import RAW_SUFFIXES
from preparing_data.raw_extract import extract_one_clip

from shared.video_sharding.run_sharded import extract_sharded
from shared.video_sharding.shard_worker import (
    EXTRACTOR_SPECS,
    build_extractor,
)

ARRAY_NAMES = ("kps", "bboxes", "scores", "kp_scores", "ndet")


def cut_first_frames(video: Path, n_frames: int, workdir: Path) -> Path:
    """ffmpeg stream-copy of roughly the first ``n_frames`` frames.

    Stream copy cuts at a keyframe, so the cut length is approximate — that is
    fine: both extraction paths consume the same cut file, and parity is a
    property of the two paths, not of the cut point.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH; needed for --limit-frames bounding")
    cap_seconds = n_frames / 25  # lower-bound fps guess; cut length is approximate anyway
    out = workdir / f"bounded_{n_frames}_{video.name}"
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-t", f"{cap_seconds:.3f}",
             "-c", "copy", str(out)],
            check=True,
        )
    return out


def run_sequential(video: Path, save_dir: Path, stem: str, n_max: int, extractor_spec: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(extractor_spec)
    ok = extract_one_clip(
        extractor=extractor,
        video_path=video,
        save_branch=str(save_dir / stem),
        n_max=n_max,
        over_det_warned=set(),
    )
    if not ok:
        raise RuntimeError(f"sequential control decoded zero frames from {video}")
    return save_dir


def load_five(save_dir: Path, stem: str) -> dict[str, np.ndarray]:
    return {
        name: np.load(save_dir / f"{stem}{suffix}")
        for name, suffix in zip(ARRAY_NAMES, RAW_SUFFIXES)
    }


def compare(seq: dict[str, np.ndarray], shard: dict[str, np.ndarray]) -> bool:
    """Print a per-array comparison; return overall exact equality."""
    all_exact = True
    shape_ok = all(seq[n].shape == shard[n].shape for n in ARRAY_NAMES)
    if not shape_ok:
        for name in ARRAY_NAMES:
            if seq[name].shape != shard[name].shape:
                print(f"ARRAY {name}: SHAPE MISMATCH seq={seq[name].shape} shard={shard[name].shape}")
        print("GATE parity: FAIL (structural)")
        return False

    ndet_diff_frames = int(np.sum(seq["ndet"] != shard["ndet"]))
    for name in ARRAY_NAMES:
        exact = np.array_equal(seq[name], shard[name], equal_nan=(seq[name].dtype != np.int8))
        all_exact &= exact
        line = f"ARRAY {name}: exact={int(exact)}"
        if not exact and name != "ndet":
            same_ndet = seq["ndet"] == shard["ndet"]
            seq_masked, shard_masked = seq[name][same_ndet], shard[name][same_ndet]
            with np.errstate(invalid="ignore"):
                diff = np.abs(seq_masked - shard_masked)
            numeric_max = float(np.nanmax(diff)) if diff.size else 0.0
            frames_differing = int(np.sum(
                np.any(np.reshape(seq_masked != shard_masked, (seq_masked.shape[0], -1))
                       & ~np.reshape(np.isnan(seq_masked) & np.isnan(shard_masked),
                                     (seq_masked.shape[0], -1)), axis=1)
            ))
            # Reorder probe: equal as multisets of rows despite unequal order?
            reordered = 0
            for frame_index in np.nonzero(same_ndet)[0][:2000]:
                a, b = seq[name][frame_index], shard[name][frame_index]
                if not np.array_equal(a, b, equal_nan=True):
                    a_sorted = np.sort(np.nan_to_num(a.reshape(a.shape[0], -1)), axis=0)
                    b_sorted = np.sort(np.nan_to_num(b.reshape(b.shape[0], -1)), axis=0)
                    if np.array_equal(a_sorted, b_sorted):
                        reordered += 1
            line += (f" frames_diff={frames_differing} max_abs_diff={numeric_max:.6g} "
                     f"pure_reorder_frames={reordered}")
        if name == "ndet":
            line += f" ndet_diff_frames={ndet_diff_frames}"
        print(line)
    print(f"GATE parity: {'PASS (exact)' if all_exact else 'NOT EXACT (see distribution above)'}")
    return all_exact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--extractor", choices=EXTRACTOR_SPECS, default="fake")
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--n-max", type=int, default=16)
    parser.add_argument("--decode-mode", choices=("seek", "scan"), default="seek")
    parser.add_argument("--limit-frames", type=int, default=None,
                        help="ffmpeg stream-copy the first ~N frames and compare on that")
    parser.add_argument("--self-variance", action="store_true",
                        help="compare sequential vs sequential (noise floor control)")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    video = args.video
    if args.limit_frames is not None:
        video = cut_first_frames(video, args.limit_frames, args.workdir)
        print(f"bounded input: {video}")

    stem = "0_paritygate"
    seq_dir = run_sequential(video, args.workdir / "seq_a", stem, args.n_max, args.extractor)
    seq = load_five(seq_dir, stem)
    print(f"sequential control: {seq['ndet'].shape[0]} frames")

    if args.self_variance:
        other_dir = run_sequential(video, args.workdir / "seq_b", stem, args.n_max, args.extractor)
        other = load_five(other_dir, stem)
        print("comparing sequential run A vs sequential run B (self-variance control):")
    else:
        publish = extract_sharded(
            video_path=video,
            out_root=args.workdir / "sharded",
            stem=stem,
            n_shards=args.n_shards,
            n_max=args.n_max,
            extractor_spec=args.extractor,
            decode_mode=args.decode_mode,
        )
        other = load_five(publish, stem)
        print(f"sharded run: {other['ndet'].shape[0]} frames, publish={publish}")

    exact = compare(seq, other)
    return 0 if exact else 2


if __name__ == "__main__":
    sys.exit(main())
