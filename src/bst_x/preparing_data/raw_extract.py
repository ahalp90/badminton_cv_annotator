"""Raw rtmlib pose extraction for a specified subset of clips.

Sibling to ``prepare_train_on_shuttleset.py``'s 2D pose step, but:

1. Operates only on the clip stems in a supplied list (e.g. the Phase 1
   "busted" set from ``validation_scripts/mmpose_heuristic_investigation/find_busted_clips.py``).
2. Applies no filtering -- no court projection, no "2 players on court"
   requirement, no normalization. Saves every detection the adapter returns.
3. Emits five raw numpy arrays per clip:

   - ``{stem}_raw_kps.npy``        ``(F, N_max, J, 2)``  float32, NaN-padded
   - ``{stem}_raw_bboxes.npy``     ``(F, N_max, 4)``     float32, NaN-padded
   - ``{stem}_raw_scores.npy``     ``(F, N_max)``        float32, NaN-padded
   - ``{stem}_raw_kp_scores.npy``  ``(F, N_max, J)``     float32, NaN-padded
   - ``{stem}_raw_ndet.npy``       ``(F,)``              int8 detection count

``_raw_ndet.npy`` is the resume marker, saved last; its presence means all
five outputs landed cleanly for this clip. NaN padding is used (not zero)
so real detected coordinates at origin are not ambiguous with padding.

A clip that decodes zero frames (unreadable, truncated, or empty mp4) writes
no npys, is logged to ``{save_dir}/failed_clips.log`` (append mode), and is
skipped. The batch aborts loudly once failures exceed 0.3 of the clips slated
for extraction this run (resume-skipped clips are excluded from that count).

The raw outputs feed downstream heuristic iteration (``apply_heuristic.py``
and the ``sticky_anchor`` variant, both out of scope for this module).

A 3D extraction path (via ``MMPoseInferencer(pose3d="human3d")``) is
deliberately out of scope for this module's current phase. The 3D stream was
removed from the tree; its design and the per-clip MMPose reload workaround
are recorded in
``docs/archive/completed_general_refactors/structure_and_guards_pass/pose_3d_stream_design.md``
for revival.

Run from the repo root with both package roots on PYTHONPATH::

    PYTHONPATH=src:src/bst_x \\
        python -m preparing_data.raw_extract --help
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from tqdm import tqdm

from pipeline.clip_index import build_clip_path_index
from pipeline.config import CLIPS_OUTPUT_DIR, COCO_N_JOINTS
from preparing_data.extract_failures import (
    FAILURE_ABORT_FRACTION,
    failed_clips_log_path,
    log_failed_clip,
)
from preparing_data.heuristics.base import RAW_SUFFIXES

if TYPE_CHECKING:  # runtime import is lazy (in main) so this module loads without rtmlib
    from preparing_data.rtmlib_pose import FrameDetections, RtmlibPoseExtractor


class PaddedRawFrame(NamedTuple):
    """One frame's raw arrays, NaN-padded to ``n_max`` along the detection dim,
    as saved by ``extract_one_clip``.
    """

    kps: np.ndarray  # (n_max, J, 2) float32, NaN-padded
    bboxes: np.ndarray  # (n_max, 4) float32, NaN-padded
    scores: np.ndarray  # (n_max,) float32, NaN-padded
    kp_scores: np.ndarray  # (n_max, J) float32, NaN-padded
    n_dets: int  # real per-frame detection count; rows [0, n_dets) are real, the rest NaN padding


def extract_raw_frame(
    det: FrameDetections,
    n_max: int,
    clip_stem: str,
    frame_num: int,
    over_det_warned: set[str],
) -> PaddedRawFrame:
    """Return per-frame raw arrays, NaN-padded to ``n_max`` along the detect dim.

    If the adapter returns more than ``n_max`` detections in a frame, keep the
    top-``n_max`` by ``bbox_score`` (descending; stable, so ties hold detector
    order). Otherwise all ``n`` detections (the per-frame detection count)
    keep detector order. Log a once-per-clip warning on truncation.

    :param det: one frame's adapter detections (``n`` real people, detector order).
    :param n_max: per-frame detection cap.
    :param clip_stem: clip id, for the once-per-clip over-detection warning.
    :param frame_num: frame index, for the warning message.
    :param over_det_warned: set of already-warned clip stems (mutated in place).
    :return: a :class:`PaddedRawFrame` -- ``kps``, ``bboxes``, ``scores``,
        ``kp_scores`` (four NaN-padded ``(n_max, ...)`` float32 arrays) and
        ``n_dets`` (the real detection count).
    """
    n = len(det.keypoints)
    order = np.arange(n)  # (n,) detector order
    if n > n_max:
        order = np.argsort(-det.bbox_scores, kind="stable")[:n_max]  # top-n_max by bbox_score
        n = n_max
        if clip_stem not in over_det_warned:
            print(
                f"  WARN: {clip_stem} frame {frame_num} had >{n_max} detections; "
                f"truncating to top-{n_max} by bbox_score"
            )
            over_det_warned.add(clip_stem)

    kps = np.full((n_max, COCO_N_JOINTS, 2), np.nan, dtype=np.float32)
    bboxes = np.full((n_max, 4), np.nan, dtype=np.float32)
    scores = np.full((n_max,), np.nan, dtype=np.float32)
    kp_scores = np.full((n_max, COCO_N_JOINTS), np.nan, dtype=np.float32)

    kps[:n] = det.keypoints[order]
    bboxes[:n] = det.bboxes[order]
    scores[:n] = det.bbox_scores[order]
    kp_scores[:n] = det.kp_scores[order]

    return PaddedRawFrame(kps, bboxes, scores, kp_scores, n)


def extract_one_clip(
    extractor: RtmlibPoseExtractor,
    video_path: Path,
    save_branch: str,
    n_max: int,
    over_det_warned: set[str],
) -> bool:
    """Run the rtmlib adapter on one clip and save the five raw arrays.

    :return: ``True`` if the clip decoded at least one frame and all five arrays
        were saved; ``False`` if it decoded zero frames (unreadable, truncated,
        or empty mp4). On ``False`` nothing is written, so the caller can log +
        skip the clip and the resume logic retries it next run.
    """
    kps_ls: list[np.ndarray] = []
    bboxes_ls: list[np.ndarray] = []
    scores_ls: list[np.ndarray] = []
    kp_scores_ls: list[np.ndarray] = []
    ndet_ls: list[int] = []

    for frame_num, det in enumerate(extractor.iter_video(video_path)):
        kps, bboxes, scores, kp_scores, n = extract_raw_frame(
            det, n_max, video_path.stem, frame_num, over_det_warned,
        )
        kps_ls.append(kps)
        bboxes_ls.append(bboxes)
        scores_ls.append(scores)
        kp_scores_ls.append(kp_scores)
        ndet_ls.append(n)

    if not ndet_ls:
        # Zero frames decoded: np.stack([]) would crash. Signal the caller to
        # log + skip instead. Writing no npys keeps resume able to retry.
        return False

    np.save(save_branch + "_raw_kps.npy", np.stack(kps_ls))
    np.save(save_branch + "_raw_bboxes.npy", np.stack(bboxes_ls))
    np.save(save_branch + "_raw_scores.npy", np.stack(scores_ls))
    np.save(save_branch + "_raw_kp_scores.npy", np.stack(kp_scores_ls))
    # _raw_ndet.npy is saved last so its presence is a reliable resume marker
    # for all five outputs.
    np.save(save_branch + "_raw_ndet.npy", np.asarray(ndet_ls, dtype=np.int8))
    return True


def inspect_one_clip(extractor: RtmlibPoseExtractor, video_path: Path) -> None:
    """Print the first frame's adapter detections (shapes/dtypes), then return."""
    print(f"Inspect: {video_path}")
    det = next(extractor.iter_video(video_path), None)
    if det is None:
        print("No frames decoded; try a different clip.")
        return
    n_dets = len(det.keypoints)
    print(f"Number of detections in frame 0: {n_dets}")
    if n_dets == 0:
        print("No detections in frame 0; try a different clip.")
        return
    for name, arr in (
        ("keypoints", det.keypoints),
        ("bboxes", det.bboxes),
        ("bbox_scores", det.bbox_scores),
        ("kp_scores", det.kp_scores),
    ):
        print(f"  {name!r}: dtype={arr.dtype} shape={arr.shape}")
    print("\nDetection[0]:")
    pprint({
        "bbox": det.bboxes[0],
        "bbox_score": float(det.bbox_scores[0]),
        "keypoints[:3]": det.keypoints[0, :3],
        "kp_scores[:3]": det.kp_scores[0, :3],
    })


def load_stems(path: Path) -> list[str]:
    with path.open() as fh:
        return [line.strip() for line in fh if line.strip()]


def _stored_n_max(save_branch: str) -> int | None:
    """Peek at `_raw_bboxes.npy` to recover the N_max dimension of an existing
    extract, or return None if the file is absent or unreadable.
    """
    path = Path(save_branch + "_raw_bboxes.npy")
    if not path.exists():
        return None
    try:
        return int(np.load(path, mmap_mode="r").shape[1])
    except (OSError, ValueError, IndexError):
        return None


def _clear_raw_files(save_branch: str) -> None:
    """Delete all five raw outputs for a given stem, if present."""
    for suffix in RAW_SUFFIXES:
        Path(save_branch + suffix).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--clips-dir", type=Path, default=CLIPS_OUTPUT_DIR,
        help="Root of raw .mp4 clips (scanned recursively). "
             "Defaults to pipeline.config.CLIPS_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--clip-stems-file", type=Path, required=True,
        help="One clip stem per line (output of find_busted_clips.py).",
    )
    parser.add_argument(
        "--save-dir", type=Path, required=True,
        help="Output dir for raw per-clip .npy files. Must not collide with "
             "the primary filtered flat dir.",
    )
    parser.add_argument(
        "--n-max", type=int, default=16,
        help="Max detections per frame. Excess is truncated by bbox_score. "
             "Default 16 matches the committed baseline schema (N_max=16).",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="onnxruntime device for the rtmlib adapter: 'cuda' (default, needs "
             "onnxruntime-gpu) or 'cpu'.",
    )
    parser.add_argument(
        "--inspect-result", action="store_true",
        help="Print the first frame's adapter detections on one clip, then exit. "
             "Run this once before any batch.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve stems to mp4 paths and exit without loading the adapter.",
    )
    parser.add_argument(
        "--force-reextract", action="store_true",
        help="If an existing clip's stored N_max differs from --n-max, "
             "delete its five raw files and re-extract. Without this flag, "
             "a shape mismatch is a hard error so we can't silently mix "
             "N_max widths in the same save-dir.",
    )
    args = parser.parse_args()

    if not args.clips_dir.is_dir():
        parser.error(f"clips-dir not found: {args.clips_dir}")
    if not args.clip_stems_file.exists():
        parser.error(f"clip-stems-file not found: {args.clip_stems_file}")
    if args.n_max > 127:
        parser.error(
            f"--n-max {args.n_max} exceeds 127: per-frame detection counts are "
            f"saved as int8 (_raw_ndet.npy), so a larger cap would silently wrap"
        )

    stems = load_stems(args.clip_stems_file)
    print(f"Loaded {len(stems)} stems from {args.clip_stems_file}")

    stem_to_path = build_clip_path_index(args.clips_dir)
    print(f"Indexed {len(stem_to_path)} mp4 files under {args.clips_dir}")

    resolved: list[tuple[str, Path]] = []
    missing: list[str] = []
    for stem in stems:
        path = stem_to_path.get(stem)
        if not path:
            missing.append(stem)
        else:
            resolved.append((stem, path))
    print(f"Resolved {len(resolved)} / {len(stems)} stems to mp4 paths")
    if missing:
        print(f"  Missing (first 10): {missing[:10]}")

    if args.dry_run:
        print("\nDry run: showing first 5 resolved pairs and exiting.")
        for stem, path in resolved[:5]:
            print(f"  {stem}  ->  {path}")
        return 0

    # Lazy import: keeps extract_raw_frame and the file's helpers importable
    # without onnxruntime (e.g. the CPU raw-schema gate), mirroring the 2D path.
    from preparing_data.rtmlib_pose import RtmlibPoseExtractor

    if args.inspect_result:
        if not resolved:
            print("No resolved clips to inspect; aborting.")
            return 1
        extractor = RtmlibPoseExtractor(device=args.device)
        inspect_one_clip(extractor, resolved[0][1])
        return 0

    args.save_dir.mkdir(parents=True, exist_ok=True)

    extractor = RtmlibPoseExtractor(device=args.device)
    over_det_warned: set[str] = set()

    # Classification pre-pass: decide skip / extract / extract-after-clear per
    # resolved clip WITHOUT deleting anything yet. Doing the N_max-mismatch hard
    # error here means it fires before any extraction starts (previously it
    # fired mid-loop, after some clips were already written). The just-in-time
    # clear then happens in the extraction loop below.
    #
    # `to_extract` fixes the abort denominator up front: resume-skipped clips
    # are never appended, so they can't dilute the 0.3 failure fraction.
    skipped = 0
    reextracted_mismatch = 0
    to_extract: list[tuple[str, Path, bool]] = []  # (stem, video_path, needs_clear)

    for stem, video_path in resolved:
        save_branch = str(args.save_dir / stem)
        ndet_path = Path(save_branch + "_raw_ndet.npy")
        if not ndet_path.exists():
            to_extract.append((stem, video_path, False))
            continue
        stored = _stored_n_max(save_branch)
        if stored is None:
            # _raw_ndet.npy present but bboxes missing or unreadable. Treat as a
            # corrupted leftover: re-extract from scratch (cleared just-in-time).
            to_extract.append((stem, video_path, True))
        elif stored == args.n_max:
            skipped += 1
        elif args.force_reextract:
            reextracted_mismatch += 1
            to_extract.append((stem, video_path, True))
        else:
            print(
                f"\nERROR: existing output for {stem} has N_max={stored} "
                f"but --n-max={args.n_max}. Rerun with --force-reextract "
                f"to delete and re-extract mismatched clips, or clear the "
                f"save-dir manually."
            )
            return 1

    n_slated = len(to_extract)
    abort_threshold = FAILURE_ABORT_FRACTION * n_slated
    failures = 0

    for stem, video_path, needs_clear in tqdm(to_extract, desc="raw_extract", unit="clip"):
        save_branch = str(args.save_dir / stem)
        if needs_clear:
            _clear_raw_files(save_branch)

        if not extract_one_clip(
            extractor=extractor,
            video_path=video_path,
            save_branch=save_branch,
            n_max=args.n_max,
            over_det_warned=over_det_warned,
        ):
            failures += 1
            log_failed_clip(args.save_dir, stem, "decoded 0 frames")
            if failures > abort_threshold:
                print(
                    f"\nERROR: {failures} clip(s) failed to decode, exceeding "
                    f"{FAILURE_ABORT_FRACTION:.0%} of the {n_slated} slated for "
                    f"extraction. Aborting; see "
                    f"{failed_clips_log_path(args.save_dir)} for the failed stems."
                )
                return 1
            continue

        # Per-clip cleanup to limit peak memory across the batch. onnxruntime
        # manages its own device memory, so (unlike the old mmpose/torch path)
        # there is no CUDA cache to clear -- gc.collect() suffices.
        gc.collect()

    print(
        f"\nDone. Extracted {n_slated - failures}, skipped {skipped} "
        f"(had _raw_ndet.npy). Missing mp4 for {len(missing)} stems."
    )
    if failures:
        print(
            f"  {failures} clip(s) decoded zero frames and were skipped; "
            f"see {failed_clips_log_path(args.save_dir)}."
        )
    if reextracted_mismatch:
        print(
            f"  Re-extracted {reextracted_mismatch} clip(s) whose stored "
            f"N_max differed from --n-max={args.n_max} (--force-reextract)."
        )
    if over_det_warned:
        print(
            f"Over-detection warnings fired for {len(over_det_warned)} clip(s) "
            f"(frames with >{args.n_max} detections, truncated to top "
            f"{args.n_max} by bbox_score):"
        )
        for stem in sorted(over_det_warned):
            print(f"  {stem}")
    else:
        print("No over-detection warnings fired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
