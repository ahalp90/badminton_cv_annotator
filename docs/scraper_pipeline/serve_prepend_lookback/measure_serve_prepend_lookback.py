"""Measure current serve-prepend evidence without changing annotator output.

Run from the repository root with the external fixture root configured::

    ANNOTATOR_FIXTURES_ROOT=/path/to/fixture/root PYTHONPATH=src \
        python docs/scraper_pipeline/serve_prepend_lookback/measure_serve_prepend_lookback.py \
        --out docs/scraper_pipeline/serve_prepend_lookback/data/run-YYYYMMDD-HHMMSS

The script uses the maintained ``FIXTURES`` and ``build_run_video_inputs`` seams. It runs the
normal committed-mask chain and, by default, the existing all-False-mask counterfactual. The
counterfactual is a measurement variant; no contacts or spans are fed back into production.

CSV and JSON files are gzip-compressed. Compact per-rally NumPy evidence tables are stored by
the native Python ``lzma`` module with XZ/LZMA compression preset 9. Those ``.npy.xz`` files are
normal NumPy ``.npy`` streams wrapped in native XZ compression and must be reloaded through
``lzma.open`` and ``np.load``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
import lzma
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from annotator.calibration.fixtures import FIXTURES, Fixture, fixtures_root
from annotator.calibration.gt_scoring import build_run_video_inputs, canonical_tolerance
from annotator.calibration.scoring import RallyBoundary, classify_all, greedy_match, load_gt_rallies
from annotator.fps_constants import scale_for_fps
from annotator.inpaint_guard import DEGRADED, FABRICATED, NO_FLAG, SUSPECT_FLAT
from annotator.run_video import RunCapture, run_video


WINDOW_SECONDS = 1.0
CLEAN_RUN_BASE30_FRAMES = 5
MASK_MODES = ("committed", "no_replay")

ARRAY_DTYPE = np.dtype([
    ("rally_index", "<i4"),
    ("gt_serve_frame", "<i8"),
    ("span_category_code", "<i1"),
    ("n_gt_strokes", "<i4"),
    ("n_matched_strokes", "<i4"),
    ("serve_matched", "<?"),
    ("later_strokes_matched", "<i4"),
    ("n_raw_candidates_in_window", "<i4"),
    ("n_accepted_candidates_in_window", "<i4"),
    ("nearest_raw_delta", "<i4"),
    ("nearest_accepted_delta", "<i4"),
    ("window_visible_fraction", "<f4"),
    ("window_clean_fraction", "<f4"),
    ("window_fabricated_fraction", "<f4"),
    ("window_suspect_flat_fraction", "<f4"),
    ("window_degraded_fraction", "<f4"),
    ("window_longest_clean_run", "<i4"),
    ("window_raw_mask_fraction", "<f4"),
    ("window_believed_mask_fraction", "<f4"),
    ("serve_on_believed_mask", "<?"),
    ("court_present_at_serve", "<?"),
    ("n_pose_detections", "<i2"),
    ("n_valid_bbox_detections", "<i2"),
    ("n_valid_wrist_detections", "<i2"),
    ("max_bbox_area", "<f4"),
])


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _fraction(values: np.ndarray) -> float:
    return round(float(values.mean()), 6) if len(values) else 0.0


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _segment_stats(
    track: np.ndarray,
    codes: np.ndarray,
    raw_mask: np.ndarray,
    believed_mask: np.ndarray,
    court_present: np.ndarray,
    start: int,
    end: int,
) -> dict[str, Any]:
    """Return compact evidence for the half-open frame interval ``[start, end)``."""
    start = max(0, start)
    end = min(len(track), end)
    if end <= start:
        return {
            "frame_count": 0,
            "visible_frames": 0,
            "clean_visible_frames": 0,
            "visible_fraction": 0.0,
            "clean_fraction": 0.0,
            "fabricated_fraction": 0.0,
            "suspect_flat_fraction": 0.0,
            "degraded_fraction": 0.0,
            "longest_clean_run": 0,
            "raw_mask_fraction": 0.0,
            "believed_mask_fraction": 0.0,
            "believed_mask_frames": 0,
            "court_present_fraction": 0.0,
        }

    track_slice = track[start:end]
    code_slice = codes[start:end]
    raw_slice = raw_mask[start:end]
    believed_slice = believed_mask[start:end]
    court_slice = court_present[start:end]
    visible = track_slice[:, 2] == 1
    clean_visible = visible & (code_slice == NO_FLAG)
    return {
        "frame_count": end - start,
        "visible_frames": int(visible.sum()),
        "clean_visible_frames": int(clean_visible.sum()),
        "visible_fraction": _fraction(visible),
        "clean_fraction": _fraction(clean_visible),
        "fabricated_fraction": _fraction(code_slice == FABRICATED),
        "suspect_flat_fraction": _fraction(code_slice == SUSPECT_FLAT),
        "degraded_fraction": _fraction(code_slice == DEGRADED),
        "longest_clean_run": _longest_true_run(clean_visible),
        "raw_mask_fraction": _fraction(raw_slice),
        "believed_mask_fraction": _fraction(believed_slice),
        "believed_mask_frames": int(believed_slice.sum()),
        "court_present_fraction": _fraction(court_slice),
    }


def _pose_stats(
    bboxes: np.ndarray, kps: np.ndarray, ndet: np.ndarray, frame: int,
) -> dict[str, Any]:
    """Summarise raw pose availability at one frame without choosing a person."""
    n_pose = max(0, min(int(ndet[frame]), bboxes.shape[1]))
    frame_boxes = bboxes[frame, :n_pose]
    finite_boxes = np.isfinite(frame_boxes).all(axis=1)
    widths = frame_boxes[:, 2] - frame_boxes[:, 0]
    heights = frame_boxes[:, 3] - frame_boxes[:, 1]
    valid_boxes = finite_boxes & (widths > 0) & (heights > 0)
    areas = widths * heights
    max_area = float(areas[valid_boxes].max()) if valid_boxes.any() else -1.0
    if kps.shape[2] >= 11:
        frame_kps = kps[frame, :n_pose, [9, 10], :]
        valid_wrists = np.isfinite(frame_kps).all(axis=(1, 2))
    else:
        valid_wrists = np.zeros(n_pose, dtype=bool)
    return {
        "n_pose_detections": n_pose,
        "n_valid_bbox_detections": int(valid_boxes.sum()),
        "n_valid_wrist_detections": int(valid_wrists.sum()),
        "max_bbox_area": round(max_area, 3),
    }


def _nearest(frames: list[int], target: int) -> tuple[int | None, int | None]:
    if not frames:
        return None, None
    frame = min(frames, key=lambda value: (abs(value - target), value))
    return frame, abs(frame - target)


def _overlapping_span_ids(extent: tuple[int, int], spans: list[tuple[int, int]]) -> list[int]:
    first, last = extent
    return [span_id for span_id, (start, end) in enumerate(spans) if start <= last and first < end]


def _span_code(category: RallyBoundary) -> int:
    return {
        RallyBoundary.COVERED: 0,
        RallyBoundary.SPLIT: 1,
        RallyBoundary.MISSED: 2,
    }[category]


def _status(serve_matched: bool, n_matched: int) -> str:
    if serve_matched:
        return "serve_matched"
    if n_matched:
        return "serve_missed_later_strokes_matched"
    return "whole_rally_missed"


def _measure_variant(fixture: Fixture, mask_mode: str) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    inputs = build_run_video_inputs(fixture)
    track, bboxes, _scores, kps, ndet = inputs.positional
    keyword = dict(inputs.keyword)
    if mask_mode == "no_replay":
        keyword["raw_exclusion_mask"] = np.zeros(len(track), dtype=bool)
    capture = RunCapture()
    keyword["capture"] = capture
    keyword["rejection_diagnostics"] = []
    result = run_video(*inputs.positional, **keyword)

    if capture.raw_exclusion_mask is None or capture.definitive_exclusion_mask is None:
        raise RuntimeError(f"{fixture.name}/{mask_mode}: run did not capture replay masks")
    raw_mask = capture.raw_exclusion_mask
    believed_mask = capture.definitive_exclusion_mask
    codes = np.asarray(inputs.keyword["inpaint_codes"])
    court_present = np.asarray(inputs.keyword["court_present"])
    gt_rallies = load_gt_rallies(inputs.master, fixture.video_id)
    classifications = classify_all(result.spans, gt_rallies)
    tolerance = canonical_tolerance(fixture.fps)
    constants = scale_for_fps(fixture.fps)
    clean_run_min_frames = max(
        1, math.floor(CLEAN_RUN_BASE30_FRAMES * fixture.fps / 30.0 + 0.5),
    )
    raw_frames = sorted(int(contact.contact_frame) for contact in result.contacts)
    accepted_frames = sorted(int(contact.contact_frame) for contact in result.filtered_contacts)
    rows: list[dict[str, Any]] = []

    for rally_index, (rally, (category, span_id)) in enumerate(zip(gt_rallies, classifications)):
        overlapping = _overlapping_span_ids(rally.extent, result.spans)
        accepted_for_rally = [
            int(contact.contact_frame)
            for contact in result.filtered_contacts
            if contact.rally_id in overlapping
        ]
        matched = greedy_match(rally.stroke_frames, accepted_for_rally, tolerance)
        matched_gt = {gt_index for gt_index, _ in matched}
        serve_matched = 0 in matched_gt
        later_strokes_matched = sum(gt_index > 0 for gt_index in matched_gt)
        serve_frame = rally.stroke_frames[0]
        first_accepted_frame, _ = _nearest(accepted_for_rally, serve_frame)
        lookback_anchor = first_accepted_frame if first_accepted_frame is not None else serve_frame
        anchor_source = (
            "first_assigned_accepted_contact"
            if first_accepted_frame is not None else "gt_serve_fallback"
        )
        lookback_start = max(0, lookback_anchor - constants.serve_start_lookback_frames)
        window_half = int(round(WINDOW_SECONDS * fixture.fps))
        window_start = max(0, serve_frame - window_half)
        window_end = min(len(track), serve_frame + window_half + 1)
        window = _segment_stats(
            track, codes, raw_mask, believed_mask, court_present, window_start, window_end,
        )
        lookback = _segment_stats(
            track, codes, raw_mask, believed_mask, court_present, lookback_start, lookback_anchor,
        )
        raw_frame, raw_delta = _nearest(raw_frames, serve_frame)
        accepted_frame, accepted_delta = _nearest(accepted_frames, serve_frame)
        raw_in_window = [frame for frame in raw_frames if window_start <= frame < window_end]
        accepted_in_window = [frame for frame in accepted_frames if window_start <= frame < window_end]
        pose = _pose_stats(bboxes, kps, ndet, serve_frame)
        span_start = result.spans[span_id][0] if span_id is not None else None
        span_end = result.spans[span_id][1] if span_id is not None else None
        row = {
            "mask_mode": mask_mode,
            "rally_index": rally_index,
            "set_id": rally.set_id,
            "rally_number": rally.rally,
            "status": _status(serve_matched, len(matched)),
            "span_category": category.value,
            "span_id": span_id if span_id is not None else "",
            "span_start": span_start if span_start is not None else "",
            "span_end": span_end if span_end is not None else "",
            "gt_first_contact": rally.extent[0],
            "gt_last_contact": rally.extent[1],
            "gt_serve_frame": serve_frame,
            "n_gt_strokes": rally.n_strokes,
            "n_matched_strokes": len(matched),
            "serve_matched": serve_matched,
            "later_strokes_matched": later_strokes_matched,
            "tolerance_frames": tolerance,
            "nearest_raw_frame": raw_frame if raw_frame is not None else "",
            "nearest_raw_delta": raw_delta if raw_delta is not None else "",
            "nearest_accepted_frame": accepted_frame if accepted_frame is not None else "",
            "nearest_accepted_delta": accepted_delta if accepted_delta is not None else "",
            "n_raw_candidates_in_window": len(raw_in_window),
            "n_accepted_candidates_in_window": len(accepted_in_window),
            "window_start": window_start,
            "window_end_exclusive": window_end,
            "window_frame_count": window["frame_count"],
            "window_visible_frames": window["visible_frames"],
            "window_clean_visible_frames": window["clean_visible_frames"],
            "window_visible_fraction": window["visible_fraction"],
            "window_clean_fraction": window["clean_fraction"],
            "window_fabricated_fraction": window["fabricated_fraction"],
            "window_suspect_flat_fraction": window["suspect_flat_fraction"],
            "window_degraded_fraction": window["degraded_fraction"],
            "window_longest_clean_run": window["longest_clean_run"],
            "window_raw_mask_fraction": window["raw_mask_fraction"],
            "window_believed_mask_fraction": window["believed_mask_fraction"],
            "window_believed_mask_frames": window["believed_mask_frames"],
            "window_court_present_fraction": window["court_present_fraction"],
            "lookback_anchor_frame": lookback_anchor,
            "lookback_anchor_source": anchor_source,
            "lookback_start": lookback_start,
            "lookback_end_exclusive": lookback_anchor,
            "lookback_frame_count": lookback["frame_count"],
            "lookback_visible_frames": lookback["visible_frames"],
            "lookback_clean_visible_frames": lookback["clean_visible_frames"],
            "lookback_longest_clean_run": lookback["longest_clean_run"],
            "lookback_believed_mask_frames": lookback["believed_mask_frames"],
            "serve_on_raw_mask": bool(raw_mask[serve_frame]),
            "serve_on_believed_mask": bool(believed_mask[serve_frame]),
            "court_present_at_serve": bool(court_present[serve_frame]),
            "replay_mask_min_frames": constants.replay_mask_min_frames,
            "serve_start_lookback_frames": constants.serve_start_lookback_frames,
            "clean_run_min_frames": clean_run_min_frames,
            **pose,
        }
        rows.append(row)

    metadata = {
        "fixture": fixture.name,
        "video_id": fixture.video_id,
        "fps": fixture.fps,
        "n_frames": len(track),
        "mask_mode": mask_mode,
        "raw_mask_true_frames": int(raw_mask.sum()),
        "believed_mask_true_frames": int(believed_mask.sum()),
        "n_spans": len(result.spans),
        "n_raw_contacts": len(result.contacts),
        "n_accepted_contacts": len(result.filtered_contacts),
        "replay_mask_min_frames": constants.replay_mask_min_frames,
        "serve_start_lookback_frames": constants.serve_start_lookback_frames,
        "clean_run_min_frames": clean_run_min_frames,
    }
    array = np.empty(len(rows), dtype=ARRAY_DTYPE)
    for index, row in enumerate(rows):
        array[index] = (
            row["rally_index"], row["gt_serve_frame"], _span_code(RallyBoundary(row["span_category"])),
            row["n_gt_strokes"], row["n_matched_strokes"], row["serve_matched"],
            row["later_strokes_matched"], row["n_raw_candidates_in_window"],
            row["n_accepted_candidates_in_window"],
            -1 if row["nearest_raw_delta"] == "" else row["nearest_raw_delta"],
            -1 if row["nearest_accepted_delta"] == "" else row["nearest_accepted_delta"],
            row["window_visible_fraction"],
            row["window_clean_fraction"], row["window_fabricated_fraction"],
            row["window_suspect_flat_fraction"], row["window_degraded_fraction"],
            row["window_longest_clean_run"], row["window_raw_mask_fraction"],
            row["window_believed_mask_fraction"], row["serve_on_believed_mask"],
            row["court_present_at_serve"], row["n_pose_detections"],
            row["n_valid_bbox_detections"], row["n_valid_wrist_detections"],
            row["max_bbox_area"],
        )
    return rows, array, metadata


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    misses = [row for row in rows if row["status"] == "serve_missed_later_strokes_matched"]
    serve_misses = [row for row in rows if not row["serve_matched"]]
    return {
        "n_rallies": len(rows),
        "n_serve_matched": sum(bool(row["serve_matched"]) for row in rows),
        "n_serve_missed": len(serve_misses),
        "n_serve_missed_later_strokes_matched": len(misses),
        "n_whole_rally_missed": sum(row["status"] == "whole_rally_missed" for row in rows),
        "n_clean_serve_window_runs": sum(
            row["window_longest_clean_run"] >= row["clean_run_min_frames"] for row in misses
        ),
        "n_clean_lookback_runs": sum(
            row["lookback_longest_clean_run"] >= row["clean_run_min_frames"] for row in misses
        ),
        "n_serve_frames_on_believed_mask": sum(row["serve_on_believed_mask"] for row in misses),
        "n_raw_candidates_near_serve": sum(
            row["nearest_raw_delta"] != "" and row["nearest_raw_delta"] <= row["tolerance_frames"]
            for row in misses
        ),
        "n_accepted_candidates_near_serve": sum(
            row["nearest_accepted_delta"] != "" and row["nearest_accepted_delta"] <= row["tolerance_frames"]
            for row in misses
        ),
        "span_categories": {
            category: sum(row["span_category"] == category for row in rows)
            for category in ("covered", "split", "missed")
        },
    }


def _write_csv_gz(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames or sum(1 for _ in reader) != len(rows):
            raise RuntimeError(f"gzip CSV reload changed schema or row count for {path}")


def _write_json_gz(value: object, path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        if json.load(handle) != value:
            raise RuntimeError(f"gzip JSON reload changed value for {path}")


def _write_array_lzma(array: np.ndarray, path: Path) -> None:
    with lzma.open(path, "wb", preset=9) as handle:
        np.save(handle, array, allow_pickle=False)
    with lzma.open(path, "rb") as handle:
        reloaded = np.load(handle, allow_pickle=False)
    if not isinstance(reloaded, np.ndarray) or reloaded.dtype != array.dtype:
        raise RuntimeError(f"native lzma reload changed array type or dtype for {path}")
    if reloaded.tobytes() != array.tobytes():
        raise RuntimeError(f"native lzma reload changed array values for {path}")


def _default_output_root() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(__file__).resolve().parent / "data" / f"serve_prepend_lookback_{stamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="dated output directory")
    parser.add_argument(
        "--mask-mode", choices=("committed", "no_replay", "both"), default="both",
        help="run the committed mask, the all-False counterfactual, or both",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = (args.out or _default_output_root()).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_modes = MASK_MODES if args.mask_mode == "both" else (args.mask_mode,)
    all_summaries: dict[str, Any] = {}
    files: list[str] = []
    for mask_mode in selected_modes:
        mode_summaries: dict[str, Any] = {}
        pooled_rows: list[dict[str, Any]] = []
        for fixture in FIXTURES:
            rows, array, run_metadata = _measure_variant(fixture, mask_mode)
            csv_path = output_root / f"{fixture.name}_{mask_mode}_rallies.csv.gz"
            array_path = output_root / f"{fixture.name}_{mask_mode}_evidence.npy.xz"
            _write_csv_gz(rows, csv_path)
            _write_array_lzma(array, array_path)
            files.extend([str(csv_path.relative_to(output_root)), str(array_path.relative_to(output_root))])
            mode_summaries[fixture.name] = {**run_metadata, **_summary(rows)}
            pooled_rows.extend(rows)
        mode_summaries["pooled"] = _summary(pooled_rows)
        all_summaries[mask_mode] = mode_summaries

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "fixture_root": str(fixtures_root()),
        "fixtures": [fixture.name for fixture in FIXTURES],
        "mask_modes": list(selected_modes),
        "window_seconds_each_side": WINDOW_SECONDS,
        "notes": [
            "Rows describe the current baseline and evidence around GT serves; they do not run a serve-prepend trigger.",
            "The no_replay variant replaces the raw mask with an all-False mask through the existing calibration precedent.",
            "The array files are native NumPy .npy streams wrapped with lzma XZ preset 9.",
        ],
        "variants": all_summaries,
        "files": sorted(files),
    }
    summary_path = output_root / "summary.json.gz"
    _write_json_gz(summary, summary_path)
    print(json.dumps(summary["variants"], indent=2, sort_keys=True))
    print(f"\nWrote current serve-prepend evidence to {output_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
