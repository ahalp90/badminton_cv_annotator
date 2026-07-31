"""Audit pinned stride-8 shuttle tracks against the live hallucination guard.

This is an exploratory instrument, not a second production detector. It uses a
local quadratic RANSAC model in pixel coordinates to produce leads for frames
whose motion is not explained by a small detector-jitter residual. Exact
``(0, 0)`` frames are treated as masking and are excluded from model windows.
Because smooth fill motion can also fit a local quadratic, this lens is biased
towards abrupt-motion outliers and can miss the smooth artefacts the guard is
designed to catch.

Run from the repository root with the project virtual environment, for example::

    ~/.venvs/badminton-cicd/bin/python \
        docs/scraper_pipeline/inpaint_hallucination_fix/analysis/audit_tracks.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from compressed_io import read_json_gz, read_npy_xz, write_json_gz, write_npy_xz


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from annotator.calibration.scoring import load_gt_rallies  # noqa: E402
from annotator.config import BaseAnnotatorConfig  # noqa: E402
from annotator.inpaint_guard import code_counts, grade_track  # noqa: E402
from annotator.rally_segmentation import detect_contact_flags  # noqa: E402
from annotator.resolve import resolve  # noqa: E402
from annotator.run_video import run_video  # noqa: E402


FRAME_WIDTH = 512
FRAME_HEIGHT = 288
WINDOW = 16
WINDOW_STEP = 4
JITTER_RADIUS_PX = 3.0
RANSAC_ITERATIONS = 32
MIN_RANSAC_INLIERS = 8
STORED_TRACK_DTYPE = np.dtype(np.float64)
SHOTS_MASTER_PATH = REPO_ROOT / "training/data/shuttleset/annotations/shots_master.csv"
SOURCE_UNCAUGHT = 1
SOURCE_IMPULSE = 2
SOURCE_TP_RALLY_ENDER = 4
SOURCE_SIDECAR_INPAINT = 8
SOURCE_BITS = (
    (SOURCE_UNCAUGHT, "uncaught"),
    (SOURCE_IMPULSE, "impulse"),
    (SOURCE_TP_RALLY_ENDER, "tp_rally_ender"),
    (SOURCE_SIDECAR_INPAINT, "sidecar_inpaint"),
)
TP_RALLY_ENDER_DEFINITION = (
    "A TP rally-ender means shuttle events that have closed a valid rally, did "
    "not overlap with another valid GT rally, and are valid within our "
    "rally-ending ruleset. This encompasses the fact that ShuttleSet's GT does "
    "not actually record the rally's final event, so we only ever know it "
    "inductively. The GT dataset only ever records the final contact."
)


@dataclass(frozen=True)
class FixtureSpec:
    name: str
    video_id: int
    video_path: str
    source_manifest: str
    source_track_path: str
    source_sidecar_path: str
    expected_track_md5: str
    fps: float

    @property
    def raw_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "raw" / f"{self.name}_track.npy.xz"

    @property
    def legacy_raw_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "raw" / f"{self.name}_track.npy"

    @property
    def sidecar_path(self) -> Path:
        return REPO_ROOT / self.source_sidecar_path


FIXTURES = (
    FixtureSpec(
        name="sset_01",
        video_id=1,
        video_path="videos_288p/pilot_288p.mp4",
        source_manifest=(
            "experiments/annotator/runs/20260730-041328/"
            "static_shuttleset_homography/sset_01/tracknet-stride-8/manifest.json"
        ),
        source_track_path="sset_01_track_npy/sset_01_track.npy",
        source_sidecar_path=(
            "local_scratch/autograder_architecture/inpaint_sidecar/"
            "backfill_staging/out_stride8/1_stride8_inpaint_mask.json.gz"
        ),
        expected_track_md5="08c5afced66b561517a43571df567b2f",
        fps=25.0,
    ),
    FixtureSpec(
        name="sset_15",
        video_id=15,
        video_path="videos_288p/vid15_288p.mp4",
        source_manifest=(
            "experiments/annotator/runs/20260730-041328/"
            "static_shuttleset_homography/sset_15/tracknet-stride-8/manifest.json"
        ),
        source_track_path="sset_15_track_npy/sset_15_track.npy",
        source_sidecar_path=(
            "local_scratch/autograder_architecture/inpaint_sidecar/"
            "backfill_staging/out_stride8/15_stride8_inpaint_mask.json.gz"
        ),
        expected_track_md5="0b9c0966ffc58a36c65f97a5a9a78deb",
        fps=25.0,
    ),
    FixtureSpec(
        name="sset_21",
        video_id=21,
        video_path="videos_288p/sset_21_288p.mp4",
        source_manifest=(
            "experiments/annotator/runs/20260730-041328/"
            "static_shuttleset_homography/sset_21/tracknet-stride-8/manifest.json"
        ),
        source_track_path="sset_21_track_npy/sset_21_track.npy",
        source_sidecar_path=(
            "local_scratch/autograder_architecture/inpaint_sidecar/"
            "backfill_staging/out_stride8/21_stride8_inpaint_mask.json.gz"
        ),
        expected_track_md5="ad00846dc78b08de728cf59ea773ad61",
        fps=30.0,
    ),
)


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixture_track(spec: FixtureSpec) -> np.ndarray:
    if spec.raw_path.exists():
        track = read_npy_xz(spec.raw_path)
    elif spec.legacy_raw_path.exists():
        legacy_track = np.load(spec.legacy_raw_path)
        if md5(spec.legacy_raw_path) != spec.expected_track_md5:
            raise ValueError(f"{spec.name}: legacy raw track hash differs from the pinned source")
        track = legacy_track.astype(STORED_TRACK_DTYPE, copy=True)
        write_npy_xz(track, spec.raw_path)
    else:
        raise FileNotFoundError(f"{spec.name}: no compressed or legacy raw track exists")

    if track.dtype != STORED_TRACK_DTYPE and spec.legacy_raw_path.exists():
        legacy_track = np.load(spec.legacy_raw_path)
        if md5(spec.legacy_raw_path) != spec.expected_track_md5:
            raise ValueError(f"{spec.name}: legacy raw track hash differs from the pinned source")
        track = legacy_track.astype(STORED_TRACK_DTYPE, copy=True)
        write_npy_xz(track, spec.raw_path)

    if track.dtype != STORED_TRACK_DTYPE:
        raise ValueError(
            f"{spec.name}: stored raw track dtype is {track.dtype}, "
            f"expected {STORED_TRACK_DTYPE}"
        )
    return track


def load_sidecar_mask(spec: FixtureSpec, n_frames: int) -> tuple[np.ndarray, dict[str, object]]:
    payload = read_json_gz(spec.sidecar_path)
    if not isinstance(payload, dict):
        raise TypeError(f"{spec.name}: sidecar payload is not a JSON object")
    if payload.get("schema") != "inpaint_fill_mask/1":
        raise ValueError(f"{spec.name}: unsupported sidecar schema {payload.get('schema')!r}")
    if payload.get("index_space") != "frame" or payload.get("stride") != 8:
        raise ValueError(f"{spec.name}: sidecar is not the pinned stride-8 frame mask")
    if payload.get("n_rows") != n_frames:
        raise ValueError(
            f"{spec.name}: sidecar has {payload.get('n_rows')} rows, expected {n_frames}"
        )
    spans = payload.get("inpaint_selected")
    if not isinstance(spans, list):
        raise TypeError(f"{spec.name}: inpaint_selected is not a span list")

    mask = np.zeros(n_frames, dtype=bool)
    previous_stop = 0
    for span in spans:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) for value in span)
        ):
            raise ValueError(f"{spec.name}: invalid sidecar span {span!r}")
        start, stop = span
        if start < previous_stop or start < 0 or stop > n_frames or stop <= start:
            raise ValueError(f"{spec.name}: unsorted or out-of-bounds sidecar span {span!r}")
        mask[start:stop] = True
        previous_stop = stop
    return mask, payload


def pixel_points(track: np.ndarray) -> np.ndarray:
    scale = np.array([FRAME_WIDTH, FRAME_HEIGHT], dtype=np.float64)
    return track[:, :2] * scale


def ransac_triples(seed: int) -> np.ndarray:
    random = np.random.default_rng(seed)
    triples = np.empty((RANSAC_ITERATIONS, 3), dtype=np.int64)
    for index in range(RANSAC_ITERATIONS):
        triples[index] = np.sort(random.choice(WINDOW, size=3, replace=False))
    return triples


def fit_quadratic_ransac(
    points: np.ndarray,
    design: np.ndarray,
    triples: np.ndarray,
    sample_solvers: np.ndarray,
    jitter_radius_px: float,
) -> np.ndarray | None:
    """Return residuals from the best local quadratic RANSAC model."""

    candidate_coefficients = np.einsum(
        "ijk,ikd->ijd", sample_solvers, points[triples]
    )
    predictions = np.einsum("wj,ijd->iwd", design, candidate_coefficients)
    residuals = np.linalg.norm(points[None, :, :] - predictions, axis=2)
    inlier_masks = residuals <= jitter_radius_px
    inlier_counts = inlier_masks.sum(axis=1)
    eligible = np.flatnonzero(inlier_counts >= MIN_RANSAC_INLIERS)
    if not len(eligible):
        return None

    best_count = inlier_counts[eligible].max()
    tied = eligible[inlier_counts[eligible] == best_count]
    median_residuals = np.nanmedian(
        np.where(inlier_masks[tied], residuals[tied], np.nan), axis=1
    )
    best_index = tied[np.argmin(median_residuals)]
    best_mask = inlier_masks[best_index]
    coefficients = np.linalg.lstsq(design[best_mask], points[best_mask], rcond=None)[0]
    return np.linalg.norm(points - design @ coefficients, axis=1)


def run_ransac(
    track: np.ndarray,
    *,
    seed: int,
    jitter_radius_px: float,
) -> dict[str, np.ndarray]:
    """Aggregate local RANSAC residuals over overlapping 16-frame windows."""

    points = pixel_points(track)
    masked = np.all(track[:, :2] == 0, axis=1)
    eligible_windows = np.zeros(len(track), dtype=np.int16)
    outlier_votes = np.zeros(len(track), dtype=np.int16)
    maximum_residual = np.zeros(len(track), dtype=np.float64)
    triples = ransac_triples(seed)
    frame_offsets = np.arange(WINDOW, dtype=np.float64)
    design = np.column_stack((
        np.ones(WINDOW, dtype=np.float64),
        frame_offsets,
        frame_offsets**2,
    ))
    sample_solvers = np.linalg.inv(design[triples])

    for start in range(0, len(track) - WINDOW + 1, WINDOW_STEP):
        window_masked = masked[start : start + WINDOW]
        if window_masked.any():
            continue
        residuals = fit_quadratic_ransac(
            points[start : start + WINDOW],
            design,
            triples,
            sample_solvers,
            jitter_radius_px,
        )
        if residuals is None:
            continue
        window_slice = slice(start, start + WINDOW)
        eligible_windows[window_slice] += 1
        outlier_votes[window_slice] += residuals > jitter_radius_px
        maximum_residual[window_slice] = np.maximum(
            maximum_residual[window_slice], residuals
        )

    minimum_votes = np.maximum(1, (eligible_windows + 1) // 2)
    candidate = (
        ~masked
        & (eligible_windows > 0)
        & (outlier_votes >= minimum_votes)
    )
    outlier_fraction = np.divide(
        outlier_votes,
        eligible_windows,
        out=np.zeros(len(track), dtype=np.float64),
        where=eligible_windows > 0,
    )
    return {
        "eligible_windows": eligible_windows,
        "outlier_votes": outlier_votes,
        "outlier_fraction": outlier_fraction,
        "maximum_residual_px": maximum_residual,
        "candidate": candidate,
        "masked": masked,
    }


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def chunk_rows(track: np.ndarray, uncaught: np.ndarray) -> list[dict[str, int | float]]:
    points = pixel_points(track)
    rows: list[dict[str, int | float]] = []
    for start, stop in contiguous_runs(uncaught):
        chunk = points[start:stop]
        variance = np.var(chunk, axis=0)
        ranges = np.ptp(chunk, axis=0)
        rows.append({
            "start_frame": start,
            "stop_frame_exclusive": stop,
            "length_frames": stop - start,
            "x_variance_px2": float(variance[0]),
            "y_variance_px2": float(variance[1]),
            "radial_variance_px2": float(np.var(chunk, axis=0).sum()),
            "x_range_px": float(ranges[0]),
            "y_range_px": float(ranges[1]),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=9) as target:
        if not fieldnames:
            target.write("\n")
            return
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_fixed_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=9) as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


IMPULSE_EVENT_FIELDS = (
    "event_frame",
    "rally_id",
    "impulse",
    "coordinate_valid",
    "uncaught_ransac",
    "guard_code",
)
TP_RALLY_ENDER_FIELDS = (
    "span_id",
    "span_start",
    "span_end_exclusive",
    "event_frame",
    "gt_set_id",
    "gt_rally",
    "gt_first_contact",
    "gt_last_contact",
    "overlapping_gt_rallies",
    "contained_gt_rallies",
    "valid_tp_rally_ender",
    "reason",
    "coordinate_valid",
    "uncaught_ransac",
    "guard_code",
)


def event_source_counts(source_codes: np.ndarray) -> dict[str, int]:
    return {
        source_label(code): int((source_codes == code).sum())
        for code in range(1 << len(SOURCE_BITS))
    }


def source_label(code: int) -> str:
    if code == 0:
        return "none"
    return "_and_".join(name for bit, name in SOURCE_BITS if code & bit)


def build_event_outputs(
    spec: FixtureSpec,
    output_dir: Path,
    track: np.ndarray,
    codes: np.ndarray,
    uncaught: np.ndarray,
    valid: np.ndarray,
    sidecar_mask: np.ndarray,
    shots_master: pd.DataFrame,
) -> dict[str, object]:
    """Build source-labelled event masks from the existing detector composition."""

    base = BaseAnnotatorConfig()
    resolved = resolve(base, spec.fps)
    result = run_video(
        track,
        fps=spec.fps,
        base=base,
        raw_exclusion_mask=np.zeros(len(track), dtype=bool),
        court_optional=True,
        stop_after_segmentation=True,
    )
    gt_rallies = load_gt_rallies(shots_master, spec.video_id)
    impulse_mask = np.zeros(len(track), dtype=bool)
    impulse_rows: list[dict[str, object]] = []
    flagged_pairs: list[tuple[int, int]] = []
    for rally_id, (start, end) in enumerate(result.spans):
        flags = detect_contact_flags(
            track,
            start,
            end,
            resolved.thresholds,
            smoothing_mode=resolved.smoothing_mode,
        )
        for frame, impulse in flags:
            if not 0 <= frame < len(track):
                raise ValueError(f"{spec.name}: impulse frame {frame} is out of bounds")
            flagged_pairs.append((rally_id, frame))
            impulse_mask[frame] = True
            impulse_rows.append({
                "event_frame": frame,
                "rally_id": rally_id,
                "impulse": impulse,
                "coordinate_valid": bool(valid[frame]),
                "uncaught_ransac": bool(uncaught[frame]),
                "guard_code": int(codes[frame]),
            })

    contact_pairs = [
        (int(contact.rally_id), int(contact.contact_frame))
        for contact in result.contacts
    ]
    if flagged_pairs != contact_pairs:
        raise ValueError(
            f"{spec.name}: resolved impulse flags disagree with run_video raw contacts"
        )

    tp_mask = np.zeros(len(track), dtype=bool)
    tp_rows: list[dict[str, object]] = []
    accepted_spans = 0
    for span_id, (start, end) in enumerate(result.spans):
        overlapping = [
            rally for rally in gt_rallies
            if start <= rally.extent[1] and rally.extent[0] < end
        ]
        contained = [
            rally for rally in gt_rallies
            if start <= rally.extent[0] and rally.extent[1] < end
        ]
        event_frame = end - 1
        valid_tp = (
            end < len(track)
            and len(overlapping) == 1
            and len(contained) == 1
        )
        if end >= len(track):
            reason = "video_end_no_close_event"
        elif len(overlapping) > 1:
            reason = "overlaps_multiple_gt_rallies"
        elif len(contained) != 1:
            reason = "does_not_contain_one_complete_gt_rally"
        else:
            reason = "valid_single_gt_rally"
        if valid_tp:
            tp_mask[event_frame] = True
            accepted_spans += 1
        contained_text = [f"{rally.set_id}:{rally.rally}" for rally in contained]
        overlapping_text = [f"{rally.set_id}:{rally.rally}" for rally in overlapping]
        gt_rally = contained[0] if len(contained) == 1 else None
        tp_rows.append({
            "span_id": span_id,
            "span_start": start,
            "span_end_exclusive": end,
            "event_frame": event_frame,
            "gt_set_id": gt_rally.set_id if gt_rally is not None else "",
            "gt_rally": gt_rally.rally if gt_rally is not None else "",
            "gt_first_contact": gt_rally.extent[0] if gt_rally is not None else "",
            "gt_last_contact": gt_rally.extent[1] if gt_rally is not None else "",
            "overlapping_gt_rallies": ";".join(overlapping_text),
            "contained_gt_rallies": ";".join(contained_text),
            "valid_tp_rally_ender": valid_tp,
            "reason": reason,
            "coordinate_valid": bool(valid[event_frame]),
            "uncaught_ransac": bool(uncaught[event_frame]),
            "guard_code": int(codes[event_frame]),
        })

    impulse_valid = impulse_mask & valid
    tp_valid = tp_mask & valid
    sidecar_valid = sidecar_mask & valid
    source_codes = np.zeros(len(track), dtype=np.uint8)
    source_codes[uncaught] |= SOURCE_UNCAUGHT
    source_codes[impulse_valid] |= SOURCE_IMPULSE
    source_codes[tp_valid] |= SOURCE_TP_RALLY_ENDER
    source_codes[sidecar_valid] |= SOURCE_SIDECAR_INPAINT
    union_uncaught_inpaint_impulse = uncaught | sidecar_valid | impulse_valid
    union_uncaught_impulse_tp = uncaught | impulse_valid | tp_valid

    impulse_mask_name = f"{spec.name}_impulse_event_mask.npy.xz"
    tp_mask_name = f"{spec.name}_tp_rally_ender_mask.npy.xz"
    source_codes_name = f"{spec.name}_event_source_codes.npy.xz"
    impulse_csv_name = f"{spec.name}_impulse_events.csv.gz"
    tp_csv_name = f"{spec.name}_tp_rally_ender_events.csv.gz"
    write_npy_xz(impulse_mask, output_dir / impulse_mask_name)
    write_npy_xz(tp_mask, output_dir / tp_mask_name)
    write_npy_xz(source_codes, output_dir / source_codes_name)
    write_fixed_csv(output_dir / impulse_csv_name, IMPULSE_EVENT_FIELDS, impulse_rows)
    write_fixed_csv(output_dir / tp_csv_name, TP_RALLY_ENDER_FIELDS, tp_rows)

    return {
        "definition": TP_RALLY_ENDER_DEFINITION,
        "rally_end_event_frame": (
            "last frame inside the accepted half-open current-default span; the "
            "span end is retained as span_end_exclusive"
        ),
        "gt_rule": (
            "accept only a pre-video-end span containing exactly one complete GT "
            "rally and no second overlapping GT rally"
        ),
        "resolved_default_path": {
            "composition": (
                "run_video(base=BaseAnnotatorConfig(), raw_exclusion_mask=zeros, "
                "court_optional=True, stop_after_segmentation=True)"
            ),
            "fps": spec.fps,
            "smoothing_mode": str(resolved.smoothing_mode),
            "span_open": str(resolved.span_open),
            "impulse_floor_half_window_frames": int(
                resolved.thresholds.impulse_floor_half_window_frames
            ),
            "contact_dedup_radius_frames": int(
                resolved.thresholds.contact_dedup_radius_frames
            ),
            "contact_impulse_multiple": float(
                resolved.thresholds.contact_impulse_multiple
            ),
        },
        "counts": {
            "gt_rallies": len(gt_rallies),
            "detected_spans": len(result.spans),
            "raw_impulse_rows": len(impulse_rows),
            "unique_impulse_frames": int(impulse_mask.sum()),
            "coordinate_valid_impulse_frames": int(impulse_valid.sum()),
            "accepted_tp_rally_ender_spans": accepted_spans,
            "accepted_tp_rally_ender_frames": int(tp_mask.sum()),
            "coordinate_valid_tp_rally_ender_frames": int(tp_valid.sum()),
            "uncaught_frames": int(uncaught.sum()),
            "uncaught_and_impulse_frames": int((uncaught & impulse_valid).sum()),
            "uncaught_and_tp_rally_ender_frames": int((uncaught & tp_valid).sum()),
            "impulse_and_tp_rally_ender_frames": int((impulse_valid & tp_valid).sum()),
            "uncaught_inpaint_impulse_union_frames": int(
                union_uncaught_inpaint_impulse.sum()
            ),
            "uncaught_impulse_tp_rally_ender_union_frames": int(
                union_uncaught_impulse_tp.sum()
            ),
            "sidecar_inpaint_frames": int(sidecar_valid.sum()),
            "uncaught_and_sidecar_inpaint_frames": int((uncaught & sidecar_valid).sum()),
            "sidecar_inpaint_and_impulse_frames": int(
                (sidecar_valid & impulse_valid).sum()
            ),
            "sidecar_inpaint_and_tp_rally_ender_frames": int(
                (sidecar_valid & tp_valid).sum()
            ),
            "valid_frames": int(valid.sum()),
            "zero_zero_frames": int((~valid).sum()),
        },
        "exclusive_source_counts_valid_frames": event_source_counts(source_codes[valid]),
        "source_code_counts_all_frames": event_source_counts(source_codes),
        "arrays": {
            "impulse_event_mask": impulse_mask_name,
            "tp_rally_ender_mask": tp_mask_name,
            "event_source_codes": source_codes_name,
        },
        "tables": {
            "impulse_events": impulse_csv_name,
            "tp_rally_ender_events": tp_csv_name,
        },
    }


def write_frame_audit(
    path: Path,
    track: np.ndarray,
    codes: np.ndarray,
    ransac: dict[str, np.ndarray],
) -> None:
    points = pixel_points(track)
    candidate = ransac["candidate"]
    caught = candidate & (codes != 0)
    uncaught = candidate & (codes == 0)
    with gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=9) as target:
        writer = csv.writer(target)
        writer.writerow([
            "frame",
            "x_normalized",
            "y_normalized",
            "visibility",
            "x_px",
            "y_px",
            "masked_zero_zero",
            "guard_code",
            "ransac_eligible_windows",
            "ransac_outlier_votes",
            "ransac_outlier_fraction",
            "ransac_max_residual_px",
            "ransac_candidate",
            "guard_caught_candidate",
            "guard_uncaught_candidate",
        ])
        for frame in range(len(track)):
            writer.writerow([
                frame,
                f"{track[frame, 0]:.12g}",
                f"{track[frame, 1]:.12g}",
                int(track[frame, 2]),
                f"{points[frame, 0]:.6f}",
                f"{points[frame, 1]:.6f}",
                int(ransac["masked"][frame]),
                int(codes[frame]),
                int(ransac["eligible_windows"][frame]),
                int(ransac["outlier_votes"][frame]),
                f"{ransac['outlier_fraction'][frame]:.6f}",
                f"{ransac['maximum_residual_px'][frame]:.6f}",
                int(candidate[frame]),
                int(caught[frame]),
                int(uncaught[frame]),
            ])


def fixture_audit(
    spec: FixtureSpec,
    output_dir: Path,
    track: np.ndarray,
    shots_master: pd.DataFrame,
) -> dict[str, object]:
    if track.ndim != 2 or track.shape[1] < 3:
        raise ValueError(f"{spec.name}: expected a (frames, 3+) track, got {track.shape}")
    if not np.isfinite(track[:, :3]).all():
        raise ValueError(f"{spec.name}: track contains non-finite datapoints")
    if (track[:, :2] < 0).any() or (track[:, :2] > 1).any():
        raise ValueError(
            f"{spec.name}: track coordinates are outside the expected [0, 1] normalised range"
        )

    codes, info = grade_track(track)
    ransac = run_ransac(track, seed=20260731, jitter_radius_px=JITTER_RADIUS_PX)
    sidecar_mask, sidecar_payload = load_sidecar_mask(spec, len(track))
    candidate = ransac["candidate"]
    valid = ~ransac["masked"]
    caught = candidate & (codes != 0)
    uncaught = candidate & (codes == 0)
    chunks = chunk_rows(track, uncaught)
    event_summary = build_event_outputs(
        spec,
        output_dir,
        track,
        codes,
        uncaught,
        valid,
        sidecar_mask,
        shots_master,
    )

    write_npy_xz(codes, output_dir / f"{spec.name}_guard_codes.npy.xz")
    write_npy_xz(candidate, output_dir / f"{spec.name}_ransac_candidate.npy.xz")
    write_npy_xz(uncaught, output_dir / f"{spec.name}_uncaught_mask.npy.xz")
    write_npy_xz(sidecar_mask, output_dir / f"{spec.name}_sidecar_inpaint_mask.npy.xz")
    write_csv(output_dir / f"{spec.name}_uncaught_chunks.csv.gz", chunks)
    write_frame_audit(output_dir / f"{spec.name}_frame_audit.csv.gz", track, codes, ransac)

    sidecar_valid = sidecar_mask & valid
    sidecar_guard_nonzero = sidecar_mask & (codes != 0)
    sidecar_ransac_candidate = sidecar_mask & candidate

    return {
        "name": spec.name,
        "video_id": spec.video_id,
        "video_path": spec.video_path,
        "source_manifest": spec.source_manifest,
        "source_track_path": spec.source_track_path,
        "raw_track_path": str(spec.raw_path.relative_to(REPO_ROOT)),
        "source_track_md5": spec.expected_track_md5,
        "stored_track_sha256": sha256(spec.raw_path),
        "shape": list(track.shape),
        "dtype": str(track.dtype),
        "fps": spec.fps,
        "resolution_px": [FRAME_WIDTH, FRAME_HEIGHT],
        "coordinate_convention": "x_normalized = x_px / 512; y_normalized = y_px / 288",
        "zero_zero_frames": int(ransac["masked"].sum()),
        "valid_frames": int(valid.sum()),
        "guard_code_counts": code_counts(codes),
        "guard_info": info,
        "ransac": {
            "model": "local quadratic x(frame), y(frame)",
            "window_frames": WINDOW,
            "window_step_frames": WINDOW_STEP,
            "jitter_radius_px": JITTER_RADIUS_PX,
            "iterations": RANSAC_ITERATIONS,
            "minimum_inliers": MIN_RANSAC_INLIERS,
            "eligible_frames": int((ransac["eligible_windows"] > 0).sum()),
            "candidate_frames": int(candidate.sum()),
            "candidate_fraction_of_valid": float(candidate.sum() / valid.sum()),
            "caught_candidate_frames": int(caught.sum()),
            "uncaught_candidate_frames": int(uncaught.sum()),
            "caught_fraction_of_candidates": float(caught.sum() / candidate.sum())
            if candidate.any()
            else None,
            "guard_nonzero_fraction_of_valid": float((codes[valid] != 0).mean()),
        },
        "sidecar": {
            "path": spec.source_sidecar_path,
            "status": sidecar_payload["inpaint_status"],
            "selected_frames": int(sidecar_mask.sum()),
            "selected_valid_frames": int(sidecar_valid.sum()),
            "selected_zero_zero_frames": int((sidecar_mask & ransac["masked"]).sum()),
            "selected_guard_nonzero_frames": int(sidecar_guard_nonzero.sum()),
            "selected_ransac_candidate_frames": int(sidecar_ransac_candidate.sum()),
            "selected_ransac_uncaught_frames": int((sidecar_mask & uncaught).sum()),
        },
        "uncaught_chunk_count": len(chunks),
        "uncaught_singleton_chunk_count": sum(
            row["length_frames"] == 1 for row in chunks
        ),
        "uncaught_zero_variance_chunk_count": sum(
            row["radial_variance_px2"] == 0 for row in chunks
        ),
        "uncaught_chunks_path": f"{spec.name}_uncaught_chunks.csv.gz",
        "frame_audit_path": f"{spec.name}_frame_audit.csv.gz",
        "event_sources": event_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workset",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="analysis workset directory (default: this script's parent workset)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workset = args.workset.resolve()
    output_dir = workset / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_manifest: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    shots_master = pd.read_csv(SHOTS_MASTER_PATH)
    for spec in FIXTURES:
        track = load_fixture_track(spec)
        raw_manifest.append({
            "fixture": spec.name,
            "video_id": spec.video_id,
            "raw_path": str(spec.raw_path.relative_to(REPO_ROOT)),
            "source_manifest": spec.source_manifest,
            "source_track_path": spec.source_track_path,
            "source_sidecar_path": spec.source_sidecar_path,
            "source_video_path": spec.video_path,
            "source_track_md5": spec.expected_track_md5,
            "stored_track_sha256": sha256(spec.raw_path),
            "stored_dtype": str(STORED_TRACK_DTYPE),
        })
        audit.append(fixture_audit(spec, output_dir, track, shots_master))

    write_json_gz(workset / "raw_manifest.json.gz", {"fixtures": raw_manifest})
    write_json_gz(output_dir / "event_union.json.gz", {
        "instrument": "audit_tracks.py",
        "status": (
            "source-labelled exploratory unions; impulse rows follow the current "
            "resolved default detector path, while TP rally-enders are inductive "
            "span-close proxies rather than direct GT events"
        ),
        "source_bits": {
            "uncaught": SOURCE_UNCAUGHT,
            "impulse": SOURCE_IMPULSE,
            "tp_rally_ender": SOURCE_TP_RALLY_ENDER,
            "sidecar_inpaint": SOURCE_SIDECAR_INPAINT,
        },
        "tp_rally_ender_definition": TP_RALLY_ENDER_DEFINITION,
        "fixtures": [row["event_sources"] for row in audit],
    })
    write_json_gz(output_dir / "track_audit.json.gz", {
            "instrument": "audit_tracks.py",
            "status": (
                "exploratory leads, not ground truth; the RANSAC lens is biased "
                "towards abrupt-motion outliers and can treat smooth-fill artefacts as inliers"
            ),
            "parameters": {
                "resolution_px": [FRAME_WIDTH, FRAME_HEIGHT],
                "window_frames": WINDOW,
                "window_step_frames": WINDOW_STEP,
                "jitter_radius_px": JITTER_RADIUS_PX,
                "ransac_iterations": RANSAC_ITERATIONS,
                "minimum_inliers": MIN_RANSAC_INLIERS,
                "zero_zero_policy": "exclude any RANSAC window containing (0, 0)",
                "candidate_policy": "at least half of eligible windows vote outlier",
            },
            "fixtures": audit,
        })
    for row in audit:
        ransac = row["ransac"]
        print(
            f"{row['name']}: {ransac['candidate_frames']} RANSAC candidates; "
            f"{ransac['caught_candidate_frames']} caught "
            f"({ransac['caught_fraction_of_candidates']!s}); "
            f"{ransac['uncaught_candidate_frames']} uncaught"
        )


if __name__ == "__main__":
    main()
