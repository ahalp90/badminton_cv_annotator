"""Stage-C wrist-distance measurement on the whole-video pilot (stage-8 item 7).

Runs the crowned stage-8 config over the cached pilot shuttle track, splits the
detected contact candidates into TRUE (greedy-matched to a GT stroke at +/-5 in
any rally's pool) and JUNK, then measures shuttle-to-nearest-wrist distance at
three kinds of frame (every GT stroke, every TRUE candidate, every JUNK
candidate) in three spaces:

  (a) image-normalised   -- A0's exact method (``wrist_contact_separation``), so
                            the GT-frame numbers stay comparable with A0's;
  (b) raw image pixels   -- the same gap in 1080p pixel units;
  (c) striker-bbox-height-- (a) divided by the normalised bbox height of the
                            detection owning the normalised-nearest wrist, so the
                            gap reads relative to the player's on-screen size.

A court-homography space (d) is only measured if a per-video homography for the
scored video is discoverable; for the scraper pilot none is (stage 9 ran on a
court-present visibility proxy, not a fitted homography), so (d) is skipped and
the CSV carries no court column.

Two checks ride along: the GT-alignment check (image-normalised p50 over GT
stroke frames must reproduce A0's ~0.049, else the fresh download's frame
numbering drifted) and the drift-guard datum (ndet at the GT rally-start frames).

The segmentation and scoring are NOT re-implemented here: the crowned config is
patched onto ``scraper.stage8_rally_segmentation`` exactly as ``stage8_sweep``
does, ``segment_video`` produces the spans and contacts, and the TRUE/JUNK split
reuses ``stage8_score``'s ``greedy_match`` / ``_spans_overlapping_extent`` over
the same per-rally candidate pools the scorer builds.

Usage:
    python -m scripts.stage8_wrist_pilot \\
        --track-npy local_scratch/.../pilot_track_npy/1.npy \\
        --pose-dir  local_scratch/.../pilot_pose_raw --vid 1 \\
        --out-dir   local_scratch/.../pilot_results/item7
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

import scraper.stage8_rally_segmentation as stage8_module  # noqa: E402  — needs the src path above

from scripts.stage8_score import (  # noqa: E402  — sibling script, imported after the src path insert
    DEFAULT_SHOTS_MASTER,
    GtRally,
    _spans_overlapping_extent,
    greedy_match,
    load_gt_rallies,
    score_stage8,
)
from scripts.stage8_sweep import Stage8Params, _patch_stage8  # noqa: E402
from scripts.wrist_contact_separation import (  # noqa: E402
    WRIST_L,
    WRIST_R,
    min_wrist_distance,
)

# The crowned boundary + contact config (stage8_block item 7). Pinned, not swept:
# this script's job is to measure the candidate set this exact config produces.
CROWNED_CONFIG = Stage8Params(
    rest_speed=0.002,
    rest_window=5,
    end_rest_frames=90,
    start_speed=0.015,
    start_min_frames=3,
    smooth_window=3,
    min_dir_change_deg=30,
    min_contact_speed=0.005,
)

# The reproduction gate (stage8_block item 7 step 1). Every one of these must
# match or the run stops: unverified spans invalidate everything downstream.
SCORED_TOLERANCES = (1, 2, 5, 10)
MATCH_TOLERANCE = 5  # +/-5 is the TRUE/JUNK split tolerance (recall-first ruling)
EXPECTED_REPRODUCTION = {
    'n_spans': 34,
    'covered': 110,
    'merged_spans': 20,
    'raw_contacts': 9002,
    'candidates_denominator': 50164,
}
EXPECTED_RECALL_5 = 0.5844  # matched 3 dp; the raw value is 959 / 1641

PERCENTILES = (10, 25, 50, 75, 90)

# Distance-column names, one per measured space. Court space (d) is absent for
# the pilot, so it contributes no column.
SPACE_IMAGE_NORM = 'dist_image_norm'
SPACE_PIXELS = 'dist_pixels'
SPACE_BBOX_HEIGHT = 'dist_bbox_height_norm'
SPACE_COLUMNS = (SPACE_IMAGE_NORM, SPACE_PIXELS, SPACE_BBOX_HEIGHT)

# Separation thresholds swept per space (spaces (a) and (c) only, per the spec).
# The image-norm grid brackets A0's 0.15 operating point; the bbox-height grid is
# scaled up because dividing by a ~0.25 player height inflates the gap ~4x.
THRESHOLD_GRID = {
    SPACE_IMAGE_NORM: (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30),
    SPACE_BBOX_HEIGHT: (0.1, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0),
}

POPULATIONS = ('gt', 'true_candidate', 'junk_candidate')


# ---------------------------------------------------------------------------
# Per-frame distance (the only nontrivial pure transform; unit-tested)
# ---------------------------------------------------------------------------
class FrameDistances(NamedTuple):
    """Shuttle-to-nearest-wrist gap at one frame, in the three measured spaces.

    Any field is NaN when the frame is unmeasurable (invisible shuttle, no
    detection, every wrist NaN, or a non-positive owner bbox for the (c) space).
    """

    image_norm: float
    pixels: float
    bbox_height_norm: float


_NAN_DISTANCES = FrameDistances(float('nan'), float('nan'), float('nan'))


def frame_distances(
    wrists_px: np.ndarray, bboxes_px: np.ndarray,
    shuttle_xy_norm: np.ndarray, width: float, height: float,
) -> FrameDistances:
    """The three-space shuttle-to-nearest-wrist gap at one frame.

    Space (a) defers to A0's ``min_wrist_distance`` unchanged, so the number is
    bit-comparable with A0's. Space (b) takes the shuttle back to pixels and
    measures the nearest-wrist gap there. Space (c) scales (a) by the owner
    player's size: it recomputes A0's normalised per-wrist distances (cheap, at
    most 16 detections) only to recover the argmin A0 discards, folds the (L, R)
    wrist axis back to a detection index, and divides (a) by that detection's
    normalised bbox height.

    :param wrists_px: ``(m, 2, 2)`` ``[detection, wrist(L, R), xy]`` pixel coords,
        already sliced to the ``m`` real detections; may hold NaN.
    :param bboxes_px: ``(m, 4)`` ``[x1, y1, x2, y2]`` pixel bboxes, detection-aligned
        to ``wrists_px``.
    :param shuttle_xy_norm: ``(2,)`` normalised shuttle xy.
    :param width: video width in pixels.
    :param height: video height in pixels.
    :return: the ``FrameDistances`` triple; all-NaN when nothing is measurable.
    """
    image_norm = min_wrist_distance(wrists_px, shuttle_xy_norm, width, height)
    if image_norm is None:
        return _NAN_DISTANCES  # ndet 0 or every wrist NaN: no wrist to measure against

    resolution = np.array([width, height])
    shuttle_px = shuttle_xy_norm * resolution
    pixels = float(np.nanmin(np.linalg.norm(wrists_px - shuttle_px, axis=-1)))  # (m, 2) -> scalar

    # Owner of the normalised-nearest wrist. Same arithmetic as min_wrist_distance,
    # recomputed only for the argmin it throws away; //2 collapses the wrist axis.
    dists_norm = np.linalg.norm(wrists_px / resolution - shuttle_xy_norm, axis=-1)  # (m, 2)
    owner = int(np.nanargmin(dists_norm) // 2)
    owner_height_norm = (bboxes_px[owner, 3] - bboxes_px[owner, 1]) / height
    bbox_height_norm = image_norm / owner_height_norm if owner_height_norm > 0 else float('nan')
    return FrameDistances(image_norm, pixels, bbox_height_norm)


def measure_frame(
    frame: int, track: np.ndarray, kps: np.ndarray, bboxes: np.ndarray,
    ndet: np.ndarray, width: float, height: float,
) -> FrameDistances:
    """Distances at one whole-video frame; all-NaN when the shuttle is invisible.

    :param frame: whole-video frame index (source-frame space; the pilot track
        starts at source frame 0, so this is also the track/pose index).
    :param track: ``(t, 3)`` ``[x_norm, y_norm, visibility]`` shuttle track.
    :param kps: ``(t, n_max, 17, 2)`` pixel keypoints, NaN-padded past ndet.
    :param bboxes: ``(t, n_max, 4)`` pixel bboxes, NaN-padded past ndet.
    :param ndet: ``(t,)`` per-frame detection count.
    :param width: video width in pixels.
    :param height: video height in pixels.
    """
    if track[frame, 2] != 1:
        return _NAN_DISTANCES  # invisible shuttle: no position to measure against
    n = int(ndet[frame])
    wrists_px = np.asarray(kps[frame, :n])[:, (WRIST_L, WRIST_R), :]  # (n, 2, 2)
    bboxes_px = np.asarray(bboxes[frame, :n])  # (n, 4)
    return frame_distances(wrists_px, bboxes_px, track[frame, :2], width, height)


# ---------------------------------------------------------------------------
# Step 1: reproduce the crowned config (hard gate)
# ---------------------------------------------------------------------------
def segment_crowned(track: np.ndarray, gt_rallies: list[GtRally]) -> tuple[list, list, dict]:
    """Patch the crowned config, segment the cached track, and score against GT.

    Mirrors ``stage8_sweep._score_config`` exactly: patch all eight module
    globals, call ``segment_video``, score at the sweep tolerances.

    :return: ``(spans, contacts, metrics)``.
    """
    _patch_stage8(CROWNED_CONFIG)
    spans, contacts = stage8_module.segment_video(track)  # positions None
    metrics = score_stage8(spans, contacts, gt_rallies=gt_rallies, tolerances=SCORED_TOLERANCES)
    return spans, contacts, metrics


def reproduction_actuals(spans: list, contacts: list, metrics: dict) -> dict:
    """The five gated counts plus recall_5, read straight off the score dict."""
    curve_5 = metrics['contacts']['tolerances']['5']
    return {
        'n_spans': len(spans),
        'covered': metrics['boundaries']['covered'],
        'merged_spans': metrics['boundaries']['merged_spans'],
        'raw_contacts': len(contacts),
        'candidates_denominator': curve_5['candidates'],
        'recall_5': curve_5['recall'],
    }


def assert_reproduction(actuals: dict) -> None:
    """Fail loud if any gated number drifts from ``EXPECTED_REPRODUCTION``.

    The spans are unverified until every count matches, so a mismatch must stop
    the run rather than let a wrong candidate set flow into the measurement.
    """
    mismatches = [
        f'{key}: got {actuals[key]}, expected {expected}'
        for key, expected in EXPECTED_REPRODUCTION.items()
        if actuals[key] != expected
    ]
    if abs(actuals['recall_5'] - EXPECTED_RECALL_5) > 5e-4:
        mismatches.append(f'recall_5: got {actuals["recall_5"]:.4f}, expected {EXPECTED_RECALL_5}')
    if mismatches:
        raise RuntimeError(
            'Step-1 reproduction gate failed; refusing to measure an unverified '
            'candidate set:\n  ' + '\n  '.join(mismatches)
        )


# ---------------------------------------------------------------------------
# Step 2: TRUE / JUNK candidate populations
# ---------------------------------------------------------------------------
def split_true_junk(
    spans: list, contacts: list, gt_rallies: list[GtRally], tolerance: int,
) -> tuple[list[int], list[int], int]:
    """Partition the physical contact candidates into TRUE and JUNK frames.

    Rebuilds the scorer's per-rally candidate pools (contacts from every span
    overlapping the rally extent), greedy-matches each pool to the rally's
    strokes at ``tolerance``, and marks a candidate TRUE if it is matched in ANY
    rally. Spans are frame-disjoint and per-span contact frames are distinct, so
    a contact frame is a unique candidate identity across the whole video.

    :param spans: detected rally spans, rally_id is the list index.
    :param contacts: ``(rally_id, contact_frame, proximity_ok)`` from segment_video.
    :param gt_rallies: ground-truth rallies for the scored video.
    :param tolerance: match tolerance in frames.
    :return: ``(true_frames, junk_frames, total_matches)`` where the frame lists
        are ascending and ``total_matches`` sums matches across rallies (equals
        the scorer's matched count; exceeds len(true_frames) only if a candidate
        matched in more than one rally's pool).
    """
    contacts_by_span: dict[int, list[int]] = defaultdict(list)
    for rally_id, contact_frame, _proximity in contacts:
        contacts_by_span[rally_id].append(contact_frame)

    matched_frames: set[int] = set()
    total_matches = 0
    for rally in gt_rallies:
        overlapping = _spans_overlapping_extent(rally.extent, spans)
        pooled = [frame for idx in overlapping for frame in contacts_by_span.get(idx, [])]
        pairs = greedy_match(rally.stroke_frames, pooled, tolerance)
        total_matches += len(pairs)
        for _gt_index, candidate_index in pairs:
            matched_frames.add(pooled[candidate_index])

    all_frames = {contact_frame for _rally_id, contact_frame, _proximity in contacts}
    true_frames = sorted(matched_frames)
    junk_frames = sorted(all_frames - matched_frames)
    return true_frames, junk_frames, total_matches


# ---------------------------------------------------------------------------
# Step 3: measure every frame of interest, one row per candidate
# ---------------------------------------------------------------------------
class MeasuredRow(NamedTuple):
    """One CSV row: a frame, its population, and the three-space distances."""

    frame: int
    population: str
    distances: FrameDistances


def measure_population(
    frames: list[int], population: str, track: np.ndarray, kps: np.ndarray,
    bboxes: np.ndarray, ndet: np.ndarray, width: float, height: float,
) -> list[MeasuredRow]:
    """Measure every frame in one population, in ascending frame order."""
    rows: list[MeasuredRow] = []
    for frame in frames:
        distances = measure_frame(frame, track, kps, bboxes, ndet, width, height)
        rows.append(MeasuredRow(int(frame), population, distances))
    return rows


def write_distances_csv(path: Path, rows: list[MeasuredRow]) -> None:
    """Write the per-candidate CSV: frame, population, one column per space.

    A NaN distance (invisible shuttle, no detection, or no positive owner bbox)
    is written as the literal ``NaN`` so pandas reads it back as a missing value.
    """
    header = ['frame', 'population', *SPACE_COLUMNS]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                row.frame,
                row.population,
                _cell(row.distances.image_norm),
                _cell(row.distances.pixels),
                _cell(row.distances.bbox_height_norm),
            ])


def _cell(value: float) -> str:
    """CSV cell for a distance: ``NaN`` when missing, else 6-dp rounded."""
    if np.isnan(value):
        return 'NaN'
    return f'{value:.6f}'


# ---------------------------------------------------------------------------
# Summaries: percentiles and the separation readout
# ---------------------------------------------------------------------------
def finite_values(rows: list[MeasuredRow], population: str, space_index: int) -> np.ndarray:
    """Finite distances of one population in one space (NaN rows dropped)."""
    values = [
        row.distances[space_index]
        for row in rows
        if row.population == population
    ]
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def population_exclusions(
    frames: list[int], track: np.ndarray, ndet: np.ndarray,
) -> tuple[int, int, int]:
    """Frame counts for one population: total, invisible-shuttle, and no-detection.

    No-detection is reported over the visible-shuttle frames (an invisible frame
    is already excluded); it is the residual reason a visible frame goes NaN.
    """
    total = len(frames)
    invisible = sum(1 for frame in frames if track[frame, 2] != 1)
    no_detection = sum(1 for frame in frames if track[frame, 2] == 1 and int(ndet[frame]) == 0)
    return total, invisible, no_detection


def write_percentiles_csv(
    path: Path, rows: list[MeasuredRow], frames_by_population: dict[str, list[int]],
    track: np.ndarray, ndet: np.ndarray,
) -> None:
    """Percentile table: one row per (population, space) with exclusion rates."""
    header = [
        'population', 'space', 'n_total', 'n_invisible', 'invisible_rate',
        'n_no_detection', 'n_finite',
        *[f'p{pct}' for pct in PERCENTILES],
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for population in POPULATIONS:
            total, invisible, no_detection = population_exclusions(
                frames_by_population[population], track, ndet
            )
            invisible_rate = invisible / total if total else float('nan')
            for space_index, space in enumerate(SPACE_COLUMNS):
                values = finite_values(rows, population, space_index)
                pcts = np.percentile(values, PERCENTILES) if len(values) else [float('nan')] * len(PERCENTILES)
                writer.writerow([
                    population, space, total, invisible, round(invisible_rate, 6),
                    no_detection, len(values),
                    *[round(float(value), 6) for value in pcts],
                ])


def separation_rows(
    rows: list[MeasuredRow], space: str, space_index: int,
) -> list[dict]:
    """True-keep vs junk-keep at each threshold for one space (A0's readout).

    A filter that keeps candidates with distance <= threshold keeps
    ``true_keep`` of the TRUE set and ``junk_keep`` of the JUNK set; their ratio
    is the separation factor (A0 reports ~1.2x at its 0.15 operating point).
    """
    true_values = finite_values(rows, 'true_candidate', space_index)
    junk_values = finite_values(rows, 'junk_candidate', space_index)
    out: list[dict] = []
    for threshold in THRESHOLD_GRID[space]:
        true_keep = float(np.mean(true_values <= threshold)) if len(true_values) else float('nan')
        junk_keep = float(np.mean(junk_values <= threshold)) if len(junk_values) else float('nan')
        junk_drop = 1.0 - junk_keep
        keep_ratio = true_keep / junk_keep if junk_keep > 0 else float('nan')
        out.append({
            'space': space,
            'threshold': threshold,
            'n_true': len(true_values),
            'n_junk': len(junk_values),
            'true_keep_rate': round(true_keep, 6),
            'junk_keep_rate': round(junk_keep, 6),
            'junk_drop_rate': round(junk_drop, 6),
            'keep_ratio': round(keep_ratio, 6) if np.isfinite(keep_ratio) else '',
        })
    return out


def write_separation_csv(path: Path, rows: list[MeasuredRow]) -> list[dict]:
    """Write the threshold-sweep separation readout for spaces (a) and (c).

    :return: the flat rows written, so the caller can headline the best ratio.
    """
    header = [
        'space', 'threshold', 'n_true', 'n_junk',
        'true_keep_rate', 'junk_keep_rate', 'junk_drop_rate', 'keep_ratio',
    ]
    all_rows: list[dict] = []
    for space, space_index in ((SPACE_IMAGE_NORM, 0), (SPACE_BBOX_HEIGHT, 2)):
        all_rows.extend(separation_rows(rows, space, space_index))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    return all_rows


# ---------------------------------------------------------------------------
# Step 4.2: drift-guard datum at GT rally starts
# ---------------------------------------------------------------------------
def rally_start_ndet(gt_rallies: list[GtRally], ndet: np.ndarray) -> dict:
    """ndet==0 / ndet==1 / ndet>=2 rates at the first stroke of each rally.

    The player-picking heuristic (sticky anchor / court-filtered pick) needs the
    per-video court homography, which is not available for the scraper pilot, so
    this ndet proxy stands in for the pick-failure rate.
    """
    starts = np.array([rally.stroke_frames[0] for rally in gt_rallies])
    counts = ndet[starts].astype(int)
    n = len(starts)
    return {
        'n_rally_starts': n,
        'ndet_0': int((counts == 0).sum()),
        'ndet_1': int((counts == 1).sum()),
        'ndet_ge2': int((counts >= 2).sum()),
        'ndet_0_rate': round(float((counts == 0).mean()), 6),
        'ndet_1_rate': round(float((counts == 1).mean()), 6),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_pose(pose_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memory-map the pilot pose arrays: keypoints, bboxes, per-frame ndet.

    kps/bboxes are memory-mapped (335 MB / 40 MB) and sliced per frame; ndet is
    small and loaded whole for the vectorised rally-start read.
    """
    kps = np.load(pose_dir / f'{prefix}_kps.npy', mmap_mode='r')
    bboxes = np.load(pose_dir / f'{prefix}_bboxes.npy', mmap_mode='r')
    ndet = np.load(pose_dir / f'{prefix}_ndet.npy')
    return kps, bboxes, ndet


def gt_stroke_frames(gt_rallies: list[GtRally]) -> list[int]:
    """Every GT stroke frame, ascending (one per stroke, the full 1,641)."""
    frames = [frame for rally in gt_rallies for frame in rally.stroke_frames]
    return sorted(frames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--track-npy', type=Path, required=True,
                        help='(t, 3) [x_norm, y_norm, visibility] cached pilot shuttle track')
    parser.add_argument('--pose-dir', type=Path, required=True,
                        help='directory of the pilot pose npys ({prefix}_kps/_bboxes/_ndet.npy)')
    parser.add_argument('--pose-prefix', type=str, default='pilot_1080p_raw',
                        help='filename stem shared by the pose npys (default pilot_1080p_raw)')
    parser.add_argument('--vid', type=int, default=1,
                        help='ShuttleSet video id to score against (default 1)')
    parser.add_argument('--shots-master', type=Path, default=DEFAULT_SHOTS_MASTER,
                        help='ShuttleSet shots_master.csv (default: the in-repo training annotations)')
    parser.add_argument('--width', type=float, default=1920.0,
                        help='pose pixel-space width (default 1920)')
    parser.add_argument('--height', type=float, default=1080.0,
                        help='pose pixel-space height (default 1080)')
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='writes wrist_distances.csv, wrist_distance_percentiles.csv, '
                             'wrist_separation.csv, item7_readout.json')
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    track = np.load(args.track_npy)
    kps, bboxes, ndet = load_pose(args.pose_dir, args.pose_prefix)
    master = pd.read_csv(args.shots_master)
    gt_rallies = load_gt_rallies(master, args.vid)
    print(f'Track {args.track_npy} ({len(track)} frames), vid {args.vid}, '
          f'{len(gt_rallies)} GT rallies, {sum(r.n_strokes for r in gt_rallies)} strokes')

    # One loud coverage check before any measurement: a truncated track or a pose
    # pass over a different cut would otherwise surface as an IndexError (or as
    # silently shifted numbers) deep inside the measurement loops.
    last_stroke = max(rally.stroke_frames[-1] for rally in gt_rallies)
    if last_stroke >= len(track):
        raise ValueError(
            f'GT strokes run to frame {last_stroke} but the track holds only '
            f'{len(track)} frames; the cached track does not cover the annotated video'
        )
    if not (len(kps) == len(bboxes) == len(ndet) == len(track)):
        raise ValueError(
            f'pose arrays and track disagree on frame count: kps {len(kps)}, '
            f'bboxes {len(bboxes)}, ndet {len(ndet)}, track {len(track)}'
        )

    # Step 1: reproduce the crowned config or stop.
    spans, contacts, metrics = segment_crowned(track, gt_rallies)
    actuals = reproduction_actuals(spans, contacts, metrics)
    assert_reproduction(actuals)
    print('Step 1 reproduction OK: ' + '  '.join(f'{k}={v}' for k, v in actuals.items()))

    # Step 2: TRUE / JUNK split.
    true_frames, junk_frames, total_matches = split_true_junk(
        spans, contacts, gt_rallies, MATCH_TOLERANCE
    )
    print(f'Step 2 candidates: TRUE={len(true_frames)}  JUNK={len(junk_frames)}  '
          f'(scorer matches summed={total_matches})')

    frames_by_population = {
        'gt': gt_stroke_frames(gt_rallies),
        'true_candidate': true_frames,
        'junk_candidate': junk_frames,
    }

    # Step 3: measure every frame of interest.
    rows: list[MeasuredRow] = []
    for population in POPULATIONS:
        rows.extend(measure_population(
            frames_by_population[population], population, track, kps, bboxes,
            ndet, args.width, args.height,
        ))

    distances_path = args.out_dir / 'wrist_distances.csv'
    percentiles_path = args.out_dir / 'wrist_distance_percentiles.csv'
    separation_path = args.out_dir / 'wrist_separation.csv'
    write_distances_csv(distances_path, rows)
    write_percentiles_csv(percentiles_path, rows, frames_by_population, track, ndet)
    separation = write_separation_csv(separation_path, rows)

    # Step 4.1: GT-alignment check (image-norm p50 over GT stroke frames vs A0's ~0.049).
    gt_image_norm = finite_values(rows, 'gt', 0)
    gt_p50 = float(np.percentile(gt_image_norm, 50))
    alignment_drift = abs(gt_p50 - 0.049)
    alignment_pass = alignment_drift <= 0.02
    print(f'Step 4.1 GT-alignment: image-norm p50 = {gt_p50:.4f} '
          f'(A0 ~0.049, drift {alignment_drift:.4f}) -> {"PASS" if alignment_pass else "FAIL"}')

    # Step 4.2: drift-guard datum at rally starts.
    start_ndet = rally_start_ndet(gt_rallies, ndet)
    print(f'Step 4.2 rally-start ndet: 0 -> {start_ndet["ndet_0"]}, 1 -> {start_ndet["ndet_1"]}, '
          f'>=2 -> {start_ndet["ndet_ge2"]} of {start_ndet["n_rally_starts"]}')

    readout = {
        'crowned_config': CROWNED_CONFIG._asdict(),
        'reproduction': actuals,
        'reproduction_gate': 'PASS',
        'candidates': {
            'true': len(true_frames),
            'junk': len(junk_frames),
            'scorer_matches_summed': total_matches,
            'raw_total': len(contacts),
        },
        'gt_alignment': {
            'image_norm_p50': round(gt_p50, 6),
            'a0_reference_p50': 0.049,
            'drift': round(alignment_drift, 6),
            'verdict': 'PASS' if alignment_pass else 'FAIL',
        },
        'rally_start_ndet': start_ndet,
        'homography_space_d': 'skipped: no per-video court homography for the pilot '
                              '(stage 9 ran on a court-present visibility proxy, not a fitted homography)',
        'files': {
            'per_candidate': str(distances_path),
            'percentiles': str(percentiles_path),
            'separation': str(separation_path),
        },
    }
    readout_path = args.out_dir / 'item7_readout.json'
    readout_path.write_text(json.dumps(readout, indent=2), encoding='utf-8')

    print(f'\nWrote:\n  {distances_path}\n  {percentiles_path}\n  {separation_path}\n  {readout_path}')
    _print_separation_headline(separation)

    if not alignment_pass:
        raise SystemExit(
            f'GT-alignment failed (p50 {gt_p50:.4f} vs ~0.049, drift {alignment_drift:.4f} > 0.02): '
            'the fresh download frame numbering drifted; downstream measurements are invalid.'
        )


# A0's readout kept ~88.8% of true; report each space at the same operating point
# (retain ~90% of real contacts) so the junk-drop is a like-for-like comparison
# rather than the degenerate high-ratio / low-keep end of the sweep.
OPERATING_TRUE_KEEP = 0.90


def _print_separation_headline(separation: list[dict]) -> None:
    """Per space, report the junk-drop at a ~90%-true-keep operating point vs A0's 1.2x."""
    print(f'\nSeparation at the ~{OPERATING_TRUE_KEEP:.0%}-true-keep operating point (A0 kept 88.8% true):')
    for space in (SPACE_IMAGE_NORM, SPACE_BBOX_HEIGHT):
        space_rows = [row for row in separation if row['space'] == space]
        # Tightest threshold that still retains the target true-keep; else the loosest.
        eligible = [row for row in space_rows if row['true_keep_rate'] >= OPERATING_TRUE_KEEP]
        row = min(eligible, key=lambda r: r['threshold']) if eligible else space_rows[-1]
        ratio = row['keep_ratio'] if isinstance(row['keep_ratio'], float) else float('nan')
        print(
            f'  {space}: at threshold {row["threshold"]} keeps {row["true_keep_rate"]:.1%} true, '
            f'drops {row["junk_drop_rate"]:.1%} junk (keep-ratio {ratio:.2f}x) [A0 ~1.2x]'
        )


if __name__ == '__main__':
    main()
