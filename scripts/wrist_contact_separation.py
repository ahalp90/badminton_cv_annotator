"""Measure shuttle-to-nearest-wrist distance at contacts and stage-8 candidates.

A0 of the stage-8 sweep plan
(``local_scratch/autograder_architecture/stage8_sweep_plan.md``). Over the
existing per-clip ShuttleSet artefacts, record the image-space distance from the
shuttle to the nearest wrist of any detected person at two kinds of frame:

  - ``gt``   : ground-truth contact frames (strokes whose ``frame_num`` lands in
               the clip's shuttle window)
  - ``cand`` : stage-8 reversal candidates, run at shipped defaults over the
               whole per-clip track

Each ``cand`` row is labelled true/false against GT in source-video frame space,
and each ``gt`` row carries whether any candidate landed near it (the recall
marker). The point is to see whether wrist proximity separates true contacts
from false candidates before any schema change is committed. Everything here is
CPU and batch; output is one CSV plus a small summary .txt.

Window semantics (confirmed against ``scripts/build_shots_master.py`` and
``shared.dataset.compute_clip_bounds``): the clip covers source frames
``[shuttle_start_f, shuttle_end_f)`` -- start inclusive, end exclusive -- so the
window length is ``shuttle_end_f - shuttle_start_f`` and the clip-local index of
a source frame is ``frame_num - shuttle_start_f`` (matches
``bst_x/validation_scripts/hit_frame_lookup.py``).

Usage:
    python -m scripts.wrist_contact_separation \\
        --raw-kps-dir <dir of {stem}_raw_kps.npy / {stem}_raw_ndet.npy> \\
        --out-csv <path> [--shots-master ...] [--shuttle-dir ...] \\
        [--resolution-csv ...] [--workers N] [--tolerance 2]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from annotator.rally_segmentation import detect_contacts  # noqa: E402  — follows sys.path insertion

# COCO wrist keypoint indices in the raw_kps (F, N_max, 17, 2) arrays.
WRIST_L, WRIST_R = 9, 10

# |len(track) - window_length| beyond this many frames is flagged (not skipped):
# the clip cut and the track occasionally differ by a frame at the seam.
WINDOW_LEN_TOLERANCE = 2
# The summary reports candidate precision / GT recall at the run tolerance and
# again at this looser one, since GT hit labels are known to be a few frames noisy.
EXTRA_TOLERANCE = 5

OUT_COLUMNS = [
    'clip_stem', 'vid', 'kind', 'frame_local', 'frame_source',
    'matched', 'shuttle_visible', 'n_det', 'min_wrist_dist',
]

PERCENTILES = (10, 25, 50, 75, 90)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no IO)
# ---------------------------------------------------------------------------
def contacts_in_window(
    vid_frame_nums: np.ndarray, start_f: int, end_f: int,
) -> np.ndarray:
    """Source frames of every stroke whose ``frame_num`` falls in the clip window.

    The window is start-inclusive, end-exclusive. Neighbouring strokes share the
    overlapping window, so more than one may land inside; the anchor stroke is
    not the only true contact.

    :param vid_frame_nums: ``(k,)`` int, every stroke ``frame_num`` of the
        clip's source video.
    :param start_f: window start frame (inclusive).
    :param end_f: window end frame (exclusive).
    :return: ``(m,)`` int, ascending source frames inside the window.
    """
    frames = np.asarray(vid_frame_nums)
    in_window = (frames >= start_f) & (frames < end_f)
    return np.sort(frames[in_window])


def matched_within(
    query_frames: np.ndarray, reference_frames: np.ndarray, tolerance: int,
) -> np.ndarray:
    """Per-query flag: True where any reference frame is within +/- tolerance.

    Both inputs must live in the same frame space (callers pass source frames);
    this is the shared primitive behind candidate labelling and GT recall.

    :param query_frames: ``(q,)`` int frames to label.
    :param reference_frames: ``(r,)`` int frames to match against.
    :param tolerance: largest absolute frame gap that counts as a match.
    :return: ``(q,)`` bool.
    """
    query = np.asarray(query_frames)
    reference = np.asarray(reference_frames)
    if len(query) == 0:
        return np.zeros(0, dtype=bool)
    if len(reference) == 0:
        return np.zeros(len(query), dtype=bool)
    gap = np.abs(query[:, None] - reference[None, :])  # (q, r)
    return (gap <= tolerance).any(axis=1)


def min_wrist_distance(
    wrists_px: np.ndarray, shuttle_xy_norm: np.ndarray,
    width: float, height: float,
) -> float | None:
    """Smallest normalised L2 gap from the shuttle to any detection's wrist.

    Wrists arrive in pixel coords and are divided by the same W/H that
    normalised the shuttle track, so both sit in the [0, 1] image space. NaN
    wrists (undetected joints) are skipped.

    :param wrists_px: ``(m, 2, 2)`` ``[detection, wrist(L, R), xy]`` pixel coords,
        may hold NaN.
    :param shuttle_xy_norm: ``(2,)`` normalised shuttle xy.
    :param width: video width in pixels.
    :param height: video height in pixels.
    :return: min distance as float, or None when no finite wrist exists
        (``ndet`` 0 or every wrist NaN).
    """
    if wrists_px.shape[0] == 0:
        return None
    wrists_norm = wrists_px / np.array([width, height])  # (m, 2, 2)
    dists = np.linalg.norm(wrists_norm - shuttle_xy_norm, axis=-1)  # (m, 2) per detection per wrist
    if np.all(np.isnan(dists)):
        return None
    return float(np.nanmin(dists))


def frame_measurement(
    frame_local: int, track: np.ndarray, raw_kps: np.ndarray,
    ndet: np.ndarray, width: float, height: float,
) -> tuple[bool, int | None, float | None]:
    """Shuttle visibility, detection count and min wrist distance at one frame.

    A short track (window-length mismatch) can leave a GT local frame past the
    array end; that is unmeasurable and returns all-blank. An invisible shuttle
    (``track[f, 2] != 1``) records the detection count but a blank distance,
    since there is no shuttle position to measure against.

    :param frame_local: clip-local frame index.
    :param track: ``(F, 3)`` ``[x_norm, y_norm, visibility]`` shuttle track.
    :param raw_kps: ``(F, N_max, 17, 2)`` pixel keypoints, NaN-padded past ndet.
    :param ndet: ``(F,)`` per-frame detection count.
    :param width: video width in pixels.
    :param height: video height in pixels.
    :return: ``(shuttle_visible, n_det, min_wrist_dist)``; the last two are None
        when out of range, and ``min_wrist_dist`` is None when the shuttle is
        invisible or no finite wrist exists.
    """
    if not 0 <= frame_local < len(track):
        return False, None, None

    shuttle_visible = bool(track[frame_local, 2] == 1)
    n_det = int(ndet[frame_local])
    if not shuttle_visible:
        return False, n_det, None

    shuttle_xy = track[frame_local, :2]  # (2,)
    wrists_px = raw_kps[frame_local, :n_det][:, (WRIST_L, WRIST_R), :]  # (n_det, 2, 2)
    min_dist = min_wrist_distance(wrists_px, shuttle_xy, width, height)
    return True, n_det, min_dist


def safe_ratio(numer: float, denom: float) -> float | None:
    """``numer / denom`` as float, or None when the denominator is 0 (undefined)."""
    if denom == 0:
        return None
    return numer / denom


# ---------------------------------------------------------------------------
# Per-clip measurement (pure over arrays; IO lives in process_clip)
# ---------------------------------------------------------------------------
class ClipResult(NamedTuple):
    """One clip's contribution to the run: rows for the CSV plus summary counts.

    ``skip_cause`` is None on a measured clip; a non-None cause means the clip
    was skipped and ``rows`` is empty. The ``*_extra`` counts mirror the tol
    counts at ``EXTRA_TOLERANCE`` for the looser precision/recall read.
    """
    rows: list[dict]
    skip_cause: str | None
    window_len_mismatch: bool
    n_gt: int
    n_cand: int
    cand_matched_tol: int
    cand_matched_extra: int
    gt_matched_tol: int
    gt_matched_extra: int
    n_gt_invisible: int


def _row(
    clip_stem: str, vid: int, kind: str, frame_local: int, frame_source: int,
    matched: bool, shuttle_visible: bool, n_det: int | None, min_dist: float | None,
) -> dict:
    """Assemble one output row dict keyed by OUT_COLUMNS."""
    return {
        'clip_stem': clip_stem,
        'vid': vid,
        'kind': kind,
        'frame_local': frame_local,
        'frame_source': frame_source,
        'matched': matched,
        'shuttle_visible': shuttle_visible,
        'n_det': n_det,
        'min_wrist_dist': min_dist,
    }


def measure_clip(
    clip_stem: str, vid: int, start_f: int, end_f: int, vid_frame_nums: np.ndarray,
    track: np.ndarray, raw_kps: np.ndarray, ndet: np.ndarray,
    width: float, height: float, tolerance: int,
) -> tuple[list[dict], dict]:
    """Measure one clip's GT contacts and stage-8 candidates.

    Candidates come from ``detect_contacts`` over the whole clip track
    (``start=0, end=len(track)``), returned in clip-local frames. Labelling is
    in source frame space: a candidate is matched when any of the video's GT
    ``frame_num`` sits within tolerance, so a candidate lining up with a GT
    stroke just outside the window (the post-hit tail clipping the next stroke)
    still counts. GT recall is the mirror: a GT row is matched when any candidate
    landed within tolerance.

    :return: ``(rows, stats)`` where stats holds the per-clip summary counts.
    """
    n_frames = len(track)

    gt_sources = contacts_in_window(vid_frame_nums, start_f, end_f)  # (m,)
    cand_locals = np.asarray(detect_contacts(track, 0, n_frames), dtype=int)  # (c,)
    cand_sources = cand_locals + start_f  # (c,)

    all_gt = np.asarray(vid_frame_nums, dtype=int)
    cand_matched_tol = matched_within(cand_sources, all_gt, tolerance)  # (c,)
    cand_matched_extra = matched_within(cand_sources, all_gt, EXTRA_TOLERANCE)
    gt_matched_tol = matched_within(gt_sources, cand_sources, tolerance)  # (m,)
    gt_matched_extra = matched_within(gt_sources, cand_sources, EXTRA_TOLERANCE)

    rows: list[dict] = []
    n_gt_invisible = 0
    for gt_source, is_recalled in zip(gt_sources, gt_matched_tol):
        frame_local = int(gt_source) - start_f
        shuttle_visible, n_det, min_dist = frame_measurement(
            frame_local, track, raw_kps, ndet, width, height
        )
        if not shuttle_visible:
            n_gt_invisible += 1
        rows.append(_row(
            clip_stem, vid, 'gt', frame_local, int(gt_source),
            bool(is_recalled), shuttle_visible, n_det, min_dist,
        ))

    for cand_local, cand_source, is_matched in zip(cand_locals, cand_sources, cand_matched_tol):
        shuttle_visible, n_det, min_dist = frame_measurement(
            int(cand_local), track, raw_kps, ndet, width, height
        )
        rows.append(_row(
            clip_stem, vid, 'cand', int(cand_local), int(cand_source),
            bool(is_matched), shuttle_visible, n_det, min_dist,
        ))

    stats = {
        'n_gt': int(len(gt_sources)),
        'n_cand': int(len(cand_locals)),
        'cand_matched_tol': int(cand_matched_tol.sum()),
        'cand_matched_extra': int(cand_matched_extra.sum()),
        'gt_matched_tol': int(gt_matched_tol.sum()),
        'gt_matched_extra': int(gt_matched_extra.sum()),
        'n_gt_invisible': n_gt_invisible,
    }
    return rows, stats


# ---------------------------------------------------------------------------
# Pool / IO glue (thin)
# ---------------------------------------------------------------------------
class ClipTask(NamedTuple):
    """One shots_master row's worth of work: the clip and its window bounds."""
    clip_stem: str
    vid: int
    start_f: int
    end_f: int


class WorkerCtx(NamedTuple):
    """Read-only lookups shared by every task, passed once per worker.

    Held in a module global so forked/spawned workers get it via the pool
    initializer rather than re-shipping the (large) per-video frame map with
    each of the ~32k tasks.
    """
    vid_frames: dict[int, np.ndarray]
    resolution: dict[int, tuple[float, float]]
    shuttle_dir: Path
    raw_kps_dir: Path
    tolerance: int


_WORKER_CTX: WorkerCtx | None = None


def _init_worker(ctx: WorkerCtx) -> None:
    """Pool initializer: stash the shared context in this worker's global."""
    global _WORKER_CTX
    _WORKER_CTX = ctx


def _empty_result(skip_cause: str) -> ClipResult:
    """A skipped clip contributes no rows and no counts, only its skip cause."""
    return ClipResult([], skip_cause, False, 0, 0, 0, 0, 0, 0, 0)


def process_clip(task: ClipTask) -> ClipResult:
    """Load one clip's arrays, guard, and measure. The pool worker entry point.

    Skip causes (logged and counted, never fatal -- one bad clip must not sink
    the batch): a missing shuttle/kps/ndet file, or a track/kps/ndet length
    disagreement. A track whose length differs from the annotated window by more
    than WINDOW_LEN_TOLERANCE is measured anyway but flagged.
    """
    ctx = _WORKER_CTX
    assert ctx is not None, 'worker context not initialised'

    shuttle_path = ctx.shuttle_dir / f'{task.clip_stem}.npy'
    raw_kps_path = ctx.raw_kps_dir / f'{task.clip_stem}_raw_kps.npy'
    ndet_path = ctx.raw_kps_dir / f'{task.clip_stem}_raw_ndet.npy'
    if not (shuttle_path.exists() and raw_kps_path.exists() and ndet_path.exists()):
        return _empty_result('missing_file')

    track = np.load(shuttle_path)
    raw_kps = np.load(raw_kps_path)
    ndet = np.load(ndet_path)
    if len(track) != len(raw_kps) or len(ndet) != len(raw_kps):
        return _empty_result('length_mismatch')

    window_len = task.end_f - task.start_f
    window_len_mismatch = abs(len(track) - window_len) > WINDOW_LEN_TOLERANCE

    width, height = ctx.resolution[task.vid]
    rows, stats = measure_clip(
        task.clip_stem, task.vid, task.start_f, task.end_f,
        ctx.vid_frames[task.vid], track, raw_kps, ndet,
        width, height, ctx.tolerance,
    )
    return ClipResult(rows=rows, skip_cause=None, window_len_mismatch=window_len_mismatch, **stats)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _fmt_ratio(value: float | None) -> str:
    """Ratio for the summary text: 3 dp, or 'n/a' when the denominator was 0."""
    return 'n/a' if value is None else f'{value:.3f}'


def _fmt_percentiles(values: list[float]) -> str:
    """Percentile block for one distance group, or 'no rows' when empty."""
    if not values:
        return 'no rows'
    pcts = np.percentile(values, PERCENTILES)
    return '  '.join(f'p{pct}={value:.4f}' for pct, value in zip(PERCENTILES, pcts))


def format_summary(
    kind_counts: dict[str, int], skip_counts: dict[str, int],
    window_mismatch_count: int, dist_groups: dict[str, list[float]],
    totals: dict[str, int], tolerance: int,
) -> str:
    """Render the end-of-run summary text from the aggregated numbers.

    :param kind_counts: measured-row counts keyed 'gt'/'cand'.
    :param skip_counts: skipped-clip counts keyed by cause.
    :param window_mismatch_count: clips whose track length missed the window by
        more than WINDOW_LEN_TOLERANCE.
    :param dist_groups: visible-shuttle min-distance lists keyed
        'gt'/'matched_cand'/'unmatched_cand'.
    :param totals: aggregate match counts (n_gt, n_cand, *_matched_tol/extra,
        n_gt_invisible).
    :param tolerance: the run's match tolerance in frames.
    """
    precision_tol = safe_ratio(totals['cand_matched_tol'], totals['n_cand'])
    recall_tol = safe_ratio(totals['gt_matched_tol'], totals['n_gt'])
    precision_extra = safe_ratio(totals['cand_matched_extra'], totals['n_cand'])
    recall_extra = safe_ratio(totals['gt_matched_extra'], totals['n_gt'])
    invisible_frac = safe_ratio(totals['n_gt_invisible'], totals['n_gt'])

    lines = [
        f'wrist_contact_separation summary  ({datetime.now():%Y-%m-%d %H:%M:%S})',
        '',
        f'Measured rows:  gt={kind_counts["gt"]:,}  cand={kind_counts["cand"]:,}',
        'Skips by cause: ' + (
            '  '.join(f'{cause}={count:,}' for cause, count in sorted(skip_counts.items()))
            or 'none'
        ),
        f'Track-vs-window length mismatch (>{WINDOW_LEN_TOLERANCE} frames): '
        f'{window_mismatch_count:,} clips',
        '',
        'min_wrist_dist percentiles (visible-shuttle rows only):',
        f'  gt contacts    : {_fmt_percentiles(dist_groups["gt"])}',
        f'  matched cands  : {_fmt_percentiles(dist_groups["matched_cand"])}',
        f'  unmatched cands: {_fmt_percentiles(dist_groups["unmatched_cand"])}',
        '',
        f'Candidate precision  @+/-{tolerance}: {_fmt_ratio(precision_tol)}'
        f'   (matched {totals["cand_matched_tol"]:,} / {totals["n_cand"]:,})',
        f'Candidate precision  @+/-{EXTRA_TOLERANCE}: {_fmt_ratio(precision_extra)}',
        f'GT recall            @+/-{tolerance}: {_fmt_ratio(recall_tol)}'
        f'   (recalled {totals["gt_matched_tol"]:,} / {totals["n_gt"]:,})',
        f'GT recall            @+/-{EXTRA_TOLERANCE}: {_fmt_ratio(recall_extra)}',
        '',
        f'GT frames with invisible shuttle: {_fmt_ratio(invisible_frac)}'
        f'   ({totals["n_gt_invisible"]:,} / {totals["n_gt"]:,})',
        '',
        'Note: clip windows overlap, so one physical contact contributes up to '
        '~3 rows across neighbouring clips. Read these as distributions, not '
        'unique-contact counts.',
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_tasks(
    master: pd.DataFrame, resolution: dict[int, tuple[float, float]],
) -> tuple[list[ClipTask], int]:
    """One task per shots_master row whose video has a resolution entry.

    :return: ``(tasks, n_missing_resolution)``; rows for videos absent from the
        resolution map are dropped and counted (they can't be normalised).
    """
    tasks: list[ClipTask] = []
    n_missing_resolution = 0
    for row in master.itertuples(index=False):
        vid = int(row.vid)
        if vid not in resolution:
            n_missing_resolution += 1
            continue
        tasks.append(ClipTask(
            clip_stem=str(row.clip_stem),
            vid=vid,
            start_f=int(row.shuttle_start_f),
            end_f=int(row.shuttle_end_f),
        ))
    return tasks, n_missing_resolution


def run(
    master: pd.DataFrame, vid_frames: dict[int, np.ndarray],
    resolution: dict[int, tuple[float, float]], shuttle_dir: Path,
    raw_kps_dir: Path, out_csv: Path, workers: int, tolerance: int,
) -> str:
    """Process every clip, stream rows to the CSV, and return the summary text.

    Results stream via ``imap_unordered`` and rows are written as they arrive, so
    a run leaves a usable partial CSV if interrupted and never holds all rows in
    memory at once.
    """
    tasks, n_missing_resolution = build_tasks(master, resolution)
    ctx = WorkerCtx(vid_frames, resolution, shuttle_dir, raw_kps_dir, tolerance)

    kind_counts = {'gt': 0, 'cand': 0}
    skip_counts: dict[str, int] = {}
    if n_missing_resolution:
        skip_counts['missing_resolution'] = n_missing_resolution
    window_mismatch_count = 0
    dist_groups: dict[str, list[float]] = {'gt': [], 'matched_cand': [], 'unmatched_cand': []}
    totals = {
        'n_gt': 0, 'n_cand': 0,
        'cand_matched_tol': 0, 'cand_matched_extra': 0,
        'gt_matched_tol': 0, 'gt_matched_extra': 0,
        'n_gt_invisible': 0,
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        for result in _iter_results(tasks, ctx, workers):
            if result.skip_cause is not None:
                skip_counts[result.skip_cause] = skip_counts.get(result.skip_cause, 0) + 1
                continue
            if result.window_len_mismatch:
                window_mismatch_count += 1
            for key in totals:
                totals[key] += getattr(result, key)
            for row in result.rows:
                kind_counts[row['kind']] += 1
                writer.writerow(_serialise_row(row))
                _accumulate_distance(row, dist_groups)

    return format_summary(
        kind_counts, skip_counts, window_mismatch_count, dist_groups, totals, tolerance,
    )


def _iter_results(tasks: list[ClipTask], ctx: WorkerCtx, workers: int):
    """Yield a ClipResult per task, serial when workers==1 else via a pool.

    The serial path sets the module context directly (easy to debug); the pool
    path hands the same context to each worker through the initializer.
    """
    if workers <= 1:
        _init_worker(ctx)
        for task in tasks:
            yield process_clip(task)
        return
    with Pool(processes=workers, initializer=_init_worker, initargs=(ctx,)) as pool:
        yield from pool.imap_unordered(process_clip, tasks, chunksize=16)


def _serialise_row(row: dict) -> dict:
    """CSV form of a row: bools as True/False, None as blank, distance rounded."""
    min_dist = row['min_wrist_dist']
    return {
        'clip_stem': row['clip_stem'],
        'vid': row['vid'],
        'kind': row['kind'],
        'frame_local': row['frame_local'],
        'frame_source': row['frame_source'],
        'matched': 'True' if row['matched'] else 'False',
        'shuttle_visible': 'True' if row['shuttle_visible'] else 'False',
        'n_det': '' if row['n_det'] is None else row['n_det'],
        'min_wrist_dist': '' if min_dist is None else round(min_dist, 6),
    }


def _accumulate_distance(row: dict, dist_groups: dict[str, list[float]]) -> None:
    """File a measured distance into its percentile group (visible shuttle only)."""
    if not row['shuttle_visible'] or row['min_wrist_dist'] is None:
        return
    if row['kind'] == 'gt':
        dist_groups['gt'].append(row['min_wrist_dist'])
    elif row['matched']:
        dist_groups['matched_cand'].append(row['min_wrist_dist'])
    else:
        dist_groups['unmatched_cand'].append(row['min_wrist_dist'])


def load_resolution(resolution_csv: Path) -> dict[int, tuple[float, float]]:
    """Map ``vid -> (width, height)`` from the shuttle-normalisation CSV."""
    res_df = pd.read_csv(resolution_csv).set_index('id')
    return {
        int(vid): (float(row['width']), float(row['height']))
        for vid, row in res_df.iterrows()
    }


def load_vid_frames(master: pd.DataFrame) -> dict[int, np.ndarray]:
    """Map ``vid -> sorted array of every stroke frame_num`` in that video."""
    return {
        int(vid): np.sort(grp['frame_num'].to_numpy(dtype=int))
        for vid, grp in master.groupby('vid')
    }


def main() -> None:
    default_master = REPO_ROOT / 'training' / 'data' / 'shuttleset' / 'annotations' / 'shots_master.csv'
    default_shuttle = REPO_ROOT / 'data' / 'shuttleset' / 'shuttle_npy'
    default_resolution = REPO_ROOT / 'data' / 'shuttleset' / 'my_raw_video_resolution.csv'

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shots-master', type=Path, default=default_master,
                        help='shots_master.csv (one row per stroke)')
    parser.add_argument('--shuttle-dir', type=Path, default=default_shuttle,
                        help='directory of per-clip {stem}.npy shuttle tracks')
    parser.add_argument('--raw-kps-dir', type=Path, required=True,
                        help='directory of {stem}_raw_kps.npy and {stem}_raw_ndet.npy')
    parser.add_argument('--resolution-csv', type=Path, default=default_resolution,
                        help='video resolution CSV; the SAME source the shuttle track was normalised by')
    parser.add_argument('--out-csv', type=Path, required=True,
                        help='output CSV, one row per measured frame')
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1,
                        help='multiprocessing pool size (default: all cores)')
    parser.add_argument('--tolerance', type=int, default=2,
                        help='frame tolerance for the true/false match (default 2)')
    args = parser.parse_args()

    master = pd.read_csv(args.shots_master)
    resolution = load_resolution(args.resolution_csv)
    vid_frames = load_vid_frames(master)
    print(f'Loaded {len(master):,} strokes across {master["vid"].nunique()} videos')
    print(f'Workers: {args.workers}   tolerance: +/-{args.tolerance} frames')

    summary = run(
        master, vid_frames, resolution, args.shuttle_dir, args.raw_kps_dir,
        args.out_csv, args.workers, args.tolerance,
    )

    print('\n' + summary)
    summary_path = args.out_csv.with_name(args.out_csv.stem + '_summary.txt')
    with summary_path.open('a', encoding='utf-8') as handle:
        handle.write(summary + '\n\n' + '=' * 70 + '\n\n')
    print(f'\nWrote rows to {args.out_csv}')
    print(f'Appended summary to {summary_path}')


if __name__ == '__main__':
    main()
