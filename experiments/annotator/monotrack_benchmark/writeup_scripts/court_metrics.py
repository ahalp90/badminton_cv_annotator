"""Per-rally court error against ShuttleSet22's official corners, for the MonoTrack report.

Three court sources are measured on every labelled rally window:

- ``ours_rally``: our detector, one court per supplied rally window, after the person
  vote. Our annotator and MonoTrack both used these in the "our courts" conditions.
- ``monotrack``: MonoTrack's released detector, one court per supplied rally window.
- ``ours_discovery``: our detector's stored fast-robust courts, one per accepted
  PySceneDetect scene. A rally's court is the per-coordinate median of the courts in
  force on each frame of its window, so a scene covering more frames weighs more.

The official corners are one set per video, in 1280 x 720 pixels. They are scaled to the
native 1920 x 1080 frames. They fit only the main camera view, so a rally filmed from
another view reads as a large error even when its court is right.

Corner error pairs corners by position on the outline (top-left with top-left, and so on)
after sorting each quadrilateral into image order. ShuttleSet22's ``downleft`` and
``downright`` columns name the near corners the other way round, so names are not used.

Usage: python court_metrics.py --out DIR
Writes ``rally_courts.csv.gz`` (one row per rally) and ``court_summary.json``.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
INPUTS = REPO / 'experiments/annotator/monotrack_benchmark/benchmark_results/inputs'
OFFICIAL_CSV = REPO / 'data/shuttleset22/set/homography.csv.gz'
OFFICIAL_SIZE = np.array([1280.0, 720.0])
OFFICIAL_COLUMNS = (('upleft_x', 'upleft_y'), ('upright_x', 'upright_y'),
                    ('downleft_x', 'downleft_y'), ('downright_x', 'downright_y'))
ROW_COLUMNS = (('upleft_x', 'upleft_y'), ('upright_x', 'upright_y'),
               ('downleft_x', 'downleft_y'), ('downright_x', 'downright_y'))
SOURCES = ('ours_rally', 'monotrack', 'ours_discovery')
HIGH_ERROR_PX = 15.0


def read_json(path: Path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def image_order(corners: np.ndarray) -> np.ndarray:
    """Return four points as image TL, TR, BR, BL: the two highest, then the two lowest.

    :param corners: (4, 2) x, y points in any order.
    :return: (4, 2) points in outline order.
    """
    top, bottom = np.split(corners[np.argsort(corners[:, 1], kind='stable')], 2)
    top = top[np.argsort(top[:, 0], kind='stable')]
    bottom = bottom[np.argsort(bottom[:, 0], kind='stable')]
    return np.array([top[0], top[1], bottom[1], bottom[0]])


def official_table() -> dict[int, dict]:
    with gzip.open(OFFICIAL_CSV, 'rt') as stream:
        return {int(row['id']): row for row in csv.DictReader(io.StringIO(stream.read()))}


def official_corners(row: dict, native_size: np.ndarray) -> np.ndarray:
    corners = np.array([[float(row[x]), float(row[y])] for x, y in OFFICIAL_COLUMNS])
    return image_order(corners * (native_size / OFFICIAL_SIZE))


def mean_corner_error(detected: np.ndarray, official: np.ndarray) -> float:
    return float(np.linalg.norm(image_order(detected) - official, axis=1).mean())


def outline_matched_error(detected: np.ndarray, official: np.ndarray) -> float:
    """Smallest mean error over the eight ways to walk a quadrilateral's outline.

    A cross-check on ``image_order``: the two agree unless a fit is badly tilted.
    """
    ordered = image_order(detected)
    walks = [np.roll(ordered, shift, axis=0) for shift in range(4)]
    walks += [np.roll(ordered[::-1], shift, axis=0) for shift in range(4)]
    return float(min(np.linalg.norm(walk - official, axis=1).mean() for walk in walks))


def iou(detected: np.ndarray, official: np.ndarray) -> float:
    """Overlap area over union area of the two court outlines.

    One badly failed fit crosses itself in image order. Its convex hull stands in for it,
    which can only overstate that fit's overlap.
    """
    first = image_order(detected).astype(np.float32)
    if not cv2.isContourConvex(first):
        first = cv2.convexHull(first).reshape(-1, 2)
    second = official.astype(np.float32)
    if not cv2.isContourConvex(second):
        raise ValueError('official corners must form a convex court')
    overlap, _ = cv2.intersectConvexConvex(first, second)
    union = cv2.contourArea(first) + cv2.contourArea(second) - overlap
    return float(overlap / union)


def per_rally_courts(path: Path, spans: list[dict]) -> list[np.ndarray | None]:
    """Court corners for each supplied rally window, or None where the window has none."""
    records = read_json(path)['scene_records']
    by_bounds = {(record['start_frame'], record['end_frame']): record for record in records}
    courts = []
    for span in spans:
        record = by_bounds[span['start_frame'], span['end_frame']]
        courts.append(np.asarray(record['corners_native_px'], dtype=float) if record['scene_valid'] else None)
    return courts


def discovery_courts(path: Path, spans: list[dict], frame_count: int) -> tuple[list, list[float]]:
    """Median court over each rally window's frames, and the fraction of frames with a court."""
    rows = read_json(path)['inputs']['homography_rows']
    table = pd.DataFrame(rows['data'], columns=rows['columns'])
    # (frames, 4, 2); NaN where no accepted scene court is in force.
    per_frame = np.full((frame_count, 4, 2), np.nan)
    for row in table.itertuples(index=False):
        corners = np.array([[getattr(row, x), getattr(row, y)] for x, y in ROW_COLUMNS])
        per_frame[row.start_frame:row.end_frame] = image_order(corners)
    courts, coverage = [], []
    for span in spans:
        window = per_frame[span['start_frame']:span['end_frame']]
        has_court = ~np.isnan(window[:, 0, 0])
        coverage.append(float(has_court.mean()))
        courts.append(np.median(window[has_court], axis=0) if has_court.any() else None)
    return courts, coverage


def collect() -> pd.DataFrame:
    table = official_table()
    manifest = read_json(INPUTS / 'manifest.json.gz')
    rows = []
    for item in manifest['items']:
        video = item['video_id']
        official_row = table[video]
        number, name = item['directory'].split(' ', 1)
        if int(number) != video or name != official_row['video']:
            raise ValueError(f'video {video}: {item["directory"]} does not match {official_row["video"]}')
        native_size = np.array(item['resolution'], dtype=float)
        official = official_corners(official_row, native_size)
        spans = item['spans']
        courts = {
            'ours_rally': per_rally_courts(INPUTS / 'ours' / item['item_id'] / 'court_evidence.json.gz', spans),
            'monotrack': per_rally_courts(INPUTS / 'monotrack' / item['item_id'] / 'court_evidence.json.gz', spans),
        }
        courts['ours_discovery'], coverage = discovery_courts(
            INPUTS / 'discovery' / item['item_id'] / 'court_evidence.json.gz', spans, item['frame_count'])
        for rally_index, span in enumerate(spans):
            row = {'video': video, 'rally': rally_index + 1, 'rally_id': span['rally_id'],
                   'start_frame': span['start_frame'], 'end_frame': span['end_frame'],
                   'midpoint_frame': (span['start_frame'] + span['end_frame']) // 2,
                   'discovery_coverage': coverage[rally_index]}
            row['official'] = json.dumps(np.round(official, 2).tolist())
            for source in SOURCES:
                detected = courts[source][rally_index]
                if detected is None:
                    row |= {f'{source}_error_px': np.nan, f'{source}_outline_error_px': np.nan,
                            f'{source}_iou': np.nan, f'{source}_corners': ''}
                    continue
                row |= {f'{source}_error_px': mean_corner_error(detected, official),
                        f'{source}_outline_error_px': outline_matched_error(detected, official),
                        f'{source}_iou': iou(detected, official),
                        f'{source}_corners': json.dumps(np.round(image_order(detected), 2).tolist())}
            rows.append(row)
    return pd.DataFrame(rows)


def describe(errors: pd.Series, ious: pd.Series) -> dict:
    present = errors.dropna()
    return {'rallies_with_court': len(present), 'rallies_without_court': int(errors.isna().sum()),
            'median_error_px': round(float(present.median()), 2),
            'p90_error_px': round(float(present.quantile(0.9)), 2),
            'over_15px': int((present > HIGH_ERROR_PX).sum()),
            'median_iou': round(float(ious.dropna().median()), 4),
            'p10_iou': round(float(ious.dropna().quantile(0.1)), 4)}


def summarise(frame: pd.DataFrame) -> dict:
    all_present = frame[[f'{source}_error_px' for source in SOURCES]].notna().all(axis=1)
    summary = {'rallies': len(frame), 'videos': int(frame['video'].nunique()),
               'all_rallies': {}, 'rallies_where_all_three_have_a_court': {'rallies': int(all_present.sum())}}
    for source in SOURCES:
        errors, ious = frame[f'{source}_error_px'], frame[f'{source}_iou']
        summary['all_rallies'][source] = describe(errors, ious)
        summary['rallies_where_all_three_have_a_court'][source] = describe(errors[all_present], ious[all_present])
        ordering_gap = (frame[f'{source}_error_px'] - frame[f'{source}_outline_error_px']).abs()
        summary['all_rallies'][source]['rallies_where_outline_matching_changes_error'] = int((ordering_gap > 0.01).sum())
        per_video = frame.groupby('video')[f'{source}_error_px'].median()
        summary['all_rallies'][source]['distinct_courts'] = int(frame[f'{source}_corners'].replace('', np.nan).nunique())
        summary['all_rallies'][source]['median_of_video_medians_px'] = round(float(per_video.median()), 2)
    over = frame[[f'{source}_error_px' for source in SOURCES]].gt(HIGH_ERROR_PX).any(axis=1)
    summary['rallies_over_15px_any_source'] = int(over.sum())
    summary['discovery'] = {
        'rallies_with_no_court_frames': int((frame['discovery_coverage'] == 0).sum()),
        'rallies_partly_covered': int(((frame['discovery_coverage'] > 0) & (frame['discovery_coverage'] < 1)).sum()),
        'median_coverage': round(float(frame['discovery_coverage'].median()), 4),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = collect()
    frame.to_csv(args.out / 'rally_courts.csv.gz', index=False)
    summary = summarise(frame)
    (args.out / 'court_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
