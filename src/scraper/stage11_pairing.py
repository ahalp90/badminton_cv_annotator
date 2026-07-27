"""Stage 11: rally-to-commentary pairing (scraper_spec.md section 9).

A mechanical time-range join: each rally span pairs with the commentary chunk
that immediately follows it. Replay-masked rallies and chunks are held out of
pairing but kept (unpaired), per the schema's keep-with-flag rule.

Mixed units, deliberate. `rally_start`/`rally_end` stay in FRAMES (provenance to
`rally_spans.csv`), while `commentary_start`/`commentary_end` are SECONDS (the
chunk sidecar's native unit). Each field keeps its producer's unit so nothing is
silently converted; downstream assembly derives seconds from frames via the
per-video fps when it wants both on one clock.

Run as `python -m scraper.stage11_pairing` with PYTHONPATH=src.
"""
import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from annotator.fps_constants import scale_for_fps
from annotator.replay_mask import believe_raw_mask
from .config import CHUNKS_DIR, MASKS_DIR, PAIRS_CSV, PAIR_WINDOW_S, RALLY_SPANS_CSV, SCRAPE_DIR

log = logging.getLogger(__name__)

# Not in config: a local path built from SCRAPE_DIR, and the video extensions
# build_video_fps_csv scans. Neither is a tunable rule constant.
VIDEO_FPS_CSV = SCRAPE_DIR / 'video_fps.csv'
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}

PAIRS_COLUMNS = [
    'video_id', 'rally_id',
    'rally_start', 'rally_end',  # FRAMES (from rally_spans.csv)
    'chunk_id',
    'commentary_start', 'commentary_end',  # SECONDS (native chunk units)
]


# ---------------------------------------------------------------------------
# fps sidecar
# ---------------------------------------------------------------------------
def build_video_fps_csv(video_dir: Path, out_csv: Path = VIDEO_FPS_CSV) -> Path:
    """Read fps per video file into `video_fps.csv` (columns video_id, fps).

    The `video_id` is the file stem, matching `<video_id>.<ext>` against the
    rally spans and chunk sidecars.

    :param video_dir: directory of downloaded video files.
    :param out_csv: destination CSV (defaults to SCRAPE_DIR/video_fps.csv).
    :return: the path written.
    """
    if not video_dir.is_dir():
        raise FileNotFoundError(f'video dir not found: {video_dir}')

    rows: list[tuple[str, float]] = []
    for path in sorted(video_dir.iterdir()):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        rows.append((path.stem, float(fps)))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['video_id', 'fps'])
        writer.writerows(rows)
    log.info('wrote fps for %d videos -> %s', len(rows), out_csv)
    return out_csv


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def _chunk_start_on_mask(start_s: float, fps: float, replay_mask: np.ndarray) -> bool:
    """True if the chunk's start time lands on a masked frame (so it is unpairable)."""
    frame = int(start_s * fps)
    return 0 <= frame < len(replay_mask) and bool(replay_mask[frame])


def _believed_replay_in_rally_interior(
    believed_mask: np.ndarray, start_frame: int, end_frame: int, grace: int,
) -> bool:
    """Return whether believed replay lies in a rally's interior after boundary grace.

    A rally's asserted start and end get ``grace`` frames for measurement error.
    Only believed replay deeper than that grace from either asserted boundary
    disqualifies the rally. An empty interior never disqualifies it.
    """
    if start_frame < 0 or end_frame < start_frame or end_frame > len(believed_mask):
        raise ValueError(
            f'rally span [{start_frame}, {end_frame}) is outside replay mask of length {len(believed_mask)}'
        )
    return bool(believed_mask[start_frame + grace : max(start_frame + grace, end_frame - grace)].any())


def pair_video(
    video_id: str,
    rally_spans: list[tuple[int, int, int]],
    chunks: list[dict],
    replay_mask: np.ndarray | None,
    fps: float,
) -> list[dict]:
    """Pair one video's rallies to commentary chunks.

    A rally pairs with the first chunk whose start falls in
    `(rally_end_t, rally_end_t + PAIR_WINDOW_S]`, where `rally_end_t = end_frame
    / fps`. A rally overlapping the replay mask is held out (kept, unpaired); a
    chunk whose start lands on a masked frame is not pairable. Every rally
    yields exactly one row (blank commentary fields when unpaired), the
    keep-with-flag default.

    A chunk pairs with at most one rally: rallies are processed in id order and a
    claimed chunk is skipped thereafter, so when two rallies' windows both cover
    a chunk the earlier rally wins. The spec is silent on this tie; earlier-rally
    -wins matches the "immediately succeeds" intent (the nearer rally in time).

    :param video_id: the video id.
    :param rally_spans: `[(rally_id, start_frame, end_frame), ...]`.
    :param chunks: `[{chunk_id, start, end, text}, ...]`, times in seconds.
    :param replay_mask: `(frames,)` bool mask, or None.
    :param fps: frames per second for this video.
    :return: one row dict per rally, keyed by PAIRS_COLUMNS.
    """
    sorted_chunks = sorted(chunks, key=lambda chunk: float(chunk['start']))
    claimed: set = set()  # chunk_ids already paired
    rows: list[dict] = []
    minimum_run = scale_for_fps(fps).replay_mask_min_frames
    believed_mask = None if replay_mask is None else believe_raw_mask(replay_mask, minimum_run)

    for rally_id, start_frame, end_frame in sorted(rally_spans):
        row = {
            'video_id': video_id, 'rally_id': rally_id,
            'rally_start': start_frame, 'rally_end': end_frame,
            'chunk_id': '', 'commentary_start': '', 'commentary_end': '',
        }
        rally_masked = believed_mask is not None and _believed_replay_in_rally_interior(
            believed_mask, start_frame, end_frame, minimum_run,
        )
        if rally_masked:
            rows.append(row)  # kept, held out of pairing
            continue

        rally_end_t = end_frame / fps
        window_hi = rally_end_t + PAIR_WINDOW_S
        for chunk in sorted_chunks:  # ascending start: first in window wins
            chunk_id = chunk['chunk_id']
            if chunk_id in claimed:
                continue
            start_s = float(chunk['start'])
            if start_s <= rally_end_t:
                continue
            if start_s > window_hi:
                break  # sorted ascending: nothing later can land in window
            if believed_mask is not None and _chunk_start_on_mask(start_s, fps, believed_mask):
                continue  # chunk start on a replay frame is unpairable
            claimed.add(chunk_id)
            row['chunk_id'] = chunk_id
            row['commentary_start'] = chunk['start']
            row['commentary_end'] = chunk['end']
            break
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_fps_map(fps_csv: Path) -> dict[str, float]:
    """Read video_fps.csv into a `{video_id: fps}` map."""
    if not fps_csv.exists():
        raise FileNotFoundError(f'{fps_csv} not found. Build it with build_video_fps_csv first.')
    with fps_csv.open(newline='', encoding='utf-8') as handle:
        return {row['video_id']: float(row['fps']) for row in csv.DictReader(handle)}


def _read_rally_spans_by_video(spans_csv: Path) -> dict[str, list[tuple[int, int, int]]]:
    """Group rally spans by video: `{video_id: [(rally_id, start, end), ...]}`."""
    if not spans_csv.exists():
        raise FileNotFoundError(f'{spans_csv} not found. Run stage 8 first.')
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    with spans_csv.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            grouped[row['video_id']].append(
                (int(row['rally_id']), int(row['start_frame']), int(row['end_frame']))
            )
    return grouped


def _load_chunks(chunks_dir: Path, video_id: str) -> list[dict]:
    """Load `<video_id>.json` chunk sidecar, or [] if absent."""
    chunk_path = chunks_dir / f'{video_id}.json'
    if not chunk_path.exists():
        return []
    with chunk_path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _load_replay_mask(masks_dir: Path, video_id: str) -> np.ndarray | None:
    """Load a one-dimensional boolean `<video_id>_replay.npy`, or None if absent."""
    mask_path = masks_dir / f'{video_id}_replay.npy'
    if not mask_path.exists():
        return None
    replay_mask = np.load(mask_path)
    if replay_mask.ndim != 1 or replay_mask.dtype != np.bool_:
        raise ValueError(f'{mask_path} must be a one-dimensional boolean array')
    return replay_mask


def main() -> None:
    parser = argparse.ArgumentParser(description='Stage 11: pair rallies to commentary chunks.')
    parser.add_argument('--rally-spans', type=Path, default=RALLY_SPANS_CSV)
    parser.add_argument('--chunks-dir', type=Path, default=CHUNKS_DIR)
    parser.add_argument('--masks-dir', type=Path, default=MASKS_DIR)
    parser.add_argument('--fps-csv', type=Path, default=VIDEO_FPS_CSV)
    parser.add_argument('--pairs-csv', type=Path, default=PAIRS_CSV)
    parser.add_argument('--build-fps-from', type=Path, default=None,
                        help='If given, (re)build the fps CSV from this video dir before pairing')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if args.build_fps_from is not None:
        build_video_fps_csv(args.build_fps_from, args.fps_csv)

    fps_map = _read_fps_map(args.fps_csv)
    spans_by_video = _read_rally_spans_by_video(args.rally_spans)

    all_rows: list[dict] = []
    for video_id, rally_spans in spans_by_video.items():
        if video_id not in fps_map:
            log.warning('no fps for %s; skipping its rallies', video_id)  # log-and-skip per video
            continue
        chunks = _load_chunks(args.chunks_dir, video_id)
        replay_mask = _load_replay_mask(args.masks_dir, video_id)
        rows = pair_video(video_id, rally_spans, chunks, replay_mask, fps_map[video_id])
        all_rows.extend(rows)
        paired = sum(1 for row in rows if row['chunk_id'])
        log.info('%s: %d rallies, %d paired', video_id, len(rows), paired)

    args.pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRS_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    log.info('wrote %d pair rows -> %s', len(all_rows), args.pairs_csv)


if __name__ == '__main__':
    main()
