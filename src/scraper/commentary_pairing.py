"""Pair rallies with commentary chunks.

A mechanical time-range join: each rally span pairs with the commentary chunk
that immediately follows it. Replay-masked rallies and chunks are held out of
pairing but kept (unpaired), per the schema's keep-with-flag rule.

Mixed units, deliberate. `rally_start`/`rally_end` stay in FRAMES (provenance to
`rally_spans.csv`), while `commentary_start`/`commentary_end` are SECONDS (the
chunk sidecar's native unit). Each field keeps its producer's unit so nothing is
silently converted; downstream assembly derives seconds from frames via the
per-video fps when it wants both on one clock.

Run as `python -m scraper.commentary_pairing` with PYTHONPATH=src.

See ``docs/scraper_pipeline/scraper_architecture.md`` for the stage and file
contracts.
"""
import argparse
import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import tomllib
from collections import defaultdict
from pathlib import Path

import cv2
from frozendict import frozendict
import numpy as np

from annotator.fps_constants import scale_for_fps
from annotator.replay_mask import filter_short_exclusion_runs
from annotator.video_metadata import VideoMetadata
from .config import (
    CHUNKS_DIR,
    MASKS_DIR,
    PAIRS_CSV,
    PAIR_WINDOW_S,
    RALLY_SPANS_CSV,
    SCRAPE_DIR,
    SOURCES_MANIFEST_NAME,
    VIDEO_EXTENSIONS,
    VIDEOS_DIR,
)

log = logging.getLogger(__name__)

# This path is local to the pairing stage, while supported extensions are
# shared with the downloader and commentary cleaner.
VIDEO_FPS_CSV = SCRAPE_DIR / 'video_fps.csv'

PAIRS_COLUMNS = [
    'video_id', 'rally_id',
    'rally_start', 'rally_end',  # FRAMES (from rally_spans.csv)
    'chunk_id',
    'commentary_start', 'commentary_end',  # SECONDS (native chunk units)
]


@dataclass(frozen=True)
class CanonicalPairing:
    """Pair rows tied to the exact canonical timing metadata they consumed."""

    video_id: str
    metadata: VideoMetadata
    rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id:
            raise ValueError('canonical pairing video_id must be a non-empty string')
        if not isinstance(self.metadata, VideoMetadata):
            raise TypeError('canonical pairing metadata must be VideoMetadata')
        object.__setattr__(self, 'rows', tuple(frozendict(row) for row in self.rows))


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
    duration_filtered_replay_mask: np.ndarray, start_frame: int, end_frame: int, grace: int,
) -> bool:
    """Return whether duration-filtered replay lies in a rally's interior after boundary grace.

    A rally's asserted start and end get ``grace`` frames for measurement error.
    Only believed replay deeper than that grace from either asserted boundary
    disqualifies the rally. An empty interior never disqualifies it.
    """
    if (start_frame < 0 or end_frame < start_frame
            or end_frame > len(duration_filtered_replay_mask)):
        raise ValueError(
            f'rally span [{start_frame}, {end_frame}) is outside replay mask '
            f'of length {len(duration_filtered_replay_mask)}'
        )
    return bool(
        duration_filtered_replay_mask[
            start_frame + grace : max(start_frame + grace, end_frame - grace)
        ].any()
    )


def pair_video(
    video_id: str,
    rally_spans: list[tuple[int, int, int]],
    chunks: list[dict],
    replay_mask: np.ndarray | None,
    fps: float,
    *,
    pair_window_s: float = PAIR_WINDOW_S,
) -> list[dict]:
    """Pair one video's rallies to commentary chunks.

    A rally pairs with the first chunk whose start falls in
    `(rally_end_t, rally_end_t + pair_window_s]`, where `rally_end_t = end_frame
    / fps`. The default window is the configured `PAIR_WINDOW_S`. A rally
    overlapping the replay mask is held out (kept, unpaired); a
    chunk whose start lands on a masked frame is not pairable. Every rally
    yields exactly one row (blank commentary fields when unpaired), the
    keep-with-flag default.

    A chunk pairs with at most one rally: rallies are processed in id order and a
    claimed chunk is skipped thereafter, so when two rallies' windows both cover
    a chunk the earlier rally wins. This is a deterministic processing-order
    tie-break and does not compare which rally is nearer to the chunk.

    :param video_id: the video id.
    :param rally_spans: `[(rally_id, start_frame, end_frame), ...]`.
    :param chunks: `[{chunk_id, start, end, text}, ...]`, times in seconds.
    :param replay_mask: `(frames,)` bool mask, or None.
    :param fps: frames per second for this video.
    :param pair_window_s: positive post-rally pairing window in seconds.
    :return: one row dict per rally, keyed by PAIRS_COLUMNS.
    """
    if (
        isinstance(pair_window_s, bool)
        or not isinstance(pair_window_s, (int, float))
        or not np.isfinite(pair_window_s)
        or pair_window_s <= 0
    ):
        raise ValueError('pair_window_s must be a positive finite number')
    sorted_chunks = sorted(chunks, key=lambda chunk: float(chunk['start']))
    claimed: set = set()  # chunk_ids already paired
    rows: list[dict] = []
    minimum_run = scale_for_fps(fps).replay_mask_min_frames
    duration_filtered_replay_mask = (
        None if replay_mask is None
        else filter_short_exclusion_runs(replay_mask, minimum_run)
    )

    for rally_id, start_frame, end_frame in sorted(rally_spans):
        row = {
            'video_id': video_id, 'rally_id': rally_id,
            'rally_start': start_frame, 'rally_end': end_frame,
            'chunk_id': '', 'commentary_start': '', 'commentary_end': '',
        }
        rally_masked = (
            duration_filtered_replay_mask is not None
            and _believed_replay_in_rally_interior(
                duration_filtered_replay_mask, start_frame, end_frame, minimum_run,
            )
        )
        if rally_masked:
            rows.append(row)  # kept, held out of pairing
            continue

        rally_end_t = end_frame / fps
        window_hi = rally_end_t + pair_window_s
        for chunk in sorted_chunks:  # ascending start: first in window wins
            chunk_id = chunk['chunk_id']
            if chunk_id in claimed:
                continue
            start_s = float(chunk['start'])
            if start_s <= rally_end_t:
                continue
            if start_s > window_hi:
                break  # sorted ascending: nothing later can land in window
            if (duration_filtered_replay_mask is not None
                    and _chunk_start_on_mask(start_s, fps, duration_filtered_replay_mask)):
                continue  # chunk start on a replay frame is unpairable
            claimed.add(chunk_id)
            row['chunk_id'] = chunk_id
            row['commentary_start'] = chunk['start']
            row['commentary_end'] = chunk['end']
            break
        rows.append(row)
    return rows


def pair_video_with_metadata(
    video_id: str,
    rally_spans: Sequence[tuple[int, int, int]],
    chunks: Sequence[Mapping[str, object]],
    replay_mask: np.ndarray,
    metadata: VideoMetadata,
) -> CanonicalPairing:
    """Pair one video on the canonical CFR timing and frame-count contract."""
    if not isinstance(metadata, VideoMetadata):
        raise TypeError('metadata must be canonical VideoMetadata')
    if not isinstance(video_id, str) or not video_id:
        raise ValueError('canonical pairing video_id must be a non-empty string')
    if (
        not isinstance(replay_mask, np.ndarray)
        or replay_mask.ndim != 1
        or replay_mask.dtype != np.bool_
    ):
        raise ValueError('canonical replay mask must be a one-dimensional boolean array')
    if len(replay_mask) != metadata.frame_count:
        raise ValueError(
            f'replay mask length {len(replay_mask)} does not match canonical '
            f'frame_count {metadata.frame_count}'
        )
    seen_rallies: set[int] = set()
    for rally_id, start_frame, end_frame in rally_spans:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (rally_id, start_frame, end_frame)
        ):
            raise ValueError('canonical rally ids and frame bounds must be integers')
        if rally_id in seen_rallies:
            raise ValueError(f'canonical pairing rally_id is duplicated: {rally_id}')
        if not 0 <= start_frame < end_frame <= metadata.frame_count:
            raise ValueError(
                f'canonical rally span [{start_frame}, {end_frame}) is outside '
                f'frame_count {metadata.frame_count}'
            )
        seen_rallies.add(rally_id)
    if seen_rallies != set(range(len(rally_spans))):
        raise ValueError('canonical pairing rally_ids must be contiguous from zero')
    rows = pair_video(
        video_id,
        list(rally_spans),
        [dict(chunk) for chunk in chunks],
        replay_mask,
        float(metadata.fps),
    )
    return CanonicalPairing(video_id, metadata, tuple(rows))


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
        raise FileNotFoundError(f'{spans_csv} not found. Run rally segmentation first.')
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
        log.warning(
            '%s: commentary-eligible but chunks sidecar is missing: %s',
            video_id,
            chunk_path,
        )
        return []
    with chunk_path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _load_replay_mask(masks_dir: Path, video_id: str) -> np.ndarray | None:
    """Load a one-dimensional boolean `<video_id>_replay.npy`, or None if absent."""
    mask_path = masks_dir / f'{video_id}_replay.npy'
    if not mask_path.exists():
        log.info(
            '%s: replay mask is missing; pairing without replay filtering: %s',
            video_id,
            mask_path,
        )
        return None
    replay_mask = np.load(mask_path)
    if replay_mask.ndim != 1 or replay_mask.dtype != np.bool_:
        raise ValueError(f'{mask_path} must be a one-dimensional boolean array')
    return replay_mask


def _read_sources_manifest(manifest_path: Path) -> dict[str, dict[str, object]]:
    """Read the validated scraper manifest used by the pairing boundary."""
    if not manifest_path.exists():
        raise FileNotFoundError(f'{manifest_path} not found')
    with manifest_path.open('rb') as handle:
        manifest = tomllib.load(handle)
    dataset = manifest.get('dataset')
    if not isinstance(dataset, str):
        raise TypeError("sources.toml 'dataset' must be a string")
    if not dataset.strip():
        raise ValueError("sources.toml 'dataset' must not be empty")
    videos = manifest.get('videos')
    if not isinstance(videos, dict):
        raise TypeError("sources.toml 'videos' must be a table")
    for basename, entry in videos.items():
        if not isinstance(entry, dict):
            raise TypeError(f"sources.toml entry for {basename!r} must be a table")
    return videos


def _manifest_pairing_index(
    videos: dict[str, dict[str, object]],
    video_dir: Path,
    video_ids: set[str],
) -> dict[str, tuple[str, dict[str, object]]]:
    """Map fps-bearing rally ids to exactly one existing manifest entry."""
    index: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for basename, entry in videos.items():
        if 'video_id' not in entry:
            continue
        video_id = entry['video_id']
        if isinstance(video_id, bool) or not isinstance(video_id, (str, int)):
            if Path(basename).stem in video_ids and (video_dir / basename).is_file():
                raise TypeError(f'manifest entry for video {Path(basename).stem!r} has invalid video_id')
            continue
        if str(Path(basename).stem) != str(video_id):
            if str(video_id) in video_ids and (video_dir / basename).is_file():
                raise ValueError(
                    f'manifest basename {basename!r} does not match video_id {video_id!r}'
                )
            continue
        video_path = video_dir / basename
        if not video_path.is_file():
            continue
        index[str(video_id)].append((basename, entry))

    pairing_index: dict[str, tuple[str, dict[str, object]]] = {}
    for video_id in video_ids:
        matches = index.get(video_id, [])
        if not matches:
            raise ValueError(f'no existing manifest entry for video {video_id!r}')
        if len(matches) != 1:
            names = ', '.join(basename for basename, _entry in matches)
            raise ValueError(f'multiple manifest entries for video {video_id!r}: {names}')
        basename, entry = matches[0]
        if 'commentary_eligible' not in entry:
            raise ValueError(f'manifest entry for video {video_id!r} lacks commentary_eligible')
        if not isinstance(entry['commentary_eligible'], bool):
            raise TypeError(
                f'manifest entry for video {video_id!r} has non-boolean commentary_eligible'
            )
        pairing_index[video_id] = (basename, entry)
    return pairing_index


def main() -> None:
    parser = argparse.ArgumentParser(description='Pair rallies to commentary chunks.')
    parser.add_argument('--rally-spans', type=Path, default=RALLY_SPANS_CSV)
    parser.add_argument('--chunks-dir', type=Path, default=CHUNKS_DIR)
    parser.add_argument('--masks-dir', type=Path, default=MASKS_DIR)
    parser.add_argument('--fps-csv', type=Path, default=VIDEO_FPS_CSV)
    parser.add_argument('--pairs-csv', type=Path, default=PAIRS_CSV)
    parser.add_argument('--video-dir', type=Path, default=None)
    parser.add_argument('--build-fps-from', type=Path, default=None,
                        help='If given, (re)build the fps CSV from this video dir before pairing')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if args.video_dir is not None and args.build_fps_from is not None:
        video_dir = args.video_dir.resolve()
        build_video_dir = args.build_fps_from.resolve()
        if video_dir != build_video_dir:
            raise ValueError('--video-dir and --build-fps-from must resolve to the same path')
    elif args.video_dir is not None:
        video_dir = args.video_dir
    elif args.build_fps_from is not None:
        video_dir = args.build_fps_from
    else:
        video_dir = VIDEOS_DIR

    if args.build_fps_from is not None:
        build_video_fps_csv(args.build_fps_from, args.fps_csv)

    fps_map = _read_fps_map(args.fps_csv)
    spans_by_video = _read_rally_spans_by_video(args.rally_spans)
    fps_video_ids = {video_id for video_id in spans_by_video if video_id in fps_map}
    pairing_index: dict[str, tuple[str, dict[str, object]]] = {}
    if fps_video_ids:
        manifest_videos = _read_sources_manifest(video_dir / SOURCES_MANIFEST_NAME)
        pairing_index = _manifest_pairing_index(manifest_videos, video_dir, fps_video_ids)

    all_rows: list[dict] = []
    for video_id, rally_spans in spans_by_video.items():
        if video_id not in fps_map:
            log.warning('no fps for %s; skipping its rallies', video_id)  # log-and-skip per video
            continue
        _basename, manifest_entry = pairing_index[video_id]
        commentary_eligible = manifest_entry['commentary_eligible']
        if commentary_eligible:
            chunks = _load_chunks(args.chunks_dir, video_id)
            replay_mask = _load_replay_mask(args.masks_dir, video_id)
        else:
            log.info('%s: commentary-ineligible; rallies kept, commentary left blank', video_id)
            chunks = []
            replay_mask = None
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
