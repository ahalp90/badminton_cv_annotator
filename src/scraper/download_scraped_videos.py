"""Download scraper candidates and record their source and audio status.

The downloader deliberately follows the ShuttleSet downloader's small worker
shape. Scraper downloads differ in two places: they request audio and verify
that the resulting file contains an audio stream before accepting it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import config


CANDIDATES_CSV = config.CANDIDATES_CSV
VIDEOS_DIR = config.VIDEOS_DIR
SOURCES_MANIFEST_NAME = config.SOURCES_MANIFEST_NAME

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}

_H264_WITH_M4A = 'bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]'
_H264_WITH_ANY_AUDIO = 'bestvideo[vcodec^=avc1]+bestaudio'
_H264_PREMUXED = 'best[vcodec^=avc1][ext=mp4]'
_YTDLP_FORMAT = f'{_H264_WITH_M4A}/{_H264_WITH_ANY_AUDIO}/{_H264_PREMUXED}'

_YTDLP_TIMEOUT_S = 1800
_FFPROBE_TIMEOUT_S = 60
_DEFAULT_WORKERS = 2
_SCALAR_TYPES = (str, bool, int, float)
_BARE_KEY = re.compile(r'^[A-Za-z0-9_-]+$')


@dataclass(frozen=True)
class DownloadOutcome:
    """Result for one selected candidate row."""

    video_id: str
    filename: str | None
    entry: dict[str, object] | None
    failed: bool


class _UnreadableMedia(Exception):
    """ffprobe could not read the media file."""


class _AudioProbeTimeout(Exception):
    """ffprobe did not finish within its per-file timeout."""


def _check_ytdlp() -> None:
    """Fail before worker creation when yt-dlp is unavailable."""
    if not shutil.which('yt-dlp'):
        raise RuntimeError('yt-dlp not found in PATH. Install with: pip install yt-dlp')


def _check_ffprobe() -> None:
    """Fail before worker creation when audio verification is required."""
    if not shutil.which('ffprobe'):
        raise RuntimeError('ffprobe not found in PATH. Install ffmpeg to verify audio streams')


def _completed_outputs(output_dir: Path, video_id: str) -> list[Path]:
    """Return completed video files for ``video_id`` in stable order."""
    return [
        path
        for path in sorted(output_dir.iterdir(), key=lambda candidate: candidate.name)
        if path.is_file()
        and path.stem == video_id
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def _bounded_stderr(stderr: object) -> str:
    """Return a short printable subprocess diagnostic."""
    return str(stderr or '').strip()[:200]


def _probe_audio(video_path: Path) -> bool:
    """Return whether ffprobe finds an audio stream in ``video_path``."""
    try:
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise _AudioProbeTimeout from exc

    if result.returncode != 0:
        detail = _bounded_stderr(result.stderr)
        suffix = f': {detail}' if detail else ''
        raise _UnreadableMedia(f'ffprobe exited {result.returncode}{suffix}')

    return any(line.strip() == 'audio' for line in (result.stdout or '').splitlines())


def _manifest_scalar(value: object, *, context: str) -> None:
    """Validate one value supported by the deliberately small TOML writer."""
    if not isinstance(value, _SCALAR_TYPES):
        raise TypeError(f'{context} must be a string, boolean, integer, or finite float')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{context} must be a finite float')


def _validate_manifest(manifest: object) -> dict[str, object]:
    """Validate and return a manifest that the writer can round-trip."""
    if not isinstance(manifest, dict):
        raise TypeError('sources.toml must contain a table')
    if manifest.get('dataset') != 'scraped':
        raise ValueError("sources.toml 'dataset' must be 'scraped'")

    videos = manifest.get('videos')
    if not isinstance(videos, dict):
        raise TypeError("sources.toml 'videos' must be a table")

    for key, value in manifest.items():
        if key != 'videos':
            _manifest_scalar(value, context=f"sources.toml '{key}'")

    for basename, entry in videos.items():
        if not isinstance(entry, dict):
            raise TypeError(f"sources.toml entry for {basename!r} must be a table")
        for key, value in entry.items():
            _manifest_scalar(value, context=f"sources.toml '{basename}.{key}'")
        if 'video_id' in entry and (
            isinstance(entry['video_id'], bool)
            or not isinstance(entry['video_id'], (str, int))
        ):
            raise TypeError(f"sources.toml '{basename}.video_id' must be a string or integer")
        for key in ('title', 'url'):
            if key in entry and not isinstance(entry[key], str):
                raise TypeError(f"sources.toml '{basename}.{key}' must be a string")
        if 'commentary_eligible' in entry and not isinstance(entry['commentary_eligible'], bool):
            raise TypeError(
                f"sources.toml '{basename}.commentary_eligible' must be a boolean"
            )

    return manifest


def _read_manifest(manifest_path: Path) -> dict[str, object]:
    """Read, validate, or initialise a scraper source manifest."""
    if not manifest_path.exists():
        return {'dataset': 'scraped', 'videos': {}}
    with manifest_path.open('rb') as handle:
        return _validate_manifest(tomllib.load(handle))


def _toml_key(key: str) -> str:
    """Quote a TOML key when it cannot use the bare-key form."""
    return key if _BARE_KEY.fullmatch(key) else json.dumps(key, ensure_ascii=False)


def _toml_value(value: object, *, context: str) -> str:
    """Serialise one supported TOML scalar."""
    _manifest_scalar(value, context=context)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    return repr(value)


def _serialise_manifest(manifest: Mapping[str, object]) -> str:
    """Serialise a validated scraper manifest without adding a TOML dependency."""
    validated = _validate_manifest(dict(manifest))
    videos = validated['videos']
    assert isinstance(videos, dict)

    lines = [
        f'{_toml_key(str(key))} = {_toml_value(value, context=f"sources.toml {key!r}")}'
        for key, value in validated.items()
        if key != 'videos'
    ]
    lines.append('')
    lines.append('[videos]')
    for basename, entry in videos.items():
        assert isinstance(entry, dict)
        lines.append('')
        lines.append(f'[videos.{json.dumps(str(basename), ensure_ascii=False)}]')
        lines.extend(
            f'{_toml_key(str(key))} = {_toml_value(value, context=f"sources.toml {basename}.{key}")}'
            for key, value in entry.items()
        )
    return '\n'.join(lines) + '\n'


def _write_manifest(manifest_path: Path, manifest: Mapping[str, object]) -> None:
    """Atomically write a validated source manifest."""
    text = _serialise_manifest(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=manifest_path.parent,
            prefix=f'.{manifest_path.name}.',
            suffix='.tmp',
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
        assert temporary_path is not None
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _source_entry(
    task: tuple[str, str, str, Path],
    commentary_eligible: bool,
    existing_entry: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build an entry while retaining unknown scalar fields from an old entry."""
    url, video_id, title, _output_dir = task
    entry = dict(existing_entry) if existing_entry is not None else {}
    entry.update(
        video_id=video_id,
        title=title,
        url=url,
        commentary_eligible=commentary_eligible,
    )
    return entry


def _failure_outcome(
    task: tuple[str, str, str, Path],
    *,
    filename: str | None = None,
    entry: dict[str, object] | None = None,
) -> DownloadOutcome:
    """Create a normal per-video failure result."""
    _url, video_id, _title, _output_dir = task
    return DownloadOutcome(video_id, filename, entry, True)


def _download_one(
    task: tuple[str, str, str, Path],
    *,
    allow_missing_audio: bool,
    existing_videos: Mapping[str, object],
    direct_existing_entry: Mapping[str, object] | None = None,
) -> DownloadOutcome:
    """Download and verify one candidate without catching programming errors."""
    url, video_id, title, output_dir = task
    existing = _completed_outputs(output_dir, video_id)
    if len(existing) > 1:
        raise RuntimeError(
            f'video {video_id} has multiple completed outputs: '
            f'{", ".join(path.name for path in existing)}'
        )

    if existing:
        output_path = existing[0]
        raw_entry = (
            direct_existing_entry
            if direct_existing_entry is not None
            else existing_videos.get(output_path.name)
        )
        if raw_entry is not None and not isinstance(raw_entry, dict):
            raise TypeError(f"sources.toml entry for {output_path.name!r} must be a table")
        existing_entry = raw_entry if isinstance(raw_entry, dict) else None
        recorded_eligibility = (
            existing_entry.get('commentary_eligible')
            if existing_entry is not None
            else None
        )

        if recorded_eligibility is False:
            print(f'  Skipping video {video_id} (already marked commentary-ineligible)')
            return DownloadOutcome(
                video_id,
                output_path.name,
                _source_entry(task, False, existing_entry),
                False,
            )
        if allow_missing_audio:
            eligibility = recorded_eligibility is True
            print(f'  Skipping video {video_id} (already exists: {output_path.name})')
            return DownloadOutcome(
                video_id,
                output_path.name,
                _source_entry(task, eligibility, existing_entry),
                False,
            )

        print(f'  Checking existing video {video_id}: {output_path.name}')
        return _verify_existing(task, output_path, existing_entry)

    output_template = str(output_dir / f'{video_id}.%(ext)s')
    try:
        result = subprocess.run(
            [
                'yt-dlp',
                '--format', _YTDLP_FORMAT,
                '--output', output_template,
                '--merge-output-format', 'mp4',
                '--no-playlist',
                '--retries', '3',
                '--sleep-interval', '5',
                '--max-sleep-interval', '15',
                '--sleep-requests', '10',
                '--limit-rate', '2M',
                '--concurrent-fragments', '1',
                url,
            ],
            capture_output=True,
            text=True,
            timeout=_YTDLP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT video {video_id}: download exceeded 30 minutes')
        return _failure_outcome(task)
    if result.returncode != 0:
        print(f'  ERROR video {video_id}: {_bounded_stderr(result.stderr)}')
        return _failure_outcome(task)

    downloaded = _completed_outputs(output_dir, video_id)
    if len(downloaded) > 1:
        raise RuntimeError(
            f'video {video_id} has multiple completed outputs: '
            f'{", ".join(path.name for path in downloaded)}'
        )
    if not downloaded:
        print(f'  ERROR video {video_id}: yt-dlp produced no matching output')
        return _failure_outcome(task)

    output_path = downloaded[0]
    if allow_missing_audio:
        print(f'  Downloaded video {video_id}: {output_path.name} (audio not checked)')
        return DownloadOutcome(
            video_id,
            output_path.name,
            _source_entry(task, False, None),
            False,
        )

    try:
        has_audio = _probe_audio(output_path)
    except _UnreadableMedia as exc:
        print(f'  ERROR video {video_id}: ffprobe could not read media: {exc}')
        output_path.unlink()
        return _failure_outcome(task)
    except _AudioProbeTimeout:
        print(f'  TIMEOUT video {video_id}: ffprobe exceeded 60 seconds')
        output_path.unlink()
        return _failure_outcome(task)

    if not has_audio:
        print(f'  ERROR video {video_id}: no audio stream')
        output_path.unlink()
        return _failure_outcome(task)

    print(f'  Downloaded video {video_id}: {output_path.name}')
    return DownloadOutcome(
        video_id,
        output_path.name,
        _source_entry(task, True, None),
        False,
    )


def _verify_existing(
    task: tuple[str, str, str, Path],
    output_path: Path,
    existing_entry: Mapping[str, object] | None,
) -> DownloadOutcome:
    """Verify an existing file, retaining it when the audio gate fails."""
    _url, video_id, _title, _output_dir = task
    try:
        has_audio = _probe_audio(output_path)
    except _UnreadableMedia as exc:
        print(f'  ERROR video {video_id}: ffprobe could not read media: {exc}')
        return _failure_outcome(
            task,
            filename=output_path.name,
            entry=_source_entry(task, False, existing_entry),
        )
    except _AudioProbeTimeout:
        print(f'  TIMEOUT video {video_id}: ffprobe exceeded 60 seconds')
        return _failure_outcome(
            task,
            filename=output_path.name,
            entry=_source_entry(task, False, existing_entry),
        )

    if not has_audio:
        print(f'  ERROR video {video_id}: no audio stream')
        return _failure_outcome(
            task,
            filename=output_path.name,
            entry=_source_entry(task, False, existing_entry),
        )

    print(f'  Skipping video {video_id} (already exists: {output_path.name})')
    return DownloadOutcome(
        video_id,
        output_path.name,
        _source_entry(task, True, existing_entry),
        False,
    )


def download_video(
    url: str,
    video_id: int | str,
    title: str,
    output_dir: Path = VIDEOS_DIR,
    *,
    allow_missing_audio: bool = False,
    existing_entry: Mapping[str, object] | None = None,
) -> str | None:
    """Download one scraper video and return its actual output basename.

    :param url: YouTube video URL.
    :param video_id: YouTube id used as the output basename stem.
    :param title: Stage-1 title recorded in the source manifest.
    :param output_dir: Directory containing downloaded videos.
    :param allow_missing_audio: Accept media without invoking ffprobe.
    :param existing_entry: Existing manifest entry for a pre-existing file.
    :return: Actual output filename on success, or ``None`` on a seed failure.
    """
    task = (url, str(video_id), title, output_dir)
    outcome = _download_one(
        task,
        allow_missing_audio=allow_missing_audio,
        existing_videos={},
        direct_existing_entry=existing_entry,
    )
    return outcome.filename if not outcome.failed else None


def _tasks_from_rows(rows: list[dict], output_dir: Path) -> list[tuple[str, str, str, Path]]:
    """Filter kept candidates and reject duplicate download seeds."""
    tasks: list[tuple[str, str, str, Path]] = []
    seen_ids: set[str] = set()
    for row in rows:
        if row['keep'] != 'True':
            continue
        video_id = str(row['video_id'])
        if video_id in seen_ids:
            raise ValueError(f'duplicate kept video_id: {video_id}')
        seen_ids.add(video_id)
        tasks.append((row['url'], video_id, row['title'], output_dir))
    return tasks


def download_all_videos(
    candidates_path: Path = CANDIDATES_CSV,
    output_dir: Path = VIDEOS_DIR,
    max_workers: int = _DEFAULT_WORKERS,
    *,
    allow_missing_audio: bool = False,
) -> list[DownloadOutcome]:
    """Download selected scraper candidates and update their source manifest.

    :param candidates_path: Candidate CSV, filtered with exact ``keep == 'True'``.
    :param output_dir: Destination for downloaded videos.
    :param max_workers: Number of parallel download threads.
    :param allow_missing_audio: Skip ffprobe and mark new files ineligible.
    :return: One outcome per selected seed when workers finish normally.
    """
    manifest_path = output_dir / SOURCES_MANIFEST_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path)
    rows = config.read_candidates(candidates_path)
    tasks = _tasks_from_rows(rows, output_dir)

    if not tasks:
        print('Nothing to download.')
        _write_manifest(manifest_path, manifest)
        return []

    _check_ytdlp()
    if not allow_missing_audio:
        _check_ffprobe()

    videos = manifest['videos']
    assert isinstance(videos, dict)
    print(f'Downloading {len(tasks)} videos...')
    print(f'Using {max_workers} parallel workers')

    outcomes: list[DownloadOutcome] = []
    first_unexpected: Exception | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[Future[DownloadOutcome]] = [
            executor.submit(
                _download_one,
                task,
                allow_missing_audio=allow_missing_audio,
                existing_videos=videos,
            )
            for task in tasks
        ]
        for future in futures:
            try:
                outcomes.append(future.result())
            except Exception as exc:
                if first_unexpected is None:
                    first_unexpected = exc

    updated_videos = dict(videos)
    for outcome in outcomes:
        if outcome.filename is not None and outcome.entry is not None:
            updated_videos[outcome.filename] = outcome.entry
    updated_manifest = dict(manifest)
    updated_manifest['videos'] = updated_videos
    _write_manifest(manifest_path, updated_manifest)

    if first_unexpected is not None:
        raise first_unexpected

    failures = sum(outcome.failed for outcome in outcomes)
    print(f'Finished: {len(outcomes) - failures}/{len(tasks)} videos accepted; {failures} failed.')
    return outcomes


def main() -> int:
    """Run the scraper downloader CLI and return its mass-failure status."""
    parser = argparse.ArgumentParser(description='Download kept scraper videos.')
    parser.add_argument(
        '--candidates-csv',
        dest='candidates_path',
        type=Path,
        default=config.CANDIDATES_CSV,
    )
    parser.add_argument('--output-dir', type=Path, default=config.VIDEOS_DIR)
    parser.add_argument('--workers', type=int, default=_DEFAULT_WORKERS)
    parser.add_argument(
        '--allow-missing-audio',
        action='store_true',
        help='Accept videos without checking for an audio stream; newly accepted files are\n'
             'marked commentary-ineligible.',
    )
    args = parser.parse_args()

    outcomes = download_all_videos(
        candidates_path=args.candidates_path,
        output_dir=args.output_dir,
        max_workers=args.workers,
        allow_missing_audio=args.allow_missing_audio,
    )
    if not outcomes:
        return 0

    failures = sum(outcome.failed for outcome in outcomes)
    if failures / len(outcomes) >= 0.5:
        print(f'Failure summary: {failures}/{len(outcomes)} selected videos failed.')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
