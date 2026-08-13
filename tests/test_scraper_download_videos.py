"""Focused tests for scraper-owned video downloads."""

from __future__ import annotations

import csv
import subprocess
import tomllib
from pathlib import Path

import pytest

from src.scraper import download_scraped_videos as downloader


def _write_candidates(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['video_id', 'url', 'title', 'keep'])
        writer.writeheader()
        writer.writerows(rows)


def _candidate(video_id: str, *, keep: str = 'True') -> dict[str, str]:
    return {
        'video_id': video_id,
        'url': f'https://example.test/{video_id}',
        'title': f'Title {video_id}',
        'keep': keep,
    }


def _run_main_with_fake_outcomes(
    tmp_path: Path,
    monkeypatch,
    rows: list[dict[str, str]],
    failed_ids: set[str],
) -> tuple[int, list[str], Path]:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, rows)
    output_dir = tmp_path / 'videos'
    manifest_path = output_dir / 'sources.toml'
    called_ids: list[str] = []

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_download(
        task: tuple[str, str, str, Path],
        *,
        allow_missing_audio: bool,
        video_only: bool,
        existing_videos: dict[str, object],
        accept_silent_video: bool = False,
    ) -> downloader.DownloadOutcome:
        del allow_missing_audio, video_only, existing_videos, accept_silent_video
        url, video_id, title, _output_dir = task
        called_ids.append(video_id)
        if video_id in failed_ids:
            return downloader.DownloadOutcome(video_id, None, None, True)
        return downloader.DownloadOutcome(
            video_id,
            f'{video_id}.mp4',
            {
                'video_id': video_id,
                'title': title,
                'url': url,
                'commentary_eligible': False,
            },
            False,
        )

    monkeypatch.setattr(downloader, '_download_one', fake_download)
    monkeypatch.setattr(
        'sys.argv',
        [
            'download_scraped_videos',
            '--candidates-csv', str(candidates_path),
            '--output-dir', str(output_dir),
            '--allow-missing-audio',
        ],
    )
    return downloader.main(), called_ids, manifest_path


def test_download_argv_requests_h264_video_and_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / 'videos'
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('abc123')])
    calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')
    monkeypatch.setattr(downloader.config, 'YTDLP_BIN', 'configured-yt-dlp')
    monkeypatch.setattr(downloader.config, 'YTDLP_RETRIES', 7)
    monkeypatch.setattr(downloader.config, 'SLEEP_INTERVAL_S', 6)
    monkeypatch.setattr(downloader.config, 'MAX_SLEEP_INTERVAL_S', 16)
    monkeypatch.setattr(downloader.config, 'SLEEP_REQUESTS_S', 11)
    monkeypatch.setattr(downloader.config, 'LIMIT_RATE', '3M')
    monkeypatch.setattr(downloader.config, 'CONCURRENT_FRAGMENTS', 2)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        (output_dir / 'abc123.mp4').write_bytes(b'video')
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)
    monkeypatch.setattr(downloader, '_probe_audio', lambda path: True)

    outcomes = downloader.download_all_videos(
        candidates_path=candidates_path,
        output_dir=output_dir,
        max_workers=1,
    )

    assert outcomes[0].filename == 'abc123.mp4'
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:3] == [
        'configured-yt-dlp',
        '--format',
        downloader._YTDLP_FORMAT,
    ]
    assert argv[argv.index('--output') + 1] == str(output_dir / 'abc123.%(ext)s')
    assert argv[argv.index('--merge-output-format') + 1] == 'mp4'
    assert argv[argv.index('--no-playlist')] == '--no-playlist'
    assert argv[argv.index('--retries') + 1] == '7'
    assert argv[argv.index('--sleep-interval') + 1] == '6'
    assert argv[argv.index('--max-sleep-interval') + 1] == '16'
    assert argv[argv.index('--sleep-requests') + 1] == '11'
    assert argv[argv.index('--limit-rate') + 1] == '3M'
    assert argv[argv.index('--concurrent-fragments') + 1] == '2'
    assert argv[-1] == 'https://example.test/abc123'
    assert kwargs['timeout'] == 1800


def test_completed_outputs_ignores_ytdlp_fragment_named_like_video(tmp_path: Path) -> None:
    (tmp_path / 'abc.f137.mp4').write_bytes(b'partial')
    (tmp_path / 'abc Match.f137.mp4').write_bytes(b'partial')

    assert downloader._completed_outputs(tmp_path, 'abc') == []


def test_video_only_mode_uses_h264_selector_without_ffprobe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('silent')])
    output_dir = tmp_path / 'videos'
    calls: list[list[str]] = []

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(argv)
        (output_dir / 'silent.mp4').write_bytes(b'video')
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)
    outcomes = downloader.download_all_videos(
        candidates_path,
        output_dir,
        max_workers=1,
        video_only=True,
    )

    assert calls[0][calls[0].index('--format') + 1] == downloader._YTDLP_VIDEO_ONLY_FORMAT
    assert outcomes[0].entry is not None
    assert outcomes[0].entry['commentary_eligible'] is False


def test_legacy_spaced_file_is_completed_output(tmp_path: Path) -> None:
    legacy_path = tmp_path / '12 Match Name.mp4'
    legacy_path.write_bytes(b'video')

    assert downloader._completed_outputs(tmp_path, '12') == [legacy_path]


def test_video_only_resume_skips_legacy_file_without_creating_exact_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('12')])
    output_dir = tmp_path / 'videos'
    output_dir.mkdir()
    legacy_path = output_dir / '12 Match Name.mp4'
    legacy_path.write_bytes(b'video')

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError(f'yt-dlp must not run: {args}, {kwargs}')

    monkeypatch.setattr(downloader.subprocess, 'run', fail_download)
    outcomes = downloader.download_all_videos(
        candidates_path,
        output_dir,
        max_workers=1,
        video_only=True,
    )

    assert outcomes[0].filename == legacy_path.name
    assert not (output_dir / '12.mp4').exists()
    with (output_dir / 'sources.toml').open('rb') as handle:
        entry = tomllib.load(handle)['videos'][legacy_path.name]
    assert entry['commentary_eligible'] is False


def test_exact_and_legacy_outputs_are_ambiguous(tmp_path: Path) -> None:
    (tmp_path / '12.mp4').write_bytes(b'video')
    (tmp_path / '12 Match Name.mp4').write_bytes(b'video')
    task = ('https://example.test/12', '12', 'Match Name', tmp_path)

    with pytest.raises(RuntimeError, match='multiple completed outputs'):
        downloader._download_one(
            task,
            allow_missing_audio=False,
            video_only=True,
            existing_videos={},
        )


def test_audio_verified_download_writes_truthful_manifest(tmp_path: Path, monkeypatch) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('abc')])
    output_dir = tmp_path / 'videos'
    manifest_path = output_dir / 'sources.toml'
    calls: list[list[str]] = []

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(argv)
        if argv[0] == 'yt-dlp':
            (output_dir / 'abc.mp4').write_bytes(b'video')
            return subprocess.CompletedProcess(argv, 0, '', '')
        return subprocess.CompletedProcess(argv, 0, 'audio\n', '')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)

    outcomes = downloader.download_all_videos(
        candidates_path, output_dir, max_workers=1,
    )

    assert len(outcomes) == 1
    assert not outcomes[0].failed
    assert [call[0] for call in calls] == ['yt-dlp', 'ffprobe']
    with manifest_path.open('rb') as handle:
        manifest = tomllib.load(handle)
    assert manifest['videos']['abc.mp4'] == {
        'video_id': 'abc',
        'title': 'Title abc',
        'url': 'https://example.test/abc',
        'commentary_eligible': True,
    }


def test_missing_audio_removes_new_file_and_counts_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('silent')])
    output_dir = tmp_path / 'videos'

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        if argv[0] == 'yt-dlp':
            (output_dir / 'silent.mp4').write_bytes(b'video')
            return subprocess.CompletedProcess(argv, 0, '', '')
        return subprocess.CompletedProcess(argv, 0, '\n', '')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)
    outcomes = downloader.download_all_videos(
        candidates_path, output_dir, max_workers=1,
    )

    assert outcomes[0].failed
    assert sum(outcome.failed for outcome in outcomes) == 1
    assert not (output_dir / 'silent.mp4').exists()
    assert 'no audio stream' in capsys.readouterr().out


def test_ffprobe_failure_is_distinct_and_removes_new_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('broken')])
    output_dir = tmp_path / 'videos'

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        if argv[0] == 'yt-dlp':
            (output_dir / 'broken.mp4').write_bytes(b'broken')
            return subprocess.CompletedProcess(argv, 0, '', '')
        return subprocess.CompletedProcess(argv, 1, '', 'moov atom not found')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)
    outcomes = downloader.download_all_videos(
        candidates_path, output_dir, max_workers=1,
    )

    assert outcomes[0].failed
    assert not (output_dir / 'broken.mp4').exists()
    output = capsys.readouterr().out
    assert 'ffprobe could not read media' in output
    assert 'moov atom not found' in output
    assert 'no audio stream' not in output


def test_override_accepts_new_silent_file_without_ffprobe(tmp_path: Path, monkeypatch) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('silent')])
    output_dir = tmp_path / 'videos'
    manifest_path = output_dir / 'sources.toml'
    calls: list[list[str]] = []

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(argv)
        (output_dir / 'silent.mp4').write_bytes(b'silent')
        return subprocess.CompletedProcess(argv, 0, '', '')

    def fail_probe(path: Path) -> bool:
        raise AssertionError(f'ffprobe must not run for {path}')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)
    monkeypatch.setattr(downloader, '_probe_audio', fail_probe)
    outcomes = downloader.download_all_videos(
        candidates_path,
        output_dir,
        max_workers=1,
        allow_missing_audio=True,
    )

    assert not outcomes[0].failed
    assert [call[0] for call in calls] == ['yt-dlp']
    with manifest_path.open('rb') as handle:
        manifest = tomllib.load(handle)
    assert manifest['videos']['silent.mp4']['commentary_eligible'] is False


def test_override_preserves_existing_eligibility_and_adopts_unmanifested_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('verified'), _candidate('unknown')])
    output_dir = tmp_path / 'videos'
    output_dir.mkdir()
    (output_dir / 'verified.mp4').write_bytes(b'verified')
    (output_dir / 'unknown.mp4').write_bytes(b'unknown')
    manifest_path = output_dir / 'sources.toml'
    manifest_path.write_text(
        'dataset = "scraped"\n\n'
        '[videos."verified.mp4"]\n'
        'video_id = "verified"\n'
        'title = "Old title"\n'
        'url = "https://old.example/verified"\n'
        'commentary_eligible = true\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fail_probe(path: Path) -> bool:
        raise AssertionError(f'ffprobe must not run for {path}')

    monkeypatch.setattr(downloader, '_probe_audio', fail_probe)
    outcomes = downloader.download_all_videos(
        candidates_path,
        output_dir,
        max_workers=1,
        allow_missing_audio=True,
    )

    assert all(not outcome.failed for outcome in outcomes)
    with manifest_path.open('rb') as handle:
        videos = tomllib.load(handle)['videos']
    assert videos['verified.mp4']['commentary_eligible'] is True
    assert videos['unknown.mp4']['commentary_eligible'] is False


def test_manifest_round_trip_preserves_rich_strings_and_unknown_scalars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    rich_url = 'https://example.test/"café"\\path\tline\nnext'
    rich_title = 'Title "révélé"\\path\tline\nnext'
    _write_candidates(
        candidates_path,
        [{
            'video_id': 'new',
            'url': rich_url,
            'title': rich_title,
            'keep': 'True',
        }],
    )
    output_dir = tmp_path / 'videos'
    manifest_path = output_dir / 'sources.toml'
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        'dataset = "scraped"\n'
        'future_drill_footage = true\n\n'
        '[videos."old.mp4"]\n'
        'video_id = "old"\n'
        'title = "Old title"\n'
        'url = "https://old.example/"\n'
        'commentary_eligible = false\n'
        'future_flag = true\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        (output_dir / 'new.mp4').write_bytes(b'video')
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)
    downloader.download_all_videos(
        candidates_path,
        output_dir,
        max_workers=1,
        allow_missing_audio=True,
    )

    with manifest_path.open('rb') as handle:
        manifest = tomllib.load(handle)
    assert manifest['future_drill_footage'] is True
    assert manifest['videos']['old.mp4']['future_flag'] is True
    assert manifest['videos']['new.mp4'] == {
        'video_id': 'new',
        'title': rich_title,
        'url': rich_url,
        'commentary_eligible': False,
    }


@pytest.mark.parametrize(
    ('manifest_text', 'error_type'),
    [
        ('[videos."clip.mp4"\n', tomllib.TOMLDecodeError),
        ('dataset = "scraped"\nvideos = []\n', TypeError),
        ('dataset = 7\n[videos]\n', TypeError),
        ('dataset = ""\n[videos]\n', ValueError),
        ('dataset = "scraped"\nfuture = [1, 2]\n[videos]\n', TypeError),
    ],
)
def test_existing_manifest_validation_fails_loudly(
    tmp_path: Path,
    manifest_text: str,
    error_type: type[Exception],
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [])
    output_dir = tmp_path / 'videos'
    manifest_path = output_dir / 'sources.toml'
    manifest_path.parent.mkdir()
    manifest_path.write_text(manifest_text, encoding='utf-8')

    with pytest.raises(error_type):
        downloader.download_all_videos(candidates_path, output_dir)


def test_empty_selected_input_writes_empty_manifest_and_exits_zero(tmp_path: Path, monkeypatch) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('excluded', keep='False')])
    output_dir = tmp_path / 'videos'
    manifest_path = output_dir / 'sources.toml'

    monkeypatch.setattr(
        'sys.argv',
        [
            'download_scraped_videos',
            '--candidates-csv', str(candidates_path),
            '--output-dir', str(output_dir),
        ],
    )

    assert downloader.main() == 0
    with manifest_path.open('rb') as handle:
        manifest = tomllib.load(handle)
    assert manifest == {'dataset': 'scraped', 'videos': {}}


def test_cli_writes_explicit_dataset_label(tmp_path: Path, monkeypatch) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [])
    output_dir = tmp_path / 'videos'

    monkeypatch.setattr(
        'sys.argv',
        [
            'download_scraped_videos',
            '--candidates-csv', str(candidates_path),
            '--output-dir', str(output_dir),
            '--dataset', 'shuttleset',
        ],
    )

    assert downloader.main() == 0
    with (output_dir / 'sources.toml').open('rb') as handle:
        manifest = tomllib.load(handle)
    assert manifest == {'dataset': 'shuttleset', 'videos': {}}


def test_existing_manifest_dataset_must_match_requested_label(tmp_path: Path) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [])
    output_dir = tmp_path / 'videos'
    output_dir.mkdir()
    manifest_path = output_dir / 'sources.toml'
    manifest_text = 'dataset = "shuttleset"\n\n[videos]\n'
    manifest_path.write_text(manifest_text, encoding='utf-8')

    with pytest.raises(ValueError, match="expected 'scraped'"):
        downloader.download_all_videos(candidates_path, output_dir)

    assert manifest_path.read_text(encoding='utf-8') == manifest_text


def test_probe_audio_pins_full_argv_timeout_and_exact_path(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / 'abc.mkv'
    video_path.write_bytes(b'video')
    observed: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed['argv'] = argv
        observed['kwargs'] = kwargs
        return subprocess.CompletedProcess(argv, 0, 'audio\n', '')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)

    assert downloader._probe_audio(video_path)
    assert observed['argv'] == [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path),
    ]
    assert observed['kwargs'] == {
        'capture_output': True,
        'text': True,
        'timeout': 60,
    }


def test_multiple_completed_video_extensions_fail_loudly(tmp_path: Path) -> None:
    (tmp_path / 'abc.mp4').write_bytes(b'video')
    (tmp_path / 'abc.mkv').write_bytes(b'video')
    task = ('https://example.test/abc', 'abc', 'Title abc', tmp_path)

    with pytest.raises(RuntimeError, match='multiple completed outputs'):
        downloader._download_one(
            task,
            allow_missing_audio=True,
            video_only=False,
            existing_videos={},
        )


def test_failed_preexisting_file_is_retained_and_recorded_false(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('existing')])
    output_dir = tmp_path / 'videos'
    output_dir.mkdir()
    video_path = output_dir / 'existing.mp4'
    video_path.write_bytes(b'silent')
    manifest_path = output_dir / 'sources.toml'

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')
    monkeypatch.setattr(downloader, '_probe_audio', lambda path: False)
    outcomes = downloader.download_all_videos(
        candidates_path, output_dir, max_workers=1,
    )

    assert outcomes[0].failed
    assert video_path.exists()
    with manifest_path.open('rb') as handle:
        entry = tomllib.load(handle)['videos']['existing.mp4']
    assert entry['commentary_eligible'] is False


def test_existing_false_entry_is_not_reprobed_or_counted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('silent')])
    output_dir = tmp_path / 'videos'
    output_dir.mkdir()
    (output_dir / 'silent.mp4').write_bytes(b'silent')
    manifest_path = output_dir / 'sources.toml'
    manifest_path.write_text(
        'dataset = "scraped"\n\n'
        '[videos."silent.mp4"]\n'
        'video_id = "silent"\n'
        'title = "Title silent"\n'
        'url = "https://example.test/silent"\n'
        'commentary_eligible = false\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fail_probe(path: Path) -> bool:
        raise AssertionError(f'ffprobe must not run for {path}')

    monkeypatch.setattr(downloader, '_probe_audio', fail_probe)
    outcomes = downloader.download_all_videos(
        candidates_path, output_dir, max_workers=1,
    )

    assert not outcomes[0].failed


def test_accept_silent_video_reprobes_false_entry_until_readable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('existing')])
    output_dir = tmp_path / 'videos'
    output_dir.mkdir()
    video_path = output_dir / 'existing.mp4'
    video_path.write_bytes(b'broken')
    manifest_path = output_dir / 'sources.toml'
    manifest_path.write_text(
        'dataset = "scraped"\n\n'
        '[videos."existing.mp4"]\n'
        'video_id = "existing"\n'
        'title = "Title existing"\n'
        'url = "https://example.test/existing"\n'
        'commentary_eligible = false\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def probe(path: Path) -> bool:
        if path.read_bytes() == b'broken':
            raise downloader._UnreadableMedia('fixture media failure')
        return False

    monkeypatch.setattr(downloader, '_probe_audio', probe)
    first = downloader.download_all_videos(
        candidates_path,
        output_dir,
        max_workers=1,
        accept_silent_video=True,
    )
    video_path.write_bytes(b'readable silent video')
    second = downloader.download_all_videos(
        candidates_path,
        output_dir,
        max_workers=1,
        accept_silent_video=True,
    )

    assert first[0].failed
    assert not second[0].failed


def test_unexpected_worker_exception_propagates_after_sibling_manifest_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('boom'), _candidate('good')])
    output_dir = tmp_path / 'videos'
    manifest_path = output_dir / 'sources.toml'
    called_ids: list[str] = []

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_download(
        task: tuple[str, str, str, Path],
        *,
        allow_missing_audio: bool,
        video_only: bool,
        existing_videos: dict[str, object],
        accept_silent_video: bool = False,
    ) -> downloader.DownloadOutcome:
        del allow_missing_audio, video_only, existing_videos, accept_silent_video
        url, video_id, title, _output_dir = task
        called_ids.append(video_id)
        if video_id == 'boom':
            raise RuntimeError('worker defect')
        return downloader.DownloadOutcome(
            video_id,
            'good.mp4',
            {
                'video_id': video_id,
                'title': title,
                'url': url,
                'commentary_eligible': True,
            },
            False,
        )

    monkeypatch.setattr(downloader, '_download_one', fake_download)
    with pytest.raises(RuntimeError, match='worker defect'):
        downloader.download_all_videos(
            candidates_path,
            output_dir,
            max_workers=2,
            allow_missing_audio=True,
        )

    assert set(called_ids) == {'boom', 'good'}
    with manifest_path.open('rb') as handle:
        manifest = tomllib.load(handle)
    assert 'good.mp4' in manifest['videos']


def test_duplicate_kept_video_ids_fail_before_worker_checks(tmp_path: Path, monkeypatch) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('duplicate'), _candidate('duplicate')])
    output_dir = tmp_path / 'videos'

    def fail_tool_check() -> None:
        raise AssertionError('duplicate seeds must fail before tool checks')

    monkeypatch.setattr(downloader, '_check_ytdlp', fail_tool_check)
    with pytest.raises(ValueError, match='duplicate kept video_id: duplicate'):
        downloader.download_all_videos(candidates_path, output_dir)


def test_exactly_half_failed_seeds_exit_two(tmp_path: Path, monkeypatch) -> None:
    rows = [_candidate(f'v{index}') for index in range(4)]
    exit_code, called_ids, _manifest_path = _run_main_with_fake_outcomes(
        tmp_path, monkeypatch, rows, {'v0', 'v1'},
    )

    assert exit_code == 2
    assert set(called_ids) == {'v0', 'v1', 'v2', 'v3'}


def test_all_failed_seeds_exit_two(tmp_path: Path, monkeypatch) -> None:
    rows = [_candidate('v0'), _candidate('v1')]

    exit_code, _called_ids, _manifest_path = _run_main_with_fake_outcomes(
        tmp_path, monkeypatch, rows, {'v0', 'v1'},
    )

    assert exit_code == 2


def test_fewer_than_half_failed_seeds_exit_zero(tmp_path: Path, monkeypatch) -> None:
    rows = [_candidate(f'v{index}') for index in range(3)]

    exit_code, _called_ids, _manifest_path = _run_main_with_fake_outcomes(
        tmp_path, monkeypatch, rows, {'v0'},
    )

    assert exit_code == 0


def test_main_uses_the_configured_failure_fraction(tmp_path: Path, monkeypatch) -> None:
    rows = [_candidate(f'v{index}') for index in range(4)]
    monkeypatch.setattr(downloader.config, 'DOWNLOAD_FAIL_FRACTION_BLOCK', 0.75)

    exit_code, _called_ids, _manifest_path = _run_main_with_fake_outcomes(
        tmp_path, monkeypatch, rows, {'v0', 'v1'},
    )

    assert exit_code == 0


def test_mass_failure_denominator_excludes_unkept_rows(tmp_path: Path, monkeypatch) -> None:
    rows = [_candidate('excluded', keep='False'), _candidate('failed'), _candidate('good')]

    exit_code, called_ids, _manifest_path = _run_main_with_fake_outcomes(
        tmp_path, monkeypatch, rows, {'failed'},
    )

    assert exit_code == 2
    assert set(called_ids) == {'failed', 'good'}


def test_successful_manifest_entry_is_written_before_mass_failure_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = [_candidate('failed'), _candidate('good')]

    exit_code, _called_ids, manifest_path = _run_main_with_fake_outcomes(
        tmp_path, monkeypatch, rows, {'failed'},
    )

    assert exit_code == 2
    with manifest_path.open('rb') as handle:
        videos = tomllib.load(handle)['videos']
    assert 'failed.mp4' not in videos
    assert videos['good.mp4']['video_id'] == 'good'


def test_produced_mkv_uses_actual_basename_and_ignores_stale_manifest_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('abc')])
    output_dir = tmp_path / 'videos'
    manifest_path = output_dir / 'sources.toml'
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        'dataset = "scraped"\n\n'
        '[videos."abc.mp4"]\n'
        'video_id = "abc"\n'
        'title = "Old"\n'
        'url = "https://old.example/abc"\n'
        'commentary_eligible = true\n',
        encoding='utf-8',
    )

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        if argv[0] == 'yt-dlp':
            (output_dir / 'abc.mkv').write_bytes(b'video')
            return subprocess.CompletedProcess(argv, 0, '', '')
        return subprocess.CompletedProcess(argv, 0, 'audio\n', '')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)
    outcomes = downloader.download_all_videos(
        candidates_path, output_dir, max_workers=1,
    )

    assert outcomes[0].filename == 'abc.mkv'
    assert downloader._completed_outputs(output_dir, 'abc') == [output_dir / 'abc.mkv']
    with manifest_path.open('rb') as handle:
        videos = tomllib.load(handle)['videos']
    assert 'abc.mkv' in videos
    assert 'abc.mp4' in videos


def test_explicit_selected_ids_override_keep_without_mutating_candidates(tmp_path: Path) -> None:
    rows = [
        {'video_id': 'first', 'url': 'https://x/first', 'title': 'First', 'keep': 'False'},
        {'video_id': 'second', 'url': 'https://x/second', 'title': 'Second', 'keep': ''},
        {'video_id': 'third', 'url': 'https://x/third', 'title': 'Third', 'keep': 'True'},
    ]

    tasks = downloader._tasks_from_rows(rows, tmp_path, {'second', 'first'})

    assert [task[1] for task in tasks] == ['first', 'second']
    assert [row['keep'] for row in rows] == ['False', '', 'True']


def test_explicit_selected_ids_reject_unknown_or_duplicate_values(tmp_path: Path) -> None:
    rows = [{'video_id': 'known', 'url': 'https://x/known', 'title': 'Known', 'keep': ''}]

    with pytest.raises(ValueError, match='absent from candidates'):
        downloader._tasks_from_rows(rows, tmp_path, {'missing'})
    with pytest.raises(ValueError, match='duplicate values'):
        downloader._tasks_from_rows(rows, tmp_path, ['known', 'known'])
    with pytest.raises(ValueError, match='not one string'):
        downloader._tasks_from_rows(rows, tmp_path, 'known')


def test_explicit_selection_can_keep_a_verified_silent_visual_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidates_path = tmp_path / 'candidates.csv'
    _write_candidates(candidates_path, [_candidate('silent')])
    output_dir = tmp_path / 'videos'

    monkeypatch.setattr(downloader.shutil, 'which', lambda name: f'/bin/{name}')

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        (output_dir / 'silent.mp4').write_bytes(b'silent')
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(downloader.subprocess, 'run', fake_run)
    monkeypatch.setattr(downloader, '_probe_audio', lambda _path: False)

    outcomes = downloader.download_all_videos(
        candidates_path,
        output_dir,
        max_workers=1,
        selected_video_ids=['silent'],
        accept_silent_video=True,
    )

    assert not outcomes[0].failed
    assert (output_dir / 'silent.mp4').is_file()
    assert outcomes[0].entry is not None
    assert outcomes[0].entry['commentary_eligible'] is False
