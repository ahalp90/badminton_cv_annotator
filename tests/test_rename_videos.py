"""Tests for the retained ShuttleSet video rename tool."""

import sys
from pathlib import Path

import pytest

from scripts import rename_videos


def test_dry_run_uses_current_numeric_video_name(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / '1_1080p_25fps.mp4').write_bytes(b'video')
    match_csv = tmp_path / 'match.csv'
    match_csv.write_text('id,video,url\n1,Match Name,https://example.test/1\n')
    flaw_csv = tmp_path / 'flaws.csv'
    flaw_csv.write_text('match,stroke_type,reason\n')

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'rename_videos.py',
            '--video-dir', str(video_dir),
            '--match-csv', str(match_csv),
            '--flaw-csv', str(flaw_csv),
            '--dry-run',
        ],
    )

    rename_videos.main()

    assert '-> 1.mp4' in capsys.readouterr().out
    assert (video_dir / '1_1080p_25fps.mp4').exists()


def test_existing_numeric_target_fails_without_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    source = video_dir / '1_1080p_25fps.mp4'
    target = video_dir / '1.mp4'
    source.write_bytes(b'source')
    target.write_bytes(b'target')
    match_csv = tmp_path / 'match.csv'
    match_csv.write_text('id,video,url\n1,Match Name,https://example.test/1\n')
    flaw_csv = tmp_path / 'flaws.csv'
    flaw_csv.write_text('match,stroke_type,reason\n')
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'rename_videos.py',
            '--video-dir', str(video_dir),
            '--match-csv', str(match_csv),
            '--flaw-csv', str(flaw_csv),
        ],
    )

    with pytest.raises(FileExistsError, match='target already exists'):
        rename_videos.main()

    assert source.read_bytes() == b'source'
    assert target.read_bytes() == b'target'
    assert not (video_dir / 'video_metadata.csv').exists()


def test_duplicate_source_ids_fail_without_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    first = video_dir / '1_1080p_25fps.mp4'
    second = video_dir / '1_720p_30fps.mp4'
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    match_csv = tmp_path / 'match.csv'
    match_csv.write_text('id,video,url\n1,Match Name,https://example.test/1\n')
    flaw_csv = tmp_path / 'flaws.csv'
    flaw_csv.write_text('match,stroke_type,reason\n')
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'rename_videos.py',
            '--video-dir', str(video_dir),
            '--match-csv', str(match_csv),
            '--flaw-csv', str(flaw_csv),
        ],
    )

    with pytest.raises(RuntimeError, match='Multiple source files for ID 1'):
        rename_videos.main()

    assert first.read_bytes() == b'first'
    assert second.read_bytes() == b'second'
    assert not (video_dir / '1.mp4').exists()
