"""Tests for ShuttleSet raw-video discovery and resolution metadata."""

from pathlib import Path

import pandas as pd
import pytest

from src.bst_x.pipeline import video_metadata


def test_find_video_files_accepts_new_and_legacy_names(tmp_path: Path) -> None:
    exact = tmp_path / '1.mp4'
    legacy = tmp_path / '2 Match Name.mov'
    exact.write_bytes(b'video')
    legacy.write_bytes(b'video')
    (tmp_path / 'sources.toml').write_text('dataset = "scraped"\n', encoding='utf-8')

    assert video_metadata.find_video_files(tmp_path) == {1: exact, 2: legacy}


def test_find_video_files_does_not_count_manifest_as_video(tmp_path: Path) -> None:
    (tmp_path / 'sources.toml').write_text('dataset = "scraped"\n', encoding='utf-8')
    (tmp_path / '1 Match.f137.mp4').write_bytes(b'partial')

    assert video_metadata.find_video_files(tmp_path) == {}


def test_find_video_files_rejects_exact_and_legacy_double_match(tmp_path: Path) -> None:
    (tmp_path / '12.mp4').write_bytes(b'video')
    (tmp_path / '12 Match Name.mp4').write_bytes(b'video')

    with pytest.raises(RuntimeError, match='multiple raw videos.*12'):
        video_metadata.find_video_files(tmp_path)


def test_build_resolution_csv_preserves_report_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / '1.mp4').write_bytes(b'video')
    (video_dir / '2 Match Name.mp4').write_bytes(b'video')
    set_dir = tmp_path / 'set'
    set_dir.mkdir()
    pd.DataFrame({'id': [1, 2, 3]}).to_csv(set_dir / 'match.csv', index=False)
    output_path = tmp_path / 'my_raw_video_resolution.csv'

    class FakeCapture:
        def __init__(self, path: str) -> None:
            self.path = Path(path)

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == video_metadata.cv2.CAP_PROP_FRAME_WIDTH:
                return 1920 if self.path.stem == '1' else 1280
            if prop == video_metadata.cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080 if self.path.stem == '1' else 720
            raise AssertionError(f'unexpected property: {prop}')

        def release(self) -> None:
            return None

    monkeypatch.setattr(video_metadata, 'SET_INFO_DIR', set_dir)
    monkeypatch.setattr(video_metadata.cv2, 'VideoCapture', FakeCapture)

    result = video_metadata.build_resolution_csv(video_dir, output_path)

    expected = pd.DataFrame(
        {'id': [1, 2], 'width': [1920, 1280], 'height': [1080, 720]}
    )
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(pd.read_csv(output_path), expected)
    report = output_path.with_name('my_raw_video_resolution_csv_missing.txt').read_text()
    assert 'Missing video IDs: [3]' in report


def test_all_unreadable_videos_preserve_existing_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video_dir = tmp_path / 'videos'
    video_dir.mkdir()
    (video_dir / '1.mp4').write_bytes(b'unreadable')
    output_path = tmp_path / 'my_raw_video_resolution.csv'
    original_csv = 'id,width,height\n1,1920,1080\n'
    output_path.write_text(original_csv)
    report_path = output_path.with_name('my_raw_video_resolution_csv_missing.txt')
    report_path.write_text('existing report\n')

    class UnreadableCapture:
        def __init__(self, path: str) -> None:
            self.path = path

        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr(video_metadata.cv2, 'VideoCapture', UnreadableCapture)

    with pytest.raises(RuntimeError, match='Could not read resolution from any video'):
        video_metadata.build_resolution_csv(video_dir, output_path)

    assert output_path.read_text() == original_csv
    assert report_path.read_text() == 'existing report\n'
