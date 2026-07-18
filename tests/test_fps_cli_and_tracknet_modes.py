"""Regression tests for fps-aware CLIs and TrackNet mode selection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

def test_stage9_main_scales_composition_min_scene_len(monkeypatch, tmp_path: Path) -> None:
    import annotator.composition_mask as stage9

    keep_vote_path = tmp_path / 'keep_vote.npy'
    np.save(keep_vote_path, np.ones(200, dtype=bool))
    captured: list[int] = []

    def fake_detect_cuts(video_path, expected_frames, threshold, min_scene_len):
        captured.append(min_scene_len)
        return np.array([100], dtype=int)

    monkeypatch.setattr(stage9, 'detect_cuts', fake_detect_cuts)
    common = [
        '--video-id', 'video-1', '--video', str(tmp_path / 'unused.mp4'),
        '--keep-vote', str(keep_vote_path), '--out-dir', str(tmp_path / 'masks'),
    ]
    monkeypatch.setattr(sys, 'argv', ['stage9', *common, '--fps', '60'])
    stage9.main()
    monkeypatch.setattr(sys, 'argv', ['stage9', *common])
    stage9.main()

    assert captured == [30, 13]


@pytest.mark.parametrize(
    ('fps_args', 'expected_impulse'),
    [
        (['--fps-csv'], 20),
        (['--fps-csv', '--missing-id'], 10),
        (['--fps', '60'], 24),
    ],
)
def test_stage8_main_resolves_fps_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    fps_args: list[str],
    expected_impulse: int,
) -> None:
    import annotator.rally_segmentation as stage8

    shuttle_dir = tmp_path / 'shuttles'
    shuttle_dir.mkdir()
    np.save(shuttle_dir / 'video-1.npy', np.zeros((4, 3)))
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('id,fps\nvideo-1,50\n', encoding='utf-8')
    captured = []

    def fake_segment_video(track, positions=None, **kwargs):
        captured.append(kwargs['thresholds'])
        return [], []

    monkeypatch.setattr(stage8, 'segment_video', fake_segment_video)
    args = [
        '--shuttle-dir', str(shuttle_dir), '--rally-spans-csv', str(tmp_path / 'spans.csv'),
        '--contact-frames-csv', str(tmp_path / 'contacts.csv'),
    ]
    if '--fps-csv' in fps_args:
        args.extend(['--fps-csv', str(fps_csv)])
    elif '--fps' in fps_args:
        args.extend(fps_args)
    if '--missing-id' in fps_args:
        (shuttle_dir / 'video-1.npy').rename(shuttle_dir / 'missing-id.npy')
    monkeypatch.setattr(sys, 'argv', ['stage8', *args])
    stage8.main()

    assert captured[0].impulse_floor_half_window_frames == expected_impulse
    if '--missing-id' in fps_args:
        assert 'absent from fps CSV' in caplog.text


@pytest.mark.parametrize(
    ('stride', 'expected_mode'), [(8, 'nonoverlap'), (1, 'weight')],
)
@pytest.mark.parametrize('large_video, present', [(False, False), (True, True)])
def test_extract_shuttle_builds_tracknet_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stride: int,
    expected_mode: str,
    large_video: bool,
    present: bool,
) -> None:
    import src.bric.perception.shuttle as shuttle

    weights_dir = tmp_path / 'weights'
    weights_dir.mkdir()
    (weights_dir / 'TrackNet_best.pt').touch()
    (weights_dir / 'InpaintNet_best.pt').touch()
    video_path = tmp_path / 'clip.mp4'
    video_path.touch()
    save_dir = tmp_path / 'output'
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)
        (save_dir / 'clip_ball.csv').parent.mkdir(parents=True, exist_ok=True)
        (save_dir / 'clip_ball.csv').touch()

    monkeypatch.setattr(shuttle.subprocess, 'run', fake_run)
    shuttle.extract_shuttle(video_path, save_dir, weights_dir, tracknet_stride=stride,
                            large_video=large_video)

    argv = captured[0]
    assert ['--eval_mode', expected_mode] == argv[argv.index('--eval_mode'):argv.index('--eval_mode') + 2]
    assert ('--large_video' in argv) is present


def test_extract_shuttle_rejects_stride_three(tmp_path: Path) -> None:
    import src.bric.perception.shuttle as shuttle

    weights_dir = tmp_path / 'weights'
    weights_dir.mkdir()
    (weights_dir / 'TrackNet_best.pt').touch()
    (weights_dir / 'InpaintNet_best.pt').touch()
    video_path = tmp_path / 'clip.mp4'
    video_path.touch()
    with pytest.raises(ValueError, match='stride must be 1 or 8'):
        shuttle.extract_shuttle(video_path, tmp_path / 'output', weights_dir, tracknet_stride=3)


@pytest.mark.parametrize('stride, expected_mode', [(1, 'weight'), (8, 'nonoverlap')])
def test_batch_shuttle_extractor_maps_mode_without_large_video_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stride: int,
    expected_mode: str,
) -> None:
    import src.bst_x.pipeline.shuttle_extractor as extractor

    tracknet_dir = tmp_path / 'tracknet'
    (tracknet_dir / 'ckpts').mkdir(parents=True)
    (tracknet_dir / 'batch_predict.py').touch()
    (tracknet_dir / 'ckpts' / 'TrackNet_best.pt').touch()
    clips_dir = tmp_path / 'clips'
    clips_dir.mkdir()
    (clips_dir / 'clip.mp4').touch()
    captured = []

    class FakeProcess:
        returncode = 0

        def wait(self):
            return 0

    def fake_popen(args, **kwargs):
        captured.append(args)
        return FakeProcess()

    monkeypatch.setattr(extractor.subprocess, 'Popen', fake_popen)
    extractor.extract_all_shuttles(
        tracknet_dir, clips_dir=clips_dir, output_csv_dir=tmp_path / 'csv',
        max_workers=1, tracknet_stride=stride, dry_run=True,
    )

    argv = captured[0]
    assert argv[argv.index('--eval_mode') + 1] == expected_mode
    assert '--large_video' not in argv


def test_bric_api_pins_tracknet_modes() -> None:
    bric_inference = pytest.importorskip(
        'src.api.bric_inference',
        reason='BRIC API runtime dependencies are unavailable in this test environment',
    )
    assert bric_inference._EVAL_MODE_BY_STRIDE == {1: 'weight', 8: 'nonoverlap'}
    assert bric_inference.TRACKNET_STRIDE == 1
    assert bric_inference.TRACKNET_LARGE_VIDEO is False
