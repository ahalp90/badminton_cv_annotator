"""Regression tests for fps-aware CLIs and TrackNet mode selection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from scraper.config import SCRAPE_TRACKNET_LARGE_VIDEO, SCRAPE_TRACKNET_STRIDE
from annotator.resolve import resolve
from annotator.run_video import AnnotatorResult

def test_stage9_main_scales_composition_min_scene_len(monkeypatch, tmp_path: Path) -> None:
    import annotator.composition_mask as stage9

    keep_vote_path = tmp_path / 'keep_vote.npy'
    np.save(keep_vote_path, np.ones(200, dtype=bool))
    captured: list[int] = []

    def fake_detect_cuts(video_path, expected_frames, threshold, min_scene_len):
        captured.append(min_scene_len)
        return np.array([100], dtype=int)

    monkeypatch.setattr(stage9, 'detect_cuts', fake_detect_cuts)
    monkeypatch.setattr(stage9, 'probe_fps', lambda _video: 50.0)
    common = [
        '--video-id', 'video-1', '--video', str(tmp_path / 'unused.mp4'),
        '--keep-vote', str(keep_vote_path), '--out-dir', str(tmp_path / 'masks'),
    ]
    monkeypatch.setattr(sys, 'argv', ['stage9', *common, '--fps', '60'])
    stage9.main()
    monkeypatch.setattr(sys, 'argv', ['stage9', *common])
    stage9.main()

    assert captured == [30, 25]


@pytest.mark.parametrize(
    ('fps_args', 'expected_impulse'),
    [
        (['--fps-csv'], 20),
        (['--fps-csv', '--missing-id'], None),
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
    import annotator.run_video as run_video_module
    import annotator.rally_segmentation as stage8

    shuttle_dir = tmp_path / 'shuttles'
    shuttle_dir.mkdir()
    np.save(shuttle_dir / 'video-1.npy', np.zeros((4, 3)))
    fps_csv = tmp_path / 'fps.csv'
    fps_csv.write_text('id,fps\nvideo-1,50\n', encoding='utf-8')
    captured = []
    calls = []

    def fake_run_video(track, **kwargs):
        calls.append((track, kwargs))
        captured.append(resolve(kwargs['base'], kwargs['fps']).thresholds)
        return AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, [])

    monkeypatch.setattr(run_video_module, 'run_video', fake_run_video)
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

    if expected_impulse is not None:
        assert captured[0].impulse_floor_half_window_frames == expected_impulse
        assert calls[0][1]['base'].span_open is None
        assert calls[0][1]['court_optional'] is True
        assert calls[0][1]['stop_after_segmentation'] is True
        np.testing.assert_array_equal(calls[0][1]['raw_exclusion_mask'], np.zeros(4, dtype=bool))
    if '--missing-id' in fps_args:
        assert not captured
        assert 'skipping missing-id: absent from fps CSV' in caplog.text


def test_stage8_main_serialises_split_verdicts_to_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The contacts CSV carries both verdict columns, blank on the no-gate path."""
    import annotator.run_video as run_video_module
    import annotator.rally_segmentation as stage8
    from annotator.types import ContactCandidate

    shuttle_dir = tmp_path / 'shuttles'
    shuttle_dir.mkdir()
    np.save(shuttle_dir / 'video-1.npy', np.zeros((4, 3)))
    contacts_csv = tmp_path / 'contacts.csv'

    def fake_run_video(track, **kwargs):
        return AnnotatorResult([(0, 20)], [
            ContactCandidate(0, 1, True, True, False),    # gate winner
            ContactCandidate(0, 5, True, True, True),     # suppression loser
            ContactCandidate(0, 9, False, False, False),  # gate failure
            ContactCandidate(0, 13, None, None, None),    # no-gate path
        ], [], {}, [], [], [], [], {}, {}, {}, {}, [])

    monkeypatch.setattr(run_video_module, 'run_video', fake_run_video)
    monkeypatch.setattr(sys, 'argv', [
        'stage8', '--shuttle-dir', str(shuttle_dir), '--fps', '30',
        '--rally-spans-csv', str(tmp_path / 'spans.csv'),
        '--contact-frames-csv', str(contacts_csv),
    ])
    stage8.main()

    assert contacts_csv.read_text(encoding='utf-8').splitlines() == [
        'video_id,rally_id,contact_frame,proximity_ok,wrist_near,suppressed',
        'video-1,0,1,True,True,False',
        'video-1,0,5,True,True,True',
        'video-1,0,9,False,False,False',
        'video-1,0,13,,,',
    ]


@pytest.mark.parametrize(
    'retired_option',
    ['--gate-dir', '--pose-dir', '--homography-csv', '--resolution-csv',
     '--court-box-csv', '--thresholds'],
)
def test_stage8_main_rejects_retired_options(monkeypatch, tmp_path, retired_option):
    import annotator.rally_segmentation as stage8

    shuttle_dir = tmp_path / 'shuttles'
    shuttle_dir.mkdir()
    option_value = 'shipped' if retired_option == '--thresholds' else str(tmp_path / 'retired')
    monkeypatch.setattr(sys, 'argv', [
        'stage8', '--shuttle-dir', str(shuttle_dir), '--fps', '30',
        retired_option, option_value,
    ])

    with pytest.raises(SystemExit, match='2'):
        stage8.main()


def test_stage8_main_requires_an_fps_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    import annotator.rally_segmentation as stage8

    shuttle_dir = tmp_path / 'shuttles'
    shuttle_dir.mkdir()
    monkeypatch.setattr(sys, 'argv', [
        'stage8', '--shuttle-dir', str(shuttle_dir), '--rally-spans-csv', str(tmp_path / 'spans.csv'),
        '--contact-frames-csv', str(tmp_path / 'contacts.csv'),
    ])
    with pytest.raises(SystemExit, match='2'):
        stage8.main()
    assert 'one of --fps or --fps-csv is required' in capsys.readouterr().err


def test_replay_main_requires_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    import annotator.replay_mask as replay

    monkeypatch.setattr(sys, 'argv', ['replay', '--video-id', 'one'])
    with pytest.raises(SystemExit, match='2'):
        replay.main()


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


@pytest.mark.parametrize(
    ('stride', 'expected_mode'), [(8, 'nonoverlap'), (1, 'weight')],
)
@pytest.mark.parametrize('large_video, present', [(False, False), (True, True), (None, False)])
def test_batch_shuttle_extractor_builds_tracknet_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stride: int,
    expected_mode: str,
    large_video: bool | None,
    present: bool,
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

    kwargs = {
        'tracknet_dir': tracknet_dir,
        'clips_dir': clips_dir,
        'output_csv_dir': tmp_path / 'csv',
        'max_workers': 1,
        'tracknet_stride': stride,
        'dry_run': True,
    }
    if large_video is not None:
        kwargs['large_video'] = large_video

    extractor.extract_all_shuttles(**kwargs)

    argv = captured[0]
    assert argv[argv.index('--eval_mode') + 1] == expected_mode
    assert ('--large_video' in argv) is present


@pytest.mark.parametrize(
    ('cli_args', 'expected_stride', 'expected_large_video'),
    [
        ([], 1, False),
        # The scrape profile's expectations come from scraper.config, so a deliberate
        # profile change moves the test with it; explicit CLI overrides stay literal.
        (['--profile', 'scrape'], SCRAPE_TRACKNET_STRIDE, SCRAPE_TRACKNET_LARGE_VIDEO),
        (['--profile', 'scrape', '--tracknet-stride', '1'], 1, SCRAPE_TRACKNET_LARGE_VIDEO),
        (['--profile', 'scrape', '--no-large-video'], SCRAPE_TRACKNET_STRIDE, False),
        (['--large-video'], 1, True),
    ],
)
def test_batch_shuttle_extractor_main_resolves_profiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cli_args: list[str],
    expected_stride: int,
    expected_large_video: bool,
) -> None:
    import src.bst_x.pipeline.shuttle_extractor as extractor

    captured = []

    def fake_extract(*args, **kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(extractor, 'extract_all_shuttles', fake_extract)
    monkeypatch.setattr(sys, 'argv', ['shuttle_extractor', '--tracknet-dir', str(tmp_path), '--dry-run', *cli_args])

    extractor.main()

    assert captured[0]['tracknet_stride'] == expected_stride
    assert captured[0]['large_video'] is expected_large_video
