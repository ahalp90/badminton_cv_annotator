"""Regression tests for fps-aware CLIs and TrackNet mode selection."""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scraper.config import SCRAPE_TRACKNET_LARGE_VIDEO, SCRAPE_TRACKNET_STRIDE


def test_composition_mask_main_scales_composition_min_scene_len(monkeypatch, tmp_path: Path) -> None:
    import experiments.annotator.scene_cut_mask as composition_mask

    keep_vote_path = tmp_path / 'keep_vote.npy'
    np.save(keep_vote_path, np.ones(200, dtype=bool))
    captured: list[int] = []

    def fake_detect_cuts(video_path, expected_frames, threshold, min_scene_len):
        captured.append(min_scene_len)
        return np.array([100], dtype=int)

    monkeypatch.setattr(composition_mask, 'detect_cuts', fake_detect_cuts)
    monkeypatch.setattr(composition_mask, 'probe_fps', lambda _video: 50.0)
    common = [
        '--video-id', 'video-1', '--video', str(tmp_path / 'unused.mp4'),
        '--keep-vote', str(keep_vote_path), '--out-dir', str(tmp_path / 'masks'),
    ]
    monkeypatch.setattr(sys, 'argv', ['composition_mask', *common, '--fps', '60'])
    composition_mask.main()
    monkeypatch.setattr(sys, 'argv', ['composition_mask', *common])
    composition_mask.main()

    assert captured == [30, 25]


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
    from src.bric.perception import shuttle

    weights_dir = tmp_path / 'weights'
    weights_dir.mkdir()
    (weights_dir / 'TrackNet_best.pt').touch()
    (weights_dir / 'InpaintNet_best.pt').touch()
    video_path = tmp_path / 'clip.mp4'
    video_path.touch()
    save_dir = tmp_path / 'output'
    captured = []

    def fake_run(cmd, **kwargs):
        captured.append((cmd, kwargs))
        (save_dir / 'clip_ball.csv').parent.mkdir(parents=True, exist_ok=True)
        (save_dir / 'clip_ball.csv').touch()

    monkeypatch.setattr(shuttle.subprocess, 'run', fake_run)
    shuttle.extract_shuttle(video_path, save_dir, weights_dir, tracknet_stride=stride,
                            large_video=large_video)

    argv, run_kwargs = captured[0]
    assert ['--eval_mode', expected_mode] == argv[argv.index('--eval_mode'):argv.index('--eval_mode') + 2]
    assert ('--large_video' in argv) is present
    expected_dir = Path(shuttle.__file__).resolve().parents[2] / 'shared' / 'tracknetv3'
    assert run_kwargs['cwd'] == expected_dir


def test_extract_shuttle_rejects_stride_three(tmp_path: Path) -> None:
    from src.bric.perception import shuttle

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
    (tracknet_dir / 'ckpts' / 'InpaintNet_best.pt').touch()
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
    assert '--inpaintnet_file' in argv


def test_batch_shuttle_extractor_accepts_an_explicit_non_mp4_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.bst_x.pipeline.shuttle_extractor as extractor

    tracknet_dir = tmp_path / 'tracknet'
    (tracknet_dir / 'ckpts').mkdir(parents=True)
    (tracknet_dir / 'batch_predict.py').touch()
    (tracknet_dir / 'ckpts' / 'InpaintNet_best.pt').touch()
    model = tmp_path / 'tracknet.pt'
    model.touch()
    video = tmp_path / 'selected.mkv'
    video.touch()
    output_dir = tmp_path / 'csv'
    captured = []

    class FakeProcess:
        returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(
        extractor.subprocess,
        'Popen',
        lambda args, **_kwargs: captured.append(args) or FakeProcess(),
    )

    extractor.extract_all_shuttles(
        tracknet_dir=tracknet_dir,
        clips_dir=tmp_path / 'unused',
        video_paths=[video],
        output_csv_dir=output_dir,
        model_path=model,
        max_workers=1,
        dry_run=True,
        enable_inpainting=False,
    )

    argv = captured[0]
    list_path = Path(argv[argv.index('--video_list') + 1])
    assert list_path.parent == output_dir
    assert '--inpaintnet_file' not in argv


def test_batch_shuttle_receiver_exposes_the_stream_contract() -> None:
    import src.bst_x.pipeline.shuttle_extractor as extractor

    parameters = inspect.signature(extractor.extract_all_shuttles).parameters

    assert {
        "input_mode",
        "ffmpeg",
        "expected_frame_count",
        "input_video_identity",
    } <= parameters.keys()


def test_exact_stream_tracknet_command_carries_identity_and_process_ownership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.bst_x.pipeline.shuttle_extractor as extractor

    tracknet_dir = tmp_path / "tracknet"
    tracknet_dir.mkdir()
    (tracknet_dir / "batch_predict.py").touch()
    model = tmp_path / "tracknet.pt"
    model.touch()
    source = tmp_path / "canonical.mkv"
    source.touch()
    captured: list[tuple[list[str], dict[str, object]]] = []

    class FakeProcess:
        returncode = 0
        pid = 12_345

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(
        extractor.subprocess,
        "Popen",
        lambda args, **kwargs: captured.append((args, kwargs)) or FakeProcess(),
    )

    extractor.extract_all_shuttles(
        tracknet_dir=tracknet_dir,
        clips_dir=tmp_path / "unused",
        video_paths=[source],
        output_csv_dir=tmp_path / "csv",
        model_path=model,
        max_workers=1,
        batch_size=16,
        tracknet_stride=8,
        large_video=True,
        dry_run=True,
        enable_inpainting=False,
        input_mode="exact_ffv1_stream",
        ffmpeg="/fixture/ffmpeg",
        expected_frame_count=99,
        input_video_identity=source,
    )

    argv, popen_kwargs = captured[0]
    assert argv[argv.index("--input_mode") + 1] == "exact_ffv1_stream"
    assert argv[argv.index("--ffmpeg") + 1] == "/fixture/ffmpeg"
    assert argv[argv.index("--expected_frame_count") + 1] == "99"
    assert argv[argv.index("--input_video_identity") + 1] == str(source)
    assert argv[argv.index("--batch_size") + 1] == "16"
    assert argv[argv.index("--eval_mode") + 1] == "nonoverlap"
    assert popen_kwargs["start_new_session"] is True
    assert popen_kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(extractor._SRC)


def test_tracknet_child_imports_stream_reader_without_inherited_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.bst_x.pipeline.shuttle_extractor as extractor

    monkeypatch.delenv("PYTHONPATH", raising=False)
    tracknet_dir = extractor._SRC / "shared" / "tracknetv3"
    command = (
        "import sys; "
        f"sys.path.insert(0, {str(tracknet_dir)!r}); "
        "from dataset import ExactFFV1StreamDataset; "
        "from dataset_builder.ffv1_stream import iter_exact_ffv1_frames; "
        "assert ExactFFV1StreamDataset and iter_exact_ffv1_frames"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=extractor._tracknet_subprocess_environment(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("failure", ["startup", "worker", "cancellation"])
def test_batch_shuttle_failure_cleans_lists_and_owned_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    import src.bst_x.pipeline.shuttle_extractor as extractor

    tracknet_dir = tmp_path / "tracknet"
    tracknet_dir.mkdir()
    (tracknet_dir / "batch_predict.py").touch()
    model = tmp_path / "tracknet.pt"
    model.touch()
    videos = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    for video in videos:
        video.touch()
    output_dir = tmp_path / "csv"
    processes: list[FakeProcess] = []
    termination_calls: list[tuple[FakeProcess, ...]] = []

    class FakeProcess:
        pid = 20_000

        def __init__(self) -> None:
            self.returncode = 7 if failure == "worker" else 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if failure == "cancellation":
                raise KeyboardInterrupt
            return self.returncode

    def popen(*_args: object, **_kwargs: object) -> FakeProcess:
        if failure == "startup" and processes:
            raise OSError("fixture startup failure")
        process = FakeProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(extractor.subprocess, "Popen", popen)
    monkeypatch.setattr(
        extractor,
        "_terminate_process_groups",
        lambda owned: termination_calls.append(tuple(owned)),
    )

    expected_error = OSError if failure == "startup" else (
        RuntimeError if failure == "worker" else KeyboardInterrupt
    )
    with pytest.raises(expected_error):
        extractor.extract_all_shuttles(
            tracknet_dir=tracknet_dir,
            clips_dir=tmp_path / "unused",
            video_paths=videos,
            output_csv_dir=output_dir,
            model_path=model,
            max_workers=2,
            dry_run=True,
            enable_inpainting=False,
        )

    assert termination_calls == [tuple(processes)]
    assert not list(output_dir.glob("_pending_clips_*.txt"))


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


def test_batch_shuttle_extractor_defaults_to_shared_tracknet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.bst_x.pipeline.shuttle_extractor as extractor

    captured = []
    monkeypatch.setattr(extractor, 'extract_all_shuttles', lambda **kwargs: captured.append(kwargs))
    monkeypatch.setattr(sys, 'argv', ['shuttle_extractor', '--dry-run'])

    extractor.main()

    assert captured[0]['tracknet_dir'] == extractor.DEFAULT_TRACKNET_DIR
