"""Failure circuit-breaker tests for the two pose-extraction batch entry points.

Covers ``preparing_data.raw_extract.main`` and
``preparing_data.prepare_train_on_shuttleset.prepare_dataset_npy_from_raw_video``:
a clip that decodes zero frames must be logged to ``failed_clips.log`` and
skipped (never crashing on ``np.stack([])``), and the batch must abort loudly
once failures exceed 0.3 of the clips slated for extraction this run.

Runs with NO rtmlib installed (CI has none). Both entry points import the
adapter lazily via ``from preparing_data.rtmlib_pose import RtmlibPoseExtractor``,
so we inject a stub module into ``sys.modules['preparing_data.rtmlib_pose']``
before the call. The stub's ``iter_video`` is driven by a per-stem frame count:
a stem mapped to 0 yields nothing (the zero-frame failure), any other yields a
few duck-typed FrameDetections whose keypoints/bboxes/scores are correctly
shaped but arbitrary in value.
"""
from __future__ import annotations

import sys
import types
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import preparing_data.raw_extract as raw_extract
from pipeline.config import COCO_N_JOINTS
from preparing_data.extract_failures import FAILED_CLIPS_LOG
from preparing_data.heuristics.base import RAW_SUFFIXES
from preparing_data.prepare_train_on_shuttleset import prepare_dataset_npy_from_raw_video

_J = COCO_N_JOINTS  # 17, the COCO keypoint count the real adapter emits

# Duck-typed stand-in for rtmlib_pose.FrameDetections: the entry points only
# read these four attributes off each yielded frame.
_FakeFrame = namedtuple("_FakeFrame", ["keypoints", "bboxes", "bbox_scores", "kp_scores"])


# ---------------------------------------------------------------------------
# Stub adapter + fixture builders
# ---------------------------------------------------------------------------


def _fake_frame(n_people: int) -> _FakeFrame:
    """One frame's detections: ``n_people`` rows of correctly-shaped dummy data.

    ``n_people >= 2`` exercises raw_extract's NaN-padding path; ``< 2`` makes
    detect_players_2d zero-fill the frame (its ``len < 2`` short-circuit fires
    before any court projection, so no homography is needed).
    """
    return _FakeFrame(
        keypoints=np.full((n_people, _J, 2), 0.5, dtype=np.float32),
        bboxes=np.tile(np.array([0.0, 0.0, 10.0, 20.0], np.float32), (n_people, 1)),
        bbox_scores=np.full((n_people,), 0.9, dtype=np.float32),
        kp_scores=np.full((n_people, _J), 0.8, dtype=np.float32),
    )


def _make_stub_rtmlib_module(
    frames_per_stem: dict[str, int],
    default_frames: int = 3,
    people_per_frame: int = 2,
) -> types.ModuleType:
    """Build a fake ``preparing_data.rtmlib_pose`` module to inject into sys.modules.

    :param frames_per_stem: per-stem frame count override; a stem mapped to 0
        yields nothing (the zero-frame failure the guard must catch).
    :param default_frames: frame count for any stem not in ``frames_per_stem``.
    :param people_per_frame: detections per yielded frame (2 for raw_extract's
        padding path, 1 for detect_players_2d's zero-fill path).
    """
    module = types.ModuleType("preparing_data.rtmlib_pose")

    class StubExtractor:
        def __init__(self, device: str = "cpu", **kwargs) -> None:
            pass

        def iter_video(self, video_path):
            stem = Path(video_path).stem
            n_frames = frames_per_stem.get(stem, default_frames)
            for _frame in range(n_frames):
                yield _fake_frame(people_per_frame)

    module.RtmlibPoseExtractor = StubExtractor
    module.FrameDetections = _FakeFrame
    return module


def _write_clips_and_stems(clips_dir: Path, stems_file: Path, stems: list[str]) -> None:
    """Create an empty ``.mp4`` per stem (content is irrelevant; iter_video is
    stubbed) and a one-stem-per-line stems file for raw_extract's --clip-stems-file."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (clips_dir / f"{stem}.mp4").write_bytes(b"")
    stems_file.write_text("\n".join(stems) + "\n")


def _seed_done_raw_clip(save_dir: Path, stem: str, n_max: int, n_frames: int = 1) -> None:
    """Write the five conforming raw npys so raw_extract's pre-pass reads the clip
    as already-done and skips it (_raw_ndet present + bboxes N_max == n_max)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    branch = str(save_dir / stem)
    np.save(branch + "_raw_kps.npy", np.zeros((n_frames, n_max, _J, 2), np.float32))
    np.save(branch + "_raw_bboxes.npy", np.zeros((n_frames, n_max, 4), np.float32))
    np.save(branch + "_raw_scores.npy", np.zeros((n_frames, n_max), np.float32))
    np.save(branch + "_raw_kp_scores.npy", np.zeros((n_frames, n_max, _J), np.float32))
    np.save(branch + "_raw_ndet.npy", np.zeros((n_frames,), np.int8))


def _run_raw_extract(monkeypatch, stub_module, clips_dir, stems_file, save_dir, n_max):
    """Inject the stub adapter, set argv, and run raw_extract.main()."""
    monkeypatch.setitem(sys.modules, "preparing_data.rtmlib_pose", stub_module)
    monkeypatch.setattr(sys, "argv", [
        "raw_extract",
        "--clips-dir", str(clips_dir),
        "--clip-stems-file", str(stems_file),
        "--save-dir", str(save_dir),
        "--n-max", str(n_max),
        "--device", "cpu",
    ])
    return raw_extract.main()


def _raw_files_present(save_dir: Path, stem: str) -> list[bool]:
    return [(save_dir / (stem + suffix)).exists() for suffix in RAW_SUFFIXES]


def _read_log_stems(save_dir: Path) -> list[str]:
    """Return the stem column (first tab field) of every failed_clips.log line."""
    lines = (save_dir / FAILED_CLIPS_LOG).read_text().splitlines()
    return [line.split("\t")[0] for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# raw_extract
# ---------------------------------------------------------------------------


def test_raw_extract_normal_path_writes_five_npys(tmp_path, monkeypatch):
    """All clips decode fine: five npys per clip, no failure log, exit 0."""
    clips_dir, save_dir = tmp_path / "clips", tmp_path / "save"
    stems_file = tmp_path / "stems.txt"
    stems = ["11_1_1_1", "11_1_1_2", "11_1_1_3"]
    _write_clips_and_stems(clips_dir, stems_file, stems)

    n_max, n_frames = 4, 3
    stub = _make_stub_rtmlib_module({}, default_frames=n_frames, people_per_frame=2)
    rc = _run_raw_extract(monkeypatch, stub, clips_dir, stems_file, save_dir, n_max)

    assert rc == 0
    assert not (save_dir / FAILED_CLIPS_LOG).exists()
    for stem in stems:
        assert all(_raw_files_present(save_dir, stem))

    # Documented per-clip shapes/dtypes (F=n_frames, N_max=n_max, J=17).
    branch = str(save_dir / stems[0])
    kps = np.load(branch + "_raw_kps.npy")
    bboxes = np.load(branch + "_raw_bboxes.npy")
    scores = np.load(branch + "_raw_scores.npy")
    kp_scores = np.load(branch + "_raw_kp_scores.npy")
    ndet = np.load(branch + "_raw_ndet.npy")
    assert kps.shape == (n_frames, n_max, _J, 2) and kps.dtype == np.float32
    assert bboxes.shape == (n_frames, n_max, 4) and bboxes.dtype == np.float32
    assert scores.shape == (n_frames, n_max) and scores.dtype == np.float32
    assert kp_scores.shape == (n_frames, n_max, _J) and kp_scores.dtype == np.float32
    assert ndet.shape == (n_frames,) and ndet.dtype == np.int8
    # 2 real detections per frame, padded to N_max.
    assert np.array_equal(ndet, np.full((n_frames,), 2, dtype=np.int8))


def test_raw_extract_single_bad_clip_below_threshold(tmp_path, monkeypatch):
    """One zero-frame clip among five: logged + skipped, others intact, exit 0."""
    clips_dir, save_dir = tmp_path / "clips", tmp_path / "save"
    stems_file = tmp_path / "stems.txt"
    stems = ["11_1_1_1", "11_1_1_2", "11_1_1_3", "11_1_1_4", "11_1_1_5"]
    _write_clips_and_stems(clips_dir, stems_file, stems)

    bad = "11_1_1_3"
    stub = _make_stub_rtmlib_module({bad: 0}, default_frames=3, people_per_frame=2)
    rc = _run_raw_extract(monkeypatch, stub, clips_dir, stems_file, save_dir, n_max=4)

    assert rc == 0
    assert _read_log_stems(save_dir) == [bad]
    assert not any(_raw_files_present(save_dir, bad))
    for stem in stems:
        if stem != bad:
            assert all(_raw_files_present(save_dir, stem))


def test_raw_extract_aborts_past_threshold(tmp_path, monkeypatch, capsys):
    """Two of five clips fail: 2 > 0.3 * 5, so the batch aborts with exit 1."""
    clips_dir, save_dir = tmp_path / "clips", tmp_path / "save"
    stems_file = tmp_path / "stems.txt"
    stems = ["11_1_1_1", "11_1_1_2", "11_1_1_3", "11_1_1_4", "11_1_1_5"]
    _write_clips_and_stems(clips_dir, stems_file, stems)

    # Two failing clips first in stems order so the abort fires at the 2nd.
    fails = ["11_1_1_1", "11_1_1_2"]
    stub = _make_stub_rtmlib_module(
        {stem: 0 for stem in fails}, default_frames=3, people_per_frame=2
    )
    rc = _run_raw_extract(monkeypatch, stub, clips_dir, stems_file, save_dir, n_max=4)

    assert rc == 1
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert FAILED_CLIPS_LOG in out  # abort message names the log
    assert _read_log_stems(save_dir) == fails


def test_raw_extract_resume_skips_excluded_from_denominator(tmp_path, monkeypatch):
    """Resume-skipped clips are NOT slated, so they don't dilute the 0.3 fraction.

    Ten clips, six already-done (skipped), four to extract, two of them fail.
    2 > 0.3 * 4 (the not-done count) aborts; 2 < 0.3 * 10 would not. The abort
    proves the denominator is the not-done count, not the total.
    """
    clips_dir, save_dir = tmp_path / "clips", tmp_path / "save"
    stems_file = tmp_path / "stems.txt"
    stems = [f"11_1_1_{i}" for i in range(10)]
    _write_clips_and_stems(clips_dir, stems_file, stems)

    n_max = 4
    done = stems[:6]
    for stem in done:
        _seed_done_raw_clip(save_dir, stem, n_max)

    fails = stems[6:8]  # two of the four not-done clips fail
    stub = _make_stub_rtmlib_module(
        {stem: 0 for stem in fails}, default_frames=3, people_per_frame=2
    )
    rc = _run_raw_extract(monkeypatch, stub, clips_dir, stems_file, save_dir, n_max)

    assert rc == 1
    assert _read_log_stems(save_dir) == fails


# ---------------------------------------------------------------------------
# prepare_dataset_npy_from_raw_video
# ---------------------------------------------------------------------------


def _dummy_court_and_res() -> tuple[dict, pd.DataFrame]:
    """Minimal court + resolution stand-ins. Never read on the paths exercised
    here: zero-frame clips return None before court info is touched, and the
    <2-detections clips zero-fill each frame before any court projection."""
    res_df = pd.DataFrame({"width": [1280], "height": [720]}, index=[11])
    res_df.index.name = "id"
    return {}, res_df


def test_prepare_zero_frame_logged_and_skipped_below_threshold(tmp_path, monkeypatch):
    """One zero-frame clip among five: logged, writes no npys, no raise.

    The other four yield single-detection frames, so detect_players_2d returns a
    valid (all-zeroed) result and writes _pos/_joints/_failed without needing
    real homography.
    """
    clips_dir, save_dir = tmp_path / "clips", tmp_path / "save"
    stems = [f"11_1_1_{i}" for i in range(5)]
    _write_clips_and_stems(clips_dir, tmp_path / "stems.txt", stems)

    bad = "11_1_1_2"
    stub = _make_stub_rtmlib_module({bad: 0}, default_frames=3, people_per_frame=1)
    monkeypatch.setitem(sys.modules, "preparing_data.rtmlib_pose", stub)
    all_court_info, res_df = _dummy_court_and_res()

    prepare_dataset_npy_from_raw_video(
        my_clips_folder=clips_dir,
        save_root_dir=save_dir,
        resolution_df=res_df,
        all_court_info=all_court_info,
        device="cpu",
    )

    assert _read_log_stems(save_dir) == [bad]
    prepare_suffixes = ("_pos.npy", "_joints.npy", "_failed.npy")
    for suffix in prepare_suffixes:
        assert not (save_dir / (bad + suffix)).exists()
    for stem in stems:
        if stem != bad:
            for suffix in prepare_suffixes:
                assert (save_dir / (stem + suffix)).exists()


def test_prepare_aborts_past_threshold_raises(tmp_path, monkeypatch):
    """Four zero-frame clips: 2 > 0.3 * 4 raises RuntimeError naming the log."""
    clips_dir, save_dir = tmp_path / "clips", tmp_path / "save"
    stems = [f"11_1_1_{i}" for i in range(4)]
    _write_clips_and_stems(clips_dir, tmp_path / "stems.txt", stems)

    stub = _make_stub_rtmlib_module(
        {stem: 0 for stem in stems}, default_frames=3, people_per_frame=1
    )
    monkeypatch.setitem(sys.modules, "preparing_data.rtmlib_pose", stub)
    all_court_info, res_df = _dummy_court_and_res()

    with pytest.raises(RuntimeError, match=r"failed_clips\.log"):
        prepare_dataset_npy_from_raw_video(
            my_clips_folder=clips_dir,
            save_root_dir=save_dir,
            resolution_df=res_df,
            all_court_info=all_court_info,
            device="cpu",
        )

    # Two failures logged before the abort; no npys written for any clip.
    assert _read_log_stems(save_dir) == stems[:2]
    for stem in stems:
        for suffix in ("_pos.npy", "_joints.npy", "_failed.npy"):
            assert not (save_dir / (stem + suffix)).exists()
