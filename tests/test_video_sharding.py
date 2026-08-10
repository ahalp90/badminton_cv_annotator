"""Deterministic tests for the video-sharding PoC (no rtmlib, no GPU).

Covers: shard planning, fake-extractor sequential-vs-sharded parity (seek and
scan decode), the stitch integrity guards, worker short-read/overrun policy,
and downstream compatibility of the published arrays with the production
loader + heuristics.

Real-video decode identity and real-inference parity are gated separately in
``shared/video_sharding/gate_*`` (they need a match video / GPU).
"""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from preparing_data.apply_heuristic import _load_raw_clip, _raw_files_present
from preparing_data.heuristics import REGISTRY
from preparing_data.heuristics.base import RAW_SUFFIXES, ClipContext
from preparing_data.raw_extract import extract_one_clip

from shared.video_sharding import shard_worker as shard_worker_module
from shared.video_sharding.fake_pose import DeterministicFakeExtractor
from shared.video_sharding.run_sharded import extract_sharded
from shared.video_sharding.shard_plan import plan_frame_shards
from shared.video_sharding.shard_worker import (
    load_gz_json,
    run_shard,
    save_gz_json,
    save_npy_xz,
    shard_stem,
)
from shared.video_sharding.stitch import (
    RUN_MANIFEST_NAME,
    StitchError,
    stitch_and_publish,
)

N_FRAMES = 120  # 4 GOPs of 30 so mid-video seeks cross keyframe boundaries


@pytest.fixture(scope="module")
def sharding_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """120 distinct-content frames, x264 lossless, 30-frame GOPs."""
    width, height = 96, 64
    output = tmp_path_factory.mktemp("sharding-video") / "tiny_match.mp4"
    ramp_x = np.arange(width, dtype=np.uint16)[None, :, None]
    ramp_y = np.arange(height, dtype=np.uint16)[:, None, None]
    frames = [
        (((ramp_x * 3 + ramp_y * 5) + index * 7) % 256).astype(np.uint8).repeat(3, axis=2)
        for index in range(N_FRAMES)
    ]
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-framerate", "30", "-i", "-",
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-g", "30",
        "-pix_fmt", "yuv420p", str(output),
    ]
    completed = subprocess.run(
        command, input=b"".join(frame.tobytes() for frame in frames),
        capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    return output


def load_five(save_dir: Path, stem: str) -> list[np.ndarray]:
    return [np.load(save_dir / f"{stem}{suffix}") for suffix in RAW_SUFFIXES]


def assert_five_equal(a_dir: Path, a_stem: str, b_dir: Path, b_stem: str) -> None:
    for suffix, left, right in zip(
        RAW_SUFFIXES, load_five(a_dir, a_stem), load_five(b_dir, b_stem)
    ):
        equal_nan = left.dtype != np.int8
        assert np.array_equal(left, right, equal_nan=equal_nan), f"{suffix} differs"


# --- shard planning -----------------------------------------------------------

def test_plan_covers_exactly() -> None:
    for n_frames, n_shards in [(120, 4), (7, 3), (100349, 8), (5, 5)]:
        plan = plan_frame_shards(n_frames, n_shards)
        assert plan[0][0] == 0 and plan[-1][1] == n_frames
        assert all(b[0] == a[1] for a, b in itertools.pairwise(plan))
        sizes = [end - start for start, end in plan]
        assert max(sizes) - min(sizes) <= 1
        assert sum(sizes) == n_frames


def test_plan_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        plan_frame_shards(0, 2)
    with pytest.raises(ValueError):
        plan_frame_shards(10, 0)
    with pytest.raises(ValueError):
        plan_frame_shards(3, 4)


# --- sequential vs sharded parity (deterministic fake) ------------------------

@pytest.mark.parametrize("decode_mode", ["seek", "scan"])
def test_fake_parity_sequential_vs_sharded(sharding_video: Path, tmp_path: Path, decode_mode: str) -> None:
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    assert extract_one_clip(
        extractor=DeterministicFakeExtractor(),
        video_path=sharding_video,
        save_branch=str(seq_dir / "1_seq"),
        n_max=16,
        over_det_warned=set(),
    )
    publish = extract_sharded(
        video_path=sharding_video,
        out_root=tmp_path / "sharded",
        stem="1_sharded",
        n_shards=5,  # uneven split of 120
        extractor_spec="fake",
        decode_mode=decode_mode,
        expected_frame_count=N_FRAMES,
    )
    assert_five_equal(seq_dir, "1_seq", publish, "1_sharded")


# --- worker failure policy ----------------------------------------------------

def make_worker_kwargs(video: Path, run_dir: Path, start: int, end: int, **overrides) -> dict:
    kwargs = {
        "video_path": str(video),
        "start": start,
        "end": end,
        "n_max": 16,
        "run_dir": str(run_dir),
        "run_id": "testrun",
        "source_md5": "irrelevant",
        "extractor_spec": "fake",
    }
    kwargs.update(overrides)
    return kwargs


def test_worker_short_read_raises(sharding_video: Path, tmp_path: Path) -> None:
    """A range past EOF must raise, not write a small-but-plausible shard."""
    with pytest.raises(RuntimeError, match="short shard"):
        run_shard(**make_worker_kwargs(sharding_video, tmp_path, N_FRAMES - 5, N_FRAMES + 5))
    assert not list(tmp_path.glob("shard_*_manifest.json.gz"))


def test_worker_probe_detects_undercounted_plan(sharding_video: Path, tmp_path: Path) -> None:
    """probe_past_end fires when the source has frames beyond the planned end."""
    with pytest.raises(RuntimeError, match="past planned end"):
        run_shard(**make_worker_kwargs(
            sharding_video, tmp_path, N_FRAMES - 20, N_FRAMES - 10, probe_past_end=True,
        ))


def test_failed_worker_aborts_run(
    sharding_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker planned past EOF exits nonzero; the run aborts, nothing publishes.

    The lying plan comes from patching the parent's frame-count probe; the
    worker's failure is genuine (it short-reads in its own process).
    """
    from shared.video_sharding import run_sharded as run_sharded_module

    monkeypatch.setattr(run_sharded_module, "metadata_frame_count", lambda _: N_FRAMES + 8)
    with pytest.raises(RuntimeError, match="worker"):
        extract_sharded(
            video_path=sharding_video,
            out_root=tmp_path,
            stem="1_bad",
            n_shards=4,
            extractor_spec="fake",
        )
    assert not list(tmp_path.glob("publish_*"))


def test_canonical_frame_count_mismatch_refuses_before_worker_spawn(
    sharding_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shared.video_sharding import run_sharded as run_sharded_module

    monkeypatch.setattr(run_sharded_module, "metadata_frame_count", lambda _: N_FRAMES + 1)

    with pytest.raises(ValueError, match="differs from canonical metadata"):
        extract_sharded(
            video_path=sharding_video,
            out_root=tmp_path,
            stem="1_bad_count",
            n_shards=4,
            extractor_spec="fake",
            expected_frame_count=N_FRAMES,
        )

    assert not list(tmp_path.iterdir())


def test_shard_compression_streams_through_atomic_xz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "shard.npy.xz"
    values = np.arange(24, dtype=np.float32).reshape(3, 8)
    real_open = shard_worker_module.lzma.open
    observed: list[tuple[str, int | None, int | None]] = []

    def tracked_open(
        target: Path,
        mode: str,
        *,
        format: int | None = None,
        preset: int | None = None,
    ) -> object:
        observed.append((mode, format, preset))
        return real_open(target, mode, format=format, preset=preset)

    def reject_buffered_write(_path: Path, _payload: bytes) -> None:
        raise AssertionError("NumPy shard compression must not buffer the whole payload")

    monkeypatch.setattr(shard_worker_module.lzma, "open", tracked_open)
    monkeypatch.setattr(shard_worker_module, "atomic_write_bytes", reject_buffered_write)

    save_npy_xz(path, values)

    assert observed == [("wb", shard_worker_module.lzma.FORMAT_XZ, 9)]
    monkeypatch.setattr(shard_worker_module.lzma, "open", real_open)
    np.testing.assert_array_equal(shard_worker_module.load_npy_xz(path), values)


# --- stitch integrity ---------------------------------------------------------

@pytest.fixture()
def completed_run(sharding_video: Path, tmp_path: Path) -> tuple[Path, Path]:
    """A finished 3-shard fake run, stitched inputs intact, not yet published."""
    out_root = tmp_path / "run_root"
    extract_sharded(
        video_path=sharding_video,
        out_root=out_root,
        stem="1_ok",
        n_shards=3,
        extractor_spec="fake",
        run_id="fixedrun",
    )
    return out_root / "run_fixedrun", out_root


def assert_stitch_refuses(run_dir: Path, tmp_path: Path, match: str) -> None:
    publish_dir = tmp_path / "retry_publish"
    with pytest.raises(StitchError, match=match):
        stitch_and_publish(run_dir, publish_dir, "1_retry")
    # A refused stitch must never leave output that looks complete: ndet (the
    # completeness marker) must not exist even if earlier arrays landed.
    assert not (publish_dir / "1_retry_raw_ndet.npy").exists()


def test_stitch_missing_shard(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    (run_dir / f"{shard_stem(40, 80)}_manifest.json.gz").unlink()
    assert_stitch_refuses(run_dir, tmp_path, "incomplete")


def test_stitch_stale_mixed_run(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    manifest_path = run_dir / f"{shard_stem(0, 40)}_manifest.json.gz"
    manifest = load_gz_json(manifest_path)
    manifest["run_id"] = "olderrun"
    save_gz_json(manifest_path, manifest)
    assert_stitch_refuses(run_dir, tmp_path, "stale or mixed")


def test_stitch_source_mismatch(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    manifest_path = run_dir / f"{shard_stem(80, 120)}_manifest.json.gz"
    manifest = load_gz_json(manifest_path)
    manifest["source_md5"] = "0" * 32
    save_gz_json(manifest_path, manifest)
    assert_stitch_refuses(run_dir, tmp_path, "different source")


def test_stitch_n_max_mismatch(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    manifest_path = run_dir / f"{shard_stem(0, 40)}_manifest.json.gz"
    manifest = load_gz_json(manifest_path)
    manifest["n_max"] = 8
    save_gz_json(manifest_path, manifest)
    assert_stitch_refuses(run_dir, tmp_path, "n_max")


@pytest.mark.parametrize("field", ["extractor", "decode_mode"])
def test_stitch_extractor_or_decode_mode_mismatch(
    completed_run: tuple[Path, Path],
    tmp_path: Path,
    field: str,
) -> None:
    run_dir, _ = completed_run
    manifest_path = run_dir / f"{shard_stem(0, 40)}_manifest.json.gz"
    manifest = load_gz_json(manifest_path)
    manifest[field] = "different"
    save_gz_json(manifest_path, manifest)
    assert_stitch_refuses(run_dir, tmp_path, field)


def test_stitch_short_shard_manifest(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    manifest_path = run_dir / f"{shard_stem(40, 80)}_manifest.json.gz"
    manifest = load_gz_json(manifest_path)
    manifest["frames_read"] = 39
    save_gz_json(manifest_path, manifest)
    assert_stitch_refuses(run_dir, tmp_path, "read 39 frames")


def test_stitch_missing_array_file(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    (run_dir / f"{shard_stem(40, 80)}_raw_kps.npy.xz").unlink()
    assert_stitch_refuses(run_dir, tmp_path, "missing array file")


def test_stitch_tampered_array_shape(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    path = run_dir / f"{shard_stem(40, 80)}_raw_scores.npy.xz"
    save_npy_xz(path, np.zeros((39, 16), dtype=np.float32))
    assert_stitch_refuses(run_dir, tmp_path, "shape")


def test_stitch_unplanned_extra_shard(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    stray = load_gz_json(run_dir / f"{shard_stem(0, 40)}_manifest.json.gz")
    save_gz_json(run_dir / f"{shard_stem(0, 20)}_manifest.json.gz", stray)
    assert_stitch_refuses(run_dir, tmp_path, "unplanned")


def test_stitch_plan_gap_and_overlap(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    run_manifest = load_gz_json(run_dir / RUN_MANIFEST_NAME)
    for bad_plan, expected in [
        ([[0, 0], [0, 120]], "invalid range"),
        ([[0, 40], [41, 80], [80, 120]], "gap"),
        ([[0, 41], [40, 80], [80, 120]], "overlap"),
        ([[0, 40], [40, 80]], "ends at"),
    ]:
        tampered = dict(run_manifest, plan=bad_plan)
        save_gz_json(run_dir / RUN_MANIFEST_NAME, tampered)
        assert_stitch_refuses(run_dir, tmp_path, expected)


def test_stitch_rejects_out_of_order_plan(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    run_manifest = load_gz_json(run_dir / RUN_MANIFEST_NAME)
    shuffled = [[40, 80], [0, 40], [80, 120]]
    save_gz_json(run_dir / RUN_MANIFEST_NAME, dict(run_manifest, plan=shuffled))
    assert_stitch_refuses(run_dir, tmp_path, "ascending frame order")


def test_stitch_shard_count_mismatch(completed_run, tmp_path: Path) -> None:
    run_dir, _ = completed_run
    run_manifest = load_gz_json(run_dir / RUN_MANIFEST_NAME)
    save_gz_json(run_dir / RUN_MANIFEST_NAME, dict(run_manifest, n_shards=2))
    assert_stitch_refuses(run_dir, tmp_path, "plan has 3 ranges")


def test_stitch_partial_write_leftover_is_ignored_but_shard_incomplete(
    completed_run, tmp_path: Path,
) -> None:
    """A crashed worker leaves ``.tmp`` files and no manifest: run refused."""
    run_dir, _ = completed_run
    manifest_path = run_dir / f"{shard_stem(80, 120)}_manifest.json.gz"
    manifest_path.unlink()
    (run_dir / f"{shard_stem(80, 120)}_manifest.json.gz.tmp").write_bytes(b"partial")
    assert_stitch_refuses(run_dir, tmp_path, "incomplete")


# --- downstream compatibility -------------------------------------------------

def test_downstream_loader_and_heuristics_consume_stitched_output(
    sharding_video: Path, tmp_path: Path,
) -> None:
    stem = "1_full_poc"  # numeric prefix: apply_heuristic parses vid from it
    publish = extract_sharded(
        video_path=sharding_video,
        out_root=tmp_path,
        stem=stem,
        n_shards=4,
        extractor_spec="fake",
    )
    assert _raw_files_present(publish, stem)
    raw = _load_raw_clip(publish, stem)
    assert raw.kps.shape[0] == N_FRAMES

    court_info = {
        "H": np.eye(3, dtype=np.float64),
        "border_L": 0.0, "border_R": 96.0, "border_U": 0.0, "border_D": 64.0,
    }
    ctx = ClipContext(
        vid=1,
        all_court_info={1: court_info},
        res_df=pd.DataFrame({"width": [96], "height": [64]}, index=[1]),
    )
    for heuristic_fn in REGISTRY.values():
        output = heuristic_fn(raw, ctx)
        assert output.pos.shape[0] == N_FRAMES
        assert output.failed.shape == (N_FRAMES,)
