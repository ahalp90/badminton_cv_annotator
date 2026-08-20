"""Focused contracts for deterministic BRIC rally clip bounds."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from bric.preprocessing import extract_shuttle, slice_rallies
from scripts import build_shots_master

add_rally_clip_bounds = build_shots_master.add_rally_clip_bounds


@pytest.mark.parametrize(
    ("fps", "expected_start", "expected_end"),
    [(25, 50, 575), (30, 40, 590)],
)
def test_rally_clip_bounds_scale_with_fps_and_use_ball_round(
    fps: int,
    expected_start: int,
    expected_end: int,
) -> None:
    contacts = pd.DataFrame({
        "set": [1, 1, 1],
        "rally": [7, 7, 7],
        "ball_round": [2, 3, 1],
        "frame_num": [600, 500, 100],
    })
    retained_shots = contacts.loc[contacts["ball_round"] != 1].copy()

    result = add_rally_clip_bounds(retained_shots, contacts, fps=fps, frame_count=1_000)

    assert result["clip_start_frame"].tolist() == [expected_start, expected_start]
    assert result["clip_end_frame"].tolist() == [expected_end, expected_end]


def test_rally_clip_bounds_clamp_to_source_video() -> None:
    contacts = pd.DataFrame({
        "set": [1, 1],
        "rally": [3, 3],
        "ball_round": [1, 2],
        "frame_num": [20, 980],
    })

    result = add_rally_clip_bounds(contacts, contacts, fps=25, frame_count=1_000)

    assert result[["clip_start_frame", "clip_end_frame"]].drop_duplicates().to_dict("records") == [
        {"clip_start_frame": 0, "clip_end_frame": 1_000},
    ]


def test_rally_clip_bounds_reject_duplicate_contact_keys() -> None:
    contacts = pd.DataFrame({
        "set": [1, 1],
        "rally": [3, 3],
        "ball_round": [1, 1],
        "frame_num": [100, 101],
    })

    with pytest.raises(ValueError, match="duplicate rally contact keys"):
        add_rally_clip_bounds(contacts.iloc[:1], contacts, fps=25, frame_count=1_000)


def test_slicer_and_extractor_consume_identical_stored_bounds() -> None:
    strokes = pd.DataFrame({
        "set_id": ["set1", "set1", "set1"],
        "rally": [1, 1, 2],
        "clip_start_frame": [10, 10, 300],
        "clip_end_frame": [200, 200, 500],
    })

    sliced = slice_rallies.compute_rally_bounds(strokes, frame_count=600)
    extracted = extract_shuttle.compute_rally_bounds(strokes, frame_count=600)

    assert {(set_id, rally): (start, end) for set_id, rally, start, end in sliced} == extracted


def test_consumers_reject_inconsistent_or_out_of_source_bounds() -> None:
    inconsistent = pd.DataFrame({
        "set_id": ["set1", "set1"],
        "rally": [1, 1],
        "clip_start_frame": [10, 11],
        "clip_end_frame": [200, 200],
    })
    outside = pd.DataFrame({
        "set_id": ["set1"],
        "rally": [1],
        "clip_start_frame": [10],
        "clip_end_frame": [601],
    })

    with pytest.raises(ValueError, match="inconsistent stored clip bounds"):
        slice_rallies.compute_rally_bounds(inconsistent, frame_count=600)
    with pytest.raises(ValueError, match="invalid for source frame count"):
        extract_shuttle.compute_rally_bounds(outside, frame_count=600)


def test_existing_clip_requires_matching_bounds_sidecar(tmp_path: Path) -> None:
    source_path = tmp_path / "1 match.mp4"
    clip_path = tmp_path / "1_set1_1.mp4"
    source_path.write_bytes(b"source")
    clip_path.write_bytes(b"clip")
    expected = slice_rallies.expected_bounds_metadata(
        source_path,
        source_frame_count=1_000,
        start_frame=50,
        end_frame=575,
        fps=25,
    )

    assert not slice_rallies.clip_bounds_are_current(clip_path, expected)
    slice_rallies.write_bounds_metadata(clip_path, expected)
    assert slice_rallies.clip_bounds_are_current(clip_path, expected)
    assert extract_shuttle.clip_bounds_are_current(
        clip_path,
        source_path,
        source_frame_count=1_000,
        bounds=(50, 575),
        fps=25,
    )

    changed = {**expected, "clip_start_frame": 49}
    assert not slice_rallies.clip_bounds_are_current(clip_path, changed)


def test_existing_shuttle_cache_requires_current_bounds_metadata(tmp_path: Path) -> None:
    cache_path = tmp_path / "1.npz"
    source_path = tmp_path / "1 match.mp4"
    bounds = {("set1", 1): (50, 575), ("set1", 2): (700, 900)}
    np.savez_compressed(cache_path, frame=np.arange(1_000, dtype=np.int32))

    assert not extract_shuttle.shuttle_cache_is_current(
        cache_path, bounds, source_path, source_frame_count=1_000,
    )

    rally_keys, starts, ends = extract_shuttle.rally_bound_arrays(bounds)
    np.savez_compressed(
        cache_path,
        bounds_metadata_version=np.int32(extract_shuttle.BOUNDS_METADATA_VERSION),
        source_video=np.asarray(source_path.name),
        source_frame_count=np.int64(1_000),
        rally_keys=rally_keys,
        rally_clip_start_frames=starts,
        rally_clip_end_frames=ends,
    )
    assert not extract_shuttle.shuttle_cache_is_current(
        cache_path, bounds, source_path, source_frame_count=1_000,
    )

    np.savez_compressed(
        cache_path,
        frame=np.arange(1_000, dtype=np.int32),
        x=np.zeros(1_000, dtype=np.float32),
        y=np.zeros(1_000, dtype=np.float32),
        visibility=np.zeros(1_000, dtype=np.int32),
        bounds_metadata_version=np.int32(extract_shuttle.BOUNDS_METADATA_VERSION),
        source_video=np.asarray(source_path.name),
        source_frame_count=np.int64(1_000),
        rally_keys=rally_keys,
        rally_clip_start_frames=starts,
        rally_clip_end_frames=ends,
    )
    assert extract_shuttle.shuttle_cache_is_current(
        cache_path, bounds, source_path, source_frame_count=1_000,
    )
    assert not extract_shuttle.shuttle_cache_is_current(
        cache_path, {("set1", 1): (49, 575)}, source_path, source_frame_count=1_000,
    )


@pytest.mark.parametrize("content", [b"", b"PK\x03\x04truncated"])
def test_corrupt_shuttle_cache_is_stale(tmp_path: Path, content: bytes) -> None:
    cache_path = tmp_path / "1.npz"
    cache_path.write_bytes(content)

    assert not extract_shuttle.shuttle_cache_is_current(
        cache_path,
        {("set1", 1): (50, 575)},
        tmp_path / "1 match.mp4",
        source_frame_count=1_000,
    )


@pytest.mark.parametrize("module", [build_shots_master, slice_rallies, extract_shuttle])
def test_source_lookup_accepts_current_and_legacy_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    monkeypatch.setattr(module, "RAW_VIDEO_DIR", tmp_path)
    source_path = tmp_path / "7.mp4"
    source_path.write_bytes(b"video")

    assert module.find_source_video(7) == source_path


@pytest.mark.parametrize("module", [build_shots_master, slice_rallies, extract_shuttle])
def test_source_lookup_rejects_ambiguous_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    monkeypatch.setattr(module, "RAW_VIDEO_DIR", tmp_path)
    (tmp_path / "7.mp4").write_bytes(b"video")
    (tmp_path / "7 match.mp4").write_bytes(b"video")

    with pytest.raises(RuntimeError, match="multiple canonical raw videos"):
        module.find_source_video(7)
