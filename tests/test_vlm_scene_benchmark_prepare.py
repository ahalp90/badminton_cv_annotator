"""Frame-preserving preparation tests for the VLM benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from annotator.vlm_scene_benchmark.prepare import (
    PreparedShardManifest,
    VideoFileSpec,
    prepare_shard,
    read_manifest,
    resolve_model_video,
    write_manifest,
)
from annotator.vlm_scene_benchmark.contracts import ShardSpec


SHA256 = "a" * 64


def _manifest() -> PreparedShardManifest:
    source = VideoFileSpec("source.mp4", "b" * 64, 25.0, 100, 640, 360)
    reference = VideoFileSpec("reference.mp4", "c" * 64, 25.0, 50, 512, 288)
    model = VideoFileSpec("model.mp4", SHA256, 1.0, 2, 512, 288)
    shard = ShardSpec(
        "sset_15", source.file_name, source.sha256, model.file_name, model.sha256, 25.0, 100, 10, 60
    )
    return PreparedShardManifest(
        shard=shard,
        original_source=source,
        reference_video=reference,
        model_video=model,
        sampled_source_frames=(10, 35),
        cut_frames=(20, 40),
        content_threshold=27.0,
        min_scene_len=15,
        ffmpeg_version="ffmpeg version test",
    )


def test_prepared_manifest_round_trip_and_duplicate_rejection(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, _manifest())

    assert read_manifest(path) == _manifest()

    encoded = path.read_text(encoding="utf-8")
    path.write_text(encoded.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_manifest(path)


def test_manifest_rejects_model_map_count_mismatch() -> None:
    value = _manifest().to_json()
    value["sampled_source_frames"] = [10]

    with pytest.raises(ValueError, match="frame count differs"):
        PreparedShardManifest.from_json(value)


def test_prepare_shard_preserves_frames_and_source_mapping(
    validation_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.prepare.detect_cuts",
        lambda _path, _frames, _threshold, _minimum: np.array([2, 5]),
    )
    output_dir = tmp_path / "prepared"

    manifest_path = prepare_shard(
        validation_video,
        output_dir,
        video_id="tiny",
        start_frame=1,
        end_frame=7,
        sample_fps=25.0,
        width=64,
        height=32,
    )
    manifest = read_manifest(manifest_path)

    assert manifest.shard.start_frame == 1
    assert manifest.shard.end_frame == 7
    assert manifest.reference_video.frame_count == 6
    assert manifest.model_video.frame_count == 6
    assert manifest.shard.source_file == validation_video.name
    assert manifest.shard.source_sha256 == manifest.original_source.sha256
    assert manifest.shard.prepared_input_file == manifest.model_video.file_name
    assert manifest.shard.prepared_input_sha256 == manifest.model_video.sha256
    assert manifest.sampled_source_frames == (1, 2, 3, 4, 5, 6)
    assert manifest.cut_frames == (3, 6)
    assert resolve_model_video(manifest_path, manifest).is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_shard(
            validation_video,
            output_dir,
            video_id="tiny",
            start_frame=1,
            end_frame=7,
            sample_fps=25.0,
            width=64,
            height=32,
        )


def test_manifest_reader_requires_arrays(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    value = _manifest().to_json()
    value["cut_frames"] = "20,40"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="must be JSON arrays"):
        read_manifest(path)


def test_prepare_shard_cleans_published_videos_if_manifest_write_fails(
    validation_video: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.prepare.detect_cuts",
        lambda _path, _frames, _threshold, _minimum: np.array([], dtype=np.int64),
    )
    def fail_write(_path: Path, _manifest_value: PreparedShardManifest) -> None:
        raise OSError("injected failure")

    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.prepare.write_manifest",
        fail_write,
    )
    output_dir = tmp_path / "failed"

    with pytest.raises(OSError, match="injected failure"):
        prepare_shard(
            validation_video,
            output_dir,
            video_id="tiny",
            start_frame=1,
            end_frame=7,
            sample_fps=25.0,
            width=64,
            height=32,
        )

    assert list(output_dir.iterdir()) == []
