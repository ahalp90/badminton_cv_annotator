"""Tests for resumable ShuttleSet22 whole-video extraction."""

from __future__ import annotations

from fractions import Fraction
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from annotator.video_metadata import VideoMetadata
from dataset_builder.manifest import artifact_integrity
from dataset_builder.shuttle_evidence import shuttle_evidence_artifacts
from dataset_builder.vision import (
    PoseArrays,
    PoseExtraction,
    load_json_gz,
    save_json_gz,
    save_pose_arrays,
)
from shuttleset22.extraction import (
    EXTRACTION_RECEIPT_FILENAME,
    ExtractionSettings,
    PoseRuntimeIdentity,
    compress_csv,
    extract_source,
    resolve_pose_identity,
)
from shuttleset22 import extraction as extraction_module
from shuttleset22.manifest import AnnotationMatch, ResolvedSource, SourceEntry, SourceKind


def _source(tmp_path: Path) -> ResolvedSource:
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "08 India_Open_Final.mp4"
    video.write_bytes(b"video")
    metadata = VideoMetadata(video.resolve(), Fraction(30), 3, 100, 50)
    entry = SourceEntry(
        match_id=8,
        video="India_Open_Final",
        kind=SourceKind.DOWNLOAD,
        url="https://www.youtube.com/watch?v=7_O5r9CLOVw",
        youtube_id="7_O5r9CLOVw",
    )
    annotations = AnnotationMatch(
        match_id=8,
        video=entry.video,
        tournament="India Open",
        round_name="Finals",
        winner="Lakshya Sen",
        loser="Loh Kean Yew",
        stroke_count=1,
        max_annotated_frame=2,
        annotation_files=(tmp_path / "set1.csv",),
    )
    return ResolvedSource(
        entry=entry,
        annotations=annotations,
        metadata=metadata,
        integrity=artifact_integrity("source", video),
    )


def _settings(tmp_path: Path) -> ExtractionSettings:
    tracknet_dir = tmp_path / "tracknet"
    tracknet_dir.mkdir(parents=True)
    (tracknet_dir / "batch_predict.py").write_bytes(b"producer")
    tracknet = tmp_path / "tracknet.pt"
    inpaint = tmp_path / "inpaint.pt"
    tracknet.write_bytes(b"tracknet")
    inpaint.write_bytes(b"inpaint")
    interpreter = Path(sys.prefix) / "bin" / "python"
    return ExtractionSettings(
        source_commit="a" * 40,
        tracknet_dir=tracknet_dir,
        tracknet_model=tracknet,
        inpaint_model=inpaint,
        tracknet_python=interpreter,
        pose_interpreter=interpreter,
    )


def _pose_identity(_settings: ExtractionSettings) -> PoseRuntimeIdentity:
    return PoseRuntimeIdentity(models=(), runtime={"fixture": True})


class FakeProducers:
    def __init__(self) -> None:
        self.tracknet_calls = 0
        self.pose_calls = 0

    def tracknet(self, **kwargs) -> None:
        self.tracknet_calls += 1
        video = kwargs["video_paths"][0]
        output = kwargs["output_csv_dir"]
        csv_path = output / f"{video.stem}_ball.csv"
        csv_path.write_text(
            "Frame,X,Y,Visibility\n0,0,0,0\n1,25,25,1\n2,50,50,1\n",
            encoding="utf-8",
        )
        artifacts = shuttle_evidence_artifacts(
            output,
            input_video=video,
            stride=kwargs["tracknet_stride"],
        )
        save_json_gz(
            artifacts.inpaint_sidecar,
            {
                "schema": "inpaint_fill_mask/1",
                "index_space": "frame",
                "inpaint_status": "applied",
                "n_rows": 3,
                "eval_mode": "nonoverlap",
                "stride": 8,
                "th_h_px": 2.5,
                "tracknet_ckpt": kwargs["model_path"].name,
                "inpaintnet_ckpt": kwargs["inpaintnet_path"].name,
                "input_video": video.name,
                "extracted_utc": "2026-08-15T00:00:00Z",
                "inpaint_selected": [[1, 2]],
            },
        )

    def pose(self, **kwargs) -> PoseExtraction:
        self.pose_calls += 1
        frame_count = kwargs["metadata"].frame_count
        n_slots = 16
        arrays = PoseArrays(
            kps=np.full((frame_count, n_slots, 17, 2), np.nan, dtype=np.float32),
            bboxes=np.full((frame_count, n_slots, 4), np.nan, dtype=np.float32),
            scores=np.full((frame_count, n_slots), np.nan, dtype=np.float32),
            kp_scores=np.full((frame_count, n_slots, 17), np.nan, dtype=np.float32),
            ndet=np.zeros(frame_count, dtype=np.int16),
        )
        artifacts = save_pose_arrays(kwargs["output_dir"], arrays, frame_count)
        return PoseExtraction(arrays, artifacts, ("fake-pose",))


def test_compress_csv_round_trip_is_exact_and_removes_plain_csv(tmp_path: Path) -> None:
    source = tmp_path / "track.csv"
    expected = b"a,b\n1,2\n"
    source.write_bytes(expected)

    destination = compress_csv(source, tmp_path / "track.csv.gz")

    assert not source.exists()
    with gzip.open(destination, "rb") as handle:
        assert handle.read() == expected


def test_compress_csv_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"a,b\n1,2\n")
    second.write_bytes(first.read_bytes())

    first_gzip = compress_csv(first, tmp_path / "first.csv.gz")
    second_gzip = compress_csv(second, tmp_path / "second.csv.gz")

    assert first_gzip.read_bytes() == second_gzip.read_bytes()


def test_extract_source_publishes_validated_compressed_artifacts(tmp_path: Path) -> None:
    source = _source(tmp_path)
    producers = FakeProducers()

    result = extract_source(
        source,
        output_root=tmp_path / "outputs",
        settings=_settings(tmp_path),
        tracknet_producer=producers.tracknet,
        pose_producer=producers.pose,
        pose_identity_resolver=_pose_identity,
    )

    assert not result.reused
    assert result.receipt.name == EXTRACTION_RECEIPT_FILENAME
    assert result.receipt.is_file()
    assert result.artifacts["tracknet_csv"].name.endswith(".csv.gz")
    assert not any(result.output_dir.glob("*_ball.csv"))
    assert len(result.artifacts) == 10
    assert producers.tracknet_calls == producers.pose_calls == 1


def test_extract_source_reuses_only_a_valid_receipt(tmp_path: Path) -> None:
    source = _source(tmp_path)
    settings = _settings(tmp_path)
    producers = FakeProducers()
    extract_source(
        source,
        output_root=tmp_path / "outputs",
        settings=settings,
        tracknet_producer=producers.tracknet,
        pose_producer=producers.pose,
        pose_identity_resolver=_pose_identity,
    )

    reused = extract_source(
        source,
        output_root=tmp_path / "outputs",
        settings=settings,
        tracknet_producer=lambda **_kwargs: pytest.fail("TrackNet should not rerun"),
        pose_producer=lambda **_kwargs: pytest.fail("pose should not rerun"),
        pose_identity_resolver=_pose_identity,
    )

    assert reused.reused
    assert producers.tracknet_calls == producers.pose_calls == 1


def test_extract_source_reruns_after_artifact_corruption(tmp_path: Path) -> None:
    source = _source(tmp_path)
    settings = _settings(tmp_path)
    first = FakeProducers()
    result = extract_source(
        source,
        output_root=tmp_path / "outputs",
        settings=settings,
        tracknet_producer=first.tracknet,
        pose_producer=first.pose,
        pose_identity_resolver=_pose_identity,
    )
    result.artifacts["pose_scores"].write_bytes(b"corrupt")
    second = FakeProducers()

    repaired = extract_source(
        source,
        output_root=tmp_path / "outputs",
        settings=settings,
        tracknet_producer=second.tracknet,
        pose_producer=second.pose,
        pose_identity_resolver=_pose_identity,
    )

    assert not repaired.reused
    assert second.tracknet_calls == second.pose_calls == 1


def test_extract_source_reruns_after_semantically_corrupt_array(tmp_path: Path) -> None:
    source = _source(tmp_path)
    settings = _settings(tmp_path)
    first = FakeProducers()
    result = extract_source(
        source,
        output_root=tmp_path / "outputs",
        settings=settings,
        tracknet_producer=first.tracknet,
        pose_producer=first.pose,
        pose_identity_resolver=_pose_identity,
    )
    corrupt = result.artifacts["pose_scores"]
    corrupt.write_bytes(b"not an xz stream")
    payload = load_json_gz(result.receipt)
    records = payload["artifacts"]
    assert isinstance(records, list)
    replacement = artifact_integrity(
        "pose_scores",
        corrupt,
        relative_to=result.output_dir,
    ).to_dict()
    payload["artifacts"] = [
        replacement if isinstance(record, dict) and record.get("name") == "pose_scores" else record
        for record in records
    ]
    save_json_gz(result.receipt, payload)
    second = FakeProducers()

    repaired = extract_source(
        source,
        output_root=tmp_path / "outputs",
        settings=settings,
        tracknet_producer=second.tracknet,
        pose_producer=second.pose,
        pose_identity_resolver=_pose_identity,
    )

    assert not repaired.reused
    assert second.tracknet_calls == second.pose_calls == 1


def test_extract_source_failure_never_writes_receipt(tmp_path: Path) -> None:
    source = _source(tmp_path)

    with pytest.raises(RuntimeError, match="producer failed"):
        extract_source(
            source,
            output_root=tmp_path / "outputs",
            settings=_settings(tmp_path),
            tracknet_producer=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("producer failed")
            ),
            pose_producer=FakeProducers().pose,
            pose_identity_resolver=_pose_identity,
        )

    output_dir = tmp_path / "outputs" / "08 India_Open_Final"
    assert not (output_dir / EXTRACTION_RECEIPT_FILENAME).exists()


def test_extract_source_rejects_pose_slot_count_different_from_n_max(tmp_path: Path) -> None:
    source = _source(tmp_path)
    producers = FakeProducers()

    def two_slot_pose(**kwargs) -> PoseExtraction:
        frame_count = kwargs["metadata"].frame_count
        arrays = PoseArrays(
            kps=np.full((frame_count, 2, 17, 2), np.nan, dtype=np.float32),
            bboxes=np.full((frame_count, 2, 4), np.nan, dtype=np.float32),
            scores=np.full((frame_count, 2), np.nan, dtype=np.float32),
            kp_scores=np.full((frame_count, 2, 17), np.nan, dtype=np.float32),
            ndet=np.zeros(frame_count, dtype=np.int16),
        )
        artifacts = save_pose_arrays(kwargs["output_dir"], arrays, frame_count)
        return PoseExtraction(arrays, artifacts, ("two-slot-pose",))

    with pytest.raises(ValueError, match="pose slot count 2"):
        extract_source(
            source,
            output_root=tmp_path / "outputs",
            settings=_settings(tmp_path),
            tracknet_producer=producers.tracknet,
            pose_producer=two_slot_pose,
            pose_identity_resolver=_pose_identity,
        )


def test_extraction_receipt_reuses_after_storage_root_changes(tmp_path: Path) -> None:
    source = _source(tmp_path / "original")
    original_settings = _settings(tmp_path / "original-settings")
    producers = FakeProducers()
    result = extract_source(
        source,
        output_root=tmp_path / "original-outputs",
        settings=original_settings,
        tracknet_producer=producers.tracknet,
        pose_producer=producers.pose,
        pose_identity_resolver=_pose_identity,
    )
    moved_source_path = tmp_path / "moved-sources" / source.metadata.source_path.name
    moved_source_path.parent.mkdir()
    shutil.copy2(source.metadata.source_path, moved_source_path)
    moved_outputs = tmp_path / "moved-outputs"
    shutil.copytree(result.output_dir.parent, moved_outputs)
    moved_source = ResolvedSource(
        entry=source.entry,
        annotations=source.annotations,
        metadata=VideoMetadata(moved_source_path.resolve(), Fraction(30), 3, 100, 50),
        integrity=artifact_integrity(source.integrity.name, moved_source_path),
    )
    moved_settings = _settings(tmp_path / "moved-settings")

    reused = extract_source(
        moved_source,
        output_root=moved_outputs,
        settings=moved_settings,
        tracknet_producer=lambda **_kwargs: pytest.fail("TrackNet should not rerun"),
        pose_producer=lambda **_kwargs: pytest.fail("pose should not rerun"),
        pose_identity_resolver=_pose_identity,
    )

    assert reused.reused


def test_input_change_during_extraction_prevents_receipt(tmp_path: Path) -> None:
    source = _source(tmp_path)
    settings = _settings(tmp_path)
    producers = FakeProducers()

    def mutate_model_after_pose(**kwargs) -> PoseExtraction:
        result = producers.pose(**kwargs)
        settings.tracknet_model.write_bytes(b"changed")
        return result

    with pytest.raises(ValueError, match="inputs changed"):
        extract_source(
            source,
            output_root=tmp_path / "outputs",
            settings=settings,
            tracknet_producer=producers.tracknet,
            pose_producer=mutate_model_after_pose,
            pose_identity_resolver=_pose_identity,
        )

    receipt = tmp_path / "outputs" / "08 India_Open_Final" / EXTRACTION_RECEIPT_FILENAME
    assert not receipt.exists()


def test_changed_source_never_reuses_extraction_receipt(tmp_path: Path) -> None:
    source = _source(tmp_path)
    settings = _settings(tmp_path)
    producers = FakeProducers()
    result = extract_source(
        source,
        output_root=tmp_path / "outputs",
        settings=settings,
        tracknet_producer=producers.tracknet,
        pose_producer=producers.pose,
        pose_identity_resolver=_pose_identity,
    )
    source.metadata.source_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="source integrity differs"):
        extract_source(
            source,
            output_root=tmp_path / "outputs",
            settings=settings,
            tracknet_producer=lambda **_kwargs: pytest.fail("TrackNet should not rerun"),
            pose_producer=lambda **_kwargs: pytest.fail("pose should not rerun"),
            pose_identity_resolver=_pose_identity,
        )

    assert result.receipt.is_file()


def test_pose_identity_verifies_models_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    detector = tmp_path / "rtmdet.onnx"
    pose = tmp_path / "rtmpose.onnx"
    detector.write_bytes(b"detector")
    pose.write_bytes(b"pose")
    payload = {
        "models": {"rtmdet": str(detector), "rtmpose": str(pose)},
        "packages": {
            "rtmlib": "0.0.15",
            "onnxruntime": None,
            "onnxruntime-gpu": "1.27.0",
            "numpy": "2.4.6",
            "opencv-python": "4.13.0.92",
        },
        "onnxruntime_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }

    def run(command, **_kwargs):
        stdout = "Python 3.12" if "--version" in command else (
            "SHUTTLESET22_POSE_IDENTITY=" + json.dumps(payload)
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(extraction_module.subprocess, "run", run)
    expected = extraction_module._POSE_MODEL_SHA256
    monkeypatch.setattr(
        extraction_module,
        "_file_digest",
        lambda path: expected["rtmdet" if path == detector else "rtmpose"],
    )

    identity = resolve_pose_identity(settings)

    assert [model.name for model in identity.models] == ["rtmdet", "rtmpose"]
    assert identity.runtime["model_sha256"] == expected
    assert identity.runtime["onnxruntime_providers"] == payload["onnxruntime_providers"]


def test_pose_identity_rejects_model_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    detector = tmp_path / "rtmdet.onnx"
    pose = tmp_path / "rtmpose.onnx"
    detector.write_bytes(b"detector")
    pose.write_bytes(b"pose")
    payload = {
        "models": {"rtmdet": str(detector), "rtmpose": str(pose)},
        "packages": {"rtmlib": "0.0.15", "onnxruntime-gpu": "1.27.0"},
        "onnxruntime_providers": ["CUDAExecutionProvider"],
    }

    def run(command, **_kwargs):
        stdout = "Python 3.12" if "--version" in command else (
            "SHUTTLESET22_POSE_IDENTITY=" + json.dumps(payload)
        )
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(extraction_module.subprocess, "run", run)
    monkeypatch.setattr(extraction_module, "_file_digest", lambda _path: "0" * 64)

    with pytest.raises(ValueError, match="SHA-256"):
        resolve_pose_identity(settings)
