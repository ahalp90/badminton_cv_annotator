"""Whole-video vision, annotation, and compressed artefact contracts."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import annotator.court_evidence as court_evidence_module
import annotator.run_video as run_video_module
from annotator.config import BaseAnnotatorConfig
from annotator.court_evidence import (
    DETECTOR_RESULT_SCHEMA,
    CourtEvidenceResult,
    CourtInputs,
    CourtSceneRecord,
    DetectorScene,
    NoAcceptedCourtError,
    SceneStatus,
    build_court_detector_evidence,
)
from annotator.point_winner import (
    GeometricVerdictRow,
    Half,
    Landing,
    Verdict,
    VerdictRow,
    VerdictSource,
)
from annotator.run_video import AnnotatorResult, RunCapture, run_video
from annotator.types import ContactCandidate, DeadMaskMode
from annotator.video_metadata import VideoMetadata
from dataset_builder import vision
from dataset_builder.shuttle_quality import ShuttleQualitySummary, summarize_shuttle_quality
from scraper.commentary_pairing import pair_video


def _metadata(tmp_path: Path, *, frame_count: int = 6, fps: int = 25) -> VideoMetadata:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    return VideoMetadata(
        source_path=source.resolve(),
        fps=Fraction(fps),
        frame_count=frame_count,
        width=100,
        height=50,
    )


def _tracknet_csv(path: Path, frames: list[float | int]) -> Path:
    pd.DataFrame({
        "Frame": frames,
        "X": [10.0 + index for index in range(len(frames))],
        "Y": [20.0 + index for index in range(len(frames))],
        "Visibility": [1] * len(frames),
    }).to_csv(path, index=False)
    return path


def _pose_arrays(frame_count: int, n_slots: int = 2) -> vision.PoseArrays:
    return vision.PoseArrays(
        kps=np.full((frame_count, n_slots, 17, 2), np.nan, dtype=np.float32),
        bboxes=np.full((frame_count, n_slots, 4), np.nan, dtype=np.float32),
        scores=np.full((frame_count, n_slots), np.nan, dtype=np.float32),
        kp_scores=np.full((frame_count, n_slots, 17), np.nan, dtype=np.float32),
        ndet=np.zeros(frame_count, dtype=np.int8),
    )


def _write_raw_pose(output_dir: Path, arrays: vision.PoseArrays) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, suffix in {
        "kps": "_raw_kps.npy",
        "bboxes": "_raw_bboxes.npy",
        "scores": "_raw_scores.npy",
        "kp_scores": "_raw_kp_scores.npy",
        "ndet": "_raw_ndet.npy",
    }.items():
        np.save(output_dir / f"pose{suffix}", getattr(arrays, name), allow_pickle=False)


def _court_vision(video_id: str, frame_count: int) -> vision.CourtVision:
    width, height = 100.0, 50.0
    homography = np.array([
        [1.0 / width, 0.0, 0.0],
        [0.0, 1.0 / height, 0.0],
        [0.0, 0.0, 1.0],
    ])
    court_info: dict[str, object] = {
        "H": homography,
        "border_L": 0.0,
        "border_R": 1.0,
        "border_U": 0.0,
        "border_D": 1.0,
    }
    homography_rows = pd.DataFrame([{
        "video_id": video_id,
        "start_frame": 0,
        "end_frame": frame_count,
        "upleft_x": 0.0,
        "upleft_y": 0.0,
        "upright_x": width,
        "upright_y": 0.0,
        "downleft_x": 0.0,
        "downleft_y": height,
        "downright_x": width,
        "downright_y": height,
    }])
    inputs = CourtInputs(
        court_info=court_info,
        gate_court_info={video_id: court_info},
        net_band=(24.0, 26.0),
        resolution=(width, height),
        gate_resolution_table=pd.DataFrame(
            {"width": [width], "height": [height]},
            index=pd.Index([video_id], dtype=object),
        ),
        homography_rows=homography_rows,
        landing_error_band_m=0.1,
        active_corners_refpx=np.array(
            [[0.0, 0.0], [1280.0, 0.0], [1280.0, 720.0], [0.0, 720.0]],
        ),
    )
    native_corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
    )
    record = CourtSceneRecord(
        video_id=video_id,
        case_id=f"case-{video_id}",
        parent="fixture-static",
        scene_index=0,
        start_frame=0,
        end_frame=frame_count,
        status=SceneStatus.COURT,
        analysed_frame=None,
        corners_native_px=native_corners,
        no_court_reason=None,
        reused_from=None,
        error=None,
        exactly_two_count=0,
        exactly_two_fraction=0.0,
        scene_valid=False,
    )
    evidence = CourtEvidenceResult(
        inputs=inputs,
        scene_records=(record,),
        keep_vote=np.zeros(frame_count, dtype=bool),
        court_present=np.ones(frame_count, dtype=bool),
    )
    return vision.CourtVision(((0, frame_count),), evidence)


def _assert_structured_equal(actual: object, expected: object) -> None:
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(expected) and not isinstance(expected, type):
        assert type(actual) is type(expected)
        for field in fields(expected):
            _assert_structured_equal(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_structured_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _direct_annotation(
    video_id: str,
    metadata: VideoMetadata,
    track: np.ndarray,
    pose: vision.PoseArrays,
    court: vision.CourtVision,
    guard_codes: np.ndarray,
) -> tuple[AnnotatorResult, RunCapture]:
    inputs = court.evidence.inputs
    assert inputs is not None
    capture = RunCapture()
    result = run_video(
        track,
        pose.bboxes,
        pose.scores,
        pose.kps,
        pose.ndet,
        fps=float(metadata.fps),
        landing_options=run_video_module.point_winner.SHIPPED_LANDING_FILTER_OPTIONS,
        net_band=inputs.net_band,
        resolution=inputs.resolution,
        video_id=video_id,
        court_info=inputs.court_info,
        homo_df=None,
        gate_court_info=inputs.gate_court_info,
        gate_resolution_table=inputs.gate_resolution_table,
        court_present=court.evidence.court_present,
        homography_rows=inputs.homography_rows,
        cut_frames=[],
        keep_vote=court.evidence.keep_vote,
        inpaint_codes=guard_codes,
        court_invalid_is_excluded=True,
        landing_error_band_m=inputs.landing_error_band_m,
        capture=capture,
    )
    return result, capture


def _quality(
    track: np.ndarray,
    *,
    fill_mask: np.ndarray | None = None,
    guard_codes: np.ndarray | None = None,
) -> ShuttleQualitySummary:
    frame_count = len(track)
    fill = np.zeros(frame_count, dtype=bool) if fill_mask is None else fill_mask
    codes = np.zeros(frame_count, dtype=np.uint8) if guard_codes is None else guard_codes
    return summarize_shuttle_quality(track, fill, codes, frozenset({1, 2, 3}))


def test_whole_video_tracknet_conversion_reindexes_and_preserves_string_id(
    tmp_path: Path,
) -> None:
    metadata = _metadata(tmp_path, frame_count=3)
    csv_path = tmp_path / "track.csv"
    pd.DataFrame({
        "Frame": [2, 0, 1],
        "X": [30.0, 10.0, 20.0],
        "Y": [15.0, 5.0, 10.0],
        "Visibility": [1, 0, 1],
    }).to_csv(csv_path, index=False)
    output = tmp_path / "vision" / vision.TRACK_FILENAME

    shuttle = vision.convert_tracknet_csv_stage(
        csv_path,
        video_id="match-alpha/001",
        metadata=metadata,
        output_path=output,
    )

    assert shuttle.video_id == "match-alpha/001"
    expected = np.array([[0.1, 0.1, 0.0], [0.2, 0.2, 1.0], [0.3, 0.3, 1.0]])
    np.testing.assert_allclose(shuttle.track, expected)
    np.testing.assert_array_equal(vision.load_npy_xz(output), shuttle.track)


@pytest.mark.parametrize(
    ("frames", "reason"),
    [
        ([0, 1, 1], "duplicates"),
        ([0, 2], "gaps"),
        ([0, 1.5, 2], "integers"),
        ([0, 1, 3], "must be in"),
    ],
)
def test_whole_video_tracknet_frame_contract_raises_original_error(
    tmp_path: Path,
    frames: list[float | int],
    reason: str,
) -> None:
    metadata = _metadata(tmp_path, frame_count=3)
    csv_path = _tracknet_csv(tmp_path / "bad.csv", frames)
    output = tmp_path / vision.TRACK_FILENAME

    with pytest.raises(ValueError, match=reason):
        vision.convert_tracknet_csv_stage(
            csv_path,
            video_id="not-a-number",
            metadata=metadata,
            output_path=output,
        )

    assert not output.exists()


def test_whole_video_tracknet_rejects_non_numeric_values(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path, frame_count=2)
    csv_path = tmp_path / "bad-values.csv"
    pd.DataFrame({
        "Frame": [0, 1],
        "X": [1.0, "not-a-number"],
        "Y": [1.0, 2.0],
        "Visibility": [1, 1],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="finite values"):
        vision.convert_tracknet_csv_stage(
            csv_path,
            video_id="video",
            metadata=metadata,
            output_path=tmp_path / vision.TRACK_FILENAME,
        )


@pytest.mark.parametrize(("x", "y"), [(-1.0, 25.0), (101.0, 25.0), (50.0, -1.0), (50.0, 51.0)])
def test_whole_video_tracknet_rejects_visible_coordinates_outside_video(
    tmp_path: Path,
    x: float,
    y: float,
) -> None:
    metadata = _metadata(tmp_path, frame_count=1)
    csv_path = tmp_path / "bad-coordinate.csv"
    pd.DataFrame({"Frame": [0], "X": [x], "Y": [y], "Visibility": [1]}).to_csv(
        csv_path,
        index=False,
    )

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        vision.convert_tracknet_csv_stage(
            csv_path,
            video_id="video",
            metadata=metadata,
            output_path=tmp_path / vision.TRACK_FILENAME,
        )


def test_whole_video_tracknet_accepts_boundaries_and_ignores_invisible_coordinates(
    tmp_path: Path,
) -> None:
    metadata = _metadata(tmp_path, frame_count=3)
    csv_path = tmp_path / "boundary-coordinates.csv"
    pd.DataFrame({
        "Frame": [0, 1, 2],
        "X": [0.0, 100.0, -500.0],
        "Y": [50.0, 0.0, 500.0],
        "Visibility": [1, 1, 0],
    }).to_csv(csv_path, index=False)

    shuttle = vision.convert_tracknet_csv_stage(
        csv_path,
        video_id="video",
        metadata=metadata,
        output_path=tmp_path / vision.TRACK_FILENAME,
    )

    np.testing.assert_array_equal(
        shuttle.track,
        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [-5.0, 10.0, 0.0]]),
    )


def test_scene_rows_keep_numeric_looking_string_video_ids() -> None:
    rows = court_evidence_module.build_scene_rows(
        "0012",
        [(0, 10)],
        [np.array([[0.0, 0.0], [1280.0, 0.0], [1280.0, 720.0], [0.0, 720.0]])],
        (100.0, 50.0),
    )

    assert rows.loc[0, "video_id"] == "0012"
    assert isinstance(rows.loc[0, "video_id"], str)


def test_rtmlib_pose_uses_configured_interpreter_and_publishes_compressed_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(tmp_path, frame_count=4)
    expected = _pose_arrays(metadata.frame_count)
    interpreter = tmp_path / "rtmlib-python"
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        raw_dir = Path(command[command.index("--output-dir") + 1])
        _write_raw_pose(raw_dir, expected)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(vision.subprocess, "run", fake_run)
    extraction = vision.extract_rtmlib_pose_stage(
        metadata=metadata,
        output_dir=tmp_path / "pose",
        interpreter=interpreter,
        device="cpu",
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == str(interpreter.resolve())
    assert command[1:4] == ["-m", "dataset_builder.vision", "_extract-rtmlib-pose"]
    assert command[command.index("--device") + 1] == "cpu"
    loaded = vision.load_pose_arrays(tmp_path / "pose", metadata.frame_count)
    for field_name in ("kps", "bboxes", "scores", "kp_scores", "ndet"):
        np.testing.assert_array_equal(
            getattr(loaded, field_name),
            getattr(expected, field_name),
        )
    for path in extraction.artifacts.as_mapping().values():
        assert path.name.endswith(".npy.xz")


def test_compressed_numpy_publication_streams_to_atomic_xz_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.arange(48, dtype=np.float32).reshape(6, 8)[:, ::2]
    output = tmp_path / "values.npy.xz"
    real_open = vision.lzma.open
    observed: list[tuple[str, int | None, int | None]] = []

    def tracked_open(
        path: Path,
        mode: str,
        *,
        format: int | None = None,
        preset: int | None = None,
    ) -> object:
        observed.append((mode, format, preset))
        return real_open(path, mode, format=format, preset=preset)

    def reject_buffered_publication(_path: Path, _payload: bytes) -> None:
        raise AssertionError("compressed NumPy output must not use a whole-payload buffer")

    monkeypatch.setattr(vision.lzma, "open", tracked_open)
    monkeypatch.setattr(vision, "_atomic_write_bytes", reject_buffered_publication)

    assert vision.save_npy_xz(output, values) == output
    assert observed == [("wb", vision.lzma.FORMAT_XZ, 9)]
    np.testing.assert_array_equal(vision.load_npy_xz(output), values)


@pytest.mark.parametrize(
    "failure",
    ["shape", "dtype", "frame_count", "kps_padding", "bboxes_padding", "kp_scores_padding"],
)
def test_pose_contract_failures_raise_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    metadata = _metadata(tmp_path, frame_count=4)
    valid = _pose_arrays(metadata.frame_count)
    if failure == "shape":
        arrays = vision.PoseArrays(
            valid.kps,
            valid.bboxes[:, :, :3],
            valid.scores,
            valid.kp_scores,
            valid.ndet,
        )
    elif failure == "dtype":
        arrays = vision.PoseArrays(
            np.zeros_like(valid.kps, dtype=np.int16),
            valid.bboxes,
            valid.scores,
            valid.kp_scores,
            valid.ndet,
        )
    elif failure == "frame_count":
        arrays = _pose_arrays(metadata.frame_count - 1)
    elif failure == "kps_padding":
        arrays = vision.PoseArrays(
            np.zeros_like(valid.kps), valid.bboxes, valid.scores, valid.kp_scores, valid.ndet,
        )
    elif failure == "bboxes_padding":
        arrays = vision.PoseArrays(
            valid.kps, np.zeros_like(valid.bboxes), valid.scores, valid.kp_scores, valid.ndet,
        )
    else:
        arrays = vision.PoseArrays(
            valid.kps, valid.bboxes, valid.scores, np.zeros_like(valid.kp_scores), valid.ndet,
        )
    interpreter = tmp_path / "rtmlib-python"
    interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter.chmod(0o755)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raw_dir = Path(command[command.index("--output-dir") + 1])
        _write_raw_pose(raw_dir, arrays)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(vision.subprocess, "run", fake_run)
    output_dir = tmp_path / "pose"
    with pytest.raises(ValueError):
        vision.extract_rtmlib_pose_stage(
            metadata=metadata,
            output_dir=output_dir,
            interpreter=interpreter,
        )

    for path in vision.pose_artifact_paths(output_dir).as_mapping().values():
        assert not path.exists()


FULL_FRAME_PX = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]  # TL, TR, BR, BL at 100x50


def _two_player_pose(frame_count: int) -> vision.PoseArrays:
    """Two people standing inside a full-frame 100x50 court in every frame."""
    pose = _pose_arrays(frame_count)
    pose.bboxes[:, 0] = (20.0, 20.0, 30.0, 30.0)
    pose.bboxes[:, 1] = (60.0, 30.0, 70.0, 40.0)
    pose.kps[:, :2] = 25.0
    pose.kp_scores[:, :2] = 0.9
    pose.scores[:, :2] = 0.9
    pose.ndet[:] = 2
    return pose


def _detector_row(start: int, end: int, status: str, **fields: object) -> dict[str, object]:
    row: dict[str, object] = {
        "view_id": f"0012_scene_{start}", "start_frame": start, "end_frame": end,
        "frame_index": (start + end) // 2, "status": status,
        "corners_native_px": FULL_FRAME_PX if status == "court" else None,
    }
    row.update(fields)
    return row


def _detector_result(frame_count: int, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": DETECTOR_RESULT_SCHEMA, "video_id": "0012", "frame_count": frame_count,
        "native_size": [100, 50], "scenes": rows,
    }


def _settings(python: Path = Path(sys.executable)) -> vision.CourtDetectorSettings:
    return vision.CourtDetectorSettings(
        python=python,
        deeplsd_source=Path("deeplsd"),
        deeplsd_weights=Path("deeplsd/weights/deeplsd_md.tar"),
        device="cuda",
        template_device="cpu",
        reuse_courts=False,
    )


def test_court_detector_command_launches_the_program_with_saved_poses() -> None:
    command = vision.court_detector_command(
        _settings(Path("/venv/bin/python")),
        video_path=Path("video.mp4"), pose_dir=Path("pose"), output_path=Path("out.json.gz"),
    )
    assert command == [
        "/venv/bin/python", "-m", "court_detector.run_video",
        "--video", "video.mp4", "--people", "pose",
        "--deeplsd-source", "deeplsd", "--deeplsd-weights", "deeplsd/weights/deeplsd_md.tar",
        "--device", "cuda", "--template-device", "cpu", "--court-mode", "video-robust",
        "--output", "out.json.gz", "--pyscenedetect",
    ]
    reused = vision.court_detector_command(
        vision.CourtDetectorSettings(
            Path("python"), Path("s"), Path("w"), "cpu", "cuda", reuse_courts=True,
        ),
        video_path=Path("v"), pose_dir=Path("p"), output_path=Path("o"), scenes_path=Path("scenes.json.gz"),
    )
    assert reused[-3:] == ["--scenes", "scenes.json.gz", "--reuse-courts"]
    assert "--pyscenedetect" not in reused
    assert reused[reused.index("--template-device") + 1] == "cuda"
    per_scene = vision.court_detector_command(
        replace(_settings(Path("python")), court_mode="scene-robust"),
        video_path=Path("v"), pose_dir=Path("p"), output_path=Path("o"),
    )
    assert per_scene[per_scene.index("--court-mode") + 1] == "scene-robust"


def test_run_court_detector_runs_a_separate_program_and_loads_its_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    result = _detector_result(4, [_detector_row(0, 4, "court")])

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        vision.save_json_gz(Path(command[command.index("--output") + 1]), result)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(vision.subprocess, "run", fake_run)
    loaded = vision.run_court_detector(
        _settings(), video_path=tmp_path / "video.mp4", pose_dir=tmp_path / "pose", work_dir=tmp_path,
    )
    assert loaded == result
    (command, kwargs), = calls
    assert command[1:3] == ["-m", "court_detector.run_video"]
    assert Path(command[0]).is_absolute()
    assert command[command.index("--output") + 1] == str(tmp_path / vision.COURT_DETECTOR_RESULT_FILENAME)
    env = kwargs["env"]
    assert isinstance(env, dict)
    source_root = Path(vision.__file__).resolve().parents[1]
    assert str(source_root) in env["PYTHONPATH"].split(":")

    def failing_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, "", "DeepLSD weights are missing")

    monkeypatch.setattr(vision.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match="exited with status 2: DeepLSD weights are missing"):
        vision.run_court_detector(
            _settings(), video_path=tmp_path / "video.mp4", pose_dir=tmp_path / "pose", work_dir=tmp_path,
        )


def test_detected_court_stage_validates_the_detector_result_and_persists_scene_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(tmp_path, frame_count=6)
    pose = _two_player_pose(metadata.frame_count)
    seen: dict[str, object] = {}
    rows = [
        _detector_row(0, 3, "court", no_court_reason=None, reused_from=None),
        _detector_row(3, 4, "scene_too_short_for_feet"),
        _detector_row(4, 6, "no_court", no_court_reason="no_candidates", reused_from=None),
    ]

    def fake_detector(settings: vision.CourtDetectorSettings, **kwargs: object) -> dict[str, object]:
        seen.update(kwargs, settings=settings)
        work_dir = kwargs["work_dir"]
        assert isinstance(work_dir, Path) and work_dir.is_dir()
        return _detector_result(metadata.frame_count, rows)

    monkeypatch.setattr(vision, "run_court_detector", fake_detector)
    output_dir = tmp_path / "court"
    court = vision.build_detected_court_stage(
        video_id="0012",
        metadata=metadata,
        pose=pose,
        pose_dir=tmp_path / "pose",
        settings=_settings(),
        output_dir=output_dir,
    )

    assert seen["video_path"] == metadata.source_path
    assert seen["pose_dir"] == tmp_path / "pose"
    assert "scenes_path" not in seen  # the detector finds the scenes with PySceneDetect
    assert Path(str(seen["work_dir"])).parent == output_dir
    assert not Path(str(seen["work_dir"])).exists()
    assert court.raw_cuts == ((0, 3), (3, 4), (4, 6))
    assert court.artifacts is not None
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(
        path.name for path in court.artifacts.as_mapping().values()
    )
    restored = vision.load_court_vision(
        output_dir, video_id="0012", frame_count=metadata.frame_count, resolution=(100.0, 50.0),
    )
    assert restored.evidence.court_present.tolist() == [True] * 3 + [False] * 3
    records = restored.evidence.scene_records
    assert [record.status for record in records] == [
        SceneStatus.COURT, SceneStatus.SCENE_TOO_SHORT_FOR_FEET, SceneStatus.NO_COURT,
    ]
    assert records[2].no_court_reason == "no_candidates"
    _assert_structured_equal(records, court.evidence.scene_records)


@pytest.mark.parametrize(("result_fields", "message"), [
    ({"schema": "court-detector-video/0", "frame_count": 99}, "unsupported court detector schema"),
    ({"frame_count": 5}, "frame count 5 differs from 6"),
    ({"native_size": [1920, 1080]}, "native size"),
])
def test_detected_court_stage_rejects_a_result_for_another_video_before_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_fields: dict[str, object],
    message: str,
) -> None:
    metadata = _metadata(tmp_path, frame_count=6)
    result = {**_detector_result(6, [_detector_row(0, 6, "court")]), **result_fields}
    monkeypatch.setattr(vision, "run_court_detector", lambda *_args, **_kwargs: result)
    with pytest.raises(ValueError, match=message):
        vision.build_detected_court_stage(
            video_id="0012", metadata=metadata, pose=_two_player_pose(6), pose_dir=tmp_path,
            settings=_settings(), output_dir=tmp_path / "court",
        )
    assert list((tmp_path / "court").iterdir()) == []


def test_detected_court_stage_fails_when_no_scene_has_an_accepted_court(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(tmp_path, frame_count=6)
    rows = [
        _detector_row(0, 1, "scene_too_short_for_feet"),
        _detector_row(1, 3, "no_court", no_court_reason="refit_camera_implausible"),
        _detector_row(3, 6, "detection_failed", error="CourtFitError()", traceback="..."),
    ]
    monkeypatch.setattr(vision, "run_court_detector", lambda *_args, **_kwargs: _detector_result(6, rows))
    with pytest.raises(NoAcceptedCourtError, match="detection_failed"):
        vision.build_detected_court_stage(
            video_id="0012", metadata=metadata, pose=_two_player_pose(6), pose_dir=tmp_path,
            settings=_settings(), output_dir=tmp_path / "court",
        )
    failure_path = tmp_path / "court" / vision.COURT_FAILURE_FILENAME
    failure = vision.load_json_gz(failure_path)
    assert failure["video_id"] == "0012"
    assert [row["status"] for row in failure["scene_records"]] == [
        "scene_too_short_for_feet", "no_court", "detection_failed",
    ]
    assert failure["scene_records"][1]["no_court_reason"] == "refit_camera_implausible"
    assert failure["scene_records"][2]["error"] == "CourtFitError()"
    assert all(not row["scene_valid"] for row in failure["scene_records"])

    rows[:] = [_detector_row(0, 6, "court")]
    vision.build_detected_court_stage(
        video_id="0012", metadata=metadata, pose=_two_player_pose(6), pose_dir=tmp_path,
        settings=_settings(), output_dir=tmp_path / "court",
    )
    assert not failure_path.exists()


def _detected_evidence(frame_count: int) -> CourtEvidenceResult:
    """Detected evidence with two courts, a reused view and every no-court status."""
    shifted = np.array(FULL_FRAME_PX) + [5.0, 0.0]
    scenes = [
        DetectorScene(0, 4, 2, SceneStatus.COURT, np.array(FULL_FRAME_PX), None, None, None),
        DetectorScene(4, 6, 5, SceneStatus.SCENE_TOO_SHORT_FOR_FEET, None, None, None, None),
        DetectorScene(6, 10, 8, SceneStatus.COURT, shifted, None, "0012_scene_0", None),
        DetectorScene(10, 12, 11, SceneStatus.NO_COURT, None, "no_candidates", None, None),
        DetectorScene(12, frame_count, 13, SceneStatus.DETECTION_FAILED, None, None, None, "CourtFitError()"),
    ]
    pose = _two_player_pose(frame_count)
    return build_court_detector_evidence(
        "case-0012", "0012", (100.0, 50.0), (100.0, 50.0), scenes, pose.bboxes, pose.scores, pose.ndet,
    )


def test_court_evidence_round_trip_keeps_each_scene_geometry_status_and_reason(tmp_path: Path) -> None:
    video_id = "0012"
    frame_count = 14
    evidence = _detected_evidence(frame_count)
    raw_cuts = tuple((record.start_frame, record.end_frame) for record in evidence.scene_records)
    vision.persist_court_vision(
        tmp_path,
        video_id=video_id,
        court=vision.CourtVision(raw_cuts, evidence),
        frame_count=frame_count,
        resolution=(100.0, 50.0),
    )
    restored = vision.load_court_vision(
        tmp_path, video_id=video_id, frame_count=frame_count, resolution=(100.0, 50.0),
    )

    assert isinstance(restored.evidence.scene_records[0].video_id, str)
    _assert_structured_equal(restored.evidence.scene_records, evidence.scene_records)
    assert restored.evidence.inputs is not None
    rows = restored.evidence.inputs.homography_rows
    assert rows[["start_frame", "end_frame"]].values.tolist() == [[0, 4], [6, 10]]
    assert rows["upleft_x"].tolist() == pytest.approx([0.0, 5.0])
    np.testing.assert_array_equal(restored.evidence.court_present, evidence.court_present)
    assert restored.evidence.court_present.tolist() == (
        [True] * 4 + [False] * 2 + [True] * 4 + [False] * 4
    )


def test_court_loader_names_a_stale_schema_before_its_fields(tmp_path: Path) -> None:
    video_id = "0012"
    frame_count = 4
    artifacts = vision.persist_court_vision(
        tmp_path,
        video_id=video_id,
        court=_court_vision(video_id, frame_count),
        frame_count=frame_count,
        resolution=(100.0, 50.0),
    )
    payload = vision.load_json_gz(artifacts.evidence)
    stale = {**payload, "schema": "court-evidence/0.1", "consensus": None}
    vision.save_json_gz(artifacts.evidence, stale)
    with pytest.raises(ValueError, match=r"unsupported court evidence schema 'court-evidence/0\.1'"):
        vision.load_court_vision(tmp_path, video_id=video_id, frame_count=frame_count, resolution=(100.0, 50.0))

    vision.save_json_gz(artifacts.evidence, {**payload, "consensus": None})
    with pytest.raises(ValueError, match="payload fields differ from court-evidence/0.2"):
        vision.load_court_vision(tmp_path, video_id=video_id, frame_count=frame_count, resolution=(100.0, 50.0))


@pytest.mark.parametrize(("scene_index", "changes", "message"), [
    (0, {"sampled_frame_indices": [2]}, "fields differ from CourtSceneRecord"),
    (0, {"status": "fallback"}, "is not a court scene status"),
    (0, {"corners_native_px": None}, "corners must be present exactly when its status is court"),
    (1, {"corners_native_px": FULL_FRAME_PX}, "corners must be present exactly when its status is court"),
    (1, {"scene_valid": True}, "without a court cannot be valid"),
    (4, {"error": None}, "error must be present exactly when detection failed"),
    (0, {"analysed_frame": 9}, "analysed frame lies outside its scene interval"),
])
def test_court_loader_rejects_inconsistent_scene_records(
    tmp_path: Path, scene_index: int, changes: dict[str, object], message: str,
) -> None:
    video_id = "0012"
    frame_count = 14
    evidence = _detected_evidence(frame_count)
    raw_cuts = tuple((record.start_frame, record.end_frame) for record in evidence.scene_records)
    artifacts = vision.persist_court_vision(
        tmp_path,
        video_id=video_id,
        court=vision.CourtVision(raw_cuts, evidence),
        frame_count=frame_count,
        resolution=(100.0, 50.0),
    )
    payload = vision.load_json_gz(artifacts.evidence)
    payload["scene_records"][scene_index].update(changes)
    vision.save_json_gz(artifacts.evidence, payload)
    with pytest.raises(ValueError, match=message):
        vision.load_court_vision(tmp_path, video_id=video_id, frame_count=frame_count, resolution=(100.0, 50.0))


def test_court_loader_rejects_missing_scene_provenance(tmp_path: Path) -> None:
    video_id = "0012"
    frame_count = 4
    artifacts = vision.persist_court_vision(
        tmp_path,
        video_id=video_id,
        court=_court_vision(video_id, frame_count),
        frame_count=frame_count,
        resolution=(100.0, 50.0),
    )
    payload = vision.load_json_gz(artifacts.evidence)
    payload["scene_records"] = []
    vision.save_json_gz(artifacts.evidence, payload)

    with pytest.raises(ValueError, match="scene record count differs"):
        vision.load_court_vision(
            tmp_path,
            video_id=video_id,
            frame_count=frame_count,
            resolution=(100.0, 50.0),
        )


def test_full_annotation_matches_direct_run_video_and_captures_both_masks(
    tmp_path: Path,
) -> None:
    metadata = _metadata(tmp_path, frame_count=60)
    video_id = "match-alpha"
    track = np.zeros((metadata.frame_count, 3), dtype=np.float64)
    pose = _pose_arrays(metadata.frame_count)
    court = _court_vision(video_id, metadata.frame_count)
    guard_codes = np.zeros(metadata.frame_count, dtype=np.uint8)
    direct_result, direct_capture = _direct_annotation(
        video_id,
        metadata,
        track,
        pose,
        court,
        guard_codes,
    )

    output = vision.run_full_annotation_stage(
        video_id=video_id,
        metadata=metadata,
        track=track,
        inpaint_fill_mask=np.zeros(metadata.frame_count, dtype=bool),
        guard_codes=guard_codes,
        pose=pose,
        court=court,
        output_dir=tmp_path / "annotation",
    )

    assert output.run.result == direct_result
    assert direct_capture.raw_exclusion_mask is not None
    assert direct_capture.definitive_exclusion_mask is not None
    np.testing.assert_array_equal(output.run.raw_replay_mask, direct_capture.raw_exclusion_mask)
    np.testing.assert_array_equal(
        output.run.definitive_exclusion_mask,
        direct_capture.definitive_exclusion_mask,
    )
    assert output.run.raw_replay_mask.dtype == np.bool_
    assert output.run.raw_replay_mask.shape == (metadata.frame_count,)
    assert output.run.definitive_exclusion_mask.dtype == np.bool_
    assert output.run.definitive_exclusion_mask.shape == (metadata.frame_count,)


def test_full_annotation_rejects_non_replay_dead_mask_mode(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path, frame_count=10)
    with pytest.raises(ValueError, match="DeadMaskMode.REPLAY"):
        vision.run_full_annotation_stage(
            video_id="video",
            metadata=metadata,
            track=np.zeros((metadata.frame_count, 3), dtype=float),
            inpaint_fill_mask=np.zeros(metadata.frame_count, dtype=bool),
            guard_codes=np.zeros(metadata.frame_count, dtype=np.uint8),
            pose=_pose_arrays(metadata.frame_count),
            court=_court_vision("video", metadata.frame_count),
            output_dir=tmp_path / "annotation",
            base=BaseAnnotatorConfig(dead_mask_mode=DeadMaskMode.COMPOSITION),
        )

    assert not (tmp_path / "annotation" / vision.ANNOTATOR_RESULT_FILENAME).exists()


def test_full_annotation_uses_guard_codes_and_keeps_fill_mask_as_measurement_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(tmp_path, frame_count=10)
    track = np.zeros((metadata.frame_count, 3), dtype=float)
    track[:, 2] = 1.0
    fill_mask = np.ones(metadata.frame_count, dtype=bool)
    guard_codes = np.zeros(metadata.frame_count, dtype=np.uint8)
    guard_codes[[2, 5, 8]] = [1, 2, 3]

    def fake_run_video(
        *_args: object,
        inpaint_codes: np.ndarray,
        shuttle_hallucination_mask: np.ndarray | None = None,
        capture: RunCapture,
        **_kwargs: object,
    ) -> AnnotatorResult:
        np.testing.assert_array_equal(inpaint_codes, guard_codes)
        assert shuttle_hallucination_mask is None
        capture.raw_exclusion_mask = np.zeros(metadata.frame_count, dtype=bool)
        capture.definitive_exclusion_mask = np.zeros(metadata.frame_count, dtype=bool)
        return AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, [])

    monkeypatch.setattr(run_video_module, "run_video", fake_run_video)
    output = vision.run_full_annotation_stage(
        video_id="video",
        metadata=metadata,
        track=track,
        inpaint_fill_mask=fill_mask,
        guard_codes=guard_codes,
        pose=_pose_arrays(metadata.frame_count),
        court=_court_vision("video", metadata.frame_count),
        output_dir=tmp_path / "annotation",
    )

    quality = output.run.shuttle_quality
    assert quality.inpaint_filled_frames == metadata.frame_count
    assert quality.inpaint_visible_filled_frames == metadata.frame_count
    assert quality.guard_counts_per_code == (7, 1, 1, 1)
    assert quality.filled_counts_per_code == (7, 1, 1, 1)
    assert quality.guard_rejected_frames == 3
    assert quality.filled_guard_rejected_frames == 3


@pytest.mark.parametrize(
    ("mask_name", "bad_mask"),
    [
        ("raw_exclusion_mask", np.zeros((10, 1), dtype=bool)),
        ("raw_exclusion_mask", np.zeros(10, dtype=np.uint8)),
        ("definitive_exclusion_mask", np.zeros((10, 1), dtype=bool)),
        ("definitive_exclusion_mask", np.zeros(10, dtype=np.uint8)),
    ],
)
def test_malformed_run_capture_masks_raise_original_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mask_name: str,
    bad_mask: np.ndarray,
) -> None:
    metadata = _metadata(tmp_path, frame_count=10)

    def fake_run_video(
        *_args: object,
        capture: RunCapture,
        **_kwargs: object,
    ) -> AnnotatorResult:
        capture.raw_exclusion_mask = np.zeros(metadata.frame_count, dtype=bool)
        capture.definitive_exclusion_mask = np.zeros(metadata.frame_count, dtype=bool)
        setattr(capture, mask_name, bad_mask)
        return AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, [])

    monkeypatch.setattr(run_video_module, "run_video", fake_run_video)
    output_dir = tmp_path / "annotation"
    with pytest.raises(ValueError, match="one-dimensional boolean"):
        vision.run_full_annotation_stage(
            video_id="video",
            metadata=metadata,
            track=np.zeros((metadata.frame_count, 3), dtype=float),
            inpaint_fill_mask=np.zeros(metadata.frame_count, dtype=bool),
            guard_codes=np.zeros(metadata.frame_count, dtype=np.uint8),
            pose=_pose_arrays(metadata.frame_count),
            court=_court_vision("video", metadata.frame_count),
            output_dir=output_dir,
        )

    assert not (output_dir / vision.ANNOTATOR_RESULT_FILENAME).exists()


@pytest.mark.parametrize(
    ("raw_mask", "definitive_mask"),
    [
        (np.zeros((5, 1), dtype=bool), np.zeros(5, dtype=bool)),
        (np.zeros(5, dtype=np.uint8), np.zeros(5, dtype=bool)),
        (np.zeros(5, dtype=bool), np.zeros((5, 1), dtype=bool)),
        (np.zeros(5, dtype=bool), np.zeros(5, dtype=np.uint8)),
    ],
)
def test_annotation_persistence_rejects_malformed_captured_masks(
    tmp_path: Path,
    raw_mask: np.ndarray,
    definitive_mask: np.ndarray,
) -> None:
    run = vision.AnnotationRun(
        video_id="video",
        result=AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, []),
        raw_replay_mask=raw_mask,
        definitive_exclusion_mask=definitive_mask,
        shuttle_quality=_quality(np.zeros((5, 3), dtype=float)),
    )

    with pytest.raises(ValueError, match="one-dimensional boolean"):
        vision.persist_annotation_run(tmp_path, run, frame_count=5)


def test_annotation_persistence_round_trips_every_primitive_and_distinct_masks(
    tmp_path: Path,
) -> None:
    accepted = ContactCandidate(0, 12, True, True, False)
    raw = ContactCandidate(0, 11, False, False, None)
    result = AnnotatorResult(
        spans=[(5, 30)],
        contacts=[raw, accepted],
        filtered_contacts=[accepted],
        filtered_by_rally={0: [12]},
        striker_halves=[Half.TOP],
        n_strokes_list=[1],
        next_servers=[Half.BOT],
        fitted_first_all=[Half.TOP],
        verdict_rows={
            0: VerdictRow(0, Half.TOP, Verdict.WON, VerdictSource.NEXT_SERVER, 0.2, False, True),
        },
        landings={0: Landing(24, (0.4, 0.8), Half.BOT, False, False)},
        geometric_verdict_rows={
            0: GeometricVerdictRow(0, Verdict.WON, Half.TOP, True, False),
        },
        hit_height_by_frame={12: 2},
        hit_height_failures=[(0, 0, 12, "unmeasured")],
    )
    raw_mask = np.zeros(50, dtype=bool)
    raw_mask[10:15] = True
    definitive_mask = np.zeros(50, dtype=bool)
    run = vision.AnnotationRun(
        "match-alpha",
        result,
        raw_mask,
        definitive_mask,
        _quality(np.zeros((50, 3), dtype=float)),
    )

    artifacts = vision.persist_annotation_run(tmp_path, run, frame_count=50)
    payload = vision.load_json_gz(artifacts.result)

    assert payload["schema"] == vision.ANNOTATOR_RESULT_SCHEMA
    assert payload["video_id"] == "match-alpha"
    primitives = payload["result"]
    assert isinstance(primitives, dict)
    assert set(primitives) == set(AnnotatorResult._fields)
    assert primitives["contacts"] == [
        {
            "rally_id": 0,
            "contact_frame": 11,
            "proximity_ok": False,
            "wrist_near": False,
            "suppressed": None,
        },
        {
            "rally_id": 0,
            "contact_frame": 12,
            "proximity_ok": True,
            "wrist_near": True,
            "suppressed": False,
        },
    ]
    assert primitives["verdict_rows"]["0"]["verdict_source"] == "next_server"
    assert primitives["landings"]["0"]["norm"] == [0.4, 0.8]
    assert primitives["hit_height_failures"] == [[0, 0, 12, "unmeasured"]]
    np.testing.assert_array_equal(vision.load_npy_xz(artifacts.raw_replay_mask), raw_mask)
    assert vision.load_json_gz(artifacts.shuttle_quality) == run.shuttle_quality.to_payload()
    np.testing.assert_array_equal(
        vision.load_npy_xz(artifacts.definitive_exclusion_mask),
        definitive_mask,
    )

    chunks = [{"chunk_id": "c0", "start": 4.0, "end": 5.0, "text": "commentary"}]
    raw_rows = pair_video("match-alpha", [(0, 0, 30)], chunks, raw_mask, fps=10.0)
    definitive_rows = pair_video(
        "match-alpha",
        [(0, 0, 30)],
        chunks,
        definitive_mask,
        fps=10.0,
    )
    assert raw_rows[0]["chunk_id"] == ""
    assert definitive_rows[0]["chunk_id"] == "c0"


def test_court_failure_keeps_the_vote_for_a_detected_but_rejected_court(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata(tmp_path, frame_count=6)
    rows = [_detector_row(0, 6, "court", corners_native_px=[[0., 0.], [10., 0.], [10., 50.], [0., 50.]])]
    monkeypatch.setattr(vision, "run_court_detector", lambda *_args, **_kwargs: _detector_result(6, rows))
    with pytest.raises(NoAcceptedCourtError):
        vision.build_detected_court_stage(
            video_id="0012", metadata=metadata, pose=_two_player_pose(6), pose_dir=tmp_path,
            settings=_settings(), output_dir=tmp_path / "court",
        )
    failure = vision.load_json_gz(tmp_path / "court" / vision.COURT_FAILURE_FILENAME)
    record, = failure["scene_records"]
    assert record["status"] == "court"
    assert record["exactly_two_count"] == 0
    assert record["exactly_two_fraction"] == 0
    assert record["scene_valid"] is False
