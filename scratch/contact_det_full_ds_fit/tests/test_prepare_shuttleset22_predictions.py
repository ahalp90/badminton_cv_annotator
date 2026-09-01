from __future__ import annotations

import ast
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scratch.contact_det.scripts.freeze_contact_evidence import FixtureSpec
from scratch.contact_det_full_ds_fit.scripts import (
    prepare_shuttleset22_predictions as prediction_program,
)
from scratch.contact_det_full_ds_fit.scripts.inpaint_shuttleset22_tracks import (
    RECEIPT_FILENAME,
    output_paths,
)
from scratch.contact_det_full_ds_fit.scripts.prepare_shuttleset22_predictions import (
    ANNOTATION_FILENAMES,
    COURT_FILENAMES,
    EXPECTED_OVERLAPS,
    EXPECTED_UNRESOLVED_IDS,
    POSE_FILENAMES,
    PREDICTION_OUTPUT_FILENAMES,
    VIDEO_IDS,
    ArtifactMetadata,
    CheckedVideo,
    ModelBundle,
    SourceSpec,
    _gzip_json_bytes,
    _input_records,
    _score_candidates,
    _span_id,
    _validate_input_records,
    _validate_output_hashes,
    _write_verified_bytes,
    fill_mask_from_sidecar,
    parse_args,
    source_specs_from_payload,
)


def _source_manifest_payload() -> dict[str, object]:
    videos: list[dict[str, object]] = []
    for video_id in range(1, 59):
        row: dict[str, object] = {"id": video_id, "video": f"video_{video_id:02d}"}
        if video_id in EXPECTED_OVERLAPS:
            row.update(
                source_kind="shuttleset_overlap",
                overlap_shuttleset_id=EXPECTED_OVERLAPS[video_id],
            )
        elif video_id in EXPECTED_UNRESOLVED_IDS:
            row["source_kind"] = "unresolved"
        else:
            row["source_kind"] = "download"
        videos.append(row)
    return {"schema": "shuttleset22-sources/1", "videos": videos}


def _checked_video(tmp_path: Path) -> CheckedVideo:
    directory = tmp_path / "08 video_08"
    directory.mkdir()
    paths = [
        directory / RECEIPT_FILENAME,
        directory / "shuttle_track_inpainted.npy.xz",
        directory / "shuttle_guard_codes_inpainted.npy.xz",
        directory / "court_receipt.json.gz",
    ]
    paths.extend(directory / filename for filename in POSE_FILENAMES + COURT_FILENAMES)
    paths.append(output_paths(directory, directory.name).sidecar_path)
    for index, path in enumerate(paths):
        path.write_bytes(f"input {index}".encode())

    return CheckedVideo(
        SourceSpec(8, "video_08"),
        tmp_path / "08 video_08.mp4",
        directory,
        ArtifactMetadata(100, 1920, 1080, "receipt", "shuttle", "code", "model"),
    )


def test_source_manifest_returns_the_fixed_downloads() -> None:
    sources = source_specs_from_payload(_source_manifest_payload())

    assert tuple(source.video_id for source in sources) == VIDEO_IDS
    assert sources[0].video_name == "video_08"
    assert sources[-1].video_name == "video_57"


def test_source_manifest_rejects_a_changed_overlap() -> None:
    payload = _source_manifest_payload()
    videos = payload["videos"]
    assert isinstance(videos, list)
    first = videos[0]
    assert isinstance(first, dict)
    first["overlap_shuttleset_id"] = 99

    with pytest.raises(ValueError, match="overlap mapping differs"):
        source_specs_from_payload(payload)


def test_fill_mask_restores_half_open_spans() -> None:
    sidecar = {
        "schema": "inpaint_fill_mask/1",
        "n_rows": 8,
        "inpaint_selected": [[1, 3], [5, 8]],
    }

    mask = fill_mask_from_sidecar(sidecar, 8)

    assert mask.tolist() == [False, True, True, False, False, True, True, True]


def test_fill_mask_rejects_overlapping_spans() -> None:
    sidecar = {
        "schema": "inpaint_fill_mask/1",
        "n_rows": 8,
        "inpaint_selected": [[1, 4], [3, 5]],
    }

    with pytest.raises(ValueError, match="ordered, disjoint"):
        fill_mask_from_sidecar(sidecar, 8)


def test_score_candidates_uses_the_recorded_field_order() -> None:
    dtype = np.dtype(
        [
            ("fixture", "S7"),
            ("interval_id", "<i4"),
            ("frame", "<i4"),
            ("fps", "<f4"),
            ("second", "<f4"),
            ("first", "<f4"),
        ]
    )
    rows = np.zeros(3, dtype=dtype)
    rows["fixture"] = b"8"
    rows["frame"] = [10, 11, 30]
    rows["fps"] = 30.0
    rows["first"] = [1.0, 2.0, 3.0]
    rows["second"] = [10.0, 20.0, 30.0]

    class RecordingModel:
        def __init__(self) -> None:
            self.matrix: np.ndarray | None = None

        def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
            self.matrix = matrix.copy()
            positive = np.asarray([0.95, 0.91, 0.20])
            return np.column_stack((1.0 - positive, positive))

    model = RecordingModel()
    scores, predictions = _score_candidates(
        rows,
        FixtureSpec("8", 8, 30.0),
        ModelBundle(model, ("first", "second")),
    )

    assert model.matrix is not None
    assert model.matrix.tolist() == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    assert predictions.tolist() == [10]
    assert scores["kept"].tolist() == [True, False, False]


def test_span_lookup_uses_half_open_ranges() -> None:
    spans = [
        {"span_id": 0, "start_frame": 10, "end_frame": 20},
        {"span_id": 1, "start_frame": 30, "end_frame": 40},
    ]

    assert _span_id(10, spans) == 0
    assert _span_id(19, spans) == 0
    assert _span_id(20, spans) is None


def test_gzip_prediction_bytes_are_deterministic() -> None:
    payload = {"status": "complete", "video_ids": [8, 9]}

    first = _gzip_json_bytes(payload)
    second = _gzip_json_bytes(payload)

    assert first == second
    assert gzip.decompress(first).endswith(b"\n")


def test_saved_input_records_must_match_live_files(tmp_path: Path) -> None:
    video = _checked_video(tmp_path)
    recorded = _input_records(video)

    _validate_input_records(video, recorded)

    (video.directory / "pose_kps.npy.xz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="saved input files differ"):
        _validate_input_records(video, recorded)


def test_saved_output_hash_list_must_be_complete(tmp_path: Path) -> None:
    root = tmp_path / "ss22_08"
    output_hashes: dict[str, str] = {}
    for name in PREDICTION_OUTPUT_FILENAMES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = name.encode()
        path.write_bytes(payload)
        output_hashes[name] = hashlib.sha256(payload).hexdigest()

    _validate_output_hashes(root, output_hashes)

    output_hashes.pop(f"annotation/{ANNOTATION_FILENAMES[-1]}")
    with pytest.raises(ValueError, match="output list differs"):
        _validate_output_hashes(root, output_hashes)


def test_combined_bytes_must_reload_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "combined_predictions.json.gz"

    def write_changed_bytes(destination: Path, _payload: bytes) -> None:
        Path(destination).write_bytes(b"changed")

    monkeypatch.setattr(prediction_program, "_write_bytes", write_changed_bytes)

    with pytest.raises(ValueError, match="differs after reload"):
        _write_verified_bytes(path, b"expected")


def test_prediction_program_has_no_label_reader_import() -> None:
    module_path = Path(
        "scratch/contact_det_full_ds_fit/scripts/prepare_shuttleset22_predictions.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "annotator.calibration.shuttleset22_features" not in imported
    assert "pandas" not in imported


def test_prediction_arguments_have_no_label_path() -> None:
    arguments = parse_args(
        [
            "--source-manifest",
            "sources.toml",
            "--inpaint-root",
            "inpaint",
            "--source-root",
            "videos",
            "--model",
            "model.joblib",
            "--model-result",
            "model.json",
            "--setting-result",
            "setting.json",
            "--output-root",
            "predictions",
            "--source-commit",
            "1234567",
        ]
    )

    assert all("label" not in name for name in vars(arguments))
