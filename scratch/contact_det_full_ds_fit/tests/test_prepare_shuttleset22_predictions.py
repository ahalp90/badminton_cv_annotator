from __future__ import annotations

import ast
import gzip
from pathlib import Path

import numpy as np
import pytest

from scratch.contact_det.scripts.freeze_contact_evidence import FixtureSpec
from scratch.contact_det_full_ds_fit.scripts.prepare_shuttleset22_predictions import (
    EXPECTED_OVERLAPS,
    EXPECTED_UNRESOLVED_IDS,
    VIDEO_IDS,
    ModelBundle,
    _gzip_json_bytes,
    _score_candidates,
    _span_id,
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
