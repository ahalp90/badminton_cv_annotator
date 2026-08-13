"""Dataset-builder shuttle provenance, guard, and resume contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dataset_builder.shuttle_evidence import (
    load_inpaint_fill_mask,
    load_shuttle_evidence,
    persist_shuttle_evidence,
    shuttle_evidence_artifacts,
)
from dataset_builder.vision import save_json_gz, save_npy_xz


def _sidecar_payload(
    input_video: Path,
    *,
    frame_count: int,
    inpaint_selected: list[list[int]] | None = None,
    inpaint_status: str = "applied",
) -> dict[str, object]:
    return {
        "schema": "inpaint_fill_mask/1",
        "index_space": "frame",
        "inpaint_status": inpaint_status,
        "n_rows": frame_count,
        "eval_mode": "nonoverlap",
        "stride": 8,
        "th_h_px": 14.4,
        "tracknet_ckpt": "tracknet.pt",
        "inpaintnet_ckpt": (
            "inpaintnet.pt" if inpaint_status == "applied" else None
        ),
        "input_video": input_video.name,
        "extracted_utc": "2026-08-13T00:00:00Z",
        "inpaint_selected": [[2, 5]] if inpaint_selected is None else inpaint_selected,
    }


def _persisted_fixture(tmp_path: Path) -> tuple[object, Path, Path, Path]:
    frame_count = 40
    input_video = tmp_path / "proxy.mp4"
    input_video.write_bytes(b"proxy")
    tracknet_model = tmp_path / "tracknet.pt"
    inpaint_model = tmp_path / "inpaintnet.pt"
    tracknet_model.write_bytes(b"tracknet")
    inpaint_model.write_bytes(b"inpaint")
    artifacts = shuttle_evidence_artifacts(
        tmp_path / "shuttle",
        input_video=input_video,
        stride=8,
    )
    artifacts.tracknet_csv.parent.mkdir(parents=True)
    artifacts.tracknet_csv.write_text("fixture CSV", encoding="utf-8")
    track = np.zeros((frame_count, 3), dtype=np.float64)
    save_npy_xz(artifacts.shuttle_track, track)
    save_json_gz(
        artifacts.inpaint_sidecar,
        _sidecar_payload(input_video, frame_count=frame_count),
    )
    evidence = persist_shuttle_evidence(
        track=track,
        artifacts=artifacts,
        input_video=input_video,
        input_height=288,
        frame_count=frame_count,
        stride=8,
        tracknet_model=tracknet_model,
        inpaint_model=inpaint_model,
    )
    return evidence, input_video, tracknet_model, inpaint_model


def test_shuttle_evidence_round_trips_provenance_and_guard_grades(
    tmp_path: Path,
) -> None:
    evidence, input_video, tracknet_model, inpaint_model = _persisted_fixture(tmp_path)

    np.testing.assert_array_equal(
        evidence.inpaint_fill_mask,
        np.array([False, False, True, True, True] + [False] * 35),
    )
    assert evidence.guard_codes.dtype == np.uint8
    assert evidence.guard_diagnostics["schema"] == "shuttle-guard/0.1"

    restored = load_shuttle_evidence(
        artifacts=evidence.artifacts,
        input_video=input_video,
        input_height=288,
        frame_count=40,
        stride=8,
        tracknet_model=tracknet_model,
        inpaint_model=inpaint_model,
    )

    np.testing.assert_array_equal(restored.track, evidence.track)
    np.testing.assert_array_equal(restored.inpaint_fill_mask, evidence.inpaint_fill_mask)
    np.testing.assert_array_equal(restored.guard_codes, evidence.guard_codes)
    assert restored.guard_diagnostics == evidence.guard_diagnostics


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "inpaint_fill_mask/2", "schema"),
        ("n_rows", 39, "n_rows"),
        ("eval_mode", "weight", "eval_mode"),
        ("stride", 1, "stride"),
        ("input_video", "other.mp4", "input_video"),
        ("tracknet_ckpt", "other.pt", "tracknet_ckpt"),
        ("inpaintnet_ckpt", "other.pt", "inpaintnet_ckpt"),
    ],
)
def test_inpaint_sidecar_rejects_mismatched_producer_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    input_video = tmp_path / "proxy.mp4"
    sidecar = tmp_path / "proxy_stride8_inpaint_mask.json.gz"
    payload = _sidecar_payload(input_video, frame_count=40)
    payload[field] = value
    save_json_gz(sidecar, payload)

    with pytest.raises(ValueError, match=message):
        load_inpaint_fill_mask(
            sidecar,
            input_video=input_video,
            input_height=288,
            frame_count=40,
            stride=8,
            tracknet_model=tmp_path / "tracknet.pt",
            inpaint_model=tmp_path / "inpaintnet.pt",
        )


@pytest.mark.parametrize(
    "spans",
    [
        [[-1, 2]],
        [[4, 3]],
        [[2, 5], [4, 7]],
        [[2, 41]],
        [[True, 2]],
        [[2.0, 3]],
    ],
)
def test_inpaint_sidecar_rejects_invalid_spans(
    tmp_path: Path,
    spans: list[list[object]],
) -> None:
    input_video = tmp_path / "proxy.mp4"
    sidecar = tmp_path / "proxy_stride8_inpaint_mask.json.gz"
    save_json_gz(
        sidecar,
        _sidecar_payload(input_video, frame_count=40, inpaint_selected=spans),
    )

    with pytest.raises(ValueError, match="span"):
        load_inpaint_fill_mask(
            sidecar,
            input_video=input_video,
            input_height=288,
            frame_count=40,
            stride=8,
            tracknet_model=tmp_path / "tracknet.pt",
            inpaint_model=tmp_path / "inpaintnet.pt",
        )


def test_disabled_inpaint_sidecar_requires_empty_spans(tmp_path: Path) -> None:
    input_video = tmp_path / "proxy.mp4"
    sidecar = tmp_path / "proxy_stride8_inpaint_mask.json.gz"
    save_json_gz(
        sidecar,
        _sidecar_payload(
            input_video,
            frame_count=40,
            inpaint_status="disabled",
            inpaint_selected=[[2, 5]],
        ),
    )

    with pytest.raises(ValueError, match="must not select"):
        load_inpaint_fill_mask(
            sidecar,
            input_video=input_video,
            input_height=288,
            frame_count=40,
            stride=8,
            tracknet_model=tmp_path / "tracknet.pt",
            inpaint_model=None,
        )


def test_restore_rejects_guard_codes_from_a_different_track(tmp_path: Path) -> None:
    evidence, input_video, tracknet_model, inpaint_model = _persisted_fixture(tmp_path)
    wrong_codes = np.ones(40, dtype=np.uint8)
    save_npy_xz(evidence.artifacts.guard_codes, wrong_codes)

    with pytest.raises(ValueError, match="differ from the final track"):
        load_shuttle_evidence(
            artifacts=evidence.artifacts,
            input_video=input_video,
            input_height=288,
            frame_count=40,
            stride=8,
            tracknet_model=tracknet_model,
            inpaint_model=inpaint_model,
        )
