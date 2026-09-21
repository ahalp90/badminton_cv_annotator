from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scratch.court_det_fix.w5_holistic import replay_marking_junctions as repair


def _case(case_id: str = "yellow_short_frame_14") -> dict:
    return {
        "id": case_id,
        "dimensions": {"width": 100, "height": 80},
        "anchor_frame_index": 14,
        "segments_px": [],
        "bbox_px": [],
    }


def _image_record(image_path: Path, case: dict | None = None) -> dict:
    case = _case() if case is None else case
    return {
        "frame_index": case["anchor_frame_index"],
        "image": image_path.name,
        "image_md5": hashlib.md5(image_path.read_bytes(), usedforsecurity=False).hexdigest(),
        "dimensions": case["dimensions"],
        "bboxes": [[1, 2, 10, 20], [30, 30, 40, 50], [50, 10, 60, 20]],
        "scores": [0.2, 0.200001, 0.1],
    }


def _pinned_image_record(image_path: Path, case: dict) -> dict:
    record = _image_record(image_path, case)
    record["image_md5"] = repair.EXPECTED_IMAGE_MD5[case["id"]]
    return record


def _gx5_record() -> dict:
    return {
        "frame_index": 5,
        "image": "gxBQ_window_00_frame_00000005.png",
        "image_md5": "a" * 32,
        "dimensions": {"width": 1920, "height": 1080},
        "bboxes": [[100, 100, 300, 700]],
        "scores": [0.9],
    }


def _write_packet(path: Path, records: list[tuple[str, dict]]) -> None:
    path.write_text(json.dumps({
        "schema": repair.DETECTION_SCHEMA,
        "model": {
            "basename": repair.DEFAULT_MODEL_BASENAME,
            "score_cutoff": 0.2,
            "score_rule": "strict_gt",
        },
        "coordinate_order": "xyxy",
        "cases": {case_id: record for case_id, record in records},
    }))


def test_strict_detection_cutoff_keeps_only_scores_above_point_two() -> None:
    boxes, scores = repair._validate_boxes_and_scores(
        [[1, 2, 10, 20], [30, 30, 40, 50], [50, 10, 60, 20]],
        [0.2, 0.200001, 0.1],
        (100, 80),
        "case",
    )

    np.testing.assert_array_equal(boxes, np.asarray([[30, 30, 40, 50]], dtype=float))
    np.testing.assert_array_equal(scores, np.asarray([0.200001], dtype=float))


def test_exact_frame_hash_and_dimensions_are_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "yellow_short_frame_00000014.png"
    assert cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))
    case = _case()
    record = _pinned_image_record(image_path, case)
    record["image"] = image_path.name

    def validate_image(path: Path, dimensions: tuple[int, int], expected_md5: str, context: str) -> str:
        assert expected_md5 == repair.EXPECTED_IMAGE_MD5[context]
        return expected_md5

    monkeypatch.setattr(repair, "_validate_image", validate_image)

    validated = repair._validate_detection_frame("yellow_short_frame_14", record, case, tmp_path, "test-detector")

    assert validated["frame_index"] == 14
    assert validated["boxes_px"] == [[30.0, 30.0, 40.0, 50.0]]

    wrong_video_path = tmp_path / "letterboxed_short_frame_00000014.png"
    assert cv2.imwrite(str(wrong_video_path), np.zeros((80, 100, 3), dtype=np.uint8))
    record = _pinned_image_record(wrong_video_path, case)
    with pytest.raises(ValueError, match="image basename"):
        repair._validate_detection_frame("yellow_short_frame_14", record, case, tmp_path, "test-detector")

    record = _pinned_image_record(image_path, case)
    record["image_md5"] = "0" * 32
    with pytest.raises(ValueError, match="pinned image"):
        repair._validate_detection_frame("yellow_short_frame_14", record, case, tmp_path, "test-detector")

    record = _pinned_image_record(image_path, case)
    record["frame_index"] = 15
    with pytest.raises(ValueError, match="frame index"):
        repair._validate_detection_frame("yellow_short_frame_14", record, case, tmp_path, "test-detector")

    record = _pinned_image_record(image_path, case)
    record["dimensions"] = {"width": 99, "height": 80}
    with pytest.raises(ValueError, match="dimensions"):
        repair._validate_detection_frame("yellow_short_frame_14", record, case, tmp_path, "test-detector")


def test_pinned_image_digest_rejects_wrong_content(tmp_path: Path) -> None:
    image_path = tmp_path / "yellow_short_frame_00000014.png"
    assert cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="expected image"):
        repair._validate_image(
            image_path,
            (100, 80),
            repair.EXPECTED_IMAGE_MD5["yellow_short_frame_14"],
            "yellow_short_frame_14",
        )


def test_detection_packet_requires_all_five_affected_cases_and_ignores_gx5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_paths = {}
    frame_indices = (14, 58, 64, 71, 319)
    for case_id, frame_index in zip(repair.AFFECTED_CASE_IDS, frame_indices, strict=True):
        prefix = case_id.rsplit("_frame_", 1)[0]
        image_path = tmp_path / f"{prefix}_frame_{frame_index:08d}.png"
        assert cv2.imwrite(str(image_path), np.zeros((80, 100, 3), dtype=np.uint8))
        image_paths[case_id] = image_path
    packet_path = tmp_path / "detections.json"
    cases = {case_id: _case(case_id) for case_id in repair.AFFECTED_CASE_IDS}
    for case_id, frame_index in zip(repair.AFFECTED_CASE_IDS, frame_indices, strict=True):
        cases[case_id]["anchor_frame_index"] = frame_index
    _write_packet(packet_path, [
        (case_id, _pinned_image_record(image_paths[case_id], cases[case_id]))
        for case_id in repair.AFFECTED_CASE_IDS
    ] + [(repair.GX5_CASE_ID, _gx5_record())])

    monkeypatch.setattr(
        repair,
        "_validate_image",
        lambda path, dimensions, expected_md5, context: expected_md5,
    )
    loaded = repair.load_affected_detections(packet_path, tmp_path, cases)

    assert set(loaded) == set(repair.AFFECTED_CASE_IDS)
    assert all(record["model_name"] == repair.DEFAULT_MODEL_BASENAME for record in loaded.values())

    packet = json.loads(packet_path.read_text())
    packet["model"]["score_rule"] = "greater_than_or_equal"
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="score_cutoff 0.2"):
        repair.load_affected_detections(packet_path, tmp_path, cases)

    packet["model"]["score_rule"] = "strict_gt"
    packet["model"]["basename"] = "different-detector"
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="model basename"):
        repair.load_affected_detections(packet_path, tmp_path, cases)

    packet["model"]["basename"] = repair.DEFAULT_MODEL_BASENAME
    del packet["cases"][repair.GX5_CASE_ID]
    packet_path.write_text(json.dumps(packet))

    with pytest.raises(ValueError, match="canonical IDs"):
        repair.load_affected_detections(packet_path, tmp_path, cases)

    packet["cases"][repair.GX5_CASE_ID] = _gx5_record()
    packet["cases"]["unexpected_case"] = _gx5_record()
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="canonical IDs"):
        repair.load_affected_detections(packet_path, tmp_path, cases)

    del packet["cases"]["unexpected_case"]
    packet["cases"][repair.GX5_CASE_ID] = {"frame_index": 5}
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="missing"):
        repair.load_affected_detections(packet_path, tmp_path, cases)


def _source_candidate(source_index: int) -> dict:
    return {
        "source_index": source_index,
        "stage": "original",
        "extra_gallery_probe": False,
        "corners_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "evidence": {"eligible": True, "scheme_eligible": {"original": True}},
        "metrics": {"corner_max_error_px": float(source_index + 1)},
    }


def _selection_entry(source: dict, metrics: dict | None = None) -> dict:
    entry_id = f"{source['source_index']:04d}:{source['stage']}"
    return {
        "id": entry_id,
        "stripe_score": 0.5,
        "metrics": source["metrics"] if metrics is None else metrics,
    }


def test_frozen_candidates_rejects_selection_metric_drift() -> None:
    source = _source_candidate(0)
    selection = {"entries": [_selection_entry(source, {"corner_max_error_px": 99.0})]}

    with pytest.raises(ValueError, match="metrics differ"):
        repair._frozen_candidates("case", {"entries": [source]}, selection)


def test_frozen_candidates_rejects_truncated_eligible_pool() -> None:
    sources = [_source_candidate(0), _source_candidate(1)]
    selection = {"entries": [_selection_entry(sources[0])]}

    with pytest.raises(ValueError, match="does not equal eligible source pool"):
        repair._frozen_candidates("case", {"entries": sources}, selection)


def test_load_archives_rejects_duplicate_ids_in_each_archive_member(monkeypatch: pytest.MonkeyPatch) -> None:
    case_ids = [f"case_{index}" for index in range(20)]
    base_payload = (
        {"cases": [{"id": case_id} for case_id in case_ids]},
        {"records": [{"id": case_id} for case_id in case_ids]},
        {"records": [{"id": case_id} for case_id in case_ids]},
        {"records": [{"id": case_id} for case_id in case_ids]},
    )
    for member_index in range(4):
        payload = [dict(member) for member in base_payload]
        records_key = "cases" if member_index == 0 else "records"
        payload[member_index][records_key] = [
            *payload[member_index][records_key],
            payload[member_index][records_key][0],
        ]
        monkeypatch.setattr(repair, "_archive_inputs", lambda _archive, _stripe, payload=tuple(payload): payload)

        with pytest.raises(ValueError, match="duplicate record IDs"):
            repair.load_archives(Path("archive.zip"), Path("stripe.zip"))


def test_corrected_case_replaces_boxes_without_changing_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = {
        "id": "0000:original",
        "corners_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "stripe_score": 0.5,
        "metrics": {"corner_max_error_px": 1.0},
    }
    seen_boxes: list[np.ndarray] = []
    monkeypatch.setattr(repair, "prepare_case", lambda case: SimpleNamespace(native_scale=np.ones(2)))
    monkeypatch.setattr(repair, "_frozen_candidates", lambda *args: [candidate])

    def measure(saved: dict, prepared: object, boxes: np.ndarray) -> dict:
        seen_boxes.append(boxes.copy())
        return {
            "id": saved["id"],
            "sites": [],
            "usable_sites": 0,
            "disagreements": 0,
        }

    monkeypatch.setattr(repair, "_measure_candidate", measure)
    result = repair._corrected_case(
        _case(), {}, {},
        {
            "frame_index": 14,
            "image_md5": "a" * 32,
            "dimensions": {"width": 100, "height": 80},
            "boxes_px": [[10, 20, 30, 40]],
            "scores": [0.9],
            "model_name": repair.DEFAULT_MODEL_BASENAME,
        },
    )

    np.testing.assert_array_equal(seen_boxes[0], np.asarray([[10, 20, 30, 40]], dtype=float))
    assert [entry["id"] for entry in result["measurements"]] == ["0000:original"]
    assert [entry["id"] for entry in result["candidates"]] == ["0000:original"]


def test_historical_gate_rejects_duplicate_junction_measurement_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = {
        "id": "0000:original",
        "corners_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "stripe_score": 0.5,
        "metrics": {"corner_max_error_px": 1.0},
    }
    monkeypatch.setattr(repair, "prepare_case", lambda case: SimpleNamespace(native_scale=np.ones(2)))
    monkeypatch.setattr(repair, "_frozen_candidates", lambda *args: [candidate])
    junction_entry = {"id": "0000:original"}
    junction_record = {"entries": [junction_entry, dict(junction_entry)]}
    selection_record = {
        "entries": [],
        "orders": {scheme: ["0000:original"] for scheme in repair.SCHEMES},
    }

    with pytest.raises(ValueError, match="duplicate IDs"):
        repair._historical_gate(_case(), {}, junction_record, selection_record)


def test_rank_preserves_candidate_ids_and_summary_denominator() -> None:
    candidates = [
        {
            "id": "0000:original",
            "stripe_score": 0.5,
            "metrics": {"corner_max_error_px": 10.0},
        },
        {
            "id": "0001:original",
            "stripe_score": 0.4,
            "metrics": {"corner_max_error_px": 20.0},
        },
    ]
    measurements = [
        {
            "id": candidate["id"],
            "sites": [],
            "usable_sites": 0,
            "disagreements": 0,
        }
        for candidate in candidates
    ]
    ranked = repair._rank_case(candidates, measurements)
    archived = {"case": {"orders": ranked["orders"]}}
    case = {"id": "case", "box_source": "exact_detection", **ranked}
    summary = repair.summarise([case], archived)

    assert summary["case_count"] == 1
    assert summary["candidate_id_count"] == 2
    assert summary["accurate_denominator"] == 1
    assert summary["accurate_picks"]["stripe_exclusive"] == 1
    assert all(set(order) == {"0000:original", "0001:original"} for order in ranked["orders"].values())
