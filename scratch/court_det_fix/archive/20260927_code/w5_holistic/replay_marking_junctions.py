"""Repair marking-junction rankings with exact-frame person detections.

The replay archive already contains every court geometry and stripe score needed
for this comparison.  This utility only changes the boxes passed to the
junction observer.  It first proves that current helpers still reproduce the
archived measurements and rankings for the fifteen unaffected cases.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "src"))

from experiments.annotator.independent_court import detector, junction_observations
from experiments.annotator.independent_court.assignment import (
    Observations,
    prepare_observations,
)
from experiments.annotator.independent_court.export_people import DEFAULT_MODEL_BASENAME
from experiments.annotator.independent_court.run_junction_selection import SCHEMES, rank

MARKING_INPUTS = "marking_inputs.json.gz"
MARKING_RESULTS = "reverse_results.json.gz"
JUNCTIONS = "junctions_v1.json.gz"
JUNCTION_SELECTION = "junction_selection_v1.json.gz"
SCORE_CUTOFF = 0.2
COORDINATE_TOLERANCE_PX = 1.0
ACCURATE_PX = 15.0
SHORT_AFFECTED_IDS = (
    "yellow_short_frame_14",
    "letterboxed_short_frame_58",
    "centre_short_frame_64",
    "centre_short_frame_71",
)
AM4_CASE_ID = "am4_window_00_frame_319"
EXPECTED_IMAGE_MD5 = {
    "yellow_short_frame_14": "ef16de686c2dd89dee9b959c9aee7668",
    "letterboxed_short_frame_58": "a34a467a28507331ce104acd27e9696d",
    "centre_short_frame_64": "90aab6dbed112b2b77a213d807d80700",
    "centre_short_frame_71": "834b1f6ce7e7acb3580cea12b20c48e2",
    AM4_CASE_ID: "dadb8c117ceb1bfc7a4db92b9a827539",
}
AFFECTED_CASE_IDS = (*SHORT_AFFECTED_IDS, AM4_CASE_ID)
GX5_CASE_ID = "gxBQ_window_00_frame_5"
DETECTION_CASE_IDS = (GX5_CASE_ID, *AFFECTED_CASE_IDS)
AFFECTED_IDS = frozenset(EXPECTED_IMAGE_MD5)
DETECTION_SCHEMA = "person-detections/1"
OUTPUT_SCHEMA = "frozen-marking-junction-repair/1"


@dataclass(frozen=True)
class PreparedCase:
    """The shared observation context for one frozen case."""

    size: tuple[int, int]
    native_scale: np.ndarray
    observations: Observations


def read_archive_json(archive: Path, member: str) -> dict[str, Any]:
    """Read one gzip-compressed JSON member from an archive."""
    with zipfile.ZipFile(archive) as bundle:
        return json.loads(gzip.decompress(bundle.read(member)))


def sha256_file(path: Path) -> str:
    """Hash one input artefact for the single output provenance record."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    """Return the image identity digest used by the exact-detection contract."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json_gz(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic compressed JSON without exposing a partial result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    temporary.write_bytes(gzip.compress(encoded, mtime=0))
    os.replace(temporary, path)


def _read_packet(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _dimensions(value: Any, context: str) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise ValueError(f"{context}: dimensions must contain width and height")
    width, height = value["width"], value["height"]
    if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int):
        raise TypeError(f"{context}: dimensions must be integer pixels")
    if width <= 0 or height <= 0:
        raise ValueError(f"{context}: dimensions must be positive")
    return width, height


def _expected_frame(case: dict[str, Any]) -> int:
    frame = case.get("anchor_frame_index")
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise TypeError(f"{case['id']}: frozen anchor frame is missing")
    return frame


def prepare_case(case: dict[str, Any]) -> PreparedCase:
    """Prepare the frozen line observations in the same working scale as the run."""
    width, height = _dimensions(case["dimensions"], case["id"])
    factor = min(1.0, 960 / max(width, height))
    size = (round(width * factor), round(height * factor))
    native_scale = np.asarray([width / size[0], height / size[1]])
    segments = np.asarray(case["segments_px"], dtype=float).reshape(-1, 4) / np.tile(native_scale, 2)
    return PreparedCase(size, native_scale, prepare_observations(segments, size))


def _validate_boxes_and_scores(
    boxes_value: Any,
    scores_value: Any,
    dimensions: tuple[int, int],
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes_value, dtype=float)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=float)
    else:
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"{context}: boxes must have shape (n, 4)")
    scores = np.asarray(scores_value, dtype=float)
    if scores.ndim != 1 or len(scores) != len(boxes):
        raise ValueError(f"{context}: boxes and scores must have the same length")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError(f"{context}: boxes and scores must be finite")
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError(f"{context}: scores must lie between zero and one")
    if len(boxes):
        x1, y1, x2, y2 = boxes.T
        if np.any(x2 < x1) or np.any(y2 < y1):
            raise ValueError(f"{context}: boxes have reversed bounds")
        width, height = dimensions
        tolerance = COORDINATE_TOLERANCE_PX
        if np.any(x1 < -tolerance) or np.any(x2 > width + tolerance):
            raise ValueError(f"{context}: boxes do not use the frozen image width")
        if np.any(y1 < -tolerance) or np.any(y2 > height + tolerance):
            raise ValueError(f"{context}: boxes do not use the frozen image height")
    keep = scores > SCORE_CUTOFF
    return boxes[keep], scores[keep]


def _validate_packet_record(
    record: Any,
    context: str,
    require_basename: bool = False,
) -> tuple[int, str, tuple[int, int], np.ndarray, np.ndarray]:
    if not isinstance(record, dict):
        raise TypeError(f"{context}: detection packet record must be an object")
    required = {"frame_index", "image", "image_md5", "dimensions", "bboxes", "scores"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"{context}: detection packet is missing {missing}")
    frame = record["frame_index"]
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f"{context}: frame index must be a non-negative integer")
    image = record["image"]
    if not isinstance(image, str) or not image or "\x00" in image:
        raise ValueError(f"{context}: image must be a non-empty path string")
    if require_basename and (
        Path(image).name != image
        or image in {".", ".."}
        or "/" in image
        or "\\" in image
    ):
        raise ValueError(f"{context}: image must be a basename")
    dimensions = _dimensions(record["dimensions"], context)
    image_md5 = record["image_md5"]
    if not isinstance(image_md5, str) or len(image_md5) != 32:
        raise ValueError(f"{context}: image_md5 must be a 32-character hexadecimal digest")
    try:
        int(image_md5, 16)
    except ValueError as error:
        raise ValueError(f"{context}: image_md5 must be hexadecimal") from error
    boxes, scores = _validate_boxes_and_scores(record["bboxes"], record["scores"], dimensions, context)
    return frame, image_md5, dimensions, boxes, scores


def _validate_image(
    image_path: Path,
    expected_dimensions: tuple[int, int],
    expected_md5: str | None,
    context: str,
) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"{context}: image does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"{context}: cannot read image {image_path}")
    width, height = expected_dimensions
    if image.shape[1] != width or image.shape[0] != height:
        raise ValueError(f"{context}: image dimensions do not match the frozen case")
    actual_md5 = md5_file(image_path)
    if expected_md5 is not None and actual_md5 != expected_md5.lower():
        raise ValueError(f"{context}: image MD5 does not match the expected image")
    return actual_md5


def _resolve_packet_image(image_root: Path, relative_name: Any, context: str) -> Path:
    if not isinstance(relative_name, str) or not relative_name or Path(relative_name).is_absolute():
        raise ValueError(f"{context}: image must be a relative path")
    root = image_root.resolve()
    image_path = (root / relative_name).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context}: image escapes the supplied image root") from error
    return image_path


def _validate_detection_frame(
    case_id: str,
    record: dict[str, Any],
    case: dict[str, Any],
    image_root: Path,
    model_basename: str,
) -> dict[str, Any]:
    context = case_id
    if case_id != case["id"]:
        raise ValueError(f"{context}: case ID does not match the frozen case")
    frame, image_md5, dimensions, boxes, scores = _validate_packet_record(record, context)
    if frame != _expected_frame(case):
        raise ValueError(f"{context}: frame index does not match the frozen anchor frame")
    expected_dimensions = _dimensions(case["dimensions"], case["id"])
    if dimensions != expected_dimensions:
        raise ValueError(f"{context}: detection dimensions do not match the frozen case")
    expected_md5 = EXPECTED_IMAGE_MD5.get(case_id)
    if expected_md5 is None:
        raise ValueError(f"{context}: no pinned image identity exists for this case")
    if image_md5.lower() != expected_md5:
        raise ValueError(f"{context}: packet image_md5 does not match the pinned image")
    image_path = _resolve_packet_image(image_root, record["image"], context)
    case_prefix = case_id.rsplit("_frame_", 1)[0]
    expected_basename = f"{case_prefix}_frame_{frame:08d}.png"
    if image_path.name != expected_basename:
        raise ValueError(f"{context}: image basename does not match {expected_basename!r}")
    actual_md5 = _validate_image(image_path, dimensions, expected_md5, context)
    return {
        "frame_index": frame,
        "image_md5": actual_md5,
        "dimensions": {"width": dimensions[0], "height": dimensions[1]},
        "boxes_px": boxes.tolist(),
        "scores": scores.tolist(),
        "model_name": model_basename,
    }


def load_affected_detections(
    packet_path: Path,
    image_root: Path,
    cases_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load and validate all five exact-frame marking detections."""
    packet = _read_packet(packet_path)
    if packet.get("schema") != DETECTION_SCHEMA:
        raise ValueError(f"Detection packet must use schema {DETECTION_SCHEMA!r}")
    model = packet.get("model")
    if not isinstance(model, dict):
        raise TypeError("Detection packet model metadata must be an object")
    model_basename = model.get("basename")
    if model_basename != DEFAULT_MODEL_BASENAME:
        raise ValueError(f"Detection packet model basename must be {DEFAULT_MODEL_BASENAME!r}")
    if model.get("score_cutoff") != SCORE_CUTOFF or model.get("score_rule") != "strict_gt":
        raise ValueError("Detection packet must declare score_cutoff 0.2 with score_rule strict_gt")
    if packet.get("coordinate_order") != "xyxy":
        raise ValueError("Detection packet must declare xyxy coordinate order")
    records = packet.get("cases")
    if not isinstance(records, dict):
        raise TypeError("Detection packet cases must be an object keyed by case ID")
    expected_case_ids = set(DETECTION_CASE_IDS)
    actual_case_ids = set(records)
    if actual_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - actual_case_ids)
        extra = sorted(actual_case_ids - expected_case_ids)
        raise ValueError(
            f"Detection packet cases must contain exactly the canonical IDs "
            f"(missing={missing}, extra={extra})",
        )
    _validate_packet_record(records[GX5_CASE_ID], GX5_CASE_ID, require_basename=True)
    by_id: dict[str, dict[str, Any]] = {}
    for case_id in AFFECTED_CASE_IDS:
        record = records[case_id]
        by_id[case_id] = _validate_detection_frame(
            case_id, record, cases_by_id[case_id], image_root, model_basename,
        )
    return by_id


def _archive_inputs(archive: Path, stripe_archive: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    marking = read_archive_json(archive, MARKING_INPUTS)
    results = read_archive_json(archive, MARKING_RESULTS)
    junctions = read_archive_json(stripe_archive, JUNCTIONS)
    selection = read_archive_json(stripe_archive, JUNCTION_SELECTION)
    if marking.get("schema") != "marking-refit-inputs/1":
        raise ValueError("unsupported marking input schema")
    if junctions.get("schema") != "frozen-junction-diagnostic/1":
        raise ValueError("unsupported junction measurement schema")
    if selection.get("schema") != "frozen-junction-selection/1":
        raise ValueError("unsupported junction selection schema")
    return marking, results, junctions, selection


def _frozen_candidates(
    case_id: str,
    results_record: dict[str, Any],
    selection_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Join saved geometry to the archived eligible ranking pool."""
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in results_record["entries"]:
        entry_id = f"{source['source_index']:04d}:{source['stage']}"
        if entry_id in source_by_id:
            raise ValueError(f"{case_id}: duplicate saved candidate ID {entry_id}")
        source_by_id[entry_id] = source
    candidates = []
    seen: set[str] = set()
    for saved in selection_record["entries"]:
        entry_id = saved["id"]
        if entry_id in seen:
            raise ValueError(f"{case_id}: duplicate selection candidate ID {entry_id}")
        seen.add(entry_id)
        source = source_by_id.get(entry_id)
        if source is None:
            raise ValueError(f"{case_id}: selection candidate {entry_id} has no saved geometry")
        if source["extra_gallery_probe"]:
            raise ValueError(f"{case_id}: gallery probe entered the eligible pool")
        evidence = source["evidence"]
        if not evidence["eligible"] or not evidence["scheme_eligible"]["original"]:
            raise ValueError(f"{case_id}/{entry_id}: selection pool contains an ineligible candidate")
        if saved["metrics"] != source["metrics"]:
            raise ValueError(f"{case_id}/{entry_id}: selection metrics differ from saved reverse metrics")
        candidates.append({
            "id": entry_id,
            "corners_px": source["corners_px"],
            "stripe_score": saved["stripe_score"],
            "metrics": saved["metrics"],
        })
    eligible_source_ids = {
        f"{source['source_index']:04d}:{source['stage']}"
        for source in results_record["entries"]
        if not source["extra_gallery_probe"]
        and source["evidence"]["eligible"]
        and source["evidence"]["scheme_eligible"]["original"]
    }
    if seen != eligible_source_ids:
        missing = sorted(eligible_source_ids - seen)
        extra = sorted(seen - eligible_source_ids)
        raise ValueError(
            f"{case_id}: selection pool does not equal eligible source pool "
            f"(missing={missing}, extra={extra})",
        )
    return candidates


def _measure_candidate(candidate: dict[str, Any], prepared: PreparedCase, boxes: np.ndarray) -> dict[str, Any]:
    corners = np.asarray(candidate["corners_px"], dtype=float).reshape(4, 2) / prepared.native_scale
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, corners.astype(np.float32))
    measured = junction_observations.measure(homography, prepared.observations, boxes, prepared.size)
    return {"id": candidate["id"], **measured}


def _ranking_inputs(candidates: list[dict[str, Any]], measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {measurement["id"]: measurement for measurement in measurements}
    ranking = []
    for candidate in candidates:
        measurement = by_id[candidate["id"]]
        complete = sum(site["usable"] and len(site["agreements"]) == 2 for site in measurement["sites"])
        ranking.append({
            "id": candidate["id"],
            "stripe_score": candidate["stripe_score"],
            "disagreements": measurement["disagreements"],
            "usable_sites": measurement["usable_sites"],
            "complete_agreements": complete,
            "metrics": candidate["metrics"],
        })
    return ranking


def _winners(orders: dict[str, list[str]], ranking_inputs: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    by_id = {entry["id"]: entry for entry in ranking_inputs}
    winners = {}
    for scheme, order in orders.items():
        if not order:
            winners[scheme] = None
            continue
        entry = by_id[order[0]]
        winners[scheme] = {"id": entry["id"], "metrics": entry["metrics"]}
    return winners


def _rank_case(candidates: list[dict[str, Any]], measurements: list[dict[str, Any]]) -> dict[str, Any]:
    ranking_inputs = _ranking_inputs(candidates, measurements)
    orders = rank(ranking_inputs)
    return {
        "measurements": measurements,
        "ranking_inputs": ranking_inputs,
        "orders": orders,
        "winners": _winners(orders, ranking_inputs),
    }


def _historical_gate(
    case: dict[str, Any],
    results_record: dict[str, Any],
    junction_record: dict[str, Any],
    selection_record: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce one archived case before any repaired evidence is assembled."""
    prepared = prepare_case(case)
    candidates = _frozen_candidates(case["id"], results_record, selection_record)
    boxes = np.asarray(case["bbox_px"], dtype=float).reshape(-1, 4) / np.tile(prepared.native_scale, 2)
    archived_entries = junction_record["entries"]
    archived_measurements = {entry["id"]: entry for entry in archived_entries}
    if len(archived_measurements) != len(archived_entries):
        raise ValueError(f"{case['id']}: archived junction measurements contain duplicate IDs")
    measurements = [_measure_candidate(candidate, prepared, boxes) for candidate in candidates]
    measured_by_id = {entry["id"]: entry for entry in measurements}
    if measured_by_id != archived_measurements:
        raise ValueError(f"{case['id']}: current helpers do not reproduce archived junction measurements")
    ranked = _rank_case(candidates, measurements)
    if ranked["orders"] != selection_record["orders"]:
        raise ValueError(f"{case['id']}: current helpers do not reproduce archived junction rankings")
    return {"candidates": candidates, **ranked}


def _corrected_case(
    case: dict[str, Any],
    results_record: dict[str, Any],
    selection_record: dict[str, Any],
    detection: dict[str, Any],
) -> dict[str, Any]:
    prepared = prepare_case(case)
    candidates = _frozen_candidates(case["id"], results_record, selection_record)
    boxes = np.asarray(detection["boxes_px"], dtype=float).reshape(-1, 4) / np.tile(prepared.native_scale, 2)
    ranked = _rank_case(candidates, [_measure_candidate(candidate, prepared, boxes) for candidate in candidates])
    return {"candidates": candidates, **ranked, "repair": detection}


def summarise(cases: list[dict[str, Any]], archived_selection: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarise all 20 rankings with the fixed 15-pixel denominator."""
    accurate = dict.fromkeys(SCHEMES, 0)
    without_usable = dict.fromkeys(SCHEMES, 0)
    changed: dict[str, list[str]] = {scheme: [] for scheme in SCHEMES}
    candidate_count = 0
    case_rows = []
    for case in cases:
        candidate_count += len(case["ranking_inputs"])
        for scheme in SCHEMES:
            winner = case["winners"][scheme]
            if winner is not None:
                accurate[scheme] += winner["metrics"]["corner_max_error_px"] <= ACCURATE_PX
                entry = next(item for item in case["ranking_inputs"] if item["id"] == winner["id"])
                without_usable[scheme] += entry["usable_sites"] == 0
            archived_order = archived_selection[case["id"]]["orders"][scheme]
            current_order = case["orders"][scheme]
            if (current_order[0] if current_order else None) != (archived_order[0] if archived_order else None):
                changed[scheme].append(case["id"])
        case_rows.append({
            "id": case["id"],
            "candidate_count": len(case["ranking_inputs"]),
            "box_source": case["box_source"],
            "winners": {scheme: (case["winners"][scheme]["id"] if case["winners"][scheme] else None) for scheme in SCHEMES},
        })
    return {
        "case_count": len(cases),
        "candidate_id_count": candidate_count,
        "accurate_picks": accurate,
        "accurate_denominator": len(cases),
        "winners_without_usable_sites": without_usable,
        "winner_changed_from_archived": changed,
        "cases": case_rows,
    }


def load_archives(archive: Path, stripe_archive: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    marking, results, junctions, selection = _archive_inputs(archive, stripe_archive)
    cases_by_id = {case["id"]: case for case in marking["cases"]}
    records_by_id = {record["id"]: record for record in results["records"]}
    junctions_by_id = {record["id"]: record for record in junctions["records"]}
    selection_by_id = {record["id"]: record for record in selection["records"]}
    if (
        len(cases_by_id) != len(marking["cases"])
        or len(records_by_id) != len(results["records"])
        or len(junctions_by_id) != len(junctions["records"])
        or len(selection_by_id) != len(selection["records"])
    ):
        raise ValueError("marking and junction archives contain duplicate record IDs")
    if len(cases_by_id) != 20 or len(junctions_by_id) != 20 or len(selection_by_id) != 20:
        raise ValueError("marking and junction archives must contain exactly 20 cases")
    expected_ids = set(cases_by_id)
    if set(records_by_id) != expected_ids or set(junctions_by_id) != expected_ids or set(selection_by_id) != expected_ids:
        raise ValueError("marking and junction archives do not contain the same 20 cases")
    return marking, results, junctions, selection, cases_by_id


def validate_unaffected_cases(
    archive: Path,
    stripe_archive: Path,
    requested_ids: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the archived-helper gate for unaffected cases without writing output."""
    _, results, junctions, selection, cases_by_id = load_archives(archive, stripe_archive)
    case_ids = (
        [case_id for case_id in cases_by_id if case_id not in AFFECTED_IDS]
        if requested_ids is None
        else requested_ids
    )
    unknown = sorted(set(case_ids) - set(cases_by_id))
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Validation-only case IDs must be unique")
    affected = sorted(set(case_ids) & AFFECTED_IDS)
    if affected:
        raise ValueError(f"Validation-only smoke excludes repaired cases: {affected}")
    results_by_id = {record["id"]: record for record in results["records"]}
    junctions_by_id = {record["id"]: record for record in junctions["records"]}
    selection_by_id = {record["id"]: record for record in selection["records"]}
    validated = {}
    for case_id in case_ids:
        validated[case_id] = _historical_gate(
            cases_by_id[case_id], results_by_id[case_id], junctions_by_id[case_id], selection_by_id[case_id],
        )
    return validated


def build_repair(
    archive: Path,
    stripe_archive: Path,
    short_packet: Path,
    image_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Run the full 20-case repair after the 15-case gate passes."""
    _, results, junctions, selection, cases_by_id = load_archives(archive, stripe_archive)
    results_by_id = {record["id"]: record for record in results["records"]}
    junctions_by_id = {record["id"]: record for record in junctions["records"]}
    selection_by_id = {record["id"]: record for record in selection["records"]}
    unaffected: dict[str, dict[str, Any]] = {}
    for case_id in cases_by_id:
        if case_id in AFFECTED_IDS:
            continue
        historical = _historical_gate(
            cases_by_id[case_id], results_by_id[case_id], junctions_by_id[case_id], selection_by_id[case_id],
        )
        unaffected[case_id] = {**historical, "box_source": "historical_archive"}

    detections = load_affected_detections(short_packet, image_root, cases_by_id)
    repaired: dict[str, dict[str, Any]] = {}
    for case_id in cases_by_id:
        if case_id not in AFFECTED_IDS:
            repaired[case_id] = unaffected[case_id]
            continue
        corrected = _corrected_case(
            cases_by_id[case_id], results_by_id[case_id], selection_by_id[case_id], detections[case_id],
        )
        repaired[case_id] = {**corrected, "box_source": "exact_detection"}

    ordered_cases = [repaired[case["id"]] | {"id": case["id"]} for case in _ordered_cases(cases_by_id)]
    summary = summarise(ordered_cases, selection_by_id)
    if summary["case_count"] != 20 or summary["candidate_id_count"] != 682:
        raise ValueError("full marking repair must contain 20 cases and 682 eligible candidate IDs")
    result = {
        "schema": OUTPUT_SCHEMA,
        "development_data": True,
        "acceptance_evaluated": False,
        "ranking_schemes": list(SCHEMES),
        "provenance": {
            "marking_archive": archive.name,
            "marking_archive_sha256": sha256_file(archive),
            "stripe_archive": stripe_archive.name,
            "stripe_archive_sha256": sha256_file(stripe_archive),
            "detection_packet": short_packet.name,
            "detection_packet_sha256": sha256_file(short_packet),
            "score_cutoff_strictly_greater_than": SCORE_CUTOFF,
        },
        "summary": summary,
        "cases": [
            {
                "id": case["id"],
                "box_source": case["box_source"],
                "candidate_count": len(case["candidates"]),
                "measurements": case["measurements"],
                "ranking_inputs": case["ranking_inputs"],
                "orders": case["orders"],
                "winners": case["winners"],
                **({"repair": case["repair"]} if "repair" in case else {}),
            }
            for case in ordered_cases
        ],
    }
    atomic_write_json_gz(output, result)
    return result


def _ordered_cases(cases_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return cases in the frozen archive order rather than sorted ID order."""
    return list(cases_by_id.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--stripe-archive", type=Path, required=True)
    parser.add_argument("--short-detections", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--ids", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_only:
        validated = validate_unaffected_cases(args.archive, args.stripe_archive, args.ids)
        print(json.dumps({"validated_cases": list(validated), "candidate_ids": sum(len(item["candidates"]) for item in validated.values())}))
        return
    if args.ids:
        raise ValueError("--ids is available only with --validation-only; a repair must contain all 20 cases")
    required = {
        "--short-detections": args.short_detections,
        "--image-root": args.image_root,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"full repair requires {', '.join(missing)}")
    result = build_repair(
        args.archive,
        args.stripe_archive,
        args.short_detections,
        args.image_root,
        args.output,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
