"""Score the label-blind contact evidence freeze against ShuttleSet tables.

Manifest and evidence checks run before this module imports any ground-truth
loader.  The matching helpers are plain functions so the timing contract can
be tested without external fixtures.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from annotator.fps_constants import ScalingKind

EVIDENCE_SCHEMA = "contact-evidence-freeze/1"
MANIFEST_SCHEMA = "contact-evidence-manifest/1"
RESULTS_SCHEMA = "contact-evidence-score/1"
FIXTURE_SPECS = {
    "sset_01": (1, 25.0),
    "sset_15": (15, 25.0),
    "sset_21": (21, 30.0),
}
FIXTURE_RESOLUTION = [1920.0, 1080.0]
TOLERANCES_BASE30 = (5, 10, 15)
FORBIDDEN_KEY_TOKENS = ("gt", "label", "truth", "correct", "score")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VerifiedFreeze:
    """A checksum-verified evidence object and its manifest metadata."""

    manifest_path: Path
    evidence_path: Path
    manifest: dict[str, Any]
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ContactRow:
    """One frozen raw or filtered contact with a stable physical identity."""

    fixture: str
    span_id: int
    contact_index: int
    contact_frame: int
    current_half: str | None
    ankle_half: str | None
    valid_ankle_slots: int
    preceding_scene_distance_frames: int | None
    filtered: bool

    @property
    def identity(self) -> tuple[str, int, int]:
        """Return the immutable fixture/span/contact identity."""
        return self.fixture, self.span_id, self.contact_index


@dataclass(frozen=True)
class ContactMatch:
    """One greedy match between a GT stroke index and a frozen contact row."""

    rally_index: int
    gt_index: int
    gt_frame: int
    candidate: ContactRow
    offset_frames: int


def _forbidden_key_error(value: object, path: str = "evidence") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                return f"{path}: non-string key"
            lower = key.casefold()
            if any(token in lower for token in FORBIDDEN_KEY_TOKENS):
                return f"{path}.{key}: forbidden frozen key"
            error = _forbidden_key_error(child, f"{path}.{key}")
            if error is not None:
                return error
    elif isinstance(value, list):
        for index, child in enumerate(value):
            error = _forbidden_key_error(child, f"{path}[{index}]")
            if error is not None:
                return error
    return None


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            value = json.load(source)
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected an object with string keys")
    return value


def _normalise_half(value: object, name: str) -> str | None:
    if value is None:
        return None
    if value == "Top":
        return "Top"
    if value in {"Bot", "Bottom"}:
        return "Bot"
    raise ValueError(f"{name}: unexpected half {value!r}")


def _validate_relative_filename(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a relative filename")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts) or len(path.parts) != 1:
        raise ValueError(f"{name} must not contain a directory or parent traversal")
    return value


def _validate_input_manifest(value: object) -> None:
    if not isinstance(value, list):
        raise TypeError("manifest.inputs must be a list")
    expected_roles = {
        "shuttle_track",
        "pose_kps",
        "pose_bboxes",
        "pose_scores",
        "pose_kp_scores",
        "pose_ndet",
        "court_evidence",
        "court_keep_vote",
        "court_present",
        "annotation",
    }
    fixture_names: list[str] = []
    for fixture_row in value:
        if not isinstance(fixture_row, Mapping):
            raise TypeError("manifest input fixture row must be an object")
        fixture = fixture_row.get("fixture")
        if not isinstance(fixture, str):
            raise TypeError("manifest input fixture identity is malformed")
        fixture_names.append(fixture)
        files = fixture_row.get("files")
        if not isinstance(files, list):
            raise TypeError(f"manifest input files are missing for {fixture}")
        roles: set[str] = set()
        for file_row in files:
            if not isinstance(file_row, Mapping):
                raise TypeError("manifest input file row must be an object")
            role = file_row.get("role")
            if not isinstance(role, str) or role in roles:
                raise ValueError(f"manifest input role is duplicated or malformed: {role!r}")
            roles.add(role)
            _validate_relative_filename(file_row.get("filename"), f"manifest.{role}.filename")
            stage = file_row.get("stage")
            if not isinstance(stage, str) or not stage:
                raise ValueError(f"manifest.{role}.stage is malformed")
            digest = file_row.get("sha256")
            if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None:
                raise ValueError(f"manifest.{role}.sha256 is malformed")
            size = file_row.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"manifest.{role}.size_bytes is malformed")
        if roles != expected_roles:
            raise ValueError(f"manifest input roles differ for {fixture}: {roles}")
    if sorted(fixture_names) != sorted(FIXTURE_SPECS):
        raise ValueError("manifest input fixtures differ from the fixed set")


def _validate_contact_row(value: object, fixture: str, span_id: int, index: int) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{fixture} span {span_id} contact {index} is not an object")
    required = {
        "rally_id",
        "contact_frame",
        "proximity_ok",
        "wrist_near",
        "suppressed",
        "filtered",
        "nearest_wrist_slot",
        "current_half",
        "ankle_half",
        "valid_ankle_slots",
        "preceding_scene_distance_frames",
    }
    if set(value) != required:
        raise ValueError(f"{fixture} span {span_id} contact {index} fields differ")
    if value["rally_id"] != span_id or value["contact_frame"] < 0:
        raise ValueError(f"{fixture} span {span_id} contact {index} identity is malformed")
    for name in ("proximity_ok", "wrist_near", "suppressed"):
        if value[name] is not None and not isinstance(value[name], bool):
            raise ValueError(f"{fixture} span {span_id} contact {index}.{name} is malformed")
    if not isinstance(value["filtered"], bool):
        raise TypeError(f"{fixture} span {span_id} contact {index}.filtered is malformed")
    slot = value["nearest_wrist_slot"]
    if slot is not None and (isinstance(slot, bool) or slot not in {0, 1}):
        raise ValueError(f"{fixture} span {span_id} contact {index}.nearest_wrist_slot is malformed")
    _normalise_half(value["current_half"], "current_half")
    _normalise_half(value["ankle_half"], "ankle_half")
    count = value["valid_ankle_slots"]
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 2:
        raise ValueError(f"{fixture} span {span_id} contact {index}.valid_ankle_slots is malformed")
    distance = value["preceding_scene_distance_frames"]
    if distance is not None and (isinstance(distance, bool) or not isinstance(distance, int) or distance < 0):
        raise ValueError(f"{fixture} span {span_id} contact {index}.preceding_scene_distance_frames is malformed")


def _validate_evidence(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("evidence must be an object")
    if value.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("evidence schema differs")
    error = _forbidden_key_error(value)
    if error is not None:
        raise ValueError(error)
    fixtures = value.get("fixtures")
    if not isinstance(fixtures, list) or [row.get("fixture") for row in fixtures if isinstance(row, Mapping)] != sorted(FIXTURE_SPECS):
        raise ValueError("evidence fixture set differs from the fixed set")
    for fixture_row in fixtures:
        if not isinstance(fixture_row, Mapping):
            raise TypeError("evidence fixture row must be an object")
        fixture = fixture_row["fixture"]
        video_id, fps = FIXTURE_SPECS[fixture]
        if fixture_row.get("video_id") != video_id or fixture_row.get("fps") != fps:
            raise ValueError(f"evidence metadata differs for {fixture}")
        if fixture_row.get("resolution") != FIXTURE_RESOLUTION:
            raise ValueError(f"evidence resolution differs for {fixture}")
        frame_count = fixture_row.get("frame_count")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError(f"evidence frame_count is malformed for {fixture}")
        spans = fixture_row.get("spans")
        if not isinstance(spans, list):
            raise TypeError(f"evidence spans are missing for {fixture}")
        for span_index, span in enumerate(spans):
            if not isinstance(span, Mapping):
                raise TypeError(f"evidence span {span_index} is not an object")
            required = {
                "span_id",
                "start_frame",
                "end_frame",
                "raw_contact_count",
                "filtered_contact_count",
                "stored_striker_half",
                "current_striker_half",
                "geometry_striker_half",
                "stored_server_half",
                "current_server_half",
                "geometry_server_half",
                "contacts",
            }
            if set(span) != required or span["span_id"] != span_index:
                raise ValueError(f"evidence span fields differ for {fixture} span {span_index}")
            start = span["start_frame"]
            end = span["end_frame"]
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > frame_count
            ):
                raise ValueError(f"evidence span bounds are malformed for {fixture} span {span_index}")
            for field in (
                "stored_striker_half",
                "current_striker_half",
                "geometry_striker_half",
                "stored_server_half",
                "current_server_half",
                "geometry_server_half",
            ):
                _normalise_half(span[field], f"{fixture}.spans[{span_index}].{field}")
            contacts = span["contacts"]
            if not isinstance(contacts, list):
                raise TypeError(f"evidence contacts are malformed for {fixture} span {span_index}")
            if span["raw_contact_count"] != len(contacts):
                raise ValueError(f"evidence raw contact count differs for {fixture} span {span_index}")
            filtered_count = 0
            for contact_index, contact in enumerate(contacts):
                _validate_contact_row(contact, fixture, span_index, contact_index)
                if not start <= contact["contact_frame"] < end:
                    raise ValueError(f"contact lies outside {fixture} span {span_index}")
                filtered_count += int(contact["filtered"])
            if span["filtered_contact_count"] != filtered_count:
                raise ValueError(f"evidence filtered contact count differs for {fixture} span {span_index}")
    return value


def verify_freeze(manifest_path: Path) -> VerifiedFreeze:
    """Verify the manifest and evidence before any GT module is imported."""
    manifest_path = Path(manifest_path)
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema differs")
    if manifest.get("labels_read") is not False:
        raise ValueError("manifest does not preserve the label-blind boundary")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise ValueError("manifest source_commit is missing")
    fixture_set = manifest.get("fixture_set")
    if fixture_set != list(FIXTURE_SPECS):
        raise ValueError("manifest fixture set differs from the fixed set")
    if manifest.get("evidence_schema") != EVIDENCE_SCHEMA:
        raise ValueError("manifest evidence schema differs")
    filename = _validate_relative_filename(manifest.get("evidence_file"), "manifest.evidence_file")
    evidence_path = manifest_path.parent / filename
    expected_digest = manifest.get("evidence_sha256")
    if not isinstance(expected_digest, str) or HEX_SHA256.fullmatch(expected_digest) is None:
        raise ValueError("manifest evidence_sha256 is malformed")
    actual_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("evidence SHA-256 does not match the manifest")
    _validate_input_manifest(manifest.get("inputs"))
    evidence = _validate_evidence(_read_json_object(evidence_path))
    return VerifiedFreeze(manifest_path, evidence_path, manifest, evidence)


def scale_base30_frames(base30: float, fps: float) -> int:
    """Scale a base-30 frame tolerance with the repository's half-up rule."""
    if isinstance(base30, bool) or not math.isfinite(float(base30)) or base30 < 0:
        raise ValueError(f"base30 tolerance is malformed: {base30!r}")
    return int(ScalingKind.FRAME_COUNT.scale(float(base30), float(fps)))


def greedy_match(
    gt_frames: Sequence[int], candidate_frames: Sequence[int], tolerance: int
) -> list[tuple[int, int]]:
    """Greedily match closest one-to-one frame pairs within a tolerance."""
    if isinstance(tolerance, bool) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative integer")
    ranked = sorted(
        (abs(gt_frame - candidate_frame), gt_index, candidate_index)
        for gt_index, gt_frame in enumerate(gt_frames)
        for candidate_index, candidate_frame in enumerate(candidate_frames)
        if abs(gt_frame - candidate_frame) <= tolerance
    )
    matched: list[tuple[int, int]] = []
    claimed_gt: set[int] = set()
    claimed_candidates: set[int] = set()
    for _distance, gt_index, candidate_index in ranked:
        if gt_index in claimed_gt or candidate_index in claimed_candidates:
            continue
        claimed_gt.add(gt_index)
        claimed_candidates.add(candidate_index)
        matched.append((gt_index, candidate_index))
    return matched


def _contact_rows(evidence_fixture: Mapping[str, Any], *, filtered: bool) -> list[ContactRow]:
    fixture = str(evidence_fixture["fixture"])
    rows: list[ContactRow] = []
    for span in evidence_fixture["spans"]:
        span_id = int(span["span_id"])
        for contact_index, contact in enumerate(span["contacts"]):
            if filtered and not contact["filtered"]:
                continue
            rows.append(
                ContactRow(
                    fixture,
                    span_id,
                    contact_index,
                    int(contact["contact_frame"]),
                    _normalise_half(contact["current_half"], "current_half"),
                    _normalise_half(contact["ankle_half"], "ankle_half"),
                    int(contact["valid_ankle_slots"]),
                    (
                        None
                        if contact["preceding_scene_distance_frames"] is None
                        else int(contact["preceding_scene_distance_frames"])
                    ),
                    bool(contact["filtered"]),
                )
            )
    return rows


def _rally_frames(rally: object) -> tuple[int, ...]:
    if hasattr(rally, "stroke_frames"):
        values = getattr(rally, "stroke_frames")  # noqa: B009
    elif isinstance(rally, Mapping):
        values = rally.get("stroke_frames")
    else:
        values = None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("GT rally does not expose stroke_frames")
    return tuple(int(frame) for frame in values)


def _overlapping_span_ids(
    spans: Sequence[Mapping[str, Any]], gt_frames: Sequence[int]
) -> list[int]:
    first, last = gt_frames[0], gt_frames[-1]
    return [
        int(span["span_id"])
        for span in spans
        if int(span["start_frame"]) <= last and first < int(span["end_frame"])
    ]


def _match_variant(
    evidence_fixture: Mapping[str, Any],
    rallies: Sequence[object],
    fps: float,
    rows: Sequence[ContactRow],
    tolerance_base30: int,
) -> tuple[dict[str, Any], list[ContactMatch]]:
    spans = evidence_fixture["spans"]
    by_span: dict[int, list[ContactRow]] = {}
    for row in rows:
        by_span.setdefault(row.span_id, []).append(row)
    for span_rows in by_span.values():
        span_rows.sort(key=lambda row: (row.contact_frame, row.contact_index))
    tolerance = scale_base30_frames(tolerance_base30, fps)
    matches: list[ContactMatch] = []
    candidate_occurrences = 0
    count_gate_pass = 0
    any_near_serve = 0
    for rally_index, rally in enumerate(rallies):
        gt_frames = _rally_frames(rally)
        if not gt_frames:
            raise ValueError("GT rally has no stroke frames")
        span_ids = _overlapping_span_ids(spans, gt_frames)
        candidate_rows = [row for span_id in span_ids for row in by_span.get(span_id, [])]
        candidate_rows.sort(key=lambda row: (row.span_id, row.contact_frame, row.contact_index))
        candidate_occurrences += len(candidate_rows)
        count_gate_pass += int(len(candidate_rows) == len(gt_frames))
        if any(abs(row.contact_frame - gt_frames[0]) <= tolerance for row in candidate_rows):
            any_near_serve += 1
        for gt_index, candidate_index in greedy_match(
            gt_frames,
            [row.contact_frame for row in candidate_rows],
            tolerance,
        ):
            candidate = candidate_rows[candidate_index]
            matches.append(
                ContactMatch(
                    rally_index,
                    gt_index,
                    gt_frames[gt_index],
                    candidate,
                    candidate.contact_frame - gt_frames[gt_index],
                )
            )
    total = sum(len(_rally_frames(rally)) for rally in rallies)
    serve_total = len(rallies)
    non_serve_total = total - serve_total
    serve_matched = sum(match.gt_index == 0 for match in matches)
    non_serve_matched = sum(match.gt_index > 0 for match in matches)
    matched = len(matches)
    candidate_count = len(rows)
    matched_candidate_count = len({match.candidate.identity for match in matches})
    overall = {
        "matched": matched,
        "total": total,
        "recall": matched / total if total else None,
        "precision": matched / candidate_occurrences if candidate_occurrences else None,
        "physical_precision": matched_candidate_count / candidate_count if candidate_count else None,
    }
    metrics = {
        "fixture": str(evidence_fixture["fixture"]),
        "tolerance_base30": tolerance_base30,
        "tolerance_frames": tolerance,
        "serve": {
            "matched": serve_matched,
            "total": serve_total,
            "recall": serve_matched / serve_total if serve_total else None,
        },
        "non_serve": {
            "matched": non_serve_matched,
            "total": non_serve_total,
            "recall": non_serve_matched / non_serve_total if non_serve_total else None,
        },
        "overall": overall,
        "any_candidate_near_serve_count": any_near_serve,
        "any_candidate_near_serve_total": serve_total,
        "any_candidate_near_serve_recall": any_near_serve / serve_total if serve_total else None,
        "candidate_count": candidate_count,
        "candidate_occurrence_count": candidate_occurrences,
        "matched_candidate_count": matched_candidate_count,
        "noise_count": candidate_count - matched_candidate_count,
        "noise_occurrence_count": candidate_occurrences - matched,
        "count_gate_pass": count_gate_pass,
        "count_gate_total": len(rallies),
    }
    return metrics, matches


def _aggregate_metrics(metrics_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-fixture metric counts without averaging fixture recalls."""
    if not metrics_rows:
        raise ValueError("cannot aggregate an empty metric set")
    first = metrics_rows[0]
    serve_matched = sum(row["serve"]["matched"] for row in metrics_rows)
    serve_total = sum(row["serve"]["total"] for row in metrics_rows)
    non_matched = sum(row["non_serve"]["matched"] for row in metrics_rows)
    non_total = sum(row["non_serve"]["total"] for row in metrics_rows)
    matched = sum(row["overall"]["matched"] for row in metrics_rows)
    total = sum(row["overall"]["total"] for row in metrics_rows)
    candidate_count = sum(row["candidate_count"] for row in metrics_rows)
    candidate_occurrences = sum(row["candidate_occurrence_count"] for row in metrics_rows)
    matched_candidate_count = sum(row["matched_candidate_count"] for row in metrics_rows)
    any_near = sum(row["any_candidate_near_serve_count"] for row in metrics_rows)
    any_total = sum(row["any_candidate_near_serve_total"] for row in metrics_rows)
    gate_pass = sum(row["count_gate_pass"] for row in metrics_rows)
    gate_total = sum(row["count_gate_total"] for row in metrics_rows)
    return {
        "tolerance_base30": first["tolerance_base30"],
        "tolerance_frames_by_fixture": {
            str(row["fixture"]): row["tolerance_frames"] for row in metrics_rows
        },
        "serve": {
            "matched": serve_matched,
            "total": serve_total,
            "recall": serve_matched / serve_total if serve_total else None,
        },
        "non_serve": {
            "matched": non_matched,
            "total": non_total,
            "recall": non_matched / non_total if non_total else None,
        },
        "overall": {
            "matched": matched,
            "total": total,
            "recall": matched / total if total else None,
            "precision": matched / candidate_occurrences if candidate_occurrences else None,
            "physical_precision": matched_candidate_count / candidate_count if candidate_count else None,
        },
        "any_candidate_near_serve_count": any_near,
        "any_candidate_near_serve_total": any_total,
        "any_candidate_near_serve_recall": any_near / any_total if any_total else None,
        "candidate_count": candidate_count,
        "candidate_occurrence_count": candidate_occurrences,
        "matched_candidate_count": matched_candidate_count,
        "noise_count": candidate_count - matched_candidate_count,
        "noise_occurrence_count": candidate_occurrences - matched,
        "count_gate_pass": gate_pass,
        "count_gate_total": gate_total,
    }


def _half_metrics(
    matches: Sequence[ContactMatch], side_by_frame: Mapping[tuple[str, int], str | None],
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    grouped_matches = {
        "all": list(matches),
        "no_players": [match for match in matches if match.candidate.valid_ankle_slots == 0],
        "one_player": [match for match in matches if match.candidate.valid_ankle_slots == 1],
        "two_players": [match for match in matches if match.candidate.valid_ankle_slots == 2],
        "within_15_base30_of_scene_start": [
            match
            for match in matches
            if match.candidate.preceding_scene_distance_frames is not None
            and match.candidate.preceding_scene_distance_frames
            <= scale_base30_frames(15, FIXTURE_SPECS[match.candidate.fixture][1])
        ],
    }
    result: dict[str, dict[str, dict[str, int | float | None]]] = {}
    for group_name, group in grouped_matches.items():
        result[group_name] = {}
        for field_name in ("current_half", "ankle_half"):
            matched = 0
            available = 0
            correct = 0
            for match in group:
                target = side_by_frame.get((match.candidate.fixture, match.gt_frame))
                if target is None:
                    continue
                matched += 1
                predicted = getattr(match.candidate, field_name)
                if predicted is None:
                    continue
                available += 1
                correct += int(predicted == target)
            result[group_name][field_name.removesuffix("_half")] = {
                "matched_contacts": matched,
                "available_predictions": available,
                "correct": correct,
                "accuracy": correct / available if available else None,
                "coverage": available / matched if matched else None,
            }
    return result


def _fit_metrics(
    evidence_fixture: Mapping[str, Any],
    rallies: Sequence[object],
    classifications: Sequence[tuple[object, int | None]],
    side_by_frame: Mapping[tuple[str, int], str | None],
    striker_field_name: str,
    server_field_name: str,
) -> dict[str, dict[str, int | float | None]]:
    fixture = str(evidence_fixture["fixture"])
    spans = evidence_fixture["spans"]
    result: dict[str, dict[str, int | float | None]] = {}
    for role, frame_index in (("final_striker", -1), ("server", 0)):
        field_name = striker_field_name if role == "final_striker" else server_field_name
        for variant in ("current", "geometry"):
            total = len(rallies)
            available = 0
            correct = 0
            covered = 0
            for rally, (boundary, span_id) in zip(rallies, classifications, strict=True):
                frames = _rally_frames(rally)
                if getattr(boundary, "value", boundary) != "covered" or span_id is None:
                    continue
                covered += 1
                span = spans[span_id]
                predicted = _normalise_half(span[field_name.format(variant=variant)], field_name)
                if predicted is None:
                    continue
                available += 1
                target = side_by_frame.get((fixture, frames[frame_index]))
                correct += int(target is not None and predicted == target)
            result[f"{role}_{variant}"] = {
                "correct": correct,
                "available": available,
                "covered_rallies": covered,
                "total_rallies": total,
                "accuracy": correct / available if available else None,
                "coverage": available / total if total else None,
            }
    return result


def _score_verified(verified: VerifiedFreeze) -> dict[str, Any]:
    """Import and load GT only after ``verify_freeze`` has completed."""
    from annotator.calibration.gt_scoring import load_gt_tables
    from annotator.calibration.scoring import classify_all, load_gt_rallies

    master, _homography, _court_info, _resolution = load_gt_tables()
    gt_by_fixture: dict[str, list[object]] = {}
    side_by_frame: dict[tuple[str, int], str | None] = {}
    classifications: dict[str, Sequence[tuple[object, int | None]]] = {}
    for fixture, (video_id, fps) in FIXTURE_SPECS.items():
        rallies = load_gt_rallies(master, video_id)
        gt_by_fixture[fixture] = rallies
        rows = master[master["vid"] == video_id]
        if "player_side" not in rows:
            raise ValueError("ShuttleSet shots_master is missing player_side")
        for frame, side in zip(rows["frame_num"], rows["player_side"], strict=True):
            side_by_frame[(fixture, int(frame))] = _normalise_half(str(side), "player_side")
        fixture_evidence = next(row for row in verified.evidence["fixtures"] if row["fixture"] == fixture)
        classifications[fixture] = classify_all(
            [(int(span["start_frame"]), int(span["end_frame"])) for span in fixture_evidence["spans"]],
            rallies,
        )
    total_rallies = sum(len(rallies) for rallies in gt_by_fixture.values())
    if total_rallies != 292:
        raise ValueError(f"expected 292 ShuttleSet rallies, found {total_rallies}")

    output: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        "source_commit": verified.manifest["source_commit"],
        "evidence_sha256": verified.manifest["evidence_sha256"],
        "fixture_set": list(FIXTURE_SPECS),
        "ground_truth_rallies": total_rallies,
        "tolerances_base30": list(TOLERANCES_BASE30),
        "fixtures": {},
        "overall": {"raw": {}, "filtered": {}},
        "half_attribution": {},
        "rally_fits": {},
    }
    all_variant_matches: dict[str, dict[int, list[ContactMatch]]] = {"raw": {}, "filtered": {}}
    all_variant_metrics: dict[str, dict[int, list[dict[str, Any]]]] = {"raw": {}, "filtered": {}}
    for fixture, (_video_id, fps) in FIXTURE_SPECS.items():
        evidence_fixture = next(row for row in verified.evidence["fixtures"] if row["fixture"] == fixture)
        rallies = gt_by_fixture[fixture]
        fixture_output: dict[str, Any] = {"ground_truth_rallies": len(rallies), "raw": {}, "filtered": {}}
        for variant, filtered in (("raw", False), ("filtered", True)):
            rows = _contact_rows(evidence_fixture, filtered=filtered)
            for tolerance_base30 in TOLERANCES_BASE30:
                metrics, matches = _match_variant(
                    evidence_fixture,
                    rallies,
                    fps,
                    rows,
                    tolerance_base30,
                )
                fixture_output[variant][str(tolerance_base30)] = metrics
                all_variant_metrics[variant].setdefault(tolerance_base30, []).append(metrics)
                all_variant_matches[variant].setdefault(tolerance_base30, []).extend(matches)
        output["fixtures"][fixture] = fixture_output
    for variant in ("raw", "filtered"):
        for tolerance_base30 in TOLERANCES_BASE30:
            output["overall"][variant][str(tolerance_base30)] = _aggregate_metrics(
                all_variant_metrics[variant][tolerance_base30]
            )
            matches = all_variant_matches[variant][tolerance_base30]
            output["half_attribution"].setdefault(variant, {})[str(tolerance_base30)] = _half_metrics(
                matches,
                side_by_frame,
            )
    filtered_fits: dict[str, dict[str, Any]] = {}
    for fixture in FIXTURE_SPECS:
        evidence_fixture = next(row for row in verified.evidence["fixtures"] if row["fixture"] == fixture)
        filtered_fits[fixture] = _fit_metrics(
            evidence_fixture,
            gt_by_fixture[fixture],
            classifications[fixture],
            side_by_frame,
            "{variant}_striker_half",
            "{variant}_server_half",
        )
    output["rally_fits"] = filtered_fits
    return output


def score(manifest_path: Path) -> dict[str, Any]:
    """Verify a freeze, then load GT and score it."""
    verified = verify_freeze(manifest_path)
    return _score_verified(verified)


def write_results(path: Path, payload: Mapping[str, object]) -> None:
    """Write deterministic plain JSON, or deterministic gzip JSON for ``.gz``."""
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.name.endswith(".gz"):
        with destination.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as zipped:
            zipped.write(encoded)
    else:
        destination.write_bytes(encoded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the scorer CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify and score one frozen evidence directory."""
    arguments = parse_args(argv)
    results = score(arguments.manifest)
    write_results(arguments.output, results)
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
