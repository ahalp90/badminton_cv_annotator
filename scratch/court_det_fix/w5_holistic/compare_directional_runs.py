"""Validate and compare the four directional W5 visibility-floor runs.

The comparator is deliberately local. It reads completed run directories, validates
their shared contract and writes a small JSON/Markdown comparison plus contact sheets
from the crops already rendered by ``render_gallery.py``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ARM_IDS = ("0_0", "3_3", "4_3", "5_3")
ARM_FLOORS = ((0, 0), (3, 3), (4, 3), (5, 3))
ARM_LABELS = {
    "0_0": "(0,0)",
    "3_3": "(3,3)",
    "4_3": "(4,3)",
    "5_3": "(5,3)",
}

REGRESSION_CASES = (
    "gxBQ_window_00_frame_0",
    "gxBQ_window_00_frame_5",
    "am2_window_00_frame_150",
    "am2_window_01_frame_28019",
    "am3_window_00_frame_0",
    "shuttleset_03_scene_0017",
    "shuttleset_03_scene_0019",
    "shuttleset_03_scene_0016",
    "shuttleset_21_scene_0020",
)
UNUSED_CASES = (
    "gxBQ_window_00_frame_689",
    "gxBQ_window_01_frame_5111",
    "gxBQ_window_02_frame_5766",
    "gxBQ_window_03_frame_77876",
    "gxBQ_window_04_frame_86088",
    "yellow_short_frame_14",
    "letterboxed_short_frame_45",
    "centre_short_frame_36",
    "am1_window_00_frame_54",
    "am3_window_01_frame_10514",
    "am4_window_00_frame_0",
    "am4_window_01_frame_13782",
    "shuttleset_03_scene_0029",
    "shuttleset_03_scene_0034",
    "shuttleset_03_scene_0038",
    "shuttleset_21_scene_0000",
    "shuttleset_21_scene_0010",
    "shuttleset_21_scene_0039",
)
EXPECTED_CASES = REGRESSION_CASES + UNUSED_CASES
CONTACT_CASES = UNUSED_CASES + (
    "gxBQ_window_00_frame_0",
    "gxBQ_window_00_frame_5",
    "am2_window_01_frame_28019",
)

VISIBILITY_COLUMNS = {
    "lengthwise": "first six projected court-template pieces (x-family): sidelines plus split centre",
    "cross_court": "second six projected court-template pieces (y-family): baselines and service lines",
}
ALLOWED_WINNER_SOURCES = {"G0", "G1", "line_template"}

ADMISSION_FIELDS = (
    "min_visible_lengthwise",
    "min_visible_cross_court",
    "visibility_columns",
    "hypotheses_before",
    "hypotheses_after",
    "hypotheses_rejected",
    "floor_zero_scanned_for_proposal_cap",
    "floor_zero_selected_count",
    "scanned_for_proposal_cap",
    "removed_from_floor_zero_count",
    "refilled_proposal_count",
    "newly_admitted_indices",
    "newly_admitted_proposal_ids",
    "removed_from_floor_zero_indices",
    "removed_from_floor_zero_proposal_ids",
)
REQUIRED_PER_VIEW_FIELDS = {
    "case_id",
    "view_status",
    "G0_count",
    "G1_count",
    "line_template_count",
}


class ComparisonError(ValueError):
    """A run packet fails the directional-comparison contract."""


@dataclass(frozen=True)
class ArmRun:
    """Validated inputs for one visibility-floor arm."""

    arm_id: str
    floor: tuple[int, int]
    path: Path
    manifest: dict[str, Any]
    per_view: list[dict[str, str]]
    rankings: dict[str, Any]
    reviews: dict[str, Any]


def _fail(message: str) -> None:
    raise ComparisonError(message)


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        _fail(f"{context}: missing {key}")
    return mapping[key]


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{context}: expected an integer")
    return value


def _pair(mapping: dict[str, Any], context: str) -> tuple[int, int]:
    return (
        _integer(
            _required(mapping, "min_visible_lengthwise", context),
            f"{context}.min_visible_lengthwise",
        ),
        _integer(
            _required(mapping, "min_visible_cross_court", context),
            f"{context}.min_visible_cross_court",
        ),
    )


def _load_json(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{context}: missing {path.name}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"{context}: cannot read {path.name}: {error}")
    if not isinstance(value, dict):
        _fail(f"{context}: {path.name} must contain an object")
    return value


def _read_per_view(path: Path, expected_cases: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        _fail(f"{path.parent.name}: missing per_view.csv")
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            if not REQUIRED_PER_VIEW_FIELDS <= fields:
                missing = sorted(REQUIRED_PER_VIEW_FIELDS - fields)
                _fail(f"{path.parent.name}: per_view.csv missing columns {missing}")
            rows = list(reader)
    except OSError as error:
        _fail(f"{path.parent.name}: cannot read per_view.csv: {error}")
    actual_cases = tuple(row.get("case_id", "") for row in rows)
    if actual_cases != expected_cases:
        _fail(
            f"{path.parent.name}: per_view case order differs from the 27-case contract"
        )
    return rows


def _validate_columns(value: Any, context: str) -> None:
    if value != VISIBILITY_COLUMNS:
        _fail(
            f"{context}: visibility descriptions do not name six lengthwise and six cross-court pieces"
        )


def _csv_integer(row: dict[str, str], field: str, context: str) -> int:
    value = row.get(field)
    if value is None or value == "":
        _fail(f"{context}: per_view is missing {field}")
    try:
        parsed = int(value)
    except ValueError:
        _fail(f"{context}: per_view.{field} is not an integer")
    return parsed


def _validate_list(value: Any, context: str, *, item_type: type) -> list[Any]:
    if not isinstance(value, list) or not all(
        isinstance(item, item_type)
        and (item_type is not int or not isinstance(item, bool))
        for item in value
    ):
        _fail(f"{context}: expected a list of {item_type.__name__} values")
    return value


def _validate_contamination(
    manifest: dict[str, Any], cases: tuple[str, ...], context: str
) -> None:
    source_stage = _required(manifest, "source_stage", context)
    if not isinstance(source_stage, dict):
        _fail(f"{context}.source_stage: expected an object")
    for field in (
        "automatic_path_free_of_reference_fields",
        "preflight_path_free_of_reference_fields",
    ):
        if _required(source_stage, field, context) is not True:
            _fail(f"{context}.source_stage.{field}: contamination check did not pass")
    checks = _required(source_stage, "full_run_contamination_checks", context)
    if not isinstance(checks, dict) or tuple(checks) != cases:
        _fail(
            f"{context}: contamination checks do not cover the exact ordered case list"
        )
    for case_id in cases:
        check = checks[case_id]
        if (
            not isinstance(check, dict)
            or check.get("match") is not True
            or check.get("fields") != []
        ):
            _fail(f"{context}: contamination check failed for {case_id}")


def _validate_admission(
    source: dict[str, Any],
    manifest_admission: dict[str, Any],
    floor: tuple[int, int],
    case_id: str,
    arm_id: str,
) -> dict[str, Any]:
    context = f"{arm_id}/{case_id}/line_template"
    if source.get("name") != "line_template":
        _fail(f"{context}: source name is not line_template")
    status = source.get("status")
    if status not in {"generated", "empty"}:
        _fail(f"{context}: unsupported or missing source status")
    contamination = _required(source, "contamination_check", context)
    if not isinstance(contamination, dict):
        _fail(f"{context}: contamination_check must be an object")
    if (
        contamination.get("references_loaded") is not False
        or contamination.get("prior_controls_loaded") is not False
    ):
        _fail(
            f"{context}: line-template source is contaminated by reference/control data"
        )
    settings = _required(source, "settings", context)
    if not isinstance(settings, dict) or _pair(settings, context) != floor:
        _fail(f"{context}: line-template settings do not record their arm floor")
    _validate_columns(
        _required(settings, "visibility_columns", context), f"{context}.settings"
    )
    generation = _required(source, "generation", context)
    if not isinstance(generation, dict):
        _fail(f"{context}: generation must be an object")
    admission = _required(generation, "visibility_admission", context)
    if not isinstance(admission, dict):
        _fail(f"{context}: visibility_admission must be an object")
    if set(admission) < set(ADMISSION_FIELDS):
        missing = sorted(set(ADMISSION_FIELDS) - set(admission))
        _fail(f"{context}: visibility_admission missing {missing}")
    if _pair(admission, context) != floor:
        _fail(f"{context}: visibility floor does not match its arm")
    _validate_columns(
        admission["visibility_columns"], f"{context}.visibility_admission"
    )
    if manifest_admission != admission:
        _fail(
            f"{context}: manifest line_template_admission disagrees with line-template source"
        )

    before = _integer(admission["hypotheses_before"], f"{context}.hypotheses_before")
    after = _integer(admission["hypotheses_after"], f"{context}.hypotheses_after")
    rejected = _integer(
        admission["hypotheses_rejected"], f"{context}.hypotheses_rejected"
    )
    if min(before, after, rejected) < 0 or before != after + rejected:
        _fail(f"{context}: visibility hypothesis counts are inconsistent")
    zero_scanned = _integer(
        admission["floor_zero_scanned_for_proposal_cap"],
        f"{context}.floor_zero_scanned_for_proposal_cap",
    )
    zero_selected = _integer(
        admission["floor_zero_selected_count"], f"{context}.floor_zero_selected_count"
    )
    scanned = _integer(
        admission["scanned_for_proposal_cap"], f"{context}.scanned_for_proposal_cap"
    )
    removed_count = _integer(
        admission["removed_from_floor_zero_count"],
        f"{context}.removed_from_floor_zero_count",
    )
    refill_count = _integer(
        admission["refilled_proposal_count"], f"{context}.refilled_proposal_count"
    )
    if min(zero_scanned, zero_selected, scanned, removed_count, refill_count) < 0:
        _fail(f"{context}: admission counts must be non-negative")
    if zero_selected > zero_scanned or scanned < 0:
        _fail(f"{context}: proposal-cap scan counts are inconsistent")

    new_indices = _validate_list(
        admission["newly_admitted_indices"],
        f"{context}.newly_admitted_indices",
        item_type=int,
    )
    new_ids = _validate_list(
        admission["newly_admitted_proposal_ids"],
        f"{context}.newly_admitted_proposal_ids",
        item_type=str,
    )
    removed_indices = _validate_list(
        admission["removed_from_floor_zero_indices"],
        f"{context}.removed_from_floor_zero_indices",
        item_type=int,
    )
    removed_ids = _validate_list(
        admission["removed_from_floor_zero_proposal_ids"],
        f"{context}.removed_from_floor_zero_proposal_ids",
        item_type=str,
    )
    if len(new_indices) != refill_count or len(new_ids) != refill_count:
        _fail(f"{context}: refill list lengths do not match refilled_proposal_count")
    if len(removed_indices) != removed_count or len(removed_ids) != removed_count:
        _fail(
            f"{context}: removal list lengths do not match removed_from_floor_zero_count"
        )
    if len(set(new_indices)) != len(new_indices) or len(set(removed_indices)) != len(
        removed_indices
    ):
        _fail(f"{context}: admission index lists contain duplicates")
    if len(set(new_ids)) != len(new_ids) or len(set(removed_ids)) != len(removed_ids):
        _fail(f"{context}: admission proposal ID lists contain duplicates")
    if set(new_ids) & set(removed_ids):
        _fail(f"{context}: a proposal appears in both refill and removal lists")
    if any(index < 0 or index >= before for index in new_indices + removed_indices):
        _fail(f"{context}: admission index is outside the hypothesis population")

    proposal_count = _integer(
        _required(source, "proposal_count", context), f"{context}.proposal_count"
    )
    if proposal_count < 0 or proposal_count > 256:
        _fail(f"{context}: proposal_count is outside the 256-entry cap")
    combined = generation.get("combined_admission_hypotheses")
    if status == "generated":
        caps = _required(source, "caps", context)
        if not isinstance(caps, dict) or not isinstance(caps.get("256"), dict):
            _fail(f"{context}: missing caps['256'] line-template pool record")
        cap_256 = caps["256"]
        cap_count = _integer(
            _required(cap_256, "count", f"{context}.caps['256']"),
            f"{context}.caps['256'].count",
        )
        cap_ids = _validate_list(
            _required(cap_256, "proposal_ids", f"{context}.caps['256']"),
            f"{context}.caps['256'].proposal_ids",
            item_type=str,
        )
        if cap_count != proposal_count or len(cap_ids) != cap_count:
            _fail(f"{context}: final proposal_count and caps['256'] count disagree")
        if len(set(cap_ids)) != len(cap_ids):
            _fail(f"{context}: caps['256'].proposal_ids contain duplicates")
        combined = _integer(
            combined, f"{context}.generation.combined_admission_hypotheses"
        )
        if combined < 0 or combined > before:
            _fail(
                f"{context}: combined admission count is outside the hypothesis population"
            )
        if combined >= 256 and cap_count != 256:
            _fail(
                f"{context}: combined admission has at least 256 hypotheses but the final pool is short"
            )
    else:
        if combined is not None and combined != 0:
            _fail(f"{context}: empty source has a non-zero combined admission count")
        combined = 0
        cap_count = 0
        cap_ids = []
        if proposal_count != 0:
            _fail(f"{context}: empty source has a non-zero proposal count")
        if any(
            admission[field] != 0
            for field in (
                "hypotheses_before",
                "hypotheses_after",
                "hypotheses_rejected",
                "floor_zero_scanned_for_proposal_cap",
                "floor_zero_selected_count",
                "scanned_for_proposal_cap",
                "removed_from_floor_zero_count",
                "refilled_proposal_count",
            )
        ):
            _fail(f"{context}: empty source has non-zero admission counts")
        if any(
            admission[field]
            for field in (
                "newly_admitted_indices",
                "newly_admitted_proposal_ids",
                "removed_from_floor_zero_indices",
                "removed_from_floor_zero_proposal_ids",
            )
        ):
            _fail(f"{context}: empty source has admission proposal IDs")

    if floor == (0, 0):
        if rejected != 0 or after != before or removed_count != 0 or refill_count != 0:
            _fail(
                f"{context}: (0,0) must have no visibility rejection, removal or refill"
            )
        if new_indices or new_ids or removed_indices or removed_ids:
            _fail(f"{context}: (0,0) must have empty refill/removal ID lists")
        if zero_selected != proposal_count or scanned != zero_scanned:
            _fail(
                f"{context}: (0,0) final pool does not match the recorded floor-zero selection"
            )

    return {
        "status": status,
        "proposal_count": proposal_count,
        "cap_256_count": cap_count,
        "caps": {"256": {"count": cap_count, "proposal_ids": cap_ids}},
        "combined_admission_hypotheses": combined,
        "hypotheses_before": before,
        "hypotheses_after": after,
        "hypotheses_rejected": rejected,
        "floor_zero_scanned_for_proposal_cap": zero_scanned,
        "floor_zero_selected_count": zero_selected,
        "scanned_for_proposal_cap": scanned,
        "refilled_proposal_count": refill_count,
        "removed_from_floor_zero_count": removed_count,
        "newly_admitted_proposal_ids": new_ids,
        "removed_from_floor_zero_proposal_ids": removed_ids,
    }


def _stable_line_template_source(source: dict[str, Any]) -> dict[str, Any]:
    """Keep every line-template source field except floor-dependent accounting."""
    value = copy.deepcopy(source)
    settings = value.get("settings")
    if isinstance(settings, dict):
        settings.pop("min_visible_lengthwise", None)
        settings.pop("min_visible_cross_court", None)
        settings.pop("visibility_columns", None)
    generation = value.get("generation")
    if isinstance(generation, dict):
        generation.pop("visibility_admission", None)
        for field in (
            "combined_admission_hypotheses",
            "scanned_for_proposal_cap",
            "full_w5_gate_count",
            "elapsed_seconds",
        ):
            generation.pop(field, None)
    value.pop("caps", None)
    value.pop("proposal_count", None)
    return value


def _validate_cross_arm_admission(
    arms: tuple[ArmRun, ...],
    admissions: dict[str, dict[str, dict[str, Any]]],
) -> None:
    for case_id in EXPECTED_CASES:
        baseline_arm = arms[0]
        baseline_source = _stable_line_template_source(
            baseline_arm.manifest["line_template_sources"][case_id]
        )
        baseline = admissions[baseline_arm.arm_id][case_id]
        baseline_cap_ids = baseline["caps"]["256"]["proposal_ids"]
        baseline_cap_set = set(baseline_cap_ids)
        for arm in arms[1:]:
            source = _stable_line_template_source(
                arm.manifest["line_template_sources"][case_id]
            )
            if source != baseline_source:
                _fail(
                    f"{arm.arm_id}/{case_id}: stable line-template source fields differ"
                )
            current = admissions[arm.arm_id][case_id]
            for field in (
                "hypotheses_before",
                "floor_zero_scanned_for_proposal_cap",
                "floor_zero_selected_count",
            ):
                if current[field] != baseline[field]:
                    _fail(
                        f"{arm.arm_id}/{case_id}: {field} differs from the (0,0) baseline"
                    )
            current_cap_ids = current["caps"]["256"]["proposal_ids"]
            current_cap_set = set(current_cap_ids)
            expected_new = [
                proposal_id
                for proposal_id in current_cap_ids
                if proposal_id not in baseline_cap_set
            ]
            expected_removed = [
                proposal_id
                for proposal_id in baseline_cap_ids
                if proposal_id not in current_cap_set
            ]
            if current["newly_admitted_proposal_ids"] != expected_new:
                _fail(
                    f"{arm.arm_id}/{case_id}: refill IDs do not match gated cap order"
                )
            if current["removed_from_floor_zero_proposal_ids"] != expected_removed:
                _fail(
                    f"{arm.arm_id}/{case_id}: removal IDs do not match baseline cap order"
                )


def _validate_case_ordered_mapping(
    mapping: Any, cases: tuple[str, ...], context: str
) -> None:
    if not isinstance(mapping, dict) or tuple(mapping) != cases:
        _fail(f"{context}: case keys are not the exact ordered 27-case list")


def _validate_origins(
    rankings: dict[str, Any],
    reviews: dict[str, Any],
    cases: tuple[str, ...],
    arm_id: str,
) -> dict[str, dict[str, Any]]:
    ranking_cases = _required(rankings, "cases", f"{arm_id}/rankings")
    _validate_case_ordered_mapping(ranking_cases, cases, f"{arm_id}/rankings.cases")
    _validate_case_ordered_mapping(reviews, cases, f"{arm_id}/review_candidates")
    result: dict[str, dict[str, Any]] = {}
    for case_id in cases:
        ranking_case = ranking_cases[case_id]
        if not isinstance(ranking_case, dict):
            _fail(f"{arm_id}/{case_id}/rankings: expected an object")
        ranking_c = _required(ranking_case, "C", f"{arm_id}/{case_id}/rankings")
        if not isinstance(ranking_c, dict):
            _fail(f"{arm_id}/{case_id}/rankings.C: expected an object")
        status = ranking_c.get("status")
        if not isinstance(status, str) or not status:
            _fail(f"{arm_id}/{case_id}: C ranking has no unambiguous status")
        selected = ranking_c.get("selected_origin_key")
        if not isinstance(selected, str) or not selected or ":" not in selected:
            _fail(
                f"{arm_id}/{case_id}: C selected winner is not a source-qualified origin_key"
            )
        review = reviews[case_id]
        if not isinstance(review, dict):
            _fail(f"{arm_id}/{case_id}/review_candidates: expected an object")
        review_selected = _required(
            review, "selected", f"{arm_id}/{case_id}/review_candidates"
        )
        if (
            not isinstance(review_selected, dict)
            or review_selected.get("C") != selected
        ):
            _fail(f"{arm_id}/{case_id}: review and ranking C winners disagree")
        candidates = _required(
            review, "candidates", f"{arm_id}/{case_id}/review_candidates"
        )
        if not isinstance(candidates, dict) or selected not in candidates:
            _fail(f"{arm_id}/{case_id}: C winner is absent from review_candidates")
        candidate = candidates[selected]
        if not isinstance(candidate, dict) or candidate.get("origin_key") != selected:
            _fail(f"{arm_id}/{case_id}: C winner does not join by its origin_key")
        candidate_id = candidate.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id == selected
        ):
            _fail(
                f"{arm_id}/{case_id}: C winner appears to be joined by a raw candidate ID"
            )
        source = candidate.get("source")
        if (
            not isinstance(source, str)
            or source not in ALLOWED_WINNER_SOURCES
            or selected != f"{source}:{candidate_id}"
        ):
            _fail(
                f"{arm_id}/{case_id}: C winner source or candidate-ID join is invalid"
            )
        result[case_id] = {
            "origin_key": selected,
            "candidate_id": candidate_id,
            "source": source,
            "status": status,
        }
    return result


def _scrub_manifest_for_arm_comparison(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove only identity and visibility-floor records from stable manifest data."""
    value = copy.deepcopy(manifest)
    value.pop("run_id", None)
    value.pop("min_visible_lengthwise", None)
    value.pop("min_visible_cross_court", None)
    value.pop("visibility_columns", None)
    value.pop("line_template_sources", None)
    value.pop("line_template_admission", None)
    value.pop("candidate_identity", None)
    global_parameters = value.get("global_parameters")
    if isinstance(global_parameters, dict):
        global_parameters.pop("workers", None)
        global_parameters.pop("min_visible_lengthwise", None)
        global_parameters.pop("min_visible_cross_court", None)
        global_parameters.pop("visibility_columns", None)
    for provenance in value.get("view_provenance", {}).values():
        if isinstance(provenance, dict):
            provenance.pop("line_template_source", None)
    return value


def _validate_arm(arm_id: str, path: Path, floor: tuple[int, int]) -> ArmRun:
    path = path.resolve()
    if not path.is_dir():
        _fail(f"{arm_id}: run directory does not exist: {path}")
    manifest = _load_json(path / "manifest.json", arm_id)
    if manifest.get("schema") != "w5-manifest/2":
        _fail(f"{arm_id}: expected manifest schema w5-manifest/2")
    if manifest.get("run_id") != path.name:
        _fail(f"{arm_id}: manifest run_id does not match the run directory name")
    cases = tuple(_required(manifest, "cases", arm_id))
    requested_cases = tuple(_required(manifest, "requested_cases", arm_id))
    if cases != EXPECTED_CASES or requested_cases != EXPECTED_CASES:
        _fail(
            f"{arm_id}: cases and requested_cases must be the exact ordered 27-case list"
        )
    if _required(manifest, "stopped_views", arm_id) != []:
        _fail(f"{arm_id}: stopped views are not allowed in a four-arm comparison")
    if _pair(manifest, arm_id) != floor:
        _fail(f"{arm_id}: manifest visibility floor does not match its arm")
    _validate_columns(
        _required(manifest, "visibility_columns", arm_id), f"{arm_id}.manifest"
    )
    global_parameters = _required(manifest, "global_parameters", arm_id)
    if not isinstance(global_parameters, dict):
        _fail(f"{arm_id}: global_parameters must be an object")
    if (
        _integer(
            _required(global_parameters, "workers", arm_id),
            f"{arm_id}.global_parameters.workers",
        )
        != 10
    ):
        _fail(f"{arm_id}: worker count must be 10")
    if _pair(global_parameters, arm_id) != floor:
        _fail(f"{arm_id}: global-parameter visibility floor does not match its arm")
    _validate_columns(
        _required(global_parameters, "visibility_columns", arm_id),
        f"{arm_id}.global_parameters",
    )
    helper_paths = _required(manifest, "imported_helper_paths", arm_id)
    helper_hashes = _required(manifest, "imported_helper_hashes", arm_id)
    if not isinstance(helper_paths, dict) or not helper_paths:
        _fail(f"{arm_id}: imported helper paths are missing or ambiguous")
    if not isinstance(helper_hashes, dict) or not helper_hashes.get("imported"):
        _fail(f"{arm_id}: imported helper hashes are missing or ambiguous")
    _validate_contamination(manifest, EXPECTED_CASES, arm_id)

    line_sources = _required(manifest, "line_template_sources", arm_id)
    _validate_case_ordered_mapping(
        line_sources, EXPECTED_CASES, f"{arm_id}.line_template_sources"
    )
    line_admission = _required(manifest, "line_template_admission", arm_id)
    if not isinstance(line_admission, dict):
        _fail(f"{arm_id}.line_template_admission: expected an object")
    if _pair(line_admission, arm_id) != floor:
        _fail(f"{arm_id}: line-template admission floor does not match its arm")
    _validate_columns(
        _required(line_admission, "visibility_columns", arm_id),
        f"{arm_id}.line_template_admission",
    )
    admission_cases = _required(
        line_admission, "cases", f"{arm_id}.line_template_admission"
    )
    _validate_case_ordered_mapping(
        admission_cases, EXPECTED_CASES, f"{arm_id}.line_template_admission.cases"
    )
    admissions: dict[str, dict[str, Any]] = {}
    for case_id in EXPECTED_CASES:
        admissions[case_id] = _validate_admission(
            line_sources[case_id], admission_cases[case_id], floor, case_id, arm_id
        )

    per_view = _read_per_view(path / "per_view.csv", EXPECTED_CASES)
    for row in per_view:
        case_id = row["case_id"]
        if row.get("view_status") != "completed":
            _fail(f"{arm_id}/{case_id}: per_view is not completed")
        for field in ("G0_count", "G1_count", "line_template_count"):
            count = _csv_integer(row, field, f"{arm_id}/{case_id}")
            _integer(count, f"{arm_id}/{case_id}/per_view.{field}")
            if field in {"G0_count", "G1_count"} and count != 256:
                _fail(f"{arm_id}/{case_id}: {field} is {count}, expected 256")
            if (
                field == "line_template_count"
                and count != admissions[case_id]["proposal_count"]
            ):
                _fail(
                    f"{arm_id}/{case_id}: per_view line-template count disagrees with proposal_count"
                )
        lengthwise = _csv_integer(row, "min_visible_lengthwise", f"{arm_id}/{case_id}")
        _integer(lengthwise, f"{arm_id}/{case_id}.lengthwise")
        if lengthwise != floor[0]:
            _fail(
                f"{arm_id}/{case_id}: per_view lengthwise floor disagrees with its arm"
            )
        cross_court = _csv_integer(
            row, "min_visible_cross_court", f"{arm_id}/{case_id}"
        )
        _integer(cross_court, f"{arm_id}/{case_id}.cross_court")
        if cross_court != floor[1]:
            _fail(
                f"{arm_id}/{case_id}: per_view cross-court floor disagrees with its arm"
            )

    rankings = _load_json(path / "rankings.json", f"{arm_id}/rankings")
    if rankings.get("schema") != "w5-rankings/1":
        _fail(f"{arm_id}: expected rankings schema w5-rankings/1")
    reviews = _load_json(path / "review_candidates.json", f"{arm_id}/review_candidates")
    _validate_origins(rankings, reviews, EXPECTED_CASES, arm_id)
    return ArmRun(arm_id, floor, path, manifest, per_view, rankings, reviews)


def _crop_path(run: ArmRun, case_id: str, origin_key: str) -> Path:
    gallery = run.path / "gallery"
    expected_name = safe_name(f"{case_id}__{origin_key}") + "__crop.png"
    matches = sorted(path for path in gallery.rglob(expected_name) if path.is_file())
    if len(matches) != 1:
        _fail(
            f"{run.arm_id}/{case_id}: C winner crop {expected_name!r} resolved {len(matches)} times"
        )
    return matches[0]


def safe_name(value: str) -> str:
    """Match render_gallery.py's exact filename sanitiser."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _arm_case_rows(
    arms: tuple[ArmRun, ...],
    winners_by_arm: dict[str, dict[str, dict[str, Any]]],
    admissions: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for case_id in EXPECTED_CASES:
        arm_rows = {}
        for arm in arms:
            winner = winners_by_arm[arm.arm_id][case_id]
            admission = admissions[arm.arm_id][case_id]
            arm_rows[arm.arm_id] = {
                "c_winner": winner["origin_key"],
                "winner_source": winner["source"],
                "status": winner["status"],
                "hypotheses_before": admission["hypotheses_before"],
                "hypotheses_after": admission["hypotheses_after"],
                "hypotheses_rejected": admission["hypotheses_rejected"],
                "combined_admission_hypotheses": admission[
                    "combined_admission_hypotheses"
                ],
                "floor_zero_scanned_for_proposal_cap": admission[
                    "floor_zero_scanned_for_proposal_cap"
                ],
                "floor_zero_selected_count": admission["floor_zero_selected_count"],
                "scanned_for_proposal_cap": admission["scanned_for_proposal_cap"],
                "proposal_count": admission["proposal_count"],
                "cap_256_count": admission["cap_256_count"],
                "caps": admission["caps"],
                "refilled_proposal_count": admission["refilled_proposal_count"],
                "removed_from_floor_zero_count": admission[
                    "removed_from_floor_zero_count"
                ],
                "newly_admitted_proposal_ids": admission["newly_admitted_proposal_ids"],
                "removed_from_floor_zero_proposal_ids": admission[
                    "removed_from_floor_zero_proposal_ids"
                ],
            }
        rows.append({"case_id": case_id, "arms": arm_rows})
    return rows


def validate_runs(
    run_paths: dict[str, Path],
) -> tuple[tuple[ArmRun, ...], dict[str, Any]]:
    """Validate all four packets before any comparison output is created."""
    if tuple(run_paths) != ARM_IDS:
        _fail(f"run paths must be supplied in exact arm order {ARM_IDS}")
    arms = tuple(
        _validate_arm(arm_id, run_paths[arm_id], floor)
        for arm_id, floor in zip(ARM_IDS, ARM_FLOORS, strict=True)
    )
    baseline = _scrub_manifest_for_arm_comparison(arms[0].manifest)
    for arm in arms[1:]:
        if arm.manifest.get("imported_helper_hashes") != arms[0].manifest.get(
            "imported_helper_hashes"
        ):
            _fail(f"{arm.arm_id}: imported helper hashes differ from the (0,0) arm")
        if arm.manifest.get("imported_helper_paths") != arms[0].manifest.get(
            "imported_helper_paths"
        ):
            _fail(f"{arm.arm_id}: imported helper paths differ from the (0,0) arm")
        if _scrub_manifest_for_arm_comparison(arm.manifest) != baseline:
            _fail(
                f"{arm.arm_id}: non-floor scorer/refit/global manifest parameters differ"
            )

    winners_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    admissions: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        winners_by_arm[arm.arm_id] = _validate_origins(
            arm.rankings, arm.reviews, EXPECTED_CASES, arm.arm_id
        )
        admissions[arm.arm_id] = {
            case_id: _validate_admission(
                arm.manifest["line_template_sources"][case_id],
                arm.manifest["line_template_admission"]["cases"][case_id],
                arm.floor,
                case_id,
                arm.arm_id,
            )
            for case_id in EXPECTED_CASES
        }
    _validate_cross_arm_admission(arms, admissions)
    for arm in arms:
        for case_id in CONTACT_CASES:
            _crop_path(arm, case_id, winners_by_arm[arm.arm_id][case_id]["origin_key"])

    case_rows = _arm_case_rows(arms, winners_by_arm, admissions)
    comparison = {
        "schema": "w5-directional-comparison/1",
        "case_order": list(EXPECTED_CASES),
        "contact_sheet_cases": list(CONTACT_CASES),
        "arms": [
            {
                "arm_id": arm.arm_id,
                "floor": list(arm.floor),
                "run_id": arm.manifest["run_id"],
            }
            for arm in arms
        ],
        "cases": case_rows,
        "refill": [
            {
                "case_id": case_id,
                "arm_id": arm.arm_id,
                "floor": list(arm.floor),
                "hypotheses_before": admissions[arm.arm_id][case_id][
                    "hypotheses_before"
                ],
                "hypotheses_after": admissions[arm.arm_id][case_id]["hypotheses_after"],
                "hypotheses_rejected": admissions[arm.arm_id][case_id][
                    "hypotheses_rejected"
                ],
                "combined_admission_hypotheses": admissions[arm.arm_id][case_id][
                    "combined_admission_hypotheses"
                ],
                "floor_zero_scanned_for_proposal_cap": admissions[arm.arm_id][case_id][
                    "floor_zero_scanned_for_proposal_cap"
                ],
                "floor_zero_selected_count": admissions[arm.arm_id][case_id][
                    "floor_zero_selected_count"
                ],
                "scanned_for_proposal_cap": admissions[arm.arm_id][case_id][
                    "scanned_for_proposal_cap"
                ],
                "proposal_count": admissions[arm.arm_id][case_id]["proposal_count"],
                "cap_256_count": admissions[arm.arm_id][case_id]["cap_256_count"],
                "caps": admissions[arm.arm_id][case_id]["caps"],
                "refilled_proposal_count": admissions[arm.arm_id][case_id][
                    "refilled_proposal_count"
                ],
                "removed_from_floor_zero_count": admissions[arm.arm_id][case_id][
                    "removed_from_floor_zero_count"
                ],
                "newly_admitted_proposal_ids": admissions[arm.arm_id][case_id][
                    "newly_admitted_proposal_ids"
                ],
                "removed_from_floor_zero_proposal_ids": admissions[arm.arm_id][case_id][
                    "removed_from_floor_zero_proposal_ids"
                ],
            }
            for case_id in EXPECTED_CASES
            for arm in arms
        ],
    }
    return arms, comparison


def _compact_winner_cell(record: dict[str, Any]) -> str:
    winner = record["c_winner"]
    source = record["winner_source"]
    status = record["status"] or "unknown"
    pool = record["proposal_count"]
    cap_256 = record["cap_256_count"]
    refill = record["refilled_proposal_count"]
    removed = record["removed_from_floor_zero_count"]
    rejected = record["hypotheses_rejected"]
    return f"`{winner}` · {source} / {status} · pool {pool}/{cap_256} · +{refill}/-{removed} (reject {rejected})"


def _markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Directional W5 arm comparison",
        "",
        (
            "The four arms use projected visibility of six lengthwise pieces and six cross-court pieces. "
            "The downstream scorer may merge the two centre halves into five line identities."
        ),
        "",
        "## C winners and final line-template pools",
        "",
        (
            "Each arm cell gives the source-qualified C winner, source/status, final line-template pool "
            "(`proposal_count`/`caps['256'].count`) and visibility rejection/refill/depletion counts."
        ),
        "",
        "| case | (0,0) | (3,3) | (4,3) | (5,3) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in comparison["cases"]:
        lines.append(
            f"| {row['case_id']} | "
            + " | ".join(
                _compact_winner_cell(row["arms"][arm_id]) for arm_id in ARM_IDS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Admission, refill and depletion counts",
            "",
            (
                "The JSON retains the exact proposal IDs. This table keeps the report compact and shows the "
                "counts that explain each final pool."
            ),
            "",
            "| case | arm | floor | hypotheses before | rejected | combined admitted | floor-zero selected | final pool | refilled | removed |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["refill"]:
        lines.append(
            f"| {row['case_id']} | {ARM_LABELS[row['arm_id']]} | "
            f"({row['floor'][0]},{row['floor'][1]}) | {row['hypotheses_before']} | "
            f"{row['hypotheses_rejected']} | {row['combined_admission_hypotheses']} | "
            f"{row['floor_zero_selected_count']} | {row['proposal_count']}/{row['cap_256_count']} | "
            f"{row['refilled_proposal_count']} | {row['removed_from_floor_zero_count']} |"
        )
    lines.extend(
        [
            "",
            "## Contact sheets",
            "",
            (
                "Each sheet is ordered left-to-right as `(0,0)`, `(3,3)`, `(4,3)`, `(5,3)`. "
                "Arm labels are printed on every panel, so colour is not needed to interpret the comparison."
            ),
            "",
        ]
    )
    for case_id in comparison["contact_sheet_cases"]:
        filename = f"contact_sheets/{safe_name(case_id)}__directional_arms.png"
        lines.append(f"- `{case_id}`: [{filename}]({filename})")
    return "\n".join(lines) + "\n"


PANEL_SIZE = (420, 300)
CONTACT_HEADER_HEIGHT = 38
ARM_COLOURS = {
    "0_0": (31, 119, 180),
    "3_3": (230, 126, 34),
    "4_3": (106, 61, 154),
    "5_3": (0, 158, 115),
}


def _contact_sheet(
    case_id: str,
    arms: tuple[ArmRun, ...],
    winners_by_arm: dict[str, dict[str, dict[str, Any]]],
) -> Image.Image:
    header_height = CONTACT_HEADER_HEIGHT
    panel_width, panel_height = PANEL_SIZE
    sheet = Image.new(
        "RGB", (panel_width * len(arms), panel_height + header_height), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, arm in enumerate(arms):
        origin_key = winners_by_arm[arm.arm_id][case_id]["origin_key"]
        crop = Image.open(_crop_path(arm, case_id, origin_key)).convert("RGB")
        crop.thumbnail((panel_width - 16, panel_height - 16), Image.Resampling.LANCZOS)
        x_offset = index * panel_width
        y_offset = header_height + (panel_height - crop.height) // 2
        sheet.paste(crop, (x_offset + (panel_width - crop.width) // 2, y_offset))
        colour = ARM_COLOURS[arm.arm_id]
        draw.rectangle(
            (x_offset, 0, x_offset + panel_width - 1, header_height - 1), fill=colour
        )
        draw.text((x_offset + 10, 11), f"arm {ARM_LABELS[arm.arm_id]}", fill="white")
        draw.rectangle(
            (
                x_offset,
                header_height,
                x_offset + panel_width - 1,
                header_height + panel_height - 1,
            ),
            outline=colour,
            width=3,
        )
    return sheet


def write_outputs(
    output_dir: Path,
    comparison: dict[str, Any],
    arms: tuple[ArmRun, ...],
) -> None:
    """Write all validated outputs through a temporary sibling directory."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        _fail(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        winners_by_arm = {
            arm.arm_id: _validate_origins(
                arm.rankings, arm.reviews, EXPECTED_CASES, arm.arm_id
            )
            for arm in arms
        }
        (temporary / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
        (temporary / "comparison.md").write_text(_markdown(comparison))
        contact_dir = temporary / "contact_sheets"
        contact_dir.mkdir()
        for case_id in CONTACT_CASES:
            sheet = _contact_sheet(case_id, arms, winners_by_arm)
            sheet.save(contact_dir / f"{safe_name(case_id)}__directional_arms.png")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def compare_runs(run_paths: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    """Validate four runs and atomically write their comparison outputs."""
    arms, comparison = validate_runs(run_paths)
    write_outputs(output_dir, comparison, arms)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-00", type=Path, required=True, help="Completed (0,0) run directory"
    )
    parser.add_argument(
        "--run-33", type=Path, required=True, help="Completed (3,3) run directory"
    )
    parser.add_argument(
        "--run-43", type=Path, required=True, help="Completed (4,3) run directory"
    )
    parser.add_argument(
        "--run-53", type=Path, required=True, help="Completed (5,3) run directory"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_paths = {
        "0_0": args.run_00,
        "3_3": args.run_33,
        "4_3": args.run_43,
        "5_3": args.run_53,
    }
    try:
        compare_runs(run_paths, args.output_dir)
    except ComparisonError as error:
        raise SystemExit(f"directional comparison failed: {error}") from error


if __name__ == "__main__":
    main()
