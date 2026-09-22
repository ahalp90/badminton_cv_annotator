"""Render a lightweight gallery from the saved original-27 W5 selections.

Use the compact review where possible, then recover omitted winners from their
saved case records. Verify working-to-native scaling against saved corners.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

OVERLAY_COLOUR_BGR = (255, 255, 0)
OVERLAY_THICKNESS = 1
FAR_MARGIN_PX = 30

BASELINE_REVIEW = Path(
    "evidence/holistic_admission/directional_20260921_r5/"
    "w5_directional_20260921_r5_43/review_candidates.json"
)
ABLATION_DETAILS = Path(
    "evidence/holistic_admission/directional_20260921_r5/evaluation/"
    "ablation_winner_details.json.gz"
)
COMPARISON = Path("wider_evaluation/runs/20260922/regression_comparison.json.gz")
MANIFEST = Path("wider_evaluation/runs/20260922/manifest.json.gz")
DEFAULT_OUTPUT = Path("wider_evaluation/runs/20260922/baseline_gallery")

ROLE_SPECS = (
    ("full/gated", "full", "gated", "historical_fullcourt"),
    ("g1_templates/gated", "g1_templates", "gated", "g1_plus_line_template_fullcourt"),
    ("full/ungated", "full", "ungated", None),
    ("g1_templates/ungated", "g1_templates", "ungated", None),
)


def read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def import_geometry_modules(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from experiments.annotator.independent_court import detector, paint_geometry

    return detector, paint_geometry


def project_overlay(
    frame: np.ndarray,
    candidate: dict[str, Any],
    native_size: np.ndarray,
    detector: Any,
    paint_geometry: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    homography = np.asarray(candidate["homography_working"], dtype=float)
    projected_working, _ = detector.project(homography[None], paint_geometry.CENTRE_SEGMENTS_M)
    resize = min(1.0, detector.Settings().max_dimension / max(native_size))
    working_size = np.rint(native_size * resize)
    scale = native_size / working_size
    projected_native = projected_working[0].reshape(-1, 2, 2) * scale
    canvas = frame.copy()
    for start, end in projected_native:
        cv2.line(
            canvas,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            OVERLAY_COLOUR_BGR,
            OVERLAY_THICKNESS,
            cv2.LINE_AA,
        )
    saved_corners = np.asarray(candidate["corners_px"], dtype=float)
    projected_corners_working, _ = detector.project(homography[None], detector.CORNER_COURT_M)
    projected_corners_native = projected_corners_working[0] * scale
    np.testing.assert_allclose(projected_corners_native, saved_corners, atol=0.05, rtol=0)
    corner_error = np.abs(projected_corners_native - saved_corners)
    geometry = {
        "working_dimensions": working_size.astype(int).tolist(),
        "native_dimensions": native_size.astype(int).tolist(),
        "native_over_working_scale": scale.tolist(),
        "saved_corners_px": saved_corners.tolist(),
        "projected_corners_native_px": projected_corners_native.tolist(),
        "corner_error_max_px": float(np.nanmax(corner_error)),
        "corner_error_mean_px": float(np.nanmean(corner_error)),
        "overlay_colour_bgr": list(OVERLAY_COLOUR_BGR),
        "overlay_thickness_px": OVERLAY_THICKNESS,
    }
    return canvas, geometry


def far_end_bounds(corners: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    ordered_y = np.sort(corners[:, 1])
    x0 = max(0, int(np.floor(np.min(corners[:, 0]) - FAR_MARGIN_PX)))
    x1 = min(width, int(np.ceil(np.max(corners[:, 0]) + FAR_MARGIN_PX + 1)))
    y0 = max(0, int(np.floor(ordered_y[:2].min() - FAR_MARGIN_PX)))
    y1 = min(height, int(np.ceil((ordered_y[1] + ordered_y[2]) / 2 + FAR_MARGIN_PX)))
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    return x0, y0, x1, y1


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def render_case(
    evidence_root: Path,
    output_root: Path,
    case_id: str,
    manifest_row: dict[str, Any],
    review_case: dict[str, Any],
    comparison_case: dict[str, Any],
    ablation_rows: dict[tuple[str, str], dict[str, Any]],
    detector: Any,
    paint_geometry: Any,
) -> dict[str, Any]:
    image_path = evidence_root / manifest_row["image"]
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)
    native_size = np.asarray([frame.shape[1], frame.shape[0]], dtype=float)
    case_output = output_root / safe_name(case_id)
    case_output.mkdir(parents=True, exist_ok=True)
    raw_path = case_output / "raw.png"
    write_image(raw_path, frame)

    candidates = review_case.get("candidates", {})
    selections = comparison_case.get("selections", {})
    roles_by_key: dict[str, list[str]] = {}
    role_records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for role, arm, gate, ablation_filter in ROLE_SPECS:
        selection = selections.get(arm, {})
        candidate_key = selection.get(gate)
        ablation = ablation_rows.get((case_id, ablation_filter)) if ablation_filter else None
        role_record: dict[str, Any] = {
            "role": role,
            "arm": arm,
            "gate": gate,
            "candidate_key": candidate_key,
            "ablation_filter": ablation_filter,
            "ablation_winner": None if ablation is None else ablation.get("winner"),
            "ablation_image_path": None if ablation is None else ablation.get("image_path"),
        }
        if candidate_key is None:
            role_record["status"] = "selection_missing"
            missing.append({**role_record, "reason": "comparison_selection_missing"})
        elif candidate_key not in candidates:
            role_record["status"] = "geometry_missing"
            missing.append({**role_record, "reason": "candidate_key_missing_from_review_candidates"})
        elif not {"corners_px", "homography_working"}.issubset(candidates[candidate_key]):
            role_record["status"] = "geometry_incomplete"
            missing.append({**role_record, "reason": "candidate_geometry_fields_missing"})
        else:
            role_record["status"] = "ready"
            roles_by_key.setdefault(candidate_key, []).append(role)
        role_records.append(role_record)

    rendered: list[dict[str, Any]] = []
    for index, (candidate_key, roles) in enumerate(roles_by_key.items()):
        candidate = candidates[candidate_key]
        canvas, geometry = project_overlay(frame, candidate, native_size, detector, paint_geometry)
        corners = np.asarray(candidate["corners_px"], dtype=float)
        x0, y0, x1, y1 = far_end_bounds(corners, frame.shape)
        full_path = case_output / f"overlay_{index:02d}__full.png"
        far_path = case_output / f"overlay_{index:02d}__far_end_raw_then_overlay.png"
        write_image(full_path, canvas)
        far = np.concatenate((frame[y0:y1, x0:x1], canvas[y0:y1, x0:x1]), axis=0)
        write_image(far_path, far)
        rendered.append(
            {
                "candidate_key": candidate_key,
                "roles": roles,
                "full": str(full_path.relative_to(output_root)),
                "far_end_raw_then_overlay": str(far_path.relative_to(output_root)),
                "far_end_crop_bounds_xyxy": [x0, y0, x1, y1],
                "source_memberships": candidate.get("source_memberships"),
                "corners_px": candidate["corners_px"],
                "homography_working": candidate["homography_working"],
                "geometry": geometry,
            }
        )

    return {
        "case_id": case_id,
        "manifest_image": manifest_row["image"],
        "manifest_image_kind": manifest_row.get("image_kind"),
        "native_dimensions": native_size.astype(int).tolist(),
        "raw": str(raw_path.relative_to(output_root)),
        "roles": role_records,
        "rendered": rendered,
        "missing": missing,
    }


def run(repo_root: Path, output: Path) -> dict[str, Any]:
    evidence_root = repo_root / "scratch/court_det_fix"
    baseline_review_path = evidence_root / BASELINE_REVIEW
    ablation_path = evidence_root / ABLATION_DETAILS
    comparison_path = evidence_root / COMPARISON
    manifest_path = evidence_root / MANIFEST
    review = json.loads(baseline_review_path.read_text(encoding="utf-8"))
    ablation = read_json_gz(ablation_path)
    comparison = read_json_gz(comparison_path)
    manifest = read_json_gz(manifest_path)
    ablation_rows = {
        (row["case_id"], row["filter"]): row for row in ablation.get("rows", [])
    }
    manifest_rows = {
        row["case_id"]: row for row in manifest["cases"] if row.get("previous_w5_case")
    }
    comparison_rows = {row["case_id"]: row for row in comparison.get("cases", [])}
    # The compact review omitted some gated and restricted winners.
    for case_id, row in comparison_rows.items():
        needed = {selection[gate] for selection in row["selections"].values() for gate in ("gated", "ungated")}
        missing = needed - review[case_id]["candidates"].keys() - {None}
        if missing:
            record = read_json_gz(baseline_review_path.parent / "case_records" / f"{case_id}.json.gz")
            candidates = record["parents"] + record["valid_children"]
            review[case_id]["candidates"].update({c["origin_key"]: c for c in candidates if c["origin_key"] in missing})
    case_ids = [row["case_id"] for row in manifest["cases"] if row.get("previous_w5_case")]
    if len(case_ids) != 27 or len(set(case_ids)) != 27:
        raise ValueError(f"expected 27 original baseline cases, found {len(case_ids)}")

    detector, paint_geometry = import_geometry_modules(repo_root)
    cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        if case_id not in review:
            cases.append({"case_id": case_id, "missing": [{"reason": "review_case_missing"}]})
            continue
        if case_id not in comparison_rows:
            cases.append({"case_id": case_id, "missing": [{"reason": "comparison_case_missing"}]})
            continue
        cases.append(
            render_case(
                evidence_root,
                output,
                case_id,
                manifest_rows[case_id],
                review[case_id],
                comparison_rows[case_id],
                ablation_rows,
                detector,
                paint_geometry,
            )
        )

    metadata = {
        "schema": "saved-review-gallery/1",
        "source": {
            "baseline_review": str(baseline_review_path.relative_to(repo_root)),
            "ablation_details": str(ablation_path.relative_to(repo_root)),
            "regression_comparison": str(comparison_path.relative_to(repo_root)),
            "frozen_manifest": str(manifest_path.relative_to(repo_root)),
        },
        "case_count": len(cases),
        "working_dimensions": "per case, verified against saved native corners",
        "overlay": {
            "colour_bgr": list(OVERLAY_COLOUR_BGR),
            "thickness_px": OVERLAY_THICKNESS,
            "projected_geometry": "CENTRE_SEGMENTS_M from saved homography_working, scaled native/working",
            "far_end_crop": "native corners_px bounds plus 30px, clipped to frame",
        },
        "cases": cases,
        "missing_geometry": [
            {"case_id": case["case_id"], **entry}
            for case in cases
            for entry in case.get("missing", [])
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json.gz"
    with gzip.open(metadata_path, "wt", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    return {
        "case_count": len(cases),
        "rendered_cases": sum(bool(case.get("rendered")) for case in cases),
        "rendered_overlays": sum(len(case.get("rendered", [])) for case in cases),
        "missing_roles": len(metadata["missing_geometry"]),
        "output": str(output),
        "metadata": str(metadata_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output = (args.output or repo_root / "scratch/court_det_fix" / DEFAULT_OUTPUT).resolve()
    print(json.dumps(run(repo_root, output), indent=2))


if __name__ == "__main__":
    main()
