"""Oracle-only frozen-reference coverage diagnostics for three 43-arm cases."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ARM_NAME = "w5_directional_20260921_r5_43"
TARGET_CASES = (
    "yellow_short_frame_14",
    "am1_window_00_frame_54",
    "shuttleset_21_scene_0010",
)
POOL_NAMES = ("complete_measured", "c_camera_hardvalid", "historical_fullcourt")


def import_local_modules(repo_root: Path):
    evaluation_dir = Path(__file__).resolve().parent
    project_root = repo_root / "scratch/court_det_fix"
    w5_dir = project_root / "w5_holistic"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(evaluation_dir))
    sys.path.insert(0, str(w5_dir))
    from audit_43_gates import candidate_map, read_json_gz, safe_name
    from render_gallery import import_verifier, load_native_frame, render_prediction
    from verifier import reference_corner_error

    return (
        candidate_map,
        import_verifier,
        load_native_frame,
        read_json_gz,
        reference_corner_error,
        render_prediction,
        safe_name,
    )


def load_case_ids(packet_dir: Path) -> list[str]:
    with (packet_dir / "per_view.csv").open(newline="", encoding="utf-8") as stream:
        return [row["case_id"] for row in csv.DictReader(stream)]


def source_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(candidate.get("source") for candidate in candidates).items()))


def gate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    gates = candidate.get("gates", {})
    return {
        "hard_valid": candidate.get("hard_valid"),
        "hard_validity_reason": candidate.get("hard_validity_reason"),
        "geometry_valid": gates.get("geometry_valid"),
        "camera_eligible": candidate.get("camera_eligible"),
        "camera_error": gates.get("camera_error"),
        "player_fractions": gates.get("player_fractions"),
    }


def candidate_summary(
    candidate: dict[str, Any],
    origin_key: str,
    pool_rank: int,
    c_rank: int | None,
    error: dict[str, Any],
    saved_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    saved_reference = None
    if saved_metrics is not None:
        frozen = saved_metrics.get("frozen_case_reference")
        if frozen is not None:
            saved_reference = {
                "maximum_native_px": frozen.get("maximum"),
                "maximum_working_px": error["maximum"],
                "relabelled_180": frozen.get("relabelled_180"),
            }
    return {
        "origin_key": origin_key,
        "candidate_id": candidate.get("candidate_id"),
        "source": candidate.get("source"),
        "source_memberships": candidate.get("source_memberships"),
        "pool_rank": pool_rank,
        "c_rank": c_rank,
        "maxerror_working_px": error["maximum"],
        "maxerror_native_px": error["native_maximum"],
        "relabelled_180": error["relabelled_180"],
        "gates": gate_summary(candidate),
        "saved_reference_diagnostic": saved_reference,
    }


def reference_error(
    candidate: dict[str, Any],
    reference_corners: np.ndarray,
    scale: np.ndarray,
    reference_corner_error: Any,
) -> dict[str, Any]:
    corners = np.asarray(candidate["corners_px"], dtype=float)
    native = reference_corner_error(corners, reference_corners)
    working = reference_corner_error(corners * scale, reference_corners * scale)
    return {
        "maximum": float(working["maximum"]),
        "native_maximum": float(native["maximum"]),
        "relabelled_180": bool(working["relabelled_180"]),
    }


def render_diagnostic(
    frame: Any,
    candidate: dict[str, Any],
    case_id: str,
    roles: list[str],
    context: dict[str, Any],
    packet_gallery: Path,
    render_dir: Path,
    safe_name: Any,
    render_prediction: Any,
    repo_root: Path,
) -> tuple[str, str]:
    stem_name = safe_name(f"{case_id}__{candidate['origin_key']}")
    packet_full = packet_gallery / f"{stem_name}__full.png"
    if packet_full.is_file():
        return packet_full.relative_to(repo_root).as_posix(), "packet_gallery"

    render_dir.mkdir(parents=True, exist_ok=True)
    stem = render_dir / stem_name
    full_path = render_dir / f"{stem_name}__full.png"
    if not full_path.is_file():
        links = render_prediction(frame, candidate, context, roles, stem, None)
        (render_dir / f"{stem_name}__crop.png").unlink(missing_ok=True)
        if full_path.name not in links or not full_path.is_file():
            raise RuntimeError(f"{case_id}/{candidate['origin_key']}: missing rendered full image")
    return full_path.relative_to(repo_root).as_posix(), "evaluation_gallery"


def run(repo_root: Path, output_path: Path) -> dict[str, Any]:
    (
        candidate_map,
        import_verifier,
        load_native_frame,
        read_json_gz,
        reference_corner_error,
        render_prediction,
        safe_name,
    ) = import_local_modules(repo_root)
    project_root = repo_root / "scratch/court_det_fix"
    packet_dir = project_root / "evidence/holistic_admission/directional_20260921_r5" / ARM_NAME
    evaluation_dir = packet_dir.parent / "evaluation"
    packet_gallery = packet_dir / "gallery"
    render_dir = evaluation_dir / "gallery"
    diagnostics = read_json_gz(packet_dir / "reference_diagnostics.json.gz")
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    camera_limit = float(manifest["global_parameters"]["camera_error_limit_historical"])
    case_ids = load_case_ids(packet_dir)
    if set(TARGET_CASES) - set(case_ids):
        raise ValueError("requested coverage cases are missing from the 43-arm packet")
    verifier = import_verifier(project_root)
    rendered_paths: set[str] = set()
    case_outputs: list[dict[str, Any]] = []

    for case_id in TARGET_CASES:
        record = read_json_gz(packet_dir / "case_records" / f"{case_id}.json.gz")
        candidates = candidate_map(record)
        measured_keys = [candidate["origin_key"] for candidate in [*record["parents"], *record["valid_children"]]]
        c_rank = record["rankings"]["C"]["provisional_rank"]
        fullcourt_rank = record["rankings"]["C"]["historical_fullcourt_subset_rank"]
        c_positions = {origin_key: index + 1 for index, origin_key in enumerate(c_rank)}
        pool_keys = {
            "complete_measured": measured_keys,
            "c_camera_hardvalid": c_rank,
            "historical_fullcourt": fullcourt_rank,
        }
        native_width, native_height = record["provenance"]["native_dimensions"]
        working_width, working_height = record["provenance"]["working_dimensions"]
        scale = np.asarray(
            [working_width / native_width, working_height / native_height],
            dtype=float,
        )
        pack_path = project_root / verifier["CASE_PACKS"][verifier["PACK_OF"][case_id]]
        reference_pack = verifier["read_json_gz"](pack_path)
        reference_corners = np.asarray(reference_pack["references"][case_id]["corners_px"], dtype=float)
        saved_case = diagnostics.get(case_id, {})
        saved_selected = saved_case.get("selected_origin_keys", {})
        saved_metrics_by_key = saved_case.get("candidates", {})
        errors = {
            origin_key: reference_error(
                candidates[origin_key], reference_corners, scale, reference_corner_error
            )
            for origin_key in measured_keys
        }
        nearest: dict[str, dict[str, Any]] = {}
        for pool_name, pool in pool_keys.items():
            if not pool:
                raise ValueError(f"{case_id}/{pool_name}: empty saved candidate pool")
            pool_positions = {origin_key: index + 1 for index, origin_key in enumerate(pool)}
            winner_key = min(pool, key=lambda origin_key: (errors[origin_key]["maximum"], pool_positions[origin_key]))
            nearest[pool_name] = candidate_summary(
                candidates[winner_key],
                winner_key,
                pool_positions[winner_key],
                c_positions.get(winner_key),
                errors[winner_key],
                saved_metrics_by_key.get(winner_key),
            )

        diagnostic_keys: list[str] = []
        diagnostic_roles: dict[str, list[str]] = {}
        for role, origin_key in saved_selected.items():
            if origin_key is None or origin_key not in candidates:
                continue
            diagnostic_roles.setdefault(origin_key, []).append(f"saved_{role}")
        for pool_name, result in nearest.items():
            origin_key = result["origin_key"]
            diagnostic_roles.setdefault(origin_key, []).append(f"nearest_{pool_name}")
        diagnostic_keys = list(diagnostic_roles)

        frame = None
        context = None
        rendered_diagnostics: list[dict[str, Any]] = []
        for origin_key in diagnostic_keys:
            candidate = candidates[origin_key]
            if frame is None:
                source = verifier["load_source"](project_root, case_id)
                provenance = verifier["load_case_provenance"](project_root, case_id)
                frame = load_native_frame(project_root, source, provenance, verifier)
                context = {
                    "id": case_id,
                    "dimensions": source["dimensions"],
                    "working_dimensions": record["provenance"]["working_dimensions"],
                }
            image_path, image_source = render_diagnostic(
                frame,
                candidate,
                case_id,
                diagnostic_roles[origin_key],
                context,
                packet_gallery,
                render_dir,
                safe_name,
                render_prediction,
                repo_root,
            )
            if image_source == "evaluation_gallery":
                rendered_paths.add(image_path)
            pool_rank = min(
                (pool_keys[pool_name].index(origin_key) + 1 for pool_name in POOL_NAMES if origin_key in pool_keys[pool_name]),
                default=None,
            )
            rendered_diagnostics.append(
                {
                    **candidate_summary(
                        candidate,
                        origin_key,
                        pool_rank,
                        c_positions.get(origin_key),
                        errors[origin_key],
                        saved_metrics_by_key.get(origin_key),
                    ),
                    "roles": diagnostic_roles[origin_key],
                    "image_path": image_path,
                    "image_source": image_source,
                }
            )

        case_outputs.append(
            {
                "case_id": case_id,
                "native_dimensions": [native_width, native_height],
                "working_dimensions": [working_width, working_height],
                "working_scale": scale.tolist(),
                "reference_diagnostic_selected_origin_keys": saved_selected,
                "reference_diagnostic_candidate_count": len(saved_metrics_by_key),
                "pool_counts": {
                    pool_name: {
                        "count": len(pool),
                        "source_counts": source_counts([candidates[key] for key in pool]),
                    }
                    for pool_name, pool in pool_keys.items()
                },
                "nearest": nearest,
                "diagnostic_candidates": rendered_diagnostics,
            }
        )

    output = {
        "schema": "w5-reference-coverage-oracle/1",
        "arm": ARM_NAME,
        "cases": list(TARGET_CASES),
        "pool_definitions": {
            "complete_measured": "parents followed by valid_children",
            "c_camera_hardvalid": "saved rankings.C.provisional_rank",
            "historical_fullcourt": "saved rankings.C.historical_fullcourt_subset_rank",
        },
        "reference_metric": "maximum corner error after native-to-working coordinate scaling; diagnostic only",
        "camera_error_limit_historical": camera_limit,
        "rendered_unique_full_images": len(rendered_paths),
        "case_results": case_outputs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    return {
        "case_count": len(case_outputs),
        "rendered_unique_full_images": len(rendered_paths),
        "output": output_path.relative_to(repo_root).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo_root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
