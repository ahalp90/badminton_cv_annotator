"""Oracle-only visible-landmark coverage diagnostics for two 43-arm cases.

The frozen reference landmarks are used only for this diagnostic. Candidate
membership, ordering, gates and homographies come from the saved W5 record.
No candidates are fitted, rescored or admitted by this script.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ARM_NAME = "w5_directional_20260921_r5_43"
TARGET_CASES = ("yellow_short_frame_14", "am1_window_00_frame_54")
POOL_NAMES = ("complete_measured", "c_camera_hardvalid", "historical_fullcourt")
COURT_SIZE_M = np.asarray([6.1, 13.4], dtype=float)


def import_local_modules(repo_root: Path):
    evaluation_dir = Path(__file__).resolve().parent
    project_root = repo_root / "scratch/court_det_fix"
    w5_dir = project_root / "w5_holistic"
    sys.path[:0] = [str(repo_root), str(repo_root / "src"), str(evaluation_dir), str(w5_dir)]
    from audit_43_gates import candidate_map, read_json_gz, safe_name
    from render_gallery import import_verifier, load_native_frame, render_prediction

    from experiments.annotator.independent_court import detector

    return candidate_map, detector, import_verifier, load_native_frame, read_json_gz, render_prediction, safe_name


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


def landmark_metrics(
    homography: np.ndarray,
    court_points: np.ndarray,
    observed_working: np.ndarray,
    detector: Any,
) -> dict[str, Any]:
    direct_points, direct_denominators = detector.project(homography[None], court_points)
    relabelled_court = COURT_SIZE_M - court_points
    relabelled_points, relabelled_denominators = detector.project(homography[None], relabelled_court)
    direct_points = direct_points[0]
    relabelled_points = relabelled_points[0]

    def errors(projected: np.ndarray, denominators: np.ndarray) -> np.ndarray:
        valid = np.isfinite(projected).all(axis=-1) & (denominators[0] > 1e-6)
        result = np.full(len(court_points), np.inf, dtype=float)
        result[valid] = np.linalg.norm(projected[valid] - observed_working[valid], axis=1)
        return result

    direct_errors = errors(direct_points, direct_denominators)
    relabelled_errors = errors(relabelled_points, relabelled_denominators)
    direct_maximum = float(np.max(direct_errors))
    relabelled_maximum = float(np.max(relabelled_errors))
    use_relabelled = relabelled_maximum < direct_maximum
    selected_errors = relabelled_errors if use_relabelled else direct_errors
    return {
        "landmark_count": len(selected_errors),
        "maximum_working_px": direct_maximum if not use_relabelled else relabelled_maximum,
        "median_working_px": float(np.median(selected_errors)),
        "p90_working_px": float(np.percentile(selected_errors, 90)),
        "relabelled_180": use_relabelled,
        "projection_valid": bool(np.isfinite(selected_errors).all()),
        "landmark_errors_working_px": selected_errors.tolist(),
    }


def candidate_summary(
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    pool_ranks: dict[str, int | None],
    roles: list[str],
    image_path: str | None,
) -> dict[str, Any]:
    return {
        "origin_key": candidate.get("origin_key"),
        "candidate_id": candidate.get("candidate_id"),
        "source": candidate.get("source"),
        "source_memberships": candidate.get("source_memberships"),
        "source_occurrences": candidate.get("source_occurrences"),
        "kind": candidate.get("kind"),
        "kind_order": candidate.get("kind_order"),
        "origin_index": candidate.get("origin_index"),
        "parent_origin_key": candidate.get("parent_origin_key"),
        "source_order": candidate.get("source_order"),
        "occurrence_count": candidate.get("occurrence_count"),
        "pool_ranks": pool_ranks,
        "gates": candidate.get("gates"),
        "gate_summary": gate_summary(candidate),
        "historical": candidate.get("historical"),
        **metrics,
        "roles": roles,
        "image_path": image_path,
    }


def render_candidate(
    frame: Any,
    candidate: dict[str, Any],
    case_id: str,
    roles: list[str],
    context: dict[str, Any],
    render_dir: Path,
    repo_root: Path,
    render_prediction: Any,
    safe_name: Any,
) -> str:
    stem = render_dir / safe_name(f"visible_landmarks__{case_id}__{candidate['origin_key']}")
    render_dir.mkdir(parents=True, exist_ok=True)
    full_path = stem.with_name(stem.name + "__full.png")
    if not full_path.is_file():
        links = render_prediction(frame, candidate, context, roles, stem, None)
        stem.with_name(stem.name + "__crop.png").unlink(missing_ok=True)
        if full_path.name not in links or not full_path.is_file():
            raise RuntimeError(f"{case_id}/{candidate['origin_key']}: missing rendered full image")
    return full_path.relative_to(repo_root).as_posix()


def run(repo_root: Path, output_path: Path) -> dict[str, Any]:
    (
        candidate_map,
        detector,
        import_verifier,
        load_native_frame,
        read_json_gz,
        render_prediction,
        safe_name,
    ) = import_local_modules(repo_root)
    project_root = repo_root / "scratch/court_det_fix"
    packet_dir = project_root / "evidence/holistic_admission/directional_20260921_r5" / ARM_NAME
    evaluation_dir = packet_dir.parent / "evaluation"
    render_dir = evaluation_dir / "visible_landmark_gallery"
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    camera_limit = float(manifest["global_parameters"]["camera_error_limit_historical"])
    verifier = import_verifier(project_root)
    reference_pack = read_json_gz(project_root / "frozen_views/packs/marking_refit_inputs.json.gz")
    rendered_paths: set[str] = set()
    case_outputs: list[dict[str, Any]] = []

    for case_id in TARGET_CASES:
        record = read_json_gz(packet_dir / "case_records" / f"{case_id}.json.gz")
        candidates = candidate_map(record)
        measured_keys = [candidate["origin_key"] for candidate in [*record["parents"], *record["valid_children"]]]
        c_rank = record["rankings"]["C"]["provisional_rank"]
        historical_rank = record["rankings"]["C"]["historical_fullcourt_subset_rank"]
        pool_keys = {
            "complete_measured": measured_keys,
            "c_camera_hardvalid": c_rank,
            "historical_fullcourt": historical_rank,
        }
        for pool_name, pool in pool_keys.items():
            if not pool or any(origin_key not in candidates for origin_key in pool):
                raise ValueError(f"{case_id}/{pool_name}: saved pool is empty or names a missing candidate")
        pool_ranks_by_key = {
            pool_name: {origin_key: index + 1 for index, origin_key in enumerate(pool)}
            for pool_name, pool in pool_keys.items()
        }

        source = verifier["load_source"](project_root, case_id)
        native_dimensions = tuple(record["provenance"]["native_dimensions"])
        working_dimensions = tuple(record["provenance"]["working_dimensions"])
        if tuple(source["dimensions"][key] for key in ("width", "height")) != native_dimensions:
            raise ValueError(f"{case_id}: record and source native dimensions disagree")
        scale = np.asarray(working_dimensions, dtype=float) / np.asarray(native_dimensions, dtype=float)
        reference = reference_pack["references"][case_id]
        landmarks = reference["landmarks"]
        court_points = np.asarray([landmark["court_m"] for landmark in landmarks], dtype=float)
        observed_working = np.asarray([landmark["image_px"] for landmark in landmarks], dtype=float) * scale

        metrics_by_key = {
            origin_key: landmark_metrics(
                np.asarray(candidates[origin_key]["homography_working"], dtype=float),
                court_points,
                observed_working,
                detector,
            )
            for origin_key in measured_keys
        }
        nearest_keys = {
            pool_name: min(
                pool,
                key=lambda origin_key: (
                    metrics_by_key[origin_key]["maximum_working_px"],
                    pool_ranks_by_key[pool_name][origin_key],
                ),
            )
            for pool_name, pool in pool_keys.items()
        }
        selected_c_key = record["rankings"]["C"].get("selected_origin_key")
        if selected_c_key not in candidates:
            raise ValueError(f"{case_id}: selected C candidate is missing")

        roles_by_key: dict[str, list[str]] = {}
        for pool_name, origin_key in nearest_keys.items():
            roles_by_key.setdefault(origin_key, []).append(f"nearest_{pool_name}")
        roles_by_key.setdefault(selected_c_key, []).append("selected_C")
        frame = load_native_frame(project_root, source, verifier["load_case_provenance"](project_root, case_id), verifier)
        context = {
            "id": case_id,
            "dimensions": source["dimensions"],
            "working_dimensions": list(working_dimensions),
        }
        image_paths = {
            origin_key: render_candidate(
                frame,
                candidates[origin_key],
                case_id,
                roles,
                context,
                render_dir,
                repo_root,
                render_prediction,
                safe_name,
            )
            for origin_key, roles in roles_by_key.items()
        }
        rendered_paths.update(image_paths.values())

        def ranks_for(
            origin_key: str, pool_ranks=pool_ranks_by_key
        ) -> dict[str, int | None]:
            return {
                pool_name: pool_ranks[pool_name].get(origin_key)
                for pool_name in POOL_NAMES
            }

        nearest = {
            pool_name: candidate_summary(
                candidates[origin_key],
                metrics_by_key[origin_key],
                ranks_for(origin_key),
                roles_by_key[origin_key],
                image_paths[origin_key],
            )
            for pool_name, origin_key in nearest_keys.items()
        }
        selected_c = candidate_summary(
            candidates[selected_c_key],
            metrics_by_key[selected_c_key],
            ranks_for(selected_c_key),
            roles_by_key[selected_c_key],
            image_paths[selected_c_key],
        )
        case_outputs.append(
            {
                "case_id": case_id,
                "native_dimensions": list(native_dimensions),
                "working_dimensions": list(working_dimensions),
                "working_scale": scale.tolist(),
                "visible_landmark_count": len(landmarks),
                "pool_counts": {
                    pool_name: {
                        "count": len(pool),
                        "source_counts": source_counts([candidates[key] for key in pool]),
                    }
                    for pool_name, pool in pool_keys.items()
                },
                "nearest": nearest,
                "selected_C": selected_c,
            }
        )

    output = {
        "schema": "w5-visible-landmark-coverage/1",
        "arm": ARM_NAME,
        "cases": list(TARGET_CASES),
        "pool_definitions": {
            "complete_measured": "parents followed by valid_children",
            "c_camera_hardvalid": "saved rankings.C.provisional_rank",
            "historical_fullcourt": "saved rankings.C.historical_fullcourt_subset_rank",
        },
        "reference_metric": (
            "all frozen visible landmarks projected in working pixels; maximum, median and p90 "
            "after choosing the lower-error direct or 180-degree court relabelling"
        ),
        "court_relabelling_180": "(x, y) -> (6.1 - x, 13.4 - y)",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "visible_landmark_coverage.json.gz",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.repo_root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
