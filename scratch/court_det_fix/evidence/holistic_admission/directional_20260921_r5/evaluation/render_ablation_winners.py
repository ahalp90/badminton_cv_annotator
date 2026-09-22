"""Render missing 43-arm ablation winners without rerunning W5 scoring."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

EVALUATION_FILTERS = (
    "historical_fullcourt",
    "g0_plus_g1",
    "g0_only",
    "g1_only",
    "g1_plus_line_template_fullcourt",
    "g0_plus_line_template_fullcourt",
)
ARM_NAME = "w5_directional_20260921_r5_43"


def import_local_modules(repo_root: Path):
    evaluation_dir = Path(__file__).resolve().parent
    w5_dir = repo_root / "scratch/court_det_fix/w5_holistic"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(evaluation_dir))
    sys.path.insert(0, str(w5_dir))
    from audit_43_gates import (
        candidate_map,
        historical_fullcourt,
        rank_filter,
        read_json_gz,
        source_memberships,
    )
    from render_gallery import (
        import_verifier,
        load_native_frame,
        render_prediction,
        safe_name,
    )

    return (
        candidate_map,
        historical_fullcourt,
        rank_filter,
        read_json_gz,
        source_memberships,
        import_verifier,
        load_native_frame,
        render_prediction,
        safe_name,
    )


def candidate_scalar(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {
            "q_paint10_span_weighted": None,
            "camera_error": None,
            "player_fractions": None,
            "source_memberships": None,
            "source": None,
        }
    evidence = candidate.get("evidence", {})
    gates = candidate.get("gates", {})
    return {
        "q_paint10_span_weighted": evidence.get("q_paint10_span_weighted"),
        "camera_error": gates.get("camera_error"),
        "player_fractions": gates.get("player_fractions"),
        "source_memberships": candidate.get("source_memberships"),
        "source": candidate.get("source"),
    }


def best_legacy_reference_error(
    case_id: str,
    diagnostics: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> float | None:
    case_diagnostics = diagnostics.get(case_id, {})
    values = []
    for origin_key, metrics in case_diagnostics.get("candidates", {}).items():
        candidate = candidates.get(origin_key)
        reference = metrics.get("frozen_case_reference", {})
        maximum = reference.get("maximum")
        memberships = set(candidate.get("source_memberships", ())) if candidate is not None else set()
        if candidate is not None and maximum is not None and memberships & {"G0", "G1"}:
            values.append(float(maximum))
    return min(values) if values else None


def render_missing(
    frame: Any,
    candidate: dict[str, Any],
    case_id: str,
    filter_name: str,
    context: dict[str, Any],
    output_dir: Path,
    safe_name: Callable[[str], str],
    render_prediction: Callable[..., list[str]],
) -> Path:
    stem = output_dir / safe_name(f"{case_id}__{candidate['origin_key']}")
    full_path = stem.with_name(stem.name + "__full.png")
    if not full_path.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        links = render_prediction(frame, candidate, context, [filter_name], stem, None)
        crop_path = stem.with_name(stem.name + "__crop.png")
        crop_path.unlink(missing_ok=True)
        if full_path.name not in links or not full_path.is_file():
            raise RuntimeError(f"{case_id}/{filter_name}: renderer did not create {full_path}")
    return full_path


def run(repo_root: Path, output_path: Path) -> dict[str, Any]:
    (
        candidate_map,
        historical_fullcourt,
        rank_filter,
        read_json_gz,
        source_memberships,
        import_verifier,
        load_native_frame,
        render_prediction,
        safe_name,
    ) = import_local_modules(repo_root)
    project_root = repo_root / "scratch/court_det_fix"
    packet_dir = repo_root / "scratch/court_det_fix/evidence/holistic_admission/directional_20260921_r5" / ARM_NAME
    evaluation_dir = packet_dir.parent / "evaluation"
    gallery_dir = packet_dir / "gallery"
    render_dir = evaluation_dir / "gallery"
    verifier = import_verifier(project_root)
    diagnostics = read_json_gz(packet_dir / "reference_diagnostics.json.gz")
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    camera_limit = float(manifest["global_parameters"]["camera_error_limit_historical"])
    with (packet_dir / "per_view.csv").open(newline="", encoding="utf-8") as stream:
        case_ids = [row["case_id"] for row in csv.DictReader(stream)]
    if len(case_ids) != 27 or len(set(case_ids)) != 27:
        raise ValueError("expected the 27-case directional packet")

    rows: list[dict[str, Any]] = []
    rendered_paths: set[Path] = set()
    for case_id in case_ids:
        record = read_json_gz(packet_dir / "case_records" / f"{case_id}.json.gz")
        candidates = candidate_map(record)
        provisional_rank = record["rankings"]["C"]["provisional_rank"]
        fullcourt = rank_filter(
            case_id,
            provisional_rank,
            candidates,
            lambda candidate: historical_fullcourt(candidate, camera_limit),
            gallery_dir,
        )
        filters = {
            "historical_fullcourt": fullcourt,
            "g0_plus_g1": rank_filter(
                case_id,
                provisional_rank,
                candidates,
                lambda candidate: bool(source_memberships(candidate) & {"G0", "G1"}),
                gallery_dir,
            ),
            "g0_only": rank_filter(
                case_id,
                provisional_rank,
                candidates,
                lambda candidate: "G0" in source_memberships(candidate),
                gallery_dir,
            ),
            "g1_only": rank_filter(
                case_id,
                provisional_rank,
                candidates,
                lambda candidate: "G1" in source_memberships(candidate),
                gallery_dir,
            ),
            "g1_plus_line_template_fullcourt": rank_filter(
                case_id,
                provisional_rank,
                candidates,
                lambda candidate: historical_fullcourt(candidate, camera_limit)
                and bool(source_memberships(candidate) & {"G1", "line_template"}),
                gallery_dir,
            ),
            "g0_plus_line_template_fullcourt": rank_filter(
                case_id,
                provisional_rank,
                candidates,
                lambda candidate: historical_fullcourt(candidate, camera_limit)
                and bool(source_memberships(candidate) & {"G0", "line_template"}),
                gallery_dir,
            ),
        }
        frame = None
        context = None
        for filter_name in EVALUATION_FILTERS:
            result = filters[filter_name]
            winner = result["winner"]
            candidate = candidates.get(winner) if winner is not None else None
            image_path = None
            image_source = "none"
            if winner is not None:
                existing_path = gallery_dir / result["gallery_full"]
                if existing_path.is_file():
                    image_path = existing_path
                    image_source = "arm_gallery"
                else:
                    if frame is None:
                        source = verifier["load_source"](project_root, case_id)
                        provenance = verifier["load_case_provenance"](project_root, case_id)
                        frame = load_native_frame(project_root, source, provenance, verifier)
                        context = {
                            "id": case_id,
                            "dimensions": source["dimensions"],
                            "working_dimensions": record["provenance"]["working_dimensions"],
                        }
                    image_path = render_missing(
                        frame,
                        candidate,
                        case_id,
                        filter_name,
                        context,
                        render_dir,
                        safe_name,
                        render_prediction,
                    )
                    image_source = "evaluation_gallery"
                    rendered_paths.add(image_path)
            scalar = candidate_scalar(candidate)
            rows.append(
                {
                    "case_id": case_id,
                    "filter": filter_name,
                    "winner": winner,
                    "original_rank": result["original_rank"],
                    **scalar,
                    "eligible_count": result["eligible_count"],
                    "image_path": None if image_path is None else image_path.relative_to(repo_root).as_posix(),
                    "image_source": image_source,
                    "g0_g1_best_frozen_reference_error": best_legacy_reference_error(
                        case_id, diagnostics, candidates
                    ),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "schema": "w5-ablation-winner-render/1",
                "arm": ARM_NAME,
                "case_count": len(case_ids),
                "filter_count": len(EVALUATION_FILTERS),
                "rendered_unique_full_images": len(rendered_paths),
                "rows": rows,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return {
        "case_count": len(case_ids),
        "rows": len(rows),
        "rendered_unique_full_images": len(rendered_paths),
        "render_dir": render_dir.relative_to(repo_root).as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repo_root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
