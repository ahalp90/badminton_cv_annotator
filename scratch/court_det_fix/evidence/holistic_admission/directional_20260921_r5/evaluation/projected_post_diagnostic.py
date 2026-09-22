"""Record-only projected-net scalars for the 43-arm C selections."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ARM_NAME = "w5_directional_20260921_r5_43"
SELECTIONS = ("base43", "historical_fullcourt")


def import_local_modules(repo_root: Path):
    evaluation_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(evaluation_dir))
    from audit_43_gates import candidate_map, historical_fullcourt, read_json_gz

    from experiments.annotator.independent_court.net_geometry import project_net

    return candidate_map, historical_fullcourt, project_net, read_json_gz


def post_scalars(segments: np.ndarray, native_height: float) -> dict[str, dict[str, float]]:
    posts: dict[str, dict[str, float]] = {}
    for side, segment in zip(("left", "right"), segments[2:]):
        base, top = segment
        direction = top - base
        length = float(np.linalg.norm(direction))
        posts[side] = {
            "dx_px": float(direction[0]),
            "dy_px": float(direction[1]),
            "length_px": length,
            "length_over_native_height": length / native_height,
        }
    return posts


def run(repo_root: Path, output_path: Path) -> dict[str, Any]:
    candidate_map, historical_fullcourt, project_net, read_json_gz = import_local_modules(repo_root)
    packet_dir = (
        repo_root
        / "scratch/court_det_fix/evidence/holistic_admission/directional_20260921_r5"
        / ARM_NAME
    )
    manifest = json.loads((packet_dir / "manifest.json").read_text(encoding="utf-8"))
    camera_limit = float(manifest["global_parameters"]["camera_error_limit_historical"])
    with (packet_dir / "per_view.csv").open(newline="", encoding="utf-8") as stream:
        case_ids = [row["case_id"] for row in csv.DictReader(stream)]
    if len(case_ids) != 27 or len(set(case_ids)) != 27:
        raise ValueError("expected the 27-case directional packet")

    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        record = read_json_gz(packet_dir / "case_records" / f"{case_id}.json.gz")
        candidates = candidate_map(record)
        provisional_rank = record["rankings"]["C"]["provisional_rank"]
        base_key = record["rankings"]["C"]["selected_origin_key"]
        fullcourt_key = next(
            (
                origin_key
                for origin_key in provisional_rank
                if historical_fullcourt(candidates[origin_key], camera_limit)
            ),
            None,
        )
        selected_keys = {"base43": base_key, "historical_fullcourt": fullcourt_key}
        native_width, native_height = record["provenance"]["native_dimensions"]
        for selection in SELECTIONS:
            origin_key = selected_keys[selection]
            candidate = candidates.get(origin_key) if origin_key is not None else None
            row: dict[str, Any] = {
                "case_id": case_id,
                "selection": selection,
                "origin_key": origin_key,
                "provisional_rank": None
                if origin_key is None
                else provisional_rank.index(origin_key) + 1,
                "source": None if candidate is None else candidate.get("source"),
                "source_memberships": None
                if candidate is None
                else candidate.get("source_memberships"),
                "candidate_camera_error": None
                if candidate is None
                else candidate.get("gates", {}).get("camera_error"),
                "player_fractions": None
                if candidate is None
                else candidate.get("gates", {}).get("player_fractions"),
                "native_dimensions": [native_width, native_height],
                "projected_camera_error": None,
                "focal_widths": None,
                "post": None,
                "projection_error": None,
            }
            if candidate is not None:
                try:
                    projection = project_net(candidate["corners_px"], (native_width, native_height))
                except ValueError as error:
                    row["projection_error"] = str(error)
                else:
                    row["projected_camera_error"] = projection.camera_error
                    row["focal_widths"] = projection.focal_widths
                    row["post"] = post_scalars(projection.segments_px, float(native_height))
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "schema": "w5-projected-post-diagnostic/1",
                "arm": ARM_NAME,
                "camera_error_limit_historical": camera_limit,
                "selection_geometry_space": "native_px",
                "case_count": len(case_ids),
                "selection_count": len(SELECTIONS),
                "projection_failure_count": sum(row["projection_error"] is not None for row in rows),
                "rows": rows,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return {
        "case_count": len(case_ids),
        "rows": len(rows),
        "projection_failure_count": sum(row["projection_error"] is not None for row in rows),
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
