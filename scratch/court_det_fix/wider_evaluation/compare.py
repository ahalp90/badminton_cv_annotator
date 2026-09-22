"""Compare full and G1/template access using one saved W5 ranking per view."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from run_cases import load_runtime, write

BASELINE = "evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_43"


def selections(record: dict) -> dict:
    candidates = {candidate["origin_key"]: candidate for candidate in record["parents"] + record["valid_children"]}
    ranking = record["rankings"]["C"]["provisional_rank"]
    results = {}
    for arm in ("full", "g1_templates"):
        allowed = ranking if arm == "full" else [
            key for key in ranking if set(candidates[key]["source_memberships"]) & {"G1", "line_template"}
        ]
        gated = [key for key in allowed if candidates[key]["historical"]["historical_fullcourt"]]
        gated_keys = set(gated)
        results[arm] = {
            "camera_eligible_ranked_count": len(allowed), "player_gated_count": len(gated),
            "ungated": allowed[0] if allowed else None, "gated": gated[0] if gated else None,
            "rejected_by_player_gate": [key for key in allowed if key not in gated_keys],
        }
    return results


def render(root: Path, gallery: Path, row: dict, record: dict, selected: dict, verifier) -> dict:
    from experiments.annotator.independent_court import detector, paint_geometry

    source = verifier.load_source(root, row["case_id"])
    frame = cv2.imread(str(root / row["image"]))
    if frame is None:
        raise FileNotFoundError(root / row["image"])
    native_size = np.array([frame.shape[1], frame.shape[0]])
    working_size = np.array(record["provenance"]["working_dimensions"])
    candidates = {candidate["origin_key"]: candidate for candidate in record["parents"] + record["valid_children"]}
    roles = {}
    for arm, result in selected.items():
        for gate in ("ungated", "gated"):
            key = result[gate]
            if key is not None:
                roles.setdefault(key, []).append(f"{arm}/{gate}")
    case_dir = gallery / row["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(case_dir / "raw.png"), frame):
        raise OSError(case_dir / "raw.png")
    links = {}
    for index, (key, candidate_roles) in enumerate(roles.items()):
        candidate = candidates[key]
        canvas = frame.copy()
        homography = np.asarray(candidate["homography_working"], dtype=float)
        markings, _ = detector.project(homography[None], paint_geometry.CENTRE_SEGMENTS_M)
        markings = markings[0].reshape(-1, 2, 2) * (native_size / working_size)
        for start, end in markings:
            cv2.line(canvas, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
                     (255, 190, 0), 1, cv2.LINE_AA)
        full = case_dir / f"selection_{index}.png"
        if not cv2.imwrite(str(full), canvas):
            raise OSError(full)
        corners = np.asarray(candidate["corners_px"], dtype=float)
        ordered_y = np.sort(corners[:, 1])
        lower = max(0, min(frame.shape[0] - 1, int(ordered_y[:2].min()) - 40))
        upper = max(lower + 1, min(frame.shape[0], int((ordered_y[1] + ordered_y[2]) / 2) + 20))
        far = np.concatenate((frame[lower:upper], canvas[lower:upper]), axis=1)
        far_path = case_dir / f"selection_{index}_far.png"
        if not cv2.imwrite(str(far_path), far):
            raise OSError(far_path)
        links[key] = {"roles": candidate_roles, "full": str(full.relative_to(gallery)),
                      "far_raw_and_overlay": str(far_path.relative_to(gallery)),
                      "corners_px": candidate["corners_px"], "player_fractions": candidate["gates"]["player_fractions"],
                      "source_memberships": candidate["source_memberships"], "visual_ruling": "pending"}
    return {"raw": str((case_dir / "raw.png").relative_to(gallery)), "selections": links,
            "player_observed_frames": len(source["all_feet_px"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--control-pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    _, verifier, _ = load_runtime(root, args.control_pack)
    manifest = verifier.read_json_gz(args.manifest)
    rows = []
    missing = []
    for row in manifest["cases"]:
        base = root / BASELINE if row["previous_w5_case"] else args.run
        path = base / "case_records" / f"{row['case_id']}.json.gz"
        if not path.is_file():
            missing.append(row["case_id"])
            continue
        record = verifier.read_json_gz(path)
        selected = selections(record)
        result = {"case_id": row["case_id"], "group": row["group"], "arm": row["arm"],
                  "record": str(path), "population_counts": record["population_counts"],
                  "selections": selected, "reference_status": row.get("reference_status"),
                  "view_status": row.get("view_status"), "previous_w5_case": row["previous_w5_case"]}
        if args.render:
            result["gallery"] = render(root, args.output.parent / "gallery", row, record, selected, verifier)
        rows.append(result)
    write(args.output, {"schema": "wider-w5-comparison/1", "cases": rows, "missing_cases": missing,
                        "complete": not missing, "planned_case_count": len(manifest["cases"])})
    print(json.dumps({"completed": len(rows), "missing": len(missing)}))


if __name__ == "__main__":
    main()
