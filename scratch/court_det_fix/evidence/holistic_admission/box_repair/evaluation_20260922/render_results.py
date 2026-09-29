"""Render the corrected matcher selections and retained nearest-control candidate."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUN = HERE / "runs/person_observations_repair_20260922/matcher"


def read(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def main() -> None:
    sys.path[:0] = [str(ROOT / "line_identity"), str(ROOT / "w5_holistic"),
                   str(ROOT.parents[1]), str(ROOT.parents[1] / "src")]
    from render_gallery import render_prediction

    import shared

    diagnosis = read(RUN / "diagnosis.json.gz")
    for diagnosed in diagnosis["records"]:
        row = diagnosed["accounting"]
        if row["stage"] != "results":
            continue
        case_id = row["case_id"]
        source = shared.load_source(case_id)
        frame = cv2.imread(str(shared.frame_path(source)), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(shared.frame_path(source))
        record = read(RUN / "person_observations/results" / f"{case_id}.json.gz")
        entries = {entry["candidate_id"]: entry for entry in record["entries"]}
        context = {"id": case_id, "dimensions": source["dimensions"],
                   "working_dimensions": record["working_size"]}
        for role, field in (("line", "line_winner_id"), ("paint", "paint_winner_id"),
                            ("nearest_control_diagnostic", "nearest_final_camera_eligible_id")):
            candidate_id = row[field]
            if candidate_id is None:
                continue
            entry = {**entries[candidate_id], "origin_key": f"person:{candidate_id}"}
            stem = HERE / "gallery" / f"{case_id}__{role}"
            stem.parent.mkdir(exist_ok=True)
            render_prediction(frame, entry, context, [role], stem, None)
            print(case_id, role, candidate_id)


if __name__ == "__main__":
    main()
