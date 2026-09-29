"""Render four saved GX5 person-observation candidates without refitting geometry."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any

import cv2

HERE = Path(__file__).resolve().parent
COURT_DET_FIX = HERE.parents[3]
RUN = HERE / "runs/person_observations_repair_20260922/matcher"
OUTPUT = HERE / "gallery/gx5_extra"
CASE_ID = "gxBQ_window_00_frame_5"


def read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def find_entry(record: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for entry in record["entries"]:
        if entry["candidate_id"] == candidate_id:
            return entry
    raise KeyError(f"{CASE_ID}: {candidate_id} is not in the saved entry pool")


def find_shortlist_entry(record: dict[str, Any], pair_id: int, candidate_id: str) -> dict[str, Any]:
    pair = next(pair for pair in record["pairs"] if pair["pair_id"] == pair_id)
    for entry in pair["shortlist"]:
        if entry["candidate_id"] == candidate_id:
            return entry
    raise KeyError(f"{CASE_ID}: {candidate_id} is not in results pair {pair_id} shortlist")


def main() -> None:
    sys.path[:0] = [
        str(COURT_DET_FIX / "line_identity"),
        str(COURT_DET_FIX / "w5_holistic"),
        str(COURT_DET_FIX.parents[1]),
        str(COURT_DET_FIX.parents[1] / "src"),
    ]
    from render_gallery import render_prediction

    import shared

    shared.add_helper_paths()
    results = read_json_gz(RUN / f"person_observations/results/{CASE_ID}.json.gz")
    all_camera = read_json_gz(RUN / f"person_observations/all_camera/{CASE_ID}.json.gz")
    source = shared.load_source(CASE_ID)
    frame_path = shared.frame_path(source)
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(frame_path)

    context = {
        "id": CASE_ID,
        "dimensions": source["dimensions"],
        "working_dimensions": results["working_size"],
    }
    targets = (
        (
            "nearest_pre_global17_836",
            "results_pair17_shortlist",
            find_shortlist_entry(results, 17, "17:836"),
        ),
        (
            "all_camera_nearest25_2615",
            "all_camera",
            find_entry(all_camera, "25:2615"),
        ),
        (
            "all_camera_line226_417",
            "all_camera",
            find_entry(all_camera, "226:417"),
        ),
        (
            "all_camera_paint46_1676",
            "all_camera",
            find_entry(all_camera, "46:1676"),
        ),
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, stage, saved_entry in targets:
        candidate = {**saved_entry, "origin_key": f"{stage}:{saved_entry['candidate_id']}"}
        output_stem = OUTPUT / f"{CASE_ID}__{name}"
        render_prediction(frame, candidate, context, [name], output_stem, None)
        print(
            json.dumps(
                {
                    "candidate_id": candidate["candidate_id"],
                    "stage": stage,
                    "gates": candidate.get("gates"),
                    "full": str(output_stem.with_name(output_stem.name + "__full.png")),
                    "crop": str(output_stem.with_name(output_stem.name + "__crop.png")),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
