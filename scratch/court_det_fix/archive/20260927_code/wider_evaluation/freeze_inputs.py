"""Freeze the 47 detector cases and separate 24 broadcast review controls."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def fingerprint(path: Path) -> str | None:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else None


def freeze(root: Path, control_pack: Path | None = None) -> dict:
    repo = root.parents[1]
    sys.path[:0] = [str(repo), str(repo / "src"), str(root / "w5_holistic")]
    import verifier

    from experiments.annotator.independent_court.case_provenance import (
        load_frozen_case_provenance,
    )

    rows = []
    packs = {}
    for group, relative in verifier.CASE_PACKS.items():
        pack_path = root / relative
        pack = read(pack_path)
        provenance = load_frozen_case_provenance(pack_path)
        packs[relative] = fingerprint(pack_path)
        for source in pack["cases"]:
            case_id = source["id"]
            image = verifier.frame_path(root, source, provenance[case_id])
            feet = source["all_feet_px"]
            video = source.get("provenance", {}).get("source_video")
            if video is None:
                video = "_".join(case_id.split("_")[:2]) if group == "broadcast" else case_id.split("_")[0]
            rows.append({
                "case_id": case_id, "group": group, "arm": "frozen_detector",
                "source_pack": relative, "video": video,
                "previous_w5_case": case_id in verifier.ALL_CASE_IDS,
                "prior_use": "development input from a previously used video",
                "image": str(image.relative_to(root)), "image_md5": fingerprint(image),
                "image_kind": provenance[case_id].image_kind.value,
                "view_status": "view_unverified" if case_id in {
                    "shuttleset_21_scene_0010", "shuttleset_21_scene_0039",
                } else "ordinary_frozen_view",
                "same_image_boxes": provenance[case_id].has_same_image_boxes,
                "line_fragment_count": len(source["segments_px"]),
                "player_observed_frames": len(feet),
                "player_observations_status": "available" if feet else "missing",
            })
    control_path = repo / "experiments/annotator/independent_court/recorded/controls.json.gz"
    control_root = root / "evidence/independent_proposals/development/inputs/controls"
    packs[str(control_path.relative_to(repo))] = fingerprint(control_path)
    prepared_controls = {}
    if control_pack is not None:
        prepared_controls = {source["id"]: source for source in read(control_pack)["cases"]}
        packs[str(control_pack.relative_to(root))] = fingerprint(control_pack)
    for source in read(control_path)["inputs"]["cases"]:
        image = control_root / source["image"]
        rows.append({
            "case_id": source["id"], "group": "broadcast_controls", "arm": "rejection_review",
            "source_pack": str(control_path.relative_to(repo)), "video": source["video_id"],
            "previous_w5_case": False, "prior_use": "previously recorded control from a familiar video",
            "image": str(image.relative_to(root)), "image_md5": fingerprint(image),
            "image_kind": source["image_kind"], "frame_index": source["frame_index"],
            "reference_status": source["reference_status"],
            "line_fragment_count": None, "player_observed_frames": None,
            "player_observations_status": "not_in_recorded_control_pack",
        })
        if control_pack is not None:
            prepared = prepared_controls[source["id"]]
            rows[-1].update({
                "observation_pack": str(control_pack.relative_to(root)),
                "line_fragment_count": len(prepared["segments_px"]),
                "player_observed_frames": len(prepared["all_feet_px"]),
                "player_detection_count": len(prepared["bbox_px"]),
                "person_score_cutoff": prepared["provenance"]["person_score_cutoff"],
                "player_observations_status": "available_single_frame",
                "same_image_boxes": True,
            })
    counts = Counter(row["arm"] for row in rows)
    if counts != {"frozen_detector": 47, "rejection_review": 24}:
        raise ValueError(f"Unexpected case population: {counts}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case IDs")
    return {
        "schema": "wider-w5-inputs/1", "input_md5": packs, "cases": rows,
        "settings": {
            "visibility_floor": [4, 3], "workers": 6, "native_threads_per_worker": 1,
            "source_arms": {"full": ["G0", "G1", "line_template"], "restricted": ["G1", "line_template"]},
            "rank": "C.provisional_rank", "player_rule": "historical_fullcourt",
            "review": ["clean", "tolerable_fallback", "unacceptable", "needs_user_review"],
            "visual_ideal": "imperceptible misalignment; preferably outer edges of white markings",
            "numerical_acceptance_threshold": None,
        },
        "summary": {
            "case_count": len(rows), "arms": dict(counts),
            "previous_w5_count": sum(row["previous_w5_case"] for row in rows),
            "missing_images": [row["case_id"] for row in rows if row["image_md5"] is None],
            "control_labels": dict(Counter(row["reference_status"] for row in rows if row["arm"] == "rejection_review")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-pack", type=Path)
    args = parser.parse_args()
    result = freeze(args.root.resolve(), args.control_pack.resolve() if args.control_pack else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
