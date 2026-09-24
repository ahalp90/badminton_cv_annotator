"""Seated removed, then the top N standing tracks by summed motion, for several N (throwaway).

Usage: build_mover_caps.py PEOPLE_DIR SHOT_CHECK_JSONL REPO_ROOT OUTPUT_DIR N...
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

import build_feet_variants as variants


def main() -> None:
    people_dir, shot_check, repo_root, output_dir = (Path(argument) for argument in sys.argv[1:5])
    caps = [int(argument) for argument in sys.argv[5:]]
    differences = {row["case_id"]: row["differences"] for row in map(json.loads, shot_check.read_text().splitlines())}
    sys.path[:0] = [str(repo_root / "src"), str(repo_root / "src/bst_x")]
    from preparing_data.heuristics.base import SITTING_THRESHOLD, is_sitting  # pyrefly: ignore[missing-import]

    by_cap = {cap: {} for cap in caps}
    for path in sorted(people_dir.glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            record = json.load(stream)
        pack_scale = np.asarray(record["pack_size"]) / np.asarray(record["video_size"])
        frames = [sample["frame_index"] for sample in record["samples"]]
        in_shot = variants.same_shot_samples(differences[record["case_id"]], frames.index(record["anchor"]))
        feet, heights = [], []
        for sample in (record["samples"][index] for index in in_shot):
            boxes = np.asarray(sample["bboxes"], dtype=float).reshape(-1, 4)
            keypoints = np.asarray(sample["keypoints"], dtype=float).reshape(-1, 17, 2)
            standing = ~is_sitting(keypoints, SITTING_THRESHOLD)
            x1, y1, x2, y2 = boxes[standing].T
            feet.append(np.column_stack(((x1 + x2) / 2, y2)) * pack_scale)
            heights.append((y2 - y1) * pack_scale[1])
        track_ids, motion = variants.link_tracks(feet, heights)
        ranked = sorted(motion, key=lambda track: -motion[track])
        for cap in caps:
            kept = np.asarray(ranked[:cap])
            keep = [np.isin(ids, kept) for ids in track_ids]
            by_cap[cap][record["case_id"]] = variants.pack_feet(feet, keep, record["pack_size"])
    for cap, feet_by_case in by_cap.items():
        with gzip.open(output_dir / f"feet_standing_movers{cap}.json.gz", "wt") as stream:
            json.dump(feet_by_case, stream)


if __name__ == "__main__":
    main()
