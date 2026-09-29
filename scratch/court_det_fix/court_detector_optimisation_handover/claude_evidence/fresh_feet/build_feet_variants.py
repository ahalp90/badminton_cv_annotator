"""Build the four player-feet variants for follow-up items 10 and 11 from the window detections.

Throwaway evaluation script. Every variant starts from the same detections (one 3 s window at
10 fps per view) and writes {case_id: all_feet_px} in the view pack's pixels:

- everyone: every detection
- standing: seated detections removed with sticky_anchor's rule (is_sitting at -0.3)
- movers: only the detections on the six tracks that move most, in body heights
- standing_movers: seated removed first, then the six top movers

A foot is the bottom-centre of a person box; a foot outside the image is unknown, as in the
frozen GX, broadcast and control packs. Only samples in the anchor's shot count: the unbroken
run of samples around the anchor whose 64x36 grey thumbnail stays within SAME_SHOT_GREY_LEVELS
of the anchor's (from shot_check.py).

Usage: build_feet_variants.py PEOPLE_DIR SHOT_CHECK_JSONL REPO_ROOT OUTPUT_DIR
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# Largest foot move between samples 0.1 s apart that still counts as the same person, in body
# heights. A sprinting player covers about a third of a body height per sample.
MAX_STEP_HEIGHTS = 1.0
TOP_MOVERS = 6
UNMATCHABLE = 1e6
# In-shot samples of the 20 court views differ from their anchor by at most 5.1 grey levels;
# the dissolve in control frame 1 and the passer-by at the lens in am3 frame 0 exceed 9.
SAME_SHOT_GREY_LEVELS = 8.0


def same_shot_samples(differences: list[float], anchor_index: int) -> range:
    """The unbroken run of sample indexes around the anchor that stay in the anchor's shot."""
    first = anchor_index
    while first > 0 and differences[first - 1] <= SAME_SHOT_GREY_LEVELS:
        first -= 1
    last = anchor_index
    while last < len(differences) - 1 and differences[last + 1] <= SAME_SHOT_GREY_LEVELS:
        last += 1
    return range(first, last + 1)


def link_tracks(feet: list[np.ndarray], heights: list[np.ndarray]) -> tuple[list[np.ndarray], dict[int, float]]:
    """Track ID for each detection in each sample, and each track's summed motion in body heights.

    Detections in consecutive samples are matched by foot distance over mean box height; a match
    further than MAX_STEP_HEIGHTS starts a new track. A missed detection ends its track.
    """
    track_ids, motion = [], {}
    next_id = 0
    for sample_index, (sample_feet, sample_heights) in enumerate(zip(feet, heights, strict=True)):
        ids = np.full(len(sample_feet), -1)
        if sample_index and len(sample_feet) and len(feet[sample_index - 1]):
            previous_feet, previous_heights = feet[sample_index - 1], heights[sample_index - 1]
            distance = np.linalg.norm(previous_feet[:, None] - sample_feet[None], axis=2)
            steps = distance / ((previous_heights[:, None] + sample_heights[None]) / 2)
            gated = np.where(steps <= MAX_STEP_HEIGHTS, steps, UNMATCHABLE)
            previous_rows, rows = linear_sum_assignment(gated)
            for previous_row, row in zip(previous_rows, rows, strict=True):
                if gated[previous_row, row] < UNMATCHABLE:
                    ids[row] = track_ids[-1][previous_row]
                    motion[ids[row]] += steps[previous_row, row]
        for row in np.flatnonzero(ids < 0):
            ids[row] = next_id
            motion[next_id] = 0.0
            next_id += 1
        track_ids.append(ids)
    return track_ids, motion


def top_movers(feet: list[np.ndarray], heights: list[np.ndarray]) -> list[np.ndarray]:
    """Per sample, a mask of the detections on the TOP_MOVERS most-moving tracks."""
    track_ids, motion = link_tracks(feet, heights)
    ranked = sorted(motion, key=lambda track: -motion[track])
    kept = np.asarray(ranked[:TOP_MOVERS])
    return [np.isin(ids, kept) for ids in track_ids]


def pack_feet(feet: list[np.ndarray], keep: list[np.ndarray], pack_size: list[int]) -> list[list]:
    """all_feet_px rows padded with None to one width, at least two slots as in the frozen packs."""
    width, height = pack_size
    rows = []
    for sample_feet, sample_keep in zip(feet, keep, strict=True):
        row = []
        for foot_x, foot_y in sample_feet[sample_keep]:
            on_image = 0 <= foot_x < width and 0 <= foot_y < height
            row.append([float(foot_x), float(foot_y)] if on_image else None)
        rows.append(row)
    slots = max(2, *(len(row) for row in rows))
    return [row + [None] * (slots - len(row)) for row in rows]


def main() -> None:
    people_dir, shot_check, repo_root, output_dir = (Path(argument) for argument in sys.argv[1:])
    differences = {row["case_id"]: row["differences"]
                   for row in map(json.loads, shot_check.read_text().splitlines())}
    sys.path[:0] = [str(repo_root / "src"), str(repo_root / "src/bst_x")]
    from preparing_data.heuristics.base import (  # pyrefly: ignore[missing-import]
        SITTING_THRESHOLD,
        is_sitting,
    )

    variants = {name: {} for name in ("everyone", "standing", "movers", "standing_movers")}
    stats = {}
    for path in sorted(people_dir.glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            record = json.load(stream)
        pack_scale = np.asarray(record["pack_size"]) / np.asarray(record["video_size"])
        frames = [sample["frame_index"] for sample in record["samples"]]
        in_shot = same_shot_samples(differences[record["case_id"]], frames.index(record["anchor"]))
        feet, heights, standing = [], [], []
        for sample in (record["samples"][index] for index in in_shot):
            boxes = np.asarray(sample["bboxes"], dtype=float).reshape(-1, 4)
            x1, y1, x2, y2 = boxes.T
            # Pack pixels for the feet; motion is in body heights, so the scale cancels there.
            feet.append(np.column_stack(((x1 + x2) / 2, y2)) * pack_scale)
            heights.append((y2 - y1) * pack_scale[1])
            keypoints = np.asarray(sample["keypoints"], dtype=float).reshape(-1, 17, 2)
            standing.append(~is_sitting(keypoints, SITTING_THRESHOLD))
        everyone = [np.ones(len(sample_feet), dtype=bool) for sample_feet in feet]
        movers = top_movers(feet, heights)
        standing_feet = [sample_feet[mask] for sample_feet, mask in zip(feet, standing, strict=True)]
        standing_heights = [sample_heights[mask] for sample_heights, mask in zip(heights, standing, strict=True)]
        standing_movers_within = top_movers(standing_feet, standing_heights)
        # Map "top mover among the standing" back onto every detection in the sample.
        standing_movers = []
        for mask, within in zip(standing, standing_movers_within, strict=True):
            full = np.zeros(len(mask), dtype=bool)
            full[np.flatnonzero(mask)[within]] = True
            standing_movers.append(full)
        case_id = record["case_id"]
        masks = {"everyone": everyone, "standing": standing, "movers": movers, "standing_movers": standing_movers}
        for name, keep in masks.items():
            variants[name][case_id] = pack_feet(feet, keep, record["pack_size"])
        stats[case_id] = {name: [int(mask.sum()) for mask in keep] for name, keep in masks.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, feet_by_case in variants.items():
        with gzip.open(output_dir / f"feet_{name}.json.gz", "wt") as stream:
            json.dump(feet_by_case, stream)
    (output_dir / "detections_kept_per_sample.json").write_text(json.dumps(stats, indent=1))
    for case_id, counts in stats.items():
        summary = {name: f"{np.median(values):.0f} ({min(values)}-{max(values)})" for name, values in counts.items()}
        print(f"{case_id:38s} {summary}")


if __name__ == "__main__":
    main()
