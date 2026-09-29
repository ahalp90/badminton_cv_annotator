"""Draw returned courts from a comparison's results on the frames they fit.

Each image is the original frame with one court as a plain 1 px red dashed outline.
Courts with equal corners on the same frame share one image. Its file name carries
the first label, and index.tsv lists every label that shares it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from court_detector.video_inputs import VideoFrames

from .sampling import read_json

RED_BGR = (0, 0, 255)
DASH_PX = 6
DASH_PERIOD_PX = 12  # a 6 px dash, then a 6 px gap
CORNER_DECIMALS = 3  # corners equal to this many decimals share one image


def draw_outline(frame: np.ndarray, corners: list[list[float]]) -> None:
    """Draw the court's four edges, clipped to the frame, as 1 px dashes."""
    height, width = frame.shape[:2]
    points = np.asarray(corners)
    for first, last in zip(points, np.roll(points, -1, axis=0), strict=True):
        visible, start, end = cv2.clipLine((0, 0, width, height), tuple(np.rint(first).astype(int)),
                                           tuple(np.rint(last).astype(int)))
        if not visible:
            continue
        start_array, end_array = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        length = float(np.linalg.norm(end_array - start_array))
        if length == 0:
            continue
        direction = (end_array - start_array) / length
        for offset in np.arange(0, length, DASH_PERIOD_PX):
            dash_start = tuple(np.rint(start_array + direction * offset).astype(int))
            dash_end = tuple(np.rint(start_array + direction * min(offset + DASH_PX, length)).astype(int))
            cv2.line(frame, dash_start, dash_end, RED_BGR, 1, cv2.LINE_8)


def overlay_items(results: dict[str, Any], scene_ids: set[str]) -> Iterator[tuple[str, int, str, list]]:
    """(video path, frame index, label, corners) for every returned court in the selected scenes.

    A label reads scene__arm__route_role_frame, with __chosen on the court each method selected.
    """
    for video in results["videos"]:
        for scene in video["scenes"] + video["later_scenes"]:
            scene_id = scene["scene_id"]
            if scene_ids and scene_id not in scene_ids:
                continue
            baseline = scene.get("baseline", {})
            if baseline.get("status") == "court":
                route = "full_search" if baseline["reused_from"] is None else "history_reuse"
                label = f'{scene_id}__baseline__{route}_middle_{scene["midpoint_frame"]}'
                yield video["video"], scene["midpoint_frame"], label, baseline["corners_native_px"]
            for method, outcome in scene.get("methods", {}).items():
                for position, frame in enumerate(outcome.get("frames", [])):
                    if frame["status"] != "court":
                        continue
                    label = f'{scene_id}__{method}__{frame["route"]}_{frame["role"]}_{frame["frame_index"]}'
                    if position == outcome["chosen_position"]:
                        label += "__chosen"
                    yield video["video"], frame["frame_index"], label, frame["corners_native_px"]


def render(results_path: Path, output_dir: Path, scene_ids: set[str]) -> int:
    """Write the images and index.tsv; return the number of images."""
    labels: dict[tuple, list[str]] = defaultdict(list)
    corners_by_key: dict[tuple, list] = {}
    for video, frame_index, label, corners in overlay_items(read_json(results_path), scene_ids):
        key = (video, frame_index, tuple(np.round(corners, CORNER_DECIMALS).ravel()))
        labels[key].append(label)
        corners_by_key[key] = corners
    output_dir.mkdir(parents=True, exist_ok=False)
    by_video: dict[str, list[tuple]] = defaultdict(list)
    for key in labels:
        by_video[key[0]].append(key)
    index = []
    for video, keys in by_video.items():
        with VideoFrames(Path(video)) as frames:
            for key in sorted(keys, key=lambda item: item[1]):
                frame, = frames.read([key[1]])
                image = frame.copy()
                draw_outline(image, corners_by_key[key])
                name = f"{labels[key][0]}.png"
                if not cv2.imwrite(str(output_dir / name), image):
                    raise OSError(f"Could not write {name}")
                index.extend(f"{name}\t{label}" for label in labels[key])
    (output_dir / "index.tsv").write_text("image\tlabel\n" + "".join(f"{line}\n" for line in index))
    return len(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="the comparison's results.json.gz")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for the images")
    parser.add_argument("--scene", action="append", default=[], help="scene ID to draw; repeat; default all")
    args = parser.parse_args()
    count = render(args.results, args.output_dir, set(args.scene))
    print(f"Wrote {count} images of plain 1 px red dashed court outlines to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
