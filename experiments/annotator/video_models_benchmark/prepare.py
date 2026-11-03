"""Decode each required frame once into a reusable, memory-mapped RGB cache."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    load_manifest,
)

from .data import (
    IMAGE_SIZE,
    context_intervals,
    development_videos,
    read_json,
    target_labels,
    write_json,
)


def selection_expression(intervals: list[list[int]]) -> str:
    """Keep FFmpeg's expression depth logarithmic in the number of rally intervals."""
    terms = [f"between(n,{start},{end - 1})" for start, end in intervals]
    while len(terms) > 1:
        paired = []
        for index in range(0, len(terms), 2):
            if index + 1 < len(terms):
                paired.append(f"({terms[index]}+{terms[index + 1]})")
            else:
                paired.append(terms[index])
        terms = paired
    return terms[0]


def prepare_video(video: dict, videos_root: Path, cache: Path) -> dict:
    directory = cache / video["item_id"]
    directory.mkdir(parents=True, exist_ok=True)
    intervals = context_intervals(video)
    frames = sum(end - start for start, end in intervals)
    metadata = {"image_size": IMAGE_SIZE, "intervals": intervals, "frames": frames, "padding": "black"}
    destination = directory / "frames.rgb"
    receipt = directory / "frames.json.gz"
    expected_bytes = frames * IMAGE_SIZE**2 * 3
    if receipt.exists():
        if read_json(receipt) != metadata or destination.stat().st_size != expected_bytes:
            raise ValueError(f"{directory}: incompatible or incomplete cache; use a new directory")
        return {"item_id": video["item_id"], "reused": True, "frames": frames}
    selection = selection_expression(intervals)
    filters = (
        f"select='{selection}',scale={IMAGE_SIZE}:{IMAGE_SIZE}:force_original_aspect_ratio=decrease,"
        f"pad={IMAGE_SIZE}:{IMAGE_SIZE}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )
    temporary = directory / "frames.partial"
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-y", "-threads", "2",
        "-i", str(videos_root / Path(video["source_video"]).name),
        "-vf", filters, "-vsync", "0", "-frames:v", str(frames), "-an", "-sn",
        "-pix_fmt", "rgb24", "-threads", "1", "-filter_threads", "1", "-f", "rawvideo", str(temporary),
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    if temporary.stat().st_size != expected_bytes:
        raise ValueError(f"{video['item_id']}: decoded fewer frames than the supplied timeline requires")
    temporary.replace(destination)
    write_json(receipt, metadata)
    return {"item_id": video["item_id"], "reused": False, "frames": frames, "seconds": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--spans", type=Path, help="Frozen development spans")
    inputs.add_argument("--test-inputs", type=Path, help="Directory containing the frozen test manifest")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke-rallies", type=int, help="Separate bounded cache: first N rallies of each video")
    args = parser.parse_args()
    if args.test_inputs is not None:
        videos = [{**item, "role": "test", "source_video": item["source_filename"]}
                  for item in load_manifest(args.test_inputs)["items"]]
        counts = {"test": {}}
    else:
        videos = development_videos(args.spans)
        counts = {"train": {}, "validation": {}}
    for role, role_counts in counts.items():
        selected = [video for video in videos if video["role"] == role]
        role_counts["centres"] = sum(span["end_frame"] - span["start_frame"] for video in selected for span in video["spans"])
        for scheme in ("A", "B"):
            role_counts[scheme] = sum(int((target_labels(span, scheme) != -100).sum())
                                       for video in selected for span in video["spans"])
    if args.smoke_rallies is not None:
        videos = [{**video, "spans": video["spans"][:args.smoke_rallies]} for video in videos]
    args.cache.mkdir(parents=True, exist_ok=True)
    index = {"videos": videos, "smoke_rallies": args.smoke_rallies, "full_counts": counts}
    index_path = args.cache / "index.json.gz"
    if index_path.exists() and read_json(index_path) != index:
        raise ValueError("Cache has different input spans; use a separate cache directory")
    required_bytes = 0
    for video in videos:
        if not (args.cache / video["item_id"] / "frames.json.gz").exists():
            required_bytes += sum(end - start for start, end in context_intervals(video)) * IMAGE_SIZE**2 * 3
    if shutil.disk_usage(args.cache).free < required_bytes:
        raise ValueError(f"Cache requires {required_bytes / 1024**3:.1f} GiB free disk")
    write_json(index_path, index)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_video, video, args.videos, args.cache) for video in videos]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
