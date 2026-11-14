"""Prepare height224 aspect-preserving frames in JPEG95 4:2:0 ZIP archives."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from PIL import Image

from experiments.annotator.monotrack_benchmark.benchmark_scripts.common import (
    load_manifest,
)
from experiments.annotator.video_models_benchmark.data import (
    development_videos,
    read_json,
    write_json,
)
from experiments.annotator.video_models_benchmark.prepare import selection_expression

from .data import merge_intervals


def source_dimensions(path: Path) -> tuple[int, int]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
               "-of", "json", str(path)]
    stream = json.loads(subprocess.check_output(command))["streams"][0]
    return stream["width"], stream["height"]


def preparation_intervals(video: dict, mode: str) -> list[list[int]]:
    if mode == "discovery":
        return [[0, video["frame_count"]]]
    return merge_intervals([[span["start_frame"], span["end_frame"]] for span in video["spans"]])


def prepare_video(video: dict, videos: Path, cache: Path, mode: str) -> dict:
    """Atomically finish one video at a time; reuse completed video archives."""
    directory = cache / video["item_id"]
    directory.mkdir(parents=True, exist_ok=True)
    source = videos / Path(video["source_video"]).name
    source_width, source_height = source_dimensions(source)
    width = int(source_width / source_height * 224)
    if width < 224:
        raise ValueError("Expected landscape source video for the authors' horizontal crop")
    intervals = preparation_intervals(video, mode)
    count = sum(end - start for start, end in intervals)
    metadata = {"storage": "jpeg_zip", "geometry": "height224-aspect", "image_height": 224, "image_width": width,
                "intervals": intervals, "frames": count, "jpeg_quality": 95, "jpeg_subsampling": 2}
    receipt = directory / "frames.json.gz"
    destination = directory / "frames.zip"
    if receipt.exists():
        if read_json(receipt) != metadata:
            raise ValueError("Cache metadata differs; choose a new cache directory")
        with ZipFile(destination) as archive:
            if len(archive.infolist()) != count:
                raise ValueError("Incomplete frame archive")
        return {"item_id": video["item_id"], "reused": True, "frames": count}
    temporary = directory / "frames.zip.partial"
    filters = f"select='{selection_expression(intervals)}',scale={width}:224:flags=bilinear,setsar=1"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-i", str(source),
               "-vf", filters, "-vsync", "0", "-an", "-sn", "-pix_fmt", "rgb24", "-threads", "1",
               "-filter_threads", "1", "-f", "rawvideo", "pipe:1"]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        try:
            assert process.stdout is not None
            with ZipFile(temporary, "w", compression=ZIP_STORED, allowZip64=True) as archive:
                for start, end in intervals:
                    for frame in range(start, end):
                        pixels = process.stdout.read(224 * width * 3)
                        if len(pixels) != 224 * width * 3:
                            raise ValueError(f"{video['item_id']}: source ended before selected frame {frame}")
                        output = BytesIO()
                        Image.frombytes("RGB", (width, 224), pixels).save(output, format="JPEG", quality=95, subsampling=2)
                        archive.writestr(f"{frame:09d}.jpg", output.getvalue())
            if process.stdout.read(1):
                raise ValueError(f"{video['item_id']}: source differs from frozen frame selection")
            if process.wait() != 0:
                raise RuntimeError("FFmpeg decoding failed")
        except BaseException:
            process.kill()
            process.wait()
            raise
    temporary.replace(destination)
    write_json(receipt, metadata)
    return {"item_id": video["item_id"], "reused": False, "frames": count, "bytes": destination.stat().st_size}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spans", type=Path, help="Canonical development spans (32 train, 8 validation)")
    source.add_argument("--test-inputs", type=Path, help="Frozen 46-video test manifest; explicit test preparation only")
    source.add_argument("--index", type=Path, help="Reuse only metadata from an existing index.json.gz")
    parser.add_argument("--mode", choices=("supplied", "discovery"), required=True)
    parser.add_argument("--videos", type=Path, required=True, help="Canonical source-video directory")
    parser.add_argument("--cache", type=Path, required=True, help="Fresh cache; old letterbox frames are incompatible")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.spans:
        videos = development_videos(args.spans)
    elif args.test_inputs:
        videos = [{**item, "role": "test", "source_video": item["source_filename"]}
                  for item in load_manifest(args.test_inputs)["items"]]
    else:
        videos = read_json(args.index)["videos"]
    args.cache.mkdir(parents=True, exist_ok=True)
    index_path = args.cache / "index.json.gz"
    if index_path.exists() and read_json(index_path)["videos"] != videos:
        raise ValueError("Cache index differs; choose a separate directory")
    if not index_path.exists():
        write_json(index_path, {"videos": videos})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_video, video, args.videos, args.cache, args.mode) for video in videos]
        for future in futures:
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
