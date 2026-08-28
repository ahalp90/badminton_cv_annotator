"""Run InpaintNet on the saved ShuttleSet22 TrackNet coordinates."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import io
import json
import lzma
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

VIDEO_IDS = (
    8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57,
)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SEQUENCE_LENGTH = 16
STRIDE = 8
GPU_BATCH_SIZE = 16
COORDINATE_THRESHOLD = 50 / math.sqrt(288**2 + 512**2)
CHECKPOINT_SHA256 = "5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c"
MODEL_CODE_SHA256 = "ab75c790228026095b44be107787d1529ba88ab5b01956600fad132fc0bf0edf"
GUARD_CODE_SHA256 = "574e3600bbca704a856618d43ebe70522a8d3aef4fc26285fbe0853dc9333844"
RUN_STATE_FILENAME = "inpaint_run_state.json"
RECEIPT_FILENAME = "inpaint_result.json.gz"


@dataclass(frozen=True)
class VideoInput:
    """One checked video and its prepared inputs."""

    video_id: int
    directory: Path
    source_video: Path
    csv_path: Path
    track_path: Path

    @property
    def name(self) -> str:
        return self.directory.name


@dataclass(frozen=True)
class OutputPaths:
    """New files written for one video."""

    csv_path: Path
    sidecar_path: Path
    track_path: Path
    guard_codes_path: Path
    guard_diagnostics_path: Path
    receipt_path: Path

    def files(self) -> tuple[Path, ...]:
        return (
            self.csv_path,
            self.sidecar_path,
            self.track_path,
            self.guard_codes_path,
            self.guard_diagnostics_path,
            self.receipt_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-src", type=Path, required=True)
    parser.add_argument("--video-id", type=int, choices=VIDEO_IDS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def discover_videos(input_root: Path, source_root: Path) -> list[VideoInput]:
    directories_by_id: dict[int, list[Path]] = {video_id: [] for video_id in VIDEO_IDS}
    for directory in sorted(input_root.iterdir()):
        if not directory.is_dir():
            continue
        prefix = directory.name.split(" ", maxsplit=1)[0]
        if prefix.isdigit() and int(prefix) in directories_by_id:
            directories_by_id[int(prefix)].append(directory)

    problems = {
        video_id: len(directories)
        for video_id, directories in directories_by_id.items()
        if len(directories) != 1
    }
    if problems:
        raise ValueError(f"Expected one prepared directory per fixed video, got {problems}")

    videos = []
    for video_id in VIDEO_IDS:
        directory = directories_by_id[video_id][0]
        source_video = source_root / f"{directory.name}.mp4"
        csv_path = directory / f"{directory.name}_ball.csv.gz"
        track_path = directory / "shuttle_track.npy.xz"
        for path in (source_video, csv_path, track_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        videos.append(VideoInput(video_id, directory, source_video, csv_path, track_path))
    return videos


def output_paths(directory: Path, video_name: str) -> OutputPaths:
    return OutputPaths(
        csv_path=directory / f"{video_name}_ball_inpainted.csv.gz",
        sidecar_path=directory / f"{video_name}_stride{STRIDE}_inpaint_mask.json.gz",
        track_path=directory / "shuttle_track_inpainted.npy.xz",
        guard_codes_path=directory / "shuttle_guard_codes_inpainted.npy.xz",
        guard_diagnostics_path=directory / "shuttle_guard_diagnostics_inpainted.json.gz",
        receipt_path=directory / RECEIPT_FILENAME,
    )


def read_tracknet_csv(
    path: Path,
    *,
    coordinates_must_be_in_frame: bool = True,
) -> dict[str, np.ndarray]:
    columns = {name: [] for name in ("Frame", "X", "Y", "Visibility")}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or set(columns) - set(reader.fieldnames):
            raise ValueError(f"TrackNet CSV columns differ in {path.name}")
        for row in reader:
            for name, values in columns.items():
                values.append(int(row[name]))

    arrays = {name: np.asarray(values, dtype=np.int64) for name, values in columns.items()}
    frame_count = len(arrays["Frame"])
    if not np.array_equal(arrays["Frame"], np.arange(frame_count, dtype=np.int64)):
        raise ValueError(f"TrackNet frames are not contiguous from zero in {path.name}")
    if not np.isin(arrays["Visibility"], (0, 1)).all():
        raise ValueError(f"TrackNet visibility is not binary in {path.name}")
    invisible = arrays["Visibility"] == 0
    if np.any(arrays["X"][invisible] != 0) or np.any(arrays["Y"][invisible] != 0):
        raise ValueError(f"Invisible TrackNet rows have non-zero coordinates in {path.name}")
    if coordinates_must_be_in_frame:
        if np.any(arrays["X"] < 0) or np.any(arrays["X"] >= FRAME_WIDTH):
            raise ValueError(f"TrackNet x coordinates are outside the frame in {path.name}")
        if np.any(arrays["Y"] < 0) or np.any(arrays["Y"] >= FRAME_HEIGHT):
            raise ValueError(f"TrackNet y coordinates are outside the frame in {path.name}")
    return arrays


def generate_inpaint_mask(y: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """Match TrackNetV3's saved-coordinate mask calculation."""
    mask = np.zeros_like(y, dtype=np.float32)
    gap_start = 0
    gap_end = 0
    threshold = FRAME_HEIGHT * 0.05
    while gap_end < len(visibility):
        while gap_start < len(visibility) - 1 and visibility[gap_start] == 1:
            gap_start += 1
        gap_end = gap_start
        while gap_end < len(visibility) - 1 and visibility[gap_end] == 0:
            gap_end += 1
        if gap_end == gap_start:
            break
        if gap_start == 0 and y[gap_end] > threshold:
            mask[:gap_end] = 1
        elif (
            gap_start > 1
            and y[gap_start - 1] > threshold
            and gap_end < len(visibility)
            and y[gap_end] > threshold
        ):
            mask[gap_start:gap_end] = 1
        gap_start = gap_end
    return mask


def validate_base_track(arrays: dict[str, np.ndarray], track_path: Path) -> None:
    saved_track = load_npy_xz(track_path)
    expected_track = normalised_track(arrays["X"], arrays["Y"], arrays["Visibility"])
    if not np.array_equal(saved_track, expected_track):
        raise ValueError(f"Saved shuttle track differs from its base CSV in {track_path.parent.name}")


def normalised_track(x: np.ndarray, y: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            x.astype(float) / FRAME_WIDTH,
            y.astype(float) / FRAME_HEIGHT,
            visibility.astype(float),
        )
    )


def load_npy_xz(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as input_file:
        return np.load(input_file, allow_pickle=False)


def load_model(checkpoint_path: Path, repo_src: Path) -> tuple[Any, Any]:
    if sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("InpaintNet checkpoint hash differs")
    model_path = repo_src / "shared" / "tracknetv3" / "model.py"
    guard_path = repo_src / "annotator" / "inpaint_guard.py"
    if sha256(model_path) != MODEL_CODE_SHA256:
        raise ValueError("InpaintNet model code hash differs")
    if sha256(guard_path) != GUARD_CODE_SHA256:
        raise ValueError("Shuttle guard code hash differs")

    import torch

    tracknet_code = model_path.parent
    sys.path.insert(0, str(tracknet_code))
    model_module = importlib.import_module("model")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint["param_dict"]["seq_len"]) != SEQUENCE_LENGTH:
        raise ValueError("InpaintNet checkpoint sequence length differs")
    model = model_module.InpaintNet().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    return model, torch


def run_inpaint(
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
    model: Any,
    torch: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = len(arrays["Frame"])
    padding = (-frame_count) % SEQUENCE_LENGTH
    x = np.pad(arrays["X"], (0, padding), mode="edge")
    y = np.pad(arrays["Y"], (0, padding), mode="edge")
    padded_mask = np.pad(mask, (0, padding), mode="edge")
    coordinates = np.stack((x / FRAME_WIDTH, y / FRAME_HEIGHT), axis=1).astype(np.float32)
    coordinate_sequences = coordinates.reshape(-1, SEQUENCE_LENGTH, 2)
    mask_sequences = padded_mask.reshape(-1, SEQUENCE_LENGTH, 1).astype(np.float32)
    output_sequences = np.empty_like(coordinate_sequences)

    for start in range(0, len(coordinate_sequences), GPU_BATCH_SIZE):
        stop = min(start + GPU_BATCH_SIZE, len(coordinate_sequences))
        coordinate_tensor = torch.from_numpy(coordinate_sequences[start:stop]).float()
        mask_tensor = torch.from_numpy(mask_sequences[start:stop]).float()
        with torch.no_grad():
            inpainted = model(coordinate_tensor.cuda(), mask_tensor.cuda()).detach().cpu()
            inpainted = inpainted * mask_tensor + coordinate_tensor * (1 - mask_tensor)
        near_origin = (inpainted[:, :, 0] < COORDINATE_THRESHOLD) & (
            inpainted[:, :, 1] < COORDINATE_THRESHOLD
        )
        inpainted[near_origin] = 0
        output_sequences[start:stop] = inpainted.numpy()

    torch.cuda.synchronize()
    output = output_sequences.reshape(-1, 2)[:frame_count]
    output_x = (output[:, 0] * FRAME_WIDTH).astype(np.int64)
    output_y = (output[:, 1] * FRAME_HEIGHT).astype(np.int64)
    output_visibility = ((output_x != 0) | (output_y != 0)).astype(np.int64)
    return output_x, output_y, output_visibility


def mask_spans(mask: np.ndarray) -> list[list[int]]:
    selected = mask.astype(bool)
    edges = np.diff(np.concatenate(([False], selected, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [[int(start), int(stop)] for start, stop in zip(starts, stops)]


def json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot save {type(value).__name__} as JSON")


def gzip_json_bytes(payload: dict[str, object]) -> bytes:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return gzip.compress(encoded, compresslevel=9, mtime=0)


def gzip_csv_bytes(frames: np.ndarray, x: np.ndarray, y: np.ndarray, visibility: np.ndarray) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("Frame", "Visibility", "X", "Y"))
    writer.writerows(zip(frames, visibility, x, y))
    return gzip.compress(buffer.getvalue().encode("utf-8"), compresslevel=9, mtime=0)


def npy_xz_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with lzma.open(buffer, "wb", format=lzma.FORMAT_XZ, preset=9) as output_file:
        np.save(output_file, array, allow_pickle=False)
    return buffer.getvalue()


def write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as output_file:
        output_file.write(payload)


def create_links(input_directory: Path, working_directory: Path) -> None:
    for source in sorted(input_directory.iterdir()):
        if source.is_file():
            (working_directory / source.name).symlink_to(source.resolve())


def guard_outputs(track: np.ndarray, repo_src: Path) -> tuple[np.ndarray, dict[str, object]]:
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
    guard_module = importlib.import_module("annotator.inpaint_guard")
    codes, info = guard_module.grade_track(track)
    counts = guard_module.code_counts(codes)
    diagnostics = {
        "schema": "shuttle-guard/0.1",
        "frame_count": len(track),
        "counts_per_code": {str(code): count for code, count in counts.items()},
        "detector": json_ready(info),
    }
    return codes, diagnostics


def process_video(
    video: VideoInput,
    output_root: Path,
    checkpoint_path: Path,
    repo_src: Path,
    model: Any,
    torch: Any,
    run_utc: str,
) -> dict[str, object]:
    final_directory = output_root / video.name
    working_directory = output_root / f"{video.name}.working"
    if final_directory.exists() or working_directory.exists():
        raise FileExistsError(f"Output directory already exists for video {video.video_id}")
    working_directory.mkdir()
    create_links(video.directory, working_directory)
    paths = output_paths(working_directory, video.name)
    started = time.perf_counter()

    arrays = read_tracknet_csv(video.csv_path)
    validate_base_track(arrays, video.track_path)
    mask = generate_inpaint_mask(arrays["Y"], arrays["Visibility"])
    output_x, output_y, output_visibility = run_inpaint(arrays, mask, model, torch)
    track = normalised_track(output_x, output_y, output_visibility)
    guard_codes, guard_diagnostics = guard_outputs(track, repo_src)

    sidecar: dict[str, object] = {
        "schema": "inpaint_fill_mask/1",
        "index_space": "frame",
        "inpaint_status": "applied",
        "n_rows": len(arrays["Frame"]),
        "eval_mode": "nonoverlap",
        "stride": STRIDE,
        "th_h_px": FRAME_HEIGHT * 0.05,
        "tracknet_ckpt": "TrackNet_best.pt",
        "inpaintnet_ckpt": checkpoint_path.name,
        "input_video": video.source_video.name,
        "extracted_utc": run_utc,
        "inpaint_selected": mask_spans(mask),
    }
    write_bytes(paths.csv_path, gzip_csv_bytes(arrays["Frame"], output_x, output_y, output_visibility))
    write_bytes(paths.sidecar_path, gzip_json_bytes(sidecar))
    write_bytes(paths.track_path, npy_xz_bytes(track))
    write_bytes(paths.guard_codes_path, npy_xz_bytes(guard_codes))
    write_bytes(paths.guard_diagnostics_path, gzip_json_bytes(guard_diagnostics))

    output_hashes = {path.name: sha256(path) for path in paths.files() if path != paths.receipt_path}
    receipt: dict[str, object] = {
        "schema": "shuttleset22-inpaint-result/1",
        "status": "complete",
        "video_id": video.video_id,
        "video_name": video.name,
        "method": "InpaintNet from saved TrackNet coordinates",
        "run_utc": run_utc,
        "frame_count": len(arrays["Frame"]),
        "selected_frame_count": int(mask.sum()),
        "visible_before": int(arrays["Visibility"].sum()),
        "visible_after": int(output_visibility.sum()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_code_sha256": MODEL_CODE_SHA256,
        "guard_code_sha256": GUARD_CODE_SHA256,
        "input_hashes": {
            video.csv_path.name: sha256(video.csv_path),
            video.track_path.name: sha256(video.track_path),
        },
        "output_hashes": output_hashes,
        "guard_counts": guard_diagnostics["counts_per_code"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_bytes(paths.receipt_path, gzip_json_bytes(receipt))
    os.replace(working_directory, final_directory)
    return receipt


def read_json_gz(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected one JSON object in {path.name}")
    return payload


def validate_completed(video: VideoInput, output_root: Path, repo_src: Path) -> dict[str, object] | None:
    directory = output_root / video.name
    if not directory.exists():
        return None
    receipt_path = directory / RECEIPT_FILENAME
    if not receipt_path.is_file():
        raise ValueError(f"Completed directory has no receipt for video {video.video_id}")
    receipt = read_json_gz(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("video_id") != video.video_id:
        raise ValueError(f"Receipt identity differs for video {video.video_id}")
    expected_code_hashes = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_code_sha256": MODEL_CODE_SHA256,
        "guard_code_sha256": GUARD_CODE_SHA256,
    }
    if any(receipt.get(name) != expected_hash for name, expected_hash in expected_code_hashes.items()):
        raise ValueError(f"Receipt model identity differs for video {video.video_id}")
    input_hashes = receipt.get("input_hashes")
    output_hashes = receipt.get("output_hashes")
    if not isinstance(input_hashes, dict) or not isinstance(output_hashes, dict):
        raise TypeError(f"Receipt hashes are missing for video {video.video_id}")
    expected_inputs = {
        video.csv_path.name: sha256(video.csv_path),
        video.track_path.name: sha256(video.track_path),
    }
    if input_hashes != expected_inputs:
        raise ValueError(f"Prepared input hash differs for video {video.video_id}")
    paths = output_paths(directory, video.name)
    expected_output_names = {path.name for path in paths.files() if path != paths.receipt_path}
    if set(output_hashes) != expected_output_names:
        raise ValueError(f"Receipt output list differs for video {video.video_id}")
    for name, expected_hash in output_hashes.items():
        path = directory / str(name)
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"Completed output hash differs for video {video.video_id}: {name}")

    arrays = read_tracknet_csv(paths.csv_path, coordinates_must_be_in_frame=False)
    validate_base_track(arrays, paths.track_path)
    if receipt.get("frame_count") != len(arrays["Frame"]):
        raise ValueError(f"Completed frame count differs for video {video.video_id}")
    sidecar = read_json_gz(paths.sidecar_path)
    required_sidecar_values = {
        "schema": "inpaint_fill_mask/1",
        "index_space": "frame",
        "inpaint_status": "applied",
        "n_rows": len(arrays["Frame"]),
        "eval_mode": "nonoverlap",
        "stride": STRIDE,
        "th_h_px": FRAME_HEIGHT * 0.05,
        "inpaintnet_ckpt": "InpaintNet_best.pt",
        "input_video": video.source_video.name,
    }
    if any(sidecar.get(name) != value for name, value in required_sidecar_values.items()):
        raise ValueError(f"Completed sidecar differs for video {video.video_id}")
    spans = sidecar.get("inpaint_selected")
    if not isinstance(spans, list):
        raise TypeError(f"Completed sidecar spans are missing for video {video.video_id}")
    selected_count = sum(int(stop) - int(start) for start, stop in spans)
    if receipt.get("selected_frame_count") != selected_count:
        raise ValueError(f"Completed selected-frame count differs for video {video.video_id}")

    track = load_npy_xz(paths.track_path)
    saved_guard_codes = load_npy_xz(paths.guard_codes_path)
    expected_guard_codes, expected_guard_diagnostics = guard_outputs(track, repo_src)
    if not np.array_equal(saved_guard_codes, expected_guard_codes):
        raise ValueError(f"Completed guard codes differ for video {video.video_id}")
    if read_json_gz(paths.guard_diagnostics_path) != expected_guard_diagnostics:
        raise ValueError(f"Completed guard diagnostics differ for video {video.video_id}")
    return receipt


def write_run_state(path: Path, run_utc: str, receipts: list[dict[str, object]]) -> None:
    payload: dict[str, object] = {
        "schema": "shuttleset22-inpaint-run/1",
        "status": "complete" if len(receipts) == len(VIDEO_IDS) else "running",
        "run_utc": run_utc,
        "expected_video_ids": list(VIDEO_IDS),
        "completed_video_ids": [receipt["video_id"] for receipt in receipts],
        "completed_count": len(receipts),
        "frame_count": sum(int(receipt["frame_count"]) for receipt in receipts),
        "selected_frame_count": sum(int(receipt["selected_frame_count"]) for receipt in receipts),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    videos = discover_videos(args.input_root, args.source_root)
    if args.video_id is not None:
        videos = [video for video in videos if video.video_id == args.video_id]

    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("InpaintNet checkpoint hash differs")
    if args.dry_run:
        for video in videos:
            arrays = read_tracknet_csv(video.csv_path)
            validate_base_track(arrays, video.track_path)
            print(f"checked video {video.video_id}: {len(arrays['Frame'])} frames")
        print(f"dry run complete: {len(videos)} videos")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / RUN_STATE_FILENAME
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        run_utc = str(state["run_utc"])
    else:
        run_utc = utc_now()

    all_videos = discover_videos(args.input_root, args.source_root)
    receipts = []
    for video in all_videos:
        receipt = validate_completed(video, args.output_root, args.repo_src)
        if receipt is not None:
            receipts.append(receipt)
    write_run_state(state_path, run_utc, receipts)

    pending = [video for video in videos if not (args.output_root / video.name).exists()]
    if not pending:
        print("all requested videos are already complete")
        return
    model, torch = load_model(args.checkpoint, args.repo_src)
    for video in pending:
        receipt = process_video(
            video,
            args.output_root,
            args.checkpoint,
            args.repo_src,
            model,
            torch,
            run_utc,
        )
        receipts.append(receipt)
        receipts.sort(key=lambda item: int(item["video_id"]))
        write_run_state(state_path, run_utc, receipts)
        print(
            f"completed video {video.video_id}: {receipt['frame_count']} frames, "
            f"{float(receipt['elapsed_seconds']):.1f} seconds",
            flush=True,
        )


if __name__ == "__main__":
    main()
