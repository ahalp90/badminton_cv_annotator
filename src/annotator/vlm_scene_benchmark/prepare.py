"""Prepare frame-mapped whole-shard videos and cut manifests for Issue 38."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

import cv2

from annotator.composition_mask import detect_cuts
from annotator.config import COMPOSITION_CONTENT_THRESHOLD
from annotator.fps_constants import scale_for_fps

from .contracts import ShardSpec
from .runtime import sha256_file


MANIFEST_SCHEMA_VERSION = 1


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} keys differ; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context} must be an integer at least {minimum}")
    return value


def _number(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _sha256(value: Any, context: str) -> str:
    raw = _string(value, context)
    if len(raw) != 64 or any(character not in "0123456789abcdef" for character in raw):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return raw


def _file_name(value: Any, context: str) -> str:
    raw = _string(value, context)
    if Path(raw).name != raw:
        raise ValueError(f"{context} must be a file name without a directory")
    return raw


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class VideoFileSpec:
    """Reloadable identity and media properties for one video file."""

    file_name: str
    sha256: str
    fps: float
    frame_count: int
    width: int
    height: int

    def __post_init__(self) -> None:
        _file_name(self.file_name, "video.file_name")
        _sha256(self.sha256, "video.sha256")
        _number(self.fps, "video.fps", positive=True)
        _integer(self.frame_count, "video.frame_count", minimum=1)
        _integer(self.width, "video.width", minimum=1)
        _integer(self.height, "video.height", minimum=1)

    @classmethod
    def from_json(cls, value: Any, context: str) -> VideoFileSpec:
        data = _object(value, context)
        _exact_keys(data, {"file_name", "sha256", "fps", "frame_count", "width", "height"}, context)
        return cls(
            file_name=_file_name(data["file_name"], f"{context}.file_name"),
            sha256=_sha256(data["sha256"], f"{context}.sha256"),
            fps=_number(data["fps"], f"{context}.fps", positive=True),
            frame_count=_integer(data["frame_count"], f"{context}.frame_count", minimum=1),
            width=_integer(data["width"], f"{context}.width", minimum=1),
            height=_integer(data["height"], f"{context}.height", minimum=1),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "sha256": self.sha256,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class PreparedShardManifest:
    """Source, prepared-video, sampling, and cut evidence for one run."""

    shard: ShardSpec
    original_source: VideoFileSpec
    reference_video: VideoFileSpec
    model_video: VideoFileSpec
    sampled_source_frames: tuple[int, ...]
    cut_frames: tuple[int, ...]
    content_threshold: float
    min_scene_len: int
    ffmpeg_version: str
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported prepared manifest schema {self.schema_version}")
        shard_frames = self.shard.end_frame - self.shard.start_frame
        if self.original_source.frame_count != self.shard.frame_count:
            raise ValueError("original source frame count differs from shard metadata")
        if not math.isclose(self.original_source.fps, self.shard.fps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("original source FPS differs from shard metadata")
        if self.reference_video.frame_count != shard_frames:
            raise ValueError("reference video must preserve every shard frame")
        if not math.isclose(self.reference_video.fps, self.shard.fps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("reference video must preserve the source FPS")
        if self.model_video.frame_count != len(self.sampled_source_frames):
            raise ValueError("model video frame count differs from the source-frame map")
        if self.shard.source_file != self.original_source.file_name:
            raise ValueError("shard source_file must name the original source video")
        if self.shard.source_sha256 != self.original_source.sha256:
            raise ValueError("shard source digest must identify the original source video")
        if self.shard.prepared_input_file != self.model_video.file_name:
            raise ValueError("shard prepared_input_file must name the prepared model video")
        if self.shard.prepared_input_sha256 != self.model_video.sha256:
            raise ValueError("shard prepared input digest must identify the prepared model video")
        if not self.sampled_source_frames:
            raise ValueError("sampled source-frame map must not be empty")
        if any(
            not self.shard.start_frame <= frame < self.shard.end_frame
            for frame in self.sampled_source_frames
        ):
            raise ValueError("sampled source-frame map leaves the shard")
        if any(
            right <= left
            for left, right in zip(self.sampled_source_frames, self.sampled_source_frames[1:])
        ):
            raise ValueError("sampled source-frame map must be strictly increasing")
        sampling_ratio = self.shard.fps / self.model_video.fps
        sampling_stride = round(sampling_ratio)
        if sampling_stride < 1 or not math.isclose(
            sampling_ratio,
            sampling_stride,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("model video FPS must divide the source FPS exactly")
        expected_sampled_frames = tuple(
            range(self.shard.start_frame, self.shard.end_frame, sampling_stride)
        )
        if self.sampled_source_frames != expected_sampled_frames:
            raise ValueError("sampled source-frame map differs from the declared uniform grid")
        if any(not self.shard.start_frame < frame < self.shard.end_frame for frame in self.cut_frames):
            raise ValueError("cut frames must be internal to the shard")
        if any(right <= left for left, right in zip(self.cut_frames, self.cut_frames[1:])):
            raise ValueError("cut frames must be strictly increasing")
        _number(self.content_threshold, "content_threshold", positive=True)
        _integer(self.min_scene_len, "min_scene_len", minimum=1)
        _string(self.ffmpeg_version, "ffmpeg_version")

    @classmethod
    def from_json(cls, value: Any) -> PreparedShardManifest:
        data = _object(value, "prepared manifest")
        _exact_keys(
            data,
            {
                "schema_version",
                "shard",
                "original_source",
                "reference_video",
                "model_video",
                "sampled_source_frames",
                "cut_frames",
                "content_threshold",
                "min_scene_len",
                "ffmpeg_version",
            },
            "prepared manifest",
        )
        sampled = data["sampled_source_frames"]
        cuts = data["cut_frames"]
        if not isinstance(sampled, list) or not isinstance(cuts, list):
            raise ValueError("sampled_source_frames and cut_frames must be JSON arrays")
        return cls(
            schema_version=_integer(data["schema_version"], "schema_version", minimum=1),
            shard=ShardSpec.from_json(data["shard"]),
            original_source=VideoFileSpec.from_json(data["original_source"], "original_source"),
            reference_video=VideoFileSpec.from_json(data["reference_video"], "reference_video"),
            model_video=VideoFileSpec.from_json(data["model_video"], "model_video"),
            sampled_source_frames=tuple(
                _integer(frame, f"sampled_source_frames[{index}]") for index, frame in enumerate(sampled)
            ),
            cut_frames=tuple(
                _integer(frame, f"cut_frames[{index}]", minimum=1) for index, frame in enumerate(cuts)
            ),
            content_threshold=_number(data["content_threshold"], "content_threshold", positive=True),
            min_scene_len=_integer(data["min_scene_len"], "min_scene_len", minimum=1),
            ffmpeg_version=_string(data["ffmpeg_version"], "ffmpeg_version"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shard": self.shard.to_json(),
            "original_source": self.original_source.to_json(),
            "reference_video": self.reference_video.to_json(),
            "model_video": self.model_video.to_json(),
            "sampled_source_frames": list(self.sampled_source_frames),
            "cut_frames": list(self.cut_frames),
            "content_threshold": self.content_threshold,
            "min_scene_len": self.min_scene_len,
            "ffmpeg_version": self.ffmpeg_version,
        }


def probe_video(path: Path) -> VideoFileSpec:
    """Read stable OpenCV metadata and hash one local video."""
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    return VideoFileSpec(path.name, sha256_file(path), fps, frame_count, width, height)


def ffmpeg_version() -> str:
    completed = subprocess.run(
        ["ffmpeg", "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg -version failed")
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    return _string(first_line, "ffmpeg version output")


def _run_ffmpeg(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed: {message}")


def _reference_command(
    source: Path,
    output: Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: float,
    width: int,
    height: int,
) -> list[str]:
    return [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS,scale={width}:{height}:flags=lanczos",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{fps:.12g}",
        "-frames:v",
        str(end_frame - start_frame),
        "-movflags",
        "+faststart",
        str(output),
    ]


def _sample_command(
    reference: Path,
    output: Path,
    *,
    stride: int,
    sample_fps: float,
    expected_frames: int,
) -> list[str]:
    return [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(reference),
        "-map",
        "0:v:0",
        "-vf",
        f"select=not(mod(n\\,{stride})),setpts=N/({sample_fps:.12g}*TB)",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        f"{sample_fps:.12g}",
        "-frames:v",
        str(expected_frames),
        "-movflags",
        "+faststart",
        str(output),
    ]


def _assert_video(
    video: VideoFileSpec,
    *,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    context: str,
) -> None:
    if not math.isclose(video.fps, fps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{context} FPS {video.fps} differs from expected {fps}")
    if (video.frame_count, video.width, video.height) != (frame_count, width, height):
        raise ValueError(
            f"{context} media {(video.frame_count, video.width, video.height)} differs from "
            f"expected {(frame_count, width, height)}"
        )


def write_manifest(path: Path, manifest: PreparedShardManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest.to_json(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if read_manifest(temporary) != manifest:
            raise RuntimeError(f"prepared manifest round trip changed values: {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> PreparedShardManifest:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs_without_duplicates)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}") from error
    return PreparedShardManifest.from_json(value)


def resolve_model_video(manifest_path: Path, manifest: PreparedShardManifest) -> Path:
    """Resolve and verify the exact model input adjacent to its manifest."""
    video_path = Path(manifest_path).parent / manifest.model_video.file_name
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    probed = probe_video(video_path)
    if probed != manifest.model_video:
        raise ValueError(f"model video metadata differs from manifest: {video_path}")
    return video_path


def prepare_shard(
    source_path: Path,
    output_dir: Path,
    *,
    video_id: str,
    start_frame: int,
    end_frame: int,
    sample_fps: float = 1.0,
    width: int = 512,
    height: int = 288,
    content_threshold: float = COMPOSITION_CONTENT_THRESHOLD,
) -> Path:
    """Encode one exact-frame reference and one frame-mapped model input."""
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    source = probe_video(source_path)
    if not 0 <= start_frame < end_frame <= source.frame_count:
        raise ValueError(
            f"shard [{start_frame}, {end_frame}) is outside source [0, {source.frame_count})"
        )
    ratio = source.fps / sample_fps
    stride = round(ratio)
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("source FPS must be an integer multiple of the requested sample FPS")
    sampled_source_frames = tuple(range(start_frame, end_frame, stride))
    stem = f"{video_id}_f{start_frame}_f{end_frame}_{width}x{height}"
    reference_name = f"{stem}_reference_{source.fps:.12g}fps.mp4"
    model_name = f"{stem}_model_{sample_fps:.12g}fps.mp4"
    manifest_name = f"{stem}_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [output_dir / reference_name, output_dir / model_name, output_dir / manifest_name]
    existing = [path for path in final_paths if path.exists()]
    if existing:
        raise FileExistsError(f"prepared output already exists: {existing[0]}")

    with TemporaryDirectory(prefix="vlm-prepare-", dir=output_dir) as temporary_dir:
        temporary_root = Path(temporary_dir)
        reference_path = temporary_root / reference_name
        model_path = temporary_root / model_name
        _run_ffmpeg(
            _reference_command(
                source_path,
                reference_path,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=source.fps,
                width=width,
                height=height,
            )
        )
        reference = probe_video(reference_path)
        _assert_video(
            reference,
            fps=source.fps,
            frame_count=end_frame - start_frame,
            width=width,
            height=height,
            context="reference video",
        )
        _run_ffmpeg(
            _sample_command(
                reference_path,
                model_path,
                stride=stride,
                sample_fps=sample_fps,
                expected_frames=len(sampled_source_frames),
            )
        )
        model_video = probe_video(model_path)
        _assert_video(
            model_video,
            fps=sample_fps,
            frame_count=len(sampled_source_frames),
            width=width,
            height=height,
            context="model video",
        )
        min_scene_len = scale_for_fps(source.fps).composition_min_scene_len
        local_cuts = detect_cuts(reference_path, reference.frame_count, content_threshold, min_scene_len)
        cut_frames = tuple(start_frame + int(frame) for frame in local_cuts)
        shard = ShardSpec(
            video_id=video_id,
            source_file=source.file_name,
            source_sha256=source.sha256,
            prepared_input_file=model_video.file_name,
            prepared_input_sha256=model_video.sha256,
            fps=source.fps,
            frame_count=source.frame_count,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        manifest = PreparedShardManifest(
            shard=shard,
            original_source=source,
            reference_video=reference,
            model_video=model_video,
            sampled_source_frames=sampled_source_frames,
            cut_frames=cut_frames,
            content_threshold=content_threshold,
            min_scene_len=min_scene_len,
            ffmpeg_version=ffmpeg_version(),
        )
        published_paths: list[Path] = []
        try:
            final_reference = output_dir / reference_name
            reference_path.replace(final_reference)
            published_paths.append(final_reference)
            final_model = output_dir / model_name
            model_path.replace(final_model)
            published_paths.append(final_model)
            write_manifest(output_dir / manifest_name, manifest)
        except BaseException:
            for published_path in reversed(published_paths):
                published_path.unlink(missing_ok=True)
            raise
    return output_dir / manifest_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--content-threshold", type=float, default=COMPOSITION_CONTENT_THRESHOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare_shard(
            args.source,
            args.output_dir,
            video_id=args.video_id,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            sample_fps=args.sample_fps,
            width=args.width,
            height=args.height,
            content_threshold=args.content_threshold,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        build_parser().error(str(error))
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
