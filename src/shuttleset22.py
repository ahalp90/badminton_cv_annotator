"""Download ShuttleSet22 and run its independent whole-video extractors."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from annotator.video_metadata import VideoMetadata


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = REPO_ROOT / "configs" / "shuttleset22" / "sources.toml"
DEFAULT_TRACKNET_DIR = REPO_ROOT / "src" / "shared" / "tracknetv3"
DEFAULT_COURT_WEIGHTS = (
    REPO_ROOT / "src" / "courtkeynet" / "weights" / "courtkeynet_finetuned.safetensors"
)
COURT_RECEIPT_SCHEMA = "shuttleset22-court/0.1"
COURT_RECEIPT_FILENAME = "court_receipt.json.gz"
COURT_PARENT = "detected_ckn_opencv_consensus"
COURT_REF_ERR_PX = 3.5
YOUTUBE_FORMAT = (
    "bv*[ext=mp4][vcodec^=avc1][fps=30][height<=1080]+ba[ext=m4a]/"
    "b[ext=mp4][vcodec^=avc1][fps=30][height<=1080]"
)
EXPECTED_IDS = tuple(range(1, 59))

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class SourceKind(StrEnum):
    DOWNLOAD = "download"
    OVERLAP = "shuttleset_overlap"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Source:
    match_id: int
    video: str
    kind: SourceKind
    url: str | None = None
    overlap_id: int | None = None
    unresolved_reason: str | None = None

    @property
    def filename(self) -> str:
        return f"{self.match_id:02d} {self.video}.mp4"

    @property
    def overlap_filename(self) -> str:
        if self.overlap_id is None:
            raise ValueError(f"source {self.match_id} is not a ShuttleSet overlap")
        return f"{self.overlap_id} {self.video}.mp4"


def load_sources(path: Path = DEFAULT_SOURCES) -> tuple[Source, ...]:
    """Load the reviewed public URLs and cross-dataset overlap mapping."""
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    rows = payload.get("videos")
    if not isinstance(rows, list):
        raise ValueError("source manifest must contain [[videos]] entries")

    sources: list[Source] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"videos[{index}] must be a table")
        match_id = _integer(row.get("id"), f"videos[{index}].id")
        video = _text(row.get("video"), f"videos[{index}].video")
        if Path(video).name != video:
            raise ValueError(f"videos[{index}].video must be a basename")
        kind = SourceKind(_text(row.get("source_kind"), f"videos[{index}].source_kind"))
        url = None
        overlap_id = None
        reason = None
        if kind in {SourceKind.DOWNLOAD, SourceKind.OVERLAP}:
            url = _text(row.get("url"), f"videos[{index}].url")
        if kind is SourceKind.OVERLAP:
            overlap_id = _integer(
                row.get("overlap_shuttleset_id"),
                f"videos[{index}].overlap_shuttleset_id",
            )
        if kind is SourceKind.UNRESOLVED:
            reason = _text(
                row.get("unresolved_reason"),
                f"videos[{index}].unresolved_reason",
            )
        sources.append(Source(match_id, video, kind, url, overlap_id, reason))

    ids = tuple(source.match_id for source in sources)
    if ids != EXPECTED_IDS:
        raise ValueError(f"source IDs must be exactly 1..58, got {ids}")
    if len({source.video for source in sources}) != len(sources):
        raise ValueError("source video names must be unique")
    return tuple(sources)


def select_sources(sources: Sequence[Source], ids: Sequence[int] | None) -> tuple[Source, ...]:
    """Select explicit IDs or the default non-overlap, downloadable corpus."""
    if ids is None:
        return tuple(source for source in sources if source.kind is SourceKind.DOWNLOAD)
    if len(ids) != len(set(ids)):
        raise ValueError(f"source IDs contain duplicates: {list(ids)}")
    by_id = {source.match_id: source for source in sources}
    unknown = sorted(set(ids).difference(by_id))
    if unknown:
        raise ValueError(f"unknown source IDs: {unknown}")
    return tuple(by_id[match_id] for match_id in ids)


def download_command(
    source: Source,
    destination: Path,
    *,
    cookies_from_browser: str | None = None,
    youtube_player_client: str | None = None,
) -> tuple[str, ...]:
    if source.kind is not SourceKind.DOWNLOAD or source.url is None:
        raise ValueError(f"source {source.match_id} is not downloadable")
    command = (
        "yt-dlp",
        "--no-playlist",
        "--continue",
        "--part",
        "--no-overwrites",
        "--format",
        YOUTUBE_FORMAT,
        "--merge-output-format",
        "mp4",
        "--output",
        os.fspath(destination),
    )
    if cookies_from_browser is not None:
        command += ("--cookies-from-browser", cookies_from_browser)
    if youtube_player_client is not None:
        command += ("--extractor-args", f"youtube:player_client={youtube_player_client}")
    return (*command, source.url)


def overlap_command(source: Source, overlap_root: Path, destination: Path) -> tuple[str, ...]:
    if source.kind is not SourceKind.OVERLAP:
        raise ValueError(f"source {source.match_id} is not an overlap")
    return (
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        os.fspath(Path(overlap_root) / source.overlap_filename),
        "-map",
        "0",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        os.fspath(destination),
    )


def download_sources(
    sources: Sequence[Source],
    *,
    source_root: Path,
    overlap_root: Path,
    cookies_from_browser: str | None = None,
    youtube_player_client: str | None = None,
    command_runner: CommandRunner | None = None,
) -> int:
    """Download public sources and remux the eight existing overlap videos."""
    root = Path(source_root)
    root.mkdir(parents=True, exist_ok=True)
    runner = command_runner or _run_command
    failures = 0
    for source in sources:
        if source.kind is SourceKind.UNRESOLVED:
            print(f"{source.match_id:02d}: unavailable: {source.unresolved_reason}")
            continue
        destination = root / source.filename
        if destination.is_file():
            print(f"{source.match_id:02d}: already downloaded")
            continue
        try:
            if source.kind is SourceKind.DOWNLOAD:
                command = download_command(
                    source,
                    destination,
                    cookies_from_browser=cookies_from_browser,
                    youtube_player_client=youtube_player_client,
                )
                temporary = None
            else:
                temporary = destination.with_suffix(".part.mp4")
                temporary.unlink(missing_ok=True)
                command = overlap_command(source, overlap_root, temporary)
            print(f"{source.match_id:02d}: running {' '.join(command)}", flush=True)
            completed = runner(command)
            if completed.returncode != 0:
                raise RuntimeError(f"command exited {completed.returncode}")
            if temporary is not None:
                os.replace(temporary, destination)
            if not destination.is_file():
                raise FileNotFoundError(f"command did not produce {destination}")
        except Exception as error:
            failures += 1
            print(f"{source.match_id:02d}: FAILED: {error}", file=sys.stderr)
    return failures


def gzip_csv(path: Path) -> Path:
    """Compress one completed TrackNet CSV with gzip level 9."""
    source = Path(path)
    destination = source.with_suffix(".csv.gz")
    temporary = destination.with_suffix(".csv.gz.part")
    try:
        with source.open("rb") as input_handle, gzip.open(
            temporary,
            "wb",
            compresslevel=9,
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle)
        os.replace(temporary, destination)
        source.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def probe_source(path: Path) -> VideoMetadata:
    """Read the frame count and geometry needed by the existing extractors."""
    from annotator.video_metadata import VideoMetadata

    source = Path(path).resolve(strict=True)
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,nb_frames,width,height,sample_aspect_ratio",
            "-of",
            "json",
            os.fspath(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe failed")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise ValueError(f"ffprobe returned no single video stream for {source}")
    stream = streams[0]
    fps = Fraction(_text(stream.get("avg_frame_rate"), "avg_frame_rate"))
    if fps != 30:
        raise ValueError(f"source must be 30 FPS, got {fps}: {source}")
    sample_aspect_ratio = _ratio(stream.get("sample_aspect_ratio"), "sample_aspect_ratio")
    return VideoMetadata(
        source_path=source,
        fps=fps,
        frame_count=_integer_text(stream.get("nb_frames"), "nb_frames"),
        width=_integer(stream.get("width"), "width"),
        height=_integer(stream.get("height"), "height"),
        sample_aspect_ratio=sample_aspect_ratio,
    )


def extract_source(
    source: Source,
    *,
    source_root: Path,
    output_root: Path,
    tracknet_dir: Path,
    tracknet_python: Path,
    pose_python: Path,
    pose_shards: int,
) -> None:
    """Run missing TrackNet and RTMLib stages for one whole video."""
    from bst_x.pipeline.shuttle_extractor import extract_all_shuttles
    from dataset_builder.pose_sharding import extract_sharded_rtmlib_pose_stage
    from dataset_builder.vision import (
        convert_tracknet_csv_stage,
        pose_artifact_paths,
    )

    video = Path(source_root) / source.filename
    output = Path(output_root) / f"{source.match_id:02d} {source.video}"
    output.mkdir(parents=True, exist_ok=True)
    csv = output / f"{video.stem}_ball.csv"
    csv_gz = csv.with_suffix(".csv.gz")
    shuttle = output / "shuttle_track.npy.xz"
    pose_paths = tuple(pose_artifact_paths(output).as_mapping().values())
    if csv_gz.is_file() and shuttle.is_file() and all(path.is_file() for path in pose_paths):
        print(f"{source.match_id:02d}: already extracted")
        return

    metadata = probe_source(video)
    if not csv_gz.is_file() or not shuttle.is_file():
        csv.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tracknet-", dir=output) as temporary_text:
            temporary = Path(temporary_text)
            temporary_csv = temporary / csv.name
            extract_all_shuttles(
                tracknet_dir=Path(tracknet_dir),
                clips_dir=video.parent,
                video_paths=[video],
                output_csv_dir=temporary,
                tracknet_python=Path(tracknet_python),
                max_workers=1,
                tracknet_stride=8,
                large_video=True,
                enable_inpainting=False,
            )
            if not temporary_csv.is_file():
                raise FileNotFoundError(f"TrackNet did not produce {temporary_csv}")
            convert_tracknet_csv_stage(
                temporary_csv,
                video_id=str(source.match_id),
                metadata=metadata,
                output_path=shuttle,
            )
            os.replace(gzip_csv(temporary_csv), csv_gz)

    if not all(path.is_file() for path in pose_paths):
        extract_sharded_rtmlib_pose_stage(
            metadata=metadata,
            output_dir=output,
            interpreter=Path(pose_python),
            shards=pose_shards,
        )


def extract_sources(
    sources: Sequence[Source],
    *,
    source_root: Path,
    output_root: Path,
    tracknet_dir: Path,
    tracknet_python: Path,
    pose_python: Path,
    pose_shards: int = 8,
) -> int:
    """Run the two requested extractors over every available source."""
    failures = 0
    for source in sources:
        if source.kind is SourceKind.UNRESOLVED:
            print(f"{source.match_id:02d}: unavailable: {source.unresolved_reason}")
            continue
        try:
            print(f"{source.match_id:02d}: extracting", flush=True)
            extract_source(
                source,
                source_root=source_root,
                output_root=output_root,
                tracknet_dir=tracknet_dir,
                tracknet_python=tracknet_python,
                pose_python=pose_python,
                pose_shards=pose_shards,
            )
        except Exception as error:
            failures += 1
            print(f"{source.match_id:02d}: FAILED: {error}", file=sys.stderr)
    return failures


def _artifact_record(name: str, path: Path, root: Path) -> dict[str, object]:
    from dataset_builder.manifest import artifact_integrity

    return dict(artifact_integrity(name, path, relative_to=root).to_dict())


def _court_output_items() -> tuple[tuple[str, str], ...]:
    from dataset_builder.vision import (
        COURT_EVIDENCE_FILENAME,
        COURT_KEEP_VOTE_FILENAME,
        COURT_PRESENT_FILENAME,
    )

    return (
        ("court_evidence", COURT_EVIDENCE_FILENAME),
        ("court_keep_vote", COURT_KEEP_VOTE_FILENAME),
        ("court_present", COURT_PRESENT_FILENAME),
    )


def _court_output_identities(output: Path) -> list[dict[str, object]]:
    return [
        _artifact_record(name, output / filename, output)
        for name, filename in _court_output_items()
    ]


def _court_receipt_base(
    source: Source,
    *,
    source_root: Path,
    output: Path,
    metadata: VideoMetadata,
    model_identity: dict[str, object],
    device: str,
    resize_mode: str,
    code_id: str,
) -> dict[str, object]:
    from dataset_builder.vision import pose_artifact_paths

    video = source_root / source.filename
    inputs = [_artifact_record("source_video", video, source_root)]
    inputs.extend(
        _artifact_record(name, path, output)
        for name, path in pose_artifact_paths(output).as_mapping().items()
    )
    return {
        "schema": COURT_RECEIPT_SCHEMA,
        "match_id": source.match_id,
        "video": source.video,
        "code_id": code_id,
        "configuration": {
            "device": device,
            "resize_mode": resize_mode,
            "parent": COURT_PARENT,
            "ref_err_px": COURT_REF_ERR_PX,
        },
        "metadata": {
            "fps_numerator": metadata.fps.numerator,
            "fps_denominator": metadata.fps.denominator,
            "frame_count": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
            "sample_aspect_ratio_numerator": metadata.sample_aspect_ratio.numerator,
            "sample_aspect_ratio_denominator": metadata.sample_aspect_ratio.denominator,
        },
        "model": model_identity,
        "inputs": inputs,
    }


def _validate_completed_court(
    source: Source,
    *,
    output: Path,
    metadata: VideoMetadata,
    expected: dict[str, object],
) -> None:
    from dataset_builder.vision import load_court_vision, load_json_gz

    receipt = load_json_gz(output / COURT_RECEIPT_FILENAME)
    required = {*expected, "completed", "scene_count", "outputs"}
    if set(receipt) != required:
        raise ValueError(f"court receipt fields differ from {COURT_RECEIPT_SCHEMA}")
    for name, value in expected.items():
        if receipt[name] != value:
            raise ValueError(f"court receipt {name} does not match current inputs")
    if receipt["completed"] is not True:
        raise ValueError("court receipt is not marked completed")
    court = load_court_vision(
        output,
        video_id=str(source.match_id),
        frame_count=metadata.frame_count,
        resolution=(float(metadata.width), float(metadata.height)),
    )
    if receipt["scene_count"] != len(court.raw_cuts):
        raise ValueError("court receipt scene count does not match persisted evidence")
    if receipt["outputs"] != _court_output_identities(output):
        raise ValueError("court receipt output identities do not match persisted files")


def court_source(
    source: Source,
    *,
    source_root: Path,
    output_root: Path,
    detector: object,
    model_identity: dict[str, object],
    device: str,
    resize_mode: str,
    code_id: str,
) -> bool:
    """Build or validate the court stage for one previously extracted video."""
    from dataset_builder.vision import (
        build_detected_court_stage,
        load_court_vision,
        load_pose_arrays,
        save_json_gz,
    )

    source_root = Path(source_root)
    video = source_root / source.filename
    output = Path(output_root) / f"{source.match_id:02d} {source.video}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = probe_source(video)
    pose = load_pose_arrays(output, metadata.frame_count)
    expected = _court_receipt_base(
        source,
        source_root=source_root,
        output=output,
        metadata=metadata,
        model_identity=model_identity,
        device=device,
        resize_mode=resize_mode,
        code_id=code_id,
    )
    receipt_path = output / COURT_RECEIPT_FILENAME
    if receipt_path.is_file():
        _validate_completed_court(
            source,
            output=output,
            metadata=metadata,
            expected=expected,
        )
        print(f"{source.match_id:02d}: already extracted")
        return False

    for _, filename in _court_output_items():
        (output / filename).unlink(missing_ok=True)
    built = build_detected_court_stage(
        video_id=str(source.match_id),
        metadata=metadata,
        pose=pose,
        detector=detector,
        output_dir=output,
        parent=COURT_PARENT,
        ref_err_px=COURT_REF_ERR_PX,
    )
    court = load_court_vision(
        output,
        video_id=str(source.match_id),
        frame_count=metadata.frame_count,
        resolution=(float(metadata.width), float(metadata.height)),
    )
    if built.raw_cuts != court.raw_cuts:
        raise ValueError("reloaded court cuts differ from the completed stage")
    save_json_gz(
        receipt_path,
        {
            **expected,
            "completed": True,
            "scene_count": len(court.raw_cuts),
            "outputs": _court_output_identities(output),
        },
    )
    return True


def court_sources(
    sources: Sequence[Source],
    *,
    source_root: Path,
    output_root: Path,
    court_weights: Path,
    device: str,
    resize_mode: str,
    code_id: str,
    detector_factory: Callable[..., object] | None = None,
) -> int:
    """Run only PySceneDetect and CourtKeyNet over existing pose outputs."""
    from dataset_builder.manifest import artifact_integrity

    if len(code_id) != 64 or any(character not in "0123456789abcdef" for character in code_id):
        raise ValueError("--code-id must be a lowercase SHA-256 digest")
    weights = Path(court_weights).resolve(strict=True)
    model_identity = dict(
        artifact_integrity(
            "courtkeynet_weights",
            weights,
            relative_to=weights.parent,
        ).to_dict()
    )
    if detector_factory is None:
        from courtkeynet.wrapper import CourtKeyNetDetector

        detector_factory = CourtKeyNetDetector
    detector = detector_factory(
        weights_path=weights,
        device=device,
        resize_mode=resize_mode,
    )
    failures = 0
    for source in sources:
        if source.kind is SourceKind.UNRESOLVED:
            print(f"{source.match_id:02d}: unavailable: {source.unresolved_reason}")
            continue
        try:
            print(f"{source.match_id:02d}: extracting court", flush=True)
            court_source(
                source,
                source_root=source_root,
                output_root=output_root,
                detector=detector,
                model_identity=model_identity,
                device=device,
                resize_mode=resize_mode,
                code_id=code_id,
            )
        except Exception as error:
            failures += 1
            print(f"{source.match_id:02d}: FAILED: {error}", file=sys.stderr)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCES)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--source-root", type=Path, required=True)
    download.add_argument("--overlap-root", type=Path, required=True)
    download.add_argument("--cookies-from-browser")
    download.add_argument("--youtube-player-client")
    download.add_argument("--ids", type=int, nargs="+")

    extract = subparsers.add_parser("extract")
    extract.add_argument("--source-root", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--tracknet-dir", type=Path, default=DEFAULT_TRACKNET_DIR)
    extract.add_argument("--tracknet-python", type=Path, required=True)
    extract.add_argument("--pose-python", type=Path, required=True)
    extract.add_argument("--pose-shards", type=int, default=8)
    extract.add_argument("--ids", type=int, nargs="+")

    court = subparsers.add_parser("court")
    court.add_argument("--source-root", type=Path, required=True)
    court.add_argument("--output-root", type=Path, required=True)
    court.add_argument("--court-weights", type=Path, default=DEFAULT_COURT_WEIGHTS)
    court.add_argument("--device", default="cuda")
    court.add_argument("--resize-mode", choices=("pad", "squash"), default="pad")
    court.add_argument("--code-id", required=True)
    court.add_argument("--ids", type=int, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    sources = select_sources(load_sources(arguments.manifest), arguments.ids)
    if arguments.command == "download":
        return int(
            download_sources(
                sources,
                source_root=arguments.source_root,
                overlap_root=arguments.overlap_root,
                cookies_from_browser=arguments.cookies_from_browser,
                youtube_player_client=arguments.youtube_player_client,
            )
            > 0
        )
    if arguments.command == "court":
        return int(
            court_sources(
                sources,
                source_root=arguments.source_root,
                output_root=arguments.output_root,
                court_weights=arguments.court_weights,
                device=arguments.device,
                resize_mode=arguments.resize_mode,
                code_id=arguments.code_id,
            )
            > 0
        )
    if arguments.pose_shards <= 1:
        raise ValueError("--pose-shards must be greater than one")
    return int(
        extract_sources(
            sources,
            source_root=arguments.source_root,
            output_root=arguments.output_root,
            tracknet_dir=arguments.tracknet_dir,
            tracknet_python=arguments.tracknet_python,
            pose_python=arguments.pose_python,
            pose_shards=arguments.pose_shards,
        )
        > 0
    )


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=False)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer_text(value: object, name: str) -> int:
    text = _text(value, name)
    try:
        result = int(text)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    return _integer(result, name)


def _ratio(value: object, name: str) -> Fraction:
    if value is None or (isinstance(value, str) and value.strip() in {"", "N/A"}):
        return Fraction(1, 1)
    text = _text(value, name)
    try:
        numerator, denominator = text.split(":", maxsplit=1)
        return Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be a ratio") from error


if __name__ == "__main__":
    raise SystemExit(main())
