"""Dataset-builder boundary for frame-range-sharded RTMLib extraction."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from annotator.video_metadata import VideoMetadata
from dataset_builder._pose_process import (
    POSE_CHILD_STEM,
    load_raw_pose_mapping,
    pose_subprocess_environment,
    resolve_pose_executable,
    run_isolated_pose_process,
)
from dataset_builder.vision import (
    PoseArrays,
    PoseExtraction,
    save_pose_arrays,
    validate_pose_arrays,
)


POSE_SHARD_DECODE_MODE = "seek"
_POSE_SHARD_CHILD_COMMAND = "_extract-sharded-rtmlib-pose"


def sharded_rtmlib_pose_command(
    *,
    executable: Path,
    video_path: Path,
    raw_output_root: Path,
    device: str,
    n_max: int,
    shards: int,
    expected_frame_count: int,
    run_id: str,
    decode_mode: str = POSE_SHARD_DECODE_MODE,
) -> list[str]:
    """Build the auditable child command for one sharded pose extraction."""
    return [
        os.fspath(executable),
        "-m",
        "dataset_builder.pose_sharding",
        _POSE_SHARD_CHILD_COMMAND,
        "--video",
        os.fspath(video_path),
        "--output-root",
        os.fspath(raw_output_root),
        "--device",
        device,
        "--n-max",
        str(n_max),
        "--shards",
        str(shards),
        "--expected-frame-count",
        str(expected_frame_count),
        "--run-id",
        run_id,
        "--decode-mode",
        decode_mode,
    ]


def extract_sharded_rtmlib_pose_stage(
    *,
    metadata: VideoMetadata,
    output_dir: Path,
    interpreter: str | Path,
    shards: int,
    device: str = "cuda",
    n_max: int = 16,
    decode_mode: str = POSE_SHARD_DECODE_MODE,
) -> PoseExtraction:
    """Run validated multi-process RTMLib extraction in the pose environment."""
    _validate_settings(shards=shards, n_max=n_max, device=device, decode_mode=decode_mode)
    executable = resolve_pose_executable(interpreter)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    with tempfile.TemporaryDirectory(prefix=".rtmlib-sharded-", dir=root) as raw_root_text:
        raw_root = Path(raw_root_text)
        command = sharded_rtmlib_pose_command(
            executable=executable,
            video_path=metadata.source_path,
            raw_output_root=raw_root,
            device=device,
            n_max=n_max,
            shards=shards,
            expected_frame_count=metadata.frame_count,
            run_id=run_id,
            decode_mode=decode_mode,
        )
        completed = run_isolated_pose_process(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=pose_subprocess_environment(),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-4000:]
            raise RuntimeError(
                f"sharded RTMLib subprocess exited with status {completed.returncode}: "
                f"{detail or 'no diagnostic output'}"
            )
        published = raw_root / f"publish_{run_id}"
        arrays = PoseArrays(**load_raw_pose_mapping(published, POSE_CHILD_STEM))
    validate_pose_arrays(arrays, metadata.frame_count)
    artifacts = save_pose_arrays(root, arrays, metadata.frame_count)
    return PoseExtraction(arrays=arrays, artifacts=artifacts, command=tuple(command))


def _validate_settings(*, shards: int, n_max: int, device: str, decode_mode: str) -> None:
    if isinstance(shards, bool) or not isinstance(shards, int) or shards <= 1:
        raise ValueError(f"shards must be an integer greater than one, got {shards!r}")
    if isinstance(n_max, bool) or not isinstance(n_max, int) or not 0 < n_max <= 127:
        raise ValueError(f"n_max must be an integer in [1, 127], got {n_max!r}")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if decode_mode not in {"seek", "scan"}:
        raise ValueError(f"decode_mode must be 'seek' or 'scan', got {decode_mode!r}")


def _extract_pose_child(
    *,
    video_path: Path,
    output_root: Path,
    device: str,
    n_max: int,
    shards: int,
    expected_frame_count: int,
    run_id: str,
    decode_mode: str,
) -> int:
    from shared.video_sharding.run_sharded import extract_sharded

    _validate_settings(shards=shards, n_max=n_max, device=device, decode_mode=decode_mode)
    if not video_path.is_file():
        raise FileNotFoundError(f"pose source video is not a regular file: {video_path}")
    published = extract_sharded(
        video_path=video_path,
        out_root=output_root,
        stem=POSE_CHILD_STEM,
        n_shards=shards,
        n_max=n_max,
        extractor_spec=device,
        decode_mode=decode_mode,
        run_id=run_id,
        expected_frame_count=expected_frame_count,
    )
    expected = output_root / f"publish_{run_id}"
    if published != expected:
        raise RuntimeError(f"sharded pose publication path {published} != {expected}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pose_parser = subparsers.add_parser(_POSE_SHARD_CHILD_COMMAND)
    pose_parser.add_argument("--video", type=Path, required=True)
    pose_parser.add_argument("--output-root", type=Path, required=True)
    pose_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    pose_parser.add_argument("--n-max", type=int, required=True)
    pose_parser.add_argument("--shards", type=int, required=True)
    pose_parser.add_argument("--expected-frame-count", type=int, required=True)
    pose_parser.add_argument("--run-id", required=True)
    pose_parser.add_argument("--decode-mode", choices=("seek", "scan"), required=True)
    arguments = parser.parse_args(argv)
    if arguments.command != _POSE_SHARD_CHILD_COMMAND:
        parser.error(f"unsupported command: {arguments.command}")
    return _extract_pose_child(
        video_path=arguments.video,
        output_root=arguments.output_root,
        device=arguments.device,
        n_max=arguments.n_max,
        shards=arguments.shards,
        expected_frame_count=arguments.expected_frame_count,
        run_id=arguments.run_id,
        decode_mode=arguments.decode_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
