"""Runtime evidence and strict response handling for VLM benchmark runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import subprocess
import threading
from typing import Any

from .contracts import PredictionSegment, ShardSpec, validate_prediction_partition


NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
GPU_MONITOR_STOP_TIMEOUT_SECONDS = 6.0
COMPACT_SEGMENT_KEYS = (
    "start_frame",
    "end_frame",
    "scene_label",
    "broadcast_phase",
    "view",
    "playback",
    "continuity_from_previous",
    "data_use",
    "confidence",
    "evidence_frames",
    "reason",
)
FRAME_CODE_WIDTH = 8
FRAME_CODE_RESPONSE_PREFIX = re.compile(r'\A\s*\{\s*"frames"\s*:\s*\[\s*')
FRAME_CODE_MAPS: tuple[Mapping[str, str | float], ...] = (
    {"L": "live", "N": "live-non-standard", "R": "replay", "C": "cutaway", "O": "other"},
    {
        "L": "live_rally",
        "B": "between_rallies",
        "R": "replay",
        "C": "cutaway",
        "O": "other",
        "U": "unknown",
    },
    {"R": "real_time", "S": "slow_motion", "F": "freeze_frame", "U": "unknown"},
    {
        "F": "full_court",
        "P": "partial_court",
        "S": "side_on",
        "C": "close_up",
        "D": "crowd",
        "G": "graphic",
        "O": "other",
        "U": "unknown",
    },
    {"S": "same_rally", "R": "new_rally", "A": "not_applicable", "U": "unknown"},
    {"S": "usable_standard", "A": "usable_alternate_view", "E": "exclude", "R": "review"},
    {**{str(value): value / 10 for value in range(10)}, "A": 1.0},
    {
        "R": "Active rally is visible.",
        "B": "Players are between rallies or preparing.",
        "P": "Replay footage is visible.",
        "C": "A cutaway view is visible.",
        "G": "Graphics or other footage is visible.",
        "U": "The visible state is unclear.",
    },
)


@dataclass(frozen=True)
class GpuSnapshot:
    """One aggregate NVIDIA device reading."""

    device_name: str
    used_memory_mib: float


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Return deterministic installed-version evidence, including absences."""
    found: list[tuple[str, str]] = []
    for name in sorted(set(names)):
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "not-installed"
        found.append((name, installed))
    return tuple(found)


def write_raw_response(path: Path, response: str) -> str:
    """Atomically retain a raw UTF-8 response before attempting to parse it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = response.encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(encoded)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(encoded)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def is_strict_json_response(response: str) -> bool:
    """Return whether the raw response is one complete JSON value."""
    try:
        json.loads(response, object_pairs_hook=_object_pairs)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def _unwrap_json_fence(response: str) -> str:
    """Remove one whole-response Markdown JSON fence, if present."""
    stripped = response.strip()
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0] == "```json" and lines[-1] == "```":
        return "\n".join(lines[1:-1])
    return response


def _prediction_segment_from_json(value: Any, index: int) -> PredictionSegment:
    """Expand the compact positional form before applying the strict contract."""
    if isinstance(value, list):
        if len(value) != len(COMPACT_SEGMENT_KEYS):
            raise ValueError(
                f"segments[{index}] compact array must contain "
                f"{len(COMPACT_SEGMENT_KEYS)} values, found {len(value)}"
            )
        value = dict(zip(COMPACT_SEGMENT_KEYS, value, strict=True))
    return PredictionSegment.from_json(value, index)


def _frame_code_segment(code: Any, index: int, start_frame: int, end_frame: int) -> PredictionSegment:
    """Expand one fixed-width sampled-frame code into the strict segment contract."""
    context = f"frames[{index}]"
    if not isinstance(code, str) or len(code) != FRAME_CODE_WIDTH:
        raise ValueError(f"{context} must be an {FRAME_CODE_WIDTH}-character string")
    decoded: list[str | float] = []
    for position, (character, code_map) in enumerate(zip(code, FRAME_CODE_MAPS, strict=True)):
        try:
            decoded.append(code_map[character])
        except KeyError as error:
            raise ValueError(f"{context}[{position}] has unknown code {character!r}") from error
    return PredictionSegment.from_json(
        {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "scene_label": decoded[0],
            "broadcast_phase": decoded[1],
            "playback": decoded[2],
            "view": decoded[3],
            "continuity_from_previous": decoded[4],
            "data_use": decoded[5],
            "confidence": decoded[6],
            "evidence_frames": [start_frame],
            "reason": decoded[7],
        },
        index,
    )


def _same_segment_state(left: PredictionSegment, right: PredictionSegment) -> bool:
    return (
        left.scene_label == right.scene_label
        and left.broadcast_phase == right.broadcast_phase
        and left.view == right.view
        and left.playback == right.playback
        and left.continuity_from_previous == right.continuity_from_previous
        and left.data_use == right.data_use
        and left.confidence == right.confidence
        and left.reason == right.reason
    )


def _merge_frame_segments(segments: Sequence[PredictionSegment]) -> tuple[PredictionSegment, ...]:
    merged: list[PredictionSegment] = []
    for segment in segments:
        if not merged or not _same_segment_state(merged[-1], segment):
            merged.append(segment)
            continue
        previous = merged[-1]
        merged[-1] = PredictionSegment(
            start_frame=previous.start_frame,
            end_frame=segment.end_frame,
            scene_label=previous.scene_label,
            broadcast_phase=previous.broadcast_phase,
            view=previous.view,
            playback=previous.playback,
            continuity_from_previous=previous.continuity_from_previous,
            data_use=previous.data_use,
            confidence=previous.confidence,
            evidence_frames=(*previous.evidence_frames, *segment.evidence_frames)[:3],
            reason=previous.reason,
        )
    return tuple(merged)


def _frame_code_segments(
    raw_frames: Any,
    shard: ShardSpec,
    sampled_source_frames: Sequence[int] | None,
) -> tuple[PredictionSegment, ...]:
    if not isinstance(raw_frames, list):
        raise ValueError("prediction response frames must be a JSON array")
    if sampled_source_frames is None:
        raise ValueError("sampled source frames are required for a frame-code response")
    frame_grid = tuple(sampled_source_frames)
    if len(raw_frames) < len(frame_grid):
        raise ValueError(
            f"prediction response contains {len(raw_frames)} frame codes, expected {len(frame_grid)}"
        )
    if not frame_grid or frame_grid[0] != shard.start_frame:
        raise ValueError("sampled source frames must begin at the shard start")
    if any(right <= left for left, right in zip(frame_grid, frame_grid[1:])):
        raise ValueError("sampled source frames must be strictly increasing")
    if frame_grid[-1] >= shard.end_frame:
        raise ValueError("sampled source frames must end inside the shard")
    end_frames = (*frame_grid[1:], shard.end_frame)
    segments: list[PredictionSegment] = []
    intervals = zip(raw_frames[: len(frame_grid)], frame_grid, end_frames, strict=True)
    for index, (code, start_frame, end_frame) in enumerate(intervals):
        segments.append(_frame_code_segment(code, index, start_frame, end_frame))
    return _merge_frame_segments(segments)


def _complete_frame_code_prefix(response: str, expected_count: int) -> list[Any] | None:
    """Read the required code prefix when generation continues past full coverage."""
    match = FRAME_CODE_RESPONSE_PREFIX.match(response)
    if match is None:
        return None
    decoder = json.JSONDecoder()
    offset = match.end()
    codes: list[Any] = []
    for index in range(expected_count):
        try:
            code, end_offset = decoder.raw_decode(response, offset)
        except json.JSONDecodeError:
            return None
        codes.append(code)
        offset = end_offset
        while offset < len(response) and response[offset].isspace():
            offset += 1
        if index < expected_count - 1:
            if offset >= len(response) or response[offset] != ",":
                return None
            offset += 1
            while offset < len(response) and response[offset].isspace():
                offset += 1
    return codes


def parse_prediction_response(
    response: str,
    shard: ShardSpec,
    sampled_source_frames: Sequence[int] | None = None,
) -> tuple[PredictionSegment, ...]:
    """Parse a bare or singly fenced JSON response and require a complete partition."""
    try:
        value = json.loads(_unwrap_json_fence(response), object_pairs_hook=_object_pairs)
    except json.JSONDecodeError as error:
        expected_count = 0 if sampled_source_frames is None else len(sampled_source_frames)
        frame_codes = None if expected_count == 0 else _complete_frame_code_prefix(response, expected_count)
        if frame_codes is None:
            raise ValueError(
                f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
            ) from error
        value = {"frames": frame_codes}
    if not isinstance(value, dict):
        raise ValueError("prediction response must be a JSON object")
    if set(value) == {"frames"}:
        segments = _frame_code_segments(value["frames"], shard, sampled_source_frames)
    elif set(value) == {"segments"}:
        raw_segments = value["segments"]
        if not isinstance(raw_segments, list):
            raise ValueError("prediction response segments must be a JSON array")
        segments = tuple(
            _prediction_segment_from_json(segment, index) for index, segment in enumerate(raw_segments)
        )
    else:
        raise ValueError(
            "prediction response keys differ; expected ['frames'] or ['segments'], "
            f"found {sorted(value)}"
        )
    validate_prediction_partition(segments, shard)
    return segments


def query_nvidia_gpu() -> GpuSnapshot:
    """Read the first GPU name and aggregate device memory from nvidia-smi."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"nvidia-smi timed out after {NVIDIA_SMI_TIMEOUT_SECONDS:.1f} seconds"
        ) from error
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown nvidia-smi failure"
        raise RuntimeError(f"nvidia-smi failed: {message}")
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    names: list[str] = []
    total_memory = 0.0
    for row in rows:
        name, separator, memory_text = row.rpartition(",")
        if not separator or not name.strip():
            raise RuntimeError(f"unexpected nvidia-smi row: {row!r}")
        try:
            total_memory += float(memory_text.strip())
        except ValueError as error:
            raise RuntimeError(f"invalid memory value in nvidia-smi row: {row!r}") from error
        names.append(name.strip())
    return GpuSnapshot(" + ".join(names), total_memory)


class GpuMemoryMonitor:
    """Poll total NVIDIA memory so child-process backends remain measurable."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        if interval_seconds <= 0:
            raise ValueError("GPU monitor interval must be positive")
        self.interval_seconds = interval_seconds
        self.device_name = "unavailable"
        self.peak_used_memory_mib: float | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot = query_nvidia_gpu()
            except (OSError, RuntimeError) as error:
                self.error = str(error)
                return
            self.device_name = snapshot.device_name
            if self.peak_used_memory_mib is None:
                self.peak_used_memory_mib = snapshot.used_memory_mib
            else:
                self.peak_used_memory_mib = max(
                    self.peak_used_memory_mib,
                    snapshot.used_memory_mib,
                )
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU monitor is already started")
        self._thread = threading.Thread(target=self._sample, name="vlm-gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=GPU_MONITOR_STOP_TIMEOUT_SECONDS)
            if self._thread.is_alive() and self.error is None:
                self.error = (
                    "GPU monitor did not stop within "
                    f"{GPU_MONITOR_STOP_TIMEOUT_SECONDS:g} seconds"
                )

    def __enter__(self) -> GpuMemoryMonitor:
        self.start()
        return self

    def __exit__(self, _error_type: object, _error: object, _traceback: object) -> None:
        self.stop()
