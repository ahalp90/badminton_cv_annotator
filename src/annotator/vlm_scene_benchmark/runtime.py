"""Runtime evidence and strict response handling for VLM benchmark runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import threading
from typing import Any

from .contracts import PredictionSegment, ShardSpec, validate_prediction_partition


NVIDIA_SMI_TIMEOUT_SECONDS = 5.0
GPU_MONITOR_STOP_TIMEOUT_SECONDS = 6.0


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


def parse_prediction_response(response: str, shard: ShardSpec) -> tuple[PredictionSegment, ...]:
    """Parse the model-only response and require a complete prediction partition."""
    try:
        value = json.loads(response, object_pairs_hook=_object_pairs)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError("prediction response must be a JSON object")
    if set(value) != {"segments"}:
        raise ValueError(
            f"prediction response keys differ; expected ['segments'], found {sorted(value)}"
        )
    raw_segments = value["segments"]
    if not isinstance(raw_segments, list):
        raise ValueError("prediction response segments must be a JSON array")
    segments = tuple(
        PredictionSegment.from_json(segment, index) for index, segment in enumerate(raw_segments)
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
