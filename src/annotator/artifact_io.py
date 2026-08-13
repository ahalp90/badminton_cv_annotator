"""Compressed artifact I/O for annotator experiment records."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import gzip
import io
import json
import lzma
import os
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

import numpy as np


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")


def _counterpart(path: Path) -> Path | None:
    name = path.name
    for compressed_suffix, plain_suffix in (
        (".npy.xz", ".npy"),
        (".json.gz", ".json"),
        (".csv.gz", ".csv"),
    ):
        if name.endswith(compressed_suffix):
            return path.with_name(name[: -len(compressed_suffix)] + plain_suffix)
        if name.endswith(plain_suffix):
            return path.with_name(name + compressed_suffix[len(plain_suffix) :])
    return None


def resolve_artifact_path(path: Path) -> Path:
    """Return an existing requested artifact or its compressed/plain counterpart."""
    source = Path(path)
    if source.exists():
        return source
    counterpart = _counterpart(source)
    return counterpart if counterpart is not None and counterpart.exists() else source


def artifacts_are_byte_equal(first: Path, second: Path) -> bool:
    """Compare requested artifacts after compressed/plain fallback resolution."""
    return resolve_artifact_path(first).read_bytes() == resolve_artifact_path(second).read_bytes()


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_gzip_bytes(path: Path, payload: bytes) -> Path:
    """Atomically write deterministic level-9 gzip bytes."""
    destination = Path(path)
    if not destination.name.endswith(".gz"):
        raise ValueError(f"gzip artifact path must end in .gz: {destination}")
    compressed = gzip.compress(payload, compresslevel=9, mtime=0)
    return _atomic_write_bytes(destination, compressed)


@contextmanager
def atomic_gzip_text_writer(path: Path, *, newline: str | None = None) -> Iterator[TextIO]:
    """Yield an atomic deterministic level-9 gzip text writer."""
    destination = Path(path)
    if not destination.name.endswith(".gz"):
        raise ValueError(f"gzip artifact path must end in .gz: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline=newline) as text_handle:
                    yield text_handle
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def open_text_artifact(path: Path, *, newline: str | None = None) -> Iterator[TextIO]:
    """Open a gzip text artifact or its legacy plain counterpart."""
    source = resolve_artifact_path(path)
    if source.name.endswith(".gz"):
        with gzip.open(source, "rt", encoding="utf-8", newline=newline) as handle:
            yield handle
        return
    with source.open("r", encoding="utf-8", newline=newline) as handle:
        yield handle


def encode_json_object(path: Path, payload: Mapping[str, object]) -> bytes:
    """Encode a JSON object exactly as its compressed or plain artifact bytes."""
    destination = Path(path)
    encoded = (
        json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if destination.name.endswith(".json.gz"):
        return gzip.compress(encoded, compresslevel=9, mtime=0)
    if destination.name.endswith(".json"):
        return encoded
    raise ValueError(f"JSON artifact path must end in .json or .json.gz: {destination}")


def write_json_object(path: Path, payload: Mapping[str, object]) -> Path:
    """Atomically write a JSON object in compressed or legacy plain form."""
    destination = Path(path)
    return _atomic_write_bytes(destination, encode_json_object(destination, payload))


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a compressed JSON object or its legacy plain counterpart."""
    with open_text_artifact(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"JSON object required: {resolve_artifact_path(path)}")
    return payload


def save_npy_xz(path: Path, values: np.ndarray) -> Path:
    """Atomically store an array as XZ-compressed NumPy bytes."""
    destination = Path(path)
    if not destination.name.endswith(".npy.xz"):
        raise ValueError(f"compressed NumPy path must end in .npy.xz: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with lzma.open(temporary, "wb", format=lzma.FORMAT_XZ, preset=9) as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_npy(path: Path) -> np.ndarray:
    """Load a compressed NumPy array or its legacy plain counterpart."""
    source = resolve_artifact_path(path)
    if source.name.endswith(".npy.xz"):
        with lzma.open(source, "rb", format=lzma.FORMAT_XZ) as handle:
            return np.load(handle, allow_pickle=False)
    if source.name.endswith(".npy"):
        return np.load(source, allow_pickle=False)
    raise ValueError(f"NumPy artifact path must end in .npy or .npy.xz: {source}")
