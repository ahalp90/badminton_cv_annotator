"""Small, shared readers and writers for the exploratory audit artefacts."""

from __future__ import annotations

import gzip
import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np


LZMA_PRESET = 9


def write_npy_xz(array: np.ndarray, path: Path) -> None:
    """Write one NumPy array as a native ``.npy`` stream inside XZ level 9."""

    with lzma.open(path, "wb", format=lzma.FORMAT_XZ, preset=LZMA_PRESET) as target:
        np.save(target, array, allow_pickle=False)


def read_npy_xz(path: Path) -> np.ndarray:
    """Load a native ``.npy`` array from an XZ level-9 stream."""

    with lzma.open(path, "rb", format=lzma.FORMAT_XZ) as source:
        value = np.load(source, allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{path}: expected a NumPy array, got {type(value).__name__}")
    return value


def write_json_gz(path: Path, payload: Any) -> None:
    """Write indented UTF-8 JSON with gzip level 9."""

    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as target:
        json.dump(payload, target, indent=2)
        target.write("\n")


def read_json_gz(path: Path) -> Any:
    """Read one UTF-8 JSON gzip stream."""

    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)
