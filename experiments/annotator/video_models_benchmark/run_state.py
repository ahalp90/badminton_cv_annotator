"""Atomic progress snapshots and recoverable training checkpoints."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from .data import write_json


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


class Progress:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.last_write = 0.0
        self.state: dict[str, Any] = {"pid": os.getpid(), "session_started": time.time()}

    def due(self) -> bool:
        return time.monotonic() - self.last_write >= 30

    def write(self, phase: str, **values: Any) -> None:
        self.state.update(phase=phase, updated_at=time.time(), **values)
        write_json(self.output / "status.json.gz", self.state)
        self.last_write = time.monotonic()
