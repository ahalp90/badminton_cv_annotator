"""Atomic progress snapshots and recoverable training checkpoints."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from .data import read_json, write_json


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".partial")
    torch.save(state, temporary)
    temporary.replace(path)


class Progress:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.last_write = 0.0
        self.state: dict[str, Any] = {"pid": os.getpid(), "session_started": time.time()}
        history_path = output / "training_progress.json.gz"
        self.training_history: list[dict[str, Any]] = read_json(history_path) if history_path.exists() else []

    def due(self) -> bool:
        return time.monotonic() - self.last_write >= 30

    def write(self, phase: str, **values: Any) -> None:
        self.state.update(phase=phase, updated_at=time.time(), **values)
        if phase == "training" and "train_loss" in values:
            # Loss is averaged since the interval began, not since the previous reading.
            # Session and interval boundaries distinguish resets and replayed updates.
            record = {key: self.state[key] for key in ("session_started", "updated_at", "step", "lr", "train_loss")}
            record["interval_start_step"] = self.state["step"] - self.state["completed"]
            self.training_history.append(record)
            write_json(self.output / "training_progress.json.gz", self.training_history)
        write_json(self.output / "status.json.gz", self.state)
        self.last_write = time.monotonic()
