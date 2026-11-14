"""Portable scientific settings; runtime paths belong to the command line."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backbone: str = "200mf"
    mode: str = "supplied"
    clip_length: int = 100
    learning_rate: float = 0.001
    batch_size: int = 8
    validation_batch_size: int = 8
    seed: int = 0
    mixup: bool = False
    cycles: int = 50
    frames_per_cycle: int = 500_000
    warmup_cycles: int = 3
    weight_decay: float = 0.01
    augmentation: str = "hybrid"
    early_stopping_patience: int = 0
    cudnn_benchmark: bool = False
    ignore_contact_halo: bool = True
    checkpoint_metric: str = "loss"
    selection_tolerance: int = 10

    def __post_init__(self) -> None:
        if self.backbone not in ("200mf", "800mf") or self.mode not in ("supplied", "discovery"):
            raise ValueError("Expected 200mf/800mf and supplied/discovery")
        if min(self.clip_length, self.batch_size, self.validation_batch_size, self.cycles, self.frames_per_cycle) < 1:
            raise ValueError("Clip length, batch, cycles and frame budget must be positive")
        if self.clip_length < 2 or not 0 <= self.warmup_cycles < self.cycles:
            raise ValueError("Need clip length >=2 and warmup shorter than training")
        if self.learning_rate <= 0 or self.weight_decay != 0.01:
            raise ValueError("Positive learning rate and fixed AdamW weight decay .01 required")
        if self.augmentation not in ("hybrid", "user", "e2e"):
            raise ValueError("Augmentation must be hybrid, user or e2e")
        if self.early_stopping_patience < 0:
            raise ValueError("Early stopping patience must be non-negative; zero disables it")
        if self.augmentation == "user" and self.mixup:
            raise ValueError("The user augmentation recipe does not include MixUp")
        if self.checkpoint_metric not in ("loss", "rally_f1"):
            raise ValueError("Checkpoint metric must be loss or rally_f1")
        if self.selection_tolerance not in (1, 2, 5, 10):
            raise ValueError("Selection tolerance must be 1, 2, 5 or 10 frames at 30 fps")

    @property
    def classes(self) -> int:
        return 3 if self.mode == "supplied" else 4


def load_settings(path: Path) -> Settings:
    with path.open("rb") as stream:
        return Settings(**tomllib.load(stream))
