"""Pinned adapters retained from PR 80 for the cleanup experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

BACKEND_KEYS = ("internvideo3", "qwen3-vl", "qwen3-8")
QWEN_BACKEND_KEYS = frozenset({"qwen3-vl", "qwen3-8"})


@dataclass(frozen=True)
class ModelIdentity:
    """Model details written beside every attempt."""

    model_id: str
    model_revision: str
    backend: str
    backend_version: str


@dataclass(frozen=True)
class BackendSpec:
    """Pinned model and runtime details."""

    key: str
    model_id: str
    model_revision: str
    backend_name: str
    backend_distribution: str
    expected_backend_version: str
    cache_dtype: str
    package_names: tuple[str, ...]

    def identity(self, backend_version: str) -> ModelIdentity:
        return ModelIdentity(
            model_id=self.model_id,
            model_revision=self.model_revision,
            backend=self.backend_name,
            backend_version=backend_version,
        )


@dataclass(frozen=True)
class GenerationEvidence:
    """One raw reply and evidence about the video input."""

    raw_response: str
    sampled_input_frames: tuple[int, ...]
    width: int
    height: int
    visual_tokens: int
    total_input_tokens: int


class SceneBackend(Protocol):
    """Small common interface for the two isolated model environments."""

    spec: BackendSpec
    backend_version: str

    def generate(
        self,
        video_path: Path,
        prompt: str,
        *,
        requested_fps: float,
        width: int,
        height: int,
        max_new_tokens: int,
    ) -> GenerationEvidence:
        """Generate one reply and report what the backend consumed."""


def require_complete_frame_grid(
    backend_name: str,
    frame_indices: tuple[int, ...],
    expected_input_frames: int,
) -> None:
    """Reject processor resampling before an expensive generation."""
    if frame_indices != tuple(range(expected_input_frames)):
        raise RuntimeError(
            f"{backend_name} processor sampled {len(frame_indices)} unexpected frames; "
            f"expected the complete grid of {expected_input_frames}"
        )


def load_backend(
    name: str,
    *,
    expected_input_frames: int,
    max_model_len: int | None = None,
) -> SceneBackend:
    """Import heavy dependencies only after command-line validation."""
    if name == "internvideo3":
        from .internvideo3 import InternVideo3Backend

        return InternVideo3Backend(expected_input_frames=expected_input_frames)
    if name == "qwen3-vl":
        from .qwen3_vl import Qwen3VLBackend

        if max_model_len is None:
            raise ValueError("Qwen3-VL requires an explicit maximum model length")
        return Qwen3VLBackend(
            expected_input_frames=expected_input_frames,
            max_model_len=max_model_len,
        )
    if name == "qwen3-8":
        from .qwen3_8 import Qwen38Backend

        if max_model_len is None:
            raise ValueError("Qwen3.8 requires an explicit maximum model length")
        return Qwen38Backend(
            expected_input_frames=expected_input_frames,
            max_model_len=max_model_len,
        )
    raise ValueError(f"unknown VLM backend {name!r}")


__all__ = [
    "BACKEND_KEYS",
    "QWEN_BACKEND_KEYS",
    "BackendSpec",
    "GenerationEvidence",
    "ModelIdentity",
    "SceneBackend",
    "load_backend",
    "require_complete_frame_grid",
]
