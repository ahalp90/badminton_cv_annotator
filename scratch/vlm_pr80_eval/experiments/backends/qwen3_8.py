"""Pinned vLLM adapter for Qwen3.8-27B-FP8."""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from . import BackendSpec, GenerationEvidence, require_complete_frame_grid
from .qwen3_vl import (
    _configure_vllm_environment,
    _metadata_frame_indices,
    _video_and_metadata,
    _video_content,
)

SPEC = BackendSpec(
    key="qwen3-8",
    model_id="Qwen/Qwen3.8-27B-FP8",
    model_revision="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    backend_name="vllm",
    backend_distribution="vllm",
    expected_backend_version="0.17.0",
    cache_dtype="bfloat16",
    package_names=(
        "qwen-vl-utils",
        "av",
        "numpy",
        "opencv-python-headless",
        "torch",
        "torchvision",
        "transformers",
        "vllm",
    ),
)


def _resolve_model_snapshot(
    snapshot_download_fn: Callable[..., str] | None = None,
) -> Path:
    """Download and return the exact pinned model snapshot."""
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download

    snapshot = Path(
        snapshot_download_fn(
            repo_id=SPEC.model_id,
            revision=SPEC.model_revision,
        )
    ).resolve()
    if snapshot.name != SPEC.model_revision:
        raise RuntimeError(
            f"Qwen3.8 snapshot resolved to {snapshot.name!r}, "
            f"expected {SPEC.model_revision!r}"
        )
    return snapshot


def _engine_config(model_path: Path, max_model_len: int) -> dict[str, Any]:
    """Return the pinned single-request vLLM engine configuration."""
    if not 4_096 <= max_model_len <= 262_144:
        raise ValueError(
            "Qwen3.8 maximum model length must be between 4,096 and 262,144"
        )
    return {
        "model": str(model_path),
        "tokenizer": str(model_path),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "cpu_offload_gb": 0,
        "swap_space": 0,
        "limit_mm_per_prompt": {"image": 0, "video": 1},
        "mm_processor_cache_gb": 0,
        "seed": 0,
    }


def _apply_chat_template(processor: Any, messages: Sequence[dict[str, Any]]) -> str:
    """Build the frozen direct-answer prompt without Qwen3.8 thinking tokens."""
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class Qwen38Backend:
    """One resident Qwen3.8 engine with deterministic single-request decoding."""

    spec = SPEC

    def __init__(self, *, expected_input_frames: int, max_model_len: int) -> None:
        if expected_input_frames < 2:
            raise ValueError("Qwen3.8 requires at least two supplied video frames")
        _configure_vllm_environment()
        from transformers import AutoProcessor
        from vllm import LLM

        self._expected_input_frames = expected_input_frames
        self.backend_version = version(SPEC.backend_distribution)
        model_path = _resolve_model_snapshot()
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self._llm = LLM(**_engine_config(model_path, max_model_len))

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
        from qwen_vl_utils import process_vision_info
        from vllm import SamplingParams

        messages = [
            {
                "role": "user",
                "content": [
                    _video_content(
                        video_path,
                        requested_fps=requested_fps,
                        width=width,
                        height=height,
                        expected_input_frames=self._expected_input_frames,
                    ),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_prompt = _apply_chat_template(self._processor, messages)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=int(self._processor.image_processor.patch_size),
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if image_inputs is not None:
            raise RuntimeError("Qwen3.8 benchmark unexpectedly produced image inputs")
        video_tensor, metadata = _video_and_metadata(video_inputs)
        frame_indices = _metadata_frame_indices(metadata)
        require_complete_frame_grid(
            "Qwen3.8",
            frame_indices,
            self._expected_input_frames,
        )
        if len(video_tensor.shape) != 4:
            raise RuntimeError(
                f"unexpected Qwen3.8 video tensor shape {tuple(video_tensor.shape)!r}"
            )
        observed_height = int(video_tensor.shape[-2])
        observed_width = int(video_tensor.shape[-1])
        video_processor = self._processor.video_processor
        temporal_factor = int(video_processor.temporal_patch_size)
        spatial_factor = int(video_processor.patch_size) * int(
            video_processor.merge_size
        )
        if observed_height % spatial_factor or observed_width % spatial_factor:
            raise RuntimeError(
                f"Qwen3.8 video resolution {observed_width}x{observed_height} "
                f"is not divisible by {spatial_factor}"
            )
        visual_tokens = (
            math.ceil(len(frame_indices) / temporal_factor)
            * (observed_height // spatial_factor)
            * (observed_width // spatial_factor)
        )
        engine_input = {
            "prompt": chat_prompt,
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": video_kwargs,
        }
        sampling = SamplingParams(
            temperature=0.0,
            top_k=-1,
            max_tokens=max_new_tokens,
            seed=0,
        )
        outputs = self._llm.generate(
            [engine_input],
            sampling_params=sampling,
            use_tqdm=False,
        )
        prompt_token_ids = None
        try:
            if len(outputs) != 1 or not outputs[0].outputs:
                raise RuntimeError("vLLM returned no generation output")
            prompt_token_ids = outputs[0].prompt_token_ids
            if prompt_token_ids is None:
                raise RuntimeError("vLLM did not expose prompt token IDs")
            raw_response = outputs[0].outputs[0].text
            total_input_tokens = len(prompt_token_ids)
        finally:
            del prompt_token_ids
            del outputs
            del engine_input
            del video_tensor
            del video_inputs
            gc.collect()
        return GenerationEvidence(
            raw_response=raw_response,
            sampled_input_frames=frame_indices,
            width=observed_width,
            height=observed_height,
            visual_tokens=visual_tokens,
            total_input_tokens=total_input_tokens,
        )
