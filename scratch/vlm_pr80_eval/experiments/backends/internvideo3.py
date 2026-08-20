"""Pinned Transformers adapter for InternVideo3-8B-Instruct."""

from __future__ import annotations

import gc
from importlib.metadata import version
from pathlib import Path
from typing import Any

from . import BackendSpec, GenerationEvidence, require_complete_frame_grid

SPEC = BackendSpec(
    key="internvideo3",
    model_id="yanziang/InternVideo3-8B-Instruct",
    model_revision="c4602918b65225650d152db2850fe34e01d21fcd",
    backend_name="transformers",
    backend_distribution="transformers",
    expected_backend_version="4.57.3",
    cache_dtype="bfloat16",
    package_names=(
        "accelerate",
        "av",
        "numpy",
        "opencv-python-headless",
        "qwen-vl-utils",
        "torch",
        "torchcodec",
        "torchvision",
        "transformers",
    ),
)


def _first_metadata(value: Any) -> Any:
    if value is None:
        raise RuntimeError("InternVideo3 processor did not return video metadata")
    if isinstance(value, (list, tuple)):
        if not value:
            raise RuntimeError("InternVideo3 processor returned empty video metadata")
        return value[0]
    return value


def _prepare_chat_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    requested_fps: float,
) -> tuple[Any, Any]:
    """Process one chat while keeping non-tensor video metadata separate."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        fps=requested_fps,
        return_metadata=True,
        padding=True,
    )
    metadata = _first_metadata(inputs.pop("video_metadata", None))
    inputs.convert_to_tensors("pt")
    return inputs, metadata


class InternVideo3Backend:
    """One resident InternVideo3 model with deterministic decoding."""

    spec = SPEC

    def __init__(self, *, expected_input_frames: int) -> None:
        if expected_input_frames < 2:
            raise ValueError("InternVideo3 requires at least two supplied video frames")
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("InternVideo3 inference requires a CUDA device")
        self._torch = torch
        self._expected_input_frames = expected_input_frames
        self.backend_version = version(SPEC.backend_distribution)
        self._processor = AutoProcessor.from_pretrained(
            SPEC.model_id,
            revision=SPEC.model_revision,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            SPEC.model_id,
            revision=SPEC.model_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._model.eval()
        devices = {parameter.device.type for parameter in self._model.parameters()}
        if devices != {"cuda"}:
            raise RuntimeError(
                f"InternVideo3 parameters are not wholly on CUDA: {sorted(devices)}"
            )
        dtypes = {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in self._model.parameters()
        }
        if dtypes != {SPEC.cache_dtype}:
            raise RuntimeError(
                f"InternVideo3 parameter dtypes differ from BF16: {sorted(dtypes)}"
            )

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
        video_processor = self._processor.video_processor
        video_processor.min_frames = min(4, self._expected_input_frames)
        video_processor.max_frames = self._expected_input_frames
        total_pixels = self._expected_input_frames * width * height
        video_processor.size = {
            "shortest_edge": total_pixels,
            "longest_edge": total_pixels,
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path.resolve()),
                        "fps": requested_fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs, metadata = _prepare_chat_inputs(
            self._processor,
            messages,
            requested_fps,
        )
        frame_indices = tuple(int(frame) for frame in metadata.frames_indices)
        require_complete_frame_grid(
            "InternVideo3",
            frame_indices,
            self._expected_input_frames,
        )
        grid_t, grid_h, grid_w = (
            int(value) for value in inputs["video_grid_thw"][0].tolist()
        )
        patch_size = int(video_processor.patch_size)
        merge_size = int(video_processor.merge_size)
        observed_width = grid_w * patch_size
        observed_height = grid_h * patch_size
        visual_tokens = grid_t * grid_h * grid_w // (merge_size**2)
        total_input_tokens = int(inputs["attention_mask"].sum().item())
        inputs = inputs.to("cuda:0")
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        generated = None
        try:
            input_length = int(inputs["input_ids"].shape[-1])
            generated = output[:, input_length:]
            raw_response = self._processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        finally:
            del generated
            del output
            del inputs
            gc.collect()
            self._torch.cuda.empty_cache()
        return GenerationEvidence(
            raw_response=raw_response,
            sampled_input_frames=frame_indices,
            width=observed_width,
            height=observed_height,
            visual_tokens=visual_tokens,
            total_input_tokens=total_input_tokens,
        )
