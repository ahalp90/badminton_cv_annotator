from __future__ import annotations

from pathlib import Path

import pytest
from experiments.backends import BackendSpec, require_complete_frame_grid
from experiments.backends.qwen3_8 import (
    SPEC as QWEN38_SPEC,
)
from experiments.backends.qwen3_8 import (
    _apply_chat_template as apply_qwen38_chat_template,
)
from experiments.backends.qwen3_8 import (
    _engine_config as qwen38_engine_config,
)
from experiments.backends.qwen3_vl import (
    SPEC as QWEN_SPEC,
)
from experiments.backends.qwen3_vl import (
    _engine_config,
    _metadata_frame_indices,
    _video_content,
)


def test_backend_identity_keeps_pinned_model_details() -> None:
    spec = BackendSpec(
        key="test",
        model_id="owner/model",
        model_revision="revision",
        backend_name="runtime",
        backend_distribution="runtime-package",
        expected_backend_version="1.0",
        cache_dtype="bfloat16",
        package_names=(),
    )

    identity = spec.identity("1.2.3")

    assert identity.model_id == "owner/model"
    assert identity.model_revision == "revision"
    assert identity.backend == "runtime"
    assert identity.backend_version == "1.2.3"


def test_complete_frame_grid_rejects_processor_resampling() -> None:
    with pytest.raises(RuntimeError, match="sampled 3 unexpected frames"):
        require_complete_frame_grid("test", (0, 2, 3), expected_input_frames=3)


def test_qwen_engine_config_keeps_single_gpu_limits(tmp_path: Path) -> None:
    config = _engine_config(tmp_path / "model", max_model_len=16_384)

    assert config["max_model_len"] == 16_384
    assert config["tensor_parallel_size"] == 1
    assert config["cpu_offload_gb"] == 0
    assert config["limit_mm_per_prompt"] == {"image": 0, "video": 1}


@pytest.mark.parametrize("max_model_len", [4_095, 262_145])
def test_qwen_engine_config_rejects_unsafe_lengths(
    tmp_path: Path,
    max_model_len: int,
) -> None:
    with pytest.raises(ValueError, match="between 4,096 and 262,144"):
        _engine_config(tmp_path / "model", max_model_len=max_model_len)


def test_qwen_metadata_accepts_mapping_and_object() -> None:
    class Metadata:
        frames_indices = (0, 1, 2)

    assert _metadata_frame_indices({"frames_indices": [0, 1]}) == (0, 1)
    assert _metadata_frame_indices(Metadata()) == (0, 1, 2)


def test_qwen_video_contract_pins_frame_count_and_pixels(tmp_path: Path) -> None:
    content = _video_content(
        tmp_path / "clip.mp4",
        requested_fps=25.0,
        width=512,
        height=288,
        expected_input_frames=50,
    )

    assert content["min_frames"] == content["max_frames"] == 50
    assert content["min_pixels"] == content["max_pixels"] == 512 * 288
    assert content["total_pixels"] == 50 * 512 * 288
    assert content["video"].startswith("file://")
    assert QWEN_SPEC.model_revision == "d9748a51ae66354c4dad665aab2c71f26cf2c8cd"


def test_qwen38_identity_and_engine_config_are_isolated(tmp_path: Path) -> None:
    config = qwen38_engine_config(tmp_path / "model", max_model_len=16_384)

    assert QWEN38_SPEC.key == "qwen3-8"
    assert QWEN38_SPEC.model_revision == "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
    assert QWEN38_SPEC.expected_backend_version == "0.17.0"
    assert config["max_model_len"] == 16_384
    assert config["tensor_parallel_size"] == 1
    assert config["max_num_seqs"] == 1
    assert config["cpu_offload_gb"] == 0
    assert config["swap_space"] == 0


def test_qwen38_chat_template_disables_thinking() -> None:
    class Processor:
        def __init__(self) -> None:
            self.arguments = None

        def apply_chat_template(self, messages, **arguments):
            self.arguments = arguments
            return "rendered"

    processor = Processor()

    rendered = apply_qwen38_chat_template(
        processor,
        [{"role": "user", "content": [{"type": "text", "text": "question"}]}],
    )

    assert rendered == "rendered"
    assert processor.arguments == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
