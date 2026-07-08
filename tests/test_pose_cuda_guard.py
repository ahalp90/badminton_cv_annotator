"""Guard test for RtmlibPoseExtractor's silent-CPU-fallback catch.

onnxruntime-gpu does NOT error when the CUDAExecutionProvider fails to load
(e.g. libcudnn.so.9 off the loader path): it logs two red lines and runs CPU,
roughly 10x slower. The adapter's __init__ checks each tool's engaged providers
after ``device='cuda'`` and raises rather than ship a silent slowdown into a
whole-video pass.

Needs the real ``preparing_data.rtmlib_pose``, which imports rtmlib +
onnxruntime at module load; CI has neither, so importorskip skips the whole
module there. Where it does import, we monkeypatch the module-level
``RTMDetScored`` / ``RTMPose`` with fakes exposing only ``.session.get_providers()``
(plus a settable ``.score_thr`` on the detector), so no onnxruntime session is
ever built.
"""
from __future__ import annotations

import pytest

rtmlib_pose = pytest.importorskip("preparing_data.rtmlib_pose")


class _FakeSession:
    """Stands in for the onnxruntime InferenceSession BaseTool stores at
    self.session; the guard only calls get_providers()."""

    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return self._providers


class _FakeTool:
    """A drop-in for RTMDetScored / RTMPose whose session reports ``providers``.

    Swallows the real (url, model_input_size, device) constructor args and skips
    the onnxruntime session build; score_thr stays settable for the detector.
    """

    def __init__(self, providers: list[str], *args, **kwargs) -> None:
        self.session = _FakeSession(providers)
        self.score_thr = 0.0


def _patch_tools(monkeypatch, providers: list[str]) -> None:
    monkeypatch.setattr(rtmlib_pose, "RTMDetScored", lambda *a, **kw: _FakeTool(providers, *a, **kw))
    monkeypatch.setattr(rtmlib_pose, "RTMPose", lambda *a, **kw: _FakeTool(providers, *a, **kw))


def test_cuda_requested_but_cpu_only_raises(monkeypatch):
    """device='cuda' with both tools on CPU only: RuntimeError naming the
    provider that failed to load."""
    _patch_tools(monkeypatch, ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        rtmlib_pose.RtmlibPoseExtractor(device="cuda")


def test_cuda_engaged_constructs_fine(monkeypatch):
    """CUDA provider present (ahead of the CPU fallback): no raise."""
    _patch_tools(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    extractor = rtmlib_pose.RtmlibPoseExtractor(device="cuda")
    assert extractor.det is not None and extractor.pose is not None


def test_cpu_device_skips_the_guard(monkeypatch):
    """device='cpu': a CPU build has no CUDA provider by design, so the guard
    must not fire even with a CPU-only provider list."""
    _patch_tools(monkeypatch, ["CPUExecutionProvider"])
    extractor = rtmlib_pose.RtmlibPoseExtractor(device="cpu")
    assert extractor.det is not None and extractor.pose is not None
