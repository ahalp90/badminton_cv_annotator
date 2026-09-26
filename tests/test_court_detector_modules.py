"""The court detector uses one package identity without changing import paths."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from scratch.court_det_fix.court_detector import image_sources, inputs, measurements
from scratch.court_det_fix.court_detector.detect import freeze_arrays, load_live_modules
from scratch.court_det_fix.court_detector.search import DIRECTION_SETTINGS

REPO = Path(__file__).resolve().parents[1]
COURT_ROOT = REPO / "scratch/court_det_fix"
LEAVES = ("search", "net_choice", "stripe_refit", "inputs", "feet")


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "src")))}
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=environment, capture_output=True, text=True,
                          timeout=300, check=False)


def test_direction_settings_equal_the_frozen_direction_file() -> None:
    path = COURT_ROOT / "frozen_views/baseline_directions/gxBQ_window_00_frame_0.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        frozen = json.load(stream)["settings"]
    # JSON distinguishes an integer setting from its floating-point replacement.
    assert json.dumps(DIRECTION_SETTINGS, sort_keys=True) == json.dumps(frozen, sort_keys=True)


@pytest.mark.parametrize("load_runtime", [False, True])
def test_package_imports_leave_paths_and_research_modules_alone(load_runtime: bool) -> None:
    imports = "\n".join(f"import scratch.court_det_fix.court_detector.{leaf}" for leaf in LEAVES)
    code = f"""
import sys
from pathlib import Path
from types import ModuleType
before = list(sys.path)
# Existing bare research names must have no influence on package imports.
names = ("run_w5", "verifier", "generation", "automatic_generation", "run_automatic",
         "run_given", "projective_seed", "vp_pruning", "measurement", "zone_net", "shared")
sentinels = {{name: ModuleType(name) for name in names}}
sys.modules.update(sentinels)
{imports}
if {load_runtime!r}:
    from scratch.court_det_fix.court_detector.detect import load_live_modules
    live = load_live_modules()
    for module in live[:8]:
        assert module.__name__.startswith("scratch.court_det_fix.court_detector."), module
court_root = Path({str(COURT_ROOT)!r})
package_root = court_root / "court_detector"
for name, module in tuple(sys.modules.items()):
    assert name != "experiments" and not name.startswith("experiments."), name
    filename = getattr(module, "__file__", None)
    if filename:
        path = Path(filename).resolve()
        if path.is_relative_to(court_root):
            assert path.is_relative_to(package_root), (name, path)
assert sys.path == before
assert all(sys.modules[name] is sentinel for name, sentinel in sentinels.items())
"""
    completed = run_python(code)
    assert completed.returncode == 0, completed.stderr


def test_runtime_measurements_share_module_identity_and_restore_sampling() -> None:
    live = load_live_modules()
    assert live.verifier is measurements
    assert live.runtime["verifier"] is vars(measurements)
    assert measurements.CaseProvenance is inputs.CaseProvenance is image_sources.CaseProvenance
    original_sample = measurements.grayscale_sample
    original_junctions = measurements.raw_junctions
    frame = np.full((4, 4, 3), 90, dtype=np.uint8)
    points = np.array([[1., 1.], [2., 2.]])
    with live.prepared_measurements(measurements) as counts:
        assert live.runtime["verifier"]["grayscale_sample"] is measurements.grayscale_sample
        assert measurements.grayscale_sample is not original_sample
        for _ in range(2):
            np.testing.assert_array_equal(live.runtime["verifier"]["grayscale_sample"](frame, points), [90., 90.])
        assert counts == {"greyscale_conversions": 1, "sampling_calls": 2}
    assert measurements.grayscale_sample is original_sample
    assert measurements.raw_junctions is original_junctions
    assert live.runtime["verifier"]["grayscale_sample"] is original_sample


@dataclass(frozen=True)
class Inner:
    values: np.ndarray


@dataclass(frozen=True)
class Outer:
    first: np.ndarray
    pair: tuple[np.ndarray, Inner]
    label: str


def test_freeze_arrays_reaches_nested_dataclasses_and_tuples() -> None:
    outer = Outer(np.zeros(2), (np.zeros(3), Inner(np.zeros(4))), "view")
    freeze_arrays(outer)
    for array in (outer.first, outer.pair[0], outer.pair[1].values):
        assert not array.flags.writeable
