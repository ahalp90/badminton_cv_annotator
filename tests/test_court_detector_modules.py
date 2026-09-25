"""The court detector loads the live research copies, and its leaf modules stay safe to import back."""

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

from scratch.court_det_fix.court_detector.detect import LIVE_MODULE_FILES, freeze_arrays
from scratch.court_det_fix.court_detector.search import DIRECTION_SETTINGS

REPO = Path(__file__).resolve().parents[1]
COURT_ROOT = REPO / "scratch/court_det_fix"
LEAVES = ("search", "net_choice", "stripe_refit", "inputs", "feet")
# Research modules that import helpers back from the leaves, as their launchers run them.
RESEARCH_MODULES = (
    "line_identity/paint_profiles.py",
    "line_identity/filter_replay.py",
    "colour_consistency/am1_recovery_trial.py",
    "colour_consistency/am1_net_selection_trial.py",
    "colour_consistency/observed_colour.py",
    "colour_consistency/edge_auto_trial.py",
    "edge_polarity/run_probe.py",
    "net_recovery/bounded_trial.py",
    "net_recovery/refit_selected.py",
)


def run_python(code: str) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "src")))}
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=environment, capture_output=True, text=True,
                          timeout=300, check=False)


def test_direction_settings_equal_the_frozen_direction_file() -> None:
    path = COURT_ROOT / "frozen_views/baseline_directions/gxBQ_window_00_frame_0.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        frozen = json.load(stream)["settings"]
    # Compared as JSON text so an int standing in for a float also fails.
    assert json.dumps(DIRECTION_SETTINGS, sort_keys=True) == json.dumps(frozen, sort_keys=True)


def test_leaf_modules_load_no_research_copies_and_leave_sys_path_alone() -> None:
    imports = "\n".join(f"import scratch.court_det_fix.court_detector.{leaf}" for leaf in LEAVES)
    code = f"""
import sys
from pathlib import Path
before = list(sys.path)
{imports}
court_root = Path({str(COURT_ROOT)!r})
bare = sorted(name for name, module in sys.modules.items()
              if not name.startswith("scratch.") and getattr(module, "__file__", None)
              and Path(module.__file__).resolve().is_relative_to(court_root))
assert bare == [], bare
assert sys.path == before
"""
    completed = run_python(code)
    assert completed.returncode == 0, completed.stderr


def test_load_live_modules_resolves_the_run_d17_copies() -> None:
    code = """
from scratch.court_det_fix.court_detector.detect import load_live_modules
load_live_modules()
"""
    completed = run_python(code)
    assert completed.returncode == 0, completed.stderr


def test_live_module_list_matches_run_d17_import_order() -> None:
    code = f"""
import json, sys
from pathlib import Path
root = Path({str(COURT_ROOT)!r})
repo = root.parents[1]
sys.path[:0] = [str(repo), str(repo / "src"), str(root / "w5_holistic"), str(root / "wider_evaluation"),
                str(root / "colour_consistency"), str(root / "net_recovery")]
import run_cases
run_cases.load_runtime(root, root / "wider_evaluation/runs/20260922/control_inputs.json.gz")
import am1_recovery_trial, automatic_generation, bounded_trial, generation, line_template_source, refit_selected
import run_automatic, run_given
from measurement import prepared_measurements
names = {sorted(LIVE_MODULE_FILES)!r}
print(json.dumps({{name: str(Path(sys.modules[name].__file__).resolve().relative_to(repo)) for name in names}}))
"""
    completed = run_python(code)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == LIVE_MODULE_FILES


@pytest.mark.parametrize("relative", RESEARCH_MODULES)
def test_research_module_imports(relative: str) -> None:
    script = COURT_ROOT / relative
    # As `python script.py` would: the script's folder first on sys.path.
    code = f"import runpy, sys; sys.path.insert(0, {str(script.parent)!r}); runpy.run_path({str(script)!r}, run_name='import_check')"
    completed = run_python(code)
    assert completed.returncode == 0, completed.stderr


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
