"""Generate the two bounded candidate populations for one wider-evaluation view."""

from __future__ import annotations

import gzip
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
from automatic_generation import SCREEN_METHOD, generate


def _module(runtime: dict, name: str) -> ModuleType:
    loaded = runtime.get(name)
    if isinstance(loaded, ModuleType):
        return loaded
    loaded = sys.modules.get(name)
    if isinstance(loaded, ModuleType):
        return loaded
    return importlib.import_module(name)

def _filter_module(root: Path) -> ModuleType:
    loaded = sys.modules.get("filter_replay")
    if isinstance(loaded, ModuleType) and all(hasattr(loaded, name) for name in ("paint_masks", "filtered_source")):
        return loaded
    path = root / "line_identity/filter_replay.py"
    spec = importlib.util.spec_from_file_location("wider_line_identity_filter_replay", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    helper_dir = str(path.parent)
    original_sys_path = sys.path.copy()
    sys.path.insert(0, helper_dir)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module

def _read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)

def _write_json_gz(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gzip.compress(json.dumps(value, allow_nan=False, separators=(",", ":")).encode(), mtime=0)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
    temporary.replace(path)

def _validate_population(record: dict, case_id: str, run_w5: ModuleType, source: str) -> dict:
    run_w5.validate_generation_record(record, case_id, source, expected_stage="results", validate_entries=False)
    entries = record.get("entries")
    if not isinstance(entries, list) or len(entries) > 256:
        raise ValueError(f"{case_id}: {source} has an invalid entry count")
    candidate_ids = [entry.get("candidate_id") for entry in entries]
    if not all(isinstance(candidate_id, str) and candidate_id for candidate_id in candidate_ids):
        raise ValueError(f"{case_id}: {source} has an invalid candidate ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{case_id}: {source} candidate IDs are not unique")
    for winner_name in ("line_winner_id", "paint_winner_id"):
        winner_id = record.get(winner_name)
        if winner_id is not None and winner_id not in set(candidate_ids):
            raise ValueError(f"{case_id}: {source} {winner_name} is outside its entries")
    return record

def screen_matches(record: dict, budget: int) -> bool:
    screen = record.get("direction_screen")
    if screen is None:
        return budget == 16
    return screen.get("method") == SCREEN_METHOD and screen.get("budget") == budget


def _cached(path: Path, case_id: str, run_w5: ModuleType, population: str, budget: int) -> dict | None:
    if not path.exists():
        return None
    record = _validate_population(_read_json_gz(path), case_id, run_w5, f"cached {population} generation")
    if not screen_matches(record, budget):
        raise ValueError(f"{case_id}: cached {population} direction screen differs from requested budget/method")
    return record

def _direction_from_record(record: dict, case_id: str) -> dict:
    try:
        direction = {
            "schema": "wider-w5-direction/1",
            "case_id": case_id,
            "working_size": record["working_size"],
            "settings": record["estimator_settings"],
            "estimator": record["estimator"],
        }
    except KeyError as error:
        raise ValueError(f"{case_id}: cached generation lacks direction data") from error
    if direction["case_id"] != record.get("case_id"):
        raise ValueError(f"{case_id}: cached generation direction identity differs")
    return direction

def _new_direction(root: Path, context: Any, verifier: dict, vp_pruning: ModuleType) -> dict:
    baseline_path = root / "frozen_views/baseline_directions/gxBQ_window_00_frame_0.json.gz"
    baseline = verifier.get("read_json_gz", _read_json_gz)(baseline_path)
    settings = vp_pruning.Settings(**baseline["settings"])
    working_size = tuple(context.size)
    _points, estimator = vp_pruning.estimate(
        np.asarray(context.segments, dtype=float), working_size, settings
    )
    return {
        "schema": "wider-w5-direction/1",
        "case_id": context.case_id,
        "working_size": list(working_size),
        "settings": baseline["settings"],
        "estimator": estimator,
    }

def _native_frame(root: Path, context: Any, verifier: dict) -> tuple[np.ndarray, Path]:
    frame_path = verifier["frame_path"](root, context.source, context.provenance)
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(frame_path)
    expected = (context.source["dimensions"]["height"], context.source["dimensions"]["width"])
    if frame.shape[:2] != expected:
        raise ValueError(f"{context.case_id}: native frame shape {frame.shape[:2]} != {expected}")
    return frame, frame_path

def ensure_populations(root: Path, context: Any, runtime: dict, output: Path,
                       direction_budget: int = 12) -> dict[str, Path]:
    """Ensure fresh G0/G1 generation records for one frozen view.

    :param root: Investigation root containing frozen packs and helper snapshots.
    :param context: Prepared verifier view with ``source``, ``segments`` and ``size``.
    :param runtime: Already-loaded W5/runtime modules and verifier functions.
    :param output: Isolated wider-evaluation run directory.
    :return: Paths for the ``G0`` and ``G1`` generation records.
    """
    case_id = context.case_id
    verifier = runtime["verifier"]
    run_w5 = _module(runtime, "run_w5")
    run_automatic = _module(runtime, "run_automatic")
    paths = {
        "G0": output / "populations/G0" / f"{case_id}.json.gz",
        "G1": output / "populations/G1" / f"{case_id}.json.gz",
        "inputs": output / "inputs" / f"{case_id}.json.gz",
    }
    if direction_budget not in (12, 16):
        raise ValueError(f"direction budget must be 12 or 16, got {direction_budget}")
    cached = {name: _cached(path, case_id, run_w5, name, direction_budget)
              for name, path in paths.items() if name != "inputs"}
    direction = next((
        _direction_from_record(record, case_id) for record in cached.values() if record is not None
    ), None)
    if direction is None:
        direction = _new_direction(root, context, verifier, _module(runtime, "vp_pruning"))

    filter_replay = _filter_module(root)
    frame, frame_path = _native_frame(root, context, verifier)
    dimensions = context.source["dimensions"]
    scale = np.asarray([dimensions["width"], dimensions["height"]], dtype=float) / np.asarray(context.size, dtype=float)
    paint_mask = filter_replay.paint_masks(context.source, frame, scale)["paint"]
    filtered = filter_replay.filtered_source(context.source, paint_mask)
    audit = {
        "schema": "wider-w5-generation-inputs/1",
        "case_id": case_id,
        "frame_path": str(frame_path.relative_to(root)),
        "direction": direction,
        "source": context.source,
        "filtered_source": filtered,
        "paint_mask": paint_mask.tolist(),
    }
    if paths["inputs"].exists():
        if _read_json_gz(paths["inputs"]) != audit:
            raise ValueError(f"{case_id}: cached input audit differs")
    else:
        _write_json_gz(paths["inputs"], audit)

    original_frame_path = run_automatic.frame_path
    run_automatic.frame_path = lambda source, _root: verifier["frame_path"](
        root, source, context.provenance
    )
    try:
        for name, source in (("G0", context.source), ("G1", filtered)):
            if cached[name] is not None:
                continue
            print(f"[{case_id}] generating {name}", flush=True)
            result = generate(source, direction, runtime["zone"], root, run_automatic, direction_budget)
            result.update({"stage": "results", "population": name})
            _validate_population(result, case_id, run_w5, f"fresh {name} generation")
            _write_json_gz(paths[name], result)
    finally:
        run_automatic.frame_path = original_frame_path
    return {name: paths[name] for name in ("G0", "G1")}
