"""Find a badminton court in a prepared image.

Search all detected line fragments, then search only paint-like fragments
(the saved records call these G0 and G1). Also build courts from crossing lines.
Measure and refit the candidates, then choose using paint, line and net support.
Finally adjust the fit to the painted stripe edges or centres. By default,
reject sideways and upside-down camera geometry.

Set the numerical-library thread variables to 1 before importing NumPy, as
run_views.py does. load_live_modules() imports this package and sets OpenCV
to one thread. README.md owns the settings and input requirements."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, NamedTuple

import cv2
import numpy as np

from scratch.court_det_fix.court_detector import feet, net_choice, search, stripe_refit
from scratch.court_det_fix.court_detector.inputs import (
    FrameReader,
    PeopleSource,
    ViewInputs,
)

ROOT = Path(__file__).resolve().parents[1]
DIRECTION_BUDGET = 16
VISIBILITY_FLOOR = (4, 3)  # lengthwise and cross-court lines a line template must show
NET_WEIGHT = 0.04
NET_OVERRUN_WORKING_PX = 4.0
# The camera roll a search pair may imply. The chosen courts of the 20 test views with a court
# imply rolls within 2.5 degrees.
MAX_HORIZON_TILT_DEG = 45.0


@dataclass(frozen=True)
class Switches:
    self_checks: bool = True  # reference-field, ranking-order, replay and zero-weight checks
    # TODO: default False once PySceneDetect cuts scenes and is checked on dissolves and
    # lens occlusions.
    enforce_scene_consistency: bool = True  # keep only feet from the anchor's shot
    # skip courts that need a camera rolled past MAX_HORIZON_TILT_DEG or upside down
    upright_camera: bool = True
    geometry_weight: float = 0.1  # share of W5's geometry score in the net choice; the rest is W5's ranking score
    timing: bool = False  # report seconds per step in CourtResult.stage_seconds
    artefacts_dir: Path | None = None  # write each view's intermediate results here
    workers: int = 1  # independent search pairs; run_views limits numerical libraries to one thread

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be positive, not {self.workers}")
        # A NaN weight would make every court's score NaN and the net choice pick none.
        if not 0 <= self.geometry_weight <= 1:
            raise ValueError(f"geometry_weight must be between 0 and 1, not {self.geometry_weight}")


@dataclass(frozen=True)
class CourtResult:
    view_id: str
    corners_native_px: np.ndarray | None  # (4, 2); None means no court
    no_court_reason: str | None  # "no_gated_court", or the refit's validity reason
    chosen_key: str | None  # origin_key the net choice picked
    stage_seconds: dict[str, float] | None  # with timing on


class LiveModules(NamedTuple):
    run_w5: ModuleType
    verifier: ModuleType  # runtime["verifier"] holds these same measurement functions
    generation: ModuleType
    automatic_generation: ModuleType
    run_automatic: ModuleType
    line_template_source: ModuleType
    vp_pruning: ModuleType
    court_model: ModuleType  # court geometry and fitting helpers
    prepared_measurements: Callable
    runtime: dict[str, Any]


class Laps:
    """Seconds between successive calls, by step name."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self._last = perf_counter()

    def lap(self, name: str) -> None:
        now = perf_counter()
        self.seconds[name] = now - self._last
        self._last = now


def load_live_modules() -> LiveModules:
    """Load the detector's package modules with one shared measurement context."""
    from . import (
        candidate_pool,
        court_checks,
        directions,
        generation,
        geometry,
        line_templates,
        measurements,
        players,
        scoring,
        search_records,
    )
    from .sampling import prepared_measurements

    cv2.setNumThreads(1)
    runtime = {
        "verifier": vars(measurements),
        "gate_evidence": court_checks.gate_evidence,
        "zone": players,
    }
    return LiveModules(scoring, measurements, search_records, generation, candidate_pool,
                       line_templates, directions, geometry, prepared_measurements, runtime)


def freeze_arrays(value: Any) -> None:
    """Make every array reachable through tuples, lists and dataclass fields read-only.

    Baseline stages each built their own context; here one context serves every stage, so a
    stray in-place write must fail loudly.
    """
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, (tuple, list)):
        for item in value:
            freeze_arrays(item)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            freeze_arrays(getattr(value, field.name))


def source_record(view: ViewInputs, all_feet_px: list[list]) -> dict:
    """The view in the frozen packs' source form, which the research stages read."""
    height, width = view.frame.shape[:2]
    return {"id": view.view_id, "dimensions": {"width": width, "height": height},
            "segments_px": view.segments_px.tolist(), "bbox_px": view.person_boxes_px.tolist(),
            "all_feet_px": all_feet_px, "provenance": {"people_source": "window_standing_feet"}}


def json_round_trip(value: Any) -> Any:
    """What a stage read back from the JSON file the baseline wrote: tuples become lists."""
    return json.loads(json.dumps(value, allow_nan=False))


class CourtDetector:
    """Loads the detector modules once, then finds the court in one view per detect() call."""

    def __init__(self, switches: Switches) -> None:
        self.switches = switches
        self.live = load_live_modules()

    def detect(self, view: ViewInputs, people: PeopleSource, frames: FrameReader) -> CourtResult:
        live, switches = self.live, self.switches
        laps = Laps()
        artefacts: dict[str, Any] = {}

        feet_window = feet.window_feet(view, people, frames, switches.enforce_scene_consistency)
        artefacts["feet"] = feet_window._asdict()
        laps.lap("feet")

        source = source_record(view, feet_window.all_feet_px)
        native_frame = view.frame.view()
        native_frame.flags.writeable = False
        context = live.verifier.view_context(view.view_id, source, view.provenance, native_frame, view.view_id)
        freeze_arrays(context)
        laps.lap("context")

        populations = self.search(context, source, native_frame, laps)
        artefacts["populations"] = populations
        seeds = search.seed_points(context.families[0])
        generated = live.line_template_source.generate(
            context, live.runtime, live.court_model, min_visible_lengthwise=VISIBILITY_FLOOR[0],
            min_visible_cross_court=VISIBILITY_FLOOR[1], seed_points=seeds,
        )
        templates = list(generated.entries)
        artefacts["line_templates"] = {"entries": templates, "metadata": generated.metadata}
        laps.lap("line_templates")

        with live.prepared_measurements(live.verifier):
            result = self.score_and_choose(view, context, populations, templates, native_frame, laps, artefacts)
        if switches.artefacts_dir is not None:
            live.verifier.write_json_gz(switches.artefacts_dir / f"{view.view_id}.json.gz", artefacts)
        if switches.timing:
            return dataclasses.replace(result, stage_seconds=laps.seconds)
        return result

    def search(self, context: Any, source: dict, native_frame: np.ndarray, laps: Laps) -> dict[str, list[dict]]:
        """G0 on every line fragment and G1 on the painted ones; entries as read back from JSON."""
        live = self.live
        direction = live.generation.direction_record(context, search.DIRECTION_SETTINGS, live.vp_pruning)
        dimensions = source["dimensions"]
        scale = np.asarray([dimensions["width"], dimensions["height"]], dtype=float) / np.asarray(context.size, dtype=float)
        filtered = search.filtered_source(source, search.paint_mask(source, native_frame, scale))
        laps.lap("directions_and_paint_filter")
        populations = {}
        for name, population_source in (("G0", source), ("G1", filtered)):
            record = live.automatic_generation.generate(
                population_source, direction, live.runtime["zone"], ROOT, live.run_automatic, DIRECTION_BUDGET,
                legacy_evidence=False,
                max_horizon_tilt_deg=MAX_HORIZON_TILT_DEG if self.switches.upright_camera else None,
                workers=self.switches.workers,
            )
            record.update({"stage": "results", "population": name})
            if self.switches.self_checks:
                live.generation.validate_population(record, context.case_id, live.run_w5, f"fresh {name} generation")
            populations[name] = json_round_trip(record["entries"])
            laps.lap(f"{name}_search")
        return populations

    def score_and_choose(self, view: ViewInputs, context: Any, populations: dict[str, list[dict]],
                         templates: list[dict], native_frame: np.ndarray, laps: Laps,
                         artefacts: dict[str, Any]) -> CourtResult:
        """W5 scoring, the net choice and the stripe refit. Runs inside prepared_measurements."""
        live, self_checks = self.live, self.switches.self_checks
        scored = live.run_w5.score_populations(
            context, populations["G0"], populations["G1"], templates, live.runtime, {},
            lambda message: print(f"[{view.view_id}] {message}", flush=True), self_checks=self_checks,
        )
        # As the baseline read it back from the case-record file.
        record = json.loads(json.dumps(live.verifier.jsonable({
            "parents": [live.run_w5.public_candidate(parent) for parent in scored.parents],
            "valid_children": [live.run_w5.public_candidate(child) for child in scored.children],
            "fit_attempts": scored.fit_rows,
            "rankings": {"C": scored.c_rankings},
        }), allow_nan=False, sort_keys=True))
        artefacts["w5"] = {"record": record, "identity_resolution": scored.identity_resolution}
        laps.lap("w5")

        return choose_court(view.view_id, record, context, native_frame, scored.line_maps, live, self.switches,
                            laps, artefacts)


def choose_court(view_id: str, record: dict, context: Any, native_frame: np.ndarray, line_maps: np.ndarray,
                 live: LiveModules, switches: Switches, laps: Laps, artefacts: dict[str, Any]) -> CourtResult:
    """The net choice and the stripe refit of its pick, from W5's case record. Runs inside prepared_measurements."""
    rows = net_choice.net_rows(record, context)
    chosen, net_scores = net_choice.choose(rows, NET_WEIGHT, NET_OVERRUN_WORKING_PX, switches.geometry_weight)
    if switches.self_checks:
        gated_top = rows[0]["origin_key"] if rows else None
        if net_choice.choose(rows, 0.0, NET_OVERRUN_WORKING_PX)[0] != gated_top:
            raise RuntimeError(f"{view_id}: zero net weight changed the gated top court")
    artefacts["net_choice"] = {"chosen": chosen, "rows": net_scores}
    laps.lap("net_choice")
    if chosen is None:
        return CourtResult(view_id, None, "no_gated_court", None, None)

    refit = stripe_refit.refit_chosen(record, chosen, context, native_frame, live.verifier, live.runtime, line_maps,
                                      replay_check=switches.self_checks)
    artefacts["stripe_refit"] = refit
    laps.lap("stripe_refit")
    corrected = refit["corrected"]
    if not corrected["valid"]:
        return CourtResult(view_id, None, corrected["validity_reason"], chosen, None)
    return CourtResult(view_id, np.asarray(corrected["corners_native_px"]), None, chosen, None)
