"""Find a badminton court in a prepared image.

Search all detected line fragments, then search only paint-like fragments
(all_lines and painted_lines; older saved records call them G0 and G1). Also
build courts from crossing lines. Measure and refit the candidates, then choose
using paint, line and net support. Finally adjust the fit to the painted stripe
edges or centres. By default, reject sideways and upside-down camera geometry.

Set the numerical-library thread variables to 1 before importing NumPy, as
run_views.py does. load_live_modules() imports this package and sets OpenCV
to one thread. README.md owns the settings and input requirements."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import cv2
import numpy as np

from . import feet, net_choice, search, stripe_refit, template_arrays
from .inputs import FrameReader, PeopleSource, ViewInputs

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

    from .reuse import KnownCourt

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
    require_people: bool = True  # otherwise allow a court supported by lines and camera geometry alone
    # TODO: default False once PySceneDetect cuts scenes and is checked on dissolves and
    # lens occlusions.
    enforce_scene_consistency: bool = True  # keep only feet from the anchor's shot
    # skip courts that need a camera rolled past MAX_HORIZON_TILT_DEG or upside down
    upright_camera: bool = True
    geometry_weight: float = 0.1  # share of the geometry score in the net choice; the rest is the C ranking's score
    timing: bool = False  # report seconds per step in CourtResult.stage_seconds
    artefacts_dir: Path | None = None  # write each view's intermediate results here
    workers: int = 1  # search pairs and scoring; run_views limits numerical libraries to one thread
    full_score_limit: int | None = None  # optional 16-sample shortlist before the usual 64-sample score
    template_device: str = "cpu"  # "cuda" scores line templates with CuPy; other stages keep their devices

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be positive, not {self.workers}")
        if self.full_score_limit is not None and self.full_score_limit < 1:
            raise ValueError("full_score_limit must be positive, or None for exhaustive scoring")
        # A NaN weight would make every court's score NaN and the net choice pick none.
        if not 0 <= self.geometry_weight <= 1:
            raise ValueError(f"geometry_weight must be between 0 and 1, not {self.geometry_weight}")
        if self.template_device not in template_arrays.TEMPLATE_DEVICES:
            raise ValueError(f"template_device must be one of {template_arrays.TEMPLATE_DEVICES}, "
                             f"not {self.template_device!r}")


@dataclass(frozen=True)
class CourtResult:
    view_id: str
    corners_native_px: np.ndarray | None  # (4, 2); None means no court
    no_court_reason: str | None  # "no_gated_court", or the refit's validity reason
    chosen_key: str | None  # origin_key the net choice picked
    stage_seconds: dict[str, float] | None  # with timing on
    paint_score: float | None = None  # final stripe refit's paint support
    reused_from: str | None = None  # source view ID; reused courts must not become reuse templates


class LiveModules(NamedTuple):
    scoring: ModuleType
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


def measurement_runtime() -> dict[str, Any]:
    """The shared measurement functions, without changing OpenCV's thread setting."""
    from . import court_checks, measurements, players

    return {"verifier": vars(measurements), "gate_evidence": court_checks.gate_evidence, "zone": players}


def load_live_modules() -> LiveModules:
    """Load the detector's package modules with one shared measurement context."""
    from . import (
        candidate_pool,
        directions,
        generation,
        geometry,
        line_templates,
        measurements,
        scoring,
        search_records,
    )
    from .sampling import prepared_measurements

    cv2.setNumThreads(1)
    runtime = measurement_runtime()
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
            "all_feet_px": all_feet_px,
            "provenance": {"people_source": "window_standing_feet" if all_feet_px else "not_supplied"}}


def json_round_trip(value: Any) -> Any:
    """What a stage read back from the JSON file the baseline wrote: tuples become lists."""
    return json.loads(json.dumps(value, allow_nan=False))


class CourtDetector:
    """Loads the detector modules once, then finds the court in one view per detect() call.

    With workers above 1, use it as a context manager. Every detect() call inside the with
    block then shares one pool of worker processes, and the pool closes when the block ends.
    Outside a with block, each search and scoring step starts and closes its own workers.
    """

    def __init__(self, switches: Switches) -> None:
        self.switches = switches
        self.live = load_live_modules()
        # Check CuPy and the GPU now, not when the first view reaches the line templates.
        template_arrays.array_module(switches.template_device)
        self.pool: ProcessPoolExecutor | None = None  # open only inside a with block with several workers

    def __enter__(self) -> Self:
        if self.pool is not None:
            raise RuntimeError("this detector's worker pool is already open")
        if self.switches.workers > 1:
            from .generation import worker_pool

            self.pool = worker_pool(self.switches.workers)
        return self

    def __exit__(self, *_exception_info: object) -> None:
        if self.pool is not None:
            # Tasks still waiting after a failure are dropped; running ones finish before the workers close.
            self.pool.shutdown(cancel_futures=True)
            self.pool = None

    def detect(self, view: ViewInputs, people: PeopleSource | None, frames: FrameReader,
               *, known_courts: Sequence[KnownCourt] = ()) -> CourtResult:
        live, switches = self.live, self.switches
        if switches.require_people and people is None:
            raise ValueError("People inputs are required; use require_people=False for the line-only fallback")
        laps = Laps()
        artefacts: dict[str, Any] = {}

        feet_window = feet.window_feet(view, people, frames, switches.enforce_scene_consistency,
                                      not switches.require_people)
        artefacts["feet"] = feet_window._asdict()
        laps.lap("feet")

        source = source_record(view, feet_window.all_feet_px)
        native_frame = view.frame.view()
        native_frame.flags.writeable = False
        context = live.verifier.view_context(view.view_id, source, view.provenance, native_frame, view.view_id)
        freeze_arrays(context)
        laps.lap("context")

        # Validate the view first. Diagnostic runs still need the actual candidate
        # records for comparison, even when the required player counts cannot pass.
        if (switches.require_people and switches.artefacts_dir is None
                and not feet.can_satisfy_player_requirement(feet_window.all_feet_px)):
            return self.finish(CourtResult(view.view_id, None, "no_gated_court", None, None), laps, artefacts)

        if known_courts:
            from .reuse import try_reuse, view_image

            artefacts["reuse"] = []
            alignment_image = view_image(native_frame) if view.alignment_image is None else view.alignment_image
            for known in known_courts:
                attempt = try_reuse(known, context, native_frame, live,
                                    max_horizon_tilt_deg=MAX_HORIZON_TILT_DEG if switches.upright_camera else None,
                                    require_people=switches.require_people, alignment_image=alignment_image)
                artefacts["reuse"].append(attempt.record)
                if attempt.court is not None:
                    court = attempt.court
                    laps.lap("reuse")
                    result = CourtResult(view.view_id, court.corners_native_px, None, "reuse", None,
                                         court.paint_score, court.source_view_id)
                    return self.finish(result, laps, artefacts)
            laps.lap("reuse")

        populations = self.search(context, source, native_frame, laps, artefacts)
        artefacts["populations"] = populations
        seeds = search.seed_points(context.families[0])
        generated = live.line_template_source.generate(
            context, live.runtime, live.court_model, min_visible_lengthwise=VISIBILITY_FLOOR[0],
            min_visible_cross_court=VISIBILITY_FLOOR[1], seed_points=seeds, device=switches.template_device,
        )
        templates = list(generated.entries)
        artefacts["line_templates"] = {"entries": templates, "metadata": generated.metadata}
        laps.lap("line_templates")

        with live.prepared_measurements(live.verifier):
            result = self.score_and_choose(view, context, populations, templates, native_frame, laps, artefacts)
        return self.finish(result, laps, artefacts)

    def finish(self, result: CourtResult, laps: Laps, artefacts: dict[str, Any]) -> CourtResult:
        """Save diagnostics and attach timings for searched and reused courts alike."""
        if self.switches.artefacts_dir is not None:
            self.live.verifier.write_json_gz(self.switches.artefacts_dir / f"{result.view_id}.json.gz", artefacts)
        if self.switches.timing:
            return dataclasses.replace(result, stage_seconds=laps.seconds)
        return result

    def search(self, context: Any, source: dict, native_frame: np.ndarray, laps: Laps,
               artefacts: dict[str, Any] | None = None) -> dict[str, list[dict]]:
        """Search every line fragment (all_lines), then painted ones (painted_lines); entries as read back from JSON."""
        live = self.live
        # Direction-pair proposals require player occupancy. The independent line
        # templates below supply the fallback when there are no people inputs.
        if not source["all_feet_px"]:
            return {"all_lines": [], "painted_lines": []}
        direction = live.generation.direction_record(context, search.DIRECTION_SETTINGS, live.vp_pruning)
        dimensions = source["dimensions"]
        scale = np.asarray([dimensions["width"], dimensions["height"]], dtype=float) / np.asarray(context.size, dtype=float)
        filtered = search.filtered_source(source, search.paint_mask(source, native_frame, scale))
        laps.lap("directions_and_paint_filter")
        populations = {}
        for name, population_source in (("all_lines", source), ("painted_lines", filtered)):
            record = live.automatic_generation.generate(
                population_source, direction, live.runtime["zone"], live.run_automatic, DIRECTION_BUDGET,
                max_horizon_tilt_deg=MAX_HORIZON_TILT_DEG if self.switches.upright_camera else None,
                workers=self.switches.workers,
                full_score_limit=self.switches.full_score_limit,
                pool=self.pool,
            )
            record.update({"stage": "results", "population": name})
            if self.switches.self_checks:
                live.generation.validate_population(record, context.case_id, live.scoring, f"fresh {name} generation")
            populations[name] = json_round_trip(record["entries"])
            if artefacts is not None and self.switches.full_score_limit is not None:
                ranks = {}
                for pair in record["pairs"]:
                    cheap_ranks = pair.get("role", {}).get("retained_cheap_ranks")
                    if cheap_ranks is not None:
                        ranks.update({court["candidate_id"]: rank
                                      for court, rank in zip(pair["shortlist"], cheap_ranks, strict=True)})
                artefacts.setdefault("cheap_score_ranks", {})[name] = ranks
            laps.lap(f"{name}_search")
        return populations

    def score_and_choose(self, view: ViewInputs, context: Any, populations: dict[str, list[dict]],
                         templates: list[dict], native_frame: np.ndarray, laps: Laps,
                         artefacts: dict[str, Any]) -> CourtResult:
        """Candidate scoring, the net choice and the stripe refit. Runs inside prepared_measurements."""
        live, self_checks = self.live, self.switches.self_checks
        scored = live.scoring.score_populations(
            context, populations["all_lines"], populations["painted_lines"], templates, live.runtime, {},
            lambda message: print(f"[{view.view_id}] {message}", flush=True), self_checks=self_checks,
            workers=self.switches.workers, pool=self.pool,
        )
        record = live.verifier.jsonable({
            "parents": [live.scoring.public_candidate(parent) for parent in scored.parents],
            "valid_children": [live.scoring.public_candidate(child) for child in scored.children],
            "fit_attempts": scored.fit_rows,
            "rankings": {"C": scored.c_rankings},
        })
        artefacts["scoring"] = {"record": record, "identity_resolution": scored.identity_resolution}
        laps.lap("scoring")

        return choose_court(view.view_id, record, context, native_frame, scored.line_maps, live, self.switches,
                            laps, artefacts)


def choose_court(view_id: str, record: dict, context: Any, native_frame: np.ndarray, line_maps: np.ndarray,
                 live: LiveModules, switches: Switches, laps: Laps, artefacts: dict[str, Any]) -> CourtResult:
    """The net choice and the stripe refit of its pick, from the scoring record. Runs inside prepared_measurements."""
    rows = net_choice.net_rows(record, context, require_people=switches.require_people)
    chosen, net_scores = net_choice.choose(rows, NET_WEIGHT, NET_OVERRUN_WORKING_PX, switches.geometry_weight,
                                           require_people=switches.require_people)
    if switches.self_checks:
        gated_top = rows[0]["origin_key"] if rows else None
        if net_choice.choose(rows, 0.0, NET_OVERRUN_WORKING_PX, require_people=switches.require_people)[0] != gated_top:
            raise RuntimeError(f"{view_id}: zero net weight changed the gated top court")
    artefacts["net_choice"] = {"chosen": chosen, "rows": net_scores}
    if not switches.require_people:
        artefacts["net_choice"]["require_people"] = False
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
    if not switches.require_people and not corrected["measurement"]["historical"]["historical_camera"]:
        return CourtResult(view_id, None, "refit_camera_implausible", chosen, None)
    return CourtResult(view_id, np.asarray(corrected["corners_native_px"]), None, chosen, None,
                       corrected["measurement"]["paint_score"])
