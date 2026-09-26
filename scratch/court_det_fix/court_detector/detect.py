"""Find the badminton court in one view: the accepted D17 chain, run in memory.

The chain: standing feet from a 3 s window, G0 and paint-filtered G1 court searches at
direction budget 16, seeded line templates with a (4, 3) visibility floor, the W5
merge/measure/refit/rank, the bounded net choice (weight 0.04, overrun 4 working px), then
the automatic stripe-polarity refit of the chosen court. By default the court searches skip
courts that need a camera rolled past 45 degrees or upside down (Switches.upright_camera).
Two switches change the final choice, both off by default: a share of W5's geometry score in
the net choice (Switches.geometry_weight), and refitting the best few courts before choosing
among them (Switches.refit_top).

Start-up contract: set the thread variables (OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
OMP_NUM_THREADS, NUMEXPR_NUM_THREADS, VECLIB_MAXIMUM_THREADS, BLIS_NUM_THREADS) to 1 before
numpy is first imported, as run_views.py does. load_live_modules() puts the research
folders on sys.path and sets OpenCV to one thread.
"""

from __future__ import annotations

import dataclasses
import json
import sys
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
REPO = ROOT.parents[1]
DIRECTION_BUDGET = 16
VISIBILITY_FLOOR = (4, 3)  # lengthwise and cross-court lines a line template must show
NET_WEIGHT = 0.04
NET_OVERRUN_WORKING_PX = 4.0
# The camera roll a search pair may imply. The chosen courts of the 20 test views with a court
# imply rolls within 2.5 degrees.
MAX_HORIZON_TILT_DEG = 45.0
# The refitted court's score that matches each W5 ranking criterion (verifier.py rank_candidates).
REFIT_EVIDENCE_FIELDS = {"q_paint10_span_weighted": "paint_score", "q_geom_span_weighted": "geometry_score"}
# The copies run_d17.py resolves. Several research folders hold same-named modules.
LIVE_MODULE_FILES = {
    "run_w5": "scratch/court_det_fix/w5_holistic/run_w5.py",
    "verifier": "scratch/court_det_fix/w5_holistic/verifier.py",
    "automatic_generation": "scratch/court_det_fix/w5_holistic/automatic_generation.py",
    "generation": "scratch/court_det_fix/wider_evaluation/generation.py",
    "run_automatic": "scratch/court_det_fix/next_steps_20260916/webui_seed/source/run_automatic.py",
    "run_given": "scratch/court_det_fix/next_steps_20260916/webui_seed/source/run_given.py",
    "projective_seed": "scratch/court_det_fix/next_steps_20260916/webui_seed/source/projective_seed.py",
    "scan_population": "scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/scan_population.py",
    "run_population": "scratch/court_det_fix/next_steps_20260916/webui_seed/source/run_population.py",
    "run_diagnosis": "scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis/run_diagnosis.py",
    "zone_net": "scratch/court_det_fix/frozen_helpers_20260914/legacy/zone_net.py",
    "line_template_source": "scratch/court_det_fix/w5_holistic/line_template_source.py",
    "vp_pruning": "scratch/court_det_fix/frozen_helpers_20260914/vp_pruning/vp_pruning.py",
    "measurement": "scratch/court_det_fix/wider_evaluation/measurement.py",
    "inspect_appearance": "scratch/court_det_fix/next_steps_20260916/webui_seed/source/inspect_appearance.py",
}


@dataclass(frozen=True)
class Switches:
    self_checks: bool = True  # reference-field, ranking-order, replay and zero-weight checks
    # TODO: default False once PySceneDetect cuts scenes and is checked on dissolves and
    # lens occlusions.
    enforce_scene_consistency: bool = True  # keep only feet from the anchor's shot
    # skip courts that need a camera rolled past MAX_HORIZON_TILT_DEG or upside down
    upright_camera: bool = True
    geometry_weight: float = 0.0  # share of W5's geometry score in the net choice; the rest is W5's ranking score
    refit_top: int = 1  # refit this many of the net choice's best courts, then choose among the refitted ones
    timing: bool = False  # report seconds per step in CourtResult.stage_seconds
    artefacts_dir: Path | None = None  # write each view's intermediate results here


@dataclass(frozen=True)
class CourtResult:
    view_id: str
    corners_native_px: np.ndarray | None  # (4, 2); None means no court
    no_court_reason: str | None  # "no_gated_court", or the refit's validity reason
    chosen_key: str | None  # origin_key the net choice picked
    stage_seconds: dict[str, float] | None  # with timing on


class LiveModules(NamedTuple):
    run_w5: ModuleType
    verifier: ModuleType  # the W5 verifier module; runtime["verifier"] holds its functions
    generation: ModuleType
    automatic_generation: ModuleType
    run_automatic: ModuleType
    line_template_source: ModuleType
    vp_pruning: ModuleType
    court_model: ModuleType  # experiments' independent-court detector: court geometry and fitting helpers
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
    """Import the chain's research modules in run_d17.py's path order and check their copies."""
    cv2.setNumThreads(1)
    sys.path[:0] = [str(REPO), str(REPO / "src"), str(ROOT / "w5_holistic"), str(ROOT / "wider_evaluation")]
    import run_w5  # pyrefly: ignore[missing-import]

    runtime = run_w5.load_runtime(ROOT)
    import automatic_generation  # pyrefly: ignore[missing-import]
    import generation  # pyrefly: ignore[missing-import]
    import line_template_source  # pyrefly: ignore[missing-import]
    import verifier  # pyrefly: ignore[missing-import]
    from measurement import prepared_measurements  # pyrefly: ignore[missing-import]

    for name, relative in LIVE_MODULE_FILES.items():
        resolved = Path(sys.modules[name].__file__).resolve()
        if resolved != REPO / relative:
            raise ImportError(f"{name} resolved to {resolved}, expected {relative}")
    return LiveModules(run_w5, verifier, generation, automatic_generation, sys.modules["run_automatic"],
                       line_template_source, sys.modules["vp_pruning"], run_w5.import_detector(),
                       prepared_measurements, runtime)


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
    """Loads the research modules once, then finds the court in one view per detect() call."""

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


def refitted_score(corrected: dict, criterion: str, context: Any, geometry_weight: float) -> float | None:
    """The net choice's score, measured on a refitted court; None if the court is invalid or unscored."""
    if not corrected["valid"]:
        return None
    measurement = corrected["measurement"]
    evidence = {"paint_score": measurement[REFIT_EVIDENCE_FIELDS[criterion]],
                "geometry_score": measurement["geometry_score"]}
    if evidence["paint_score"] is None:
        return None
    reward = net_choice.net_reward(*net_choice.net_posts(corrected["corners_native_px"], context),
                                   NET_OVERRUN_WORKING_PX)
    return net_choice.evidence_score(evidence, geometry_weight) + NET_WEIGHT * reward


def choose_court(view_id: str, record: dict, context: Any, native_frame: np.ndarray, line_maps: np.ndarray,
                 live: LiveModules, switches: Switches, laps: Laps, artefacts: dict[str, Any]) -> CourtResult:
    """The net choice and the stripe refit, from W5's case record. Runs inside prepared_measurements.

    The best switches.refit_top courts by the net choice's score are refitted, then scored again on
    their refitted geometry; the best valid one wins. With refit_top 1 this refits only the net
    choice's pick, as the accepted chain does.
    """
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

    # The sort is stable, so ties keep W5 order and the first court is the net choice's pick.
    shortlist = sorted(net_scores, key=lambda row: -row["combined_score"])[:switches.refit_top]
    criterion = record["rankings"]["C"]["r2_criterion"]
    refits, refit_rows = [], []
    best_index, best_score = 0, -np.inf
    for index, row in enumerate(shortlist):
        refit = stripe_refit.refit_chosen(record, row["origin_key"], context, native_frame, live.verifier,
                                          live.runtime, line_maps, replay_check=switches.self_checks)
        score = refitted_score(refit["corrected"], criterion, context, switches.geometry_weight)
        if score is not None and score > best_score:
            best_index, best_score = index, score
        refits.append(refit)
        refit_rows.append({"origin_key": row["origin_key"], "combined_score": row["combined_score"],
                           "refitted_score": score, "corrected": refit["corrected"]})
    refit = refits[best_index]
    artefacts["stripe_refit"] = refit
    artefacts["refit_choice"] = refit_rows
    laps.lap("stripe_refit")
    corrected = refit["corrected"]
    if not corrected["valid"]:
        return CourtResult(view_id, None, corrected["validity_reason"], chosen, None)
    return CourtResult(view_id, np.asarray(corrected["corners_native_px"]), None, refit["selected_origin_key"], None)
