"""Time the fresh D17 court detector on one view, stage by stage.

The chain follows the launch plan (handover_20260923/01_LAUNCH_OPTIMISATION.md, step 3):
fresh G0 and paint-filtered G1 generation at the requested direction budget, seeded line
templates, the W5 merge/score/refit/rank record, the bounded net choice (weight 0.04,
overrun 4 working px), then the automatic stripe-polarity refit of the chosen court.
Selection happens before correction, as in the accepted 20-case gallery.

Every step calls the existing code. This script adds three things: the seeded-template
hook, the net-choice rows (as in bounded_trial.measure_case, without its frozen-data
checks), and timers. Line fragments and player feet come from the frozen view packs, so
DeepLSD and pose inference are not timed. One view and one arm run per process, so each
run pays start-up once. Launch with D17_LAUNCH=$(date +%s.%N) to include interpreter
start-up.
"""

from __future__ import annotations

import time

SCRIPT_START = time.time()

import os

# One numerical thread per process, as in the SVD-search runs; set before numpy loads.
for thread_variable in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[thread_variable] = "1"

import argparse
import functools
import gzip
import json
import resource
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
NET_WEIGHT = 0.04
NET_OVERRUN_PX = 4.0
VISIBILITY_FLOOR = (4, 3)


class StageClock:
    """Add up wall time per call path, so a nested stage stays inside its caller."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.path: list[str] = []

    @contextmanager
    def stage(self, label: str):
        self.path.append(label)
        key = "/".join(self.path)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.totals[key] += time.perf_counter() - started
            self.calls[key] += 1
            self.path.pop()

    def wrap(self, owner, name: str, label: str | None = None) -> None:
        function = owner[name] if isinstance(owner, dict) else getattr(owner, name)

        @functools.wraps(function)
        def timed(*positional, **named):
            with self.stage(label or name):
                return function(*positional, **named)

        if isinstance(owner, dict):
            owner[name] = timed
        else:
            setattr(owner, name, timed)

    def report(self) -> dict[str, dict]:
        """Seconds per path, plus self seconds: the part no timed child accounts for."""
        stages = {}
        for path, seconds in sorted(self.totals.items()):
            depth = path.count("/") + 1
            children = sum(
                child_seconds for child, child_seconds in self.totals.items()
                if child.startswith(path + "/") and child.count("/") + 1 == depth + 1
            )
            stages[path] = {"seconds": seconds, "self_seconds": seconds - children, "calls": self.calls[path]}
        return stages


def net_rows(record: dict, context, net, bounded_trial, np) -> list[dict]:
    """Rows for bounded_trial.choose, built as bounded_trial.measure_case builds them."""
    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    ranking = record["rankings"]["C"]["provisional_rank"]
    criterion = record["rankings"]["C"]["r2_criterion"]
    rows = []
    full_rank = 0
    for rank, key in enumerate(ranking, start=1):
        candidate = candidates[key]
        if not candidate["historical"]["historical_fullcourt"]:
            continue
        full_rank += 1
        projection = net.project_pieces(candidate["corners_px"], context)
        posts = {}
        if projection["state"] == "measured":
            for name, piece_index in (("post_left", 2), ("post_right", 3)):
                posts[name] = bounded_trial.post_features(
                    np.asarray(projection["pieces_working_px"][piece_index]), context.segments, context.size,
                )
        rows.append({
            "origin_key": key, "candidate_id": candidate["candidate_id"], "source": candidate["source"],
            "original_rank": rank, "full_court_rank": full_rank, "historical_fullcourt": True,
            "camera_eligible": candidate["camera_eligible"], "gate_camera_error": candidate["gates"]["camera_error"],
            "paint_score": candidate["evidence"][criterion], "net_state": projection["state"], "posts": posts,
        })
    return rows


def population_summary(record: dict) -> dict:
    statuses: dict[str, int] = defaultdict(int)
    for pair in record["pairs"]:
        statuses[pair["status"]] += 1
    return {"entries": len(record["entries"]), "pair_statuses": dict(statuses),
            "direction_screen": {key: record["direction_screen"][key]
                                 for key in ("method", "budget", "selected_original_ids", "ranked_original_ids")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--direction-budget", type=int, choices=(12, 16), required=True)
    parser.add_argument("--output", type=Path, required=True, help="arm directory, one per direction budget")
    parser.add_argument("--max-matched-pairs", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()
    case_id = args.case
    arm_dir = args.output.resolve()
    summary_path = arm_dir / "d17" / f"{case_id}.json.gz"
    # Existing populations would be reused silently by ensure_populations and skip the timed work.
    for leftover in (arm_dir / "populations/G0" / f"{case_id}.json.gz", arm_dir / "populations/G1" / f"{case_id}.json.gz",
                     arm_dir / "case_records" / f"{case_id}.json.gz", summary_path):
        if leftover.exists():
            raise FileExistsError(f"{leftover}: use a fresh arm directory")

    startup: dict[str, float] = {}
    if "D17_LAUNCH" in os.environ:
        startup["interpreter_start"] = SCRIPT_START - float(os.environ["D17_LAUNCH"])
    clock_start = time.perf_counter()
    cpu_start = time.process_time()

    started = time.perf_counter()
    import cv2
    import numpy as np

    startup["import_numpy_cv2"] = time.perf_counter() - started

    started = time.perf_counter()
    cv2.setNumThreads(1)
    sys.path[:0] = [str(REPO), str(REPO / "src"), str(ROOT / "w5_holistic"), str(ROOT / "wider_evaluation"),
                    str(ROOT / "colour_consistency"), str(ROOT / "net_recovery")]
    import run_cases  # pyrefly: ignore[missing-import]

    run_w5, verifier_module, runtime = run_cases.load_runtime(ROOT, ROOT / "wider_evaluation/runs/20260922/control_inputs.json.gz")
    import am1_recovery_trial  # pyrefly: ignore[missing-import]
    import automatic_generation  # pyrefly: ignore[missing-import]
    import bounded_trial  # pyrefly: ignore[missing-import]
    import generation  # pyrefly: ignore[missing-import]
    import line_template_source  # pyrefly: ignore[missing-import]
    import refit_selected  # pyrefly: ignore[missing-import]
    import run_automatic  # pyrefly: ignore[missing-import]
    import run_given  # pyrefly: ignore[missing-import]
    from measurement import prepared_measurements  # pyrefly: ignore[missing-import]

    from experiments.annotator.independent_court import (
        assignment,
        detector,
        fixed_stripe_refit,
        stripe_observations,
    )

    manifest = verifier_module.read_json_gz(refit_selected.MANIFEST)
    label = verifier_module.CASE_LABELS[case_id]
    startup["import_and_load_runtime"] = time.perf_counter() - started

    vp_pruning = sys.modules["vp_pruning"]
    net = bounded_trial.net
    # The seeded generator must patch the same modules that run_w5 and generation use.
    assert am1_recovery_trial.line_template_source is line_template_source
    assert am1_recovery_trial.vp_pruning is vp_pruning
    assert generation.generate is automatic_generation.generate

    clock = StageClock()
    # G0 and G1 generation, per direction pair and for the pooled shortlist.
    clock.wrap(generation, "_new_direction", "direction_estimate")
    clock.wrap(generation, "_write_json_gz", "write_population")
    clock.wrap(run_automatic, "prepare", "load_fragments")
    clock.wrap(assignment, "prepare_observations")
    clock.wrap(run_automatic, "propose_role", "pair")
    clock.wrap(run_given, "basis_for")
    clock.wrap(run_given, "match_axis")
    clock.wrap(run_given, "combine")
    clock.wrap(run_given, "canonicalise")
    clock.wrap(run_given, "geometry")
    clock.wrap(runtime["zone"], "player_fractions")
    clock.wrap(run_given, "finite_scores")
    clock.wrap(run_automatic, "retain")
    clock.wrap(run_automatic, "evaluate_pool", "pool_evidence")
    clock.wrap(run_automatic, "gate_evidence")
    clock.wrap(run_automatic, "profiles", "paint_profiles")
    # Seeded line templates.
    clock.wrap(vp_pruning, "estimate", "vp_estimate")
    clock.wrap(vp_pruning, "rectangle_population")
    clock.wrap(vp_pruning, "select", "vp_select")
    clock.wrap(line_template_source, "geometry_and_support")
    clock.wrap(line_template_source, "camera_errors_with_frontier_recheck", "template_camera_check")
    clock.wrap(line_template_source, "select_with_visibility_floor", "template_selection")
    clock.wrap(line_template_source, "_ranked_records", "template_records")
    clock.wrap(line_template_source, "attach_w5_gates")
    # W5 record. process_case loads its own runtime, which picks up these module functions.
    clock.wrap(run_w5, "load_runtime")
    clock.wrap(run_w5, "canonicalise_populations")
    clock.wrap(run_w5, "make_parent_record")
    clock.wrap(run_w5, "attempt_refit")
    clock.wrap(run_w5, "rank_sensitivity")
    clock.wrap(run_w5, "load_control_entry")
    clock.wrap(sys.modules["run_diagnosis"], "gate_evidence")
    for name in ("prepare_view", "measure_candidate", "physical_marking_evidence", "rank_candidates",
                 "permutation_determinism", "candidate_review", "write_json_gz", "read_json_gz"):
        clock.wrap(verifier_module, name)
    clock.wrap(np, "savez_compressed", "write_arrays")
    # Stripe-polarity refit.
    clock.wrap(fixed_stripe_refit, "prepare", "refit_prepare")
    clock.wrap(fixed_stripe_refit, "refine", "refit_refine")
    clock.wrap(refit_selected.probe, "relabel", "polarity_relabel")
    clock.wrap(refit_selected.edge_auto_trial, "infer_polarity")
    # Shared helpers appear under whichever stage called them.
    clock.wrap(detector, "_distance_maps", "distance_maps")
    clock.wrap(stripe_observations, "measure", "stripe_measure")
    clock.wrap(stripe_observations, "score_model", "stripe_score_model")

    population_labels = ["G0_generation", "G1_generation"]
    generate = generation.generate

    def generate_population(*positional, **named):
        if args.max_matched_pairs is not None:
            named["max_matched_pairs"] = args.max_matched_pairs
        with clock.stage(population_labels.pop(0)):
            return generate(*positional, **named)

    generation.generate = generate_population

    def seeded_line_templates(context, runtime, detector, *, min_visible_lengthwise, min_visible_cross_court):
        if (min_visible_lengthwise, min_visible_cross_court) != VISIBILITY_FLOOR:
            raise ValueError("D17 uses the (4, 3) visibility floor")
        with clock.stage("line_templates"):
            generated, _seeds = am1_recovery_trial.generate_seeded(context, runtime, detector)
        return generated

    import_runtime = run_w5.import_runtime

    def import_runtime_with_seeds(root: Path) -> dict:
        loaded = import_runtime(root)
        loaded["line_template"] = seeded_line_templates
        return loaded

    run_w5.import_runtime = import_runtime_with_seeds

    phases: dict[str, float] = {}
    started = time.perf_counter()
    with clock.stage("prepare_view"):
        context = verifier_module.prepare_view(ROOT, case_id)
    phases["prepare_view"] = time.perf_counter() - started

    started = time.perf_counter()
    with clock.stage("populations"):
        population_paths = generation.ensure_populations(ROOT, context, runtime, arm_dir, args.direction_budget)
        populations = {name: verifier_module.read_json_gz(path) for name, path in population_paths.items()}
    phases["populations"] = time.perf_counter() - started
    if population_labels:
        raise RuntimeError(f"{case_id}: expected fresh G0 and G1 generation, missing {population_labels}")

    # As in wider_evaluation/run_cases.run_case: W5 reads the fresh populations.
    run_w5.load_g0 = lambda *_args: (populations["G0"]["entries"], str(population_paths["G0"]))
    run_w5.load_g1 = lambda *_args: (populations["G1"]["entries"], str(population_paths["G1"]))
    started = time.perf_counter()
    with clock.stage("w5_record"), prepared_measurements(verifier_module) as measurement_counts:
        w5_result = run_w5.process_case(
            ROOT, case_id, arm_dir,
            min_visible_lengthwise=VISIBILITY_FLOOR[0], min_visible_cross_court=VISIBILITY_FLOOR[1],
        )
    phases["w5_record"] = time.perf_counter() - started
    record_path = arm_dir / w5_result["case_record"]

    started = time.perf_counter()
    with clock.stage("net_choice"):
        record = verifier_module.read_json_gz(record_path)
        rows = net_rows(record, context, net, bounded_trial, np)
        chosen, scored = bounded_trial.choose(rows, NET_WEIGHT, NET_OVERRUN_PX)
    phases["net_choice"] = time.perf_counter() - started
    gated_baseline = rows[0]["origin_key"] if rows else None
    if bounded_trial.choose(rows, 0.0, NET_OVERRUN_PX)[0] != gated_baseline:
        raise RuntimeError(f"{case_id}: zero net weight changed the gated baseline")

    started = time.perf_counter()
    polarity = None
    if chosen is not None:
        # The refit replays the chosen court's saved fit first. A chosen parent whose own W5
        # refit failed has nothing to replay, and the refit raises. Keep the view's timings.
        try:
            with clock.stage("stripe_polarity_refit"):
                polarity = refit_selected.refit_selection(
                    case_id, str(record_path), chosen, label, verifier_module, runtime, manifest,
                )
        except (ValueError, AssertionError) as error:
            print(f"{case_id}: stripe-polarity refit failed: {error!r}", file=sys.stderr)
            polarity = {"error": repr(error)}
    phases["stripe_polarity_refit"] = time.perf_counter() - started

    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    summary = {
        "schema": "d17-timing-case/1",
        "case_id": case_id,
        "label": label,
        "direction_budget": args.direction_budget,
        "max_matched_pairs": args.max_matched_pairs,
        "native_size_wh": list(context.native_size),
        "working_size_wh": list(context.size),
        "net_rule": {"weight": NET_WEIGHT, "overrun_px": NET_OVERRUN_PX},
        "startup_s": startup,
        "phases_s": phases,
        "run_wall_s": time.perf_counter() - clock_start,
        "run_cpu_s": time.process_time() - cpu_start,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "stages": clock.report(),
        "populations": {name: population_summary(value) for name, value in populations.items()},
        "population_counts": w5_result["population_counts"],
        "measurement_counts": measurement_counts,
        "selection": {
            "w5_top": next(iter(record["rankings"]["C"]["provisional_rank"]), None),
            "gated_baseline": gated_baseline,
            "bounded": chosen,
            "bounded_corners_native_px": None if chosen is None else candidates[chosen]["corners_px"],
            "net_rows": scored,
            "polarity_refit": polarity,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(summary_path, "wt", encoding="utf-8") as stream:
        json.dump(verifier_module.jsonable(summary), stream, allow_nan=False)
    print(json.dumps({"case_id": case_id, "direction_budget": args.direction_budget, "run_wall_s": summary["run_wall_s"],
                      "phases_s": phases, "bounded": chosen}))


if __name__ == "__main__":
    main()
