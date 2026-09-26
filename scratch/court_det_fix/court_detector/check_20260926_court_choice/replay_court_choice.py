"""Replay the court detector's final choice for each court-choice arm, from the upright check's saved results.

Both switches act only after W5 scoring, so the saved W5 record and feet, with the view's frame, line
fragments and person boxes, give what a full run would. For each view this rebuilds the scoring context,
then calls the detector's own choose_court once per arm, with self-checks on. The baseline arm must
reproduce the saved final court: the same chosen court and outcome, and corners within 0.0001 native px.

Some W5 fits are rank deficient: they have no unique answer, so this machine's libraries can land far from
where Carmack's did, and the refit's replay check fails. Such a view and arm is logged with its error, and
the replay carries on.

The arms ran at commit 8e28ec7d. Later commits made the blend the default and removed the refit switch, so
run this at that commit.

Usage, from the repository root:
  python -m scratch.court_det_fix.court_detector.check_20260926_court_choice.replay_court_choice UPRIGHT_RUN OUT
    UPRIGHT_RUN: the upright check's run folder, with results/ and artefacts/
    OUT: gets one folder per arm, with results/ and artefacts/ as run_views.py writes them
"""

import os
import sys
from pathlib import Path

# Before numpy loads: one thread per process, as run_views.py does.
for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[thread_variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import json
from time import perf_counter

import cv2
import numpy as np

from scratch.court_det_fix.court_detector.detect import (
    ROOT,
    Laps,
    Switches,
    choose_court,
    freeze_arrays,
    json_round_trip,
    load_live_modules,
    source_record,
)
from scratch.court_det_fix.court_detector.inputs import ViewInputs
from scratch.court_det_fix.court_detector.run_views import (
    VIEWS,
    first_difference,
    pack_sources,
    read_json_gz,
    without_keys,
)

ARMS = {
    "baseline": Switches(),
    "blend": Switches(geometry_weight=0.1),
    "refit": Switches(refit_top=15),
    "both": Switches(geometry_weight=0.1, refit_top=15),
}
TOLERANCE_NATIVE_PX = 1e-4


def reproduction_problem(result_row: dict, saved: dict) -> str | None:
    """Why the baseline arm's result differs from the saved one, or None."""
    for field in ("chosen_key", "no_court_reason"):
        if result_row[field] != saved[field]:
            return f"{field}: {result_row[field]} against saved {saved[field]}"
    if (result_row["corners_native_px"] is None) != (saved["corners_native_px"] is None):
        return "one of the two has no corners"
    if result_row["corners_native_px"] is None:
        return None
    gap = np.abs(np.asarray(result_row["corners_native_px"]) - np.asarray(saved["corners_native_px"])).max()
    return None if gap <= TOLERANCE_NATIVE_PX else f"corners differ by up to {gap:.2e} native px"


def main() -> None:
    upright_run, out = Path(sys.argv[1]), Path(sys.argv[2])
    live = load_live_modules()
    verifier = live.verifier
    sources, provenances, frame_paths = pack_sources(verifier.CASE_PACKS)
    anchors = {view["case_id"]: view["anchor"] for view in json.loads(VIEWS.read_text())}
    for arm in ARMS:
        (out / arm / "results").mkdir(parents=True, exist_ok=True)
        (out / arm / "artefacts").mkdir(exist_ok=True)

    problems = []
    saved_paths = sorted((upright_run / "results").glob("*.json"))
    for saved_path in saved_paths:
        saved = json.loads(saved_path.read_text())
        view_id = saved["view_id"]
        saved_artefacts = read_json_gz(upright_run / "artefacts" / f"{view_id}.json.gz")
        source = sources[view_id]
        frame_path = frame_paths.get(view_id) or verifier.frame_path(ROOT, source, provenances[view_id])
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(frame_path)
        view = ViewInputs(
            view_id=view_id, frame=frame, frame_index=anchors[view_id],
            # Only the feet read the scene's frames, and they come from the saved results.
            scene_frames=(anchors[view_id], anchors[view_id]),
            segments_px=np.asarray(source["segments_px"], dtype=float).reshape(-1, 4),
            person_boxes_px=np.asarray(source["bbox_px"], dtype=float).reshape(-1, 4),
            provenance=provenances[view_id],
        )
        native_frame = view.frame.view()
        native_frame.flags.writeable = False
        detector_source = source_record(view, saved_artefacts["feet"]["all_feet_px"])
        context = verifier.view_context(view_id, detector_source, view.provenance, native_frame, view_id)
        freeze_arrays(context)
        line_maps = live.run_w5.view_line_maps(context)
        record = saved_artefacts["w5"]["record"]

        with live.prepared_measurements(verifier):
            for arm, switches in ARMS.items():
                artefacts: dict = {}
                started = perf_counter()
                try:
                    result = choose_court(view_id, record, context, native_frame, line_maps, live, switches,
                                          Laps(), artefacts)
                except AssertionError as error:
                    message = " ".join(str(error).split())[:200]
                    print(f"{view_id} {arm}: {message}", flush=True)
                    problems.append(f"{view_id} {arm}: {message}")
                    row = {"view_id": view_id, "error": message}
                    (out / arm / "results" / f"{view_id}.json").write_text(json.dumps(row, indent=1))
                    continue
                row = {"view_id": view_id, "error": None, "choose_seconds": perf_counter() - started,
                       "chosen_key": result.chosen_key, "no_court_reason": result.no_court_reason,
                       "corners_native_px": None if result.corners_native_px is None
                       else result.corners_native_px.tolist()}
                if arm == "baseline":
                    problem = reproduction_problem(row, saved)
                    if problem is not None:
                        problems.append(f"{view_id} baseline: {problem}")
                    # Whole-record equality is reported, not required: other library builds can differ
                    # in the last bits.
                    # A view with no eligible court has no refit record.
                    written = json_round_trip(verifier.jsonable(artefacts))
                    row["stripe_refit_difference"] = first_difference(
                        without_keys(written.get("stripe_refit"), {"timings_seconds"}),
                        without_keys(saved_artefacts.get("stripe_refit"), {"timings_seconds"}))
                    row["net_choice_difference"] = first_difference(written["net_choice"],
                                                                    saved_artefacts["net_choice"])
                (out / arm / "results" / f"{view_id}.json").write_text(json.dumps(row, indent=1))
                verifier.write_json_gz(out / arm / "artefacts" / f"{view_id}.json.gz", artefacts)
                print(json.dumps({key: row.get(key) for key in ("view_id", "chosen_key", "no_court_reason",
                                                                 "stripe_refit_difference")}) + f" {arm}",
                      flush=True)

    print(f"\n{len(saved_paths)} views; problems: " + ("none" if not problems else "\n".join(problems)))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
