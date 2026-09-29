"""Compare fixed label corrections on the earliest frozen frame of each amateur source."""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from scratch.court_det_fix.edge_polarity import run_probe as probe
from scratch.court_det_fix.edge_polarity.local_audit import diagnose

OUTPUT = Path(__file__).resolve().parent


def sample_cases() -> list[dict]:
    selected = {}
    for row in probe.read(probe.BASE / "manifest.json.gz")["cases"]:
        if row["group"] != "gx" and "amateur" not in row["image"]:
            continue
        frame = int(row["case_id"].rsplit("_", 1)[1])
        video = row["video"]
        if video not in selected or frame < selected[video]["frame"]:
            selected[video] = {"case_id": row["case_id"], "video": video, "frame": frame,
                               "image": row["image"], "image_md5": row["image_md5"]}
    return sorted(selected.values(), key=lambda row: row["video"])


def run_case(sample: dict) -> dict:
    cv2.setNumThreads(1)
    case_id = sample["case_id"]
    _, verifier, runtime = probe.load_runtime(probe.ROOT)
    context, parent, starting_corners, constraints = diagnose.load_case(case_id, verifier)
    comparison = next(case for case in probe.read(probe.BASE / "comparison.json.gz")["cases"]
                      if case["case_id"] == case_id)
    record = probe.read(Path(comparison["record"]))
    attempts = [row for row in record["fit_attempts"] if row["origin_key"] == parent["origin_key"]]
    maps = verifier.detector._distance_maps(verifier.detector._wide_line_families(context.segments), context.size)
    with probe.prepared_measurements(verifier) as counts:
        polarity, rows, polarity_metadata = probe.relabel(context, parent, constraints, verifier)
        choices = diagnose.centre_choices(context, parent, rows, verifier)
        positions = polarity.positions.copy()
        for fragment_id, position in choices.items():
            positions[polarity.fragment_ids == fragment_id] = position
        centres = replace(polarity, positions=positions)
        for field in ("points", "intervals", "weights", "fragment_ids", "sample_ids"):
            assert getattr(centres, field) is getattr(constraints, field)
        fits = {}
        for name, current in (("original", constraints), ("polarity", polarity), ("strong_centres", centres)):
            fit = fitting.refine(starting_corners, current, context.size, True,
                                 centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
            fit["measurement"] = None
            if fit["successful"]:
                fit["measurement"] = probe.describe(np.asarray(fit["corners_px"]), context, verifier, runtime, maps)
            fits[name] = fit
        candidates = {entry["origin_key"]: entry for entry in record["parents"] + record["valid_children"]}
        selected = candidates[comparison["selections"]["full"]["gated"]]
        selected_corners = np.asarray(selected["corners_px"]) / (np.asarray(context.native_size) / context.size)
        fits["saved_selection"] = {
            "successful": True, "status": "saved_selection", "corners_px": selected_corners.tolist(),
            "measurement": probe.describe(selected_corners, context, verifier, runtime, maps),
        }
        assert counts["greyscale_conversions"] == 1, counts
    replay_difference = None
    if attempts and fits["original"]["successful"]:
        expected = np.asarray(attempts[0]["attempted_corners_native"])
        actual = np.asarray(fits["original"]["corners_px"]) * (np.asarray(context.native_size) / context.size)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
        replay_difference = float(np.abs(actual - expected).max())
    result = {**sample, "title": f"{sample['video'].replace('_', ' ').title()} · frame {sample['frame']}",
              "working_size": context.size, "native_size": context.native_size,
              "parent_key": parent["origin_key"], "selected_key": comparison["selections"]["full"]["gated"],
              "record": comparison["record"], "baseline_replay_max_native_px": replay_difference,
              "baseline_replay_status": "matched_saved_attempt" if replay_difference is not None else "no_successful_replay",
              "centre_choices": choices, "fragments": rows, "polarity": polarity_metadata,
              "frozen_arrays_preserved": True, "measurement_counts": counts, "fits": fits}
    destination = OUTPUT / "amateur_cases" / f"{case_id}.json.gz"
    destination.write_bytes(gzip.compress(json.dumps(result, indent=2, allow_nan=False).encode(), mtime=0))
    print(case_id, "centres", len(choices), result["baseline_replay_status"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", help="Rerun named cases, then assemble the complete saved sample.")
    args = parser.parse_args()
    samples = sample_cases()
    (OUTPUT / "amateur_cases").mkdir(exist_ok=True)
    plan = {"selection": "Earliest saved frame per amateur video, including GX; chosen before fitting.", "cases": samples}
    (OUTPUT / "amateur_sample.json.gz").write_bytes(gzip.compress(json.dumps(plan, indent=2).encode(), mtime=0))
    chosen = samples
    if args.cases is not None:
        assert set(args.cases) <= {row["case_id"] for row in samples}
        chosen = [row for row in samples if row["case_id"] in args.cases]
    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(run_case, chosen))
    cases = [probe.read(OUTPUT / "amateur_cases" / f"{sample['case_id']}.json.gz") for sample in samples]
    result = {"schema": "amateur-fixed-centre-label-comparison/1", "selection": plan["selection"], "cases": cases}
    (OUTPUT / "amateur_results.json.gz").write_bytes(gzip.compress(json.dumps(result, indent=2, allow_nan=False).encode(), mtime=0))
    print(f"Completed {len(cases)} amateur sources")


if __name__ == "__main__":
    main()
