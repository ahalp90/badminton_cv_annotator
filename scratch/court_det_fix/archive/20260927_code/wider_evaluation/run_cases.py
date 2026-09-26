"""Measure frozen views with SVD-screened directions and the existing W5 scoring."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from measurement import prepared_measurements

LOGGER = logging.getLogger(__name__)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def load_runtime(root: Path, control_pack: Path | None = None) -> tuple[Any, Any, dict]:
    sys.path[:0] = [str(root.parents[1]), str(root.parents[1] / "src"), str(root / "w5_holistic")]
    import run_w5

    runtime = run_w5.load_runtime(root)
    import verifier

    from experiments.annotator.independent_court.case_provenance import (
        CaseProvenance,
        ImageKind,
    )

    for group, relative in verifier.CASE_PACKS.items():
        for source in verifier.read_json_gz(root / relative)["cases"]:
            verifier.PACK_OF[source["id"]] = group
            verifier.CASE_LABELS[source["id"]] = source["id"]
    if control_pack is not None:
        controls = {source["id"]: source for source in verifier.read_json_gz(control_pack)["cases"]}
        original_source = verifier.load_source
        original_provenance = verifier.load_case_provenance
        original_frame_path = verifier.frame_path

        def load_source(input_root: Path, case_id: str) -> dict:
            return controls[case_id] if case_id in controls else original_source(input_root, case_id)

        def load_provenance(input_root: Path, case_id: str) -> CaseProvenance:
            if case_id not in controls:
                return original_provenance(input_root, case_id)
            frame_index = controls[case_id]["frame_index"]
            return CaseProvenance(case_id, ImageKind.SOURCE_FRAME, (frame_index,), frame_index)

        def frame_path(input_root: Path, source: dict, provenance: CaseProvenance) -> Path:
            if source["id"] in controls:
                return input_root / source["image"]
            return original_frame_path(input_root, source, provenance)

        verifier.load_source = load_source
        verifier.load_case_provenance = load_provenance
        verifier.frame_path = frame_path
        verifier.CASE_LABELS.update({case_id: case_id for case_id in controls})
        runtime = run_w5.load_runtime(root)
    return run_w5, verifier, runtime


def run_case(root: Path, output: Path, case_id: str, control_pack: Path | None = None,
             direction_budget: int = 12) -> dict:
    import cv2

    if direction_budget not in (12, 16):
        raise ValueError(f"direction budget must be 12 or 16, got {direction_budget}")
    cv2.setNumThreads(1)
    run_w5, verifier, runtime = load_runtime(root, control_pack)
    from generation import SCREEN_METHOD, ensure_populations, screen_matches

    context = verifier.prepare_view(root, case_id)
    result_path = output / "results" / f"{case_id}.json.gz"
    if result_path.is_file():
        result = verifier.read_json_gz(result_path)
        record = verifier.read_json_gz(output / result["case_record"])
        if record["case_id"] != case_id or record["min_visible_lengthwise"] != 4 or record["min_visible_cross_court"] != 3:
            raise ValueError(f"{case_id}: checkpoint identity/settings mismatch")
        if not screen_matches(result, direction_budget):
            raise ValueError(f"{case_id}: checkpoint direction screen differs from requested budget/method")
        if not (output / result["array_file"]).is_file():
            raise FileNotFoundError(output / result["array_file"])
        return {"case_id": case_id, "status": "reused", "result": str(result_path),
                "direction_screen": result.get("direction_screen", {"budget": 16, "method": "legacy-full"})}
    population_paths = ensure_populations(root, context, runtime, output, direction_budget)
    populations = {name: verifier.read_json_gz(path)["entries"] for name, path in population_paths.items()}
    original_g0, original_g1 = run_w5.load_g0, run_w5.load_g1
    run_w5.load_g0 = lambda *_args: (populations["G0"], str(population_paths["G0"]))
    run_w5.load_g1 = lambda *_args: (populations["G1"], str(population_paths["G1"]))
    try:
        with prepared_measurements(verifier) as counts:
            result = run_w5.process_case(root, case_id, output, min_visible_lengthwise=4, min_visible_cross_court=3)
        result["measurement_optimisation"] = {"junction_diagnostics": "omitted", **counts}
        result["direction_screen"] = {"method": SCREEN_METHOD, "budget": direction_budget}
        write(result_path, verifier.jsonable(result))
    finally:
        run_w5.load_g0, run_w5.load_g1 = original_g0, original_g1
    return {"case_id": case_id, "status": "completed", "result": str(result_path),
            "direction_screen": result["direction_screen"], **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--control-pack", type=Path)
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--direction-budget", type=int, choices=(12, 16), default=12)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    with gzip.open(args.manifest, "rt", encoding="utf-8") as stream:
        manifest = json.load(stream)
    available = {
        row["case_id"] for row in manifest["cases"]
        if row["arm"] == "frozen_detector" and not row["previous_w5_case"]
    }
    if args.control_pack is not None:
        available.update(row["case_id"] for row in manifest["cases"] if row["arm"] == "rejection_review")
    cases = args.cases if args.cases is not None else sorted(available)
    if not cases or len(cases) != len(set(cases)) or not set(cases) <= available:
        raise ValueError("Choose unique added frozen cases from the supplied manifest")
    for row in manifest["cases"]:
        if row["case_id"] in cases:
            image = root / row["image"]
            if hashlib.md5(image.read_bytes()).hexdigest() != row["image_md5"]:
                raise ValueError(f"{row['case_id']}: image differs from frozen manifest")
    records = []
    # A fresh process per case isolates the legacy helper module overrides.
    with ProcessPoolExecutor(max_workers=min(args.workers, len(cases)), max_tasks_per_child=1) as executor:
        futures = {
            executor.submit(run_case, root, output, case_id, args.control_pack, args.direction_budget): case_id
            for case_id in cases
        }
        for future in as_completed(futures):
            case_id = futures[future]
            try:
                record = future.result()
            except Exception as error:
                LOGGER.exception("Case %s failed", case_id)
                record = {"case_id": case_id, "status": "failed", "direction_budget": args.direction_budget,
                          "error": repr(error),
                          "traceback": traceback.format_exc()}
                write(output / "failures" / f"{case_id}.json.gz", record)
            records.append(record)
            print(json.dumps(record), flush=True)
    write(output / "completion.json.gz", {"cases": records, "workers": args.workers,
                                           "direction_budget": args.direction_budget,
                                           "direction_screen_method": "svd-family-residual/1"})
    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
