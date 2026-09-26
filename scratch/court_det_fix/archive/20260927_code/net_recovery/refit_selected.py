"""Replay fixed W5 fits after a saved court selection, with automatic stripe polarity."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from time import perf_counter

import cv2

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(ROOT / "wider_evaluation")]

from run_cases import load_runtime  # pyrefly: ignore[missing-import]

from scratch.court_det_fix.colour_consistency import am1_recovery_trial
from scratch.court_det_fix.court_detector.stripe_refit import refit_chosen

MANIFEST = ROOT / "wider_evaluation/runs/20260922/manifest.json.gz"
CONTROL_PACK = ROOT / "wider_evaluation/runs/20260922/control_inputs.json.gz"


def checked_context(case_id: str, record: dict, verifier: object, manifest: dict):
    context = verifier.prepare_view(ROOT, case_id)
    provenance = record["provenance"]
    if record["case_id"] != case_id or provenance["case_id"] != case_id:
        raise ValueError(f"{case_id}: record case differs")
    if (provenance["frame_path"] != context.frame_relative_path
            or provenance["native_dimensions"] != list(context.native_size)
            or provenance["working_dimensions"] != list(context.size)):
        raise ValueError(f"{case_id}: source image or dimensions differ")
    matching = [row for row in manifest["cases"] if row["case_id"] == case_id]
    if len(matching) != 1 or matching[0]["image"] != context.frame_relative_path:
        raise ValueError(f"{case_id}: full manifest image differs")
    frame_md5 = hashlib.md5((ROOT / context.frame_relative_path).read_bytes()).hexdigest()
    if frame_md5 != matching[0]["image_md5"]:
        raise ValueError(f"{case_id}: image MD5 differs from full manifest")
    return context, frame_md5


def resolve_source_record(source_record: str) -> Path:
    source = Path(source_record)
    local = source if source.is_absolute() else REPO / source
    if local.is_file():
        return local.resolve()
    return am1_recovery_trial.resolve_saved_path(source_record)


def refit_selection(case_id: str, source_record: str, origin_key: str, label: str,
                    verifier: object, runtime: dict, manifest: dict) -> dict:
    started = perf_counter()
    path = resolve_source_record(source_record)
    record = verifier.read_json_gz(path)
    context, frame_md5 = checked_context(case_id, record, verifier, manifest)
    frame_path = ROOT / context.frame_relative_path
    native_frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if native_frame is None:
        raise FileNotFoundError(frame_path)
    line_maps = verifier.detector._distance_maps(verifier.detector._wide_line_families(context.segments), context.size)
    refit = refit_chosen(record, origin_key, context, native_frame, verifier, runtime, line_maps)
    refit["timings_seconds"]["total"] = perf_counter() - started
    return {"case_id": case_id, "label": label, "source_record": str(path), "frame_md5": frame_md5, **refit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    cv2.setNumThreads(1)
    _, verifier, runtime = load_runtime(ROOT, CONTROL_PACK)
    requests = verifier.read_json_gz(arguments.requests)
    manifest = verifier.read_json_gz(MANIFEST)
    results = []
    for request in requests["selections"]:
        result = refit_selection(**request, verifier=verifier, runtime=runtime, manifest=manifest)
        results.append(result)
        print(f"{len(results)}/{len(requests['selections'])} {result['case_id']}: "
              f"{result['corrected']['status']}", flush=True)
    verifier.write_json_gz(arguments.output, {"schema": "selected-polarity-refit/1", "selections": results})


if __name__ == "__main__":
    main()
