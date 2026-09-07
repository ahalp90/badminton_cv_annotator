"""Run the frozen CourtKeyNet baseline beside the independent detector proposals."""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any

import cv2

from .evaluate import _load_manifest, _metrics


def _record_model(detection: Any) -> dict[str, Any]:
    return {
        "corners_px": detection.corners_px.tolist(),
        "peak": detection.peak.tolist(),
        "entropy": detection.entropy.tolist(),
        "flags": list(detection.flags),
        "valid": bool(detection.passed),
    }


def _record_hybrid(quad: Any) -> dict[str, Any] | None:
    if quad is None:
        return None
    return {"corners_px": quad.corners_px.tolist(), "source": quad.source}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)

    cases = _load_manifest(arguments.manifest)
    # These imports bring in torch and model weights only when the CLI is run.
    import torch

    from courtkeynet.court_corners import pick_scene_corners
    from courtkeynet.wrapper import CourtKeyNetDetector

    if arguments.device == "cpu":
        torch.set_num_threads(2)
    model_detector = CourtKeyNetDetector(device=arguments.device)
    records: list[dict[str, Any]] = []
    for case in cases:
        image = cv2.imread(
            str(arguments.manifest.parent / case["image"]), cv2.IMREAD_COLOR
        )
        if image is None or image.size == 0:
            raise OSError(f"{case['id']}: could not read image {case['image']!r}")
        height, width = image.shape[:2]
        started = time.perf_counter()
        detections = model_detector.detect_batch([image])
        raw = detections[0]
        hybrid = pick_scene_corners(
            [image],
            detections,
            corner_min_peak_conf=model_detector.corner_min_peak_conf,
        )
        elapsed = time.perf_counter() - started
        matching = case["reference_status"] == "matching_view"
        raw_record = _record_model(raw)
        raw_record["metrics"] = (
            _metrics(raw.corners_px, case, width, height) if matching else None
        )
        hybrid_record = _record_hybrid(hybrid)
        if hybrid_record is not None:
            hybrid_record["metrics"] = (
                _metrics(hybrid.corners_px, case, width, height) if matching else None
            )
        records.append(
            {
                "id": case["id"],
                "dimensions": {"width": width, "height": height},
                "time_seconds": elapsed,
                "model": raw_record,
                "hybrid": hybrid_record,
            }
        )
        print(
            f"{case['id']} model_valid={int(raw.passed)} hybrid={None if hybrid is None else hybrid.source}",
            flush=True,
        )
    summary = {
        "total": len(records),
        "model_valid": sum(record["model"]["valid"] for record in records),
        "hybrid_available": sum(record["hybrid"] is not None for record in records),
    }
    result = {
        "manifest": arguments.manifest.name,
        "settings": {
            "device": arguments.device,
            "weights": "default",
            "resize_mode": "pad",
            "corner_min_peak_conf": float(model_detector.corner_min_peak_conf),
        },
        "cases": records,
        "summary": summary,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(arguments.output, "wt", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(
        f"summary total={summary['total']} model_valid={summary['model_valid']} hybrid_available={summary['hybrid_available']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
