"""Evaluate the independent line-based court detector on a gzip JSON manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from . import detector

REFERENCE_SIZE = (1280, 720)
SAFE_COORDINATE = 1_000_000
REFERENCE_STATUSES = frozenset(
    ("matching_view", "view_unverified", "unlabelled", "non_court")
)


def _array(value: Any, shape: tuple[int, ...], where: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where} must contain numeric points") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{where} must contain finite points with shape {shape}")
    return result


def _validate_manifest(manifest: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise TypeError(f"{path}: manifest 'cases' must be a list")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        where = f"{path}: case {index}"
        if not isinstance(case, dict):
            raise TypeError(f"{where} must be an object")
        case_id, image_name, status = (
            case.get("id"),
            case.get("image"),
            case.get("reference_status"),
        )
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"{where} id must be non-empty and unique")
        seen.add(case_id)
        if (
            not isinstance(image_name, str)
            or not image_name
            or Path(image_name).is_absolute()
        ):
            raise ValueError(f"{where} image must be a non-empty relative path")
        if status not in REFERENCE_STATUSES:
            raise ValueError(
                f"{where} reference_status must be one of {sorted(REFERENCE_STATUSES)}"
            )
        if case.get("corners_px") is not None:
            case["corners_px"] = _array(
                case["corners_px"], (4, 2), f"{where} corners_px"
            )
        landmarks = case.get("landmarks")
        if landmarks is not None and not isinstance(landmarks, list):
            raise TypeError(f"{where} landmarks must be a list")
        for landmark_index, landmark in enumerate(landmarks or []):
            landmark_where = f"{where} landmark {landmark_index}"
            if not isinstance(landmark, dict):
                raise TypeError(f"{landmark_where} must be an object")
            if "court_m" not in landmark or "image_px" not in landmark:
                raise ValueError(f"{landmark_where} must contain court_m and image_px")
            landmark["court_m"] = _array(
                landmark["court_m"], (2,), f"{landmark_where} court_m"
            )
            landmark["image_px"] = _array(
                landmark["image_px"], (2,), f"{landmark_where} image_px"
            )
        baseline = case.get("baseline")
        if baseline is not None:
            if (
                not isinstance(baseline, dict)
                or not isinstance(baseline.get("accepted"), bool)
                or "corners_px" not in baseline
            ):
                raise ValueError(
                    f"{where} baseline must contain corners_px and boolean accepted"
                )
            baseline["corners_px"] = _array(
                baseline["corners_px"], (4, 2), f"{where} baseline corners_px"
            )
    return cases


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read gzip JSON manifest {path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise TypeError(f"{path}: manifest root must be an object")
    return _validate_manifest(manifest, path)


def _homography(corners: np.ndarray) -> np.ndarray | None:
    result, _ = cv2.findHomography(
        np.asarray(detector.CORNER_COURT_M, dtype=np.float64), corners, 0
    )
    return result if result is not None and np.isfinite(result).all() else None


def _metrics(
    predicted: np.ndarray | None, reference: dict[str, Any], width: int, height: int
) -> dict[str, Any] | None:
    if predicted is None:
        return None
    scale = np.array([REFERENCE_SIZE[0] / width, REFERENCE_SIZE[1] / height])
    result: dict[str, Any] = {
        "corner_mean_error_px": None,
        "corner_max_error_px": None,
        "landmark_rms_px": None,
        "landmark_status": "no_landmarks",
    }
    reference_corners = reference.get("corners_px")
    if reference_corners is not None:
        errors = np.linalg.norm((predicted - reference_corners) * scale, axis=1)
        result["corner_mean_error_px"], result["corner_max_error_px"] = (
            float(errors.mean()),
            float(errors.max()),
        )
    landmarks = reference.get("landmarks")
    if not landmarks:
        return result
    homography = _homography(predicted)
    if homography is None:
        result["landmark_status"] = "invalid_homography"
        return result
    court = np.stack([landmark["court_m"] for landmark in landmarks])
    image = np.stack([landmark["image_px"] for landmark in landmarks])
    visible = (
        (image[:, 0] >= 0)
        & (image[:, 0] < width)
        & (image[:, 1] >= 0)
        & (image[:, 1] < height)
    )
    projected, denominators = detector.project(homography[None], court)
    if not np.isfinite(projected[0, visible]).all() or not np.all(
        denominators[0, visible] > 1e-9
    ):
        result["landmark_status"] = "invalid_projection"
        return result
    if not visible.any():
        result["landmark_status"] = "no_visible_landmarks"
        return result
    errors = (projected[0, visible] - image[visible]) * scale
    result["landmark_rms_px"] = float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))
    result["landmark_status"] = "ok"
    return result


def _safe_pair(points: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if not np.isfinite(points).all():
        return None
    bounded = np.clip(points, -SAFE_COORDINATE, SAFE_COORDINATE)
    return tuple(np.rint(bounded[0]).astype(int)), tuple(
        np.rint(bounded[1]).astype(int)
    )


def _clip(
    points: np.ndarray, width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    safe = _safe_pair(points)
    if safe is None:
        return None
    visible, start, end = cv2.clipLine((0, 0, width, height), *safe)
    return (start, end) if visible else None


def _dash(image: np.ndarray, points: np.ndarray, colour: tuple[int, int, int]) -> None:
    clipped = _clip(points, image.shape[1], image.shape[0])
    if clipped is None:
        return
    start, end = np.asarray(clipped, dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    for offset in np.arange(0, length, 16):
        a = start + (end - start) * offset / length
        b = start + (end - start) * min(offset + 8, length) / length
        cv2.line(
            image,
            tuple(np.rint(a).astype(int)),
            tuple(np.rint(b).astype(int)),
            colour,
            2,
            cv2.LINE_AA,
        )


def _dot(image: np.ndarray, point: np.ndarray, colour: tuple[int, int, int]) -> None:
    safe = _safe_pair(np.asarray([point, point], dtype=np.float64))
    if safe is not None:
        cv2.circle(image, safe[0], 4, colour, -1, cv2.LINE_AA)


def _overlay(
    image: np.ndarray,
    predicted: np.ndarray | None,
    reference: dict[str, Any] | None,
    path: Path,
) -> None:
    output = image.copy()
    homography = _homography(predicted) if predicted is not None else None
    if homography is not None:
        projected, denominators = detector.project(
            homography[None], detector.SEGMENTS_M
        )
        for segment, denominator in zip(
            projected[0].reshape(-1, 2, 2), denominators[0].reshape(-1, 2)
        ):
            if np.isfinite(denominator).all() and np.all(denominator > 1e-9):
                clipped = _clip(segment, output.shape[1], output.shape[0])
                if clipped is not None:
                    cv2.line(output, *clipped, (255, 255, 0), 2, cv2.LINE_AA)
    if reference is not None:
        corners = reference.get("corners_px")
        if corners is not None:
            for start, end in zip(corners, np.roll(corners, -1, axis=0)):
                _dash(output, np.stack([start, end]), (0, 128, 255))
            for point in corners:
                _dot(output, point, (0, 128, 255))
        for landmark in reference.get("landmarks") or []:
            _dot(output, landmark["image_px"], (0, 128, 255))
    if not cv2.imwrite(str(path), output):
        raise OSError(f"could not write overlay {path}")


def _positive(value: str) -> int:
    if (parsed := int(value)) <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _load_line_cache(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        cache = json.load(source)
    if not isinstance(cache, dict) or not isinstance(cache.get("cases"), list):
        raise TypeError("line cache must contain a cases list")
    if not isinstance(cache.get("variant"), str) or not cache["variant"]:
        raise ValueError("line cache must name its extractor variant")
    records: dict[str, dict[str, Any]] = {}
    for record in cache["cases"]:
        case_id = record["id"]
        if not isinstance(case_id, str) or not case_id or case_id in records:
            raise ValueError("line cache case IDs must be non-empty and unique")
        records[case_id] = record
    metadata = {key: value for key, value in cache.items() if key != "cases"}
    metadata["file"] = path.name
    metadata["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return metadata, records


def _cached_segments(record: dict[str, Any], image_path: Path, image: np.ndarray) -> np.ndarray:
    if hashlib.md5(image_path.read_bytes()).hexdigest() != record["image_file_md5"]:
        raise ValueError(f"{record['id']}: cached lines belong to a different image")
    dimensions = record["dimensions"]
    if (image.shape[1], image.shape[0]) != (dimensions["width"], dimensions["height"]):
        raise ValueError(f"{record['id']}: cached native dimensions do not match the image")
    segments = np.asarray(record["segments_px"], dtype=np.float64)
    return segments.reshape(0, 4) if segments.shape == (0,) else segments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extractor", choices=("hough", "ridge", "lsd"), default="hough")
    parser.add_argument("--line-cache", type=Path, help="Use cached native XYXY lines matched by image ID and hash")
    parser.add_argument("--wide-families", action="store_true", help="Probe overlapping direction groups for oblique views")
    parser.add_argument("--min-supported-lines", type=_positive, default=detector.DEFAULT_SETTINGS.min_supported_lines)
    parser.add_argument(
        "--max-family-lines",
        type=_positive,
        default=detector.DEFAULT_SETTINGS.max_family_lines,
    )
    parser.add_argument(
        "--merge-distance", type=float, default=detector.DEFAULT_SETTINGS.merge_distance
    )
    parser.add_argument("--limit", type=_positive)
    arguments = parser.parse_args(argv)
    cases = _load_manifest(arguments.manifest)
    if arguments.limit is not None:
        cases = cases[: arguments.limit]
    line_metadata, line_records = (
        _load_line_cache(arguments.line_cache) if arguments.line_cache is not None else (None, {})
    )
    if line_metadata is not None:
        missing = [case["id"] for case in cases if case["id"] not in line_records]
        if missing:
            raise ValueError(f"line cache is missing case IDs: {missing}")
    settings = detector.Settings(
        extractor=arguments.extractor,
        max_family_lines=arguments.max_family_lines,
        merge_distance=arguments.merge_distance,
        wide_families=arguments.wide_families,
        min_supported_lines=arguments.min_supported_lines,
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    overlay_dir = arguments.output / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    for case in cases:
        image_path = arguments.manifest.parent / case["image"]
        image = cv2.imread(
            str(image_path), cv2.IMREAD_COLOR
        )
        if image is None or image.size == 0:
            raise OSError(f"{case['id']}: could not read image {case['image']!r}")
        height, width = image.shape[:2]
        segments = None if line_metadata is None else _cached_segments(line_records[case["id"]], image_path, image)
        started = time.perf_counter()
        detection = (
            detector.detect(image, settings) if segments is None
            else detector.detect(image, settings, segments_px=segments)
        )
        elapsed = time.perf_counter() - started
        candidates = []
        for candidate in detection.candidates:
            candidates.append(
                {
                    "corners_px": candidate.corners_px.tolist(),
                    "score": float(candidate.score),
                    "family_support": [
                        float(value) for value in candidate.family_support
                    ],
                    "supported_lines": [
                        int(value) for value in candidate.supported_lines
                    ],
                }
            )
        top_corners = (
            detection.candidates[0].corners_px if detection.candidates else None
        )
        matching = case["reference_status"] == "matching_view"
        baseline = case.get("baseline")
        baseline_record = (
            None
            if baseline is None
            else {
                "accepted": baseline["accepted"],
                "corners_px": baseline["corners_px"].tolist(),
                "metrics": _metrics(baseline["corners_px"], case, width, height)
                if matching
                else None,
            }
        )
        overlay_name = f"{quote(case['id'], safe='._-') or 'case'}.png"
        _overlay(
            image, top_corners, case if matching else None, overlay_dir / overlay_name
        )
        records.append(
            {
                "id": case["id"],
                "dimensions": {"width": width, "height": height},
                "time_seconds": elapsed,
                "reference_status": case["reference_status"],
                "reason": detection.reason,
                "accepted": bool(detection.accepted),
                "gap": None
                if detection.score_gap is None
                else float(detection.score_gap),
                "family_line_counts": [
                    int(value) for value in detection.family_line_counts
                ],
                "hypotheses": int(detection.hypotheses_scored),
                "candidates": candidates,
                "raw_metrics": _metrics(top_corners, case, width, height)
                if matching
                else None,
                "baseline": baseline_record,
                "overlay": str(Path("overlays") / overlay_name),
            }
        )
        print(
            f"{case['id']} accepted={int(detection.accepted)} candidates={len(candidates)} reason={detection.reason}",
            flush=True,
        )
    summary = {
        "total": len(records),
        "accepted": sum(record["accepted"] for record in records),
        "rejected": sum(not record["accepted"] for record in records),
    }
    result = {
        "manifest": arguments.manifest.name,
        "settings": asdict(settings),
        "cases": records,
        "summary": summary,
    }
    if line_metadata is not None:
        result["settings"]["extractor"] = line_metadata["variant"]
        result["line_cache"] = line_metadata
    with gzip.open(
        arguments.output / "results.json.gz", "wt", encoding="utf-8"
    ) as target:
        json.dump(result, target, indent=2, allow_nan=False)
    print(
        f"summary total={summary['total']} accepted={summary['accepted']} rejected={summary['rejected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
