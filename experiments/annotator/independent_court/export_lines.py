"""Export cached line fragments from DeepLSD or LINEA checkpoints."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

MAX_DIMENSION = 960
LINEA_INPUT_SIZE = (640, 640)
LINEA_MEAN = (0.538, 0.494, 0.453)
LINEA_STD = (0.257, 0.263, 0.273)
LINEA_THRESHOLD = 0.2
DEEPLSD_LINE_PARAMS = {
    "filtering": "normal",
    "merge": False,
    "grad_thresh": 3,
}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        manifest = json.load(source)
    cases = manifest.get("cases") if isinstance(manifest, dict) else None
    if not isinstance(cases, list):
        raise TypeError(f"{path}: manifest must contain a cases list")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise TypeError(f"{path}: case {index} must be an object")
        case_id, image_name = case.get("id"), case.get("image")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"{path}: case {index} has a duplicate or invalid id")
        if not isinstance(image_name, str) or not image_name or Path(image_name).is_absolute():
            raise ValueError(f"{path}: case {case_id} image must be a relative path")
        seen.add(case_id)
        validated.append(case)
    return validated


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(source: Path) -> str:
    if not source.is_dir():
        raise FileNotFoundError(f"model source directory does not exist: {source}")
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"--source is not a usable git checkout: {source}") from error
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError(f"git rev-parse returned no commit for --source {source}")
    return commit


def _add_source_path(source: Path) -> None:
    source_text = str(source.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)


def _write_bundle(path: Path, metadata: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as target:
        json.dump({**metadata, "cases": cases}, target, allow_nan=False)
    os.replace(temporary, path)


def _segment_array(segments: Any) -> Any:
    import numpy as np

    array = np.asarray(segments, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim == 3 and array.shape[1:] == (2, 2):
        array = array.reshape(-1, 4)
    if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
        raise ValueError(f"line detector returned unexpected shape {array.shape}")
    return array


def _record(
    case: dict[str, Any],
    image_md5: str,
    dimensions: tuple[int, int],
    segments: Any,
    elapsed: float,
    working_dimensions: tuple[int, int] | None = None,
    scores: Any | None = None,
) -> dict[str, Any]:
    import numpy as np

    width, height = dimensions
    array = _segment_array(segments)
    if working_dimensions is not None:
        working_width, working_height = working_dimensions
        array = array.reshape(-1, 2, 2) * np.asarray(
            [width / working_width, height / working_height]
        )
    record: dict[str, Any] = {
        "id": case["id"],
        "dimensions": {"width": width, "height": height},
        "image_file_md5": image_md5,
        "segments_px": array.reshape(-1, 4).tolist(),
        "time_seconds": elapsed,
    }
    if working_dimensions is not None:
        record["working_dimensions"] = {
            "width": working_dimensions[0],
            "height": working_dimensions[1],
        }
    if scores is not None:
        score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(score_array) != len(array) or not np.isfinite(score_array).all():
            raise ValueError("line detector returned incompatible scores")
        record["scores"] = score_array.tolist()
    if "population" in case:
        record["population"] = case["population"]
    return record


def _working_image(image: Any) -> tuple[Any, tuple[int, int]]:
    import cv2

    height, width = image.shape[:2]
    scale = min(1.0, MAX_DIMENSION / max(width, height))
    working_width, working_height = round(width * scale), round(height * scale)
    if min(working_width, working_height) <= 0:
        raise ValueError(f"invalid working dimensions {(working_width, working_height)}")
    if scale < 1.0:
        image = cv2.resize(
            image, (working_width, working_height), interpolation=cv2.INTER_LINEAR
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != "uint8":
        raise TypeError(f"expected uint8 grayscale image, got {gray.dtype}")
    return gray, (working_width, working_height)


def _load_deeplsd(source: Path, weights: Path, device: Any) -> Any:
    import torch

    _add_source_path(source)
    DeepLSD = import_module("deeplsd.models.deeplsd_inference").DeepLSD

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    network = DeepLSD({"detect_lines": False})
    network.load_state_dict(checkpoint["model"], strict=True)
    return network.to(device).eval()


def _deeplsd_fields(network: Any, gray: Any, working_dimensions: tuple[int, int], device: Any) -> tuple[Any, Any]:
    import numpy as np
    import torch

    input_tensor = torch.from_numpy(gray).to(device=device, dtype=torch.float32) / 255.0
    with torch.inference_mode():
        prediction = network({"image": input_tensor[None, None]})
    expected_shape = (working_dimensions[1], working_dimensions[0])
    fields = []
    for name in ("df", "line_level"):
        field = prediction[name]
        if field.ndim != 3 or field.shape[0] != 1 or tuple(field.shape[1:]) != expected_shape:
            raise ValueError(f"{name} has unexpected shape {tuple(field.shape)}")
        array = field[0].detach().cpu().numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        fields.append(array)
    return fields[0], fields[1]


def _deeplsd_metadata(name: str, weights: Path, model_sha256: str, source_commit: str, grad_nfa: bool) -> dict[str, Any]:
    return {
        "config": {
            "detect_lines": False,
            "line_detection_params": {**DEEPLSD_LINE_PARAMS, "grad_nfa": grad_nfa},
        },
        "model_basename": weights.name,
        "model_sha256": model_sha256,
        "source_commit": source_commit,
        "max_dimension": MAX_DIMENSION,
        "variant": name,
        "timing": "Model fields plus this variant's line extraction; excludes decode and serialisation.",
    }


def _run_deeplsd(
    source: Path,
    weights: Path,
    cases: list[dict[str, Any]],
    image_root: Path,
    output_root: Path,
    device_text: str,
    source_commit: str,
    model: str,
) -> None:
    import cv2
    import torch

    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu explicitly")
    variants = (
        (("deeplsd_md_hard", False), ("deeplsd_md_default", True))
        if model == "deeplsd-md"
        else (("deeplsd_wireframe_hard", False),)
    )
    model_sha256 = _digest(weights, "sha256")
    metadata = {
        name: _deeplsd_metadata(name, weights, model_sha256, source_commit, grad_nfa)
        for name, grad_nfa in variants
    }
    records = {name: [] for name, _ in variants}
    for name, _ in variants:
        _write_bundle(output_root / f"{name}.json.gz", metadata[name], records[name])
    network = _load_deeplsd(source, weights, device)
    for case in cases:
        image_path = image_root / case["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise OSError(f"{case['id']}: could not read image {case['image']!r}")
        image_md5 = _digest(image_path, "md5")
        gray, working_dimensions = _working_image(image)
        started = time.perf_counter()
        df, angle = _deeplsd_fields(network, gray, working_dimensions, device)
        fields_seconds = time.perf_counter() - started
        for name, grad_nfa in variants:
            started = time.perf_counter()
            lines = network.detect_afm_lines(
                gray,
                df,
                angle,
                filtering="normal",
                merge=False,
                grad_thresh=3,
                grad_nfa=grad_nfa,
            )
            elapsed = fields_seconds + time.perf_counter() - started
            records[name].append(
                _record(
                    case,
                    image_md5,
                    (image.shape[1], image.shape[0]),
                    lines,
                    elapsed,
                    working_dimensions,
                )
            )
            _write_bundle(output_root / f"{name}.json.gz", metadata[name], records[name])
            print(f"{name} {case['id']} segments={len(records[name][-1]['segments_px'])}", flush=True)


def _load_linea(source: Path, weights: Path, device: Any) -> tuple[Any, Any]:
    import torch

    _add_source_path(source)
    import_module("models.linea")
    MODULE_BUILD_FUNCS = import_module("models.registry").MODULE_BUILD_FUNCS
    SLConfig = import_module("util.slconfig").SLConfig

    config = SLConfig.fromfile(str(source / "configs/linea/linea_hgnetv2_l.py"))
    config.pretrained = False
    config.multiscale = None
    build_func = MODULE_BUILD_FUNCS.get(config.modelname)
    if build_func is None:
        raise KeyError(f"LINEA builder is not registered: {config.modelname}")
    model, postprocessor = build_func(config)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    return model.deploy().to(device).eval(), postprocessor.deploy().to(device).eval()


def _linea_input(image: Any, device: Any) -> tuple[Any, Any]:
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(rgb)
    transform = transforms.Compose(
        [
            transforms.Resize(LINEA_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=LINEA_MEAN, std=LINEA_STD),
        ]
    )
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    native_size = torch.tensor([[width, height]], device=device)
    return input_tensor, native_size


def _linea_predictions(lines: Any, scores: Any) -> tuple[Any, Any]:
    import numpy as np
    import torch

    if isinstance(lines, torch.Tensor):
        lines = lines.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    line_array = np.asarray(lines, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if line_array.ndim == 3 and line_array.shape[0] == 1:
        line_array = line_array[0]
    if score_array.ndim == 2 and score_array.shape[0] == 1:
        score_array = score_array[0]
    if line_array.ndim != 2 or line_array.shape[1] != 4:
        raise ValueError(f"LINEA returned unexpected line shape {line_array.shape}")
    if score_array.ndim != 1 or score_array.shape[0] != line_array.shape[0]:
        raise ValueError(f"LINEA returned incompatible scores shape {score_array.shape}")
    if not np.isfinite(line_array).all() or not np.isfinite(score_array).all():
        raise ValueError("LINEA returned non-finite lines or scores")
    selected = score_array > LINEA_THRESHOLD
    return line_array[selected], score_array[selected]


def _linea_metadata(weights: Path, model_sha256: str, source_commit: str) -> dict[str, Any]:
    return {
        "variant": "linea_hgnetv2_l",
        "config": "linea_hgnetv2_l.py",
        "model_basename": weights.name,
        "model_sha256": model_sha256,
        "source_commit": source_commit,
        "threshold": LINEA_THRESHOLD,
        "input_size": {"width": LINEA_INPUT_SIZE[0], "height": LINEA_INPUT_SIZE[1]},
        "preprocessing": {
            "colour": "RGB",
            "resize": "stretch",
            "mean": list(LINEA_MEAN),
            "std": list(LINEA_STD),
        },
        "timing": "LINEA forward pass plus postprocessing; excludes decode and serialisation.",
    }


def _run_linea(
    source: Path,
    weights: Path,
    cases: list[dict[str, Any]],
    image_root: Path,
    output_root: Path,
    device_text: str,
    source_commit: str,
) -> None:
    import cv2
    import torch

    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu explicitly")
    model_sha256 = _digest(weights, "sha256")
    metadata = _linea_metadata(weights, model_sha256, source_commit)
    output_path = output_root / "linea_hgnetv2_l.json.gz"
    records: list[dict[str, Any]] = []
    _write_bundle(output_path, metadata, records)
    network, postprocessor = _load_linea(source, weights, device)
    for case in cases:
        image_path = image_root / case["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise OSError(f"{case['id']}: could not read image {case['image']!r}")
        image_md5 = _digest(image_path, "md5")
        input_tensor, native_size = _linea_input(image, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            raw_output = network(input_tensor)
            lines, scores = postprocessor(raw_output, native_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        selected_lines, selected_scores = _linea_predictions(lines, scores)
        records.append(
            _record(
                case,
                image_md5,
                (image.shape[1], image.shape[0]),
                selected_lines,
                elapsed,
                scores=selected_scores,
            )
        )
        _write_bundle(output_path, metadata, records)
        print(f"linea_hgnetv2_l {case['id']} segments={len(records[-1]['segments_px'])}", flush=True)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("deeplsd-md", "deeplsd-wireframe", "linea-large"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=_positive)
    arguments = parser.parse_args(argv)
    if not arguments.weights.is_file():
        raise FileNotFoundError(f"--weights must name a file: {arguments.weights}")
    source = arguments.source.resolve()
    source_commit = _source_commit(source)
    cases = _load_cases(arguments.manifest)
    if arguments.limit is not None:
        cases = cases[: arguments.limit]
    image_root = (arguments.image_root or arguments.manifest.parent).resolve()
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.model == "linea-large":
        _run_linea(
            source,
            arguments.weights,
            cases,
            image_root,
            arguments.output,
            arguments.device,
            source_commit,
        )
    else:
        _run_deeplsd(
            source,
            arguments.weights,
            cases,
            image_root,
            arguments.output,
            arguments.device,
            source_commit,
            arguments.model,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
