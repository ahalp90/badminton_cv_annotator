"""Evaluate a saved checkpoint using a prepared cache, without further fitting."""
from __future__ import annotations

import argparse
import lzma
from copy import copy
from pathlib import Path

import numpy as np
import torch

from .data import ClipDataset, write_json
from .evaluate import score_probabilities
from .run_state import Progress
from .train import Settings, make_loader, model_from_settings, validate


def sampled_dataset(dataset: ClipDataset, stride: int, offset: int) -> tuple[ClipDataset, np.ndarray]:
    """Select absolute video-frame centres while retaining every clip's 16-frame context."""
    if stride < 1 or not 0 <= offset < stride:
        raise ValueError("Stride must be positive and offset must be between zero and stride minus one")
    indices = np.flatnonzero(dataset.rows[:, 2] % stride == offset)
    sampled = copy(dataset)
    sampled.rows = dataset.rows[indices]
    sampled.labels = dataset.labels[indices]
    return sampled, indices


def inference_stride(model_name: str, role: str, requested: int | None) -> int:
    """VideoMAE tests use every third centre; validation and X3D retain dense defaults."""
    return requested if requested is not None else (3 if model_name == "videomae" and role == "test" else 1)


def evaluate(
    model: torch.nn.Module, dataset: ClipDataset, settings: Settings, device: torch.device,
    stride: int, offset: int, progress: Progress | None = None,
) -> tuple[dict, np.ndarray]:
    sampled, indices = sampled_dataset(dataset, stride, offset)
    loader = make_loader(sampled, settings, training=False)
    if progress is not None:
        progress.write("inference", stride=stride, offset=offset, total=len(loader), completed=0)
    result = validate(model, loader, device, settings.precision, progress, model_name=settings.model_name)
    sampled_probabilities = result.pop("probabilities")
    # The shared event decoder expects a row per centre; skipped centres cannot emit contacts.
    probabilities = np.zeros((len(dataset), 3), dtype=sampled_probabilities.dtype)
    probabilities[:, 0] = 1
    probabilities[indices] = sampled_probabilities
    result["sampling"] = {"stride": stride, "offset": offset, "total_centres": len(dataset),
                          "evaluated_centres": len(sampled), "unsampled_probabilities": "background"}
    return result, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--role", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, help="Centre-frame stride; defaults to 3 for VideoMAE test, otherwise 1")
    parser.add_argument("--offset", type=int, default=0, help="Absolute video-frame offset within the stride")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings = Settings(**checkpoint["settings"])
    device = torch.device("cuda")
    model = model_from_settings(settings, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = ClipDataset(args.cache, args.role, checkpoint["scheme"], training=False)
    stride = inference_stride(settings.model_name, args.role, args.stride)
    args.output.mkdir(parents=True, exist_ok=False)
    progress = Progress(args.output)
    progress.write("inference", role=args.role, checkpoint=str(args.checkpoint))
    result, probabilities = evaluate(model, dataset, settings, device, stride, args.offset, progress)
    progress.write("scoring")
    write_json(args.output / "frame_metrics.json.gz", result)
    events = score_probabilities(dataset, probabilities)
    events["sampling"] = result["sampling"]
    write_json(args.output / "events.json.gz", events)
    with lzma.open(args.output / "probabilities.npy.xz", "wb", preset=9) as stream:
        np.save(stream, probabilities)
    progress.write("complete")


if __name__ == "__main__":
    main()
