"""Evaluate a saved checkpoint using a prepared cache, without further fitting."""
from __future__ import annotations

import argparse
import lzma
from pathlib import Path

import numpy as np
import torch

from .data import ClipDataset, write_json
from .evaluate import score_probabilities
from .run_state import Progress
from .train import (
    Settings,
    make_loader,
    model_from_settings,
    normalise_settings_record,
    validate,
)


def evaluate(
    model: torch.nn.Module, dataset: ClipDataset, settings: Settings, device: torch.device,
    stride: int, offset: int, progress: Progress | None = None,
) -> tuple[dict, np.ndarray]:
    loader = make_loader(dataset, settings, training=False, stride=stride, offset=offset)
    if progress is not None:
        progress.write("inference", stride=stride, offset=offset, total=len(loader), completed=0)
    result = validate(model, loader, device, settings.precision, progress, model_name=settings.model_name)
    return result, result.pop("probabilities")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--role", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3, help="Centre-frame stride; defaults to 3 for both models")
    parser.add_argument("--offset", type=int, default=0, help="Absolute video-frame offset within the stride")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings = Settings(**normalise_settings_record(checkpoint["settings"]))
    device = torch.device("cuda")
    model = model_from_settings(settings, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = ClipDataset(args.cache, args.role, checkpoint["scheme"], training=False)
    args.output.mkdir(parents=True, exist_ok=False)
    progress = Progress(args.output)
    progress.write("inference", role=args.role, checkpoint=str(args.checkpoint))
    result, probabilities = evaluate(model, dataset, settings, device, args.stride, args.offset, progress)
    progress.write("scoring")
    write_json(args.output / "frame_metrics.json.gz", result)
    events = score_probabilities(dataset, probabilities, break_alternation_ties=settings.break_alternation_ties)
    events["sampling"] = result["sampling"]
    write_json(args.output / "events.json.gz", events)
    with lzma.open(args.output / "probabilities.npy.xz", "wb", preset=9) as stream:
        np.save(stream, probabilities)
    progress.write("complete")


if __name__ == "__main__":
    main()
