"""Evaluate a saved checkpoint using a prepared cache, without further fitting."""
from __future__ import annotations

import argparse
import lzma
from pathlib import Path

import numpy as np
import torch

from .data import ClipDataset, write_json
from .evaluate import score_probabilities
from .model import build_model
from .train import Settings, make_loader, validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--role", choices=("validation",), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    settings = Settings(**checkpoint["settings"])
    device = torch.device("cuda")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = ClipDataset(args.cache, args.role, checkpoint["scheme"], training=False)
    loader = make_loader(dataset, settings, training=False)
    result = validate(model, loader, device, settings.precision)
    probabilities = result.pop("probabilities")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "frame_metrics.json.gz", result)
    write_json(args.output / "events.json.gz", score_probabilities(dataset, probabilities))
    with lzma.open(args.output / "probabilities.npy.xz", "wb", preset=9) as stream:
        np.save(stream, probabilities)


if __name__ == "__main__":
    main()
