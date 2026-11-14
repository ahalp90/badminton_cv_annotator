"""Train, resume, evaluate and profile E2E-Spot without implicit test evaluation."""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from experiments.annotator.video_models_benchmark.data import write_json

from .config import Settings, load_settings
from .data import DenseClips
from .engine import (
    configure_runtime,
    device_batch,
    fit,
    make_loader,
    optimise_batch,
    save_validation_artifacts,
    seed_everything,
    validate,
)
from .model import SpotModel


def infer(checkpoint: Path, cache: Path, output: Path, role: str, device: str, workers: int) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    settings = Settings(**state["settings"])
    device_object = torch.device(device)
    configure_runtime(settings, device_object)
    model = SpotModel(settings.backbone, settings.clip_length, settings.classes, pretrained=False).to(device_object)
    model.load_state_dict(state["model"])
    dataset = DenseClips(cache, role, settings.mode, settings.clip_length, augmentation=settings.augmentation,
                         ignore_contact_halo=settings.ignore_contact_halo)
    loader = make_loader(dataset, settings.validation_batch_size, workers)
    result, probabilities = validate(model, dataset, loader, device_object, settings)
    result.update(settings=asdict(settings), role=role, checkpoint_cycle=state["cycle"],
                  probability_rows="original zero-based source frame; supplied rows outside target scopes are zero")
    save_validation_artifacts(output, dataset.videos, probabilities, result)
    return {"loss": result["loss"], "metrics": result["metrics"], "seconds": result["seconds"]}


def measure(cache: Path, settings: Settings, *, device: str, workers: int, steps: int, warmup: int,
            full_validation: bool = False, compare_device_only: bool = True) -> dict:
    """Run a fit check, optionally comparing it with a device-resident batch."""
    seed_everything(settings.seed)
    device_object = torch.device(device)
    configure_runtime(settings, device_object)
    model = SpotModel(settings.backbone, settings.clip_length, settings.classes, pretrained=False).to(device_object)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    dataset = DenseClips(cache, "train", settings.mode, settings.clip_length, seed=settings.seed, training=True,
                         cycles=1, frames_per_cycle=(steps + warmup) * settings.batch_size * settings.clip_length,
                         augmentation=settings.augmentation, mixup=settings.mixup,
                         ignore_contact_halo=settings.ignore_contact_halo)
    loader = make_loader(dataset, settings.batch_size, workers)
    weights = torch.tensor([1, 5, 5] + ([1] if settings.classes == 4 else []), device=device_object, dtype=torch.float32)
    iterator = iter(loader)
    measurements = {}
    # Reuse the final device batch for the second measurement; feeder and H2D
    # costs are excluded there, while augmentation/training arithmetic stay equal.
    for mode in (("end_to_end", "device_only") if compare_device_only else ("end_to_end",)):
        elapsed, frames = 0.0, 0
        for step in range(warmup + steps):
            if device_object.type == "cuda":
                torch.cuda.synchronize(device_object)
                if step == warmup:
                    torch.cuda.reset_peak_memory_stats(device_object)
            started = time.perf_counter()
            if mode == "end_to_end":
                clips, labels, valid_frames, positions, paired = device_batch(next(iterator), device_object)
            pair_options = {"paired": paired} if paired is not None else {}
            loss = optimise_batch(model, optimiser, clips, labels, valid_frames, settings, device_object, weights,
                                  **pair_options)
            if device_object.type == "cuda":
                torch.cuda.synchronize(device_object)
            if step >= warmup:
                elapsed += time.perf_counter() - started
                frames += int(positions.sum())
        measurements[mode] = {"steps": steps, "warmup_steps": warmup, "frames": frames, "seconds": elapsed,
                              "frames_per_second": frames / elapsed, "last_loss": float(loss),
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_object)
                              if device_object.type == "cuda" else None}
    result = {**measurements, "pretrained": False, "settings": asdict(settings)}
    if compare_device_only:
        result["device_to_end_to_end_ratio"] = (measurements["device_only"]["frames_per_second"]
                                               / measurements["end_to_end"]["frames_per_second"])
    if full_validation:
        validation_data = DenseClips(cache, "validation", settings.mode, settings.clip_length,
                                     augmentation=settings.augmentation,
                                     ignore_contact_halo=settings.ignore_contact_halo)
        validation_loader = make_loader(validation_data, settings.validation_batch_size, workers)
        validation, _ = validate(model, validation_data, validation_loader, device_object, settings)
        validation.pop("streams")
        result["validation"] = validation
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "smoke", "timing", "infer"):
        command = commands.add_parser(name)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--workers", type=int, default=0)
        if name == "infer":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--role", choices=("validation", "test"), required=True)
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--batch-size", type=int, help="Explicit training batch override; does not scale LR")
            command.add_argument("--validation-batch-size", type=int, help="Independent validation batch override")
        if name == "train":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--max-steps", type=int, help="Stop after this many additional steps; leave recovery checkpoint")
        elif name in ("smoke", "timing"):
            command.add_argument("--full-validation", action="store_true", help="Also time a full deterministic validation pass")
            command.add_argument("--steps", type=int, default=1 if name == "smoke" else 10)
            command.add_argument("--warmup", type=int, default=0 if name == "smoke" else 3)
            command.add_argument("--output", type=Path, help="Optional compressed JSON measurement")
    args = parser.parse_args()
    if args.command != "infer":
        settings = load_settings(args.config)
        overrides = {name: getattr(args, name) for name in ("batch_size", "validation_batch_size")
                     if getattr(args, name) is not None}
        settings = replace(settings, **overrides)
    if args.command == "infer":
        result = infer(args.checkpoint, args.cache, args.output, args.role, args.device, args.workers)
    elif args.command == "train":
        result = fit(args.cache, args.output, settings, device=args.device,
                     workers=args.workers, resume=args.resume, max_steps=args.max_steps)
    else:
        if args.steps < 1 or args.warmup < 0:
            parser.error("steps must be positive and warmup non-negative")
        result = measure(args.cache, settings, device=args.device, workers=args.workers,
                         steps=args.steps, warmup=args.warmup, full_validation=args.full_validation,
                         compare_device_only=args.command == "timing")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
