"""Fine-tune X3D-M; measure the same path without creating a trained checkpoint."""
from __future__ import annotations

import argparse
import lzma
import math
import random
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ClipDataset, read_json, write_json
from .evaluate import score_probabilities
from .model import build_model, prepare_clips


@dataclass(frozen=True)
class Settings:
    seed: int = 0
    batch_size: int = 16
    workers: int = 12
    learning_rate: float = 0.02
    warmup_start_lr: float = 0.002
    momentum: float = 0.9
    weight_decay: float = 1e-5
    plateau_factor: float = 0.1
    plateau_patience: int = 3
    early_stop_patience: int = 15
    improvement_threshold: float = 1e-4
    max_epochs: int = 1000
    precision: str = "bf16"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: ClipDataset, settings: Settings, *, training: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=settings.batch_size, shuffle=training,
        num_workers=settings.workers, pin_memory=True,
        persistent_workers=settings.workers > 0,
        **({"prefetch_factor": 2} if settings.workers else {}),
        generator=torch.Generator().manual_seed(settings.seed),
    )


def autocast(device: torch.device, precision: str):
    return torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def train_batch(
    model: nn.Module, batch: tuple, optimiser: torch.optim.Optimizer, device: torch.device, precision: str,
) -> torch.Tensor:
    clips, targets = (value.to(device, non_blocking=True) for value in batch)
    optimiser.zero_grad(set_to_none=True)
    clips = prepare_clips(clips, training=True)
    with autocast(device, precision):
        logits = model(clips)
        loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimiser.step()
    return loss.detach()


@torch.inference_mode()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, precision: str) -> dict:
    model.eval()
    probabilities = []
    loss_sum = torch.zeros((), device=device)
    eligible_count = torch.zeros((), device=device, dtype=torch.long)
    confusion = torch.zeros(9, device=device, dtype=torch.long)
    for clips, targets in loader:
        clips, targets = clips.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model(prepare_clips(clips, training=False))
            loss_sum += F.cross_entropy(logits, targets, reduction="sum")
        eligible = targets != -100
        eligible_count += eligible.sum()
        predicted = logits.argmax(1)
        confusion += torch.bincount(targets[eligible] * 3 + predicted[eligible], minlength=9)
        probabilities.append(logits.float().softmax(1).cpu().numpy())
    matrix = confusion.reshape(3, 3).cpu().numpy()
    denominator = matrix.sum(0) + matrix.sum(1)
    class_f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(3), where=denominator > 0)
    return {"loss": float(loss_sum / eligible_count), "probabilities": np.concatenate(probabilities),
            "macro_f1": float(class_f1.mean()), "confusion": matrix.tolist()}


def fit(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, settings: Settings,
    device: torch.device, warmup_steps: int, output: Path, scheme: str,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "settings.json.gz", {**asdict(settings), "scheme": scheme, "warmup_steps": warmup_steps})
    optimiser = torch.optim.SGD(model.parameters(), lr=settings.learning_rate,
                                momentum=settings.momentum, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=settings.plateau_factor, patience=settings.plateau_patience,
        threshold=settings.improvement_threshold, threshold_mode="abs",
    )
    best_loss, stopping_best, bad_epochs, step = float("inf"), float("inf"), 0, 0
    history = []
    for epoch in range(settings.max_epochs):
        started = time.perf_counter()
        model.train()
        training_loss = torch.zeros((), device=device)
        for batch in train_loader:
            if step < warmup_steps:
                progress = step / max(warmup_steps - 1, 1)
                learning_rate = settings.warmup_start_lr + progress * (settings.learning_rate - settings.warmup_start_lr)
                for group in optimiser.param_groups:
                    group["lr"] = learning_rate
            training_loss += train_batch(model, batch, optimiser, device, settings.precision) * len(batch[1])
            step += 1
        result = validate(model, val_loader, device, settings.precision)
        probabilities = result.pop("probabilities")
        event_scores = score_probabilities(val_loader.dataset, probabilities)
        loss = result["loss"]
        if not np.isfinite(loss) or not torch.isfinite(training_loss):
            raise ValueError("Non-finite training or validation loss")
        if loss < best_loss:
            best_loss = loss
            torch.save({"model": model.state_dict(), "settings": asdict(settings), "scheme": scheme,
                        "epoch": epoch + 1, "step": step, "validation_loss": loss}, output / "best.pt")
            with lzma.open(output / "validation_probabilities.npy.xz", "wb", preset=9) as stream:
                np.save(stream, probabilities)
            write_json(output / "validation_events.json.gz", event_scores)
        if step >= warmup_steps:
            scheduler.step(loss)
            if loss < stopping_best - settings.improvement_threshold:
                stopping_best, bad_epochs = loss, 0
            else:
                bad_epochs += 1
        row = {"epoch": epoch + 1, "steps": step, "train_loss": float(training_loss / len(train_loader.dataset)),
               "validation": result, "events": event_scores["metrics"], "next_lr": optimiser.param_groups[0]["lr"],
               "bad_epochs": bad_epochs, "seconds": time.perf_counter() - started}
        history.append(row)
        write_json(output / "history.json.gz", history)
        print(row, flush=True)
        if bad_epochs >= settings.early_stop_patience:
            break
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model"])


def time_pipeline(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, settings: Settings,
    device: torch.device, output: Path, *, warmup_batches: int = 10, measured_batches: int = 40,
) -> None:
    """Discard all timed updates, including batch-normalisation running statistics."""
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimiser = torch.optim.SGD(model.parameters(), lr=settings.learning_rate,
                                momentum=settings.momentum, weight_decay=settings.weight_decay)
    model.train()
    iterator = iter(train_loader)
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup_batches):
        train_batch(model, next(iterator), optimiser, device, settings.precision)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(measured_batches):
        train_batch(model, next(iterator), optimiser, device, settings.precision)
    torch.cuda.synchronize(device)
    train_seconds = (time.perf_counter() - started) / measured_batches
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    model.eval()
    with torch.inference_mode():
        iterator = iter(val_loader)
        for clips, _targets in islice(iterator, 3):
            with autocast(device, settings.precision):
                model(prepare_clips(clips.to(device, non_blocking=True), training=False))
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        count = 0
        for clips, targets in islice(iterator, measured_batches):
            with autocast(device, settings.precision):
                logits = model(prepare_clips(clips.to(device, non_blocking=True), training=False))
                F.cross_entropy(logits, targets.to(device, non_blocking=True), reduction="sum")
            logits.float().softmax(1).cpu().numpy()
            count += 1
        torch.cuda.synchronize(device)
    val_seconds = (time.perf_counter() - started) / count
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    output.mkdir(parents=True, exist_ok=True)
    counts = read_json(train_loader.dataset.cache / "index.json.gz")["full_counts"]
    val_batches = math.ceil(counts["validation"]["centres"] / settings.batch_size)
    epoch_hours = {}
    for scheme in ("A", "B"):
        train_batches = math.ceil(counts["train"][scheme] / settings.batch_size)
        epoch_hours[scheme] = (train_batches * train_seconds + val_batches * val_seconds) / 3600
    write_json(output / "timing.json.gz", {
        "settings": asdict(settings), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "warmup_batches": warmup_batches, "measured_train_batches": measured_batches,
        "measured_val_batches": count, "train_batch_seconds": train_seconds, "val_batch_seconds": val_seconds,
        "peak_allocated_gib": peak_gib, "timed_train_centres": len(train_loader.dataset),
        "timed_validation_centres": len(val_loader.dataset),
        "full_epoch_hours": epoch_hours,
        "scope": "Shuffled cached clips, transfer, GPU augmentation, full forward/backward/SGD; weights restored",
    })
    print(read_json(output / "timing.json.gz"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("time", "train"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheme", choices=("A", "B"), default="A")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pilot.toml"))
    parser.add_argument("--warmup-steps", type=int)
    args = parser.parse_args()
    settings = Settings(**tomllib.loads(args.config.read_text()))
    if args.command == "train" and (args.warmup_steps is None or args.warmup_steps < 1):
        parser.error("Training requires an explicitly agreed positive --warmup-steps")
    if args.command == "train" and read_json(args.cache / "index.json.gz")["smoke_rallies"] is not None:
        parser.error("The bounded smoke cache cannot be used for benchmark training")
    seed_everything(settings.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model = build_model().to(device)
    train_loader = make_loader(ClipDataset(args.cache, "train", args.scheme, training=True), settings, training=True)
    val_loader = make_loader(ClipDataset(args.cache, "validation", args.scheme, training=False), settings, training=False)
    if args.command == "time":
        time_pipeline(model, train_loader, val_loader, settings, device, args.output)
    else:
        fit(model, train_loader, val_loader, settings, device, args.warmup_steps, args.output, args.scheme)


if __name__ == "__main__":
    main()
