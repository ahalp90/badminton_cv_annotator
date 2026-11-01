"""Fine-tune X3D-M; measure the same path without creating a trained checkpoint."""
from __future__ import annotations

import argparse
import lzma
import math
import random
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
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
from .run_state import Progress, save_checkpoint
from .sampling import ContactBatchSampler


@dataclass(frozen=True)
class Settings:
    seed: int = 0
    batch_size: int = 16
    workers: int = 12
    learning_rate: float = 0.02
    warmup_start_lr: float | None = None
    momentum: float = 0.9
    weight_decay: float = 1e-5
    max_steps: int = 30_000
    warmup_steps: int = 2_000
    validation_steps: list[int] = field(default_factory=lambda: [2500, 5000, 10000, 15000, 20000, 25000, 30000])
    contact_fraction: float = 0.5
    precision: str = "bf16"

    def __post_init__(self) -> None:
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be non-negative and smaller than max_steps")
        if (not self.validation_steps or self.validation_steps != sorted(set(self.validation_steps))
                or self.validation_steps[0] < 1 or self.validation_steps[-1] != self.max_steps):
            raise ValueError("validation_steps must increase strictly from a positive step through max_steps")


def learning_rate_at_step(settings: Settings, step: int) -> float:
    """Return the LR for a zero-based optimiser update, including the final decay."""
    if step < settings.warmup_steps:
        start = settings.warmup_start_lr
        if start is None:
            start = settings.learning_rate * 0.1
        fraction = step / max(settings.warmup_steps - 1, 1)
        return start + fraction * (settings.learning_rate - start)
    fraction = (step - settings.warmup_steps) / max(settings.max_steps - settings.warmup_steps - 1, 1)
    return settings.learning_rate * 0.5 * (1 + math.cos(math.pi * min(fraction, 1.0)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: ClipDataset, settings: Settings, *, training: bool) -> DataLoader:
    batching = ({"batch_sampler": ContactBatchSampler(dataset.labels, settings.batch_size, settings.contact_fraction,
                                                     settings.max_steps, settings.seed)}
                if training else {"batch_size": settings.batch_size, "shuffle": False})
    return DataLoader(
        dataset, **batching, num_workers=settings.workers, pin_memory=True,
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
def validate(
    model: nn.Module, loader: DataLoader, device: torch.device, precision: str, progress: Progress | None = None,
) -> dict:
    model.eval()
    probabilities = []
    loss_sum = torch.zeros((), device=device)
    eligible_count = torch.zeros((), device=device, dtype=torch.long)
    confusion = torch.zeros(9, device=device, dtype=torch.long)
    for batch_index, (clips, targets) in enumerate(loader, start=1):
        clips, targets = clips.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model(prepare_clips(clips, training=False))
            loss_sum += F.cross_entropy(logits, targets, reduction="sum")
        eligible = targets != -100
        eligible_count += eligible.sum()
        predicted = logits.argmax(1)
        confusion += torch.bincount(targets[eligible] * 3 + predicted[eligible], minlength=9)
        probabilities.append(logits.float().softmax(1).cpu().numpy())
        if progress is not None and progress.due():
            progress.write("validation", completed=batch_index)
    matrix = confusion.reshape(3, 3).cpu().numpy()
    denominator = matrix.sum(0) + matrix.sum(1)
    class_f1 = np.divide(2 * matrix.diagonal(), denominator, out=np.zeros(3), where=denominator > 0)
    return {"loss": float(loss_sum / eligible_count), "probabilities": np.concatenate(probabilities),
            "macro_f1": float(class_f1.mean()), "confusion": matrix.tolist()}


def save_recovery(
    output: Path, model: nn.Module, optimiser: torch.optim.Optimizer, loader: DataLoader,
    device: torch.device, step: int, best_loss: float, history_length: int,
) -> None:
    save_checkpoint(output / "last.pt", {
        "model": model.state_dict(), "optimiser": optimiser.state_dict(), "step": step,
        "best_loss": best_loss, "history_length": history_length, "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "loader_rng": loader.generator.get_state() if loader.generator is not None else None,
    })


def fit(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, settings: Settings,
    device: torch.device, output: Path, scheme: str, *, resume: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    run_settings = {**asdict(settings), "scheme": scheme}
    if not resume and (output / "settings.json.gz").exists():
        raise FileExistsError("Run already exists; use --resume or a new output directory")
    if resume and read_json(output / "settings.json.gz") != run_settings:
        raise ValueError("Resume settings differ from the saved run")
    write_json(output / "settings.json.gz", run_settings)
    progress = Progress(output)
    optimiser = torch.optim.SGD(model.parameters(), lr=settings.learning_rate,
                                momentum=settings.momentum, weight_decay=settings.weight_decay)
    best_loss, step, history = float("inf"), 0, []
    if resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        step, best_loss = saved["step"], saved["best_loss"]
        # The best checkpoint may have been saved just before an interrupted commit.
        if (output / "best.pt").exists():
            best_loss = min(best_loss, torch.load(output / "best.pt", map_location="cpu",
                                                weights_only=True)["validation_loss"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if train_loader.generator is not None:
            train_loader.generator.set_state(saved["loader_rng"])
        history = read_json(output / "history.json.gz")[:saved["history_length"]]
    else:
        write_json(output / "history.json.gz", [])
        save_recovery(output, model, optimiser, train_loader, device, 0, best_loss, 0)
    sampler = train_loader.batch_sampler
    if not isinstance(sampler, ContactBatchSampler):
        raise TypeError("Training requires ContactBatchSampler for update-boundary recovery")
    sampler.start_step = step
    loader_rng = train_loader.generator.get_state() if train_loader.generator is not None else None
    iterator = iter(train_loader)
    # Workers only load pixels; augmentations use the restored main-process GPU RNG.
    if resume and step > 0 and train_loader.generator is not None:
        train_loader.generator.set_state(loader_rng)
    for target_step in settings.validation_steps:
        if target_step <= step:
            continue
        started, first_step = time.perf_counter(), step
        model.train()
        training_loss = torch.zeros((), device=device)
        progress.write("training", step=step, max_steps=settings.max_steps, completed=0,
                       total=target_step - step, phase_started=time.time(), lr=learning_rate_at_step(settings, step))
        for batch in islice(iterator, target_step - step):
            for group in optimiser.param_groups:
                group["lr"] = learning_rate_at_step(settings, step)
            training_loss += train_batch(model, batch, optimiser, device, settings.precision)
            step += 1
            if progress.due():
                if not torch.isfinite(training_loss):
                    raise ValueError("Non-finite training loss")
                progress.write("training", completed=step - first_step, step=step,
                               train_loss=float(training_loss / (step - first_step)),
                               lr=optimiser.param_groups[0]["lr"])
        if step != target_step:
            raise RuntimeError("Training loader ended before the configured update budget")
        progress.write("validation", completed=0, total=len(val_loader), phase_started=time.time(), step=step)
        result = validate(model, val_loader, device, settings.precision, progress)
        probabilities = result.pop("probabilities")
        progress.write("event scoring", completed=0, total=0, val_loss=result["loss"])
        event_scores = score_probabilities(val_loader.dataset, probabilities)
        loss = result["loss"]
        if not np.isfinite(loss) or not torch.isfinite(training_loss):
            raise ValueError("Non-finite training or validation loss")
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(output / "best.pt", {"model": model.state_dict(), "settings": asdict(settings),
                                                "scheme": scheme, "step": step, "validation_loss": loss})
            predictions_path = output / "validation_probabilities.npy.xz"
            temporary = predictions_path.with_name(predictions_path.name + ".partial")
            with lzma.open(temporary, "wb", preset=9) as stream:
                np.save(stream, probabilities)
            temporary.replace(predictions_path)
            write_json(output / "validation_events.json.gz", event_scores)
        row = {"steps": step, "train_loss": float(training_loss / (step - first_step)),
               "validation": result, "events": event_scores["metrics"],
               "next_lr": learning_rate_at_step(settings, step), "seconds": time.perf_counter() - started}
        history.append(row)
        write_json(output / "history.json.gz", history)
        save_recovery(output, model, optimiser, train_loader, device, step, best_loss, len(history))
        progress.write("validation complete", total=0, best_loss=best_loss, val_loss=loss)
        print(row, flush=True)
        if (output / "PAUSE").exists() and step < settings.max_steps:
            progress.write("paused", total=0)
            return
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model"])
    progress.write("complete", step=step, max_steps=settings.max_steps, total=0)


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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = Settings(**tomllib.loads(args.config.read_text()))
    if args.warmup_steps is not None:
        settings = replace(settings, warmup_steps=args.warmup_steps)
    if args.command == "train" and read_json(args.cache / "index.json.gz")["smoke_rallies"] is not None:
        parser.error("The bounded smoke cache cannot be used for benchmark training")
    seed_everything(settings.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model = build_model(pretrained=not args.resume).to(device)
    train_loader = make_loader(ClipDataset(args.cache, "train", args.scheme, training=True), settings, training=True)
    val_loader = make_loader(ClipDataset(args.cache, "validation", args.scheme, training=False), settings, training=False)
    if args.command == "time":
        time_pipeline(model, train_loader, val_loader, settings, device, args.output)
    else:
        fit(model, train_loader, val_loader, settings, device, args.output, args.scheme,
            resume=args.resume)


if __name__ == "__main__":
    main()
