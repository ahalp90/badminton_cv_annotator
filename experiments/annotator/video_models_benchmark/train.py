"""Fine-tune video contact models with shared sampling, scoring and recovery."""
from __future__ import annotations

import argparse
import lzma
import math
import random
import shutil
import time
import tomllib
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from itertools import chain, count, islice
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
from .sampling import ContactBatchSampler, EvaluationSampler
from .schedule import (
    HALF_CYCLE_STEPS,
    PATIENCE,
    VALIDATION_INTERVAL,
    checks_without_improvement,
    continuation_rate,
)
from .videomae import CHECKPOINT_REVISION, VideoMAE, parameter_groups


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
    model_name: str = "x3d"
    microbatch_size: int | None = None
    checkpoint_revision: str = CHECKPOINT_REVISION
    layer_decay: float = 0.75
    stochastic_depth: float = 0.2
    continuation_half_cycle_steps: int | None = None
    continuation_validation_interval: int | None = None
    # Records predating stride selection used dense validation; new configs use 3.
    validation_stride: int = 1
    max_total_steps: int | None = None
    continue_until_patience: bool = False
    break_alternation_ties: bool = False

    def __post_init__(self) -> None:
        if self.max_total_steps is not None and self.max_total_steps < self.max_steps:
            raise ValueError("max_total_steps must include the initial max_steps budget")
        if self.validation_stride < 1:
            raise ValueError("validation_stride must be positive")
        if self.model_name not in ("x3d", "videomae"):
            raise ValueError("model_name must be x3d or videomae")
        if self.batch_size < 2 or self.workers < 0:
            raise ValueError("batch_size must be at least two and workers non-negative")
        if self.microbatch_size is not None and (
            self.model_name != "videomae" or not 1 <= self.microbatch_size <= self.batch_size
        ):
            raise ValueError("microbatch_size requires VideoMAE and must be between one and batch_size")
        if not 0 < self.layer_decay <= 1 or not 0 <= self.stochastic_depth < 1:
            raise ValueError("layer_decay must be in (0, 1] and stochastic_depth in [0, 1)")
        if self.continuation_half_cycle_steps is not None and not 0 < self.continuation_half_cycle_steps < self.max_steps:
            raise ValueError("continuation_half_cycle_steps must be positive and smaller than max_steps")
        if self.continuation_validation_interval is not None and self.continuation_validation_interval < 1:
            raise ValueError("continuation_validation_interval must be positive")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be non-negative and smaller than max_steps")
        if (not self.validation_steps or self.validation_steps != sorted(set(self.validation_steps))
                or self.validation_steps[0] < 1 or self.validation_steps[-1] != self.max_steps):
            raise ValueError("validation_steps must increase strictly from a positive step through max_steps")


def normalise_settings_record(record: dict) -> dict:
    """Read the former tie-break flag from existing configs and checkpoints."""
    record = dict(record)
    if "force_alternation" in record:
        if "break_alternation_ties" in record:
            raise ValueError("Use only break_alternation_ties, not both tie-break setting names")
        record["break_alternation_ties"] = record.pop("force_alternation")
    return record


def settings_record(settings: Settings) -> dict:
    """Keep X3D records compatible with the existing training and inference tools."""
    record = asdict(settings)
    if settings.model_name == "x3d":
        for name in ("model_name", "microbatch_size", "checkpoint_revision", "layer_decay", "stochastic_depth"):
            del record[name]
    for name in ("continuation_half_cycle_steps", "continuation_validation_interval", "max_total_steps"):
        if record[name] is None:
            del record[name]
    return record


def model_from_settings(settings: Settings, *, pretrained: bool) -> nn.Module:
    return build_model(pretrained=pretrained, model_name=settings.model_name,
                       checkpoint_revision=settings.checkpoint_revision, stochastic_depth=settings.stochastic_depth)


def make_optimiser(model: nn.Module, settings: Settings) -> torch.optim.Optimizer:
    if settings.model_name == "videomae":
        if not isinstance(model, VideoMAE):
            raise TypeError("VideoMAE settings require a VideoMAE model")
        optimiser = torch.optim.AdamW(parameter_groups(model, settings.layer_decay, settings.weight_decay),
                                      lr=settings.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        set_learning_rate(optimiser, settings.learning_rate)
        return optimiser
    return torch.optim.SGD(model.parameters(), lr=settings.learning_rate,
                           momentum=settings.momentum, weight_decay=settings.weight_decay)


def set_learning_rate(optimiser: torch.optim.Optimizer, rate: float) -> None:
    for group in optimiser.param_groups:
        group["lr"] = rate * group.get("lr_scale", 1.0)


def learning_rate_at_step(settings: Settings, step: int) -> float:
    """Return the LR for a zero-based optimiser update, including the final decay."""
    if settings.continue_until_patience and step >= settings.max_steps:
        half_cycle = settings.continuation_half_cycle_steps or HALF_CYCLE_STEPS
        peak = learning_rate_at_step(settings, settings.max_steps - half_cycle)
        return continuation_rate(step, settings.max_steps, peak, half_cycle_steps=half_cycle)
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


def make_loader(
    dataset: ClipDataset, settings: Settings, *, training: bool, extend: bool = False,
    stride: int | None = None, offset: int = 0,
) -> DataLoader:
    batching = ({"batch_sampler": ContactBatchSampler(dataset.labels, settings.batch_size, settings.contact_fraction,
                                                     settings.max_steps, settings.seed,
                                                     continuous=extend or settings.continue_until_patience)}
                if training else {"batch_size": settings.microbatch_size or settings.batch_size,
                                  "sampler": EvaluationSampler(dataset.rows[:, 2],
                                                               settings.validation_stride if stride is None else stride,
                                                               offset)})
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
    *, model_name: str = "x3d", microbatch_size: int | None = None,
) -> torch.Tensor:
    clips, targets = batch
    optimiser.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    batch_size = len(targets)
    for start in range(0, batch_size, microbatch_size or batch_size):
        stop = min(start + (microbatch_size or batch_size), batch_size)
        part = clips[start:stop].to(device, non_blocking=True)
        labels = targets[start:stop].to(device, non_blocking=True)
        part = (prepare_clips(part, training=True) if model_name == "x3d"
                else prepare_clips(part, training=True, model_name=model_name))
        with autocast(device, precision):
            logits = model(part)
            loss = F.cross_entropy(logits, labels) * ((stop - start) / batch_size)
        loss.backward()
        total_loss += loss.detach()
    optimiser.step()
    return total_loss


@torch.inference_mode()
def validate(
    model: nn.Module, loader: DataLoader, device: torch.device, precision: str, progress: Progress | None = None,
    *, model_name: str = "x3d",
) -> dict:
    model.eval()
    probabilities = []
    loss_sum = torch.zeros((), device=device)
    eligible_count = torch.zeros((), device=device, dtype=torch.long)
    confusion = torch.zeros(9, device=device, dtype=torch.long)
    for batch_index, (clips, targets) in enumerate(loader, start=1):
        clips, targets = clips.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with autocast(device, precision):
            prepared = (prepare_clips(clips, training=False) if model_name == "x3d"
                        else prepare_clips(clips, training=False, model_name=model_name))
            logits = model(prepared)
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
    result = {"loss": float(loss_sum / eligible_count), "probabilities": np.concatenate(probabilities),
              "macro_f1": float(class_f1.mean()), "confusion": matrix.tolist()}
    if isinstance(loader.sampler, EvaluationSampler):
        sampler = loader.sampler
        sampled_probabilities = result["probabilities"]
        # Event scoring keeps every rally/contact; skipped centres cannot emit events.
        full_probabilities = np.zeros((len(loader.dataset), 3), dtype=sampled_probabilities.dtype)
        full_probabilities[:, 0] = 1
        full_probabilities[sampler.indices] = sampled_probabilities
        result["probabilities"] = full_probabilities
        result["sampling"] = {"stride": sampler.stride, "offset": sampler.offset,
                              "total_centres": len(loader.dataset), "evaluated_centres": len(sampler),
                              "unsampled_probabilities": "background"}
    return result


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
    device: torch.device, output: Path, scheme: str, *, resume: bool = False, extend: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model_settings = settings_record(settings)
    run_settings = {**model_settings, "scheme": scheme}
    if not resume and (output / "settings.json.gz").exists():
        raise FileExistsError("Run already exists; use --resume or a new output directory")
    if resume:
        saved_settings = read_json(output / "settings.json.gz")
        saved_scheme = saved_settings.pop("scheme")
        if Settings(**normalise_settings_record(saved_settings)) != settings or saved_scheme != scheme:
            raise ValueError("Resume settings differ from the saved run")
    if extend and not resume:
        raise ValueError("Continuation requires a saved 30k run and --resume")
    continuing = extend or settings.continue_until_patience
    if not continuing and (output / "continuation.json.gz").exists():
        raise ValueError("This run is a continuation; resume with --extend")
    write_json(output / "settings.json.gz", run_settings)
    progress = Progress(output)
    optimiser = make_optimiser(model, settings)
    update_batch = train_batch
    validate_model = validate
    score_events = (partial(score_probabilities, break_alternation_ties=True)
                    if settings.break_alternation_ties else score_probabilities)
    if settings.model_name == "videomae":
        update_batch = partial(train_batch, model_name=settings.model_name, microbatch_size=settings.microbatch_size)
        validate_model = partial(validate, model_name=settings.model_name)
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
    rate_at_step = partial(learning_rate_at_step, settings)
    targets = iter(settings.validation_steps)
    unsuccessful = checks_without_improvement(history)
    if continuing:
        if extend and step < settings.max_steps:
            raise ValueError("Complete the original training budget before continuing")
        half_cycle = settings.continuation_half_cycle_steps or HALF_CYCLE_STEPS
        interval = settings.continuation_validation_interval or VALIDATION_INTERVAL
        peak = learning_rate_at_step(settings, settings.max_steps - half_cycle)
        protocol = {"start_step": settings.max_steps, "peak_lr": peak, "half_cycle_steps": half_cycle,
                    "validation_interval": interval, "patience": PATIENCE}
        if settings.max_total_steps is not None:
            protocol["max_total_steps"] = settings.max_total_steps
        protocol_path = output / "continuation.json.gz"
        if protocol_path.exists() and read_json(protocol_path) != protocol:
            raise ValueError("Continuation protocol differs from the saved run")
        write_json(protocol_path, protocol)
        if extend:
            rate_at_step = partial(continuation_rate, start_step=settings.max_steps, peak=peak, half_cycle_steps=half_cycle)
            targets = count(step + interval, interval)
        else:
            targets = chain(settings.validation_steps, count(settings.max_steps + interval, interval))
        progress.state.update(patience=PATIENCE, bad_checks=unsuccessful)
    sampler = train_loader.batch_sampler
    if not isinstance(sampler, ContactBatchSampler):
        raise TypeError("Training requires ContactBatchSampler for update-boundary recovery")
    if continuing and sampler.batches is not None:
        raise ValueError("Continuation requires the continuous contact sampler")
    sampler.start_step = step
    loader_rng = train_loader.generator.get_state() if train_loader.generator is not None else None
    iterator = iter(train_loader)
    # Workers only load pixels; augmentations use the restored main-process GPU RNG.
    if resume and step > 0 and train_loader.generator is not None:
        train_loader.generator.set_state(loader_rng)
    for target_step in targets:
        if continuing and step >= settings.max_steps and unsuccessful >= PATIENCE:
            break
        if settings.max_total_steps is not None:
            if step >= settings.max_total_steps:
                break
            target_step = min(target_step, settings.max_total_steps)
        if target_step <= step:
            continue
        started, first_step = time.perf_counter(), step
        model.train()
        training_loss = torch.zeros((), device=device)
        progress.write("training", step=step, max_steps=settings.max_total_steps if continuing else settings.max_steps,
                       stage="continuation" if step >= settings.max_steps else "initial",
                       completed=0,
                       total=target_step - step, phase_started=time.time(), lr=rate_at_step(step))
        for batch in islice(iterator, target_step - step):
            set_learning_rate(optimiser, rate_at_step(step))
            training_loss += update_batch(model, batch, optimiser, device, settings.precision)
            step += 1
            if progress.due():
                if not torch.isfinite(training_loss):
                    raise ValueError("Non-finite training loss")
                progress.write("training", completed=step - first_step, step=step,
                               train_loss=float(training_loss / (step - first_step)),
                               lr=rate_at_step(step - 1))
        if step != target_step:
            raise RuntimeError("Training loader ended before the configured update budget")
        progress.write("validation", completed=0, total=len(val_loader), phase_started=time.time(), step=step)
        result = validate_model(model, val_loader, device, settings.precision, progress)
        probabilities = result.pop("probabilities")
        progress.write("event scoring", completed=0, total=0, val_loss=result["loss"])
        event_scores = score_events(val_loader.dataset, probabilities)
        if "sampling" in result:
            event_scores["sampling"] = result["sampling"]
        loss = result["loss"]
        if not np.isfinite(loss) or not torch.isfinite(training_loss):
            raise ValueError("Non-finite training or validation loss")
        check_output = output / "validation_checks" / f"step{step:06d}"
        check_output.mkdir(parents=True, exist_ok=True)
        save_checkpoint(check_output / "checkpoint.pt", {
            "model": model.state_dict(), "settings": model_settings,
            "scheme": scheme, "step": step, "validation_loss": loss,
        })
        with lzma.open(check_output / "probabilities.npy.xz", "wb", preset=9) as stream:
            np.save(stream, probabilities)
        write_json(check_output / "events.json.gz", event_scores)
        write_json(check_output / "frame_metrics.json.gz", result)
        if loss < best_loss:
            best_loss = loss
            for source, destination in (("checkpoint.pt", "best.pt"),
                                        ("probabilities.npy.xz", "validation_probabilities.npy.xz"),
                                        ("events.json.gz", "validation_events.json.gz")):
                temporary = output / (destination + ".partial")
                shutil.copyfile(check_output / source, temporary)
                temporary.replace(output / destination)
        row = {"steps": step, "train_loss": float(training_loss / (step - first_step)),
               "validation": result, "events": event_scores["metrics"],
               "next_lr": rate_at_step(step), "seconds": time.perf_counter() - started}
        history.append(row)
        unsuccessful = checks_without_improvement(history)
        write_json(output / "history.json.gz", history)
        save_recovery(output, model, optimiser, train_loader, device, step, best_loss, len(history))
        progress.write("validation complete", total=0, best_loss=best_loss, val_loss=loss,
                       bad_checks=unsuccessful)
        print(row, flush=True)
        if continuing and step >= settings.max_steps and unsuccessful >= PATIENCE:
            break
        if (output / "PAUSE").exists() and (continuing or step < settings.max_steps):
            progress.write("paused", total=0)
            return
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model"])
    progress.write("complete", step=step, max_steps=settings.max_total_steps if continuing else settings.max_steps, total=0,
                   stage="continuation" if continuing else "initial",
                   best_loss=best_loss, bad_checks=unsuccessful)


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
    val_batches = math.ceil(counts["validation"]["centres"] / settings.validation_stride / settings.batch_size)
    epoch_hours = {}
    for scheme in ("A", "B"):
        train_batches = math.ceil(counts["train"][scheme] / settings.batch_size)
        epoch_hours[scheme] = (train_batches * train_seconds + val_batches * val_seconds) / 3600
    write_json(output / "timing.json.gz", {
        "settings": settings_record(settings), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "warmup_batches": warmup_batches, "measured_train_batches": measured_batches,
        "measured_val_batches": count, "train_batch_seconds": train_seconds, "val_batch_seconds": val_seconds,
        "peak_allocated_gib": peak_gib, "timed_train_centres": len(train_loader.dataset),
        "timed_validation_centres": len(val_loader.sampler),
        "full_epoch_hours": epoch_hours,
        "scope": "Shuffled cached clips, transfer, GPU augmentation, full forward/backward/SGD; weights restored",
    })
    print(read_json(output / "timing.json.gz"), flush=True)


def load_settings(config: Path, output: Path, *, resume: bool) -> Settings:
    """Default new runs to stride three and preserve a resumed run's validation."""
    record = tomllib.loads(config.read_text())
    if resume:
        record["validation_stride"] = read_json(output / "settings.json.gz").get("validation_stride", 1)
    else:
        record.setdefault("validation_stride", 3)
    return Settings(**normalise_settings_record(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("time", "train"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheme", choices=("A", "B"), default="A")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pilot.toml"))
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend", action="store_true", help="Continue a completed run with cosine cycling and patience")
    args = parser.parse_args()
    if args.extend and (not args.resume or args.command != "train"):
        parser.error("--extend requires train --resume")
    settings = load_settings(args.config, args.output, resume=args.resume)
    overrides = {name: getattr(args, name) for name in ("batch_size", "microbatch_size", "workers")
                 if getattr(args, name) is not None}
    settings = replace(settings, **overrides)
    if args.warmup_steps is not None:
        settings = replace(settings, warmup_steps=args.warmup_steps)
    if args.command == "train" and read_json(args.cache / "index.json.gz")["smoke_rallies"] is not None:
        parser.error("The bounded smoke cache cannot be used for benchmark training")
    seed_everything(settings.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model = model_from_settings(settings, pretrained=not args.resume).to(device)
    train_loader = make_loader(ClipDataset(args.cache, "train", args.scheme, training=True), settings,
                               training=True, extend=args.extend)
    val_loader = make_loader(ClipDataset(args.cache, "validation", args.scheme, training=False), settings, training=False)
    if args.command == "time":
        if settings.model_name == "videomae":
            from .timing import time_videomae

            time_videomae(model, train_loader, val_loader, settings, device, args.output)
        else:
            time_pipeline(model, train_loader, val_loader, settings, device, args.output)
    else:
        fit(model, train_loader, val_loader, settings, device, args.output, args.scheme,
            resume=args.resume, extend=args.extend)


if __name__ == "__main__":
    main()
