"""Frame-budgeted training, exhaustive validation and atomic recovery."""
from __future__ import annotations

import lzma
import math
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.annotator.video_models_benchmark.data import read_json, write_json
from experiments.annotator.video_models_benchmark.run_state import (
    Progress,
    save_checkpoint,
)

from .config import Settings
from .data import IGNORE, DenseClips, cycle_batches
from .evaluate import score_predictions
from .model import SpotModel, transform_clips

type PairedBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def configure_runtime(settings: Settings, device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = settings.cudnn_benchmark


def device_batch(batch: tuple | list, device: torch.device) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PairedBatch | None
]:
    clips, labels, _video_ids, frame_ids, positions, *partner = batch
    paired = None
    if partner:
        paired_clips, paired_labels, paired_ids, mixing = partner
        paired = (paired_clips.to(device, non_blocking=True), paired_labels.to(device, non_blocking=True),
                  (paired_ids >= 0).to(device, non_blocking=True),
                  mixing.to(device=device, dtype=torch.float32, non_blocking=True))
    return (clips.to(device, non_blocking=True), labels.to(device, non_blocking=True),
            (frame_ids >= 0).to(device, non_blocking=True), positions, paired)


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None,
                *, paired_targets: torch.Tensor | None = None, mixing: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Weighted mean CE; each MixUp component keeps its own ignore mask.

    Normalise by the total participating target weight. Ignored positions never
    become a class, nor count as zero-valued observations in the denominator.
    """
    numerator = logits.sum() * 0
    denominator = logits.new_zeros(())
    components = [(targets, 1.0)] if paired_targets is None else [(targets, mixing), (paired_targets, 1 - mixing)]
    for labels, fraction in components:
        valid = labels != IGNORE
        losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), weight=weights,
                                 ignore_index=IGNORE, reduction="none").reshape_as(labels)
        if isinstance(fraction, torch.Tensor):
            fraction = fraction[:, None]
            frame_weights = valid if weights is None else weights[labels.masked_fill(~valid, 0)] * valid
            numerator = numerator + (fraction * losses).sum()
            denominator = denominator + (fraction * frame_weights).sum()
        else:
            numerator = numerator + fraction * losses.sum()
            mass = valid.sum() if weights is None else weights[labels[valid]].sum()
            denominator = denominator + fraction * mass
    return numerator / denominator.clamp_min(1)


def optimise_batch(model: nn.Module, optimiser: torch.optim.Optimizer, clips: torch.Tensor,
                   labels: torch.Tensor, valid_frames: torch.Tensor, settings: Settings, device: torch.device,
                   weights: torch.Tensor, *, paired: PairedBatch | None = None) -> torch.Tensor:
    """Shared training arithmetic, with independent partner clips for E2E MixUp."""
    pixels = transform_clips(clips, valid_frames, training=True, augmentation=settings.augmentation)
    paired_targets = None
    mixing: float | torch.Tensor = 1.0
    if settings.mixup and settings.augmentation == "e2e":
        if paired is None:
            raise ValueError("E2E MixUp requires independently sampled partner clips")
        paired_clips, paired_targets, paired_valid, mixing = paired
        partner_pixels = transform_clips(paired_clips, paired_valid, training=True, augmentation="e2e")
        fractions = mixing[:, None, None, None, None]
        pixels = fractions * pixels + (1 - fractions) * partner_pixels
    elif settings.mixup:
        mixing = float(np.random.beta(0.2, 0.2))
        order = torch.randperm(len(pixels), device=device)
        pixels = mixing * pixels + (1 - mixing) * pixels[order]
        paired_targets = labels[order]
    optimiser.zero_grad(set_to_none=True)
    with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
        loss = masked_loss(model(pixels).float(), labels, weights, paired_targets=paired_targets, mixing=mixing)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    loss.backward()
    optimiser.step()
    return loss.detach()


def learning_rate(settings: Settings, completed_frames: int) -> float:
    """Linear warmup, followed by cosine over the remaining frame budget."""
    cycle = completed_frames / settings.frames_per_cycle
    if settings.warmup_cycles and cycle < settings.warmup_cycles:
        return settings.learning_rate * cycle / settings.warmup_cycles
    progress = (cycle - settings.warmup_cycles) / (settings.cycles - settings.warmup_cycles)
    return settings.learning_rate * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def make_loader(dataset: DenseClips, batch_size: int, workers: int, *, batches: list[list[int]] | None = None) -> DataLoader:
    # DataLoader's worker base seed must not consume the model/augmentation RNG,
    # particularly when a resumed process creates a fresh iterator.
    generator = torch.Generator().manual_seed(dataset.seed)
    kwargs: dict[str, Any] = {"num_workers": workers, "persistent_workers": workers > 0,
                              "pin_memory": torch.cuda.is_available(), "generator": generator}
    if workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
    if batches is not None:
        return DataLoader(dataset, batch_sampler=batches, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)


def assemble_probabilities(model: nn.Module, dataset: DenseClips, loader: DataLoader,
                           device: torch.device, classes: int) -> list[np.ndarray]:
    model.eval()
    sums = [np.zeros((video["frame_count"], classes), dtype=np.float32) for video in dataset.videos]
    counts = [np.zeros(video["frame_count"], dtype=np.int32) for video in dataset.videos]
    with torch.inference_mode():
        for clips, _labels, video_ids, frame_ids, _positions in loader:
            pixels = transform_clips(clips.to(device, non_blocking=True),
                                     (frame_ids >= 0).to(device, non_blocking=True), training=False,
                                     augmentation=dataset.augmentation)
            with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.bfloat16):
                probabilities = model(pixels).float().softmax(-1).cpu().numpy()
            for video_id, ids, rows in zip(video_ids.tolist(), frame_ids.numpy(), probabilities, strict=True):
                valid = ids >= 0
                np.add.at(sums[video_id], ids[valid], rows[valid])
                np.add.at(counts[video_id], ids[valid], 1)
    for video_index, start, end in dataset.scopes:
        if np.any(counts[video_index][start:end] == 0):
            raise RuntimeError("Exhaustive inference left uncovered original frames")
    return [total / np.maximum(count[:, None], 1).astype(np.float32) for total, count in zip(sums, counts, strict=True)]


def validation_loss(probabilities: list[np.ndarray], targets: list[np.ndarray], classes: int) -> dict:
    losses = np.zeros(classes, dtype=np.float64)
    counts = np.zeros(classes, dtype=np.int64)
    for rows, labels in zip(probabilities, targets, strict=True):
        valid = labels != IGNORE
        selected = labels[valid]
        per_frame = -np.log(np.maximum(rows[valid, selected], np.finfo(np.float32).tiny))
        losses += np.bincount(selected, weights=per_frame, minlength=classes)
        counts += np.bincount(selected, minlength=classes)
    if counts.sum() == 0:
        raise ValueError("Validation has no eligible frames")
    return {"loss": float(losses.sum() / counts.sum()), "class_counts": counts.tolist(),
            "class_loss": [float(loss / count) if count else None for loss, count in zip(losses, counts, strict=True)],
            "contact_loss": float(losses[1:3].sum() / counts[1:3].sum()) if counts[1:3].sum() else None}


def validate(model: nn.Module, dataset: DenseClips, loader: DataLoader, device: torch.device,
             settings: Settings) -> tuple[dict, list[np.ndarray]]:
    started = time.perf_counter()
    probabilities = assemble_probabilities(model, dataset, loader, device, settings.classes)
    result = validation_loss(probabilities, dataset.targets, settings.classes)
    result.update(score_predictions(dataset.videos, probabilities, settings.mode))
    result["seconds"] = time.perf_counter() - started
    return result, probabilities


def save_validation_artifacts(directory: Path, videos: list[dict], probabilities: list[np.ndarray], result: dict) -> None:
    """Retain the probabilities and decoded streams from this exact validation pass."""
    directory.mkdir(parents=True, exist_ok=True)
    for video, rows in zip(videos, probabilities, strict=True):
        destination = directory / f"{video['item_id']}.npy.xz"
        temporary = destination.with_name(destination.name + ".partial")
        with lzma.open(temporary, "wb", format=lzma.FORMAT_XZ, preset=9) as stream:
            np.save(stream, rows.astype(np.float32))
        temporary.replace(destination)
    write_json(directory / "result.json.gz", result)


def recovery_state(model: nn.Module, optimiser: torch.optim.Optimizer, settings: Settings, *, cycle: int,
                   batch: int, frames: int, step: int, best_loss: float, best_cycle: int,
                   history: list[dict], cycle_counts: list[int], cycle_loss_sum: float,
                   cycle_batches_done: int, elapsed_seconds: float) -> dict:
    return {"settings": asdict(settings), "model": model.state_dict(), "optimiser": optimiser.state_dict(),
            "rng": rng_state(), "cycle": cycle, "batch": batch, "frames": frames, "step": step,
            "best_loss": best_loss, "best_cycle": best_cycle, "history": history,
            "best_metric": best_loss if settings.checkpoint_metric == "loss" else -math.inf,
            "cycle_counts": cycle_counts, "cycle_loss_sum": cycle_loss_sum,
            "cycle_batches_done": cycle_batches_done, "elapsed_seconds": elapsed_seconds}


def load_recovery(path: Path, model: nn.Module, optimiser: torch.optim.Optimizer, settings: Settings) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if Settings(**state["settings"]) != settings:
        raise ValueError("Recovery settings differ from requested protocol")
    model.load_state_dict(state["model"])
    optimiser.load_state_dict(state["optimiser"])
    restore_rng(state["rng"])
    state["settings"] = asdict(settings)
    state.setdefault("best_metric", state["best_loss"])
    return state


def patience_exhausted(state: dict, settings: Settings) -> bool:
    return (settings.early_stopping_patience > 0 and state["best_cycle"] > 0
            and state["cycle"] - state["best_cycle"] >= settings.early_stopping_patience)


def checkpoint_value(validation: dict, settings: Settings) -> float:
    if settings.checkpoint_metric == "loss":
        return float(validation["loss"])
    return float(validation["metrics"][str(settings.selection_tolerance)]["fully_correct_f1"])


def fit(cache: Path, output: Path, settings: Settings, *, device: str = "cpu", workers: int = 0,
        resume: bool = False, max_steps: int | None = None) -> dict:
    """Train to the cycle cap or selected-metric patience, preserving recoverability."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "settings.json.gz"
    if settings_path.exists() and Settings(**read_json(settings_path)) != settings:
        raise ValueError("Output directory has a different protocol")
    recovery_path = output / "recovery.pt"
    if recovery_path.exists() and not resume:
        raise ValueError("Recovery checkpoint already exists; use --resume or a fresh directory")
    if resume and not recovery_path.exists():
        raise ValueError("No recovery checkpoint exists")
    write_json(settings_path, asdict(settings))
    device_object = torch.device(device)
    configure_runtime(settings, device_object)
    seed_everything(settings.seed)
    train_data = DenseClips(cache, "train", settings.mode, settings.clip_length, seed=settings.seed,
                            frames_per_cycle=settings.frames_per_cycle, cycles=settings.cycles, training=True,
                            augmentation=settings.augmentation, mixup=settings.mixup,
                            ignore_contact_halo=settings.ignore_contact_halo)
    validation_data = DenseClips(cache, "validation", settings.mode, settings.clip_length,
                                 augmentation=settings.augmentation,
                                 ignore_contact_halo=settings.ignore_contact_halo)
    inputs = {"videos": train_data.videos + validation_data.videos,
              "frames": [reader.metadata for reader in train_data.readers + validation_data.readers]}
    inputs_path = output / "inputs.json.gz"
    if inputs_path.exists() and read_json(inputs_path) != inputs:
        raise ValueError("Cache labels, split or frame preparation differ from this run")
    write_json(inputs_path, inputs)
    streams_directory = output / "validation-streams"
    streams_directory.mkdir(exist_ok=True)
    model = SpotModel(settings.backbone, settings.clip_length, settings.classes,
                      pretrained=not resume).to(device_object)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    state = recovery_state(model, optimiser, settings, cycle=0, batch=0, frames=0, step=0,
                           best_loss=math.inf, best_cycle=-1, history=[], cycle_counts=[0] * settings.classes,
                           cycle_loss_sum=0.0, cycle_batches_done=0, elapsed_seconds=0.0)
    if resume:
        state = load_recovery(recovery_path, model, optimiser, settings)
    batches = cycle_batches(train_data.samples_per_cycle, settings.batch_size, settings.cycles,
                            start_cycle=state["cycle"], start_batch=state["batch"])
    training_loader = make_loader(train_data, settings.batch_size, workers, batches=batches)
    validation_loader = make_loader(validation_data, settings.validation_batch_size, workers)
    iterator = iter(training_loader) if not patience_exhausted(state, settings) else iter(())
    weights = torch.tensor([1, 5, 5] + ([1] if settings.classes == 4 else []), device=device_object, dtype=torch.float32)
    progress = Progress(output)
    session_start = time.perf_counter()
    elapsed_before = state["elapsed_seconds"]
    initial_step = state["step"]
    batches_per_cycle = math.ceil(train_data.samples_per_cycle / settings.batch_size)
    while state["cycle"] < settings.cycles and not patience_exhausted(state, settings):
        model.train()
        while state["batch"] < batches_per_cycle:
            clips, labels, valid_frames, positions, paired = device_batch(next(iterator), device_object)
            frame_positions = int(positions.sum())
            rate = learning_rate(settings, state["frames"] + frame_positions)
            for group in optimiser.param_groups:
                group["lr"] = rate
            pair_options = {"paired": paired} if paired is not None else {}
            loss = optimise_batch(model, optimiser, clips, labels, valid_frames, settings, device_object, weights,
                                  **pair_options)
            state["frames"] += frame_positions
            state["step"] += 1
            state["batch"] += 1
            state["cycle_batches_done"] += 1
            state["cycle_loss_sum"] += float(loss.detach())
            observed = torch.bincount(labels[labels != IGNORE], minlength=settings.classes).cpu().tolist()
            state["cycle_counts"] = [old + new for old, new in zip(state["cycle_counts"], observed, strict=True)]
            bounded = max_steps is not None and state["step"] - initial_step >= max_steps
            if progress.due() or bounded or state["batch"] == batches_per_cycle:
                state["elapsed_seconds"] = elapsed_before + time.perf_counter() - session_start
                state["rng"] = rng_state()
                state["model"], state["optimiser"] = model.state_dict(), optimiser.state_dict()
                save_checkpoint(recovery_path, state)
                progress.write("training", step=state["step"], completed=state["cycle_batches_done"],
                               cycle=state["cycle"], frames=state["frames"], lr=rate,
                               train_loss=state["cycle_loss_sum"] / state["cycle_batches_done"])
            if bounded:
                return {"complete": False, "frames": state["frames"], "steps": state["step"],
                        "seconds": time.perf_counter() - session_start}
        result, probabilities = validate(model, validation_data, validation_loader, device_object, settings)
        eligible_count = sum(state["cycle_counts"])
        result.update(cycle=state["cycle"] + 1, frames=state["frames"], step=state["step"],
                      sampled_class_proportions=[count / eligible_count if eligible_count else 0.0
                                                 for count in state["cycle_counts"]],
                      ignored_positions=settings.frames_per_cycle - eligible_count,
                      train_loss=state["cycle_loss_sum"] / state["cycle_batches_done"],
                      sampled_class_counts=state["cycle_counts"])
        streams_path = streams_directory / f"cycle{result['cycle']:04d}.json.gz"
        write_json(streams_path, {"cycle": result["cycle"], "settings": asdict(settings), "streams": result["streams"]})
        result["streams_file"] = str(streams_path.relative_to(output))
        metric_value = checkpoint_value(result, settings)
        result.update(checkpoint_metric=settings.checkpoint_metric, selection_tolerance=settings.selection_tolerance,
                      checkpoint_value=metric_value)
        summary = {key: value for key, value in result.items() if key != "streams"}
        state["history"].append(summary)
        previous_best = state["best_cycle"]
        state["best_loss"] = min(state["best_loss"], result["loss"])
        improved = (metric_value < state["best_metric"] if settings.checkpoint_metric == "loss"
                    else metric_value > state["best_metric"])
        if improved:
            state["best_metric"], state["best_cycle"] = metric_value, state["cycle"] + 1
            artifact_directory = f"validation-cycle{state['best_cycle']:04d}"
            save_validation_artifacts(output / artifact_directory, validation_data.videos, probabilities,
                                      {**result, "settings": asdict(settings)})
            save_checkpoint(output / "best.pt", {"settings": asdict(settings), "model": model.state_dict(),
                                                "cycle": state["best_cycle"], "validation": summary,
                                                "best_metric": metric_value,
                                                "validation_artifacts": artifact_directory})
        state.update(cycle=state["cycle"] + 1, batch=0, cycle_counts=[0] * settings.classes,
                     cycle_loss_sum=0.0, cycle_batches_done=0)
        state["elapsed_seconds"] = elapsed_before + time.perf_counter() - session_start
        state["rng"] = rng_state()
        state["model"], state["optimiser"] = model.state_dict(), optimiser.state_dict()
        save_checkpoint(recovery_path, state)
        if previous_best > 0 and previous_best != state["best_cycle"]:
            shutil.rmtree(output / f"validation-cycle{previous_best:04d}")
        write_json(output / "history.json.gz", state["history"])
        print({key: result[key] for key in ("cycle", "frames", "loss", "checkpoint_value", "seconds")}, flush=True)
    # A recovery checkpoint can precede history publication when a process exits.
    write_json(output / "history.json.gz", state["history"])
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    result = {"complete": True, "completed_cycles": state["cycle"],
              "stopping_reason": "early_stopping" if patience_exhausted(state, settings) else "max_cycles",
              "settings": asdict(settings), "best_cycle": state["best_cycle"],
              "best_metric": state["best_metric"], "best_loss": state["best_loss"],
              "validation": best["validation"], "frames": state["frames"], "steps": state["step"],
              "seconds": state["elapsed_seconds"], "validation_artifacts": best["validation_artifacts"]}
    write_json(output / "result.json.gz", result)
    return result
