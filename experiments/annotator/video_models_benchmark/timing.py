"""Measure VideoMAE GPU work, data feeding and complete validation separately."""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import read_json, write_json
from .evaluate import score_probabilities
from .train import Settings, make_optimiser, train_batch, validate


def time_videomae(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, settings: Settings,
    device: torch.device, output: Path, *, warmup_batches: int = 10, measured_batches: int = 200,
) -> None:
    """Discard timed updates; report throughput at the requested effective batch."""
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    results = {}
    try:
        for mode in ("gpu_only", "with_loading"):
            model.load_state_dict(initial)
            model.train()
            optimiser = make_optimiser(model, settings)
            iterator = iter(train_loader)
            resident_batch = tuple(value.to(device) for value in next(iterator)) if mode == "gpu_only" else None
            torch.cuda.reset_peak_memory_stats(device)
            for batch_index in range(warmup_batches + measured_batches):
                if batch_index == warmup_batches:
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                batch = resident_batch if resident_batch is not None else next(iterator)
                train_batch(model, batch, optimiser, device, settings.precision,
                            model_name=settings.model_name, microbatch_size=settings.microbatch_size)
            torch.cuda.synchronize(device)
            seconds = (time.perf_counter() - started) / measured_batches
            results[mode] = {"update_seconds": seconds, "clips_per_second": settings.batch_size / seconds,
                             "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3}
            print(f"{mode}: {results[mode]}", flush=True)
            del optimiser, iterator, resident_batch

        evaluated_centres = len(val_loader.sampler)
        print(f"Validating {evaluated_centres:,} of {len(val_loader.dataset):,} centres", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        validation = validate(model, val_loader, device, settings.precision, model_name=settings.model_name)
        torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - started
        started = time.perf_counter()
        score_probabilities(val_loader.dataset, validation["probabilities"])
        scoring_seconds = time.perf_counter() - started
        results["validation"] = {"seconds": validation_seconds, "event_scoring_seconds": scoring_seconds,
                                 "centres": evaluated_centres, "total_centres": len(val_loader.dataset),
                                 "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3}
    finally:
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(torch_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    output.mkdir(parents=True, exist_ok=True)
    full_cache = read_json(train_loader.dataset.cache / "index.json.gz")["smoke_rallies"] is None
    initial_hours = (settings.max_steps * results["with_loading"]["update_seconds"]
                     + len(settings.validation_steps) * (validation_seconds + scoring_seconds)) / 3600
    write_json(output / "timing.json.gz", {
        "settings": asdict(settings), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "warmup_batches": warmup_batches, "measured_batches": measured_batches, "measurements": results,
        "full_cache": full_cache, "initial_run_hours": initial_hours if full_cache else None,
        "four_run_initial_hours": 4 * initial_hours if full_cache else None,
        "scope": "Timing only; weights restored. Duration excludes continuation, startup and checkpoint writes.",
    })
    print(results, flush=True)
