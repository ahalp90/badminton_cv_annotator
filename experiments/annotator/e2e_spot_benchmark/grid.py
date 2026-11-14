"""Continue the agreed two-seed screening grid after the initial LR queue."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from experiments.annotator.video_models_benchmark.data import read_json, write_json

from .config import Settings
from .search import (
    PREDECESSORS,
    STAGES,
    rank_candidates,
    run_name,
    run_queue,
    stage_candidates,
)


def continue_grid(root: Path, supplied: Settings, discovery: Settings, *,
                  prepare: Callable[[str, list[Settings]], list[Settings]],
                  caches: dict[str, Path], device: str, workers: int) -> None:
    """Freeze each comparison before running it; reuse results and recovery files.

    The caller must hold the grid lock and wait for the original queue to exit
    successfully. Preparation runs only between stages and may resolve batches;
    it must check both recipes before the discovery LR reference runs begin.
    """
    selections: dict[str, dict] = {}
    for stage in STAGES[:-1]:
        selection = selections.get(PREDECESSORS.get(stage, ""))
        base = supplied if stage.startswith("supplied") else discovery
        if selection is not None:
            base = Settings(**selection["settings"])
        requested = stage_candidates(stage, base, selection)
        path = root / f"{stage}-grid-plan.json.gz"
        request = {"candidates": [asdict(value) for value in requested],
                   "caches": {key: str(value) for key, value in caches.items()}}
        if path.exists():
            plan = read_json(path)
            if plan["request"] != request:
                raise ValueError(f"Changed grid request: {path}")
            candidates = [Settings(**row) for row in plan["candidates"]]
            _check_prepared(requested, candidates)
        else:
            candidates = requested if stage == STAGES[0] else prepare(stage, requested)
            _check_prepared(requested, candidates)
            write_json(path, {"stage": stage, "request": request,
                              "candidates": [asdict(value) for value in candidates]})
        write_json(root / "grid-status.json.gz", {"stage": stage, "phase": "running"})
        if stage != STAGES[0]:
            run_queue(candidates, None, root, limit=len(candidates), device=device, workers=workers,
                      cache_by_augmentation=caches)
        ranking = rank_candidates(candidates, root)
        selection = {"stage": stage, **ranking[0], "ranking": ranking}
        write_json(root / f"{stage}-selection.json.gz", selection)
        selections[stage] = selection
        print(f"Completed {stage}: {run_name(Settings(**selection['settings']))}", flush=True)
    write_json(root / "grid-status.json.gz", {"stage": STAGES[-2], "phase": "complete"})


def _check_prepared(requested: list[Settings], candidates: list[Settings]) -> None:
    """Only batch reductions may differ from the agreed candidate order."""
    if len(candidates) != len(requested):
        raise ValueError("Preparation changed the number of candidates")
    for intended, resolved in zip(requested, candidates, strict=True):
        for name, value in asdict(intended).items():
            actual = getattr(resolved, name)
            if name == "validation_batch_size" or (name == "batch_size" and intended.backbone == "800mf"):
                if actual > value:
                    raise ValueError(f"Preparation increased {name}")
            elif actual != value:
                raise ValueError(f"Preparation changed {name}")
    groups: dict[tuple[str, str, int], tuple[int, int]] = {}
    for settings in candidates:
        key = (settings.mode, settings.backbone, settings.clip_length)
        batches = (settings.batch_size, settings.validation_batch_size)
        if key in groups and groups[key] != batches:
            raise ValueError("Candidates in one comparison require common batches")
        groups[key] = batches
