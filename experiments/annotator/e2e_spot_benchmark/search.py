"""Explicit staged validation search; two seeds rank, five complete a frozen choice."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev

from experiments.annotator.video_models_benchmark.data import read_json, write_json

from .config import Settings, load_settings

STAGES = ("supplied200-lr", "supplied200-mixup", "supplied800-lr", "discovery200-length-lr", "discovery800-lr", "complete")
SEARCHED_FIELDS = ("mode", "backbone", "clip_length", "learning_rate", "mixup")
PREDECESSORS = {
    "supplied200-mixup": "supplied200-lr",
    "supplied800-lr": "supplied200-mixup",
    "discovery200-length-lr": "supplied200-mixup",
    "discovery800-lr": "discovery200-length-lr",
}
COMPLETION_WINNERS = {
    ("supplied", "200mf"): "supplied200-mixup",
    ("supplied", "800mf"): "supplied800-lr",
    ("discovery", "200mf"): "discovery200-length-lr",
    ("discovery", "800mf"): "discovery800-lr",
}


def run_name(settings: Settings) -> str:
    return (f"{settings.mode}-{settings.backbone}-clip{settings.clip_length}-lr{settings.learning_rate:g}"
            f"-mixup{int(settings.mixup)}-seed{settings.seed}")


def stage_candidates(stage: str, base: Settings, selection: dict | None = None, *,
                     batch_by_length: dict[int, int] | None = None) -> list[Settings]:
    """New stages use the supplied runtime config; completion preserves the winner."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    if batch_by_length is not None:
        if stage != "discovery200-length-lr":
            raise ValueError("Per-length batch overrides apply only to discovery200-length-lr")
        if any(length not in (250, 500) or batch < 1 for length, batch in batch_by_length.items()):
            raise ValueError("Per-length batches require clip length 250 or 500 and a positive batch size")
    batch_by_length = batch_by_length or {}
    if stage != "supplied200-lr" and selection is None:
        raise ValueError("This stage needs an explicit frozen selection from the preceding stage")
    if stage == "supplied200-lr" and selection is not None:
        raise ValueError("The first stage does not take a preceding selection")
    selected = None
    inherited = base
    if selection is not None:
        selected = Settings(**selection["settings"])
        predecessor = (COMPLETION_WINNERS[(selected.mode, selected.backbone)] if stage == "complete"
                       else PREDECESSORS[stage])
        if selection["stage"] != predecessor:
            raise ValueError(f"{stage} requires a selection from {predecessor}, got {selection['stage']}")
        inherited = replace(base, **{name: getattr(selected, name) for name in SEARCHED_FIELDS}, seed=selected.seed)
        if stage == "complete":
            conflicts = [name for name, value in asdict(inherited).items() if value != getattr(selected, name)]
            if conflicts:
                raise ValueError("Completion must preserve the winner's full protocol; conflicting config fields: "
                                 + ", ".join(conflicts))
            inherited = selected
    expected = {"supplied200-mixup": ("supplied", "200mf"), "supplied800-lr": ("supplied", "200mf"),
                "discovery200-length-lr": ("supplied", "200mf"), "discovery800-lr": ("discovery", "200mf")}
    if selected is not None and stage in expected and (selected.mode, selected.backbone) != expected[stage]:
        raise ValueError("Selection is from the wrong preceding mode/backbone")
    if stage == "supplied200-lr":
        choices = [replace(base, backbone="200mf", mode="supplied", clip_length=100,
                           mixup=False, learning_rate=rate) for rate in (0.001, 0.0003)]
    elif stage == "supplied200-mixup":
        choices = [replace(inherited, mixup=mixup) for mixup in (False, True)]
    elif stage == "supplied800-lr":
        choices = [replace(inherited, backbone="800mf", learning_rate=rate) for rate in (0.001, 0.0003)]
    elif stage == "discovery200-length-lr":
        choices = [replace(inherited, backbone="200mf", mode="discovery", clip_length=length, learning_rate=rate,
                           batch_size=batch_by_length.get(length, inherited.batch_size))
                   for length in (250, 500) for rate in (0.001, 0.0003)]
    elif stage == "discovery800-lr":
        choices = [replace(inherited, backbone="800mf", learning_rate=rate) for rate in (0.001, 0.0003)]
    else:
        choices = [inherited]
    seeds = range(5) if stage == "complete" else range(2)
    return [replace(choice, seed=seed) for choice in choices for seed in seeds]


def rank_candidates(candidates: list[Settings], root: Path) -> list[dict]:
    groups: dict[Settings, list[dict]] = {}
    for settings in candidates:
        if settings.seed not in (0, 1):
            continue
        path = root / run_name(settings) / "result.json.gz"
        if not path.exists():
            raise ValueError(f"Missing completed validation result: {path}")
        result = read_json(path)
        if not result["complete"] or result["settings"] != asdict(settings):
            raise ValueError(f"Incomplete or changed protocol: {path}")
        key = replace(settings, seed=0)
        groups.setdefault(key, []).append(result)
    ranking = []
    for settings, results in groups.items():
        if sorted(result["settings"]["seed"] for result in results) != [0, 1]:
            raise ValueError("Each ranked configuration needs exactly seeds 0 and 1")
        metrics = [result["validation"]["metrics"]["10"] for result in results]
        ranking.append({"settings": asdict(settings),
                        "mean_fully_correct_recall": sum(row["fully_correct_percent"] for row in metrics) / 2,
                        "mean_correct_player_f1": sum(row["correct_player_f1"] for row in metrics) / 2})
    return sorted(ranking, key=lambda row: (-row["mean_fully_correct_recall"], -row["mean_correct_player_f1"]))


def summarise_seeds(candidates: list[Settings], root: Path) -> dict:
    """Describe completed five-seed validation results without selecting on test."""
    if sorted(value.seed for value in candidates) != list(range(5)):
        raise ValueError("Summary requires the frozen five-seed completion stage")
    results = []
    for settings in candidates:
        result = read_json(root / run_name(settings) / "result.json.gz")
        if not result["complete"] or result["settings"] != asdict(settings):
            raise ValueError("All five seeds must complete the unchanged selected protocol")
        results.append(result)
    metrics = [result["validation"]["metrics"]["10"] for result in results]
    summary = {}
    for name in metrics[0]:
        values = [row[name] for row in metrics]
        summary[name] = {"mean": mean(values), "standard_deviation": stdev(values)}
    return {"settings": asdict(replace(candidates[0], seed=0)), "seeds": list(range(5)),
            "validation_metrics": summary, "best_cycles": [result["best_cycle"] for result in results]}


def run_queue(candidates: list[Settings], cache: Path, root: Path, *, limit: int, device: str,
              workers: int) -> list[dict]:
    from .engine import fit

    completed = []
    for settings in candidates:
        output = root / run_name(settings)
        result_path = output / "result.json.gz"
        if result_path.exists():
            result = read_json(result_path)
            if result["settings"] != asdict(settings) or not result["complete"]:
                raise ValueError(f"Existing run has a different or incomplete protocol: {output}")
            continue
        completed.append(fit(cache, output, settings, device=device, workers=workers,
                             resume=(output / "recovery.pt").exists()))
        if len(completed) == limit:
            break
    return completed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "rank", "summarise"))
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-size-for-length", action="append", metavar="FRAMES=BATCH",
                        help="Repeat for profiled discovery200 clip lengths; unspecified lengths use --config")
    parser.add_argument("--selection", type=Path, help="selection.json.gz from the preceding rank")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1, help="Maximum new runs; never launches an entire grid implicitly")
    args = parser.parse_args(argv)
    batch_by_length = None
    if args.batch_size_for_length is not None:
        batch_by_length = {}
        for entry in args.batch_size_for_length:
            try:
                length, batch = (int(value) for value in entry.split("="))
            except ValueError:
                parser.error("--batch-size-for-length expects integer FRAMES=BATCH")
            if length in batch_by_length:
                parser.error(f"Repeated batch override for clip length {length}")
            batch_by_length[length] = batch
    selection = read_json(args.selection) if args.selection else None
    candidates = stage_candidates(args.stage, load_settings(args.config), selection, batch_by_length=batch_by_length)
    args.runs.mkdir(parents=True, exist_ok=True)
    if args.action == "plan":
        path = args.runs / f"{args.stage}-plan.json.gz"
        write_json(path, {"stage": args.stage, "candidates": [asdict(value) for value in candidates]})
        print(path)
    elif args.action == "rank":
        ranking = rank_candidates(candidates, args.runs)
        path = args.runs / f"{args.stage}-selection.json.gz"
        write_json(path, {**ranking[0], "stage": args.stage, "ranking": ranking})
        print(path)
    elif args.action == "summarise":
        if args.stage != "complete":
            parser.error("summarise requires --stage complete")
        path = args.runs / "five-seed-validation.json.gz"
        write_json(path, summarise_seeds(candidates, args.runs))
        print(path)
    else:
        if args.cache is None or args.limit < 1:
            parser.error("run requires --cache and --limit >=1")
        print(run_queue(candidates, args.cache, args.runs, limit=args.limit, device=args.device, workers=args.workers))


if __name__ == "__main__":
    main()
