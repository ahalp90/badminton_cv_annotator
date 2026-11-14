"""Two-seed recipe comparisons with separate supplied and discovery LR searches."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev

from experiments.annotator.video_models_benchmark.data import read_json, write_json

from .config import Settings, load_settings

STAGES = (
    "supplied200-lr", "supplied200-recipe", "supplied800",
    "discovery200-lr", "discovery200-recipe", "discovery200-length", "discovery800-length", "complete",
)
FIRST_STAGES = ("supplied200-lr", "discovery200-lr")
LENGTH_STAGES = ("discovery200-length", "discovery800-length")
SEARCHED_FIELDS = ("mode", "backbone", "clip_length", "learning_rate", "augmentation", "mixup")
PREDECESSORS = {
    "supplied200-recipe": "supplied200-lr",
    "supplied800": "supplied200-recipe",
    "discovery200-recipe": "discovery200-lr",
    "discovery200-length": "discovery200-recipe",
    "discovery800-length": "discovery200-length",
}
COMPLETION_WINNERS = {
    ("supplied", "200mf"): "supplied200-recipe",
    ("supplied", "800mf"): "supplied800",
    ("discovery", "200mf"): "discovery200-length",
    ("discovery", "800mf"): "discovery800-length",
}


def run_name(settings: Settings) -> str:
    return (f"{settings.mode}-{settings.backbone}-clip{settings.clip_length}-{settings.augmentation}"
            f"-lr{settings.learning_rate:g}-mixup{int(settings.mixup)}-seed{settings.seed}")


def stage_candidates(stage: str, base: Settings, selection: dict | None = None, *,
                     batch_by_length: dict[int, int] | None = None) -> list[Settings]:
    """Generate one agreed comparison; recipe stages reuse the full user protocol."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    if batch_by_length is not None:
        if stage not in LENGTH_STAGES:
            raise ValueError("Per-length batch overrides apply only to discovery length stages")
        if any(length not in (250, 500) or batch < 1 for length, batch in batch_by_length.items()):
            raise ValueError("Per-length batches require clip length 250 or 500 and a positive batch size")
    batch_by_length = batch_by_length or {}
    if stage in FIRST_STAGES:
        if selection is not None:
            raise ValueError("Each mode starts a fresh LR search without a preceding selection")
        mode, length = ("supplied", 100) if stage == "supplied200-lr" else ("discovery", 250)
        choices = [replace(base, backbone="200mf", mode=mode, clip_length=length,
                           augmentation="user", mixup=False, learning_rate=rate) for rate in (0.001, 0.0003)]
    else:
        if selection is None:
            raise ValueError("This stage needs an explicit frozen selection from the preceding stage")
        selected = Settings(**selection["settings"])
        changed = [name for name in ("ignore_contact_halo", "checkpoint_metric", "selection_tolerance")
                   if getattr(selected, name) != getattr(base, name)]
        if changed:
            raise ValueError("Selection used a different protocol: " + ", ".join(changed))
        predecessor = (COMPLETION_WINNERS[(selected.mode, selected.backbone)] if stage == "complete"
                       else PREDECESSORS[stage])
        if selection["stage"] != predecessor:
            raise ValueError(f"{stage} requires a selection from {predecessor}, got {selection['stage']}")
        if selected.augmentation not in ("user", "e2e"):
            raise ValueError("The hybrid pilot is separate from user/E2E recipe comparisons")
        inherited = replace(base, **{name: getattr(selected, name) for name in SEARCHED_FIELDS}, seed=selected.seed)
        if stage == "complete" or stage.endswith("-recipe"):
            conflicts = [name for name, value in asdict(inherited).items() if value != getattr(selected, name)]
            if conflicts:
                raise ValueError("Reuse requires the full selected protocol; conflicting config fields: "
                                 + ", ".join(conflicts))
        if stage == "complete":
            return [replace(selected, seed=seed) for seed in range(5)]
        mode = "supplied" if stage.startswith("supplied") else "discovery"
        if (selected.mode, selected.backbone) != (mode, "200mf"):
            raise ValueError("Selection is from the wrong preceding mode/backbone")
        if stage.endswith("-recipe"):
            if selected.augmentation != "user" or selected.mixup:
                raise ValueError("Recipe comparison requires the winning user-recipe LR")
            choices = [selected, replace(selected, augmentation="e2e", mixup=True)]
        elif stage == "supplied800":
            choices = [replace(inherited, backbone="800mf", clip_length=100)]
        else:
            choices = []
            for length in (250, 500):
                batch = batch_by_length.get(length, max(1, base.batch_size * base.clip_length // length))
                choices.append(replace(inherited, backbone="800mf" if stage == "discovery800-length" else "200mf",
                                       clip_length=length, batch_size=batch,
                                       validation_batch_size=max(1, base.validation_batch_size * base.clip_length // length)))
            if stage == "discovery200-length":
                # Reuse the selected 250-frame runs; only the 500-frame pair is new.
                reference = replace(choices[0], seed=selected.seed)
                if reference != selected:
                    raise ValueError("Discovery length comparison must preserve the selected 250-frame protocol")
    return [replace(choice, seed=seed) for choice in choices for seed in range(2)]


def rank_candidates(candidates: list[Settings], root: Path) -> list[dict]:
    groups: dict[Settings, list[dict]] = {}
    for settings in candidates:
        if settings.seed not in (0, 1):
            continue
        path = root / run_name(settings) / "result.json.gz"
        if not path.exists():
            raise ValueError(f"Missing completed validation result: {path}")
        result = read_json(path)
        if not result["complete"] or Settings(**result["settings"]) != settings:
            raise ValueError(f"Incomplete or changed protocol: {path}")
        key = replace(settings, seed=0)
        groups.setdefault(key, []).append(result)
    ranking = []
    for settings, results in groups.items():
        if sorted(result["settings"]["seed"] for result in results) != [0, 1]:
            raise ValueError("Each ranked configuration needs exactly seeds 0 and 1")
        metrics = [result["validation"]["metrics"][str(settings.selection_tolerance)] for result in results]
        rally_metric = "fully_correct_f1" if settings.checkpoint_metric == "rally_f1" else "fully_correct_percent"
        ranking.append({"settings": asdict(settings),
                        "rally_metric": rally_metric,
                        "mean_rally_score": mean(row[rally_metric] for row in metrics),
                        "mean_correct_player_f1": sum(row["correct_player_f1"] for row in metrics) / 2})
    return sorted(ranking, key=lambda row: (-row["mean_rally_score"], -row["mean_correct_player_f1"]))


def summarise_seeds(candidates: list[Settings], root: Path) -> dict:
    """Describe completed five-seed validation results without selecting on test."""
    if sorted(value.seed for value in candidates) != list(range(5)):
        raise ValueError("Summary requires the frozen five-seed completion stage")
    results = []
    for settings in candidates:
        result = read_json(root / run_name(settings) / "result.json.gz")
        if not result["complete"] or Settings(**result["settings"]) != settings:
            raise ValueError("All five seeds must complete the unchanged selected protocol")
        results.append(result)
    metrics = [result["validation"]["metrics"][str(candidates[0].selection_tolerance)] for result in results]
    summary = {}
    for name in metrics[0]:
        values = [row[name] for row in metrics]
        summary[name] = {"mean": mean(values), "standard_deviation": stdev(values)}
    return {"settings": asdict(replace(candidates[0], seed=0)), "seeds": list(range(5)),
            "validation_metrics": summary, "best_cycles": [result["best_cycle"] for result in results]}


def run_queue(candidates: list[Settings], cache: Path | None, root: Path, *, limit: int, device: str,
              workers: int, cache_by_augmentation: dict[str, Path] | None = None) -> list[dict]:
    from .engine import fit

    completed = []
    for settings in candidates:
        output = root / run_name(settings)
        result_path = output / "result.json.gz"
        if result_path.exists():
            result = read_json(result_path)
            if Settings(**result["settings"]) != settings or not result["complete"]:
                raise ValueError(f"Existing run has a different or incomplete protocol: {output}")
            continue
        selected_cache = (cache_by_augmentation or {}).get(settings.augmentation, cache)
        if selected_cache is None:
            raise ValueError(f"No cache supplied for the {settings.augmentation} recipe")
        completed.append(fit(selected_cache, output, settings, device=device, workers=workers,
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
                        help="Discovery length stages: override scaled training batches, e.g. 250=8 and 500=4")
    parser.add_argument("--selection", type=Path, help="selection.json.gz from the preceding rank")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="Single-recipe cache fallback")
    parser.add_argument("--user-cache", type=Path, help="Fresh letterbox256 JPEG cache for the user recipe")
    parser.add_argument("--e2e-cache", type=Path, help="Height224 aspect-preserving JPEG cache for the E2E recipe")
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
        if not any((args.cache, args.user_cache, args.e2e_cache)) or args.limit < 1:
            parser.error("run requires cache paths and --limit >=1")
        caches = {name: path for name, path in (("user", args.user_cache), ("e2e", args.e2e_cache)) if path is not None}
        print(run_queue(candidates, args.cache, args.runs, limit=args.limit, device=args.device,
                        workers=args.workers, cache_by_augmentation=caches))


if __name__ == "__main__":
    main()
