"""Collect the November stride-three runs without weights or frame probabilities.

Run with PYTHONPATH=src:. and --source pointing to the downloaded benchmark
scratch directory. Saved frame probabilities are needed only during collection.
"""
from __future__ import annotations

import argparse
import csv
import gzip
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from annotator.types import ContactEvent, ContactSequence, CourtHalf
from experiments.annotator.video_models_benchmark.data import read_json, write_json
from experiments.annotator.video_models_benchmark.evaluate import CLASS_SIDES
from experiments.annotator.video_models_benchmark.scoring import (
    COUNT_NAMES,
    TOLERANCES,
    add_rates,
    score_video_rules,
)
from experiments.annotator.video_models_benchmark.writeup_scripts.rescore_videomae import (
    earliest_minimum,
    read_probabilities,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "stride3_20261105"
QUEUES = {
    "x3d": "x3d-full-stride3-20261105",
    "videomae": "videomae-after-x3d-20261105",
    "x3d_extra": "x3d-winner-seeds234-20261107",
}


def public_records(value: Any) -> Any:
    """Keep source filenames and run-relative paths without machine-specific roots."""
    if isinstance(value, dict):
        return {key: public_records(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_records(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        parts = Path(value).parts
        if "video-models" in parts:
            return "/".join(parts[parts.index("video-models") + 1:])
        return Path(value).name
    return value


def collect_predictions(job: dict) -> list[dict]:
    """Save both hitter labels per hit and verify their aggregate scores."""
    saved = read_json(job["events"])
    assert saved["hitter_rule"] == "forced_alternation_probability_ties"
    assert (saved["sampling"]["stride"], saved["sampling"]["offset"]) == (3, 0)
    probabilities = read_probabilities(job["probabilities"], job["videos"])
    totals = {}
    for rule in ("raw", "alternating"):
        totals[rule] = {str(tolerance): dict.fromkeys(COUNT_NAMES, 0) for tolerance in TOLERANCES}
    videos = []
    offset = 0
    for video in job["videos"]:
        sequences = {"raw": [], "alternating": []}
        saved_sequences = saved["streams"][video["item_id"]]["sequences"]
        compact_sequences = []
        for span, sequence in zip(video["spans"], saved_sequences, strict=True):
            for key in ("span_id", "start_frame", "end_frame"):
                assert span[key] == sequence[key]
            compact_events, raw_events, alternating_events = [], [], []
            for frame, confidence, forced_side in sequence["events"]:
                assert span["start_frame"] <= frame < span["end_frame"] and frame % 3 == 0
                peak = probabilities[offset + frame - span["start_frame"]]
                predicted_class = int(peak.argmax())
                raw_side = CLASS_SIDES[predicted_class]
                assert float(peak[predicted_class]) == confidence
                compact_events.append([frame, confidence, raw_side.value, forced_side])
                raw_events.append(ContactEvent(frame, confidence, raw_side))
                alternating_events.append(ContactEvent(frame, confidence, CourtHalf(forced_side)))
            for rule, events in (("raw", raw_events), ("alternating", alternating_events)):
                sequences[rule].append(ContactSequence(
                    span["span_id"], span["start_frame"], span["end_frame"], tuple(events),
                ))
            compact_sequences.append({
                "span_id": span["span_id"], "rally_id": span["rally_id"],
                "start_frame": span["start_frame"], "end_frame": span["end_frame"],
                "events": compact_events,
            })
            offset += span["end_frame"] - span["start_frame"]
        videos.append({"item_id": video["item_id"], "fps": video["fps"], "sequences": compact_sequences})
        for tolerance in TOLERANCES:
            for rule, counts in score_video_rules(video, sequences, tolerance).items():
                for name in COUNT_NAMES:
                    totals[rule][str(tolerance)][name] += counts[name]
    assert offset == len(probabilities)
    metrics, rows = {}, []
    for rule, scores in totals.items():
        label = "raw" if rule == "raw" else "forced_alternation"
        metrics[label] = {}
        for tolerance, counts in scores.items():
            if rule == "alternating":
                for name in COUNT_NAMES:
                    assert counts[name] == saved["metrics"][tolerance][name], (job["label"], tolerance, name)
            rates = add_rates(counts)
            metrics[label][tolerance] = rates
            rows.append({**job["label"], "hitter_rule": label, "tolerance": int(tolerance), **rates})
    destination = job["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, {
        **job["label"], "sampling": saved["sampling"], "frame_metrics": job["frame_metrics"],
        "event_columns": ["frame", "raw_confidence", "raw_hitter", "forced_alternation_hitter"],
        "metrics": metrics, "videos": videos,
    })
    print(f"Verified {job['label']['model']} seed {job['label']['seed']} {job['label']['split']}", flush=True)
    return rows


def collect_runs(source: Path) -> tuple[dict, dict, list[dict]]:
    """Retain every checkpoint summary and use the already-frozen winners."""
    records, selections, winners = {}, {}, []
    for model, queue in QUEUES.items():
        directory = source / queue
        manifest = read_json(directory / "grid.json.gz")
        assert read_json(directory / "grid-status.json.gz")["phase"] == "complete"
        runs = []
        for run in manifest["runs"]:
            training = directory / run["name"]
            assert read_json(training / "status.json.gz")["phase"] == "complete"
            history = read_json(training / "history.json.gz")
            best = earliest_minimum(history)
            runs.append({
                **run, "settings": read_json(training / "settings.json.gz"),
                "continuation": read_json(training / "continuation.json.gz"), "history": history,
                "best_step": best["steps"], "final_step": history[-1]["steps"],
            })
        records[model] = {"source_queue": queue, "manifest": manifest, "runs": runs}
        if model == "x3d_extra":
            assert manifest["x3d_winner"] == selections["x3d"]["ranking"][0]
            selected = manifest["runs"]
        else:
            selections[model] = read_json(directory / "selection.json.gz")
            selected = selections[model]["ranking"][0]["runs"]
        for run in selected:
            winners.append({"model": "x3d" if model == "x3d_extra" else model, "queue": queue, **run})
    assert [len(records[model]["runs"]) for model in QUEUES] == [16, 4, 3]
    assert sorted(run["seed"] for run in winners if run["model"] == "x3d") == list(range(5))
    assert sorted(run["seed"] for run in winners if run["model"] == "videomae") == [0, 1]
    return records, selections, winners


def prediction_jobs(source: Path, output: Path, winners: list[dict], data: dict) -> list[dict]:
    jobs = []
    for run in winners:
        training = source / run["queue"] / run["name"]
        best = earliest_minimum(read_json(training / "history.json.gz"))
        if "best_step" in run:
            assert run["best_step"] == best["steps"]
        for split in ("validation", "test"):
            directory = training if split == "validation" else source / run["queue"] / "test" / f"seed{run['seed']}"
            events = directory / ("validation_events.json.gz" if split == "validation" else "events.json.gz")
            if split == "validation":
                assert read_json(events)["metrics"] == best["events"]
                frame_metrics = best["validation"]
            else:
                status = read_json(directory / "status.json.gz")
                assert status["phase"] == "complete" and Path(status["checkpoint"]).parent.name == run["name"]
                frame_metrics = read_json(directory / "frame_metrics.json.gz")
            videos = data["development" if split == "validation" else "test"]["videos"]
            jobs.append({
                "label": {"model": run["model"], "run": run["name"], "seed": run["seed"],
                          "split": split, "best_step": best["steps"]},
                "videos": [video for video in videos if video["role"] == split],
                "events": events, "frame_metrics": frame_metrics,
                "probabilities": directory / (
                    "validation_probabilities.npy.xz" if split == "validation" else "probabilities.npy.xz"
                ),
                "destination": output / "predictions" / run["model"] / f"seed{run['seed']}" / f"{split}.json.gz",
            })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    records, selections, winners = collect_runs(args.source)
    data = {
        "development": read_json(args.source / "cache-indexes/rgb256/index.json.gz"),
        "test": read_json(args.source / "cache-indexes/test-rgb256/index.json.gz"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = prediction_jobs(args.source, args.output, winners, data)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for result in pool.map(collect_predictions, jobs) for row in result]
    write_json(args.output / "runs.json.gz", public_records(records))
    write_json(args.output / "selection.json.gz", selections)
    write_json(args.output / "data.json.gz", public_records(data))
    with gzip.open(args.output / "scores.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected {len(winners)} winners, {len(jobs)} prediction files and {len(rows)} score rows.")


if __name__ == "__main__":
    main()
