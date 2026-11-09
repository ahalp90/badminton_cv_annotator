"""Copy compact VideoMAE records and rescore saved probabilities at dense and stride-three centres.

Run from the repository root after copying the VideoMAE queue directory from the
training host. The script copies each run's settings and histories into
``results/videomae_pilot/``, then rescores every run's minimum-validation-CE
probabilities. With ``--test-index``, it also rescores both test seeds of the
learning rate in the queue's ``selection.json.gz`` and copies their test scores into
``results/videomae_test/``. Test scores never feed back into selection.

    PYTHONPATH=src:. python experiments/annotator/video_models_benchmark/writeup_scripts/rescore_videomae.py \\
        --runs /path/to/videomae-runs --validation-index /path/to/rgb256-cache/index.json.gz \\
        [--test-index /path/to/test-rgb256-cache/index.json.gz] [--streams /path/to/event-predictions]

Dense inference predicts at every centre frame. Stride three keeps only centres
whose original video frame number satisfies ``frame % 3 == offset``, so the
sampling grid continues across rally boundaries. Raw hitters keep each retained
peak's own most likely class; alternating hitters come from the benchmark's decoder
unchanged. Duplicate suppression, matching and timing tolerances are the shared ones.

For validation or tests evaluated sparsely, the event record's ``sampling`` metadata
restricts rescoring to that measured stride and offset. Background-filled skipped
rows are not dense model predictions and cannot supply other offsets.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import lzma
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from annotator.types import ContactSequence
from experiments.annotator.contact_records import stream_payload
from experiments.annotator.video_models_benchmark.data import read_json, write_json
from experiments.annotator.video_models_benchmark.evaluate import (
    CLASS_SIDES,
    decode_span,
)
from experiments.annotator.video_models_benchmark.scoring import (
    COUNT_NAMES,
    TOLERANCES,
    add_rates,
    score_video_rules,
)

BENCHMARK = Path(__file__).resolve().parents[1]
RESULTS = BENCHMARK / "results"
SAMPLINGS = ((1, 0), (3, 0), (3, 1), (3, 2))  # (stride, offset); dense first
HITTER_RULES = ("raw", "alternating")
NO_CONTACT = (1.0, 0.0, 0.0)
RUN_RECORDS = ("settings.json.gz", "history.json.gz", "continuation.json.gz", "training_progress.json.gz")
TEST_RECORDS = ("events.json.gz", "frame_metrics.json.gz")


def earliest_minimum(history: list[dict]) -> dict:
    """The check whose model ``fit`` saved: strict CE improvements keep the earliest of equal losses."""
    return min(history, key=lambda check: check["validation"]["loss"])


def role_videos(index: Path, role: str) -> list[dict]:
    """Videos in cache-index order, which is the order of every saved probability row."""
    return [video for video in read_json(index)["videos"] if video["role"] == role]


def read_probabilities(path: Path, videos: list[dict]) -> np.ndarray:
    """Load one exhaustive inference pass: (centres, 3) in video, rally and frame order."""
    with lzma.open(path) as stream:
        probabilities = np.load(stream, allow_pickle=False)
    centres = sum(span["end_frame"] - span["start_frame"] for video in videos for span in video["spans"])
    if probabilities.shape != (centres, 3):
        raise ValueError(f"{path}: shape {probabilities.shape}, but the index has {centres} centres")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError(f"{path}: rows are not finite probability distributions")
    return probabilities


def decode_video(video: dict, probabilities: np.ndarray, stride: int, offset: int) -> dict[str, list[ContactSequence]]:
    """Decode one video's rallies using only centres on the stride's video-frame grid.

    :param probabilities: (centres, 3) for this video in rally then frame order;
        columns are no contact, bottom-player contact and top-player contact
    :return: one sequence per rally for each hitter rule; both rules share contact
        frames and scores, because suppression runs before alternation
    """
    sequences: dict[str, list[ContactSequence]] = {"raw": [], "alternating": []}
    row = 0
    for span in video["spans"]:
        centres = np.arange(span["start_frame"], span["end_frame"])
        rally_probabilities = probabilities[row:row + len(centres)]
        sampled = rally_probabilities.copy()
        sampled[centres % stride != offset] = NO_CONTACT
        alternating = decode_span(span, sampled, video["fps"])
        raw_events = []
        for event in alternating.events:
            peak = rally_probabilities[event.frame - span["start_frame"]]
            raw_events.append(replace(event, side=CLASS_SIDES[int(peak.argmax())]))
        sequences["raw"].append(replace(alternating, events=tuple(raw_events)))
        sequences["alternating"].append(alternating)
        row += len(centres)
    if row != len(probabilities):
        raise ValueError(f"{video['item_id']}: {len(probabilities)} rows for {row} centres")
    return sequences


def check_saved_counts(totals: dict, saved_metrics: dict, name: str) -> None:
    """Rescoring at the evaluated sampling must reproduce its saved event counts."""
    for tolerance in TOLERANCES:
        for count in COUNT_NAMES:
            rescored, saved = totals["alternating", tolerance][count], saved_metrics[str(tolerance)][count]
            if rescored != saved:
                raise ValueError(f"{name}: evaluated-sampling {count} at ±{tolerance} is {rescored}, saved {saved}")


def rescore(job: dict) -> list[dict]:
    """Score one probability array under every sampling and hitter rule.

    :param job: ``label`` columns for each output row, ``videos`` in index order,
        the ``probabilities`` path, ``saved_metrics`` from the same inference pass
        and an optional ``streams`` directory for decoded event predictions
    """
    videos = job["videos"]
    probabilities = read_probabilities(job["probabilities"], videos)
    lengths = [sum(span["end_frame"] - span["start_frame"] for span in video["spans"]) for video in videos]
    rows = []
    sampling = job.get("sampling", {"stride": 1, "offset": 0})
    evaluated = (sampling["stride"], sampling["offset"])
    conditions = SAMPLINGS if evaluated == (1, 0) else (evaluated,)
    for stride, offset in conditions:
        totals = {(rule, tolerance): dict.fromkeys(COUNT_NAMES, 0) for rule in HITTER_RULES for tolerance in TOLERANCES}
        streams: dict[str, dict] = {rule: {} for rule in HITTER_RULES}
        start = 0
        for video, length in zip(videos, lengths, strict=True):
            sequences = decode_video(video, probabilities[start:start + length], stride, offset)
            for rule, items in sequences.items():
                events = [event for item in items for event in item.events]
                streams[rule][video["item_id"]] = stream_payload(items, events)
            for tolerance in TOLERANCES:
                for rule, counts in score_video_rules(video, sequences, tolerance).items():
                    for count in COUNT_NAMES:
                        totals[rule, tolerance][count] += counts[count]
            start += length
        if (stride, offset) == evaluated:
            check_saved_counts(totals, job["saved_metrics"], job["label"]["run"])
        for (rule, tolerance), counts in totals.items():
            rows.append({**job["label"], "stride": stride, "offset": offset, "hitter_rule": rule,
                         "tolerance": tolerance, **add_rates(counts)})
        if job["streams"] is not None:
            for rule, rule_streams in streams.items():
                metrics = {str(tolerance): add_rates(totals[rule, tolerance]) for tolerance in TOLERANCES}
                path = job["streams"] / f"stride{stride}_offset{offset}_{rule}.json.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, {**job["label"], "stride": stride, "offset": offset, "hitter_rule": rule,
                                  "metrics": metrics, "streams": rule_streams})
    print(f"{job['label']['split']} {job['label']['run']}: saved counts reproduced; {conditions} scored", flush=True)
    return rows


def validation_jobs(runs: Path, manifest: dict, videos: list[dict], streams: Path | None) -> list[dict]:
    jobs = []
    for run in manifest["runs"]:
        directory = runs / run["name"]
        best = earliest_minimum(read_json(directory / "history.json.gz"))
        events = read_json(directory / "validation_events.json.gz")
        saved = events["metrics"]
        # fit writes best.pt, probabilities and events before appending the history row.
        if saved != best["events"]:
            raise ValueError(f"{run['name']}: saved validation outputs are not from the minimum-CE check at "
                             f"{best['steps']} updates; copy the run directory again")
        label = {"split": "validation", "run": run["name"], "learning_rate": run["learning_rate"],
                 "seed": run["seed"], "best_step": best["steps"]}
        run_streams = None if streams is None else streams / "validation" / run["name"]
        jobs.append({"label": label, "videos": videos, "probabilities": directory / "validation_probabilities.npy.xz",
                     "saved_metrics": saved, "sampling": events.get("sampling", {"stride": 1, "offset": 0}),
                     "streams": run_streams})
    return jobs


def test_jobs(runs: Path, videos: list[dict], streams: Path | None) -> list[dict]:
    """Both seeds of the learning rate the queue selected from validation alone."""
    selected = read_json(runs / "selection.json.gz")["ranking"][0]
    jobs = []
    for run in selected["runs"]:
        best = earliest_minimum(read_json(runs / run["name"] / "history.json.gz"))
        if best["steps"] != run["best_step"]:
            raise ValueError(f"{run['name']}: selection records {run['best_step']} updates, history {best['steps']}")
        directory = runs / "test" / f"seed{run['seed']}"
        if read_json(directory / "status.json.gz")["phase"] != "complete":
            raise ValueError(f"{directory}: test inference has not completed")
        label = {"split": "test", "run": run["name"], "learning_rate": run["learning_rate"],
                 "seed": run["seed"], "best_step": best["steps"]}
        events = read_json(directory / "events.json.gz")
        jobs.append({"label": label, "videos": videos, "probabilities": directory / "probabilities.npy.xz",
                     "saved_metrics": events["metrics"], "sampling": events.get("sampling", {"stride": 1, "offset": 0}),
                     "streams": None if streams is None else streams / "test" / f"seed{run['seed']}"})
    return jobs


def copy_records(runs: Path, manifest: dict, test: list[dict], results: Path) -> None:
    """Copy the compact records that the figures and report read; arrays and weights stay behind."""
    for run in manifest["runs"]:
        destination = results / "videomae_pilot" / run["name"]
        destination.mkdir(parents=True, exist_ok=True)
        for name in RUN_RECORDS:
            shutil.copy2(runs / run["name"] / name, destination / name)
    if (runs / "selection.json.gz").exists():
        shutil.copy2(runs / "selection.json.gz", results / "videomae_pilot" / "selection.json.gz")
    for job in test:
        seed = job["label"]["seed"]
        destination = results / "videomae_test" / f"seed{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        for name in TEST_RECORDS:
            shutil.copy2(runs / "test" / f"seed{seed}" / name, destination / name)


def write_scores(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="VideoMAE queue directory copied from training")
    parser.add_argument("--validation-index", type=Path, required=True, help="index.json.gz of the development cache")
    parser.add_argument("--test-index", type=Path, help="index.json.gz of the test cache; rescores the selected seeds")
    parser.add_argument("--streams", type=Path, help="Optional directory for decoded event predictions")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--results", type=Path, default=RESULTS, help="Destination for compact records and scores")
    args = parser.parse_args()
    manifest = read_json(args.runs / "grid.json.gz")
    jobs = validation_jobs(args.runs, manifest, role_videos(args.validation_index, "validation"), args.streams)
    test = [] if args.test_index is None else test_jobs(args.runs, role_videos(args.test_index, "test"), args.streams)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(rescore, jobs + test))
    copy_records(args.runs, manifest, test, args.results)
    write_scores(args.results / "videomae_stride3" / "validation_scores.csv.gz",
                 [row for rows in results[:len(jobs)] for row in rows])
    if test:
        write_scores(args.results / "videomae_stride3" / "test_scores.csv.gz",
                     [row for rows in results[len(jobs):] for row in rows])


if __name__ == "__main__":
    main()
