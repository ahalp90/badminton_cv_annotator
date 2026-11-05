"""Copy the head LR 5e-4 trial's records and score it beside the pilot at stride 3, offset 0.

The trial is one extra validation-only run, seed 0 at head learning rate 5e-4. It
validated at stride 3, offset 0, so its saved probabilities hold no-contact
placeholders at the other two thirds of the centres. The pilot's dense validation
probabilities cover the same stride-three centres, which gives each pilot run's
selected checkpoint a matching score.

The script copies the trial's compact records into ``results/videomae_lr5e4/``,
rescores the trial's events under both hitter rules, and computes validation
cross-entropy and macro F1 for every run at stride 3, offset 0, plus dense for
the pilot runs. Frame scores skip the centres two frames from a hit that the
label scheme ignores, as validation loss does.

    PYTHONPATH=src:. python experiments/annotator/video_models_benchmark/writeup_scripts/compare_videomae_lr5e4.py \\
        --trial /path/to/lr5e-4-run --runs /path/to/videomae-runs \\
        --validation-index /path/to/rgb256-cache/index.json.gz
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from experiments.annotator.video_models_benchmark.data import read_json, target_labels
from experiments.annotator.video_models_benchmark.writeup_scripts.rescore_videomae import (
    RESULTS,
    RUN_RECORDS,
    earliest_minimum,
    read_probabilities,
    rescore,
    role_videos,
    validation_jobs,
    write_scores,
)

COMPARISON_SAMPLING = (3, 0)  # (stride, offset): the only sampling the trial's probabilities support
IGNORED = -100  # target_labels' mark for centres that cross-entropy skips
# Validation sums the loss in float32 on the GPU, so a recomputed loss agrees to about
# four significant figures; confusion counts must agree exactly.
LOSS_TOLERANCE = 1e-4


def frame_targets(videos: list[dict], scheme: str) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and video frame numbers, one per validation centre in probability-row order."""
    spans = [span for video in videos for span in video["spans"]]
    labels = np.concatenate([target_labels(span, scheme) for span in spans])
    frames = np.concatenate([np.arange(span["start_frame"], span["end_frame"]) for span in spans])
    return labels, frames


def frame_scores(probabilities: np.ndarray, labels: np.ndarray, scored: np.ndarray) -> dict:
    """Mean cross-entropy, macro F1 and confusion over the scored centres, as validation computes them.

    :param probabilities: (centres, 3); columns are no contact, bottom-player and
        top-player contact
    :param labels: one class per centre, ``IGNORED`` where the label scheme skips it
    :param scored: one flag per centre: on the sampling grid and not ignored
    :return: ``confusion`` has true classes as rows and predicted classes as columns
    """
    targets = labels[scored]
    true_class_probabilities = probabilities[scored, targets].astype(np.float64)
    predicted = probabilities[scored].argmax(axis=1)
    confusion = np.bincount(targets * 3 + predicted, minlength=9).reshape(3, 3)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    class_f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(3), where=denominator > 0)
    return {"scored_centres": int(scored.sum()), "cross_entropy": float(-np.log(true_class_probabilities).mean()),
            "macro_f1": float(class_f1.mean()), "confusion": confusion}


def score_run(job: dict, labels: np.ndarray, frames: np.ndarray) -> list[dict]:
    """Frame scores at the run's evaluated sampling and at stride 3, offset 0.

    The evaluated sampling must reproduce the loss and confusion that training
    recorded for the selected checkpoint; that check confirms the labels, the
    ignored centres and the row order all line up.
    """
    selected = earliest_minimum(read_json(job["probabilities"].parent / "history.json.gz"))["validation"]
    probabilities = read_probabilities(job["probabilities"], job["videos"])
    evaluated = (job["sampling"]["stride"], job["sampling"]["offset"])
    rows = []
    for stride, offset in dict.fromkeys((evaluated, COMPARISON_SAMPLING)):
        on_grid = frames % stride == offset
        counted = labels != IGNORED
        scores = frame_scores(probabilities, labels, on_grid & counted)
        if (stride, offset) == evaluated:
            if scores["confusion"].tolist() != selected["confusion"]:
                raise ValueError(f"{job['label']['run']}: recomputed confusion {scores['confusion'].tolist()} "
                                 f"differs from the recorded {selected['confusion']}")
            if not np.isclose(scores["cross_entropy"], selected["loss"], rtol=LOSS_TOLERANCE, atol=0):
                raise ValueError(f"{job['label']['run']}: recomputed loss {scores['cross_entropy']:.6f} "
                                 f"differs from the recorded {selected['loss']:.6f}")
        rows.append({**job["label"], "stride": stride, "offset": offset, "scored_centres": scores["scored_centres"],
                     "cross_entropy": scores["cross_entropy"], "macro_f1": scores["macro_f1"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", type=Path, required=True, help="Run directory of the LR 5e-4 trial")
    parser.add_argument("--runs", type=Path, required=True, help="VideoMAE pilot queue directory")
    parser.add_argument("--validation-index", type=Path, required=True, help="index.json.gz of the development cache")
    parser.add_argument("--results", type=Path, default=RESULTS, help="Destination for compact records and scores")
    args = parser.parse_args()
    videos = role_videos(args.validation_index, "validation")
    settings = read_json(args.trial / "settings.json.gz")
    # The pilot queue names runs this way, so the trial's records sit beside them consistently.
    name = f"videomae-lr{settings['learning_rate']}-seed{settings['seed']}"
    manifest = {"runs": [{"name": args.trial.name, "learning_rate": settings["learning_rate"],
                          "seed": settings["seed"]}]}
    (trial,) = validation_jobs(args.trial.parent, manifest, videos, None)
    trial["label"]["run"] = name
    if (trial["sampling"]["stride"], trial["sampling"]["offset"]) != COMPARISON_SAMPLING:
        raise ValueError(f"{args.trial}: validated at {trial['sampling']}, not stride 3, offset 0")
    pilot = validation_jobs(args.runs, read_json(args.runs / "grid.json.gz"), videos, None)
    schemes = {read_json(job["probabilities"].parent / "settings.json.gz")["scheme"] for job in [*pilot, trial]}
    if len(schemes) != 1:
        raise ValueError(f"runs use different label schemes {schemes}, so their losses are not comparable")
    labels, frames = frame_targets(videos, schemes.pop())
    destination = args.results / "videomae_lr5e4"
    (destination / name).mkdir(parents=True, exist_ok=True)
    for record in RUN_RECORDS:
        shutil.copy2(args.trial / record, destination / name / record)
    write_scores(destination / "validation_scores.csv.gz", rescore(trial))
    write_scores(destination / "frame_scores.csv.gz",
                 [row for job in [*pilot, trial] for row in score_run(job, labels, frames)])


if __name__ == "__main__":
    main()
