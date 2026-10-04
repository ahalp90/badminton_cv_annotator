"""Rerun the selected new-court model on ShuttleSet22 and keep its player-pick state.

The refit evaluation saved each video's final stream but not the per-frame player
picks behind it. This reruns the same base bundle on the same inputs, checks that
the rerun stream equals the saved one, and keeps the picks for the closeout's
input-state tables. The refit run directory is only read.

Run from the refit run directory, with the run's own checkout first on the path
and the environment recorded in its ``run.json``:

    PYTHONPATH=code:code/src "$RUN_PYTHON" recompute_player_state.py --run . --output "$WORK/player_state"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

THREADS_PER_WORKER = 4
FEATURE_FIELDS = (
    "frame", "interval_id", "pose_valid_top_t+0", "pose_valid_bot_t+0",
    "wrist_valid_top_t+0", "wrist_valid_bot_t+0", "shuttle_visible_t+0",
)


def rerun_video(run: Path, output: Path, identity: str) -> dict[str, object]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(THREADS_PER_WORKER)
    import numpy as np

    from annotator.models import load_models
    from annotator.run_video import RunCapture
    from dataset_builder.vision import load_json_gz
    from experiments.annotator.good_court_refit import runner
    from experiments.annotator.old_court_regression import retrain
    from experiments.annotator.old_court_regression.old_inputs import run_old_video

    started = time.perf_counter()
    config = runner.load_config(run / "inputs.json", run / "output")
    development = retrain.load_development(runner.settings_for(config, config.output / "shared"))
    context = runner.RefitContext(config, development, runner.development_resolutions(config, development))
    models = load_models(config.output / "base" / "bundle")
    inputs = runner.load_video_inputs(context, "test", identity)
    capture = RunCapture()
    run_old_video(inputs, models, capture)
    prediction, evidence = capture.hybrid, capture.contact_evidence
    if prediction is None or evidence is None:
        raise ValueError(f"{identity}: the rerun did not capture its prediction and evidence")

    # A JSON round trip gives the rerun stream the same float text as the saved file.
    rerun = json.loads(json.dumps(retrain.stream_payload(prediction.refined.sequences, prediction.refined.events)))
    saved = load_json_gz(config.output / "base" / "eval" / "test" / identity / "stream.json.gz")
    same_stream = rerun["events"] == saved["events"] and rerun["sequences"] == saved["sequences"]

    rows = prediction.features.rows
    tracker = np.zeros(len(inputs.track), dtype=bool)
    for start, end in evidence.tracker_intervals:
        tracker[start:end] = True
    np.savez_compressed(
        output / f"{identity}.npz",
        picks=np.asarray(evidence.sticky.picks, dtype=np.int16),  # one row per frame: far (Top), near (Bot)
        tracker_covered=tracker,
        court_present=np.asarray(inputs.court_present, dtype=bool),
        exclusion_mask=np.asarray(evidence.exclusion_mask, dtype=bool),
        **{name.replace("+", "p"): rows[name] for name in FEATURE_FIELDS},
    )
    summary = {
        "video": identity, "frames": len(inputs.track), "feature_rows": len(rows),
        "same_stream_as_saved": same_stream, "events": len(rerun["events"]),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (output / f"{identity}.json").write_text(json.dumps(summary) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    arguments.output.mkdir(parents=True, exist_ok=True)
    identities = sorted((path.name for path in (run / "output/base/eval/test").iterdir()), key=int)
    assert len(identities) == 46 and "15" not in identities
    pending = [identity for identity in identities[:arguments.limit]
               if not (arguments.output / f"{identity}.json").is_file()]
    with ProcessPoolExecutor(arguments.jobs) as pool:
        futures = {pool.submit(rerun_video, run, arguments.output, identity): identity for identity in pending}
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    summaries = [json.loads((arguments.output / f"{identity}.json").read_text())
                 for identity in identities[:arguments.limit]]
    mismatched = [row["video"] for row in summaries if not row["same_stream_as_saved"]]
    print(json.dumps({"videos": len(summaries), "stream_mismatches": mismatched}), flush=True)


if __name__ == "__main__":
    main()
