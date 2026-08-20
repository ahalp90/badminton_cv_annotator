# VLM cleanup experiment tools

These scripts build, run, and score small VLM trials without changing the
production annotator. The model sees only `cases/inference/manifest.json` and
its clips. Human truth stays under `cases/scoring/` and is not mounted into the
model container.

The retained code is the reusable part of the PR 80 follow-up. Old run folders,
machine paths, logs, caches, and session notes are deliberately absent.

For the human-readable experiment history, start with
[`../experiments.md`](../experiments.md). The exact prompt variants are indexed
in [`../prompts.md`](../prompts.md).

## What is here

- `build_trials.py`: balanced contact-timing cases and broadcast controls.
- `build_track_trials.py`: marked tracker-path checks.
- `run_trials.py`: one resident model and an immutable JSON result per case.
- `score_trials.py`: candidate-level scoring with completeness checks.
- `evaluate_rally_cleanup.py`: replays retained contacts through the normal
  rally boundary.
- `analyse_event_consensus.py`: compares the two model decisions.
- `analyse_broadcast_priors.py`: measures existing scene signals.
- `backends/`: the exact pinned PR 80 adapters made local to this experiment.
- `signals.md`: useful annotator fields for routing and prompts.
- `next_experiment.md`: the next test after the binary contact model exists.
- `results/summary.json`: compact results from the completed bounded trials.

## Requirements

Case building and scoring use the repository's normal Python environment. VLM
inference needs Linux, Apptainer, an NVIDIA GPU, a model-specific image, and a
Python environment inside that image. The adapters download their exact pinned
Hugging Face revisions unless they are already cached.

Run from the repository root:

```bash
export PYTHONPATH="$PWD/scratch/vlm_pr80_eval:$PWD/src"
```

Set your own data locations. The names below are examples, not expected paths:

```bash
export VLM_WORK_ROOT=/path/to/vlm-work
export ANNOTATOR_ARTIFACTS=/path/to/annotator-artifacts
export SCENE_LABELS=/path/to/scene-labels
export TRACK_REVIEW=/path/to/human_visual_review.csv.gz
```

Every output directory must be new. The builders and runner use exclusive
writes so that a retry cannot silently mix with an earlier result.

## Build a marked tracker trial

The known hallucinations are the negative group. The positive group contains
orientation controls near ShuttleSet contacts. Those controls are not
human-labelled real tracker paths, so keep the two groups separate when
reporting results.

```bash
python -m experiments.build_track_trials \
  --artifacts-root "$ANNOTATOR_ARTIFACTS" \
  --repo-root "$PWD" \
  --scene-labels-dir "$SCENE_LABELS" \
  --review "$TRACK_REVIEW" \
  --out "$VLM_WORK_ROOT/tracker/cases" \
  --expected-negative-cases 12 \
  --positive-cases 12 \
  --slow-target \
  --zoom-target
```

`--clean-target-replay` creates the conservative counterfactual. It shows clean
target pixels first and the same pixels with the marker second. It rejected too
many controls in the completed trial and is retained only for comparison.

## Build contact and broadcast controls

```bash
python -m experiments.build_trials \
  --artifacts-root "$ANNOTATOR_ARTIFACTS" \
  --repo-root "$PWD" \
  --scene-labels-dir "$SCENE_LABELS" \
  --out "$VLM_WORK_ROOT/balanced/cases" \
  --event-cases 60 \
  --event-source filtered_contacts \
  --broadcast-cases 12 \
  --dense-broadcast-target
```

Use `--event-span VIDEO:SPAN` for a complete-rally replay. The builder records
input hashes and writes truth separately from inference inputs.

## Run a model

The launchers have no host-specific defaults. Supply the paths for the current
GPU machine:

```bash
export VLM_QWEN_IMAGE=/path/to/qwen.sif
export VLM_QWEN_ENV_ROOT=/path/to/qwen-python-environment
export VLM_QWEN_PYTHON=/path/in/container/to/python
export VLM_HF_CACHE=/path/to/huggingface-cache

scratch/vlm_pr80_eval/experiments/run_qwen_trials_remote.sh \
  "$VLM_WORK_ROOT/tracker/cases/inference/manifest.json" \
  "$VLM_WORK_ROOT/tracker/attempts/qwen-01" \
  --arm video-only
```

For InternVideo3 set `VLM_INTERN_IMAGE`, `VLM_INTERN_ENV_ROOT`,
`VLM_INTERN_PYTHON`, `VLM_INTERN_FFMPEG_PREFIX`, and `VLM_HF_CACHE`. Then use
`run_intern_trials_remote.sh` with the same arguments.

Both launchers:

- refuse to start while the GPU has a compute process;
- stop after 25 minutes;
- mount the package and manifest read-only;
- record the GPU process list on exit;
- keep model caches and temporary files under `VLM_WORK_ROOT`.

Run chunky jobs inside `tmux` on a shared machine.

## Score the result

Scoring runs outside the model container:

```bash
python -m experiments.score_trials \
  --manifest "$VLM_WORK_ROOT/tracker/cases/inference/manifest.json" \
  --truth "$VLM_WORK_ROOT/tracker/cases/scoring/truth.json" \
  --attempts "$VLM_WORK_ROOT/tracker/attempts/qwen-01" \
  --expected-backend qwen3-vl \
  --expected-arm video-only \
  --out "$VLM_WORK_ROOT/tracker/scores/qwen-01.json"
```

Treat a score as usable only when `complete` and `parse_complete` are true.
The scorer reports invalid, missing, and unexpected attempts instead of hiding
them.

For a complete-rally replay, use `evaluate_rally_cleanup.py` after candidate
scoring. Its main comparison is exact contact count, alternating attribution,
point outcome, and structurally usable rallies. Candidate precision alone is
diagnostic.

## Finished result and next run

The completed bounded measurements are summarised in `../results.md` and
`results/summary.json`. Do not rerun them merely to recreate old logs. The next
worthwhile run is the contact-detector comparison in `next_experiment.md`.
