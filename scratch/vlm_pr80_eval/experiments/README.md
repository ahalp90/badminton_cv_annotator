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
- `build_multiscale_trials.py`: builds paired 90- and 120-second cut-aware
  storyboards from suspected spans.
- `run_multiscale_trials.py` and `score_multiscale_trials.py`: run and score
  the broad segment-reading prompt.
- `build_detail_from_context.py`: builds identical 120-frame close clips for
  the three prompt arms. It can also build them without a broad VLM pass.
- `run_detail_trials.py` and `score_detail_trials.py`: run and score all detail
  arms, or only named arms with repeated `--only-arm` arguments.
- `combined_visual_trials.py`: reproduces the failed 80-frame broad plus
  120-frame close input.
- `replay_pair_trials.py`: builds, runs, and scores the failed 120-frame
  earlier-action plus 120-frame target comparison.
- `replay_pair_pilot_ids.txt`: the frozen requested cases for that bounded
  trial.
- `replay_pair_remote.sh`: GPU-safe launcher for either pinned backend.
- `analyse_detail_routes.py`: compares detail arms and simple two-model rules.
- `backends/`: the exact pinned PR 80 adapters made local to this experiment.
- `signals.md`: useful annotator fields for routing and prompts.
- `next_experiment.md`: the multiscale protocol and its current status.
- `contact_model_followup.md`: the later contact-model and VLM comparison.
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
- keep the older bounded trials under their recorded 25-minute limit;
- mount the package and manifest read-only;
- record the GPU process list on exit;
- keep model caches and temporary files under `VLM_WORK_ROOT`.

The multiscale launchers have no time cap. They still run one recorded job at
a time and record the empty GPU state when they finish.

Run chunky jobs inside `tmux` on a shared machine.

For the multiscale work, each job must write `status.json` and a final `DONE`
or `FAILED` marker. The main session must not repeatedly poll the remote host.
Use one blocking `tmux wait-for` watcher, preferably owned by a cheap Luna
agent, or make one delayed marker check near the expected finish time. See
[`next_experiment.md`](next_experiment.md#remote-runs-without-repeated-polling).

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

## Reproduce the direct replay-pair trial

The builder selects an earlier reference using only automatic span order and
signals. Human scene truth is read only by the scorer.

```bash
python -m experiments.replay_pair_trials build \
  --detail-manifest "$VLM_WORK_ROOT/detail/inference/short_only/manifest.json" \
  --source-video sset_01=/path/to/sset_01.avi \
  --source-video sset_15=/path/to/sset_15.avi \
  --source-video sset_21=/path/to/sset_21.avi \
  --case-id-file scratch/vlm_pr80_eval/experiments/replay_pair_pilot_ids.txt \
  --out "$VLM_WORK_ROOT/replay-pair/cases"

scratch/vlm_pr80_eval/experiments/replay_pair_remote.sh \
  internvideo3 \
  "$VLM_WORK_ROOT/replay-pair/cases/inference/manifest.json" \
  "$VLM_WORK_ROOT/replay-pair/attempts/intern"

python -m experiments.replay_pair_trials score \
  --manifest "$VLM_WORK_ROOT/replay-pair/cases/inference/manifest.json" \
  --attempts "$VLM_WORK_ROOT/replay-pair/attempts/intern" \
  --backend internvideo3 \
  --parent-score "$VLM_WORK_ROOT/detail/scores/intern.json" \
  --truth "$VLM_WORK_ROOT/detail/scoring/truth.json" \
  --out "$VLM_WORK_ROOT/replay-pair/scores/intern.json"
```

Keep every path under a new run directory. The public wrapper refuses to mix a
retry with an existing output.

## Current result and next step

The completed bounded measurements are summarised in `../results.md` and
`results/summary.json`. Do not rerun them merely to recreate old logs. The
multiscale work is complete. Its broad, fact-bearing, combined-visual, and
two-model paths all lost to InternVideo3's 120-frame short-only arm. The
463-target wide run then found only 44.7% of material unsafe targets. Stronger
replay wording and the direct replay pair also failed. The scene rule is not a
safe automatic keep rule. The next meaningful comparison follows the frozen
contact-detector output in `contact_model_followup.md`.
