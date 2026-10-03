# Reproduce the new-court model comparison

This guide reproduces the comparison that selected **base on 3 October 2026**.
Base uses the existing hit-selection policy. The alternative, veto, discards
possible hits when the shuttle track is flagged as unreliable and no player
has been selected nearby in time. Both were evaluated on eight validation
videos and 46 ShuttleSet22 videos; validation also tested switching the veto
on without retraining. The [evaluation report](model_selection.md) explains why the
small, inconsistent gains favoured keeping base.

The runner starts from saved court-detector outputs, converts them into the
annotator's scene geometry and per-frame court records, then fits and evaluates
the models. It uses the [old-court runner](refit_reproduction.md)'s
parallel stages and resume support.

## Input configuration

[`good_court_refit/inputs.json`](../good_court_refit/inputs.json) is a template
for an untracked working config. Its `null` court paths need explicit per-video
detector results; plain JSON and `.json.gz` are accepted. The shuttle, pose and
label roots refer to the original Carmack inputs. A new input set has its own
output directory.

The preflight command below checks file availability, schemas and frame counts
before either model build starts.

The tracked [`good_court_refit/shuttleset22_labels.json.gz`](../good_court_refit/shuttleset22_labels.json.gz) preserves the earlier cleaned contact
frames and sides for 46 videos: 3,327 rallies and 37,184 contacts. It omits only
video 15 from the original cleaned label set. Unused shot-type fields and local
run metadata were left out; the original label file remains unchanged.

The config includes 40 development videos from the original ShuttleSet and
46 test videos from ShuttleSet22. Test video 15 is excluded because its labels
are misaligned; the development video named `sset_15` is different footage and
stays included. Every court path must be supplied explicitly.

Court results must use `court-detector-video/1`, cover the same frame timeline
as their shuttle/pose arrays, and declare the expected source resolution. The existing court loader and builder handle the conversion. The first check
reads the files and verifies the scene records without changing them. Conversion
then uses player detections to help decide which courts to accept. If every scene
in a video is rejected, the run stops before fitting so that video can be inspected.

### Revisions used for the completed run

The recorded comparison used annotator revision `97b5e4de` and the 86-video
`experiments/court_detector/fast_robust_20261002/court_sharing_patched` release
from `chore/new-courts-eval` at `e0151791dd1c525311b497d460eff65b4c455021`.
It copied the released court files into the run while keeping the annotator
code on its own branch. Machine-specific config and job locations remain
outside Git; the checked-in `inputs.json` is a template with blank court paths.

## Recorded Carmack environment

The original remote setup used the project's `~/.local/bin/hpcssh carmack` and
`hpcrsync` wrappers with a separate source checkout. The checked Python
executable was `~/.venvs/venv-annotator/bin/python`: Python 3.12.13,
scikit-learn 1.9.1. At preparation time, `venv-pipeline` lacked scikit-learn.

The read-only input check, from the repository root:

```bash
bash experiments/annotator/good_court_refit/run.sh \
  --config /path/to/chosen-inputs.json \
  --output /scratch/ahalperi/annotator-good-court-refit \
  --check
```

The supplied template should fail until its court paths are filled in. Running
without an action also selects the check. Only the explicit `--run` action
starts fitting:

```bash
bash experiments/annotator/good_court_refit/run.sh \
  --config /path/to/chosen-inputs.json \
  --output /scratch/ahalperi/annotator-good-court-refit \
  --run
```

`ANNOTATOR_PYTHON` can override the Python executable. The intended Carmack
budget allows eight worker processes with up to four numerical threads each
(32 total). During contact fitting, six workers get up to five threads each
(30 total). The sequence-fit stage runs alone with a 32-thread allowance.
Actual CPU use depends on the work within each stage.

Base finishes before veto starts; a failure stops the pair. Contact features
and contact trees are computed once and shared. Each build then fits its own
sequence and review models, using the candidates allowed by its rejection
setting.

Repeating the same command and config resumes the run; successful item
summaries identify completed work. Changed extracts or code that changes
computed inputs require a fresh output directory. Old-court caches belong to
the earlier experiment.

`run.json` records each fit/evaluation invocation, code revision, working-tree
changes and package versions. Resuming with different package versions fails.
A changed revision produces a reminder to check its effect on results. Input
identity includes path, size and modification time, so copying inputs can also
invalidate a resumed run.

## What the comparison holds fixed

| Setting | Both builds |
|---|---|
| Contact features, sampling and tree parameters | Existing defaults, seed 20260824 |
| Held-group negative sampling | A–D group order, omitting the held group |
| Contact fit row order | Video-ID order after sampling |
| Final 40-video and validation 32-video contact fits | Video-ID order for sampling and fitting |
| Downstream sequence and confidence inputs | A–D group order |
| Contact-side geometry | Whole-video net band |
| Court reference-pixel error | Existing 3.5 px default for conversion and annotation |
| Test population | The fixed ShuttleSet22 set with video 15 removed |

The 2 October check showed that changing the order of training examples can
change the tree even with the same examples and random seed. This runner fixes
that order so the fit can be reproduced. The check did not establish which
order would work better on unseen videos.
The generic training API keeps its default input order; this runner supplies
`fit_video_order` explicitly.

In the veto build, `ContactModelConfig.reject_masked_without_player` is true.
A possible hit is rejected when the shuttle guard flags its frame as unreliable
and neither player is selected at any of the five nearby time samples. The
default rejected guard grades are 1–3. A selected person is enough to keep the
frame in consideration, even without a usable wrist or a sudden shuttle-velocity
change. Rejection happens before nearby hits compete and before rally repairs
are assembled. The saved model carries this setting into later annotation. The base
bundle stores false.

Each build saves two model directories. `bundle_v32` is for checking the eight
validation videos: its contact tree was trained on the other 32 videos, in
groups A–D. `bundle` is for ShuttleSet22 and later annotation: its contact tree
uses all 40 development videos. Both directories use sequence and review models
trained on the 32 A–D videos.

`dev_heldout/` keeps the training-stage predictions made while each video's group
was left out. These can help diagnose fitting later; they are separate from the
eight-video validation results.

## Evaluate after fitting

The completed run evaluated both models on validation and ShuttleSet22 before
selection, to check whether the rule behaved differently across datasets. The
commands below repeat those checks with saved models. Evaluation loads the
models without changing them:

```bash
bash experiments/annotator/good_court_refit/run.sh \
  --config /path/to/chosen-inputs.json \
  --output /scratch/ahalperi/annotator-good-court-refit \
  --evaluate validation --arm base

bash experiments/annotator/good_court_refit/run.sh \
  --config /path/to/chosen-inputs.json \
  --output /scratch/ahalperi/annotator-good-court-refit \
  --evaluate validation --arm veto
```

The optional `--arm base_inference_veto` turns the rejection rule on while
keeping the base models. Comparing that output with the fitted veto shows what
changes when the sequence and review models are retrained as well. It writes separate outputs and leaves the base bundle
unchanged. `--evaluate test` selects the fixed 46-video comparison.

The saved-stream comparison runs without further inference:

```bash
PYTHONPATH=.:src ~/.venvs/venv-annotator/bin/python \
  -m experiments.annotator.good_court_refit.compare \
  --run /scratch/ahalperi/annotator-good-court-refit \
  --split validation --left base --right veto
```

The report includes complete-rally gains/losses and timing/correct-side label
changes, with IDs for follow-up. It requires the complete pinned population.
The same command also accepts `base` versus `base_inference_veto` when those
outputs exist. The [evaluation report](model_selection.md) combines this paired
comparison with coverage, court-failure and review-queue analysis.

## Outputs

- `shared/labels`, `shared/features` and `shared/contact` hold the shared inputs
  and fitted contact trees; feature and model caches stay untracked
- `base/` and `veto/` hold their two bundles, stage summaries and later
  `eval/<split>/<video>/stream.json.gz` and `scores.json.gz`
- `base_inference_veto/` holds only the optional diagnostic evaluation
- `comparisons/` holds paired saved-output reports

The retained [evaluation evidence](../good_court_refit/evidence/) contains the
small comparison records. Large training caches and temporary logs remain
outside Git.

### Selected model from the completed run

The selected final model is `base/bundle`, fitted with Python 3.12.13 and
scikit-learn 1.9.1. It keeps `reject_masked_without_player=False` and uses one
net position for the whole video when assigning hits to the near or far player.
`base/bundle_v32` is the corresponding validation model. The selected final
files are committed in [`data/annotator/sset_and_sset22_trained_20261003T041112Z`](../../../data/annotator/sset_and_sset22_trained_20261003T041112Z/).

On ShuttleSet22, base recovered 1,744 of 3,327 complete rallies: ten more than
the fresh old-court refit, along with 671 more labelled contacts. The fitted
veto gained three complete rallies on validation but lost four on ShuttleSet22.
Its advantage in small review queues did not persist across the tested queue
sizes. The [old-court findings](refit_regression.md) explain the
earlier baseline; the [new-court evaluation](model_selection.md) records the final
selection.
