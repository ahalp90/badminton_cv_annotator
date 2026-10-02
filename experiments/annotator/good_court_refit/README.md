# Refit the annotator with the new court detections

This prepares two annotators from the same chosen court detections: the base
settings, then the optional rule that rejects masked candidates without a
nominated player. The builds run in that order. **Neither build has been run.**
The court extracts are still being corrected; their final paths belong in the
input config when they are ready.

The runner reuses the [old-court runner](../old_court_regression/README.md)'s
parallel stages and resume support. It converts saved custom-detector JSON
into operational court evidence, then calls the maintained annotator fitting
functions. It does not run court detection or change the source extracts.
The [evaluation template](evaluation.md) describes what to measure afterwards.

## Finish the inputs tomorrow

1. Copy `inputs.json` to an untracked working config. Fill the `null` court
   paths with the chosen per-video detector results. Use full paths on Carmack;
   plain JSON and `.json.gz` are accepted
2. Check the existing shuttle, pose and label locations in that config. The
   supplied roots are the earlier Carmack inputs; only the court files are
   intended to change. Use a fresh output directory for a different input set
3. Run the preflight below. Resolve every missing file, schema or frame-count
   failure before starting either build

The tracked `shuttleset22_labels.json.gz` preserves the earlier cleaned contact
frames and sides for 46 videos: 3,327 rallies and 37,184 contacts. It omits only
video 15 from the original cleaned label set. Unused shot-type fields and local
run metadata were left out; the original label file remains unchanged.

The config includes 40 original-ShuttleSet videos and 46 ShuttleSet22 videos.
ShuttleSet22 video 15 is excluded because its labels are misaligned. Original
ShuttleSet `sset_15` stays included. There are no guessed court paths or automatic
fallbacks to the old CourtKeyNet detections.

Court results must use `court-detector-video/1`, cover the same frame timeline
as their shuttle/pose arrays, and declare the expected source resolution. The
conversion uses the existing court-evidence builder and current artefact loader.
The read-only check validates the files and scene records. Court conversion
then applies the player vote; a video with no accepted scene stops the run
before fitting, so it can be inspected rather than silently dropped.

## Run on Carmack when the extracts are chosen

Use the project's `~/.local/bin/hpcssh carmack` and `hpcrsync` wrappers from the
local machine. Put this branch's code in a separate Carmack checkout or source
copy; do not overwrite another branch's checkout. The old temporary regression
checkout may have been removed. The environment checked during preparation is
`~/.venvs/venv-annotator/bin/python`: Python 3.12.13, scikit-learn 1.9.1.
`venv-pipeline` currently lacks scikit-learn.

From the copied repository on Carmack, run the read-only input check:

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
budget is eight worker processes with four numerical threads each. The six
contact fits use five threads each; the single sequence-fit stage gets all 32.
The base build completes before the
veto build starts; a failure stops the pair. Their unchanged features and
contact trees are computed once and shared. This saves repeated work and
holds the upstream model fixed for the comparison.

Resume with the same command and config. Successful item summaries let the
runner skip completed work. Use a new output for changed extracts or code that
changes computed inputs. Do not copy old-court caches into this run.
`run.json` records each fit/evaluation invocation, code revision, working-tree
changes and package versions. Resuming with different package versions fails;
a changed revision prints a reminder to check whether its changes affect results.
Keep the same checkout and input paths through fitting and evaluation, preserving
input modification times when copying files. The input check treats a changed
path, size or modification time as a changed input.

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

The fit order follows the 2 October order check: changing the order of selected
rows changes the tree even when the examples and seed stay the same. This is
a reproduction choice, not evidence that one order generalises better.
The generic training API keeps its default input order; this runner supplies
`fit_video_order` explicitly.

In the veto build, `ContactModelConfig.reject_masked_without_player` is true.
A candidate is rejected only when its frame is masked by the fitted guard
settings and neither player is nominated at any of the five sampled offsets.
The default guard grades are 1–3. Player nomination is sufficient; valid wrists
or shuttle impulse are not required. The rule acts before contact suppression
and repair shortlists, and the bundle retains it for later inference. The base
bundle stores false.

Each build saves `bundle_v32` for V validation and `bundle` for ShuttleSet22
and later inference. V remains outside the contact tree used to evaluate it.
The final contact tree uses all 40 development videos; downstream and confidence
models use only A–D. Keep those bundle roles separate.
The already-computed held-group predictions are retained in each build's
`dev_heldout/` for later training diagnostics.

## Evaluate after fitting

Evaluate both builds on validation, then record the choice before examining
ShuttleSet22 results. Evaluation is a separate action and performs no fitting:

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

The optional `--arm base_inference_veto` applies the rule to the base bundle
without refitting. It helps distinguish the rule's immediate effect from the
new downstream fits. It writes separate outputs and leaves the base bundle
unchanged. Use `--evaluate test` later for the fixed 46-video comparison.

Compare saved streams without further inference:

```bash
PYTHONPATH=.:src ~/.venvs/venv-annotator/bin/python \
  -m experiments.annotator.good_court_refit.compare \
  --run /scratch/ahalperi/annotator-good-court-refit \
  --split validation --left base --right veto
```

The report includes complete-rally gains/losses and timing/correct-side label
changes, with IDs for follow-up. It requires the complete pinned population.
Repeat for `base` versus `base_inference_veto` when that output exists.
Use the [evaluation template](evaluation.md) for coverage, court failure checks,
review-queue quality and the final decision; the comparison JSON does not cover
all of those questions.

## Files worth keeping

- `shared/labels`, `shared/features` and `shared/contact` hold the shared inputs
  and fitted contact trees; feature and model caches stay untracked
- `base/` and `veto/` hold their two bundles, stage summaries and later
  `eval/<split>/<video>/stream.json.gz` and `scores.json.gz`
- `base_inference_veto/` holds only the optional diagnostic evaluation
- `comparisons/` holds paired saved-output reports

Keep the finished evaluation and small useful result tables in a dated tracked
experiment directory. Keep large arrays, model binaries, run logs and temporary
review records outside Git. The [old-court findings](../old_court_regression/findings.md)
are the baseline narrative; they do not establish quality on the new extracts.
