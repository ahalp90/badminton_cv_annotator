# What changed after the annotator refactor

The refactored code reproduced **all 47 historical output streams** when given
the historical contact scores and sequence models. Training fresh models on
the old court data changed the result: fully correct rallies fell from **1,763
to 1,734**, with 141 rallies becoming correct and 170 becoming incorrect.

[Reproduction commands and inputs](#reproduction)

The tests below separate those two changes. They support keeping the refactor
and attribute the gap to the fitted models in the evaluated contact and sequence
path. The later [new-court evaluation](model_selection.md) selected
the final model; this report explains the earlier regression and what the
attempted fixes established.

The comparison uses 47 ShuttleSet22 videos and 3,422 labelled rallies. A fully
correct rally contains every labelled contact, the correct known player sides,
valid bounds and no extra contacts. Timing allows ±10 frames at 30 fps, scaled
to the source frame rate. One rally has unknown human sides and cannot count
as fully correct. These videos have been inspected repeatedly, so the results
help diagnose known failures rather than measure performance on new footage.

## Why the fitted results differ

Swapping contact probabilities and the models that choose contact sequences
separately gives these fully correct rally counts:

| Contact probabilities | Historical sequence models | Fresh sequence models |
|---|---:|---:|
| Historical | 1,763 | 1,777 |
| Fresh | 1,729 | 1,734 |

Starting with historical models, changing contact probabilities costs 34
rallies; changing sequence models afterwards recovers five. In the opposite
order, changing sequence models gains 14; changing contact probabilities
afterwards loses 43. The effect of either change depends on which other models it is paired with.
That means there is no single number of lost rallies that can be assigned to
each training change. [Saved component comparisons](../old_court_regression/evidence/component-swaps.json.gz)
retain the aggregate and per-video results.

Three training details matter to future comparisons:

- **Fit row order:** a held-group test used the same 738,652 selected examples,
  labels and seed. Fitting in video-ID order reproduced all 367,951 historical
  candidate probabilities. Group order reproduced the later frozen-feature
  rerun instead ([saved order check](../old_court_regression/evidence/fit-order.json.gz)). Negative sampling order and fit row order are separate reproducibility settings.
- **Library version:** restoring old histogram binning under scikit-learn 1.9.1
  gave 1,702 correct rallies, below the fresh fit's 1,734. The historical models
  used 1.6.1. Library version is therefore another variable in a refit comparison.
- **Regenerated inputs:** a contact tree fitted on historical development inputs,
  with fresh inference and sequence models fixed, gave 1,754. That +20 measures
  the whole input change, including feature values and candidate rows; it is
  not a measured court-geometry contribution.

Two fits with different random seeds differed by 33 complete rallies. That
shows fitting choices can move the total appreciably. Two fits are still too
few to say how much variation is typical or to set an acceptable-loss threshold.
Both the recent old-court refit and the later new-court base track players
within each scene. When assigning a hit to the near or far player, both use
one net position for the whole video. That side-assignment setting stayed fixed
while the models changed.

## Which small changes helped?

Of the 170 rallies that became incorrect after refitting, 94 still had a
correct repair available, but the model scored a wrong repair higher. Another
37 were blocked by the final rule requiring a repair to beat the previous
choice by 0.05. For the remaining 39, none of the generated repair choices
could make the rally fully correct. The historical limits of two earlier-serve
candidates and six later-contact candidates were preserved: all 1,424,337
historical option records matched the original generator. At least 18 of the
39 unavailable cases need more edits than the current repair family permits,
so wider candidate lists alone cannot fix them.

| Change | Result | Implication |
|---|---|---|
| Remove the final margin | 79 rallies gained, 86 lost; total 1,727 | The margin was retained |
| Judge training options after extending rally bounds | Historical trial: +9 on development; 21 gained, 23 lost on test | The target mismatch is real; this trial did not establish a quality improvement |
| Block grade-1 added repair contacts when all pose/wrist flags are absent | Nine rallies gained, none lost | This narrow rule affects added repair contacts only |
| Block every grade-1 added repair contact | 15 gained, 58 lost | The guard flag does not prove a contact is absent |

These trials used the 47-video historical comparison. The following follow-up
excluded the misaligned ShuttleSet22 video 15 labels and used **46 videos,
3,327 labelled rallies**. It retained the same 1,734 baseline correct rallies.

## The nomination rule proposed for the new-court refit

The subsequent [new-court evaluation](model_selection.md),
completed on 3 October 2026, selected the base configuration with this rule off. The fitted veto
gained three complete rallies on V but lost four on ShuttleSet22, with three
fewer timing matches on each. The rule remains an optional evaluated setting;
its old-court gains did not establish a consistent benefit on the new inputs.
Court geometry affects player nomination, so its usefulness depends on the
input and fitted system. The historical results above remain unchanged.

The following results describe the earlier old-court trial.

Reject a candidate when its frame has shuttle guard grade 1, 2 or 3 **and**
neither player is nominated at any of five nearby samples. The samples are
−10, −5, 0, +5 and +10 frames at 30 fps, scaled to the source frame rate and
bounded by the candidate's search interval. A nomination means the player
selector picked a detected person. It does not require a valid or nearby wrist.
Grades 2–3 include degraded tracks as well as fabricated ones. A flagged
shuttle position can therefore still be near a real hit.

This trial reused the fitted models, contact scores, features and rough rally
bounds. It removed the flagged candidate frames before choosing among nearby
hits, then rebuilt the possible rally repairs from the remaining candidates. The rule gained **nine fully correct rallies and lost
none**: 1,734 → 1,743, or 52.1% → 52.4%. This is a different rule from the earlier
nine-gain repair-only control; six of their gained rallies overlap.

A rule can lose real hits inside a rally that was already incorrect, without
changing the complete-rally total. The contact counts show that damage:

| Label coverage | Gained | Lost | Net |
|---|---:|---:|---:|
| Contact matched within tolerance | 10 | 7 | +3 |
| Timing match with the correct player side | 12 | 13 | −1 |

**A correctly sided hit is also a timing match, so these rows overlap.** Across
all outputs, predicted contacts
fell from 39,428 to 39,422 and matched labels rose from 33,529 to 33,532.
Of 487,423 possible contact frames flagged by the shuttle guard, the rule
removed 91,536. That changed 68 rally sequences. Of the original final hits,
42 were directly rejected: 41 had no retained label close enough to match,
and the remaining labelled hit was found again two frames later. The seven
lost timing matches in the table came from other changes made when the rally
sequences were rebuilt.

Useful controls were less promising:

| Reject a masked candidate when… | Rallies gained / lost |
|---|---:|
| No nominated player | 9 / 0 |
| No nominated player OR weak shuttle impulse | 11 / 2 |
| No nominated player AND weak shuttle impulse | 0 / 0 |
| No raw person box | 0 / 0 |
| No raw person box OR weak shuttle impulse | 2 / 2 |

The variants using shuttle impulse—the sudden change in speed or direction
near a possible hit—lost two labelled serves in video 54. The raw-box variants
only asked whether any person had been detected, so spectators counted too.
Replacing selected-player absence with that weaker check removed all nine gains.



## Retained evidence and reproduction

The [runner](#reproduction) reproduces the original complete old-court comparison.
Its historical fit order and inclusion of video 15 remain unchanged. The
[nomination summary](../old_court_regression/evidence/nomination-veto.json.gz),
[contact totals](../old_court_regression/evidence/nomination-contact-totals.json.gz),
[label audit](../old_court_regression/evidence/nomination-contact-audit.json.gz) and
[raw-box controls](../old_court_regression/evidence/raw-box-veto.json.gz) preserve the small useful
follow-up results. Each follow-up baseline reproduced its saved contact stream
and correct-rally set. No model was retrained for those veto comparisons.

Large intermediate arrays and dated investigation scripts remain local. This
report keeps the conclusions, measurement definitions and useful controls;
coordination logs are not needed to interpret it.

## Reproduction

The runner fits models from the original court, shuttle and pose data, annotates
held-out videos, and scores the historical PR149 output and new output against
the same labels. The later new-court comparison has its own
[reproduction section](model_selection.md#reproduction).

### Inputs

The runner needs saved shuttle, pose and court data for 40 development videos
and 47 ShuttleSet22 test videos. The test set includes video 15, matching the
historical comparison. Its two input roots are:

- `--dev-stages DIR`: `DIR/stages/{shuttle,pose,court}/sset_NN/` with
  `shuttle_track.npy.xz`, `shuttle_guard_codes.npy.xz`, five `pose_*.npy.xz`,
  `court_evidence.json.gz`, `court_keep_vote.npy.xz` and `court_present.npy.xz`.
  `scratch/contact_det/raw/region_v2_inputs` has this layout for three videos.
- `--test-inputs DIR`: the ShuttleSet22 inpaint root. Each `DIR/"NN <name>"/`
  holds `shuttle_track_inpainted.npy.xz`, `shuttle_guard_codes_inpainted.npy.xz`,
  the pose files and the three court files.

The other defaults use the repository's saved `shots_master.csv`, group and
split records, and PR149 outputs in `scratch/contact_det_closing_pass/`.
`--clean-labels` defaults to the untracked file
`scratch/contact_det_full_ds_fit/raw/shuttleset22-test-result/clean_labels.json.gz`.

The old court files use `court-evidence/0.1`. Because current loaders require
0.2, this runner uses `old_inputs.py` to read the fields annotation needs:
`raw_cuts`, `inputs`, keep vote and court presence. It passes those fields to
`run_video()` using the same arguments as the normal annotation path. The old
per-scene provenance records are left out.

#### Optional checks against saved research inputs

These options help trace a difference back to features or training examples:

| Option | Additional check |
|---|---|
| `--frozen-dev-features .../raw/full_raw` | Fit a contact tree on the saved research-time features for all 40 development videos, and compare features per video. |
| `--frozen-test-features .../raw/shuttleset22-test-predictions` | Compare regenerated test features with the saved features. |
| `--contact-records scratch/contact_det_full_ds_fit/raw` | Compare example counts and the 80 saved check-row scores. |
| `--shuttleset22-annotations DIR` | Also evaluate against all source annotations, containing 3,965 rallies. |

### Run

```bash
PYTHONPATH=.:src python -m experiments.annotator.old_court_regression.runner \
  --dev-stages DEV_RUN --test-inputs SS22_INPAINT_ROOT --output OUT \
  --frozen-dev-features scratch/contact_det_full_ds_fit/raw/full_raw \
  --frozen-test-features scratch/contact_det_full_ds_fit/raw/shuttleset22-test-predictions \
  --contact-records scratch/contact_det_full_ds_fit/raw \
  --jobs 4 --threads 4
```

Parallel stages run `--jobs` worker processes, each with a numerical-thread
allowance of `--threads`. A stage containing one item receives the combined
`jobs × threads` allowance. Workers exit after completing their item.

The sequence fit is one long item, so its completion line appears only when
the fit finishes; the log emits a heartbeat each minute in the meantime. A
three-video sequence fit took about 20 minutes.

Repeating the same command resumes the run. An item's summary `.json.gz` marks it as
complete. If an item fails, its traceback is saved under `failures/<stage>/`;
the other items in that stage finish, then the run stops with a `FAILED` marker.
Repeating completed work requires removal of that stage's outputs and all
downstream outputs. Changed inputs require a fresh `--output` directory.

### What it runs

The development videos are divided into groups A–D and validation group V.
The contact tree used for the final test fit sees all 40 development videos.
The validation tree sees only the 32 videos in A–D, keeping V out of training.

The sequence models need examples of the contact tree's mistakes. Each A–D
video is therefore scored by a contact tree trained on the other three groups.
Those held-out predictions supply the sequence training examples. The sequence
and confidence models use A–D only.

| Stage | Work | Current functions used |
|---|---|---|
| `labels` | Label CSVs for 40 development and 47 test videos | `load_labels` contract |
| `old-streams` | PR149 test stream, test confidence, development held-group stream | — |
| `features` | Heuristic pass and contact features per development video | `run_video(heuristic_only=True)`, `features_from_evidence` |
| `contact-fits` | 40-video final tree, four 24-video fold trees, the 32-video tree for V | `fit_contact_model` |
| `pools` | A-D option pools from each video's held-group tree | `sequence_training_video` |
| `sequence-fit` | Nested sequence fits, confidence on held-group output, two bundles | `fit_sequence_models`, `fit_confidence_model`, `save_models` |
| `evaluate` | Full annotation of 8 V and 47 test videos | `load_models`, `run_video` with models |
| `compare` | Old and new streams, same labels and scorer, per video | `evaluation_counts`, `whole_rally_correctness` |
| `report` | `comparison.json.gz` and `comparison.md` | — |

For reproduction, the final 40-video contact tree and the 32-video validation
tree fit their selected examples in video-ID order. The smaller trees that
hold out one of A–D fit in group order. The historical
`final_contact_scores/group_V` records establish the validation tree's order.

### Outputs

The output directory contains two complete model bundles:

- `bundle/`: the 40-video contact tree and fitted sequence/review models, used
  for the 47 test videos;
- `bundle_v32/`: the 32-video contact tree and fitted sequence/review models,
  used for the eight V videos.

`eval/<split>/<id>/` holds each video's `annotator_result.json.gz`, final contact
stream, confidence scores and summary. `comparison.md` reports contact precision
and recall, fully correct labelled rallies, rallies gained and lost, and the
quality of the confidence ranking. `comparison.json.gz` adds per-video detail.

The logs let a resumed or inspected run be traced back to its inputs:
`progress.log` records each item's UTC time, stage, done/total count, identity,
status and duration; `timings.json` records stage wall times; and `run.json`
records invocation inputs, library versions and the Git commit.

### How to interpret the comparison

The scorer was checked by replaying PR149's saved test output. It reproduced
the published trusted-label results:

| Measurement | Reproduced result |
|---|---:|
| Labelled contacts found | 33,716 of 38,218 |
| Predicted contacts | 41,605 |
| Fully correct rallies | 1,763 of 3,422 |
| Clips above the historical 0.757 confidence cutoff | 784 |
| Fully correct clips among the 740 that trusted labels could judge | 616 |

The remaining 44 selected clips could not be judged with the trusted labels.
An equally sized review queue allows a direct quality comparison after
refitting. The same numerical cutoff can select a different number and quality
of clips with a new confidence model.

The end-to-end old/new comparison combines regenerated features, intended
edge-case changes in the code, the move from scikit-learn 1.6.1 to 1.9.1, and
fresh fitting. For example, regenerated features match the saved rows exactly
for `sset_15`, but differ in 312 of 54,656 rows for `sset_01`. The optional
frozen-feature fit helps measure how much those input differences affect the
contact tree. The [component experiments](#why-the-fitted-results-differ)
separate the fitted-model changes more closely.

A few details determine which results can be compared directly:

- **Loader coverage:** the runner tests the current `run_video()` path through
  an adapter for old court files. The current saved-input loaders and
  `run_full_annotation_stage()` need a separate check with 0.2 court evidence.
- **Validation:** only the new system has integrated V output, so V has no
  corresponding PR149 result in this run.
- **Development:** both sides use predictions made while each video's group
  was held out. The historical stream combines PR149's local-insertion contacts
  with its corrected rally boundaries.
- **Exact reproduction:** the fold trees retain group order. Their fitted
  probabilities can change with the scikit-learn version; aggregate results
  alone do not establish identical fits.
