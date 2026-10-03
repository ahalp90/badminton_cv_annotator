# Court geometry repair: evidence and reproduction

The saved evidence supports checking the [issue #148 results](../../../scratch/court_det_fix/evidence/retirement/README.md)
from saved before/after outputs. It includes court and contact results, separate
court-geometry checks and the contact model used for the comparison. The fitted
models stayed fixed, so these results measure changes in the inputs and processing
around them rather than a retraining.

The saved-output checks below require no videos or GPU.
The [pipeline reproduction recipe](court_repair_reproduction.md) lists the larger inputs needed
to regenerate detections, annotations and model predictions.

The files described below are in
[`court_geometry_repair/`](../court_geometry_repair/).

## What is included

| Location | Contents and purpose |
| --- | --- |
| `evidence/video_17/`, `evidence/video_53/` | Baseline and final court evidence, contact/rally streams, run metadata and player-selection probes; `expected.json.gz` holds the reported scores |
| `evidence/controls/video_3/`, `evidence/controls/video_21/` | Old and repaired court estimates for the same frames, neural predictions, detected court lines, reference corners and expected errors |
| `evidence/labels/` | The saved clean contact/rally labels for ShuttleSet22 videos 17 and 53 |
| `evidence/followup.json.gz` | Retained grouping, fresh-broadcast, player-feature and painted-line pilot outputs; scope below |
| `models/` | Unchanged contact model and its original training/settings records; the other frozen selection models already live in the repository |
| `scripts/` | Saved-output checks and the cleaned experiment entry points |
| `figures/` | Two sheets of **prototype** comparison images, explained below |

JSON and CSV records are gzip-compressed. Court corners use the recorded image
resolution; the geometry controls use 1280×720 reference pixels. Scene intervals
are half-open: the start frame is included and the end frame is excluded.

The baseline represents saved production evidence at `a111181`. Source changes
are recorded in `87f66e7` and `2b486d5`. The final pipeline records here come from
the completed runs after the replay-mask correction. Earlier failed integration
outputs are excluded from the numerical results.

## Check the saved results

The commands run from the repository root in its Python environment. The
scripts use the project dependencies and existing scoring functions.
The [CPU CI setup](../../../docs/ci.md) provides a suitable starting point for
these saved-output checks. The frozen-model rerun needs the older versions
listed separately in [REPRODUCE.md](court_repair_reproduction.md#inputs-and-environment).
They compare recomputed results with the supplied expected records and fail on
a mismatch.

```bash
export PYTHONPATH="$PWD/src:$PWD"
BUNDLE="$PWD/experiments/annotator/court_geometry_repair"
OUT="$(mktemp -d)"

for VIDEO in 17 53; do
  python "$BUNDLE/scripts/evaluate_court_comparison.py" \
    --video-id "$VIDEO" \
    --baseline "$BUNDLE/evidence/video_$VIDEO/before" \
    --changed "$BUNDLE/evidence/video_$VIDEO/after" \
    --labels-root "$BUNDLE/evidence/labels" \
    --expected "$BUNDLE/evidence/video_$VIDEO/expected.json.gz" \
    --output "$OUT/video_$VIDEO.json.gz"
done

python "$BUNDLE/scripts/check_geometry.py"
python "$BUNDLE/scripts/check_followup.py"
```

The contact checks cover both ±10-frame and ±5-frame matching at 30 fps.
With a ±10-frame allowance, video 17 keeps its 17 fully correct rallies.
Video 53 improves from 7 to 35. **All previously correct rallies remain correct.** The report explains
the denominators and the precision loss on video 17.

Separate geometry checks use original-ShuttleSet videos 3 and 21. In video 3,
seven scene outlines disagreed with the outlines accepted in other scenes.
The checker rebuilds replacements for those seven using the median corner
positions from the other accepted scenes. It then checks that the replacements
fit the detected painted court lines better than the original outlines.
Video 21 needed no repairs: all 45 original outlines were kept.

The final corner positions are compared with each video's reference court at
an image size of 1280 × 720 pixels:

| Video | Scenes with an accepted court | Outlines repaired | Average distance from a predicted corner to its reference corner |
|---|---:|---:|---:|
| 3 | 46 / 47 | 7 | 5.17 pixels |
| 21 | 45 / 45 | 0 | 4.74 pixels |

These checks reproduce the reported calculations from saved outputs. Testing
the full production path—including how it selects source scenes and uses player
detections to accept courts—requires the separate pipeline reproduction. New
footage would require its own evaluation.

## Checks performed for this bundle

Both saved-stream evaluations matched the supplied expected records. The two
geometry checks also passed. The failure checks also worked: the scripts rejected an incorrect expected
contact count and a repaired court with missing line evidence.
The new scripts passed syntax, CLI and scoped lint checks.

The cleaned preparation and scoring scripts were also run on all 119,849 frames
of video 53. That earlier rerun rebuilt the before/after annotations and features from
saved detection outputs, then applied the unchanged fitted models. It matched
the expected result at both timing tolerances. The neural detections were reused
while checking that the packaged scripts could reproduce the result.

The whole-project type check still reported the same 11 existing import errors.
Production source was unchanged while packaging this bundle.

## Retained follow-up evidence

`evidence/followup.json.gz` collects 31 later output files under their original
filenames. It preserves the measured values while shortening local and remote
paths to filenames. The records are grouped here by the question they help
check:

| Records | Evidence |
|---|---|
| Video 17/53 grouping outputs, rally evaluations and player-feature summaries | How court sharing changed geometry and annotation on the two problem videos. |
| Original-data matching, sharing-guard comparisons and pooled-fit summaries | Which scene estimates were grouped and how the alternative fits behaved. |
| Full before/after court payloads, masks and review summaries for videos 8/9/10 | What happened on the fresh-video checks. |
| The 26-scene painted-line pilot | How well proposed outlines followed the detected court markings. |
| `mask_runs` | Which frames were flagged before and after each change, recorded as start-inclusive, end-exclusive ranges alongside the frame count. These ranges were checked against the original masks before packaging. |

These are the saved outputs of the historical revisions. The check script
recalculates their reported comparisons; reproducing them with current code
requires the separate pipeline recipe.

`check_followup.py` checks how many scenes have accepted courts and how many
share an outline. It verifies that scenes in the same sharing group use exactly
the same corners.

To measure how much the original scene estimates disagreed, it takes the same
105 image points and maps them onto the court using each scene's original
calibration. It compares the resulting on-court positions and also measures
the largest difference between corresponding corners. This uses the saved group
median and the original repaired corners in `evidence/video_{17,53}/after/`.

For the fresh-video checks, it compares the before/after court records while
ignoring only their `case_id` labels, then compares the frame masks. For the
painted-line pilot, it first takes each line's median coverage across frames,
then averages the lines in each direction group. The fields `mean_x_family`
and `mean_y_family` average all samples directly, which is a different calculation.

Other archived summaries can be inspected directly by their source keys. Source
videos, decoded images and full player arrays remain external. The archive records what the earlier visual review concluded. Checking those
judgements again requires the source images; rerunning the full experiments
also requires the external video and player data.

## Selected visual evidence

These images show an early prototype that fitted courts scene by scene. It
predates the final rule that checks the outline against painted court lines.
Blue shows the old outline and orange the prototype outline; dashed outlines
were rejected. The captions identify which example scenes the final run accepted or rejected.

### Preserved view and false close-ups — ShuttleSet22 video 17

The opening court is correctly restored in the top-left panel. Several other
panels show false courts on close-ups. The final run rejects scenes 62, 90, 115
and 165 shown here; its decisions are in `evidence/video_17/after/court_evidence.json.gz`.

![Prototype court comparisons for video 17](../court_geometry_repair/figures/prototype_video17.png)

### Boundary recovery — ShuttleSet22 video 53

The top-right panel is the reported scene 334. The old fallback follows a
diagonal; the recovered outline follows the painted court. Scene and frame
identifiers in every panel allow a reader to locate the underlying records.
These six scenes also survive final acceptance.

![Prototype boundary recovery comparisons for video 53](../court_geometry_repair/figures/prototype_video53_boundary_recovery.png)

## Provenance and limits

Numerical evidence retains the recorded measurements. Machine-specific source
paths in control metadata were replaced by video filenames and public input
references. Control interval CSVs were reconstructed from the saved intervals.
The saved model records keep their original contents so the existing model
loader can read them. The figures are unchanged excerpts from the original
comparison images, using ShuttleSet22 broadcast footage.

The clean labels are a two-video subset of the earlier frozen evaluation's
ShuttleSet22 label record. They retain its frame coordinates, rally membership
and player sides. The source annotation project is
[CoachAI ShuttleSet22](https://github.com/wywyWang/CoachAI-Projects/tree/main/CoachAI-Challenge-IJCAI2023/ShuttleSet22).

The reference court templates apply to their matching wide camera views. The
video 3 and 21 checks test the court geometry; a full annotation run would also
need to repeat player-based court acceptance and rally detection. The
ShuttleSet22 videos 17 and 53 were chosen because their failures were already
known, so their improvements describe those cases rather than unseen footage.

Grouping repeated matching views and evaluating partial courts remain next
steps, as described in the [investigation trail](../../../scratch/court_det_fix/evidence/retirement/README.md).
