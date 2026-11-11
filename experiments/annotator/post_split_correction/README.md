# Correcting rallies after splitting

The annotator now uses a separately trained rally selector to correct the pieces
created when a long rally is split. Training reuses the existing contact models.
This method gave the most fully correct validation rallies in the comparison,
with almost unchanged contact and player F1. Fitting extra contact models added
complexity without a useful improvement.

The deployed model uses seed 100, selected by fully correct validation rallies:
361 of 668. Its validation contact F1 is 90.875% and player F1 is 89.910%.
The corresponding test results are 2,014 of 3,327 fully correct rallies,
93.720% contact F1 and 91.775% player F1. Against the same seed's baseline,
validation gains two fully correct rallies; contact F1 falls by 0.023 percentage
points and player F1 by 0.040 points.

## What changed

The baseline already uses standing-person counts and splits long contact gaps.
See the [court-rule comparison](../tree_simplification/results/court_acceptance/README.md)
for those results.

The first rally selector chooses a contact list and assigns players. A long gap
between contacts can then split that rally into pieces. Previously, the annotator
kept each piece's contacts as they were and copied the parent's confidence score.

Now, a second rally selector considers alternatives for each split piece. It can
recover candidates rejected during the first selection and reconsider players
using the original hitter evidence. Piece bounds stay within the parent rally,
with a small exclusion margin around each split midpoint to prevent two
nearby candidates from becoming separate hits in neighbouring pieces.

The second selector changes only split pieces. Ordinary rallies keep their
contacts, players and bounds. Supplied rally bounds bypass splitting. The
refitted confidence model scores every final rally using its own features.
Ordinary feature rows stay the same, but their confidence can change because the
model was refitted. There is no further splitting or merging.

## Accuracy across five seeds

All methods used the same 32 training videos and seeds 0, 100, 200, 300 and 400.
Validation contains eight videos, 668 labelled rallies and 5,696 contacts. Test
contains 46 videos, 3,327 rallies and 37,184 contacts. Scoring uses the established
exclusion windows and a contact tolerance of ten frames at 30 fps, scaled to each
video's frame rate.

Contact F1 measures matched contact times. Player F1 also requires the right
player. A fully correct rally must contain all labelled contacts, no extra
contacts and the right players under the benchmark's rally matching rules.
These counts do not establish that every video boundary or outcome is correct.
See the [benchmark scoring definitions](../monotrack_benchmark/annotator_monotrack_own_weights_compared_writeup_v7.md).

The following values are means across the five seeds, rather than scores for the
single deployed model.

| Method | Validation correct rallies | Contact F1 | Player F1 |
|---|---:|---:|---:|
| Baseline | 353.2 | 90.844% | 89.822% |
| Reuse the first selector after splitting | 353.0 | 90.822% | 89.790% |
| **Train a second selector; reuse contact fits** | **354.4** | **90.855%** | **89.810%** |
| Train a second selector with extra contact fits | 353.4 | 90.853% | 89.818% |

| Method | Test correct rallies | Contact F1 | Player F1 |
|---|---:|---:|---:|
| Baseline | 1,992.0 | 93.729% | 91.563% |
| Reuse the first selector after splitting | 2,003.8 | 93.765% | 91.915% |
| **Train a second selector; reuse contact fits** | **2,024.6** | **93.818%** | **91.986%** |
| Train a second selector with extra contact fits | 2,020.6 | 93.816% | 91.985% |

The selected method gains 1.2 fully correct validation rallies and 32.6 test
rallies over the baseline. Validation player F1 falls by 0.013
percentage points; test player F1 rises by 0.424 points. The validation gain is
small. The larger test population provides a sanity check, rather than a reason
to claim a decisive validation improvement.

| Fully correct validation rallies | Seed 0 | 100 | 200 | 300 | 400 |
|---|---:|---:|---:|---:|---:|
| Baseline | 346 | 359 | 355 | 355 | 351 |
| Reuse the first selector after splitting | 345 | 358 | 356 | 356 | 350 |
| Train a second selector; reuse contact fits | 348 | 361 | 356 | 355 | 352 |
| Train a second selector with extra contact fits | 346 | 360 | 355 | 355 | 351 |

Among split pieces, the selected method repairs a mean of 1.4 validation pieces
and damages 0.2 previously correct pieces per seed. On test it repairs 40.6 and
damages 8.0. Reusing the first selector repairs 32.6 test pieces but damages 20.8;
training a separate selector reduces that damage.

## Training and runtime

Both retrained methods keep the experiment's full contact model and first rally
selector fixed. They train temporary rally selectors to generate split examples,
then fit the second selector and confidence model. Ordinary and split examples
use the same targets and weights. The original ordinary examples are retained
once, without adding duplicate unsplit parents.

To train confidence, each video group is held out in turn. A contact model fitted
on the other groups scores all videos used in that turn. Temporary first rally
selectors produce split examples for training the second selector. The held-out
group is excluded from all of these fits. Its final predicted rallies then
provide examples for confidence training.

The alternative with extra contact fits excludes another group when fitting
contact models for intermediate training examples. That prevents a contact model
from scoring its own training videos at this intermediate step. It requires ten
extra contact fits per seed and another level of group bookkeeping. The measured
accuracy did not justify retaining that alternative in normal training.

| Retrained method | Training time per seed | New model fits per seed |
|---|---:|---:|
| Reuse contact fits | 16.1 min | 22 |
| Extra contact fits | 22.3 min | 26 |

These times cover the additional training in this experiment, starting from the
saved contact models and first selector. They exclude fitting those original
models. Five seeds ran together, using six CPU cores per seed and GPU fitting.
The two methods ran sequentially. Normal training first fits the contact models
and first selector, then follows the selected method above.

| Core inference | Mean seconds per video |
|---|---:|
| Baseline | 9.47 |
| Second selector trained by reusing contact fits | 9.99 |
| Second selector trained with extra contact fits | 9.99 |

The added cost is about 0.52 seconds per video, or 5.5%. Each of the eight
validation videos was timed for all five seeds, with six video workers per seed
and GPU models warmed before timing. This measures contact-feature construction
through final confidence from prepared tracking evidence. Disk and model loading,
detectors and outcome estimation are excluded. These are timings from one matched
run, not a measurement of the complete raw-video pipeline.

## Compared with the earlier published result

The earlier README's 94.2% contact F1 came from a pipeline that rejected some
contacts using court geometry. The test population, exclusions and main tolerance
are the same. The following values are five-seed means.

| Test metric | Earlier published pipeline | Selected method |
|---|---:|---:|
| Contact precision | 95.557% | 94.440% |
| Contact recall | 92.849% | 93.205% |
| Contact F1 | 94.184% | 93.818% |
| Player F1 | 91.763% | 91.986% |
| Fully correct rallies | 1,963.2 | 2,024.6 |

The selected method finds about 132 more true contacts and 435 more false contacts
per seed than that published pipeline. The precision loss exceeds the recall
gain. Most of the contact-F1 gap predates this change: the baseline
already scored 93.729%. The selected method trades some contact precision for
more fully correct rallies and better player F1 relative to the published result.

The [saved published scores](../monotrack_benchmark/benchmark_results/hp_search/primary/contact_summary.json.gz)
retain that earlier comparison. Confidence figures for the newly fitted model
are reported separately in the updated MonoTrack benchmark.

## Data and reproduction

Download [per-seed metrics, training times and inference timings](results.json.gz)
to inspect the comparisons. The [updated benchmark data](benchmark/) also includes
supplied-boundary scores and the refitted confidence model's results. Precision, recall and F1 are percentages. The saved
comparisons include counts and paired differences for each seed. In that file,
`existing_selector` means reusing the first selector after splitting;
`reuse_contact_fits` and `extra_contact_fits` are the two retrained methods.
Each comparison pairs the `baseline` with the named method's `trial` results.
The earlier
published contact metrics are in
[the original summary](../monotrack_benchmark/benchmark_results/hp_search/primary/contact_summary.json.gz).

Use the [normal training command](../../../docs/annotator/retuning.md) to fit the
four-model pipeline. The five fitted bundles are under [`models`](models), one directory per seed.
The default uses the seed-100 files in
[`runtime/deployed/annotator`](../../../runtime/deployed/annotator).
For the exact experimental implementations of both training alternatives, use
Git commit `a899364d`; commit `2e24e551` only clarifies a training-pool name.
Normal training retains the selected method alone.
