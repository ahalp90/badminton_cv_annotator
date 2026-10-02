# What changed after the annotator refactor

The old-court check supports keeping the refactor. Replaying the historical
contact scores and sequence models through current inference reproduces all
47 historical output streams. Refitting changes the result: fully correct
rallies fall from **1,763 to 1,734**, with 141 gained and 170 lost. The tests
attribute that output gap to changed fitted models; they found no remaining
porting defect in the evaluated contact and sequence path. The practical goal
is usable rally annotations for a performance dataset: complete contacts with
correct player sides, and fewer errors for a human reviewer to repair.

The comparison uses 47 ShuttleSet22 videos and 3,422 labelled rallies. A fully
correct rally needs every labelled contact, the right known player sides and
valid bounds, with no extra contacts. Timing allows ±10 frames at 30 fps,
scaled for other frame rates. One rally has unknown human sides and cannot
count as fully correct. These videos have been inspected repeatedly, so the
results are diagnostic rather than a fresh test of generalisation.

## Why the fitted results differ

Swapping contact probabilities and the models that choose contact sequences
separately gives these fully correct rally counts:

| Contact probabilities | Historical sequence models | Fresh sequence models |
|---|---:|---:|
| Historical | 1,763 | 1,777 |
| Fresh | 1,729 | 1,734 |

Changing probabilities first costs 34 rallies; changing sequence models then
recovers five. Reversing the order gives +14 then −43. The effects interact,
so these experiments do not assign a unique share of the regression to each
training change. [Saved component comparisons](evidence/component-swaps.json.gz)
retain the aggregate and per-video results.

Three training details matter to future comparisons:

- **Fit row order:** a held-group test used the same 738,652 selected examples,
  labels and seed. Fitting in video-ID order reproduced all 367,951 historical
  candidate probabilities. Group order reproduced the later frozen-feature
  rerun instead ([saved order check](evidence/fit-order.json.gz)). Keep negative sampling order separate from fit row order.
- **Library version:** restoring old histogram binning under scikit-learn 1.9.1
  gave 1,702 correct rallies, below the fresh fit's 1,734. The historical models
  used 1.6.1. Keep the library fixed within a new comparison.
- **Regenerated inputs:** a contact tree fitted on historical development inputs,
  with fresh inference and sequence models fixed, gave 1,754. That +20 measures
  the whole input change, including feature values and candidate rows; it is
  not a measured court-geometry contribution.

A seed pair moved the total by 33 rallies. This shows sensitivity; it supplies
neither a variance estimate nor a threshold for accepting a regression.
Per-scene player tracking also differs from assigning a contact to the far or
near player. The fresh comparison used a whole-video net band for that latter
assignment. Per-scene contact-side attribution explains none of this result.

## Which small changes helped?

Of the 170 lost rallies, 94 still had a correct repair option, but a wrong
option scored higher. The final 0.05 improvement margin blocked 37; the other
39 had no fully correct option. The historical limits of two earlier-serve
candidates and six later-contact candidates were preserved: all 1,424,337
historical option records matched the original generator. At least 18 of the
39 unavailable cases need more edits than the current repair family permits,
so wider candidate lists alone cannot fix them.

| Change | Result and implication |
|---|---|
| Remove the final margin | 79 rallies gained, 86 lost; total 1,727. Keep the margin |
| Judge training options after extending rally bounds | Historical trial gained nine on development, lost two on test (21 gained, 23 lost). The target mismatch is real; this trial did not establish a quality improvement |
| Block grade-1 added repair contacts when all pose/wrist flags are absent | Nine rallies gained, none lost. This narrow rule affects added repair contacts only |
| Block every grade-1 added repair contact | 15 gained, 58 lost. The guard flag does not prove a contact is absent |

These trials used the 47-video historical comparison. The following follow-up
excluded the misaligned ShuttleSet22 video 15 labels and used **46 videos,
3,327 labelled rallies**. It retained the same 1,734 baseline correct rallies.

## The nomination rule proposed for the new-court refit

Reject a candidate when its frame has shuttle guard grade 1, 2 or 3 **and**
neither player is nominated at any of five nearby samples. The samples are
−10, −5, 0, +5 and +10 frames at 30 fps, scaled to the source frame rate and
bounded by the candidate's search interval. A nomination means the player
selector picked a detected person. It does not require a valid or nearby wrist.
Grades 2–3 include degraded tracks, so “fabricated coordinates” is too narrow
a description of the mask.

The experiment kept fitted models, probabilities, features and segmentation
fixed. It removed candidate rows before nearby-contact suppression, then
rebuilt repair choices. The rule gained **nine fully correct rallies and lost
none**: 1,734 → 1,743, or 52.1% → 52.4%. This is a different rule from the earlier
nine-gain repair-only control; six of their gained rallies overlap.

Whole-rally totals hide changes inside already-incorrect rallies:

| Label coverage | Gained | Lost | Net |
|---|---:|---:|---:|
| Contact matched within tolerance | 10 | 7 | +3 |
| Timing match with the correct player side | 12 | 13 | −1 |

The rows overlap and must not be added. Across all outputs, predicted contacts
fell from 39,428 to 39,422 and matched labels rose from 33,529 to 33,532.
The rule excluded 91,536 of 487,423 masked candidate rows and changed 68 rally
sequences. It directly forbade 42 final baseline contacts: 41 had no retained
label within tolerance; the remaining labelled hit was recovered two frames
later. The lost label matches arose from subsequent sequence changes.

Useful controls were less promising:

| Reject a masked candidate when… | Rallies gained / lost |
|---|---:|
| No nominated player | 9 / 0 |
| No nominated player OR weak shuttle impulse | 11 / 2 |
| No nominated player AND weak shuttle impulse | 0 / 0 |
| No raw person box | 0 / 0 |
| No raw person box OR weak shuttle impulse | 2 / 2 |

The impulse variants lost two labelled serves in video 54. Raw boxes include
spectators. Their presence is a weaker test of usable player evidence, and
substituting box absence removed all nine gains.

Retain the nomination rule as an optional experiment until the new-court
comparison is evaluated. Court geometry itself affects player nomination, so
better courts may change both its usefulness and its risks. Measure complete
rallies, missed contacts and correct-side coverage together before choosing it.
The [prepared refit](../good_court_refit/README.md) keeps the baseline rule off
and saves the experimental setting in its model bundle.

## Retained evidence and reproduction

The [runner](README.md) reproduces the original complete old-court comparison.
Its historical fit order and inclusion of video 15 remain unchanged. The
[nomination summary](evidence/nomination-veto.json.gz),
[contact totals](evidence/nomination-contact-totals.json.gz),
[label audit](evidence/nomination-contact-audit.json.gz) and
[raw-box controls](evidence/raw-box-veto.json.gz) preserve the small useful
follow-up results. Each follow-up baseline reproduced its saved contact stream
and correct-rally set. No model was retrained for those veto comparisons.

Large intermediate arrays and dated investigation scripts remain local. This
report keeps the conclusions, measurement definitions and useful controls;
coordination logs are not needed to interpret it.
