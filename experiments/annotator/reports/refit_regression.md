# What changed after the annotator refactor

The refactored code reproduced **all 47 historical output streams** when given
the historical contact scores and sequence models. Training fresh models on
the old court data changed the result: fully correct rallies fell from **1,763
to 1,734**, with 141 rallies becoming correct and 170 becoming incorrect.

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
  rerun instead ([saved order check](../old_court_regression/evidence/fit-order.json.gz)). Keep negative sampling order separate from fit row order.
- **Library version:** restoring old histogram binning under scikit-learn 1.9.1
  gave 1,702 correct rallies, below the fresh fit's 1,734. The historical models
  used 1.6.1. Keep the library fixed within a new comparison.
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
| Remove the final margin | 79 rallies gained, 86 lost; total 1,727 | Keep the margin |
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

The [runner](refit_reproduction.md) reproduces the original complete old-court comparison.
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
