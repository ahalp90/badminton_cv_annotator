# Earlier three-model annotator

The earlier bundle uses a contact classifier, one rally selector and a separate
review model. Player attribution uses the court for the current camera scene.
Both heuristic and fitted annotation resolve tied alternating player orders
using summed wrist-to-shuttle distance, normalised by player height.

The selected seed-0 refit gets **341 of 668 validation rallies completely right**.
The preceding first-known tie rule and its fitted models scored 336. The new rule
with those earlier models scored 337; refitting the selector and review model
increased that to 341. The contact models were reused.

## Inputs and training targets

The inference order is:

`raw contact candidates → contact XGB → rally-selector XGB → player alternation`

The selector receives original player guesses. Final alternation chooses the
order matching the most guesses; a vote tie uses the lower summed wrist distance.
A contact contributes to the distance comparison only when both court halves
have usable evidence. Equal or unavailable sums follow the first known guess.
If every guess is unknown, all players remain unknown.

Selector targets judge whether a proposed sequence would be correct **after**
that same alternation rule. Review targets judge held-group selector predictions
using the same correctness rule. The ground-truth labels and contact targets
are unchanged. Review features retain raw player guesses and use final rally
boundaries; review scores rank rallies for inspection without changing contacts.

Training judges proposed contact sequences directly, without simulating the
final boundaries of neighbouring rallies. Runtime boundary padding is unchanged.
Training fits one selector on all training videos and one per held-out group to
provide review-training predictions.

## Validation

Both refits used the same 32 training videos, eight validation videos, vision
inputs, tree settings and seed 0. All 668 validation rallies are included,
including 314 marked with flaws. Contact matching uses a ten-frame tolerance
at 30 FPS, scaled to each video's frame rate.

| Validation measure | First-known tie refit | Wrist-distance tie refit |
| --- | ---: | ---: |
| Fully correct rallies | 336/668 | **341/668** |
| Contact F1 | 90.530% | 90.580% |
| Correct-player F1 | 89.479% | 89.511% |
| Fully correct unflagged rallies | 258/354 | 260/354 |

The new refit recovers 17 rallies and loses 12 relative to the first-known tie
refit. Exact changes are in [rally-changes.json.gz](rally-changes.json.gz).
Saved predictions and fresh CPU inference produce the same reported counts.
These are single-seed comparisons.

Five paired fits supported keeping scene geometry throughout training and
inference. Mean fully correct validation rallies were 335 with scene-based
training and 334.2 with video-band training evaluated using scene geometry.
The selected model retains seed 0 rather than choosing the best validation seed.

## Test confirmation

The selected model completed confirmation on the familiar 46-video test set,
scoring 3,327 rallies and 37,184 labelled contacts. The existing exclusion rules
removed 524 rally windows. The primary contact tolerance was ten frames.

| Test measure | First-known tie refit | Wrist-distance tie refit |
| --- | ---: | ---: |
| Fully correct rallies | 1,872/3,327 (56.267%) | **1,873/3,327 (56.297%)** |
| Contact F1 | 93.672% | 93.694% |
| Correct-player F1 | 91.472% | 91.498% |

The selected refit adds one fully correct test rally, with similar contact and
player scores. Test results were used for confirmation after selection on
validation. See [summary.json.gz](summary.json.gz) for counts, the five-frame
sensitivity check and review-ranking measures.

## Review ranking

On validation, the separate review model's correctness AUC was 0.810 versus
0.781 for the selector score. In the first 10% inspected, review ranking found
55 incorrect rallies versus 57 for the selector; at 50%, it found 210 versus
200. These comparisons use 616 judgeable predicted rallies, of which 275 were
incorrect; another 138 predictions could not be judged.

## Runtime and rebuilding

Use scikit-learn 1.9.1 and the published XGBoost 3.4.1 package to load
[the seed 0 bundle](models/raw_baseline-seed-0/). Fitting used XGBoost
3.4.2 on a V100. The trees were exported through XGBoost's native UBJSON format,
loaded in 3.4.1, and saved as the deployment bundle in that runtime. The project
dependency pins remain unchanged.

Use the staged trainer with raw-baseline preprocessing, seed 0,
five-frame-preferred selector targets and the current tree settings. Reuse
prepared vision evidence and compatible contact fits. Regenerate option pools
and correctness targets made before the current player-attribution rule.
Review fitting requires regenerated held-group selector predictions.

Production manifest-based fitting exposes contact radii and independent
selector/review tolerances. The staged experiment retains its fixed 5/10-frame
policy. See [retuning](../../../../../docs/annotator/retuning.md) for fitting
entry points and saved settings.
