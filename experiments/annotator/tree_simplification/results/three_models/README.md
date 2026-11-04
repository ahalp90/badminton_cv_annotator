# Three-model annotator

The default bundle uses a contact classifier, one rally selector and a separate
review model. The selector chooses between the original contacts and proposals
from one additions/deletions routine. Training fits one selector on all training
videos and one per held-out group to provide review-training predictions.

The current rebuild improves complete-rally validation results from 298 to 304
out of 668. Both models used the same 32 training videos, eight validation videos,
vision inputs, tree settings and seed 0. Validation includes all 314 rallies
marked with flaws. Contact features are built only for frames nominated by the
existing contact-search rules; this reduced preparation work without changing
the rebuilt model's validation scores.

| Validation measure | Previous selector | Current rebuild |
| --- | ---: | ---: |
| Fully correct rallies | 298/668 | 304/668 |
| Contact F1 | 90.209% | 90.509% |
| Correct-player F1 | 85.391% | 85.631% |
| Fully correct unflagged rallies | 227/354 | 227/354 |

The selected model also completed confirmation on the familiar 46-video test set.
Scoring included 3,327 rallies and 37,184 labelled contacts, with 524 rally
windows excluded under the existing test rules. The primary contact tolerance
was ten frames.

| Test measure | Previous selector | Current rebuild |
| --- | ---: | ---: |
| Fully correct rallies | 1,848/3,327 (55.545%) | 1,851/3,327 (55.636%) |
| Contact F1 | 93.677% | 93.734% |
| Correct-player F1 | 91.015% | 91.121% |

The test confirms the selected model; it was not used to select a new variant.
These are single-seed results. See [summary.json.gz](summary.json.gz) for counts,
the five-frame test sensitivity check and review-ranking measures.

## Review ranking

The separate review model ranks likely incorrect rallies for inspection. On
validation, its correctness AUC was 0.786 versus 0.767 for the selector score.
It found 56 incorrect rallies in the first 10% inspected, versus 58 using the
selector score; at 50%, it found 222 versus 213. The improvement is therefore
not uniform across review budgets. These comparisons use the 616 predicted
rallies that could be judged against labels; 138 could not be judged.

## Runtime

Use scikit-learn 1.9.1 and the published XGBoost 3.4.1 package to load
[the default seed 0 bundle](models/raw_baseline-seed-0/). The project dependency
pins remain unchanged.

Fitting used XGBoost 3.4.2 on a V100. The three trees were exported through
XGBoost's native UBJSON format, loaded in 3.4.1, and saved as the deployment
bundle in that runtime. The model metadata records the 3.4.1 loading environment;
the fitting version remains 3.4.2. Keep historical bundles with their corresponding
source checkout.

## Rebuild

Use the existing staged trainer with the raw-baseline preprocessing candidate,
seed 0, five-frame-preferred targets and the current tree settings. Reuse prepared
vision evidence and compatible contact fits and option pools. Regenerate prepared
inputs from older checkouts that stored dense frame rows and region columns.
Score the exported validation streams and run a fresh CPU annotation.

Production manifest-based fitting exposes contact radii and independent
selector/review tolerances. The staged sidecar experiment retains its fixed
5/10-frame policy. See [retuning](../../../../../docs/annotator/retuning.md)
for the fitting entry points and saved settings.
