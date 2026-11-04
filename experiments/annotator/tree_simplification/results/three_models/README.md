# Three-model annotator

The default bundle uses a contact classifier, one rally selector and a separate
review model. The selector chooses between the original contacts and proposals
from one additions/deletions routine. Training fits one selector on all training
videos and one per held-out group to provide review-training predictions.

Training now judges proposed contact sequences directly, without simulating the
final boundaries of neighbouring rallies. Runtime boundary padding is unchanged.
This removes a training-only finishing path. Contact features are prepared only
for frames nominated by the existing contact-search rules.

The refit gets 306 of 668 validation rallies completely right, versus 304 for the
previous selected model: 13 gains and 11 losses on the same rallies. Both fits
used the same 32 training videos, eight validation videos, vision inputs, tree
settings and seed 0. Validation includes all 314 rallies marked with flaws.
The simpler rule was selected for readability with similar measured performance;
these single-seed results do not establish a reliable accuracy gain.

| Validation measure | Previous selected model | Direct proposal targets |
| --- | ---: | ---: |
| Fully correct rallies | 304/668 | 306/668 |
| Contact F1 | 90.509% | 90.446% |
| Correct-player F1 | 85.631% | 85.648% |
| Fully correct unflagged rallies | 227/354 | 230/354 |

Across 1,065,983 training options, the selected target policy changed 1,821 labels
(0.171%) in 24 rally spans. Every change was from incorrect to unjudgeable;
no span gained or lost all its correct options. Exact validation gains and losses
are in [rally-changes.json.gz](rally-changes.json.gz).

The selected model also completed confirmation on the familiar 46-video test set.
Scoring included 3,327 rallies and 37,184 labelled contacts, with 524 rally
windows excluded under the existing test rules. The primary contact tolerance
was ten frames.

| Test measure | Previous selected model | Direct proposal targets |
| --- | ---: | ---: |
| Fully correct rallies | 1,851/3,327 (55.636%) | 1,842/3,327 (55.365%) |
| Contact F1 | 93.734% | 93.691% |
| Correct-player F1 | 91.121% | 91.139% |

Test confirmation is nine fully correct rallies lower, a decrease of 0.27
percentage points. This modest trade-off is accepted for the simpler training
rule. The test was not used to tune another variant. These are single-seed
results. See [summary.json.gz](summary.json.gz) for counts, the five-frame test
sensitivity check and review-ranking measures.

## Review ranking

The separate review model ranks likely incorrect rallies for inspection. On
validation, its correctness AUC was 0.793 versus 0.775 for the selector score.
Both found 57 incorrect rallies in the first 10% inspected; at 50%, the review
model found 222 versus 217. These comparisons use the 617 predicted rallies that
could be judged against labels; 137 could not be judged.

## Runtime

Use scikit-learn 1.9.1 and the published XGBoost 3.4.1 package to load
[the default seed 0 bundle](models/raw_baseline-seed-0/). The project dependency
pins remain unchanged.

Fitting used XGBoost 3.4.2 on a V100. The three models were exported through
XGBoost's native UBJSON format, loaded in 3.4.1, and saved as the deployment
bundle in that runtime. The model metadata records the 3.4.1 loading environment;
the fitting version remains 3.4.2. Keep historical bundles with their corresponding
source checkout.

## Rebuild

Use the existing staged trainer with the raw-baseline preprocessing candidate,
seed 0, five-frame-preferred targets and the current tree settings. Reuse prepared
vision evidence and compatible contact fits. Regenerate option pools and target
sidecars made before direct proposal judgement. Regenerate prepared inputs from
older checkouts that stored dense frame rows and region columns.
Score the exported validation streams and run a fresh CPU annotation.

Production manifest-based fitting exposes contact radii and independent
selector/review tolerances. The staged sidecar experiment retains its fixed
5/10-frame policy. See [retuning](../../../../../docs/annotator/retuning.md)
for the fitting entry points and saved settings.
