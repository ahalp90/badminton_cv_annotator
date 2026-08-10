# Corrected contact and refit experiment

Status: completed. The measured result and limits are in `report.md`.

## The question

Can shuttle motion before the earliest accepted contact tell us that this contact was the first return rather than the serve?

The anchor will be the earliest accepted geometry/impulse contact in each predicted rally. Its player will come from `attribute_half` at that exact frame. The experiment will never use the existing alternating-fit server label to decide which player the shuttle approaches.

We will run two related experiments across all ShuttleSet rallies in `sset_01`, `sset_15` and `sset_21`.

### Experiment 1: decide whether the anchor is a return

Look backwards at most 30 base-30 frames from the anchor. Scale that maximum to each video's frame rate. Search only before the contact and choose the closest continuous run; do not reach farther back because an older run looks cleaner. The path and anchor must remain inside one court scene from `tracker_segments`.

A usable path has at least five consecutive visible points, ends no more than two scaled base-30 frames before the anchor, has recurrence code zero, and has finite positions for the anchor player. A minimum total movement will be swept in player body heights. A second sweep will cap the largest one-frame movement divided by the median non-zero movement, which rejects a path dominated by one wild jump. Both cut-offs will be shown rather than hidden in a combined score. The other player's path may be recorded as a diagnostic, but will not be required.

Measure whether the path closes on the anchor player. Compare:

1. a plain direction rule based on net distance closed and the percentage of consecutive movements that reduce distance to the anchor player;
2. a separately labelled variant that may add path-shape measurements, including a quadratic fit.

The direction-only detector is the main experiment. Curve shape will remain diagnostic unless its alternate threshold curve shows a clear gain. A quadratic always has more freedom than a line, and earlier inpaint work found plausible-looking false paths.

If the rule says the anchor is a return, the inferred server is the other player. We will show two results when it does not fire. The forced result names the anchor player as server. The evidence-only result abstains when there is no usable path, while a measured path below the threshold counts as evidence for the anchor player. If the anchor player is unknown, both results abstain.

Ground truth is used only for scoring and exploratory threshold choice. For a covered rally, count how many ShuttleSet contacts lie within the usual frame tolerance of the anchor. Zero means unmatched; more than one means ambiguous; exactly one means contact 1, contact 2, or a later contact according to its stroke number. Only the unique contact-1 and contact-2 groups enter the first-return threshold curve. The other groups remain separate.

Choose the displayed setting by the highest first-return F1. Break exact ties by higher precision, then by requiring a higher percentage of incoming movements. This is an exploratory result on the same three videos, not a held-out result. Final server F1 will not choose the setting.

### Experiment 2: prepend and refit

Recalculate the direct player guess for every accepted contact. These frame-backed guesses form the complete input sequence. When Experiment 1 says the anchor is a return, compare two one-step additions before calling the existing `fit_alternation` function once:

1. prepend `None`, which tests the effect of one missing contact on sequence parity without adding a player vote;
2. prepend the inferred other player, which tests parity plus one extra vote derived from Experiment 1.

The second vote is independent of the old alternating fit, but it is not a new measurement of player identity: it is the opposite of the anchor player already in the sequence. The prepended serve has an unknown frame. We will not invent a timestamp or pass a fake frame through `run_video`. We will report the refitted first player, the original fit margin, whether the supplied vote changes the fitted final player, and server accuracy and macro-F1 for both variants.

### What the report must make obvious

- counts for every denominator;
- the old alternating-fit baseline, contact-local estimate, and prepended refit as separate methods;
- all rallies and the frozen geometry/contact failure subset: the 99 covered rallies whose released server label is wrong plus the 22 whose released label is missing;
- first-contact, second-contact, later-contact, ambiguous and unmatched anchor groups;
- precision, recall and F1 for detecting a first return;
- server accuracy, per-class precision/recall/F1, macro-F1 and abstentions;
- threshold curves with axes in named physical or count units;
- a headline threshold plot whose x-axis is “minimum movements towards the contact player (%)” and whose y-axis shows precision, recall and F1 as percentages;
- TP, FP and FN counts printed beside the selected threshold, with separate lines for plainly labelled minimum net-closure requirements in player body heights;
- readable labels that state the numerator and denominator;
- a small, labelled set of representative paths and all false positives at the chosen threshold.

## Work order

1. **Rebuild the evidence table.** Reuse the frozen release annotations, tracks, court masks and pinned pose arrays. Join GT rallies through `classify_all`. Recalculate direct contact halves with `attribute_half`.
2. **Verify the anchor.** Confirm the first accepted frame, direct half, raw earlier candidates, GT contact match and trajectory window for every covered rally.
3. **Measure incoming motion.** Produce net closure in player body heights, percentage of movements towards the anchor player, and conservative path-quality flags. Keep quadratic measurements as a named comparison. Do not publish a combined feature under an unexplained name such as “motion score”.
4. **Score Experiment 1.** Plot threshold curves and plain feature distributions. Score the contact-local server estimate on all requested groups.
5. **Score Experiment 2.** Compare the parity-only and player-labelled prepends, rerun `fit_alternation`, and score the refitted first player. Do not recalculate winners or rally boundaries.
6. **Inspect errors and write the result.** Validate reported arithmetic from compressed tables. Explain what was tested within the first 800 words.

## Checks

Synthetic tests will cover incoming, outgoing, stationary, jumping, gapped and curved paths. Sequence tests will compare the experiment's unmodified fit with the frozen result, prove that prepending `None` leaves the fitted final player unchanged, and check the one-vote effects of the labelled prepend. An output validator will recompute counts and threshold rows from `.csv.gz` files.

Before completion, run the dedicated tests and the repository Ruff, Pyrefly and pytest commands. Stop if the reconstructed accepted contacts, direct guesses or unmodified alternating fits disagree with the frozen release for unexplained reasons.

## Files

Useful scripts, tests and documents will live in this directory and be tracked. Inputs, symlinks, compressed result tables, plots, case images and delegated-agent records will remain ignored. Generated NumPy, JSON and CSV data will use `.npy.xz`, `.json.gz` and `.csv.gz`.

## Outside this experiment

- no changes under `src/**`;
- no exact serve frame or fake contact geometry;
- no changes to rally spans, winners, landings, replay masks or production output;
- no learned model;
- no use of GT as an inference feature;
- no claim that an unseen serve occurred when the path does not support it;
- no tracked release data, pose arrays, generated plots or bulky result files;
- no work on `main`, `.env`, `.claude/**` or `experiments/**`.

## Proposed commits

These messages were approved on 10 August 2026.

```text
Track the corrected serve trajectory experiment

Narrow the ignore rules and add the dedicated plan, scripts and tests. Keep downloaded inputs, generated results and agent records out of Git.
```

```text
Measure incoming motion at accepted contacts

Use each anchor contact's direct player attribution. Compare clear direction and path-quality rules against ShuttleSet first- and second-contact truth.
```

```text
Refit rallies after inferred missing serves

Prepend an order-only server guess when motion identifies a first return. Reuse the existing alternating fitter and report its full effect on server attribution.
```

```text
Explain the corrected serve trajectory results

Add readable plots, checked tables and a plain account of what the two experiments did and did not establish.
```
