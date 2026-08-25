# Contact detector exploration

This directory contains three readable reports plus the separate BST-X implementation plan.

The reports and PDFs stay here. Supporting Python modules live in `scripts/`, generated plots live in `figures/`, and retained experiment data lives in `raw/`.

Each report has one job:

| File | Question it answers |
| --- | --- |
| [`README.md`](README.md) | **What did we learn, and which document should I read next?** |
| [`auto_annotator_progress.md`](auto_annotator_progress.md) | **How good is each annotator output now?** |
| [`tree_contact_detector_results.md`](tree_contact_detector_results.md) | **Exactly what contact-detector experiments produced those results?** |
| [`bst_x_contact_detector_plan.md`](bst_x_contact_detector_plan.md) | **How should the proposed BST-X detector be implemented?** |

## Bottom line

The old heuristic contact path tries to produce a usable contact list with hand-written rules alone.

The experimental path separates the problem:

```text
broad deterministic search region
→ learned contact classifier
→ contact events
→ existing Top/Bottom attribution rule
```

The search region is deliberately relaxed. It is not a detector and is not produced by a tree. Its job is only to discard obviously irrelevant video without making real contacts unrecoverable.

Region v2 searches about **31.9% of the video** and contains a candidate within ±10 of **98.3% of labelled contacts**.

Inside that fixed region, HGB is the best contact-timing model tested:

- current final heuristics: **72.6% contact-timing F1**
- region-v2 RF: **84.6%**
- region-v2 HGB: **87.4%**

Player side is scored separately. When the existing Top/Bottom rule is applied to the frozen event streams (saved once, then held unchanged during scoring):

- current final heuristics get **70.6%** of all labelled contacts correct in both time and player side;
- region-v2 HGB gets **75.7%**;
- region-v2 RF gets **73.1%**.

For serves, the corresponding “right time + correct serving side” result is:

- current final: **46.2%**
- region-v2 HGB: **56.2%**
- region-v2 RF: **37.0%**

So the useful conclusion is:

> **Use region v2 as the search surface. HGB is the best region-v2 timing model tested. Region v1 still has slightly better timing F1 and joint event+side F1; region v2 keeps more contacts and serves reachable and gives higher timing+side recall.**

The new end-to-end rally check is much stricter than event F1. At the default
zero score cut, the system keeps 291 predicted spans and only 21 are fully
correct. A 0.90 timing-score cut keeps 51 spans and nine are fully correct.
Confidence filtering alone does not yet produce a useful clean rally subset.

The frame-rate check did not improve that result. Removing raw motion values
reduced timing F1 to **84.8%** and left 16 fully correct spans at the open
confidence setting. Converting motion to a common 30 fps scale gave **87.0%**
timing F1 and 15 fully correct spans. The existing raw-motion model remains the
baseline with **87.4%** timing F1 and 21 fully correct spans.


![The old standalone path and the experimental search-plus-classifier path.](figures/contact_pipeline_architecture.png)

## Why use region v2

Region v1 is slightly more selective and has marginally better event F1 on these three fixtures.

Use region v2 because its job is **search-space reduction**, not final event prediction. It keeps more serves reachable:

- region v1 serve coverage at ±10: **91.4%**
- region v2 serve coverage at ±10: **97.9%**

![Region v2's search-cost versus coverage trade-off.](figures/region_v2_search_tradeoff.png)

For the next classifier, that extra coverage matters more than v1's small precision/F1 edge.

## What is still open

The RF/HGB work does **not** change the rally-span finder. It now measures the
current HGB events and Top/Bottom answers inside those fixed spans.

Direct Top/Bottom attribution is measured on frozen HGB/RF contact events.

The separate **rally-level alternation fit** has not been rerun on those new events, so we do not yet know whether its final-hitter or server-side scores improve.

The strict result shows what usually blocks complete output. Of 311 predicted
spans, 210 map to one real rally but fail contact timing or event count before
player side is considered. Only 29 first fail at player side after exact
timing.

The missed-contact audit also narrows the next work. It finds a seeded HGB
candidate near 244 of 296 missed contacts. This includes 89 of 95 missed
serves. All 13 predicted spans that are otherwise exact apart from one missing
contact already have a region-v2 candidate nearby.

`sset_21` is the main warning sign: region v2 contains **94.7%** of its serves, but HGB finds only **44.0%**.

## Where to go next

Read [`auto_annotator_progress.md`](auto_annotator_progress.md) if you care about the annotator's outputs: rally spans, contact time + player side, serve time + serving side, and rally-level server attribution.

Read [`tree_contact_detector_results.md`](tree_contact_detector_results.md) for the experiment itself: region construction, exact feature sets, RF/HGB training, controls, player-side scoring and failure cases.

The frame-rate concern is now settled for this pilot. The next bounded pass is
the pre-specified small set of score cut-off and duplicate-removal decisions,
including separate handling near rally starts. Rescue search is deferred. A
second learned stage must wait for a label-blind shortlist that shows useful
extra coverage without a large false-alarm burden.

Read [`bst_x_contact_detector_plan.md`](bst_x_contact_detector_plan.md) only when moving on to the neural detector implementation. That specification is intentionally separate from the experiment reports.

## Full scorecard

This puts the main measured outputs in one place. The focused plots in the two reports are easier to read when comparing contact or serve performance alone.

![Full scorecard across the main setups.](figures/dense_scorecard.png)
