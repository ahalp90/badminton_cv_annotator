# RF/HGB contact detector experiments

This report contains the technical detail behind the contact-detector results.

It does not restate rally-segmentation results or the overall annotator scorecard. Those belong in [`auto_annotator_progress.md`](auto_annotator_progress.md).

## Table of contents

- [Experiment structure](#experiment-structure)
- [Search-region experiment](#search-region-experiment)
- [Exact tree inputs](#exact-tree-inputs)
- [Training and timing evaluation](#training-and-timing-evaluation)
- [Timing-only model sweep](#timing-only-model-sweep)
- [Player-side scoring](#player-side-scoring)
- [Region v1 versus region v2](#region-v1-versus-region-v2)
- [Boundary sensitivity](#boundary-sensitivity)
- [Where it still fails](#where-it-still-fails)
- [Recommendation](#recommendation)
- [Reproduction](#reproduction)

## Experiment structure

The work has three stages:

```text
deterministic search-region construction
→ RF/HGB contact scoring
→ existing Top/Bottom attribution on frozen predicted events
```

The current **final** heuristic event list is only a comparison point; HGB does not use it as an input.

The current **raw** proposals are one of several inputs used to build the search region.

## Search-region experiment

### Why a new region was needed

The current raw proposals make too many real contacts unreachable by a later classifier.

At ±10 they cover only:

- **83.8% of non-serves**
- **66.1% of serves**

The search-region experiment therefore asks:

> **How much of the video can we discard while still leaving almost every contact available to the classifier?**

### Region-v2 construction

Region v2 is deterministic and label-blind.

It takes temporal neighbourhoods around:

- current raw contact proposals;
- relaxed shuttle impulse / direction-change peaks;
- local shuttle-to-wrist minima;
- shuttle visibility changes;
- detected rally starts;
- scene starts;
- a 45-base-30-frame look-back immediately before eligible court-view intervals.

The final item exists because some serves happen during the close-up immediately before the broadcast returns to the full court.

Ground truth is loaded only after a region has been constructed, to measure coverage.

### Coverage

Region v2 scores **128,824 candidate centres**, about **31.9% of 404,229 source frames**.

At ±10:

| Search surface | All contacts | Non-serves | Serves |
| --- | ---: | ---: | ---: |
| Court-view intervals only | 97.9% | 98.3% | 93.2% |
| **Region v2** | **98.3%** | **98.4%** | **97.9%** |

Per fixture:

| Fixture | Non-serve coverage | Serve coverage | All-contact coverage |
| --- | ---: | ---: | ---: |
| `sset_01` | 99.4% | 98.2% | 99.3% |
| `sset_15` | 100.0% | 100.0% | 100.0% |
| `sset_21` | 93.7% | 94.7% | 93.8% |

These are **search coverage** numbers, not model recall.

![Region v2 reduces the search workload while keeping almost every labelled contact reachable.](figures/region_v2_search_tradeoff.png)

Region v2 is intentionally permissive. An extra candidate frame only gives the classifier more work; a real contact left outside the region cannot be recovered.

## Exact tree inputs

The main `physics` feature set contains **85 columns**.

### Physical signals: 60 columns

Twelve signals sampled at offsets −10, −5, 0, +5 and +10 base-30 frames:

- `shuttle_vx`
- `shuttle_vy`
- `shuttle_speed`
- `shuttle_impulse`
- `shuttle_impulse_ratio`
- `wrist_gap_min`
- `wrist_gap_top`
- `wrist_gap_bot`
- `nearest_wrist_dx`
- `nearest_wrist_dy`
- `ankle_speed_top`
- `ankle_speed_bot`

### Validity / missingness: 25 columns

Five signals at the same five offsets:

- `shuttle_visible`
- `pose_valid_top`
- `pose_valid_bot`
- `wrist_valid_top`
- `wrist_valid_bot`

### Optional context: 20 columns

The context block contains centre-frame absolute positions, ankle positions, bbox heights, standing count, interval timing, scene timing and seven region-source flags.

The four tested feature sets are therefore:

| Feature set | Columns |
| --- | ---: |
| `physics` | **85** |
| `physics_context` | **105** |
| `context_only` | **20** |
| `missingness_only` | **25** |

### PySceneDetect

PySceneDetect affects where candidates can occur through scene/court interval construction and the scene-start seed.

Scene timing is also present in the optional context block.

The winning 85-column HGB `physics` model does **not** receive scene timing or region-source flags as explicit inputs.

## Training and timing evaluation

The scorer uses three outer leave-one-fixture-out folds.

For each held-out fixture:

- the other two fixtures are the training side;
- inner out-of-fold predictions choose the probability threshold;
- the held-out fixture does not choose its threshold.

Training rows:

- positive through ±1 base-30 frame of a labelled contact;
- ignored through ±4;
- hard negatives through ±15;
- easy negatives sampled toward at most 12:1 negatives to positives.

Temporal NMS radius: **5 base-30 frames**.

Metrics are reported at ±5, ±10 and ±15. The main comparison is ±10.

## Heuristic proposal and cleanup

Before the learned models, the existing heuristic path moves from a broad raw proposal stream to a cleaner final event list:

| Timing-only event stream at ±10 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Current raw proposals | 41.7% | **82.2%** | 55.3% |
| Current final heuristic cleanup | **66.9%** | 79.3% | **72.6%** |

The wrist/proximity cleanup greatly improves precision, but it also removes some real contacts. That is why the new path keeps the search region relaxed and lets the classifier do the cleanup.

## Timing-only model sweep

This table is **timing-only**.

| Event stream at ±10 | Precision | Recall | F1 | Non-serve recall | Serve recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current final heuristics | 66.9% | 79.3% | 72.6% | 81.2% | 61.0% |
| **HGB physical** | **84.5%** | **90.5%** | **87.4%** | **92.9%** | **67.5%** |
| RF physical | 84.1% | 85.2% | 84.6% | 89.6% | 42.8% |
| HGB + context | 81.7% | 89.8% | 85.5% | 92.5% | 63.4% |
| RF + context | 83.9% | 86.5% | 85.2% | 90.4% | 47.9% |

Controls:

| Timing-only control | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| HGB context only | 47.0% | 79.6% | 59.1% |
| RF context only | 37.1% | 82.2% | 51.1% |
| HGB missingness only | 16.1% | 96.9% | 27.5% |
| RF missingness only | 16.0% | 96.9% | 27.5% |

The main conclusions from the sweep are:

- HGB is the strongest timing model;
- context hurts HGB;
- missingness alone only achieves high recall by producing far too many events;
- there is no clear reason to spend more time tuning RF.

## Player-side scoring

The tree does not predict player side.

For the existing heuristic, region-v1 and region-v2 event streams, the player-side scorer leaves every prediction frame, threshold and NMS decision unchanged, then applies `attribute_half` at each predicted frame.

The **eligible-court-only HGB** row is the one exception: it is a new full HGB refit on the eligible-court rows, including threshold selection and NMS. That entire event stream is still completed and frozen **before** player-side ground truth is loaded.

Before scoring side, the script checks that the original timing counts are unchanged.

At ±10:

| Event stream | Timing recall | Side accuracy given timing match | Timing + correct-side recall | Joint event+side precision | Joint event+side F1 | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current final heuristics | 79.3% | **89.0%** | 70.6% | 59.6% | 64.6% | 46.2% |
| **HGB physical** | **90.5%** | 83.7% | **75.7%** | 70.7% | **73.1%** | **56.2%** |
| HGB + context | 89.8% | 81.9% | 73.4% | 66.8% | 69.9% | 46.2% |
| RF physical | 85.2% | 85.8% | 73.1% | **72.1%** | 72.6% | 37.0% |
| Eligible-court-only HGB | 89.1% | 84.1% | 74.8% | 72.5% | **73.7%** | 53.4% |

Adding side correctness narrows the gap:

- HGB is still the region-v2 tree model we would use;
- RF nearly catches HGB once correct player side is required;
- on timing matches, the side rule is right **83.7%** of the time for HGB and **89.0%** for the current final heuristics. We do not know whether the extra recovered contacts explain that difference.

For serves, the side rule is right on **84.1%** of HGB timing matches versus **75.8%** of current-final timing matches, while serve timing recall rises from **61.0% to 67.5%**.

A simpler check using centre-frame `wrist_gap_top` and `wrist_gap_bot` gives essentially the same HGB side result.

## Region v1 versus region v2

The region choice is a simple trade-off: v1 is more selective; v2 keeps more contacts reachable.

| HGB physical metric | Region v1 | Region v2 |
| --- | ---: | ---: |
| Serve search coverage | 91.4% | **97.9%** |
| Timing precision | **87.5%** | 84.5% |
| Timing recall | 88.4% | **90.5%** |
| Timing F1 | **87.9%** | 87.4% |
| Timing + correct-side recall | 74.6% | **75.7%** |
| Joint event+side F1 | **74.1%** | 73.1% |
| Serve timing + correct-side recall | 52.7% | **56.2%** |

Region v1 is slightly more selective.

Region v2 keeps more contacts, especially serves, available to the classifier. Since the region's job is search-space reduction rather than final event prediction, use v2 as the search surface.

## Boundary sensitivity

We also refit HGB using eligible court-view rows only, to check whether the extra rows just before court view were somehow driving the result.

| HGB physical search | Timing P | Timing R | Timing F1 | Timing + correct-side recall | Joint event+side F1 | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main region v2 | 84.5% | **90.5%** | 87.4% | **75.7%** | 73.1% | **56.2%** |
| Eligible court only | **86.3%** | 89.1% | **87.7%** | 74.8% | **73.7%** | 53.4% |

Only nine of the main HGB's 3,350 detections are in the extra before-court rows.

Those rows help **search coverage**, especially for serves, but HGB scores almost the same without them.

## Where it still fails

### `sset_21`

HGB physical at ±10:

| Fixture | Timing F1 | Timing + correct-side recall | Serve timing recall | Serve timing + correct-side recall |
| --- | ---: | ---: | ---: | ---: |
| `sset_01` | 92.6% | 77.3% | 74.3% | 60.2% |
| `sset_15` | 84.2% | 76.5% | 76.9% | 67.3% |
| `sset_21` | **79.6%** | **70.7%** | **44.0%** | **34.7%** |

Region v2 contains **71 / 75 = 94.7%** of `sset_21` serves, while HGB finds **33 / 75 = 44.0%**. Four serves sit outside region v2, but most misses happen after search. With only three fixtures, we cannot tell whether `sset_21` is simply different from the other two or whether the model has fit the other two too closely.

### Contacts outside region v2

The 37 `sset_21` non-serves outside region v2 all have visible shuttle evidence near the labelled contact, but none has sticky player analysis or a player pick and none lies inside a detected rally span.

That is an off-court/live-close-up search problem.

### Very broad shuttle-only check

A deliberately broad shuttle-only check searches **90.6% of the broadcasts** and covers every non-serve plus 291 of 292 serves at ±10.

That shows the shuttle evidence exists, but the search is far too broad to use and often lacks usable player evidence.

## Other useful experiments

Three other decisions are worth keeping.

**Ankle side attribution.** On current-final timing matches, the shipped box/net rule scores **2,207 / 2,480 = 89.0%** correct Top/Bottom answers. Replacing the final half test with ankle height gives **2,208 / 2,481 = 89.0%**. That is effectively no change.

**PR88 server rule.** PR88 used shuttle motion before and after a chosen contact to decide whether it looked like a serve or return. When that was unclear, it fell back to the older PR82 alternation answer. It got the server side right on **170 / 239 = 71.1%** of an older fixed 239-rally development set. This is not directly comparable with the 292-rally evaluation used here. On a later held-out test it tied the older rule overall and got worse on one video, so we did not keep it.

**X3D-S.** Leave it aside for now. RGB may contain useful contact clues that pose and shuttle do not. Revisit it only if the failure cases point to missing visual information.

## Recommendation

Use:

- **region v2** as the search region;
- **HGB physical + validity** as the simple learned contact model.

Do not add the tested context block to HGB.

The **87.4% timing F1** is only the timing score. When player side matters, the corresponding HGB results are:

- **75.7% timing + correct-side recall**
- **73.1% joint event+side F1**
- **56.2% serve timing + correct-side recall**

For the annotator-level result, see [`auto_annotator_progress.md`](auto_annotator_progress.md).

## Reproduction

Feature freezer:

```text
scratch/contact_det/scripts/freeze_tree_contact_features.py
```

Timing scorer:

```text
scratch/contact_det/scripts/score_tree_contact_detector.py
```

Player-side scorer:

```text
scratch/contact_det/scripts/score_contact_player_attribution.py
```

Summary figures:

```text
scratch/contact_det/scripts/plot_summary_figures.py
```

Saved region-v2 timing result:

```text
scratch/contact_det/raw/region_v2/tree_contact_results.json.gz
```

Saved player-side result:

```text
scratch/contact_det/raw/contact_player_attribution_score.json.gz
```
