# Tree contact detector trial

## Bottom line

Histogram boosting is a credible contact detector for the current physical inputs. On leave-one-fixture-out predictions, its physics model reached 87.5% precision and 88.4% recall at ±10 frames. Non-serve recall was 91.1%. Serve recall was only 62.7%.

The tree is doing more than spotting broadcast gaps. The context-only and missingness-only controls had much lower F1. Adding context to the physics model raised recall, but lowered precision and F1. The best result came from physical features plus their validity masks, without absolute court position, rally progress, scene timing or proposal-source flags.

This does not settle the model choice. The result makes histogram boosting the baseline that BST-X must beat. It also confirms that serve search is still the immediate problem. The broader regions contain 98.3% of non-serve frames and 82.5% of serve frames exactly. Within the ±10 scoring margin, their operational ceilings are 98.3% and 91.4%.

## What was tested

The trial used `sset_01`, `sset_15` and `sset_21`. One fixture was held out for each outer fold. The other two fixtures supplied training data and inner held-out predictions for threshold selection.

The feature freezer generated 112,502 rows without loading ground truth. Each row is one possible centre frame inside a label-blind search region. The regions cover 27.8% of the 404,229 source frames across the three fixtures.

The region union contains six expanded channels:

- current raw proposals
- shuttle impulse ratio at least 1.25
- local wrist-gap minima below 3 body heights
- shuttle visibility boundaries
- detected rally starts
- scene starts

Each model sees a 21-frame physical window through values sampled at offsets -10, -5, 0, 5 and 10 base-30 frames. Positives are within one base-30 frame of GT. Frames from 2 through 4 away are ignored. All negatives within 15 frames are retained as hard negatives. Easy negatives fill the sample towards a 12:1 negative-to-positive ratio.

The threshold is selected at ±5 on inner fixture-held-out predictions. The scorer then applies temporal NMS with a five-base-30-frame radius. The threshold and NMS do not read the outer test fixture.

## Search-region ceiling

This is strict coverage: the GT contact frame itself must be present in the expanded region. It is a stronger test than allowing the edge of a region to fall within the scoring tolerance.

| Region | Non-serves | Serves | All contacts |
| --- | ---: | ---: | ---: |
| Current raw proposal region | 2,528 / 2,836 (89.1%) | 218 / 292 (74.7%) | 2,746 / 3,128 (87.8%) |
| Relaxed impulse region | 2,789 / 2,836 (98.3%) | 241 / 292 (82.5%) | 3,030 / 3,128 (96.9%) |
| Broad six-channel union | 2,789 / 2,836 (98.3%) | 241 / 292 (82.5%) | 3,030 / 3,128 (96.9%) |

The pooled number hides a large fixture difference:

| Fixture | Non-serve coverage | Serve coverage |
| --- | ---: | ---: |
| `sset_01` | 1,519 / 1,528 (99.4%) | 108 / 113 (95.6%) |
| `sset_15` | 719 / 720 (99.9%) | 84 / 104 (80.8%) |
| `sset_21` | 551 / 588 (93.7%) | 49 / 75 (65.3%) |

Allowing the normal event-scoring margin gives this operational ceiling for the broad union:

| Margin | Non-serve ceiling | Serve ceiling |
| --- | ---: | ---: |
| ±5 | 2,789 / 2,836 (98.3%) | 253 / 292 (86.6%) |
| ±10 | 2,789 / 2,836 (98.3%) | 267 / 292 (91.4%) |
| ±15 | 2,789 / 2,836 (98.3%) | 272 / 292 (93.2%) |

At ±10, the serve ceilings are 98.2% for `sset_01`, 92.3% for `sset_15` and 80.0% for `sset_21`.

The relaxed impulse region supplies the union's entire strict GT coverage. The other five channels add 409 search frames, but no additional exact GT contact frames. Pooled operational coverage clears a 97% non-serve gate and a 90% serve gate at ±10.

It does not clear either gate on every fixture. `sset_21` is the problem case for both contact groups, so the region design is not ready to serve as the shared tree/BST-X search surface.

The main restriction is that every region is clipped to a saved detected rally span. A missed or late-opened serve cannot be recovered outside that span. The next region version should work over live court-view scene intervals and add a serve lookback around scene and tracker starts.

## Main model results

### Histogram boosting with physical features

The physical input includes shuttle motion, impulse, player–shuttle wrist gaps, relative wrist position, ankle motion and explicit validity masks. It excludes absolute image position, player size, rally progress, scene timing and proposal-source flags.

| Margin | Precision | Recall | F1 | Non-serve recall | Serve recall | Median error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ±5 | 84.3% | 85.2% | 84.8% | 88.6% | 52.7% | 1 frame |
| ±10 | 87.5% | 88.4% | 87.9% | 91.1% | 62.7% | 1 frame |
| ±15 | 87.9% | 88.9% | 88.4% | 91.3% | 65.8% | 1 frame |

At ±10, the model emits 3,162 events and matches 2,766 of 3,128 GT contacts.

The held-out fixture results vary enough to matter:

| Held-out fixture | Precision | Recall | F1 | Non-serve recall | Serve recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sset_01` | 94.5% | 90.6% | 92.5% | 92.3% | 67.3% |
| `sset_15` | 81.5% | 86.9% | 84.1% | 89.3% | 70.2% |
| `sset_21` | 79.4% | 84.9% | 82.1% | 90.0% | 45.3% |

The threshold is 0.80 in all three folds. The variation therefore comes from the fixture, not fold-specific threshold drift.

### Model and feature-set comparison at ±10

| Model | Features | Precision | Recall | F1 | Non-serve recall | Serve recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Histogram boosting | Physics + validity | 87.5% | 88.4% | **87.9%** | 91.1% | 62.7% |
| Histogram boosting | Physics + context + validity | 82.7% | **90.4%** | 86.4% | **93.0%** | **65.4%** |
| Random forest | Physics + validity | 83.2% | 88.4% | 85.7% | 92.0% | 53.4% |
| Random forest | Physics + context + validity | 84.1% | 87.9% | 86.0% | 91.6% | 51.4% |

Histogram boosting is the better baseline. Its physics model has the best F1 and precision. Its context-added model is available when recall matters more than the extra false positives.

Compared with the current final contacts at ±10, the histogram-boosting physics model:

- raises non-serve recall from 81.2% to 91.1%
- raises serve recall from 61.0% to 62.7%
- reduces event count from 3,706 to 3,162
- raises overall event precision from about 67.0% to 87.5%

The gain is concentrated in later contacts. The tree barely improves serves.

## Shortcut controls

| Model | Control at ±10 | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Histogram boosting | Context only | 43.6% | 81.4% | 56.8% |
| Random forest | Context only | 38.4% | 73.6% | 50.5% |
| Histogram boosting | Missingness only | 16.3% | 96.6% | 27.9% |
| Random forest | Missingness only | 16.3% | 96.5% | 27.9% |

Missingness can spray predictions across most search regions and obtain high recall. Its very low precision makes the shortcut obvious. Context alone also carries real timing information, but remains far behind the physics models.

Adding context to histogram boosting increases non-serve recall by 1.9 points and serve recall by 2.7 points. Precision falls by 4.8 points and F1 falls by 1.6 points. There is no evidence that the best tree needs absolute broadcast context to work.

These controls do not prove that every physics feature is causal. Body-relative geometry can still inherit camera and detector behaviour. They show that the full result does not survive when physical values are removed.

## Recommendation

Keep histogram boosting as the cheap reference detector. Do not spend time tuning the random forest.

Build the second search-region version before comparing the tree with BST-X. Generate regions across live court-view scene intervals rather than only inside detected rally spans. Add serve lookback around scene cuts and tracker resets. Freeze that region file, rerun both tree models unchanged, then give BST-X the same centres.

Use two histogram-boosting operating points:

- physics plus validity when false contacts are costly
- physics, context and validity when later-contact recall is more important

BST-X is still the stronger candidate for a final learned detector. It has a better physical bias for full pose and shuttle sequences, and ShuttleSet22 can test whether its gains transfer. The tree result sets a serious target: BST-X should beat 87.9% event F1 at comparable precision on the same regions, then retain the gain on ShuttleSet22.

## Limits

There are only three independent fixture videos. Leave-one-fixture-out prevents direct window leakage, but it is still a small in-domain test.

The trial uses saved detected rally spans. It cannot assess contacts outside those spans. This matters most for serves.

The current negative sampler uses the known three-fixture GT after the label-blind freeze. That is correct for supervised fitting, but ShuttleSet22 or another whole-match dataset is needed before treating the operating threshold as portable.

The reported controls are feature-family ablations. They are not per-feature importance claims.

Event scoring uses the project's established closest-pair greedy matcher so the tree numbers remain comparable with the current detector. A maximum-cardinality sensitivity check leaves the ±5 results unchanged. At ±10 it leaves the histogram-boosting physics total unchanged, and adds one match to physics plus context. At ±15 it adds four and two matches respectively.

## Reproduction record

The feature file is a deterministic `tree-contact-features/1` freeze from standard-stage source commit `ad8da4f`. That commit identifies the saved vision artefact producer, not the new trial scripts. Two independent freezes had the same SHA-256. The retained compressed table, manifest, model predictions and result JSON live under the ignored `scratch/contact_det/raw/tree_trial/` directory.

The tracked scripts are:

- `freeze_tree_contact_features.py`
- `score_tree_contact_detector.py`
- `test_tree_contact_detector.py`

Focused Ruff and pytest checks pass. Whole-repository gate results are recorded in the ignored worklog after the final run.
