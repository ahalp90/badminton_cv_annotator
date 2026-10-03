# New-court annotator: base selected

**Use base, with the optional nomination veto off.** Base keeps the existing
contact-candidate policy. Veto adds a rule: reject a possible hit when the
shuttle track is flagged as unreliable and no player has been selected nearby
in time.

The evaluation covers eight held-out validation videos and 46 previously
inspected ShuttleSet22 videos. A complete rally requires all labelled contacts
within ±10 frames at 30 fps (scaled to source fps), correct player sides, valid
bounds and no extra contacts. ShuttleSet22 has already been used to investigate failures and develop changes,
so its results describe performance on familiar footage.

Veto gets three more validation rallies completely right, but four fewer on
ShuttleSet22. It also matches three fewer labelled hits on each dataset. Its
highest-ranked review items are a little better, but that advantage fades as
more items are reviewed. Base
keeps the simpler candidate policy.

Against the fresh annotator fitted with the old, faulty court inputs, the new
base recovers **10 more complete rallies and 671 more labelled contacts** on
ShuttleSet22. Complete-rally recovery rises from **1,734/3,327 (52.1%) to
1,744/3,327 (52.4%)**; contact recall rises from **90.2% to 92.0%**. That small overall gain hides a lot of change: 223 rallies become fully
correct and 213 stop being fully correct. Some failures improve substantially;
other rallies get worse.

## What was tested and how it was scored

The goal is complete rally annotations with correct contact times and player
sides, plus a useful ordering for human review. This comparison tests a new
court release and whether an additional candidate-rejection rule helps it.

| Arm | Candidate policy | Fitted models |
|---|---|---|
| Base | Existing policy | Fresh contact, sequence and confidence trees |
| Base + rule | Reject masked candidates with no nominated player | Same base trees; inference-only diagnostic |
| Veto | Same rejection rule | Shared base contact tree; separately fitted sequence and confidence trees |

The veto combines two checks. First, the shuttle guard must have flagged the
track as unreliable, normally with grade 1, 2 or 3. Second, the player selector
must have found neither player at any of five nearby time samples. Selecting
a person is enough to keep the candidate; a usable wrist position is not required.
Rejected candidates are removed before nearby duplicate hits are suppressed
and before the later models try to repair rallies. All three arms use the same
whole-video net position to decide which side of the court a hit belongs to.

The two evaluation sets use different contact trees so that each video stays
outside the contact tree's training data:

| Evaluation set | Videos | Labelled rallies | Labelled contacts | Contact-tree training |
|---|---:|---:|---:|---|
| Validation (V), from original ShuttleSet | 8 | 668 | 5,696 | `bundle_v32`: the 32 development videos in groups A–D |
| ShuttleSet22 | 46 | 3,327 | 37,184 | `bundle`: all 40 development videos |

The sequence and confidence models use A–D in both bundles. ShuttleSet22 video
15 is excluded because its labels are misaligned. The development video named
`sset_15` belongs to the original ShuttleSet dataset and stays included.

Both evaluation sets were inspected before the final model choice. Earlier
ShuttleSet22 investigations had also helped motivate the veto, so that dataset
checks behaviour on familiar footage rather than providing a fresh test on
unseen videos.

A **complete rally** has every labelled contact matched one-to-one within
±10 frames at 30 fps, scaled to source fps, with correct known player sides,
no extra contacts and valid section bounds. A rally with an unknown side cannot
be certified complete. Coverage counts distinct labelled rallies and includes
wholly missed rallies in its denominator.

The results also distinguish rallies that were partly recovered from those
missed entirely. A **reached** rally has at least one labelled hit matched
somewhere in the model's video output. Its remaining hits or clip boundaries
may still be wrong. **Unjudgeable** sections are predicted clips whose labels cannot settle whether
they are correct. Contact **precision** asks how many predicted hits match a
label; contact **recall** asks how many labelled hits the model finds.

## Results across both datasets

### Validation: a small, mixed veto effect

| Measure | Base | Base + rule | Veto |
|---|---:|---:|---:|
| Complete rallies / 668 | 273 (40.9%) | 270 (40.4%) | 276 (41.3%) |
| Reached but incomplete / wholly missed rallies | 333 / 62 | 336 / 62 | 331 / 61 |
| Predicted / matched contacts | 5,924 / 5,214 | 5,921 / 5,211 | 5,916 / 5,211 |
| Contact precision / recall | 88.01% / 91.54% | 88.01% / 91.49% | 88.08% / 91.49% |
| Timing matches with correct known side / 5,696 | 4,889 | 4,885 | 4,918 |
| Correct / wrong / unjudgeable predicted sections | 273 / 338 / 143 | 270 / 341 / 143 | 276 / 336 / 142 |

Simply turning the rule on makes validation worse: 270 complete rallies
instead of 273. Retraining the models that choose contact sequences and rank
rallies for review brings the total to 276, three above base. The +29 correct-side
matches are concentrated: `sset_31` accounts for +26, while losing three timing
matches. Most of the side-assignment gain therefore comes from one video.

### ShuttleSet22: base retains a small coverage advantage

| Measure | Base | Veto |
|---|---:|---:|
| Complete rallies / 3,327 | 1,744 (52.4%) | 1,740 (52.3%) |
| Reached but incomplete / wholly missed rallies | 1,355 / 228 | 1,358 / 229 |
| Predicted / matched contacts | 40,246 / 34,200 | 40,216 / 34,197 |
| Contact precision / recall | 84.98% / 91.98% | 85.03% / 91.97% |
| Timing matches with correct known side / 37,183 known-side labels | 33,156 | 33,153 |
| Correct / wrong / unjudgeable predicted sections | 1,744 / 1,273 / 1,004 | 1,740 / 1,277 / 1,004 |

One of the 37,184 labelled hits has no known player side. That leaves 37,183
hits against which side assignments can be checked. A hit counts in the
correct-side row only when both its timing and its player side match the label.

Base + rule was not run on ShuttleSet22. Veto removes 30 predicted contacts but
also loses three timing matches and three correct-side matches. Its largest
per-video complete-rally gain is +3 (video 37); its largest loss is −2
(videos 9, 11 and 34). The small validation advantage reverses here.

### Changes to individual rallies matter more than the net totals

Each cell below is **gained / lost** when the same labels are checked against
base and the alternative. For example, 13 / 10 means 13 previously incomplete
rallies become correct while ten previously correct rallies become incomplete.
A correctly sided hit is also a timing match, so those two rows overlap.

| Paired change | V: Base + rule | V: Veto | ShuttleSet22: Veto |
|---|---:|---:|---:|
| Complete rallies | 1 / 4 | 13 / 10 | 44 / 48 |
| Label timing matches | 1 / 4 | 25 / 28 | 112 / 115 |
| Correct-side timing matches | 0 / 4 | 63 / 34 | 228 / 231 |

The saved outputs identify errors in all 58 rallies that were complete under
base and incomplete under the fitted veto. Missing hits are the most common
problem, often at the start of the rally:

| Error in the best-matching predicted clip | Validation: 10 lost rallies | ShuttleSet22: 48 lost rallies |
|---|---:|---:|
| Missing at least one labelled hit | 9 | 40 |
| Missing the first labelled hit | 9 | 27 |
| Incomplete clip bounds | 6 | 10 |
| Extra contacts | 5 | 21 |
| Wrong player side | 1 | 10 |

A clip can have several errors, so the rows overlap. For each labelled rally,
the audit chooses the predicted clip with the most matching hits. Ties favour
the clip with fewer kinds of error, then the lower section ID.

Whole-video first-contact changes are less one-sided: veto gains/loses 16/17
on V and 50/46 on ShuttleSet22. Thus veto damages some previously complete
rally starts without reducing aggregate ShuttleSet22 first-contact recall.
The missing hits were not necessarily rejected by the veto itself. Retraining
and choosing a new sequence can change other contacts in the rally too.

## How much better is this than the old-court annotator?

Both old models are compared on the same 46 ShuttleSet22 videos and cleaned
labels as the new base. The **historical annotator** is the original saved
system. The **fresh old-court refit** is the recent retraining on the old court
inputs. Keeping those baselines separate matters: the new base is slightly
ahead of the recent refit, but still behind the original system in complete rallies.
The fresh baseline is the recent refit on scikit-learn 1.9.1. It uses per-scene
player tracking but whole-video geometry for contact-side assignment, as does
the new base.

| System | Complete rallies / 3,327 | Matched contacts / 37,184 |
|---|---:|---:|
| Historical annotator, old courts | 1,763 (53.0%) | 33,551 (90.2%) |
| Fresh refit, old courts | 1,734 (52.1%) | 33,529 (90.2%) |
| **New-court base** | **1,744 (52.4%)** | **34,200 (92.0%)** |

Against the fresh old-court refit, base gains 223 complete rallies and loses
213: **+10, or +0.30 percentage points**. Predicted contacts rise by 818, matched labels by 671, and unmatched predictions
by 147. These are net changes across the two complete streams. Contact precision consequently changes little: 85.04% → 84.98%.
Against the historical model, base gains 224 complete rallies and loses 243,
leaving it 19 below that benchmark despite 649 more timing matches.

The distribution is uneven:

| Video | Fresh old-court complete rallies | New base | Gained / lost |
|---|---:|---:|---:|
| 53: earlier corrupted-court case | 9 | 32 | 26 / 3 |
| 17: earlier far-player-selection case | 18 | 20 | 2 / 0 |
| 12 | 21 | 29 | 10 / 2 |
| 39 | 38 | 46 | 12 / 4 |
| 42: largest regression | 28 | 15 | 1 / 14 |
| 19 | 56 | 47 | 3 / 12 |

Video 53 alone adds 23 complete rallies against the fresh old-court model;
the other 45 videos together lose 13. Against the historical output, video 53
improves from 195/937 to 880/937 matched contacts and 7/76 to 32/76 complete
rallies. Video 17 improves from 17/73 historical complete rallies to 20/73.
These counts show that the annotation output improved. Checking the court
outlines and player picks themselves would require looking at the same video
frames before and after the change; that visual check was not part of this pass.

The learned models appear to have handled poor court inputs surprisingly
well: the new system recovers many more hits, but only a few more complete
rallies overall. A rally still fails the strict check if even one hit, side or
boundary is wrong, and the large gains and losses across videos largely cancel
out. Together, those effects explain why the rally total moved much less than
hit recovery.

The comparison combines new courts with a fixed contact-training row order
and freshly fitted models. It therefore measures the whole refit, including
how the later models assemble rallies and rank them for review. Repeated fits
with several random seeds would be needed to establish whether the small
overall gain persists across fits.

Earlier analysis found 2,374 of 3,633 missed contacts (65.3%) in scenes whose
court detection was rejected. That figure describes **where the missed hits
occurred**. The proportion of court detections that were wrong was not measured
by that analysis.

## What still fails in the selected base?

Base leaves **395/668 validation rallies** and **1,583/3,327 ShuttleSet22 rallies**
incomplete. Most are reached at least once: 333 and 1,355 respectively. Better
contact recall therefore has substantial room to improve complete sequences.
There are still 482 unmatched validation labels and 2,984 on ShuttleSet22.

| Label position | Validation: timing matches / labels | ShuttleSet22: timing matches / labels |
|---|---:|---:|
| First contact | 445 / 668 (66.6%) | 2,722 / 3,327 (81.8%) |
| Middle contact | 4,230 / 4,405 (96.0%) | 28,746 / 30,570 (94.0%) |
| Last contact | 539 / 623 (86.5%) | 2,732 / 3,287 (83.1%) |

A one-hit rally is counted in the first-contact row only. Each labelled hit
therefore appears in exactly one row. The first-contact weakness is especially pronounced on V. Correct-side
first-contact recovery is lower again: 384/668 (57.5%) on V and 2,587/3,327
(77.8%) on ShuttleSet22. These are positional first contacts; the audit does not
independently establish shot type.

For hits the model does find, timing is usually close: half the matches are
within one frame of the label, and 95% are within four frames, measured at
30 fps. That leaves the missing hits out of the calculation. They, along with
wrong sides, extra hits and badly placed clip boundaries, remain the more
useful problems to investigate.

Video 42's 14 losses against the fresh old-court model include ten with missing
matches, nine with side errors, six with extras and two with incomplete bounds.
Video 19's 12 losses include nine with missing matches and two with no overlapping
section. A clip can have several of these errors at once. The counts describe the
predicted clip with the most matching hits; they do not yet explain why the
model chose it.

The next bounded diagnosis should start with video 42 and the missing first
contacts on V. The saved outputs show which hits are missing. To find the cause, a follow-up
would need to trace a hit through court acceptance, player selection, the list
of possible contact frames and the model's choice of rally repair. Those
intermediate records were not collected here. This pass therefore leaves open
how often each input problem causes a miss, which candidates the veto directly
removed, and whether the court outlines look right. The existing results were
enough to choose base; no fitting or inference rerun was needed.

## Does confidence give a better review queue?

The practical question is: if someone checks the highest-ranked clips first,
how many complete, correct rallies will they get? The table compares the same
number of clips from each model within each dataset. Each cell is
**correct / wrong / unjudgeable**; unjudgeable means the labels cannot settle
the clip's correctness. The correct clips here each recover a different rally.
These counts illustrate the trade-off at several workloads; they do not select
a review budget for deployment.

| Population and number reviewed | Base: correct / wrong / unjudgeable | Veto: correct / wrong / unjudgeable |
|---|---:|---:|
| V: 100 | 80 / 19 / 1 | 85 / 14 / 1 |
| V: 200 | 164 / 34 / 2 | 162 / 35 / 3 |
| V: 400 | 231 / 155 / 14 | 231 / 155 / 14 |
| ShuttleSet22: 500 | 407 / 84 / 9 | 423 / 67 / 10 |
| ShuttleSet22: 1,000 | 802 / 169 / 29 | 798 / 174 / 28 |
| ShuttleSet22: 2,000 | 1,356 / 462 / 182 | 1,361 / 478 / 161 |

Veto's best argument is the smallest queue: +5 correct rallies in validation's top 100,
and +16 in ShuttleSet22's top 500 (81.4% → 84.6% of all reviewed sections).
That advantage disappears at 200 and 1,000 respectively. Retraining also changes the review scores. Using the same score cutoff for
both models would therefore give reviewers different numbers of clips.
At the historical 0.757 threshold, base keeps 725 ShuttleSet22 sections
(591 correct, 117 wrong, 17 unjudgeable); veto keeps 712 (584, 114, 14).
No new threshold is selected. Human review time was not measured.

## Final configuration, limits and evidence

**The final model choice is the new-court base `bundle`: contact cutoff 0.9,
`reject_masked_without_player=False`, and `side_geometry=video`.** Use the
40-development-video bundle for final annotation; `bundle_v32` exists for the
held-out V evaluation. The choice favours the simpler policy, essentially tied
aggregate quality, and the absence of a consistent veto benefit across both
datasets. The launch's conservative rule also favoured base when rally gains
came with timing or side losses. Keep the veto as an evaluated optional setting.

The selected bundle loaded successfully with Python 3.12.13 and scikit-learn
1.9.1, matching its fitting environment. Annotation requires that same
scikit-learn version; the local analysis environment's 1.8.0 is incompatible.
The model files are committed in
[`data/annotator/sset_and_sset22_trained_20261003T041112Z`](../../../data/annotator/sset_and_sset22_trained_20261003T041112Z/).
The branch awaits merging.

The new inputs are the 86-video `court_sharing_patched` release at court revision
`e0151791dd1c525311b497d460eff65b4c455021`; annotator code is
`97b5e4de524fe2698492c7f72aceb2a5fddee6a3`. Contact sampling retains the historical
per-fit order, then selected training rows are fitted in ShuttleSet video-ID
order. Development labels come from `shots_master.csv`; ShuttleSet22 uses the
[retained cleaned labels](../good_court_refit/shuttleset22_labels.json.gz). Private run paths and
full input snapshots remain with the run. The [reproduction guide](model_refit.md) records
how the inputs and two builds were prepared.

The labels used here tell us about hits and player sides. Winner, landing and
hit-height fields still need their own checks before a dataset rebuild can rely
on them. The annotator is useful for producing draft rally annotations and
putting promising clips first for review; many clips still need correction.

Compact evidence retained with this report:

- [Selected model record](../good_court_refit/evidence/selected-model.json.gz): settings, environment,
  revisions and SHA-256 fingerprints of both model files.

- [Validation: base versus veto](../good_court_refit/evidence/validation-base-vs-veto.json.gz) and
  [base versus inference-only rule](../good_court_refit/evidence/validation-base-vs-base_inference_veto.json.gz).
- [ShuttleSet22: base versus veto](../good_court_refit/evidence/test-base-vs-veto.json.gz), including
  every paired gained/lost rally and contact identity.
- [Saved-stream analysis](../good_court_refit/evidence/saved-stream-analysis.json.gz): review budgets,
  contact positions, all 58 veto rally losses, old/new paired rally identities
  and bounded diagnoses of the largest regressions.
- [Old-court findings](refit_regression.md) and their retained
  component comparisons establish the fresh baseline. The
  [historical evaluation tables](../../../scratch/annotator_wrapup_evaluation/evaluation_tables.md)
  establish the separate historical benchmark.
