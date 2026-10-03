# Understanding the annotator results

The main measure of success is a **complete, correctly annotated rally**.
Good contact detection helps, but a single remaining mistake can invalidate a
rally. Contact and whole-rally scores show different aspects of an improvement.

## What the scores mean

- **Contact precision:** the share of predicted hits that match a labelled hit
- **Contact recall:** the share of labelled hits the annotator finds
- **Contact F1:** a combined precision and recall score; it describes contact
  timing, not the correctness of a whole rally
- **Correct-side matches:** hits with both the right timing and the right
  known court half, `Top` for far and `Bot` for near
- **Complete rallies:** all labelled contacts matched one-to-one, correct known
  sides, no extra contacts, and valid clip bounds
- **Review precision:** the share of a selected review queue that is correct;
  its meaning depends on queue size and the treatment of unjudgeable clips

`Top` and `Bot` identify physical court halves, not named players. A player
changing ends changes court-half label.

The current comparison allows a timing error of **±10 frames at 30 fps**, scaled
to the source frame rate. Some earlier reports use ±5 frames. A wider allowance
is easier to satisfy, so those scores cannot be compared directly.

## Selected model and useful baselines

The model selected on 3 October 2026 uses the new court inputs and keeps the
optional unreliable-shuttle/no-player rejection rule off. The experiment calls
this configuration **base**.

The table below uses one common population: 46 ShuttleSet22 videos, 3,327
labelled rallies and 37,184 labelled contacts. All rows use the same cleaned
labels and ±10-frame allowance.

| System | Complete rallies | Matched contacts |
|---|---:|---:|
| Historical annotator on old courts | 1,763 (53.0%) | 33,551 (90.2%) |
| Freshly fitted models on old courts | 1,734 (52.1%) | 33,529 (90.2%) |
| **Selected model on new courts** | **1,744 (52.4%)** | **34,200 (92.0%)** |

The selected model finds more labelled hits, but its complete-rally count is
still 19 below the historical system. Against the fresh old-court fit, it gains
223 complete rallies and loses 213. The net gain of ten hides substantial
changes to individual rallies. Court inputs, fitted models and training order
changed together; this comparison does not isolate the effect of court quality.

The [full comparison](reports/model_selection.md) records per-video changes,
missed rallies, player assignment, review queues and court failures. The
[refit investigation](reports/refit_regression.md) explains why a new fit can
differ even when the refactored code reproduces the historical outputs.

## Which videos were used?

**Original ShuttleSet** supplied 40 development videos. Groups A–D contain 32
videos; group V contains eight validation videos. The contact model used for
validation is fitted on A–D. The final contact model uses all 40 videos. Both
model bundles use sequence and confidence models trained on A–D.

**ShuttleSet22** supplied the cross-dataset evaluation. Earlier studies used 47
videos and 3,422 cleaned labelled rallies. Later comparisons exclude video 15
because its labels are misaligned, leaving 46 videos and 3,327 rallies. The
original-ShuttleSet video `sset_15` is different footage and remains included.

ShuttleSet22 was initially held out from fitting, but its errors have since
guided development decisions. Both validation and ShuttleSet22 results were
inspected for the latest selection. A new team needs fresh footage to measure
generalisation independently.

## Why historical totals differ

The early side-assignment study reports **901 / 3,982 predicted sections** at
±5 frames. The later sequence studies report **1,763 / 3,422 labelled rallies**
at ±10 frames. One counts predicted clips and the other counts labelled rallies;
the timing rules and label populations also differ. Each is a milestone within
its own experiment.

Current coverage includes wholly missed labelled rallies in the denominator.
A *reached* rally has at least one labelled hit recovered somewhere in the video
output; it can still have missing hits, extra hits or bad clip bounds. An
*unjudgeable* predicted section lacks enough label evidence to certify it.
Unjudgeable sections still contribute to the review workload, although their
correctness is unresolved.

## Comparing a proposed change

A useful comparison fixes the video population, labels, timing allowance and
scorer. Its provenance includes the code revision, input versions, model files,
library versions and seed. Related videos stay in the same data split. Later
models train on predictions made while the earlier model held out that video's
group, so they learn from realistic errors.

Complete-rally gains **and losses**, contact precision and recall, side errors
and wholly missed rallies reveal changes that a net total can hide. Per-video
results show whether the benefit is widespread or concentrated in a few cases.

Review comparisons count correct, wrong and unjudgeable clips at fixed queue
sizes. An automatic-acceptance threshold needs separate evaluation. The report
also identifies whether it checks saved outputs, fresh annotation or a full
extraction and refit, because each covers different stages.

[Reproduction](reproducing.md) maps these checks to the available runners.
The generic [refit guide](../../docs/annotator/retuning.md) defines training inputs,
group splits and saved-model evaluation.
