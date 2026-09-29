# Acceptance and benchmark runbook

## 1. Freeze the baseline

Use branch `exp/court-det-opt` at a named SHA and the same environment for baseline and candidate runs. The committed handover identifies the Carmack reference environment as Python 3.12.13, NumPy 2.5.3, SciPy 1.17.1, and OpenCV 5.0.0.93, one numerical thread per worker.

Record:

- commit SHA and dirty diff;
- CPU model, physical cores, NUMA topology, RAM;
- Python/NumPy/SciPy/OpenCV versions;
- worker count and affinity;
- all thread environment variables;
- input pack/feet hashes; and
- whether control files and fresh detections are present.

## 2. Exact deployment-path gate

Before any proposal pruning:

1. Run current D17 and the integrated `exact` path on all 28 views.
2. Compare corrected selection, bounded selection, populations, parent/child records, and arrays using the repository's float-bit-aware comparer.
3. Require identical output except explicitly removed research-only fields and timing/path metadata.
4. Run with at least two hash seeds and two worker completion orders.
5. Require deterministic results.

Performance report:

- process CPU seconds and scene wall seconds;
- p50, p90, p95, p100;
- peak resident memory per worker and aggregate;
- stage self time;
- worker utilisation and queue idle time;
- bytes copied to workers; and
- startup/import time.

## 3. Parallelism gate

Test worker counts 1, 2, 4, 8, 16, and the deployment maximum.

Require:

- identical results at every worker count;
- original pair and parent order restored before stable retention/ranking;
- no oversubscription from BLAS/OpenCV threads;
- no memory-growth failure on GX views; and
- speed-up reported separately for search, W5, and total wall time.

Amdahl diagnostics:

- compute effective parallel fraction from measured 1-worker and N-worker results;
- identify memory-bandwidth saturation;
- distinguish pool startup from steady-state service latency.

The 90 s contract should use a warm persistent service unless cold-start latency is explicitly part of deployment.

## 4. K=8,192 cascade shadow gate

Implement the cascade but continue full exact scoring in shadow mode.

For every one of the 3,369 recorded-style pair calls, save:

- court count;
- coarse timing;
- exact top-K timing;
- exact full timing;
- coarse rank of every full retained court;
- proposal shortlist IDs and full shortlist IDs;
- weakest retained exact score;
- source, view, pair ID, G0/G1 label; and
- whether the downstream final court changed.

Acceptance on the committed 28 views:

- 3,369 / 3,369 pair shortlists identical;
- every selected parent/child/corrected court identical;
- every court/non-court decision identical;
- no new peak-memory regression; and
- measured whole-run wall saving consistent with the 8.5% projection within normal node noise.

Then run the wider scene corpus. The 28-view corpus is a regression set, not a sufficient precision/recall estimate.

## 5. Adaptive K gate

Do not deploy K=4,096 until a fallback trigger is evaluated from saved call-level rows.

Split views/videos by source into development and hold-out groups. Tune no thresholds on the hold-out group.

Required hold-out outcomes:

- zero pair-shortlist misses, or every would-be miss invokes full fallback;
- identical final corrected court/abstention;
- fallback rate low enough to preserve a measurable gain;
- no source class with materially higher fallback or miss rate.

Report fallback rate by:

- GX, amateur, broadcast, letterboxed, and non-court;
- G0 versus G1;
- pair court-count decile;
- visibility/crop status; and
- previous-view reuse versus fresh search.

## 6. Accuracy metrics

Use the full16 current detector as the primary behavioural baseline.

For court views report:

- same candidate/origin ID;
- max corner displacement in working and native pixels;
- median/worst landmark pixel error;
- median/worst floor error in centimetres;
- camera gate and validity result;
- parent and child ranks; and
- whether the bounded net choice and polarity correction agree.

For non-court views report:

- abstention agreement;
- false-court additions/removals;
- which proposal source supplied a false court;
- fresh-detection versus frozen-feet input; and
- fallback path.

Do not use SVD12 as the accuracy reference; it is known to change five court views and worsen two.

## 7. Previous-view reuse gate

Construct positive and negative transitions:

- same camera, same crop, adjacent scenes;
- same camera with lighting/player changes;
- crop/letterbox change;
- hard cut to a different camera;
- non-court inserts/replays;
- scene with temporary line occlusion.

For every transition record:

- camera-signature distance;
- previous-court evidence before and after local refit;
- whether reuse was accepted;
- whether full fallback ran;
- reuse/full output difference; and
- reuse latency.

Acceptance:

- no false reuse across different-camera/crop transitions;
- same corrected court as full search on accepted same-view reuse;
- full fallback on uncertain transitions; and
- subsequent-scene p100 below 90 s on the deployment machine.

## 8. Final latency contract

The final report must state:

- hardware SKU and worker count;
- cold and warm latency separately;
- scene p50/p95/p100;
- first-scene versus reused-scene latency;
- full-fallback p100;
- fallback rate; and
- CPU seconds per scene.

A result that meets 90 s only on average does not meet the requested maximum. A result that meets 90 s by dropping direction groups does not meet the accuracy requirement.
