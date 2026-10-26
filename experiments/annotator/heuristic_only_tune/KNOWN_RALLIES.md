# Tuned heuristic with known rally boundaries

17 October 2026

Giving the tuned heuristic **two seconds before and three seconds after** each
labelled rally raises correct-player contact F1 from **70.95% to 71.35%** and fully
correct rallies from **221 to 272 of 3,327** at ±10 frames, using our known-rally
courts. Contact F1 rises from **88.70% to 89.13%**. MonoTrack’s courts give almost
identical results. Scoring keeps the original ±1-second windows throughout.
The heuristic parameters are unchanged.

The 2/3-second comparison follows the dataset’s existing clip margins. Extra
context helps the heuristic modestly; the full annotator does slightly worse
with those margins. The original 1/1-second benchmark remains below for comparison.

## Original one-second context: comparison at ±10 frames

| Method | Courts | Contact F1 | Correct-player contact F1 | Fully correct rallies |
|---|---|---:|---:|---:|
| Tuned heuristic | Our known-rally courts | 88.70% | 70.95% | 221 (6.64%) |
| Tuned heuristic | MonoTrack courts | 88.71% | 70.88% | 221 (6.64%) |
| Full annotator | Our known-rally courts | 96.68% | 94.48% | 2,029 (60.99%) |
| Full annotator | MonoTrack courts | 96.62% | 94.41% | 2,024 (60.84%) |
| MonoTrack | Our known-rally courts | 96.02% | 95.85% | 1,607 (48.30%) |
| MonoTrack | MonoTrack courts | 96.02% | 95.82% | 1,602 (48.15%) |

All rows cover the same 46 ShuttleSet22 videos, 3,327 rallies and 37,184 labelled
contacts. The full annotator and MonoTrack rows reuse the published benchmark
results. MonoTrack uses its selected zeroed shuttle input with interpolation off.

## Heuristic precision, recall and tighter timing

| Measure | Our courts ±10 | MonoTrack courts ±10 | Our courts ±5 | MonoTrack courts ±5 |
|---|---:|---:|---:|---:|
| Contact precision | 86.13% | 86.14% | 81.06% | 81.07% |
| Contact recall | 91.43% | 91.43% | 86.04% | 86.04% |
| Contact F1 | 88.70% | 88.71% | 83.48% | 83.48% |
| Correct-player precision | 68.90% | 68.83% | 65.27% | 65.20% |
| Correct-player recall | 73.14% | 73.05% | 69.28% | 69.20% |
| Correct-player F1 | 70.95% | 70.88% | 67.22% | 67.14% |
| Fully correct rallies | 221 (6.64%) | 221 (6.64%) | 140 (4.21%) | 140 (4.21%) |

For comparison, the published results at ±5 are:

| Method | Courts | Contact F1 | Correct-player contact F1 | Fully correct rallies |
|---|---|---:|---:|---:|
| Full annotator | Our known-rally courts | 94.74% | 92.74% | 1,630 (48.99%) |
| Full annotator | MonoTrack courts | 94.68% | 92.67% | 1,626 (48.87%) |
| MonoTrack | Our known-rally courts | 94.83% | 94.75% | 1,352 (40.64%) |
| MonoTrack | MonoTrack courts | 94.85% | 94.73% | 1,347 (40.49%) |

## Inputs and scoring

The supplied windows run from one second before the first labelled contact to
one second after the last. All videos are 30 fps, so the stored half-open bounds
are `[first_contact − 30, last_contact + 31)`. The windows are supplied during
inference as well as scoring for the original comparison. The context checks below
widen only the inference windows. The full-video shuttle tracks and poses stay unchanged.

Our court condition uses the saved video-robust known-rally courts after the person
vote. The MonoTrack condition uses its saved courts with the corner order adapted
for our annotator; every successful fit is accepted without our person vote.
Our saved inputs have 14 rallies without an accepted court: ten without a fit and
four rejected by the player-presence vote. MonoTrack has nine failed fits. These rallies
remain in the denominators. Neither new heuristic run had an inference failure.

The existing core benchmark scorer matches contacts one-to-one within the timing
allowance. Contact precision counts predictions inside the supplied windows.
Correct-player metrics also require the emitted player to match the label.
A fully correct rally requires every labelled contact, no extra contacts and
correct players after the scorer’s existing alternation fit within that window.

The frozen settings are those selected by the [discovery search](RESULTS.md).
Supplying spans bypasses boundary discovery. This comparison also changes court
inputs relative to discovery, so its gain cannot be attributed to boundaries alone.
The heuristic retains scene-specific net geometry; the full annotator retains its
fitted video-wide geometry. This is a comparison of the complete configured systems.

## Discovery under the same benchmark scoring

The core benchmark judges complete rallies inside labelled windows even when
inference discovers its own boundaries. Rescoring the saved heuristic discovery
predictions this way preserves every contact count and changes only the rally measure.
Contacts outside labelled windows still count against discovery precision.

| Discovery inference | Player F1 ±10 | Fixed-window rallies ±10 | Fixed-window rallies ±5 |
|---|---:|---:|---:|
| Tuned heuristic | 60.38% | 330 (9.92%) | 198 (5.95%) |
| Full annotator | 85.64% | 1,944 (58.43%) | 1,558 (46.83%) |

Requiring the actual discovered sequence to contain the whole rally instead gives
**107/3,327 (3.22%)** for the heuristic and **1,744/3,327 (52.42%)** for the full
annotator at ±10. The [discovery report](RESULTS.md) uses that stricter definition.

The supplied-window heuristic finds 3,197 more labelled contacts than discovery,
but also adds 3,123 unmatched predictions inside the scoring windows. It gains
114 fully correct rallies and loses 223, leaving 221 instead of 330. Of those
223 losses, 214 include extra contacts and 176 fail only because of extras.
Most extras in the lost rallies lie before the first or after the last labelled
hit; 197 of 303 are within ten frames of a supplied window edge. This motivates
checking inference context separately from the fixed scoring windows.

## More context before and after each rally

This follow-up checks whether the supplied windows truncate useful inference
context. Every heuristic run uses the frozen settings. All runs use the original
±1-second scoring windows. The requested inference margins are measured from the first and
last labelled contacts. Extra context stops at saved scene cuts, neighbouring-window
midpoints or video bounds. The original scoring windows remain intact.

The court coverage expands with the inference windows, using the same fitted
geometry and acceptance decisions. No court is refitted and no player-presence
vote is repeated. The stored vote fraction changes with the interval duration
to keep the court record consistent; the original acceptance decision is retained.

Known spans bypass rally-start and rally-end discovery. Landing and outcome
inference happen after contact detection and player assignment, so a longer
landing search cannot improve these contact scores. Added context can still
affect player-tracking history, replay masking, contact filtering and the fitted
alternating player order.

### Main comparison: dataset clip margins

The dataset’s [clip settings](../../../src/dataset_builder/features.py) add two
seconds before and three seconds after inferred rally bounds. Here the same
margins are anchored to the first and last labelled contacts. These anchors can
differ from inferred bounds. We use 2/3 seconds as the main comparison with the
1/1-second baseline because it follows that existing dataset convention.

### Same two-second start and three-second end context at ±10

| Method | Courts | Contact F1 | Correct-player F1 | Fully correct rallies |
| --- | --- | --- | --- | --- |
| Tuned heuristic | Our courts | 89.13% | 71.35% | 272 (8.18%) |
| Tuned heuristic | MonoTrack courts | 89.14% | 71.32% | 272 (8.18%) |
| Full annotator | Our courts | 96.49% | 94.27% | 1,934 (58.13%) |
| Full annotator | MonoTrack courts | 96.42% | 94.21% | 1,929 (57.98%) |

### Same two-second start and three-second end context at ±5

| Method | Courts | Contact F1 | Correct-player F1 | Fully correct rallies |
| --- | --- | --- | --- | --- |
| Tuned heuristic | Our courts | 83.89% | 67.58% | 177 (5.32%) |
| Tuned heuristic | MonoTrack courts | 83.89% | 67.55% | 177 (5.32%) |
| Full annotator | Our courts | 94.58% | 92.56% | 1,560 (46.89%) |
| Full annotator | MonoTrack courts | 94.52% | 92.51% | 1,556 (46.77%) |

### Heuristic precision and recall with two-/three-second context

| Measure | Our courts ±10 | MonoTrack courts ±10 | Our courts ±5 | MonoTrack courts ±5 |
| --- | --- | --- | --- | --- |
| Contact precision | 86.94% | 86.96% | 81.82% | 81.84% |
| Contact recall | 91.44% | 91.44% | 86.05% | 86.05% |
| Contact F1 | 89.13% | 89.14% | 83.89% | 83.89% |
| Correct-player precision | 69.60% | 69.57% | 65.92% | 65.90% |
| Correct-player recall | 73.19% | 73.16% | 69.33% | 69.29% |
| Correct-player F1 | 71.35% | 71.32% | 67.58% | 67.55% |

The full annotator loses 95 fully correct rallies with our courts (2,029 to
1,934), and its correct-player F1 falls from 94.48% to 94.27%. The wider context
adds unmatched predictions (425 to 586), while missing contacts fall slightly
(1,989 to 1,977). Of the 142 previously correct rallies it loses, 101 fail only
because of extra contacts; it also gains 47 correct rallies. The wider context
therefore does not close the large gap between the two systems. MonoTrack itself
was not rerun with wider windows; its original reference results remain above.

### Other margins checked

These variants measure sensitivity on the same benchmark. They do not establish
a universally best context length. Two seconds on each side gives the highest
correct-player F1 among the tried heuristic settings; 2/3 seconds gives two more
fully correct rallies. Three seconds before the rally adds no complete rallies
over two seconds when the end margin stays at one second.

The 2/2 and 2/3 heuristic runs have identical in-window contact frames. Player
assignments change in 143 rallies: two become fully correct, but the total number
of correctly attributed contacts falls by 56. This explains their different
rankings on correct-player F1 and fully correct rallies.

### Heuristic with our courts at ±10 frames

| Before / after (seconds) | Contact F1 | Correct-player F1 | Fully correct rallies |
| --- | --- | --- | --- |
| 1 / 1 | 88.70% | 70.95% | 221 (6.64%) |
| 2 / 1 | 88.73% | 70.96% | 233 (7.00%) |
| 3 / 1 | 88.73% | 70.81% | 233 (7.00%) |
| 1 / 2 | 89.11% | 71.43% | 255 (7.66%) |
| 1 / 3 | 89.11% | 71.48% | 256 (7.69%) |
| 2 / 2 | 89.13% | 71.50% | 270 (8.12%) |
| 2 / 3 | 89.13% | 71.35% | 272 (8.18%) |

### Heuristic with our courts at ±5 frames

| Before / after (seconds) | Contact F1 | Correct-player F1 | Fully correct rallies |
| --- | --- | --- | --- |
| 1 / 1 | 83.48% | 67.22% | 140 (4.21%) |
| 2 / 1 | 83.50% | 67.22% | 150 (4.51%) |
| 3 / 1 | 83.50% | 67.07% | 150 (4.51%) |
| 1 / 2 | 83.86% | 67.67% | 162 (4.87%) |
| 1 / 3 | 83.86% | 67.72% | 163 (4.90%) |
| 2 / 2 | 83.89% | 67.73% | 175 (5.26%) |
| 2 / 3 | 83.89% | 67.58% | 177 (5.32%) |

The remaining 2/2-second controls give the following supporting results:

| Method | Courts | Tolerance | Contact F1 | Correct-player F1 | Fully correct rallies |
|---|---|---|---:|---:|---:|
| Tuned heuristic | MonoTrack courts | ±10 | 89.14% | 71.46% | 270 (8.12%) |
| Tuned heuristic | MonoTrack courts | ±5 | 83.89% | 67.69% | 175 (5.26%) |
| Full annotator | Our courts | ±10 | 96.51% | 94.33% | 1,941 (58.34%) |
| Full annotator | Our courts | ±5 | 94.57% | 92.60% | 1,562 (46.95%) |
| Full annotator | MonoTrack courts | ±10 | 96.45% | 94.25% | 1,936 (58.19%) |
| Full annotator | MonoTrack courts | ±5 | 94.51% | 92.52% | 1,558 (46.83%) |

Most of the one-sided end-context gain comes from fewer extra contacts. Moving
from 1/1 to 1/2 seconds removes 352 unmatched predictions inside the scoring
windows, while matched contacts fall by one. The 1/2 and 1/3 runs have identical
in-window contact frames; the third second changes player assignment, adding
20 correctly attributed contacts and one fully correct rally.

Extra context moves the court-mask edges away from the scoring windows. This is
consistent with an edge effect, but the experiment changes several pipeline
stages together and does not isolate its cause.

### Context actually added

| Requested before / after (s) | Clipped starts | Clipped ends | Mean extra before (s) | Mean extra after (s) |
| --- | --- | --- | --- | --- |
| 2 / 2 | 577 | 385 | 0.905 | 0.959 |
| 2 / 3 | 577 | 1,554 | 0.905 | 1.672 |

The means above are additions to the original one-second margins, averaged over
all 3,327 rallies. For 2/3 seconds, 2,750 starts and 1,773 ends receive the full
requested margin. The remaining edges stop earlier at the saved limits. Saved
scene cuts come from automatic detection; they were not manually verified here.

## Saved results and reproduction

Use the [benchmark runner instructions](../monotrack_benchmark/benchmark_scripts/README.md#run)
with `--heuristic-settings` pointing to [selected.json.gz](results/selected.json.gz).
Run `ours_our_courts` and `ours_mono_courts` into fresh directories and score each
with the original benchmark scorer. For the discovery row, [rescore_discovery.py](rescore_discovery.py)
packages the saved test predictions for that scorer without new inference.

The new per-video streams and pooled scores are saved under
[known_our_courts](results/known_our_courts/), [known_mono_courts](results/known_mono_courts/)
and [discovery_benchmark](results/discovery_benchmark/). They reuse the committed
[benchmark manifest and courts](../monotrack_benchmark/benchmark_results/inputs/).
The [published comparison](../monotrack_benchmark/archive/results/comparison/)
contains the reference streams and full precision/recall tables at both tolerances.
[diagnostics.py](diagnostics.py) recomputes the rally gains and losses, failure
reasons and extra-contact counts quoted in this report from these committed files.

The two new inference runs used 13 workers each, with one numerical-library thread
per worker. Together they took 80 seconds of inference wall time, including loading
but excluding scoring. Both completed without inference failures; the
[execution summary](results/known_rallies_execution.json.gz) records the run settings.

The [context results](results/context/) contain every additional per-video stream
and score. Their [input manifests](results/context/inputs/) record the actual
inference windows and requested margins. Use [padding.py](padding.py) as shown in
[the run instructions](README.md#check-additional-inference-context) to recreate
the court coverage, then run inference with those inputs. **Score against the
original benchmark inputs**, or the helper’s `score_inputs` directory, to retain
the original ±1-second evaluation windows.

All 552 additional video jobs completed without inference failures. The three
context batches took 224, 352 and 325 seconds of wall time, including inference
and scoring; the first two also include input preparation. They used at most
25 workers in total, with one numerical-library thread per worker. The
[context execution summary](results/context/execution.json.gz) records the
conditions and timings. Independent checks reproduced pooled counts and spot
scores at both tolerances. Input checks confirmed the original scoring windows,
labels, court geometry and acceptance decisions were retained.
