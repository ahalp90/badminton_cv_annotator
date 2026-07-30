# Current annotator measurement verification

## Bottom line

The fixed CUDA measurement succeeded for all eight configurations. The
retrieved record passed its input, output, schema and arithmetic checks.

The main comparison used three ShuttleSet videos with stride-8 TrackNet
outputs. Live CourtKeyNet/OpenCV court detection gave slightly better contact
F1 than the supplied static homographies, with slightly lower rally coverage.
The difference is small enough that live detection remains the operational
default. Static homography remains a controlled reference and a manual
fixed-camera fallback.

The measurement ran from source commit
`189c5af58e45d23ae827dde516924194eb238e18`. The tracked record was added at
repository merge `726b155afad87ee1beca7062d498b4ee4b84eea7`.

## Run and integrity checks

- Requested and resolved device: CUDA
- Elapsed time: 24 minutes 10 seconds
- Fixed cases: four succeeded
- Court configurations: eight succeeded
- Runner output: 103 files totalling about 12.2 MB
- Failures: none

The original retrieval was about 2.25 GB because it also contained the staged
videos, pose arrays and shuttle tracks. The runner output itself was much
smaller.

The verification checked:

- all 20 staged input file sizes and MD5 digests;
- every output file recorded in a closed manifest;
- fixed CSV and JSON schemas and row counts;
- frame-aligned boolean mask lengths;
- shared scene cuts and parent-specific court evidence;
- the rule that every court-invalid frame enters the final exclusion mask;
  and
- every pooled total and timing distribution reported below.

The tracked [summary](./summary.json) contains the per-configuration values.
The generated [run report](./report.md) gives the compact eight-row view.

## What the contact measures mean

**Rally coverage** is the share of the 292 labelled rallies assigned one
corresponding predicted rally span.

**All-contact score** is called `existing_calibration` in the machine-readable
files. Every unique filtered contact predicted by the annotator contributes to
its precision denominator, including contacts in spurious predicted spans.

**Covered-rally timing diagnostic** is called `strict_contacts` in the files.
It assesses contact candidates through each labelled rally's assigned
predicted span. Split and missed rallies have no candidate rows. One merged
predicted span can be assigned to more than one labelled rally, so this
precision value diagnoses timing inside covered or merged spans. It is not a
global deployment precision estimate.

**Base-30 tolerance** means a frame limit specified for 30 fps and scaled to
each video's actual frame rate. For example, the base-30 +/-5 limit becomes
four frames on the 25 fps fixtures and stays five frames on the 30 fps
fixture.

## Main three-video result

The table pools the normal stride-8 cases from `sset_01`, `sset_15` and
`sset_21`. It excludes the separate stride-1 sensitivity case.

| Measure | Static court | Live court detection |
| --- | ---: | ---: |
| Rally coverage | 249/292 = 0.8527 | 241/292 = 0.8253 |
| All-contact precision / recall / F1 | 0.5387 / 0.6793 / 0.6009 | 0.5800 / 0.6675 / 0.6207 |
| Covered-rally +/-5 precision / recall / F1 | 0.6124 / 0.6793 / 0.6441 | 0.6302 / 0.6675 / 0.6483 |
| Covered-rally +/-10 precision / recall / F1 | 0.6865 / 0.7615 / 0.7220 | 0.7066 / 0.7484 / 0.7269 |
| Mean absolute timing error at +/-5 | 1.922 frames | 1.914 frames |
| Median absolute timing error at +/-5 | 2 frames | 2 frames |

Live detection covered eight fewer labelled rallies. It improved all-contact
F1 by 0.0198 and covered-rally +/-5 F1 by 0.0042. Three fixtures do not
establish broad deployment accuracy, but they provide no appreciable
performance reason to keep static homography as the default.

Each `strict_contacts.csv` stores the labelled frame, predicted frame and
signed offset for every row. Recomputing the base-30 +/-5 matches from those
rows reproduced every timing count, mean and median in the older scorer.

## Other end-to-end labels

These scores include earlier rally and contact errors.

| Label | Static court | Live court detection |
| --- | ---: | ---: |
| Rally winner | 111/271 = 0.4096 | 117/271 = 0.4317 |
| Landing half | 64/287 = 0.2230 | 62/287 = 0.2160 |
| Hit height | 976/3127 = 0.3121 | 953/3127 = 0.3048 |

The results are weak. They are not comparable with the historical 91.2%
point-winner result, which supplied the true rally boundaries before scoring.

## Stride sensitivity

Only `sset_01` has a current stride-1 comparison.

| Court mode | Track mode | Precision | Recall | Covered-rally +/-5 F1 |
| --- | --- | ---: | ---: | ---: |
| Static | stride 8, non-overlap | 0.6237 | 0.6789 | 0.6501 |
| Static | stride 1, weight | 0.7674 | 0.4985 | 0.6044 |
| Live detection | stride 8, non-overlap | 0.6155 | 0.6789 | 0.6456 |
| Live detection | stride 1, weight | 0.7710 | 0.4985 | 0.6055 |

Stride 1 raised precision but lost much more recall than it gained. Stride 8
therefore remains the operational default. This one-video comparison does not
settle the producer question because stride and aggregation mode both change,
and the two modes produce different inpaint artefacts. Stride 8 makes the
repeating hallucination pattern easier to spot. The comparison should be
repeated after the inpaint guards improve.

## Rally-boundary and landing diagnostics

The [first/last-stroke buffer analysis](../../first_last_stroke_buffered_search_20260730/README.md)
checked whether a correct contact sat just outside a predicted rally. It found
19 extra correct candidate-to-target associations per court mode, all in
split-rally cases. The same buffers selected 229 wrong static candidates and
223 wrong live-detection candidates. Every correct contact from an otherwise
covered rally was already inside its predicted span.

The buffer was useful for answering the boundary question, but it is too noisy
to use as a matcher. The routine `wide_edge_contacts.csv` diagnostic should be
removed after its direct consumers are updated.

The landing-horizon check asked how often one, two or three seconds after the
last contact were enough for the current landing result. It used detected
annotator spans with a resolved striker, not labelled rallies.

| Court mode | Horizon | Eligible spans | Limited by horizon | Landing changed | Winner changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static | 1 s | 293 | 120 | 46 | 0 |
| Static | 2 s | 293 | 50 | 11 | 0 |
| Static | 3 s | 293 | 14 | 4 | 0 |
| Live detection | 1 s | 265 | 105 | 32 | 0 |
| Live detection | 2 s | 265 | 35 | 5 | 0 |
| Live detection | 3 s | 265 | 6 | 1 | 0 |

Longer observation changed a small number of landing calls. No capped horizon
changed the inferred winner in this run.

## Verified operation trace

The source trace below supports the detailed schematic in the project
overview. PyCharm semantic navigation confirmed the named definitions and
signatures. Direct source inspection established the call and branch
relationships. PyCharm's Python call-hierarchy provider did not resolve the
module-level function names used here, so call hierarchy is not claimed as
evidence.

| Operation | Principal current symbols | Verified relationship |
| --- | --- | --- |
| CLI and run record | [`main`, `_run_cli_measurement`, `run_annotator_measurement`](../../../../src/annotator/e2e_court_annotator.py) | The CLI makes one timestamped run directory. The measurement writes initial files, sets up four cases, runs both court modes, scores successful configurations and closes the run manifest. |
| Inputs and raw cuts | [`_setup`, `_load_case`](../../../../src/annotator/e2e_court_annotator.py), `build_raw_cut_intervals` | Setup verifies the manifest and CourtKeyNet pins. Each case loads the fixed video, shuttle and pose arrays and derives raw scene cuts. |
| Court evidence | [`_run_one_configuration`](../../../../src/annotator/e2e_court_annotator.py), [`build_static_court_evidence`, `detect_scene_evidence`, `build_detected_court_evidence`](../../../../src/annotator/court_evidence.py) | Each case runs once with supplied ShuttleSet homography and once with live CourtKeyNet/OpenCV evidence. Both branches provide geometry, `keep_vote` and `court_present`. |
| Rally and replay handling | [`run_video`](../../../../src/annotator/run_video.py), [`tracker_segments`, `build_sticky_result`, `find_rally_spans`, `segment_video`](../../../../src/annotator/rally_segmentation.py), [`build_dead_mask`](../../../../src/annotator/dead_mask.py) | `run_video` resolves fps-scaled settings, builds scene-gated sticky player picks, finds preliminary rally spans to establish the replay mask's in-rally speed baseline, builds and short-run-filters the exclusion mask, adds court-invalid frames, then runs final segmentation. |
| Contacts and labels | [`run_video`](../../../../src/annotator/run_video.py), [`attribute_half`, `fit_alternation`, `pick_landing_to_end`, `rally_verdict`, `build_hit_height_rows`](../../../../src/annotator/point_winner.py) | Filtered contacts feed striker order, stroke count and server inference. The final usable contact feeds landing and winner; each filtered contact feeds hit-height work. |
| Ground-truth scoring | [`_score_configurations`, `_write_scoring_outputs`](../../../../src/annotator/e2e_court_annotator.py), [`score_video`](../../../../src/annotator/calibration/gt_scoring.py) | The master ground-truth table loads during setup but is retained until this scoring stage. Per-set ground-truth files are verified after inference. Ground truth does not enter `run_video`. |
| Post-success publication | [`write_summary_and_report`, `clean_run`](../../../../src/annotator/experiment_records.py) | At current commit `726b155`, the CLI writes the compact report and summary, backs up the non-NPY record when needed, sanitises tracked files and reports ignored NPY size. This wrapper was added after the measured `189c5af` run. |

The maintained calibration
[`run_cli`](../../../../src/annotator/calibration/run_cli.py) and
[`sweep`](../../../../src/annotator/calibration/sweep.py) follow a different
evaluation path. They load frozen `Fixture` arrays and masks, then call the
same `run_video` core. The sweep can replace the stored mask with
`--mask-npy`.

## Evidence boundary

This verification checks the retrieved files and their arithmetic. It does not
rerun CourtKeyNet, the annotator or the scorer. No threshold changed after the
run.

The ignored NPY arrays are published in the
[ShuttleSet annotator heuristic reference arrays v1 Release](https://github.com/ahalp90/badminton_cv_annotator/releases/tag/shuttleset-annotator-heuristic-reference-v1).
The Release contains four shuttle-track arrays with their producer CSVs and
inpaint sidecars. It also contains, for each of the eight configurations, the
court-valid, keep-vote, replay and final exclusion masks.
