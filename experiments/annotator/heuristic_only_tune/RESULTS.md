# Tuned heuristic and full annotator

17 October 2026

On ShuttleSet22, the tuned heuristic reaches **60.38% correct-player contact F1** and **107 of 3,327 fully correct rallies (3.22%)** at ±10 base-30 frames. The full annotator reaches **85.64%** and **1,744 rallies (52.42%)**.

The follow-up [known-rally comparison](KNOWN_RALLIES.md) evaluates these same
settings with the core MonoTrack benchmark's supplied rally windows and court inputs.

On training, tuning substantially improves the recovered historical heuristic settings. The full system still finds more true contacts, assigns more of them to the right player, and reconstructs many more complete rallies. These are the best settings found by this staged search; the search does not establish a global optimum.

## Test comparison

Both conditions use the same 46 videos, 37,184 labelled contacts and saved shuttle, pose and scene-court inputs. Both discover rally boundaries. The heuristic disables all eight learned annotator trees. Neither condition had an inference failure.

| Metric | Heuristic ±10 | Full ±10 | Heuristic ±5 | Full ±5 |
|---|---:|---:|---:|---:|
| Contact precision | 67.18% | 84.98% | 63.95% | 83.25% |
| Contact recall | 82.83% | 91.98% | 78.85% | 90.10% |
| Contact F1 | 74.19% | 88.34% | 70.62% | 86.54% |
| Correct-player precision | 54.68% | 82.38% | 52.28% | 80.82% |
| Correct-player recall | 67.42% | 89.17% | 64.47% | 87.47% |
| Correct-player F1 | 60.38% | 85.64% | 57.74% | 84.01% |
| Fully correct rallies | 107 (3.22%) | 1,744 (52.42%) | 66 (1.98%) | 1,394 (41.90%) |

A fully correct rally must be one discovered sequence containing exactly one complete labelled rally, with every contact and player correct. Splitting or merging rallies fails this rule. The earlier MonoTrack report’s fixed-window discovery score was 58.43%; **52.42% is the full annotator’s actual discovered-rally score used here**. The fresh run reproduces its previous 1,744-rally total exactly.

At ±10, the heuristic predicts 45,850 contacts: 30,800 match labels and 25,070 also have the right player. The full annotator predicts 40,246: 34,200 match and 33,156 have the right player. False positives anywhere in the video count against precision.

Requiring the right player lowers heuristic F1 by 13.80 points, compared with 2.70 points for the full annotator. Player assignment therefore contributes 11.10 points to the 25.26-point gap in correct-player F1.

Rally boundaries are another major weakness. Of 3,306 heuristic sequences touching exactly one labelled rally, 2,013 (60.9%) fail to contain all its labelled contacts; the full annotator has 128 such failures among 2,968 sequences (4.3%). These are sequence counts, so multiple fragments can refer to the same rally. The scorer checks containment first; those sequences may also have contact or player errors.

## Search and selection

The search used the frozen original ShuttleSet split: 32 training videos with 2,691 rallies and 27,571 contacts; then eight validation videos with 668 rallies and 5,696 contacts. Training first ranked pooled correct-player F1 at ±10, then followed with a local search for fully correct rallies. Training results determined all range extensions. ShuttleSet22 was not used to tune the heuristic.

The two best-score columns can describe different settings.

| Training stage | Settings | Best cumulative player F1, ±10 | Most cumulative fully correct rallies, ±10 | Wall time |
|---|---:|---:|---:|---:|
| Initial search | 128 | 54.54% | 44 | 26.0 min |
| Contact refinement 1 | 145 | 57.69% | 54 | 29.4 min |
| Contact refinement 2 | 132 | 59.62% | 70 | 26.8 min |
| Wrist-distance probe | 35 | 63.80% | 99 | 7.3 min |
| Rally refinement | 156 | 63.85% | 101 | 29.6 min |
| Final wrist values | 5 | 63.85% | 101 | 1.4 min |

The 601 settings required 19,232 video jobs and 120.4 minutes of search wall time, excluding setup and time between rounds. Each job used one numerical-library thread, with up to 26 workers. The queue admitted individual video/configuration jobs as workers became available. Every training job completed without an inference failure.

The initial search used recovered tighter settings, one-parameter changes and seeded joint draws. Contact refinement explored adjacent values and selected parameter pairs. Training winners prompted wider ranges for several parameters. The wrist gate was then exposed as a dimensionless setting, retaining its original default of 1.4 player-box heights. Five default-value controls exactly reproduced all earlier per-video scores. The final probe checked a low-range rally winner and values between the contact and rally leaders.

The top three training settings per objective produced five distinct validation candidates. Both validation objectives chose the same setting. The shortlist was frozen in `bb91d32d`; the final settings were frozen in `af167eac`, before the test runs.

| Setting | Body-height half-window | Wrist distance | Train player F1, ±10 | Train rallies | Validation player F1, ±10 | Validation rallies |
|---|---:|---:|---:|---:|---:|---:|
| ff98c89d0992 (selected) | 3 | 0.8 | 63.83% | 99 | 60.94% | 32 |
| c77e02bb18ce | 6 | 0.8 | 63.82% | 98 | 60.80% | 31 |
| fe32987dbd95 | 12 | 0.8 | 63.85% | 99 | 60.77% | 31 |
| 9cee49d9e694 | 12 | 0.6 | 62.21% | 101 | 60.31% | 31 |
| 30248f157977 | 12 | 0.5 | 57.71% | 101 | 54.93% | 28 |

The `c77e02bb18ce` setting also uses `blip_max_frames=6`; the other shortlisted settings use 12.

The selected setting leads validation by only 0.14 F1 points and one rally over the next candidate. Those small differences do not establish that its exact parameter values are generally better. The earlier integer starting point scored 42.51% player F1 on training; the selected setting scores 63.83%. That gain is relative to the historical starting point, not the shipped loose configuration.

The selected body-height half-window of 3 is the smallest tried. A value of 1 was available in an extended range but was never generated by the local search. Smaller windows remain untested.

## Selected settings

Frame counts and speeds use the existing base-30 convention. Distances in body heights and the impulse multiple are dimensionless. The opening mode is `REGION_START`; its start-speed and minimum-start-frame settings are retained but inactive under that mode.

The table lists the values actually tried across all stages, including the historical starting settings.

| Parameter | Selected | Values tried |
|---|---:|---|
| `blip_max_frames` | 12 | 6, 12, 24 |
| `body_unit_half_window` | 3 | 3, 6, 12, 24 |
| `contact_dedup_radius_frames` | 7 | 1, 2, 3, 3.6, 5, 7, 9 |
| `contact_impulse_multiple` | 2 | 1.5, 2, 3, 4, 6, 8 |
| `contact_suppression_radius_frames` | 12 | 3, 5, 7, 9, 10.8, 12, 15 |
| `end_rest_frames` | 54 | 18, 24, 30, 36, 54, 72, 90, 108 |
| `gap_state_demotion_bound` | 30 | 15, 30, 75, 150 |
| `impulse_floor_half_window_frames` | 18 | 6, 9, 12, 14.4, 18, 24 |
| `replay_mask_min_frames` | 15 | 5, 15, 30 |
| `rest_speed` | 0.016666667 | 0.0016666667, 0.0025, 0.0041666667, 0.0083333333, 0.01, 0.016666667, 0.02 |
| `rest_window` | 5 | 3, 5, 9, 15, 18, 25 |
| `smooth_window` | 2 | 1, 2, 3, 5, 6, 7, 9 |
| `start_min_frames` | 2 | 1, 2, 3, 3.6, 4, 6 |
| `start_speed` | 0.025 | 0.0083333333, 0.0125, 0.016666667, 0.025, 0.03, 0.041666667 |
| `wrist_distance_body_heights` | 0.8 | 0.25, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.8, 2.2 |

## Frame rates and limits

Training has 16 videos at 25 fps and 16 at 30 fps. Validation has one at 25 fps and seven at 30 fps; all test videos are 30 fps. The ±5/±10 tolerances become ±4/±8 at 25 fps. Frame-rate groups also differ in footage, so their score differences cannot be attributed to frame rate alone.

| Selected setting | Videos | Player F1, ±10 | Fully correct rallies, ±10 |
|---|---:|---:|---:|
| Train overall | 32 | 63.83% | 99/2691 (3.68%) |
| Train 25 fps | 16 | 62.62% | 52/1418 (3.67%) |
| Train 30 fps | 16 | 65.21% | 47/1273 (3.69%) |
| Validation overall | 8 | 60.94% | 32/668 (4.79%) |
| Validation 25 fps | 1 | 37.33% | 4/64 (6.25%) |
| Validation 30 fps | 7 | 62.87% | 28/604 (4.64%) |

The comparison preserves each system’s existing geometry policy: scene-specific net geometry for the heuristic, and the fitted bundle’s video-wide geometry for the full annotator. It therefore measures the difference between these complete systems, including that architectural difference.

The full model’s contact tree was fitted on all 40 original ShuttleSet videos. Its base configuration was chosen over an optional veto after both validation and ShuttleSet22 scores had been seen; ShuttleSet22 cases also helped motivate that veto. ShuttleSet22 stayed out of this heuristic search, but had already influenced development and selection of the full model. See the [model-selection report](../reports/model_selection.md) and [selected bundle](../../../data/annotator/sset_and_sset22_trained_20261003T041112Z/README.md).

## Saved results and reproduction

The [run instructions](README.md#run) describe the input paths and commands. [Selected settings](results/selected.json.gz), [shortlist](results/shortlist.json.gz), [test labels](results/test_labels.json.gz), and [selected result tables](results/selected_tables.json.gz) are saved with this report. Each training-stage directory contains all settings, per-video scores and runtime. The test directories contain the [heuristic streams](results/test/predictions.json.gz) and [full streams](results/test_full/predictions.json.gz), plus their summaries. The streams retain discovered spans and per-contact player assignments.

Both test arms ran concurrently with 13 workers each. The remaining arm could use 26 after the first finished. The heuristic arm took 117.2 seconds; the full arm took 156.6 seconds. These are elapsed times for this shared run, including input loading and scoring.
