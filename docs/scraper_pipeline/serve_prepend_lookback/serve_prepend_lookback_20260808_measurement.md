# Serve-prepend lookback measurement

Measured 2026-08-08 against the reviewed `sset_01`, `sset_15` and `sset_21`
broadcast timelines.

## Decision

Do not build the proposed central-pose serve-lookback prototype.

The evidence-only rule selected 14 triggers and recovered 0 of 136 target
rows. Each target is a rally whose first ShuttleSet stroke was unmatched while
a later stroke matched. All 14 selections were false positives against that
target. Fixed-span injection also recovered 0 of all 164 rallies whose first
ShuttleSet stroke was unmatched.

Nineteen candidates passed shuttle, pose and court-absence evidence. One failed
the existing contact-suppression distance from the accepted anchor. All 18
remaining candidates were on the definitive mask, so the current mask policy
selected nothing. A narrow one-frame exemption accepted all 14 selected
candidates. It changed 14 stroke-count rows and 10 next-server rows without
recovering a target first stroke.

The middle-half and middle-two-thirds pose bands produced identical results.
Widening the central band added no useful evidence.

## Results

| Video | Target rows | Opportunities | Raw candidates | Evidence pass | Suppression pass | Selected | Recovered | False positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sset_01` | 63 | 113 | 125 | 6 | 5 | 3 | 0 | 3 |
| `sset_15` | 39 | 145 | 182 | 8 | 8 | 6 | 0 | 6 |
| `sset_21` | 34 | 86 | 104 | 5 | 5 | 5 | 0 | 5 |
| **Pooled** | **136** | **344** | **411** | **19** | **18** | **14** | **0** | **14** |

These counts apply to either pose band. A candidate passes evidence when it has
a visible clean shuttle track, an absent court detection, and a wrist distance
within the existing body-height threshold. It must also be far enough from the
accepted anchor to pass the existing contact-suppression radius. Selection
keeps the largest impulse per lookback opportunity.

### Percentages and denominators

The committed baseline contains 292 GT rallies. Of these, 136 had an unmatched
first ShuttleSet stroke and at least one matched later stroke. This target is
46.58% of all rallies. Another 28 rallies had no matched stroke. In total, 164
of 292 rallies, or 56.16%, had an unmatched first ShuttleSet stroke.

| Baseline observation | Count | Denominator | Percentage |
| --- | ---: | ---: | ---: |
| Target unmatched first stroke with later match | 136 | 292 rallies | 46.58% |
| Any unmatched first stroke | 164 | 292 rallies | 56.16% |
| Clean first-stroke-centred window | 114 | 136 target rows | 83.82% |
| Clean pre-anchor lookback | 55 | 136 target rows | 40.44% |
| Raw candidate near the target first stroke | 15 | 136 target rows | 11.03% |
| Accepted candidate near the target first stroke | 1 | 136 target rows | 0.74% |
| Target first stroke on the believed mask | 17 | 136 target rows | 12.50% |

The candidate sweep found 411 raw rows across 344 lookback opportunities for
either pose band. Nineteen rows, or 4.62%, passed the shuttle, court and pose
evidence. Eighteen rows, or 4.38%, also passed anchor suppression. Fourteen
triggers were selected. That is 4.07% of the 344 opportunities and 77.78% of
the 18 suppression-passing candidates.

The target recovery was 0 of 136, or 0.00%. Target precision was 0 of 14, or
0.00%. The false-positive share was 14 of 14, or 100.00%. Injection accepted
14 of 14 selected contacts. It changed the stroke count for 14 of 14 affected
rows and changed the next-server value for 10 of 14, or 71.43%.

The reviewed timelines classify the 14 selected frames as:

| Human class | Selected frames | Percentage of selections |
| --- | ---: | ---: |
| `cutaway` | 9 | 64.29% |
| `replay` | 3 | 21.43% |
| `live-non-standard` | 2 | 14.29% |
| `live` | 0 | 0.00% |
| `other` | 0 | 0.00% |

The definitive mask correctly blocks the 12 replay or cutaway selections. It
also blocks two non-standard live selections. Exempting those selections did
not recover a target first stroke, so the false block does not support this
trigger design.

For the 136 target rows, 128 had no raw lookback candidate within the canonical
matching tolerance. This is 94.12%. Six, or 4.41%, had no clean shuttle
evidence at the nearby candidate. Two, or 1.47%, occurred where court detection
was present. No target row reached the central-pose selection stage within
tolerance.

One selected `sset_15` frame was within tolerance of a GT first stroke from a
whole-rally or unresolved miss. Fixed-span injection did not recover that
stroke because the selected candidate belonged to another pipeline span. This
is evidence against contact injection without a separate span reassignment
rule.

## Comparison with the earlier baseline

The 2026-07-31 historical fixture run recorded 137 target rows. This run records
136: 63 in `sset_01`, 39 in `sset_15`, and 34 in `sset_21`. The one-case
difference is in `sset_01`.

The difference is expected because this run uses the explicit UNE
`189c5af58e45d23ae827dde516924194eb238e18` static stride-8 profile from the
project release. The older run used the maintained historical calibration
pins. The new profile is named in the output summary rather than being treated
as the historical fixture.

## Inputs and method

The measurement followed these steps:

1. It loaded the `sset_01`, `sset_15` and `sset_21` ShuttleSet fixtures. Every
   external file read by `build_run_video_inputs` was checked against its
   recorded MD5 before loading.
2. It ran the normal `run_video` chain with the committed raw replay mask. This
   produced the baseline spans, raw contacts, accepted contacts and downstream
   results.
3. It loaded ShuttleSet stroke frames as GT. A target row means the first GT
   stroke was unmatched while one or more later strokes in that rally matched
   an accepted contact. Matching used the existing greedy matcher and
   canonical tolerance: four frames at 25 FPS and five frames at 30 FPS. The
   historical output name `gt_serve_frame` does not establish that the first
   stroke is typed or visibly confirmed as a physical serve.
4. For every claimed pipeline span, it anchored the search at the first
   accepted contact. A span with no accepted contact used its span start. The
   bounded lookback was 21 frames at 25 FPS and 25 frames at 30 FPS.
5. It ran the current raw impulse contact finder in that lookback. Each row was
   joined to the reviewed broadcast class for its exact frame.
6. A row passed basic evidence when the shuttle track was visible and clean,
   court detection was absent, and a usable central pose passed the 1.4
   body-height wrist-distance threshold. The pose selector chose the largest
   valid bounding box whose centre was inside the tested band. Score and slot
   order broke ties.
7. A row anchored to an accepted contact also had to pass the existing contact
   suppression distance. The radius was eight frames at 25 FPS and nine frames
   at 30 FPS. The largest remaining impulse was selected for each opportunity.
8. The current-policy arm kept the definitive replay mask. The evidence-only
   sensitivity arm cleared the raw mask only at selected frames. It was used to
   test what would happen if those exact candidates received an exemption.
9. The counterfactual copied the natural accepted-contact map, added selected
   frames and kept every span fixed. It reran the downstream pipeline and
   recorded target-first-stroke recovery, contact counts, stroke counts and
   next-server changes.
10. The complete sweep was repeated for the middle half and middle two-thirds
    pose bands. Both bands produced the same counts and selected frames.

- Shuttle tracks, raw replay masks, court-presence arrays and scene rows come
  from the GitHub release
  `shuttleset-annotator-heuristic-reference-v1`.
- Pose boxes, scores, keypoints and detection counts come from
  `/scratch/comp320a/ahalperi/sset_measure_189c5af/fixtures` on Bourbaki.
- Every consumed external file is checked against its recorded MD5 before
  loading. The unused keypoint-score arrays are not required.
- The committed mask is the decision baseline. The evidence-only arm is a
  sensitivity test that clears the raw mask only at selected frames.
- Contact injection copies the accepted-contact map, adds selected frames and
  keeps pipeline spans fixed. It does not run a production serve-prepend path.
- Human labels determine whether selected frames are live, non-standard live,
  replay, cutaway or other. ShuttleSet rows supply first-stroke timing. The
  separate visual audit supplies visible-serve truth.

Run from the repository root:

~~~bash
ANNOTATOR_FIXTURES_ROOT="$PWD/local_scratch/issue28-fixtures-189c5af" \
PYTHONPATH=src python -u \
  docs/scraper_pipeline/serve_prepend_lookback/measure_serve_prepend_lookback.py \
  --fixture-profile une-189c5af-static-stride8 \
  --mask-mode committed \
  --labels-dir docs/scraper_pipeline/broadcast_nonstandard_camera_id/data \
  --out docs/scraper_pipeline/serve_prepend_lookback/data/serve_prepend_lookback_189c5af_20260808
~~~

The reload-checked evidence pack is
[data/serve_prepend_lookback_189c5af_20260808/](data/serve_prepend_lookback_189c5af_20260808/).
Its [summary](data/serve_prepend_lookback_189c5af_20260808/summary.json.gz)
records profile pins, label hashes, measurement code hash, baseline rows,
candidate verdicts and counterfactual results.

## Limits

This result covers three reviewed ShuttleSet broadcasts and one specific
candidate rule. It rejects the proposed raw-impulse plus central-pose trigger.
It does not prove that every possible serve detector will fail. A future study
would need a different source of serve evidence and an explicit span
reassignment design before another full-chain run is justified.
