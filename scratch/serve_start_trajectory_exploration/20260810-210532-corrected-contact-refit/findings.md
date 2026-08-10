# Findings for the corrected experiment

## Bottom line

The corrected motion rule found 11 of 16 clear first returns and made 3 false return calls. Across all 16 covered rallies where the rule fired, directly naming the other player as server was right in 13 cases. Adding an unknown-player contact before the alternating fit was right in 8 cases. Adding a contact by the inferred other player was right in 9 cases.

The simple complete motion rule raises covered-rally server accuracy from 61.8% when using the anchor player alone to 65.9%. The alternating refits barely improve on the released alternating fit. The useful result is therefore the direct motion inference, not the recursive refit.

The stricter check that excludes filled or interpolated TrackNet points has 100% precision and 56.2% recall on the 16 clear returns. The main rule has 78.6% precision and 68.8% recall. Both settings were chosen on these same videos, so neither is a held-out estimate.

The earlier EDA started from `fitted_first_all`, which is the result of the rally-wide alternating fit. It then asked whether motion supported reversing that already uncertain label. The corrected experiment instead anchors the trajectory on the earliest contact's own player attribution.

## Accepted and raw contacts

`assemble_contacts` records raw impulse candidates with wrist-gate and suppression fields. `scoring_filter` accepts a candidate unless `wrist_near` is explicitly false or `suppressed` is explicitly true. `build_contact_data` then removes candidates under the definitive exclusion mask.

The first frame in `filtered_by_rally[rally_id]` is therefore the earliest accepted contact by construction. There cannot be an earlier accepted contact in the same rally. There may be earlier raw candidates that failed a gate, were suppressed, or were excluded. Those candidates are useful diagnostics, but treating them as credible would require another rule that the current pipeline does not define.

Sources:

- `src/annotator/rally/contacts.py`: `assemble_contacts`
- `src/annotator/video_outcomes.py`: `scoring_filter`, `build_contact_data`
- `src/annotator/types.py`: `ContactCandidate`

## The anchor player is an independent measurement

`attribute_half` looks at the accepted frame's shuttle position and sticky player distances. It chooses the nearest sticky slot, then maps that player's bounding-box foot position above or below the net band. It returns `None` when the track or player evidence is unavailable, or when the foot lies in the net band.

The function does not consult the alternating fit. Exact distance ties resolve to the top slot because NumPy chooses the first minimum. The experiment should record this rare edge case rather than silently describe it as strong evidence.

Source: `src/annotator/point_winner.py`: `attribute_half`.

## What the existing fitted server means

For every accepted contact, `build_contact_data` calls `attribute_half`. `fit_alternation` scores the two possible Top/Bot alternating phases against all non-missing guesses. The winning phase names the final striker. `_first_stroke_half` then works backwards through the contact count to produce `fitted_first_all`.

This explains why a missed serve can reverse the fitted first player. It also explains why `fitted_first_all` must not be used to identify the anchor player in the corrected motion test.

Sources:

- `src/annotator/point_winner.py`: `fit_alternation`
- `src/annotator/video_outcomes.py`: `_first_stroke_half`, `build_contact_data`

## An unknown-time prepend has a clean seam

The existing contact-injection helper adds real frame numbers and reruns `run_video`. That path would force an unknown serve to have invented shuttle and pose geometry.

The alternating fit itself only needs an ordered list of `Top`, `Bot` or `None` guesses. Adding one position leaves the phase assignment of every original contact unchanged. It changes the fitted first player through the longer sequence's parity. Supplying an inferred half at the new position also adds one vote to its matching phase; supplying `None` adds no vote.

The corrected refit should therefore show both cases. Prepending `None` isolates the consequence of detecting one missing contact. Prepending the inferred server half measures the same parity change plus the supplied player evidence. Neither case claims temporal localisation.

Sources:

- `src/annotator/calibration/serve_prepend_measurement.py`: `run_contact_injection_counterfactual`
- `src/annotator/point_winner.py`: `fit_alternation`

## Reusable data and joins

The frozen static-homography stride-8 release leaves contain `annotations.json`, `scene_rows.csv` and `court_present.npy`. Matching pose arrays are present under `local_scratch/autograder_architecture/` and match the fixture checksums.

Use `(video_id, set_id, rally)` as the stable GT key. Map GT rallies to predicted spans through `classify_all`; do not equate a GT rally index with a predicted span ID. Load exact ShuttleSet first and second strokes from `collect_shots`, using `ball_round == 1` and `ball_round == 2`. Twenty rallies have no second stroke.

The previous extraction found 292 GT rallies, of which 249 are covered by a predicted span. The earliest accepted contact matched GT contact 1 in 87 covered rallies, contact 2 in 17, contact 3 in 3, and no GT contact within tolerance in 142. One accepted anchor was within tolerance of both contacts 1 and 2, leaving 16 strict contact-2 positives. Separately, four GT first/second pairs lie within the canonical tolerance of each other. These counts must be regenerated and checked rather than copied into the final report.

## Minimal trajectory safeguards

Use a continuous visible run in court view. Require recurrence guard code zero and, as a stricter sensitivity, producer-original points. Require enough observations of the anchor player. Reject stationary paths, gaps and gross single-frame jumps. The other player can remain a diagnostic rather than a coverage gate.

Measure net distance closed on the direct anchor player and the proportion of consecutive steps that close that distance. Keep values in normalised image distance or body heights and state the units.

A quadratic fit is diagnostic only. It has more freedom than a line, so lower error does not prove a parabola or a genuine shuttle. The archived inpaint investigation also found that curved flight and camera cuts made straight-chord tests unreliable.

Sources:

- `src/annotator/inpaint_guard.py`
- `src/annotator/rally/evidence.py`
- `docs/tracknet/evidence/inpaint_fabrications_20260722/detector_options.md`
- `docs/tracknet/evidence/inpaint_fabrications_20260722/stride1_retrack/summary.txt`

## Verification

Read-only reviews traced the contact flow, GT and release loaders, sticky evidence, alternating fitter, and prior inpaint checks. The focused trajectory tests pass 24 synthetic cases. The output checker independently recalculates every threshold row, server table and headline count from the compressed per-rally results.

An independent audit reproduced the main confusion counts, stricter TrackNet comparison, raw-candidate comparison and all three triggered-rally outcomes. Replacing all stored GT fields left every runtime motion decision unchanged. GT is used to choose the displayed threshold, which the report states plainly.
