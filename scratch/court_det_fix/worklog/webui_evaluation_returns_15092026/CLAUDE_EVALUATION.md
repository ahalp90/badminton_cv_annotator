# Evaluation of the WebUI assessment packet against the direction experiment

Written 2026-09-15 by a Claude Code session (Claude Fable 5.1) after the direction experiment finished. Assessment only; no repository files were changed. Written for a colleague who knows the court-detection work but has not followed this branch.

**Bottom line.** The three WebUI assessments and the direction experiment tell one consistent story, and the experiment behaved the way the WebUI said it would before the run. A lower direction residual is not a closer court. A closer control fit is not a better generated court. Coverage-first allocation discards the precise GX0 directions before any representative rule can act, and the midpoint anchor reallocates without recovering them. The ranking failures the WebUI dissected are the same candidates that sit in the experiment's baseline rows, and neither M nor R touches them. Taken together, the packet points away from more single-frame direction tuning and toward two other places: court identity in ranking, and the bounded per-view scoring audit with Amateur-3 as the regression control. Nobody in the packet makes a readiness claim, and nothing here supports one.

## What was read

WebUI returns in this folder:

- `WEBUI_EVALUATION_PROMPTS.md`: the three prompts.
- `does_direction_assessment_preserve_court_geometry.md` and `badminton_direction_audit_7299ff3.zip`: Prompt 1, with its executed synthetic checks in `synthetic_geometry_checks.py` and `synthetic_geometry_checks_output.json`.
- `can_available_evidence_separate_court_from_false_match.md`, `court_ranking_review_7299ff3.zip` and `court_identifiability_extension_7299ff3.zip`: Prompt 2.
- `camera_calibration_assessment.md`, `calibration_per_view_handover.zip` and `court_temporal_extension_handover.zip`: Prompt 3.

Experiment record on `fix/court-det` at commit `9e0dec8`, under `scratch/court_det_fix/direction_agreement/`: `results.md`, `hypotheses.md`, `evidence.md`, the loss-decomposition table in `runs/direction_agreement_20260915_144900/summary.md`, and the E2 and E4 records checked directly. Also the protocol packet `../worklog/CLAUDE_DIRECTION_EXPERIMENTS.md`, the three audit reports under `../worklog/claude_session_15092026_14h32m/`, and the two evidence commits `7299ff3` and `d0c9a12`, including the cached `temporal_assessment.md`.

Each WebUI return has two parts: an initial assessment and a follow-up extension. Both parts pin their reads to the named evidence commit and state what they could and could not decode. The WebUI opened no images and decoded only prefixes of the large compressed records. Its numerical claims are verified where it could decode, and it says so plainly. Its visual claims rest on the written judgements in the repository. I did not run the bundled verification scripts.

Units differ between the two bodies of work. Prompt 2 reports maximum corner distances in 1280 by 720 display pixels. The experiment reports 960 by 540 working pixels. The scale is 0.75. Candidate identity below was matched by candidate ID in the experiment's accounting table, not by converting numbers.

## Prompt 1: the direction objective

**What the WebUI found.** Every direction score in play measures line concurrence in a chosen coordinate gauge, not court accuracy. The foot-anchored angle and the midpoint-anchored angle divide a normal incidence error by a tangential lever arm, so moving the anchor changes which vanishing point wins without changing the observed line. The capped-residual representative score and the unit-normal SVD objective have the same limitation in different forms. The WebUI executed synthetic constructions showing all four scores improving while the fitted court moves away from truth.

The stronger material is its replay of the committed GX0 bank. It reconstructed the 5,671-candidate bank from the 106 merged rows and matched all 16 original selections and both control candidates exactly. It then showed:

- Candidate 122, the control's y-direction, is capped. Its largest overlap with any leader is 7 of 9, below the strict 0.8 suppression threshold, so it sits outside every bucket and no within-bucket rule can reach it.
- Candidate 1183, the control's x-direction, is inside leader 2152's bucket but ranks 27th of 74 under the precision score, so R cannot pick it either.
- A lower bound on maximum corner error for any court that preserves a fixed direction pair. Across all 542 candidates in the bucket union, even with oracle choice, no pair can fit the approved control closer than about 3.5 working pixels. The binding constraint is the y-direction, and the best union candidate for it is 854.
- The closer control pair scores worse than the original pair under the capped-angle score on the original support masks. On shared rows the control pair scores better. Three merged rows that only the original candidates support flip the preference: rows 31 and 61 for x, row 51 for y.

**What the run showed.** I read the E2 record for GX0 directly. Every checkable prediction held:

| WebUI claim, made before the run | Run outcome |
| --- | --- |
| R cannot pick candidate 122; it sits outside every bucket | 122 absent from all four arms |
| R cannot pick 1183; it ranks 27th in leader 2152's bucket | R's representative for that bucket is 802 |
| No union candidate beats 3.50 px; the union's best y-direction is 854 | R chose 854 for that bucket; R's best GX0 fit is 6.18 px |
| M reallocates and might free capped candidates | M shares 1 of 16 GX0 leaders with B; 122 still absent; fit worsens to 19.65 px |

The WebUI also insisted that residual, control fit, generated population and ranked winner are four different results. The experiment's loss-decomposition table is that separation made concrete, and it found losses at each step. R improves the E3 control fit on seven views, mostly by under a pixel, and those gains reach the paint winner on two. R minimises the residual by construction yet regresses E3 on Am3-0 and SS21-20, which is the score-versus-geometry reversal the WebUI built synthetically, now seen on real frames. M's GX0 loss appears at allocation and again at availability: the E3 fit is 19.7 px and the nearest pooled court is 33.8 px. R's Am3-0 loss appears only at availability: the E3 fit is 4.3 px and the nearest pooled court is 44.2 px.

The E1 finding that the midpoint anchor moves the blind spot rather than removing it is the WebUI's "anchor effect without observational ambiguity" in the experiment's own words. The WebUI's interpretation table pre-registered the reading for these outcomes: a no-change or regression under R cannot refute the direction-loss explanation, because the allocated union was already insufficient.

**What remains open.** The WebUI worked without fragment memberships and said it could not attribute the three flipping rows to the anchor or predict whether M would repair them. The experiment recovered the memberships and answered the second part: M made GX0 worse. Nobody has yet looked at what the three rows are. The bound does not cover SVD outputs, which the WebUI states; the experiment's fixed-support SVD on B reaches 3.96 px, just above it.

## Prompt 2: court identity

**What the WebUI found.** The paint profile is a local appearance test with a missing-data denominator and no ownership check, so a perfect profile does not identify a court. A candidate can raise its mean by projecting inconvenient markings outside the image. The floor gate rejects the approved GX0 controls as readily as the false winners, so it is not a discriminator. Choosing line ranking everywhere loses the approved Amateur-2 frame 150 paint court; choosing paint everywhere loses the usable scene 19 line court.

Its extension separated two failure mechanisms:

- On ShuttleSet 03 scene 19 the false winner's five available markings are four sidelines clipped at the image edge and one short-service line. That leaves court length along the sidelines unconstrained. A family of courts differing by about 400 working pixels passes the same profiles and stays under the camera bound. The named-line constraint matrix has rank 7; one more transverse marking would raise it to 8.
- On Amateur-2 frame 28019 the false winner projects the near baseline within half a working pixel of the near long-service line, so two markings share one ridge and both pass. The saved assignments show the competition: the baseline's exclusive support collapses to zero. The approved frame 150 court has the same all-pass profile and a similar but smaller exclusive-support drop, which rules out a simple veto.

**What the run showed.** The experiment's accounting names the same candidates:

| WebUI candidate | Where it sits in the experiment's baseline rows |
| --- | --- |
| Scene 19 paint winner 165:6702 | B all-camera paint winner, 6,297 working px off |
| Am2-28019 paint winner 184:4123 | B all-camera paint winner, 892 working px off |
| Am2-150 line winner 30:30, mat border | B line winner at every stage, 141 px off |
| Am2-28019 line winner 16:1800, mat border | B line winner at both rescoring stages, 145 px off |

Neither M nor R removes any of these. M adds a new all-camera paint failure on scene 19 and R adds one on GX5. Both arms still pick a court about 140 px off by line ranking on both Amateur-2 views. Pair IDs are not comparable across arms, so whether it is the same mat-border court is unverified; the distances match B's. The floor gate passes nothing on GX0, GX5 or Am2-28019 in any arm. This is why the experiment's rescoring stages rescue nothing: those failures live in ranking, not in directions.

**What remains open.** The WebUI's proposed fixed-pool comparison over the 19 exported winners has not been run. Its geometric parts can be computed from saved homographies now. The ownership question, which structures supplied the passing samples, needs image-level judgement.

## Prompt 3: calibration per stable camera view

**What the WebUI found.** The cached evidence at `d0c9a12` is 13 frames from three videos with cached lines and selected directions, no fragment memberships, and no verified camera stability. The WebUI's recommendation is to test independent versus cross-frame scoring on one frozen automatic candidate union, then guarded reuse, before touching pooled-line generation or refinement. It specified a bounded audit: four 90-frame windows, three calibration frames each, B's direction selection frozen, an automatic registration guard with tolerances fixed before outcomes are seen, Amateur-3 as a mandatory regression control, and separate reporting of generation, ranking, acceptance and reuse.

Its extension recovered the old Yellow clip's 33 saved candidates and made the comparison the original headline table lacked. With cues and candidates held fixed, shared scoring across the three frames improves selection: the worst error falls from about 1,271 to 221 px under floor-only scoring and from 221 to 26 px with the net cue. But the wrong candidate beats the closer one on every frame under the floor score alone, so no monotone aggregation rule could rescue it within that saved matrix. The extension also showed that holding out a frame's score is not holding out its geometry, since the good candidate was generated from the held-out frame. This is historical conditional evidence, not a modern Amateur-2 line- or paint-score result.

**How it relates to the run.** The experiment session never studied this evidence, but three of its findings bear on the proposal:

- Amateur-3 is where both M and R regress from a line winner judged essentially perfect, so it is the right regression control.
- The protocol freezes B's selection. Neither M nor R earned promotion, so B is what would be frozen.
- A consistently misleading structure wins under every arm on the Amateur-2 views in the direction experiment. That is a same-frame ranking observation, not evidence that modern temporal scoring is dominated in the way the older Yellow floor matrix was.

**What remains open.** Freezing B means GX0 and GX5 enter the union with directions that cannot produce a usable court, unless some other frame's automatic selection kept the precise ones. The WebUI flags this but had no way to check it. The cached temporal records provide descriptive direction comparisons, but they do not establish attainable fits, generated courts or modern shared scoring on one verified candidate union.

## The story in one paragraph

The detector loses the court in three separable places. Direction allocation discards precise directions through coverage-first bucketing, and neither within-bucket precision nor a different anchor recovers them; on GX0 this is now a proven bound, not a suspicion. The matcher then loses more between the direction fit and the pooled proposals on some views, and the records cannot yet say where. Ranking cannot tell a court from a convincing false match because the paint profile tests local appearance with a missing-data denominator and no ownership check, and the floor gate rejects good and bad alike. Calibration per stable view reduces how often those three steps must succeed, not whether they can. The old Yellow data show a wrong candidate can dominate on every frame in that historical floor-score matrix; that does not establish modern Amateur-2 line/paint-score dominance.

## Next steps implied by the sum

These are suggestions, not decisions. Two are nearly free and should come before the expensive one.

1. **Close the direction experiment as a bounded negative result.** Neither change is promoted and B stays the baseline. The editor and cold-reader passes on `results.md` have not been run. Cost is editing time. Benefit is a citable record for the later audit.

2. **Inspect the three GX0 merged rows with the recovered memberships.** The WebUI named rows 31, 51 and 61 as the rows that flip the score against the control pair. The E0 record holds their member fragments. Looking at which image structures those fragments come from settles whether the loss is a wrong-structure membership or an orientation error. That decides whether any anchor or score change can ever help on GX0. Cost is minutes with existing records.

3. **Check the cached GX directions against the GX0 control pair.** `temporal_records.json.gz` holds the 16 selected directions for seven GX frames, and `gx0_control_measurements.json.gz` holds the two precise directions. Measuring the angular distance tells you whether any automatic selection along the video keeps them. If none does, per-view pooling cannot fix GX from automatic selection and the GX views need a different lever. Cost is minutes. This is the one observation that would tell you whether step 5 can help the hardest views.

4. **Compute the two identity diagnostics on the exported winners.** The constraint rank of available markings and the minimum projected separation between distinct markings are both computable now from saved homographies. They separate "unconstrained", "aliased" and "contradicted" before any score changes. The WebUI's contrary examples show neither is a veto on its own: the approved frame 150 court has a separation under 2 px, and the approved GX0 controls are cropped. The ownership question still needs image-level judgement.

5. **Run the bounded per-view audit as specified in `calibration_per_view_handover.zip`.** Four windows, 17 generation positions, B frozen, one fixed union, independent versus shared scoring, Amateur-3 as control, and three guard sanity checks. From the experiment's timings, the 12 new 1080p generations take roughly 3 to 8 hours serial, or under 2 hours as four streams. The registration guard is new engineering with tolerances frozen before outcomes are seen. Benefit is the one result the WebUI says would change its judgement.

6. **Avoid** anchor or score sweeps, threshold tuning, and promoting M for its GX5 win alone. That win is real, but M's losses land on an approved control and a court judged perfect.

## Confidence and limits

- Candidate identity between the WebUI's analysis and the experiment's baseline rows is certain: matched by ID in `e4/accounting.csv.gz`.
- The GX0 predictions were checked against `e2/gxBQ_window_00_frame_0.json.gz` directly. The bound itself was not recomputed here.
- Whether M's and R's 140 px Amateur-2 line winners are the same mat-border court as B's is unverified.
- The WebUI's synthetic constructions and Yellow-clip computations were not rerun. Its bundles carry their own verification scripts and state their access limits.
- The three audit reports concern the experiment's own integrity and their fixes were applied before commit `9e0dec8`. They did not use the WebUI material and the WebUI did not see them.
