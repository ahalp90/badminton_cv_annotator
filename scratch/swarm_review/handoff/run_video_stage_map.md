# Verified stage map of `run_video()`

Produced 2026-08-04 by gpt-5.6-luna (reasoning effort max), read-only source
analysis at main commit 1afc86a, commissioned by the review orchestrator.

`run_video()` is defined at `src/annotator/run_video.py:199-231`, with the executable body running from `src/annotator/run_video.py:242-649`. The line references below use `RV:Lx-Ly` for `src/annotator/run_video.py` lines `x` through `y`.

## Execution-order stage map

### 1. Reset caller capture and validate horizon options — `RV:L242-L262`

- Purpose: clear the caller-owned capture fields, validate `landing_horizons_s` as finite positive and strictly increasing, and copy a supplied `raw_exclusion_mask` into the capture [RV:L242-L262]
- Later-stage handoff: the same `capture` object is written again by the optional-path mask code, the normal-path mask code, and the horizon loop [RV:L349-L357; RV:L420-L430; RV:L610-L627]. This stage produces no computation value for the normal chain [RV:L242-L262]
- Parameters consumed: `capture`, `landing_horizons_s`, and `raw_exclusion_mask` [RV:L242-L262]
- Candidate seam: `prepare_run_capture(capture, landing_horizons_s, raw_exclusion_mask) -> None`, retaining the existing `capture` mutation because later stages use the same object [RV:L242-L262; RV:L349-L357; RV:L420-L430]

### 2. Resolve configuration, validate the mode contract, and normalise homography rows — `RV:L263-L329`

- Purpose: resolve `base` against `fps`, construct `span_options`, reject incompatible or missing mode inputs, and convert an object exposing `to_dict` to record rows [RV:L263-L329]
- Later-stage handoff: `resolved` supplies the hallucination-mask grades, sticky-cache constants, replay-mask settings, smoothing mode, serve options, and landing constants [RV:L330-L339; RV:L353-L365; RV:L405-L440; RV:L529-L555]. `span_options` is passed to span finding and segmentation [RV:L359-L367; RV:L384-L386; RV:L402-L404; RV:L434-L440]. The normalised `homography_rows` value is passed to sticky construction and replay-mask construction [RV:L327-L339; RV:L390-L410]
- Parameters consumed: `base` and `fps` are consumed by resolution [RV:L263-L272]; `serve_start`, `spans`, `court_optional`, and `stop_after_segmentation` are consumed by mode checks [RV:L273-L279]; `homography_rows`, `court_present`, `bboxes`, `scores`, `kps`, `ndet`, `resolution`, `video_id`, `gate_court_info`, and `gate_resolution_table` are consumed by the normal/optional input checks [RV:L281-L312]; `landing_options`, `net_band`, `court_info`, `homo_df`, and `landing_error_band_m` are consumed by the full-chain checks [RV:L313-L325]
- Candidate seams: `resolve_config(base, fps) -> resolved` already has a narrow call boundary [RV:L263-L263]. `normalise_homography_rows(homography_rows) -> homography_rows` can contain the `to_dict('records')` conversion [RV:L327-L328]. The combined mode-validation block is not narrow as one helper because its required and rejected inputs differ between `court_optional` mode, segmentation-only mode, and full-chain mode [RV:L274-L325]

### 3. Build the event-rule shuttle hallucination mask — `RV:L330-L332`

- Purpose: adapt either `inpaint_codes` or `shuttle_hallucination_mask` into the boolean mask used by later event rules, while retaining source codes when codes were supplied [RV:L330-L332]
- Later-stage handoff: `shuttle_hallucination_mask` is passed to replay-mask construction and later gates final contacts, landing windows, landing selection, and rejection recording [RV:L390-L410; RV:L501-L560]. `source_codes` is passed to rejection recording [RV:L511-L560]
- Parameters consumed: `track`, `inpaint_codes`, and `shuttle_hallucination_mask`, together with `resolved.rejected_grades` from stage 2 [RV:L330-L332]
- Candidate seam: `_build_shuttle_hallucination_mask(len(track), resolved.rejected_grades, inpaint_codes, shuttle_hallucination_mask) -> (shuttle_hallucination_mask, source_codes)` [RV:L330-L332]

### 4. Build the scene-gated sticky cache — `RV:L333-L339`

- Purpose: when `court_optional` is false, derive tracker segments from the homography and court-presence inputs, then build the sticky result over those segments [RV:L333-L339]
- Later-stage handoff: `sticky` supplies serve-start evidence, tracker distances for segmentation, half attribution, and landing kinematics [RV:L413-L415; RV:L434-L440; RV:L463-L482]
- Parameters consumed: `track`, `homography_rows`, `court_present`, `bboxes`, `scores`, `kps`, `ndet`, `video_id`, `gate_court_info`, `gate_resolution_table`, and `resolution`, plus the resolved body-unit half-window [RV:L334-L339]
- Candidate seam: `build_scene_sticky(track, homography_rows, court_present, bboxes, scores, kps, ndet, video_id, gate_court_info, gate_resolution_table, resolution, body_unit_half_window) -> sticky` [RV:L335-L339]
- Control coupling: the entire stage is skipped when `court_optional` is true, and the optional path returns before the normal consumers of `sticky` run [RV:L334-L378]

### 5. Run the court-optional segmentation path — `RV:L340-L378`

- Purpose: default or validate the raw exclusion mask, filter short exclusion runs, either segment the video or materialise injected contacts, and return a partial `AnnotatorResult` without court-dependent outputs [RV:L340-L378]
- Later-stage handoff: `definitive_exclusion_mask` is consumed by the `segment_video` call within this stage [RV:L353-L364]. `final_spans` and `raw_contacts` are consumed by the partial result construction, and the return prevents them from entering the normal stages [RV:L365-L378]
- Parameters consumed: `court_optional` selects the path [RV:L340-L341]; `raw_exclusion_mask`, `track`, `positions`, `spans`, `contacts`, and `capture` are read or updated in the path [RV:L342-L378]. `resolved.constants.replay_mask_min_frames`, `resolved.smoothing_mode`, and `span_options` supply filtering and segmentation settings [RV:L353-L367]
- Candidate seam: `run_court_optional_segmentation(track, positions, raw_exclusion_mask, spans, contacts, replay_mask_min_frames, smoothing_mode, span_options, capture) -> AnnotatorResult`, with the partial result retaining the empty downstream fields shown in the return [RV:L341-L378]
- Local coupling: the seam has two internal behaviours, because `contacts is None` invokes `segment_video` while supplied contacts bypass that call and are converted into `ContactCandidate` rows [RV:L358-L372]

### 6. Prepare normal-mode spans, injected contacts, replay mask, and serve options — `RV:L380-L416`

- Purpose: assert the normal sticky cache, choose or bootstrap rally spans, build the raw replay mask when the caller did not supply one, materialise injected contacts when present, and build serve-start options for the automatic-contact path [RV:L380-L416]
- Later-stage handoff: `raw_exclusion_mask` goes to final mask validation and filtering [RV:L390-L411; RV:L418-L424]. `final_spans`, `raw_contacts`, and `serve_options` feed the segmentation/adoption stage [RV:L434-L442]
- Parameters consumed: `contacts`, `spans`, `raw_exclusion_mask`, `track`, `fps`, `court_present`, `homography_rows`, `cut_frames`, `keep_vote`, `shuttle_hallucination_mask`, `serve_start`, and `resolution` [RV:L381-L416]. The stage also consumes `resolved` and `span_options` from stage 2, and `sticky` from stage 4 [RV:L381-L415]
- Candidate seams: the injected-contact branch can be `prepare_injected_contacts(track, spans, contacts, raw_exclusion_mask, fps, court_present, homography_rows, cut_frames, keep_vote, shuttle_hallucination_mask, resolved, span_options) -> (final_spans, raw_contacts, raw_exclusion_mask)` [RV:L381-L398]. The automatic-contact branch can be `prepare_detected_contacts(track, spans, raw_exclusion_mask, fps, court_present, homography_rows, cut_frames, keep_vote, shuttle_hallucination_mask, serve_start, sticky, resolution, resolved, span_options) -> (final_spans, raw_exclusion_mask, serve_options)` [RV:L399-L416]
- Awkward coupling: one helper for the whole stage would need branch-specific outputs, because `raw_contacts` is made only when contacts are injected, while `serve_options` is made only on the non-injected branch when `serve_start` is supplied [RV:L381-L416]. The next stage consumes the resulting values through one shared call site [RV:L434-L442]

### 7. Validate and finalise the definitive exclusion mask — `RV:L418-L432`

- Purpose: enforce frame alignment and at least one live frame, filter short exclusion runs, optionally add invalid-court frames, and copy the definitive mask into the capture [RV:L418-L432]
- Later-stage handoff: `definitive_exclusion_mask` is passed to final segmentation, contact filtering, safe-window calculation, and window-end comparisons [RV:L434-L455; RV:L529-L549]
- Parameters consumed: `raw_exclusion_mask`, `track`, `capture`, `court_invalid_is_excluded`, `stop_after_segmentation`, and `court_present` [RV:L418-L430]
- Candidate seam: `finalise_exclusion_mask(raw_exclusion_mask, track_length, replay_mask_min_frames, court_invalid_is_excluded, stop_after_segmentation, court_present, capture) -> definitive_exclusion_mask` [RV:L418-L432]
- Awkward coupling: the `stop_after_segmentation` flag controls whether court-invalid frames are added even though the segmentation-only return occurs later [RV:L427-L450]

### 8. Segment automatically or adopt injected contacts — `RV:L434-L442`

- Purpose: call `segment_video` when contacts were not injected, otherwise retain the prepared spans and contacts, then bind the final values to the public `spans` and `contacts` locals [RV:L434-L442]
- Later-stage handoff: `spans` and `contacts` feed the segmentation-only return and the full-chain contact filtering, attribution, landing, hit-height, and final-result stages [RV:L442-L450; RV:L452-L649]
- Parameters consumed: `contacts`, `track`, `positions`, `spans`, together with `definitive_exclusion_mask`; the call also consumes `sticky.distances`, `serve_options`, `resolved.smoothing_mode`, and `span_options` from earlier stages [RV:L434-L442]
- Candidate seam: `segment_or_adopt_contacts(track, positions, definitive_exclusion_mask, sticky_distances, serve_options, final_spans, smoothing_mode, span_options, injected_contacts) -> (spans, contacts)` [RV:L434-L442]
- Awkward coupling: the same local names are used first as inputs or branch outputs and then rebound by `spans, contacts = final_spans, raw_contacts`, so a helper must represent both the injected and generated-contact cases [RV:L381-L442]

### 9. Return the segmentation-only result — `RV:L444-L450`

- Purpose: when `stop_after_segmentation` is true, return spans and contacts with every downstream result collection empty [RV:L444-L450]
- Later-stage handoff: there is no later stage in this invocation because the return occurs before scoring, attribution, landing, and hit-height work [RV:L444-L450; RV:L452-L649]
- Parameters consumed: `stop_after_segmentation` selects the return, and `spans` and `contacts` populate it [RV:L444-L450]
- Candidate seam: `make_segmentation_result(spans, contacts) -> AnnotatorResult` [RV:L444-L450]

### 10. Filter scoring contacts and group them by rally — `RV:L452-L461`

- Purpose: apply `scoring_filter`, remove contacts on definitive exclusion frames, and group both the scored and filtered contact frames by rally [RV:L452-L461]. `scoring_filter` keeps contacts whose wrist gate is not false and whose suppression flag is not true [RV:L22-L25]
- Later-stage handoff: `filtered_by_rally` feeds attribution, per-rally landing/verdict work, and hit-height extraction [RV:L463-L478; RV:L501-L504; RV:L631-L640]. `scored_by_rally` feeds the diagnostic for rallies whose fitted striker is absent [RV:L493-L499]. `filtered_contacts` is returned in the final result [RV:L642-L645]
- Parameters consumed: `contacts` and `definitive_exclusion_mask` [RV:L452-L461]
- Candidate seam: `filter_and_group_contacts(contacts, definitive_exclusion_mask) -> (filtered_contacts, filtered_by_rally, scored_by_rally)` [RV:L452-L461]

### 11. Fit rally attribution and server-side metadata — `RV:L463-L478`

- Purpose: attribute each filtered contact to a half, fit the alternating half sequence per rally, count strokes, derive next servers, and derive each rally's fitted first-stroke half [RV:L463-L478]
- Later-stage handoff: `striker_halves` and `next_servers` feed the per-rally verdict pass [RV:L492-L518; RV:L563-L565]. `n_strokes_list` feeds next-server and first-stroke derivation [RV:L473-L478]. `fitted_first_all` is carried to the final result [RV:L475-L478; RV:L645-L646]
- Parameters consumed: `spans`, `filtered_by_rally`, `track`, `sticky`, `bboxes`, and `net_band` [RV:L463-L478]
- Candidate seam: `fit_rally_attribution(spans, filtered_by_rally, track, sticky, bboxes, net_band) -> (striker_halves, n_strokes_list, next_servers, fitted_first_all)` [RV:L463-L478]

### 12. Build landing kinematics and the active error band — `RV:L480-L487`

- Purpose: build landing kinematics from the track and sticky cache, then select the explicitly supplied landing error band or compute one from the video homography and court inputs [RV:L480-L487]
- Later-stage handoff: `kin` is passed to strict and capped landing selection [RV:L529-L557; RV:L597-L601]. `band_m` is passed to strict and capped rally verdicts [RV:L563-L565; RV:L602-L604]
- Parameters consumed: `track`, `sticky`, `kps`, `resolution`, `landing_error_band_m`, `video_id`, `homo_df`, `court_info`, and `ref_err_px` [RV:L480-L487]
- Candidate seam: `build_landing_context(track, sticky, kps, resolution, landing_error_band_m, video_id, homo_df, court_info, ref_err_px) -> (kin, band_m)` [RV:L480-L487]

### 13. Evaluate each rally's strict landing, verdict, and geometric diagnostic — `RV:L489-L582`

- Purpose: initialise the three per-rally output maps, skip rallies without a fitted striker after recording trusted-mask evidence where applicable, and for usable rallies compute the safe window, landing, shipped verdict winner, and geometric comparison [RV:L489-L582]
- Later-stage handoff: the per-rally locals `final_contact`, `safe_window`, `landing`, `verdict`, and `shipped_winner` are consumed by the nested horizon stage [RV:L527-L582; RV:L583-L627]. `verdict_rows`, `landings`, and `geometric_verdict_rows` are carried to the final result [RV:L489-L491; RV:L642-L648]
- Parameters consumed: `spans`, `rejection_diagnostics`, `track`, `definitive_exclusion_mask`, `shuttle_hallucination_mask`, `landing_options`, `net_band`, `resolution`, and `court_info` [RV:L492-L560]. The stage also consumes `scored_by_rally`, `filtered_by_rally`, `striker_halves`, `next_servers`, `kin`, `band_m`, `source_codes`, and resolved landing constants from earlier stages [RV:L493-L565]
- Candidate seam: `evaluate_rally(rally_id, span, striker, scored_frames, filtered_frames, next_server, track, definitive_exclusion_mask, shuttle_hallucination_mask, source_codes, sustained_loss_frames, kin, landing_options, net_band, resolution, court_info, band_m, rejection_diagnostics) -> (final_contact, safe_window, landing, verdict, geometric_verdict_row, shipped_winner)` [RV:L492-L582]
- Awkward coupling: the loop writes three outer dictionaries, conditionally mutates the caller's `rejection_diagnostics`, and leaves the locals needed by the nested horizon loop in the same lexical scope [RV:L489-L582]

### 14. Optionally compare shorter landing horizons — `RV:L583-L627`

- Purpose: for each requested horizon on a rally with a usable final contact, cap the safe endpoint, recompute landing and verdict, and append one `LandingHorizonRow` comparison to the caller-owned capture [RV:L583-L627]
- Later-stage handoff: rows are stored directly in `capture.landing_horizon_rows`; no later local stage consumes them, although the capture remains visible to the caller after return [RV:L610-L627; RV:L642-L649]
- Parameters consumed: `landing_horizons_s`, `fps`, `capture`, `track`, `landing_options`, `net_band`, `resolution`, `court_info`, and `shuttle_hallucination_mask` [RV:L583-L610]. The stage also consumes `final_contact`, `safe_window`, `landing`, `verdict`, `shipped_winner`, `striker`, `rally_id`, `kin`, `band_m`, `next_servers`, and resolved constants from the surrounding stages [RV:L583-L627]
- Candidate seam: `build_landing_horizon_rows(rally_id, landing_horizons_s, fps, final_contact, safe_window, track, kin, landing_options, striker, net_band, resolution, court_info, constants, shuttle_hallucination_mask, next_server, band_m, strict_landing, strict_verdict, strict_winner) -> list[LandingHorizonRow]`; the caller can extend `capture.landing_horizon_rows` with the returned rows [RV:L583-L627]
- Awkward coupling: this is nested inside the per-rally loop and requires the base pass's final-contact, safe-window, strict-landing, strict-verdict, and shipped-winner locals, while also writing directly to `capture` [RV:L527-L627]

### 15. Compute hit heights for filtered contacts — `RV:L629-L640`

- Purpose: call hit-height construction once per filtered contact, record successful frame-to-height results, and retain `ValueError` details for contacts where hit-height construction fails [RV:L629-L640]
- Later-stage handoff: `hit_height_by_frame` and `hit_height_failures` are passed to the final `AnnotatorResult` [RV:L629-L649]
- Parameters consumed: `spans`, `filtered_by_rally`, `track`, `net_band`, and `resolution` [RV:L629-L640]
- Candidate seam: `build_hit_height_results(spans, filtered_by_rally, track, net_band, resolution) -> (hit_height_by_frame, hit_height_failures)` [RV:L629-L640]

### 16. Assemble the full result — `RV:L642-L649`

- Purpose: construct and return `AnnotatorResult` from the spans, raw contacts, filtered contacts, attribution outputs, landing outputs, and hit-height outputs [RV:L642-L649]
- Later-stage handoff: the return is the terminal handoff for the full-chain path [RV:L642-L649]
- Parameters consumed: no run-time parameter is read directly in this stage; it consumes the local values produced by stages 8, 10, 11, 13, and 15 [RV:L642-L649]
- Candidate seam: `make_annotator_result(spans, contacts, filtered_contacts, filtered_by_rally, striker_halves, n_strokes_list, next_servers, fitted_first_all, verdict_rows, landings, geometric_verdict_rows, hit_height_by_frame, hit_height_failures) -> AnnotatorResult` [RV:L642-L649]

## `homography_rows` input forms

- In normal mode, `homography_rows` must be non-`None`; the same normal-mode block also requires `court_present` [RV:L297-L299]
- In `court_optional` mode, `homography_rows` must be `None` at this function boundary because any non-`None` value is included in `supplied_optional_inputs` and rejected [RV:L278-L296]
- In normal mode, an object with a `to_dict` attribute is converted by calling `homography_rows.to_dict('records')` [RV:L327-L328]
- In normal mode, a non-`None` object without a `to_dict` attribute is left unchanged by `run_video()` and passed to `tracker_segments` [RV:L327-L339]. When a replay mask is built, that unchanged value is also passed to `build_dead_mask` [RV:L390-L410]
- Therefore the forms evidenced at this boundary are: `None` for the court-optional path; a non-`None` `to_dict('records')`-capable object for conversion in normal mode; or a non-`None` already downstream-compatible value passed through unchanged in normal mode [RV:L278-L339]

## Control flow and cross-stage state

### Early returns and skipped stages

- `court_optional` skips sticky construction, enters its own mask/segmentation branch, and returns a partial result at line 373, so the normal replay-mask, attribution, landing, and hit-height stages do not run [RV:L334-L378]
- `stop_after_segmentation` returns after stage 8 and before scoring-contact filtering, so stages 10 through 16 are skipped in that path [RV:L444-L450; RV:L452-L649]
- The `contacts is not None` condition switches between injected-contact preparation and automatic span/mask preparation, and then switches the final segmentation stage between adopting contacts and calling `segment_video` [RV:L381-L442]
- A supplied `raw_exclusion_mask` skips `build_dead_mask`; an absent mask triggers that construction in either normal contact branch [RV:L389-L411]
- `serve_start` is rejected with injected `spans`, and serve options are built only when contacts are not injected [RV:L273-L277; RV:L399-L415]
- Horizon diagnostics run only when `landing_horizons_s` is non-empty and only after the rally has a usable non-hallucinated final contact [RV:L246-L258; RV:L501-L527; RV:L583-L627]
- A rally with no fitted striker skips landing and verdict construction after recording a trusted-mask rejection when all scored contacts were filtered [RV:L492-L500]. A rally whose filtered contacts are all hallucination frames receives a verdict with `landing = None` and skips landing selection [RV:L501-L526]

### Variables crossing or changing stage boundaries

- `shuttle_hallucination_mask` is the input parameter to stage 3 and is rebound to the helper's canonical mask result before later stages read it [RV:L222-L223; RV:L330-L332]
- `raw_exclusion_mask` is copied into capture at entry, defaulted or supplied in the optional path, possibly generated by `build_dead_mask` in normal mode, and then consumed by definitive-mask finalisation [RV:L259-L262; RV:L342-L357; RV:L389-L424]
- `definitive_exclusion_mask` is made in the optional path and again in the normal path; the normal value can then be modified by the `court_invalid_is_excluded` branch before segmentation and filtering [RV:L353-L357; RV:L418-L432]
- `final_spans` and `raw_contacts` are produced in branch-specific preparation or by final segmentation, then rebound into the `spans` and `contacts` locals used by all full-chain stages [RV:L381-L442]
- `serve_options` is initialised before the optional path, populated only by the automatic-contact branch when requested, and consumed by final segmentation [RV:L340-L340; RV:L399-L415; RV:L434-L440]
- `capture` is caller-owned mutable state written during entry validation, optional-mask handling, normal-mask handling, and horizon-row collection [RV:L242-L262; RV:L342-L357; RV:L418-L430; RV:L610-L627]
- `rejection_diagnostics` is a caller-owned list mutated from several points inside the per-rally stage for trusted-mask, final-contact, lost-shuttle, and landing-descent rejections [RV:L493-L500; RV:L505-L514; RV:L534-L543; RV:L558-L561]

## Summary

The body has identifiable handoffs for mask construction, sticky setup, segmentation, attribution, landing/verdict work, horizon diagnostics, hit-height extraction, and result assembly, while the court-optional and injected-contact branches make the segmentation boundary conditional [RV:L330-L450; RV:L463-L649].

The single most awkward coupling is the nested horizon loop because it depends on final-contact, safe-window, landing, verdict, shipped-winner, kinematics, geometry inputs, the event mask, and caller-owned `capture` state from the surrounding per-rally pass [RV:L527-L627].
