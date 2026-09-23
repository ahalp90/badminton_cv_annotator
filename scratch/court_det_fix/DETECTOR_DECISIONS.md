# Findings and decisions

**Keep G0, G1 and templates while settling reliable selection and fallback.**
SVD12 now reduces experimental generation cost; deeper matching has a confirmed
benefit on Am2. Centre-to-edge fitting remains experimental. Colour has been
measured on fixed fits, but no colour-driven detector improvement has been tried.
CourtKeyNet repair is retired; the replacement scene runtime and dependency
removal remain unfinished. [pickup.md](pickup.md) owns the next actions.

The nine-view stress panel and expanded 27-case corpus are development data,
not representative held-out evaluations. A nearby reference fit, a generated
candidate, an automatic winner and a visually usable court are different
outcomes. Keep those distinctions when interpreting every result below.

| ID | Finding and decision | Evidence |
| --- | --- | --- |
| D01 | The old chain passed 0/11 labelled amateur frames. Broadcast repair lessons survive, but CourtKeyNet is excluded from the replacement design. | [Retirement](evidence/retirement/README.md) |
| D02 | Better line evidence and useful candidate pools did not solve court identity. Preserve earlier inputs, proposal records and contrary examples; do not revive failed acceptance rules. | [Independent proposals](evidence/independent_proposals/README.md) |
| D03 | Midpoint and representative changes caused regressions. Historical fixed-membership SVD fitting is distinct from the now-implemented SVD12 direction-family screen (D12). The proposed per-assignment SVD refit before capping remains unimplemented. | [Direction search](evidence/direction_search/README.md); [historical SVD assessment](evidence/webui_followup3_20260922/ASSESSMENT.md) |
| D04 | Score ordering and the 512-assignment cap discard court-compatible matches on traced pairs. A broader fixed-budget rule helped one case and badly regressed another. That diversity rule remains rejected; the later 640-axis experiment is a separate change (D13). | [Exact traces and stopped probe](evidence/direction_search/README.md#cap-and-duplicate-corrections) |
| D05 | The expanded crossed comparison supports G1 proposal access, but not universally better S1 ranking. G0 retains some complementary coverage. Adding candidates can worsen selection. | [G0/G1](evidence/g0_g1/README.md) |
| D06 | W5 `(4,3)` and `(5,3)` select identical courts, with 21/27 usable development selections. The existing player gate raises this to 24/27. G1 plus templates preserves that count and the same failures. The wider test needs G0 for Am4-319. No tested floor is a sufficient acceptance rule. | [Evaluation](archive/20260922/evaluation_results_20260922.md) |
| D07 | All five corrected person-mask matcher comparisons are complete. Useful broadcast proposals survive, including SS03-17 despite its poor paint winner. Masks alone do not rescue GX5. Historical direction-changing masked arms remain qualified. | [Provenance and repairs](evidence/holistic_admission/box_provenance.md) |
| D08 | On the same registered GX union, per-frame and shared paint ranking identify the played court on 7/7 frames, versus 5/7 with native-only access. Far-end errors remain; these are not precise-fit counts. Shared line ranking selects a wall on all seven. Prioritise stable-view proposal reuse and measure far-marking accuracy. | [Pixel and temporal evidence](evidence/pixel_temporal/README.md) |
| D09 | The wider comparison completes 47 frozen cases and 24 controls. Removing G0 loses the tolerable Am4-319 fit. Both arms accept one of eight labelled non-court controls. Static-grid residuals show inward width bias; SS03-34 has a confirmed inner/outer paint-edge misinterpretation. | [Wider results and checks](archive/20260922/wider_evaluation_20260922.md#completed-numeric-comparison) |
| D10 | A fixed-point polarity check partly corrects the left inset. Across six video-03 scenes, median left-edge disagreement with the shared static grid falls from 3.745 to 1.799 working pixels. Scores are mixed; the approved GX control moves under 0.05 pixels. The subsequent centre-label investigation is recorded in D11; fitting defaults remain unchanged. | [Seven-case polarity test](edge_polarity/README.md) |
| D11 | Centre-to-edge relabelling is preferred in the user's four-case review and is a partial correction. The eight-source amateur extension did not solve Am1's net-tape identity error. Keep it as an experimental fitting candidate; neither a fixed corner offset nor a changed physical stripe width is justified. | [Local assessment](edge_polarity/local_audit/ASSESSMENT.md) |
| D12 | SVD12 is the default in fresh experimental W5 generation; full16 remains available. Nine development cases retain all eight approved automatic witnesses and best reference-agreement candidates, but lose three score-winner roles with mixed substitutes. The user accepted this efficiency tradeoff. The completed benchmark saves 52.9% summed matcher time; it is not an end-to-end or general accuracy guarantee. | [Implemented state](svd_runtime/README.md); [matcher timing](svd_runtime/RESULTS.md) |
| D13 | In six G0-only cases, deeper axes (640 versus 512) cost 26.6% more summed full-trial time and give a visually suitable Am2 result where baseline fails. Wider shortlists cost 10.0% more without a comparable visual win. Deeper Am2 and both deeper SS03 selections reproduce saved W5 geometry. Retain G1/templates; bad G0-only fits do not establish that SVD caused the failure. No new depth default is adopted. | [Completed search comparison and user rulings](svd_search/WORKLOG.md) |
| D14 | Automatic colour cues are not adopted. The floor trial rejects 0/9 saved choices; final hue-only paint rejects 0/12 (one usable comparison, eleven inconclusive). Hue-grouped raw colour rejects Am1 but still fails a pale same-hue control. Keep weak/neutral colour inconclusive. Similar paint cannot certify geometry. | [Colour trials and limits](colour_consistency/PLAN.md) |
| D15 | The final net-evidence packet has been pre-evaluated. It does not justify another net experiment before integration. Its ownership idea overlaps the completed colour trials; their negative result does not validate a new net cue. The old net scorer penalises missing evidence and must not be restored unchanged. | [Net assessment](NET_EVIDENCE_ASSESSMENT.md); [resume](pickup.md) |
| D16 | Carry automatic stripe polarity forward as the preferred fitting integration candidate. The user finds it practically identical to the bright-paint rule and both better than the original in the six-case gallery. Automatic polarity handles synthetic dark-stripe inversion with modest extra cost; real dark-court robustness is unproven. Preserve original labels when unresolved. Am1 remains a separate assignment/acceptance failure. Current production fitting is unchanged. | [Final edge policy and timing](edge_polarity/local_audit/ASSESSMENT.md) |

### Interpret the earlier colour diagnostic gallery correctly

It displays preserved fits, not geometry changed by colour. Saved W5 and saved
G1/templates use one earlier ranking with different source access: 50 of 64
available fits have identical geometry, 14 differ, and seven records have no
fit. Source IDs and rendered coordinates match the saved records. Duplicate
columns therefore say nothing about whether colour can improve selection.
The user confirmed the repaired shared-template UI works; no colour efficacy
ruling follows. Future visual review should answer a specific decision question
and show genuine before/after behaviour when testing an intervention.

## What counts as good enough

The recorded 24/27 W5 result means qualitatively usable **for development**.
It is not a count of clean fits. The GX temporal result of 7/7 means that paint
ranking identifies the played court; it does not mean seven accurate fits.
Do not silently relabel either historical count under a stricter standard.

Use three separate judgements in the next visual review:

- **Clean fit:** visible boundaries and service markings follow the court
- **Tolerable fallback:** the right court has a limited, visible error that
  could be accepted on a rare difficult case; record the affected region
- **Unacceptable fit:** wrong court or substantial missing/extra court area;
  also record any exceptional last-resort use separately

User review places frame-0 G1 `143:158` in the fallback category: it is skewed
and clips a sliver of the far-left court. GX86088 G1 `16:44` is a more serious
last resort. Its close near-end alignment hides loss of nearly the whole far
long-service-to-baseline strip. An error of that size at any reasonable
frequency would block shipping. These are the temporal candidates, not an
automatic ruling on their separately refined W5 children.

Review far-end crops alongside full images. Record far-baseline, long-service
and sideline errors separately from near-end error. Where annotations permit,
measure missing court area as well as landmark distance. Report error severity
and frequency, not one pooled “usable” rate. The user clarified the standard on
22 September: imperceptible misalignment is the visual ideal. Following the
outer edges of the white markings is preferred, not a minimum deployment
threshold. Numerical errors support the visual review; this comparison sets no
new pixel cutoff or acceptable fallback frequency. Borderline fits need user
review.

## Older ideas worth bringing back

These mechanisms preserve useful history; they are not an active experiment
queue. The user now expects the colour work and final net-evidence lead to close
the search for new ideas. Use older findings to inform bounded integration and
failure handling. [pickup](pickup.md) owns the remaining order and stop rules.

| Priority | Lead and why it is promising | What is missing or unsafe |
| --- | --- | --- |
| Core design | **Sparse multi-frame agreement and stable-view proposal reuse.** Decide a scene's court from several sampled frames, with shared proposals and supporting observations. The complete GX comparison identifies the played court on 7/7 frames with the common pool, versus 5/7 with each frame's own pool. | This is an access gain, not accurate fitting or a camera-change detector. Check camera stability and support in each sampled image. The failed shared line median rules out that selector, not the need for multi-frame agreement. |
| Historical net/refit evidence | **Net-image evidence plus pooled-fragment refinement.** The older three-frame Yellow replay changed worst-corner error from 221.19 px with floor scoring to 26.01 with net evidence, then 21.22 after pooled refinement. W5 has same-view refinement but not this combined temporal method. | These are 1280×720 errors against old references, not the current working-pixel metric. Letterboxed worsened from 10.06 to 14.75; the net cue and refinement were not isolated temporal ablations. Player-pair identity and camera stability remain prerequisites. |
| Historical fitting evidence | **Constrained stripe/marking refinement, with explicit far-end assessment.** Older fixed-assignment refits sometimes improved a plausible start. Keep parents beside children and inspect both near and far markings. | W5 already has same-view fixed-assignment refinement, so simply adding a refit is not a new mechanism. Across 357 old parent/refit pairs, an improvement rule chose 35 harmful refits. A lower fitting objective does not certify better court geometry. |
| Alongside reuse | **Scene-repair safeguards without CourtKeyNet.** Borrowed geometry should agree with the target's own visible lines. Keep inferred repairs from becoming donors. Separate mixed broadcast composites from ordinary views. | Those safeguards came from broadcast repair; they do not validate amateur detection. Camera novelty, colour and temporal consistency alone cannot certify a court. |
| Implemented experimental speed gain | **SVD12 screening and deeper matching.** The family screen reduces matching; the later 640-axis trial recovers useful Am2 geometry. | D12/D13 give the measured evidence. Per-assignment SVD fitting and graph search remain different, untested mechanisms; the current scope does not require trying them. |
| Conditional on a traced coverage failure | **Better retention before candidate caps.** Exact traces show useful assignments discarded before whole-court scoring sees them. | The tested diversity rule helped one pair and harmed another. Screen all pairs and regression controls before changing the 512-assignment cap; this is distinct from W5's 256-template retention limit. |
| After the search rules settle | **Graph search / shared partial states.** Reuse equivalent partial assignments rather than revisiting the same work along many search paths. | This is a speed/implementation lead, not a new accuracy result. Define state equivalence from the final matcher constraints and verify equivalent outputs before adopting it. Its complexity makes it a later step. |

The [older temporal replay account](../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/evaluation/temporal_assessment.md#an-earlier-shared-court-experiment-already-exists)
includes the Yellow/Letterboxed/Centre comparisons and links the saved code.
That replay uses footpoints and raw fragments; the historical spatial-mask
error does not invalidate it. Its net cue measures image support along
projected net segments. The newer W5 projected-post diagnostic only records
geometry, so it is not a test of that image cue.

See [refit evidence](evidence/independent_proposals/README.md#other-branches-and-why-their-records-remain-useful),
[donor safeguards](evidence/retirement/README.md#non-model-lessons-worth-keeping)
and [cap traces](evidence/direction_search/README.md#cap-and-duplicate-corrections).
Keep global midpoint/direction replacements, connected-pixel vetoes and
junction-first ranking out of the main design for now. Do not equate the old
fixed-group SVD fitting diagnostic with the implemented family screen.
Retain player
evidence, but compare rejection/fallback behaviour rather than assuming a
hard gate is always preferable to a ranking feature.

### Scene-level operation is the target

Sample several frames from a scene or verified stable camera view. Share
proposal search and aggregate compatible line, paint and player observations
across those samples. Require multi-frame support before assigning the court
to the scene, then reuse it with cheap checks for camera change or loss of
support. A failed check should trigger resampling/recalibration or abstention.
Choose the sample count, spacing and agreement rule through the next tests.

The detector should neither decide a whole scene from one frame nor repeat
full independent search on every frame. Agreement must include visible court
geometry, especially at the far end; a wall can score consistently across
frames. The existing fixed-pool replay tests only part of this design. It does
not yet implement multi-frame generation, a scene sampler or change handling.

## Completed evaluation protocol

The wider evaluation and its input coverage are historical records. The
[archive map](archive/README.md) links the original plan, execution and results.
[pickup.md](pickup.md) owns today's priorities and the planned finishing steps.

## What remains before a detector can ship

Address the Am4-319 regression before removing G0. Yellow14 and Am1-54
remain difficult identity/coverage cases; Am1 also needs net-tape rejection. The user excludes unreliable edit-transition
views such as SS21-10 from required detection; confidence rejection is optional.
Retain player evidence while testing rejection and fallback on new
cameras. Preserve the greyscale and prepared-measurement reuse already present
in the experimental runners. Removing unused diagnostics remains a small
optional compute change; the [compute audit](svd_runtime/COMPUTE_AUDIT.md)
records its scope. Historical 2.37× local timing and current Carmack savings
are not deployment latency estimates.

Representative labelled evaluation, ordinary-hardware latency with player
detections supplied, camera-change handling and annotator integration remain
separate requirements. Development search time on Carmack is not deployment
latency. Remove CourtKeyNet and accidental model/weight-loading dependencies
before the branch finishes.
