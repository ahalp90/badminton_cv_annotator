# Court detector findings and decisions

**The experimental direction is to keep G0, G1 and line templates, use bounded net support for selection, and apply automatic stripe correction to the selected fit.** SVD12 reduces matcher work but does not make the detector affordable for deployment. The fresh combination, scene-level reuse and CourtKeyNet removal remain untested or unfinished. [pickup.md](pickup.md) owns the current work order; this file records why the decisions were made.

These are development findings. The nine-view stress panel and 27-case corpus are not representative held-out evaluations. A reference-near candidate, a generated candidate, an automatic winner and a visually usable fit are different outcomes. Do not combine their counts into an accuracy claim.

## Retired chain, proposal access and ranking

### D01 — Retire CourtKeyNet repair

The old detector chain passed 0/11 labelled amateur frames. Its broadcast repair work still contains useful safeguards, but CourtKeyNet is excluded from the replacement design. Removing the model and accidental weight-loading dependencies remains an integration requirement. [Retirement evidence](evidence/retirement/README.md).

### D02 — Preserve proposal evidence, reject old acceptance rules

Better line evidence and useful candidate pools did not settle which visible court was being played. Preserve the earlier inputs, proposal records and contrary examples. The failed acceptance rules are closed as defaults. [Independent proposal evidence](evidence/independent_proposals/README.md).

### D03 — Keep direction mechanisms distinct

Global midpoint and representative changes caused regressions. Historical SVD fitting held group membership fixed; it is distinct from the implemented SVD12 direction-family screen in D12. A proposed per-assignment SVD refit before capping has not been implemented. [Direction search](evidence/direction_search/README.md); [historical SVD assessment](evidence/webui_followup3_20260922/ASSESSMENT.md).

### D04 — Do not adopt the tested diversity cap rule

Exact traces show that score ordering and the 512-assignment cap discard court-compatible matches on traced pairs. A broader fixed-budget diversity rule helped one case but badly regressed another, so that rule remains rejected. D13's 640-axis trial changes search depth and is a separate experiment. The matcher assignment cap is also distinct from W5's 256-template retention limit. [Traces and stopped probe](evidence/direction_search/README.md#cap-and-duplicate-corrections).

### D05 — Retain complementary proposal sources

The expanded crossed comparison supports access to G1 proposals, which use paint-filtered fragments. It does not show that S1 scoring is universally better. G0, which uses original fragments, retains complementary coverage. More candidates can also worsen the automatic selection. [G0/G1 comparison](evidence/g0_g1/README.md).

### D06 — Keep the W5 development result in context

W5 `(4,3)` and `(5,3)` select identical courts. In the 27-case development review, 21 selections were usable; the existing player gate raised this to 24. G1 plus line templates preserved that count and the same failures in this review, but the wider test needs G0 for Am4-319. No tested floor score is a sufficient acceptance rule. [Four-part evaluation](archive/20260922/evaluation_results_20260922.md).

### D07 — Person masks help proposal access in limited cases

All five corrected person-mask matcher comparisons are complete. Useful broadcast proposals survive, including SS03-17 despite its poor paint winner. Masks alone do not rescue GX5. The historical direction-changing masked arms remain qualified because their changes were not isolated to masking. [Box provenance and repairs](evidence/holistic_admission/box_provenance.md).

### D08 — Shared candidate access helps GX court identity

With the same registered GX candidate union, per-frame and shared paint ranking identify the played court on 7/7 frames. Native-only access identifies it on 5/7. These are court-identity results, not precise-fit counts: far-end errors remain. Shared line ranking selects a wall on all seven. This supports stable-view proposal reuse, with far-marking accuracy and camera stability still requiring checks. [Pixel and temporal evidence](evidence/pixel_temporal/README.md).

### D09 — Wider evaluation retains G0 and exposes fit bias

The wider comparison completed 47 frozen cases and 24 controls. Removing G0 loses the tolerable Am4-319 fit. Both arms accept one of eight labelled non-court controls. Static-grid residuals show an inward width bias, and SS03-34 has a confirmed inner/outer paint-edge misinterpretation. [Wider numeric comparison](archive/20260922/wider_evaluation_20260922.md#completed-numeric-comparison).

## Fitting, colour and net evidence

### D10 — Fixed-point polarity partly corrects the left inset

Across six video-03 scenes, a fixed-point polarity check reduced median left-edge disagreement with the shared static grid from 3.745 to 1.799 working pixels. Scores were mixed; the approved GX control moved under 0.05 pixels. D11 records the subsequent centre-label investigation. This test did not change production fitting defaults. [Seven-case polarity test](edge_polarity/README.md).

### D11 — Centre-to-edge relabelling is a partial correction

The user preferred centre-to-edge relabelling in a four-case review. The eight-source amateur extension did not solve Am1's net-tape identity error. Keep the method as an experimental fitting candidate. Neither a fixed corner offset nor a changed physical stripe width is justified. [Local fitting assessment](edge_polarity/local_audit/ASSESSMENT.md).

### D12 — SVD12 is an accepted experimental screen

Fresh experimental W5 generation defaults to SVD12, which retains 12 of 16 direction-support groups; full16 remains available. Nine development cases retain all eight approved automatic witnesses and the best reference-agreement candidates. Three score-winner roles change, with mixed substitutes. The user accepted that efficiency tradeoff. The completed benchmark saved 52.9% of summed matcher time, not end-to-end time or a general accuracy guarantee. [Implementation](svd_runtime/README.md); [matcher benchmark](svd_runtime/RESULTS.md).

### D13 — Deeper axes recover one useful G0-only case

In six G0-only cases, 640 rather than 512 axes cost 26.6% more summed full-trial time and gave a visually suitable Am2 result where baseline failed. A wider shortlist cost 10.0% more without a comparable visual win. Deeper Am2 and both deeper SS03 selections reproduced saved W5 geometry. Keep G1 and templates: G0-only failures do not show that SVD caused them. No global search-depth default was changed. [Search comparison and user rulings](svd_search/WORKLOG.md).

### D14 — Do not adopt an automatic colour veto

The floor trial rejected 0/9 saved choices. Final hue-only paint rejected 0/12: one comparison was usable and eleven were inconclusive. Hue-grouped raw colour rejected Am1 but still failed a pale same-hue control. Keep weak or neutral colour evidence inconclusive; similar paint cannot certify court geometry. These were fixed-fit diagnostics and decision trials, not a demonstrated colour-driven detector improvement. [Colour trials and limits](colour_consistency/PLAN.md).

### D15 — Recover Am1 without unrestricted net preference

Three automatic line-group seeds exposed an Am1-54 candidate that positive net support selects; the user accepts the result. The subsequent 71-case scan changed 13 choices and lost good courts on Letterboxed78 and frame 52563. Unrestricted net preference is rejected. Missing net support does not establish a bad court. The later bounded rule is D17. [Net assessment](net_recovery/ASSESSMENT.md); [Am1 recovery](colour_consistency/AM1_RECOVERY.md); [scan record](net_recovery/WORKLOG.md).

### D16 — Carry automatic stripe polarity into integration

In the six-case gallery, the user found automatic polarity practically identical to the bright-paint rule and both better than the original. Automatic polarity also handles synthetic dark-stripe inversion with modest extra cost. Robustness on real dark courts is unproven. Keep original labels when the polarity is unresolved. Stripe correction alone did not repair Am1; D15 records the seeded recovery. Production fitting is unchanged. [Final edge policy and timing](edge_polarity/local_audit/ASSESSMENT.md).

### D17 — Use bounded net weight 0.04 provisionally

For the fresh combined trial, retain net weight **0.04** and projected post-base overrun **4 working pixels**. A post earns support when any of the first six of 24 samples from its projected base matches a DeepLSD fragment, provided none of its matched fragments extends more than 4 px below that base. Existing match tolerances are 4 px lateral, 8 degrees and 2 px extent. Zero, one or two supported posts earn 0, 0.5 or 1; this reward is added to the frozen paint score at the chosen weight. Gates remain, and exact ties retain original rank. Missing evidence is neutral.

Only 4/26 supported posts on changed choices match the base sample, so this is lower-post support, not post detection or verified post identity. The bounded rule protects the five known regression views from the unrestricted scan. Overrun 2, 4 and 8 selected identical courts in the saved set. [Rule and limitations](net_recovery/ASSESSMENT.md); [trial record](net_recovery/WORKLOG.md).

The trial measured 72 pools from 71 unique frames because Am1-54 has both old and seeded pools. Weight 0.04 changed 15/72 pool selections. The primary 71-frame comparison replaces the old Am1 pool with its seeded pool. Of these, 45 have usable references: 27 landmark and 18 four-corner frames. Median per-frame reference-point errors for weights 0, 0.02, 0.04 and 0.08 are 3.40, 2.75, 2.75 and 2.76 working pixels. At 0.04, Q90 is 5.62 versus 6.20 at 0.02. Differences between positive weights are small, and ten-source bootstrap intervals include zero. The data do not establish a statistically reliable optimum. Am1-5352 and Yellow14 retain large errors; all settings accept one of eight labelled non-court controls. Seven frames have no gated selection. Two unverified views and 24 controls lack usable reference geometry. [Paired statistics](net_recovery/statistics/paired_reference_report.md).

The accepted 20-case gallery selects with bounded net support, then applies automatic stripe correction to that selected fit. The user called the page very usable despite tiny amateur imperfections. This is gallery-level feedback, not 20 independent case labels. Net evidence does not constrain the correction. The comparison reused older candidate pools; the fresh SVD12, G0/G1/template, bounded-net and stripe-corrected combination has not been tested. Fresh and historical paint measurements differ even on the same image and geometry, with the cause unresolved. [Combined replay](net_recovery/run_combined.py); [trial account](net_recovery/WORKLOG.md).

## Runtime and deployment boundary

### D18 — Optimise before deployment

Six SVD12 G0-only trials still took **4.3–33.6 minutes per frame**. Generation accounted for 91.6% of summed full-trial time; refitting and scoring alone took 59–107 seconds per frame. These runs used six workers with one numerical thread each and cached line inputs. They excluded imports, output writing, G1/templates and video processing, so they are not complete detector or ordinary-hardware latency measurements. [Saved stage timings](svd_search/WORKLOG.md); [launch evidence](handover_20260923/01_LAUNCH_OPTIMISATION.md).

The user's expectation is roughly **30 seconds for a five-minute video**, with **90 seconds tolerable**. Report startup and warmup separately. The user rejected the earlier 5%-of-duration/15-second proposal. For longer videos, expensive search should roughly follow distinct compatible camera views. Existing code applies ContentDetector to find cuts, samples frames for court evidence, and only then groups initially valid scenes by perceptual hash and alignment. That grouping saves no first-pass court searches. Cut detection and decoding still scan the video. [Shared requirement](handover_20260923/00_SHARED_CONTRACT.md); [scene code anchors](handover_20260923/01_LAUNCH_OPTIMISATION.md).

### Scene-level operation is the target

The intended detector samples several frames from a scene or verified stable view. It shares compatible proposals, uses line, paint and player evidence across samples, and checks geometry before reusing a court. Loss of support or camera change should trigger new search or abstention. Sample count, spacing and agreement rule remain open. The fixed-pool GX replay tests only proposal access and ranking; it does not implement multi-frame generation, a scene sampler or change handling. A wall can score consistently across frames, so agreement must include court geometry at the far end. [Temporal evidence](evidence/pixel_temporal/README.md); [scene contract](handover_20260923/00_SHARED_CONTRACT.md).

## How to judge fit quality

The 24/27 W5 result is qualitative development usability, not 24 clean fits. The GX 7/7 result means the played court was identified, not that all fits were accurate. Use three visual categories: **clean fit** where boundaries and service markings follow the court; **tolerable fallback** where the right court has a limited visible error; and **unacceptable fit** for a wrong court or substantial missing or extra area. Record exceptional last-resort use separately.

The user placed frame-0 G1 `143:158` in the fallback category: it is skewed and clips a sliver of far-left court. GX86088 G1 `16:44` is a more serious last resort. Near-end alignment conceals loss of nearly the whole far long-service-to-baseline strip. Such a loss at reasonable frequency would block shipping. These rulings concern temporal candidates, not their separately refined W5 children. [Temporal review](evidence/pixel_temporal/README.md).

Review far-end crops with full images. Where annotations permit, measure far-baseline, long-service and sideline errors separately, as well as missing court area. Report severity and frequency instead of one usable rate. On 22 September the user described imperceptible misalignment as the visual ideal and preferred outer edges of white markings. This set no pixel cutoff or accepted fallback frequency; borderline cases need review. The later 20-case gallery feedback accepts frequent tiny amateur imperfections without assigning each image to a category. [Fitting assessment](edge_polarity/local_audit/ASSESSMENT.md); [bounded-gallery account](net_recovery/WORKLOG.md).

### Interpret the earlier colour diagnostic gallery correctly

That gallery displays preserved fits, not geometry changed by colour. Saved W5 and saved G1/templates use one earlier ranking with different source access: 50 of 64 available fits have identical geometry, 14 differ, and seven records have no fit. Source IDs and rendered coordinates match the saved records. Duplicate columns therefore do not measure colour efficacy. The user confirmed the repaired shared-template UI worked; that was a UI ruling. [Colour diagnostic](colour_consistency/PLAN.md).

## Older ideas worth bringing back

These are closed or conditional leads, not a new experiment queue. The [older temporal replay](../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/evaluation/temporal_assessment.md#an-earlier-shared-court-experiment-already-exists) changed Yellow's worst-corner error from 221.19 px under floor scoring to 26.01 with net evidence and 21.22 after pooled refinement. Letterboxed worsened from 10.06 to 14.75. Those are 1280×720 errors against old references, not current working-pixel errors. Net cue and temporal refinement were not isolated ablations. The replay used footpoints and raw fragments, so the historical spatial-mask error does not invalidate it. Its cue measured image support along projected net segments; W5's projected-post diagnostic recorded only geometry. Player-pair identity and stable camera geometry are prerequisites.

Older fixed-assignment refits sometimes improved a plausible court, but an improvement rule chose 35 harmful refits among 357 parent/refit pairs. W5 already has same-view fixed-assignment refinement. Keep parents and children together and inspect both near and far markings; a lower fitting objective does not certify better geometry. [Refit evidence](evidence/independent_proposals/README.md#other-branches-and-why-their-records-remain-useful).

Broadcast scene-repair safeguards remain useful design context: borrowed geometry should agree with target-visible lines, inferred repairs should not become donors, and mixed broadcast composites need separate treatment. They do not validate amateur detection. Camera novelty, colour and temporal consistency alone cannot certify a court. [Donor safeguards](evidence/retirement/README.md#non-model-lessons-worth-keeping).

Earlier cap traces justify investigating retention only after a concrete coverage failure. The tested diversity rule regressed one pair; any replacement needs paired checks and controls. Graph search or shared partial states are later implementation leads, not accuracy findings. Define equivalent states from the matcher constraints before using them. Per-assignment SVD fitting is also untested. Keep global midpoint and direction replacements, connected-pixel vetoes and junction-first ranking out of the main design. Retain player evidence while comparing rejection and fallback; the historical hard gate need not be the final scoring form. [Cap traces](evidence/direction_search/README.md#cap-and-duplicate-corrections); [historical proposal branches](evidence/independent_proposals/README.md).

## Frozen sample coverage

The [archived evaluation](archive/20260922/evaluation_results_20260922.md) and [wider comparison](archive/20260922/wider_evaluation_20260922.md) retain the original protocol and input coverage. The current 71-frame statistics use the seeded Am1 replacement pool and report original-pool results separately. Do not infer performance on unseen cameras from these development sets. Unreliable edit-transition views such as SS21-10 are excluded from required detection by the user; confidence rejection remains optional.

Representative labelled evaluation, ordinary-hardware latency with player detections supplied, camera-change handling and annotator integration remain distinct requirements. The experimental runners already reuse greyscale and prepared measurements. Removing unused diagnostics is only a small optional compute change. Historical 2.37× local timing and Carmack matcher savings are not deployment latency. [Compute audit](svd_runtime/COMPUTE_AUDIT.md); [current runtime evidence](handover_20260923/01_LAUNCH_OPTIMISATION.md).
