# Find court-detector files by idea

This maps ideas to their files. Start with your question, then follow one route.
Most dated folders are frozen
experiments, not successive versions of a deployable detector.

## Start here

| I want to… | Open |
| --- | --- |
| Review the final colour and automatic-edge trials | [Trial worklog](colour_consistency/PLAN.md); [decision gallery](colour_consistency/decision_gallery/index.html) |
| Read settled findings and their evidence | [DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md) |
| Resume practical work and choose the next test | [pickup.md](pickup.md) |
| Read the completed fitting-objective investigation | [Local assessment](edge_polarity/local_audit/ASSESSMENT.md); [original prompt](edge_polarity/WEBUI_OBJECTIVE_AUDIT.md) |
| Read the completed four-part comparison | [evaluation_results_20260922.md](archive/20260922/evaluation_results_20260922.md) |
| Read the historical SVD return assessment and locate its packages | [Critical assessment](evidence/webui_followup3_20260922/ASSESSMENT.md); [packages](evidence/webui_followup3_20260922/README.md) |
| Read the implemented SVD screen and measured saving | [Current state](svd_runtime/README.md); [nine-case matcher timings](svd_runtime/RESULTS.md) |
| Compare deeper versus wider SVD searches | [Six-case worklog and visual rulings](svd_search/WORKLOG.md); [gallery](svd_search/gallery/index.html) |
| Understand the colour diagnostic and its duplicate columns | [Findings and limits](colour_consistency/PLAN.md); [measurement code](colour_consistency/run.py); [saved-source selection](wider_evaluation/compare.py) |
| Build a gallery using the working shared renderer | [Shared template](svd_search/gallery_template.html); [SVD adapter](svd_search/build_gallery.py); [colour adapter](colour_consistency/build_gallery.py) |
| Follow the net evidence and Am1 recovery result | [Assessment](NET_EVIDENCE_ASSESSMENT.md); [recovery report](colour_consistency/AM1_RECOVERY.md); [gallery](colour_consistency/am1_gallery/index.html); visual review pending |
| Read the returned far-end and sparse-frame follow-ups | [WebUI tasks 1 and 2](evidence/webui_followups_20260922/README.md) |
| Judge the results myself using only GitHub | [Raw review packet](evidence/review_20260922/README.md) |
| Find execution details or check results | [Evaluation worklog](archive/20260922/evaluation_20260922.md); older records are in local-only `worklog/` |

## The main ideas and their evidence

| Idea / question | Experiment names you will see | Where to look |
| --- | --- | --- |
| Generate courts from detected line fragments; filter fragments by painted-line appearance | **G0** = original fragments; **G1** = paint-filtered fragments | [G0/G1 explanation](evidence/g0_g1/README.md); [latest crossed comparison](evidence/g0_g1/evaluation_20260922/) |
| Separate better proposals from better scoring | **S0** = original scoring fragments; **S1** = filtered scoring fragments; **L2** = earlier four-view comparison | [G0/G1 explanation](evidence/g0_g1/README.md#historical-four-case-l2-comparison); [L2 tables and replay](next_steps_20260916/L2_scoring/) |
| Combine proposal sources, camera checks, paint evidence, players and refinement | **W5**, **stages 2–5**, **A/B/C** score orders | [Whole-court experiments](evidence/holistic_admission/README.md); [current W5 review](evidence/holistic_admission/directional_20260921_r5/visual_review.md) |
| Recover courts missing from line-matcher proposals | **Line templates**, **source admission** | [Admission protocol and experiment](evidence/holistic_admission/source_admission/); [why templates rescued GX5](evidence/holistic_admission/README.md#why-test-directional-floors) |
| Test how many projected markings a template must show before admission | **33 / 43 / 53** = lengthwise/cross-court visibility counts; **r5** = completed 27-view W5 packet | [Directional-floor packet](evidence/holistic_admission/directional_20260921_r5/README.md); [source and player-gate comparisons](evidence/holistic_admission/directional_20260921_r5/evaluation/) |
| Remove person fragments without mixing up image coordinates | **Person masks**, **box repair** | [Box error and valid comparisons](evidence/holistic_admission/box_provenance.md); [corrected five-case comparison](evidence/holistic_admission/box_repair/evaluation_20260922/) |
| Use player locations to choose the played court | **Player-support / full-court gate**; distinct from person masks | [W5 gate review](evidence/holistic_admission/directional_20260921_r5/visual_review.md#existing-player-support-gate); [older player-guided work](evidence/independent_proposals/README.md#temporal-paint-and-projective-branches) |
| Reuse courts across frames; distinguish proposal pooling from score averaging | **L3** = 30-court temporal pilot; **W4** = rank-sum replay; **automatic union** = complete GX/Am3 comparison | [Temporal findings](evidence/pixel_temporal/README.md); [complete replay and results](evidence/pixel_temporal/evaluation_20260922/) |
| Build a practical scene-level detector without searching every frame | **Sparse sampling**, multi-frame agreement, **SVD search reduction**, later **graph/shared-state search** | [Deployment leads and scene-level design](DETECTOR_DECISIONS.md#older-ideas-worth-bringing-back); [historical direction diagnostics](evidence/direction_search/README.md) |
| Understand why a net, wall or seam can score like court paint | **W2**, interval/pixel atlas, identity diagnostics | [Pixel evidence and caveats](evidence/pixel_temporal/README.md); [W2 atlas assets](evidence/pixel_temporal/w2/evidence/); [identity diagnostics](evidence/pixel_temporal/diagnostics/identity_diagnostics/) |
| Improve directions or stop useful assignments being discarded too early | **B/M/R/MR**, **E0–E4**, **C2**, **L1** | [Direction and cap findings](evidence/direction_search/README.md); [direction experiment](direction_agreement/); [exact cap witnesses](next_steps_20260916/C2_traces/); [stopped diversity probe](next_steps_20260916/L1_admission/) |
| Improve a plausible court's geometry with stripes, multiple fragments or net geometry | **Stripe / marking refit**, **physical paint**, **projective patterns** | [Earlier proposal and refit findings](evidence/independent_proposals/README.md); [reusable experiment package](../../experiments/annotator/independent_court/README.md) |
| Trace recurring inward corner bias | **Paint-side polarity**, **centre-to-edge correction** | [Original polarity probe](edge_polarity/README.md); [completed objective and amateur assessment](edge_polarity/local_audit/ASSESSMENT.md); [historical fits](evidence/independent_proposals/history_audit_20260922.md) |
| Reduce fresh matching work without changing original directions | **SVD12** = retain 12 of 16 support groups; distinct from SVD fitting | [Generator](w5_holistic/automatic_generation.py); [runtime evidence](svd_runtime/README.md) |
| Check whether more search supplies useful courts | **512/256 baseline**, **640/256 deeper**, **512/512 shortlist** | [Runner](svd_search/run.py); [completed results](svd_search/WORKLOG.md); `svd_search/run_20260923/cases/` |
| Measure paint and floor colour on fixed courts | **Raw paint**, **floor-relative chroma**, exploratory ambiguity | [Diagnostic and current limits](colour_consistency/PLAN.md); [measurements](colour_consistency/measurements.json.gz); [original returned probe](edge_polarity/webui_return_colour_consistency/scripts/boundary_paint_followup.py) |
| Find out why CourtKeyNet is being removed | **Retirement**, old fallback chain | [Retirement findings](evidence/retirement/README.md) |

`GX` names views from the difficult amateur video. `Am1`–`Am4` name other
amateur videos; `SS03` and `SS21` identify ShuttleSet videos. A candidate such
as `G1:16:44` is source-qualified within a case. It is not globally unique.
**Saved W5** selects from the full pool; **Saved G1/templates** restricts the
same saved ranking by source membership. They are not before/after colour arms:
50/64 available fits have identical geometry. **Reference-best** chooses a fit
retrospectively using annotations and is not a detector selection.
`/child` denotes a separately refined candidate; do not transfer a ruling
between a parent and child without checking their geometry.

## Code, inputs and big working packets

| Subtree | Its job |
| --- | --- |
| [w5_holistic/](w5_holistic/) | Whole-court experiment runner, measurements, refinement and galleries; not the integrated detector |
| [wider_evaluation/](wider_evaluation/) | Preserved 71-view manifest, numeric fits and full-versus-G1/template comparison; source of the colour diagnostic's saved courts |
| [svd_runtime/](svd_runtime/README.md) | SVD integration, retention links, completed matcher benchmark and optional compute audit |
| [svd_search/](svd_search/WORKLOG.md) | Three-arm six-case experiment, all 18 result records, receipts and the shared gallery template |
| [colour_consistency/](colour_consistency/PLAN.md) | Fixed-geometry diagnostic, completed automatic floor/paint decision trials, automatic edge comparison and galleries; no adopted colour veto |
| [line_identity/](line_identity/) | Automatic line-matching proposal experiments; `line_run_matcher.py` is the line-identity matcher |
| [direction_agreement/](direction_agreement/) | Fixed direction experiments and their E0–E4 records |
| [frozen_helpers_20260914/](frozen_helpers_20260914/) | Fixed helper code used to reproduce older experiments |
| [frozen_views/](frozen_views/README.md) | Shared saved frames, controls and input packs; start here for the actual input pixels |
| [evidence/](evidence/) | Findings grouped by idea, with retained tables, scripts and selected artefacts |
| [next_steps_20260916/](next_steps_20260916/) | Historical L1/L2/L3 and C2 experiments, despite the name; not the current task list |
| `worklog/remote_records_20260921/` (local-only) | Owner of the large preserved G0/G1 inputs and populations; start with its `README.md` |
| [experiments/annotator/independent_court/](../../experiments/annotator/independent_court/README.md) | Older reusable implementations and compressed recorded experiments outside this scratch subtree |

Some working packets are intentionally too large for Git. The
[GitHub review packet](evidence/review_20260922/README.md) states exactly what
is published and what remains local. A local path in an old run record is not
a promise that GitHub contains that file. GitHub does not render the old HTML
atlases as applications; use their linked images or the Markdown review pages.

Recovery archives are for retrieving superseded material, not normal reading.
Their route remains in [INDEX.md](INDEX.md#recovery).
