# Find court-detector files by idea

This is a map of what the directories mean. Nothing has moved. Start with the
question you have, then follow one route. Most dated folders are frozen
experiments, not successive versions of a deployable detector.

## Start here

| I want to… | Open |
| --- | --- |
| Understand where the detector stands and what to test next | [DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md) |
| Resume practical work | [pickup.md](pickup.md) |
| Read the completed four-part comparison | [evaluation_results_20260922.md](evaluation_results_20260922.md) |
| Judge the results myself using only GitHub | [Raw review packet](evidence/review_20260922/README.md) |
| Find execution details or check results | [Evaluation worklog](evaluation_20260922.md); older records are in local-only `worklog/` |

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
| Find out why CourtKeyNet is being removed | **Retirement**, old fallback chain | [Retirement findings](evidence/retirement/README.md) |

`GX` names views from the difficult amateur video. `Am1`–`Am4` name other
amateur videos; `SS03` and `SS21` identify ShuttleSet videos. A candidate such
as `G1:16:44` is source-qualified within a case. It is not globally unique.
`/child` denotes a separately refined candidate; do not transfer a ruling
between a parent and child without checking their geometry.

## Code, inputs and big working packets

| Subtree | Its job |
| --- | --- |
| [w5_holistic/](w5_holistic/) | Whole-court experiment runner, measurements, refinement and galleries; not the integrated detector |
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
Their route remains in [INDEX.md](INDEX.md#recovery-not-another-reading-path).
