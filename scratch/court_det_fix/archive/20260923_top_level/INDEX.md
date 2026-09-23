> Archived 23 September 2026 before the top-level refresh. Live role: [INDEX.md](../../INDEX.md). Prose retained; relative Markdown links adjusted for this location. Inline code paths retain their original court_det_fix context.

# Court-detector investigation

**Fresh session:** read the [shared contract](../../handover_20260923/00_SHARED_CONTRACT.md)
and [optimisation launch prompt](../../handover_20260923/01_LAUNCH_OPTIMISATION.md).
[pickup.md](../../pickup.md) retains the investigation summary and evidence links.
This index explains the
experiment history through the 23 September handover. [FP_INDEX.md](../../FP_INDEX.md) maps ideas directly to files;
[DETECTOR_DECISIONS.md](../../DETECTOR_DECISIONS.md) retains lasting findings.

## Experiment history

| Stage or question | Record |
| --- | --- |
| Retire the old CourtKeyNet fallback chain | [Retirement](../../evidence/retirement/README.md) |
| Recover courts from line fragments, paint and player evidence | [Independent proposals and earlier fits](../../evidence/independent_proposals/README.md) |
| Explain direction, assignment-cap and line-identity failures | [Direction/search](../../evidence/direction_search/README.md); [G0/G1 comparison](../../evidence/g0_g1/README.md) |
| Combine proposal sources and test whole-court admission | [W5 findings](../../evidence/holistic_admission/README.md); [completed directional-floor packet](../../evidence/holistic_admission/directional_20260921_r5/README.md) |
| Correct person-mask provenance | [Box repair and valid comparisons](../../evidence/holistic_admission/box_provenance.md) |
| Separate shared proposals from shared scoring | [Pixel and temporal evidence](../../evidence/pixel_temporal/README.md) |
| Read the completed four-part evaluation and its execution record | [Archive map](../README.md), then the 22 September results/worklog |
| Follow the wider 47-view plus 24-control evaluation | [Archived wider worklog](../20260922/wider_evaluation_20260922.md#completed-numeric-comparison) |
| Check whether earlier approved geometry was lost | [Historical fit audit](../../evidence/independent_proposals/history_audit_20260922.md) |
| Test whether an inner paint edge is mistaken for the outer boundary | [Paint-side experiment and rerun commands](../../edge_polarity/README.md) |
| Explain the remaining fitting bias and test centre-to-edge labels | [Completed local assessment](../../edge_polarity/local_audit/ASSESSMENT.md); [original WebUI brief](../../edge_polarity/WEBUI_OBJECTIVE_AUDIT.md) |
| Follow the WebUI calculations | [Original prompts](../../evidence/webui_followups_20260922/PROMPTS.md); [returns 1/2 and local replay](../../evidence/webui_followups_20260922/README.md); [SVD returns and critical assessment](../../evidence/webui_followup3_20260922/ASSESSMENT.md) |
| Integrate SVD12 and measure its matcher cost | [Current SVD state](../../svd_runtime/README.md); [completed nine-case timings](../../svd_runtime/RESULTS.md) |
| Spend the saving on deeper axes or wider shortlists | [Completed six-case comparison and user review](../../svd_search/WORKLOG.md); [local comparison gallery](../../svd_search/gallery/index.html) |
| Compare automatic paint and edge decisions | [Completed trial record](../../colour_consistency/PLAN.md); [six-case review gallery](../../colour_consistency/decision_gallery/index.html) |
| Recover Am1 proposals and prefer independently supported nets | [Bounded recovery result](../../colour_consistency/AM1_RECOVERY.md); [three-case gallery](../../colour_consistency/am1_gallery/index.html) |
| Check wider net preference and its regressions | [71-case scan and current worklog](../../net_recovery/WORKLOG.md); [changed-choice gallery](../../net_recovery/gallery/index.html) |
| Choose bounded net weight and carry forward accepted stripe fitting | [Paired statistics](../../net_recovery/statistics/paired_reference_report.md); [selected-court fitting gallery](../../net_recovery/polarity_gallery/index.html) |
| Investigate Am1 net tape using colour | [Received WebUI colour probe](../../edge_polarity/webui_return_colour_consistency/README.md); [local replay and broader diagnostic](../../colour_consistency/PLAN.md) |
| Understand what the colour gallery does and does not establish | [Diagnostic findings](../../colour_consistency/PLAN.md); [local gallery](../../colour_consistency/gallery/index.html). Existing fits were measured, not reranked or refitted |

The SVD efficiency result and the deeper-search benefit on Am2 are completed
findings. Automatic floor and hue-only decision trials are also complete:
neither supplies a useful rejection on the saved choices. See the
[colour trial record](../../colour_consistency/PLAN.md). Broader detector work belongs
in [pickup](../../pickup.md).

## Am1 net-evidence result

The [net assessment](../../net_recovery/ASSESSMENT.md) now distinguishes the old-pool
no-go from the new [automatic recovery trial](../../colour_consistency/AM1_RECOVERY.md).
Extra line-group seeds admit useful candidates; a fixed net preference selects
an Am1 fit the user accepts, with a small remaining right-edge inset. The two
saved-pool controls are unchanged. The subsequent 71-case scan finds clear
regressions from unrestricted preference, so core promotion is held. See the
[wider worklog](../../net_recovery/WORKLOG.md). Follow
[pickup](../../pickup.md) for the current next step.

## Inputs and code

Use [frozen views](../../frozen_views/README.md) for inputs and [FP_INDEX](../../FP_INDEX.md#code-inputs-and-big-working-packets)
for code ownership. The [GitHub review packet](../../evidence/review_20260922/README.md)
provides a bounded published subset. Large preserved G0/G1 populations remain
local under `worklog/remote_records_20260921/`; its README is their map.
Experiment code, data and manifests retain their original locations. Recent
colour code, measurements, galleries and received packets are included in the
close-out checkpoint. Complete raw search outputs remain local; the compact
comparison and completion receipt are included. Local links do not imply every
large working packet has been published to GitHub.

<a id="recovery-not-another-reading-path"></a>

## Recovery

The [archive map](../README.md) records moved documents, retired names
and verified snapshots. Its [earlier recovery section](../README.md#earlier-sealed-recovery)
explains the sealed 21 September cleanup and W5 preparation archives.

## Keep this usable

- `pickup.md` is the only live handover; rewrite it at close-out, never add a sibling
- Keep pickup plus this index under roughly 4,000 orientation tokens; open evidence for a named question
- INDEX maps the experiment history; FP_INDEX maps ideas to files; decisions records lasting rulings
- Put each result and its run record beside the relevant evidence, with a reproducible command
- Archive completed plans/worklogs whole, label their old resume text, and update links
- Keep the next actions in pickup rather than duplicating them across reports
- Preserve source identity, units, reference limits and visual-review status
- Keep original inputs, useful candidate populations and recovery snapshots
- File future returns under evidence with explicit review status; the current net packet stays at its received top-level path until assessed
- Use numerical comparisons and focused visual questions; broad image review needs a specific reason
