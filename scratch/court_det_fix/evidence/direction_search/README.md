# Direction search evidence

The direction-selection changes tested here do not provide a general detector fix. The midpoint anchor (M) rescues GX5 and improves both Amateur-2 views, but it breaks GX0 and Amateur-3. The precise representative (R) gives small gains on some views, but breaks GX0, Amateur-3 and ShuttleSet 21. Fixed-support SVD is diagnostic only. The strongest surviving mechanism is downstream: the axis matcher’s per-direction score cap can discard a court-compatible matching before court ranking sees it.

This is development evidence from nine frozen views at 960 by 540 working pixels. Six controls are visually approved and three are manual references. Distances are maximum corner error with the existing 180-degree relabelling. A close control distance is not a usability ruling.

## Direction experiment

The direction run is `direction_agreement_20260915_144900`. E0 reproduced the merge, estimator, coverage selection and baseline records exactly. E1 compared the original foot anchor with the midpoint of each line’s longest visible fragment. E2 froze four arms: baseline B, midpoint M, precise representative R, and combined MR. E3 fit a court to each control from every ordered direction pair. E4 sent M and R through the unchanged matcher and both rescoring stages.

E3 and E4 answer different questions. E3 is a least-squares fit with fixed vanishing directions. It measures direction-fit potential and is not a generated court. E4 generates courts without reading the control, then measures their distance afterwards. Membership counts describe merge assignment, not physical court stripes.

At generation-stage paint winners, M changes B as follows: GX5 525.7→9.9 px, Am2-150 12.4→8.8, Am2-28019 none→7.0, and SS03-16 3.9→3.3. It regresses GX0 7.7→379.3, Am3-0 8.5→23.8, and SS03-17 7.6→8.4. R improves Am2-28019 none→6.1, SS03-17 7.6→6.9, and SS03-19 3.93→3.80, while it regresses GX0 7.7→13.3, Am2-150 12.4→12.6, Am3-0 8.5→58.2, and SS21-20 2.0→2.8. The direction change agrees with E3 potential on eight of nine views for M and five for R. That agreement is not enough to promote either arm.

SVD refits each retained representative on its own support mask. It moves directions by a median 0.1–0.4 degrees and at most 3.3 degrees. It helps baseline B on eight cases, hurts Amateur-3, and behaves unevenly for M and R. No SVD arm reached the matcher. Keep this as a bounded diagnostic, not as a proposed production change.

The [full run](../../direction_agreement/runs/direction_agreement_20260915_144900/)
retains E0–E4 measurements, the frozen arm definitions and every M/R candidate
record. Start with [accounting.csv.gz](../../direction_agreement/runs/direction_agreement_20260915_144900/e4/accounting.csv.gz)
for a stage-by-stage comparison. The original producing revisions were
`0caae0a` for stage records and scientific checkpoint `9fcec94`.

The earlier three-case fixed-membership SVD test is a different diagnostic;
see [independent proposals](../independent_proposals/README.md). Neither run
establishes that refitting a fixed group repairs wrongly assigned support.

## Cap and duplicate corrections

The C2 replay confirms two exact traces from saved raw records and the frozen producing helper. On GX0, the pair-143 one-pair proxy improves under `paint_observations` from 34.3268 to 12.5775 px, but the full pre-global pool worsens from 7.6621 to 9.0360 px. Both full-pool winners come from pair 22. The proxy is therefore not a valid substitute for the full candidate pool.

On Amateur-3 under R, pair 43 contains an eligible 4.2783 px enumeration. Its assignment’s actual duplicate representative is 8.5405 px because that row has the higher score. The 6.2409 px row is a different assignment at distinct rank 11,518. The closest row retained by the 512 cap is 49.6472 px. The earlier explanation that duplicate removal retained the 6.24 px row is false and is corrected in the C2 findings.

| Amateur-3 R horizontal enumeration | Role | Error, working pixels |
| --- | --- | ---: |
| 29686 | Closest eligible enumeration | 4.2783 |
| 29665 | Higher-scoring representative of that same assignment | 8.5405 |
| 30886 | Closest distinct assignment; not the duplicate representative | 6.2409 |
| 18981 | Closest assignment retained under the cap | 49.6472 |

The first two share assignment `(7, 197, 32, 3, 19)`, with scores
0.6579144844 and 0.7844496413 respectively. Their distinction matters more
than the small differences in rounded error.

Retain [exact C2 witnesses](../../next_steps_20260916/C2_traces/witnesses.json),
[the producing check](../../next_steps_20260916/C2_traces/check_traces.py) and
[the frozen source map](../../next_steps_20260916/webui_seed/source/SOURCE_MAP.md).
The [cap](diagnostics/cap_loss/) and [pre-gate](diagnostics/pregate_loss/)
records preserve the raw arrays. Their analysis scripts read those arrays;
summary tables are not a substitute.

A possible later change is cap ordering or retention of diverse scale/offset candidates. It needs all-pair local screening before another expensive matcher run. The six-pair L1 admission screen is not that trial: a fixed-budget score-plus-diversity arm improves Am2-B pair 15 from 61.6207 to 7.5291 px, but breaks Am3-B pair 43 from 4.2708 to 23.2484 px. The exact variant was stopped without an all-pair matcher run and without a full-pool pixel score. Keep its witnesses as a stopped probe.

## What remains useful

The [six inspection helpers](inspection_tools/) preserve the 14 September
direction-precision review's source. They are historical diagnostic tools,
not validated alternatives to the retained experiment producers.

The stopped L1 probe retains its [comparison](../../next_steps_20260916/L1_admission/comparison.csv),
[witnesses](../../next_steps_20260916/L1_admission/witnesses.json) and
[overlays](../../next_steps_20260916/L1_admission/overlays/). It tested
keeping 256 score-leading assignments plus 256 from fixed scale/offset bins.
Both arms still formed 512×512 candidate combinations. The control measured
the outcome; it did not choose admissions.

The [G0/G1 note](../g0_g1/README.md) separates access to proposals from changed
scoring observations. Its expanded 27-case comparison is unfinished.
The [pixel/temporal note](../pixel_temporal/README.md) owns the completed
rank-sum result and the distinct independent-versus-shared scoring question.

Do not promote a local pair improvement to a full-pool gain. Any future cap
change needs all-pair screening, the existing regression controls, and a
ranked-winner check. Graph/SVD search remains an untested alternative, not
a proven improvement. Count repeated partial states in saved cases before
justifying a larger search rewrite.

Historical direction, line-identity and C2/L1 narratives remain recoverable
by their original paths in the tidy backup. They are replaced here by
tested claims, exact witnesses and explicit unfinished comparisons.
