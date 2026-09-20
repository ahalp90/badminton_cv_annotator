# W5 holistic court-detector pilot

This is a development/fit packet on the frozen corpus. It compares the legacy paint-first baseline, whole-court scoring of original candidates, and the same scoring over original plus locally adjusted candidates. The automatic pool includes the cached-fragment `line_template` source. The JSON and CSV retain `A`, `B` and `C` as historical field names.

## Completed views

| view | legacy paint-first | original + adjusted: initial | original + adjusted: camera-filtered | original + adjusted: span-weighted | final status | adjusted candidates | determinism |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| GX0 | G0:22:4588 | line_template:rectangle_19114:template_134/child | line_template:rectangle_19114:template_134/child | line_template:rectangle_19114:template_134/child | provisional_for_review | 720 | pass |
| GX5 | G0:181:973 | line_template:rectangle_20061:template_59 | line_template:rectangle_20061:template_59 | line_template:rectangle_85795:template_72/child | provisional_for_review | 599 | pass |
| Am2-150 | G0:30:33 | G1:30:9149/child | G1:30:9149/child | G1:30:137/child | provisional_for_review | 290 | pass |
| Am2-28019 | G1:15:168 | G1:0:35/child | line_template:rectangle_116543:template_43 | line_template:rectangle_210761:template_48 | provisional_for_review | 414 | pass |
| Am3-0 | G1:43:296 | G1:44:25 | G1:43:299/child | G1:43:299/child | provisional_for_review | 632 | pass |
| SS03-17 | G1:1:71 | G1:1:1308/child | G1:1:1308/child | G1:1:1308/child | provisional_for_review | 676 | pass |
| SS03-19 | G0:1:60 | line_template:rectangle_164603:template_55 | line_template:rectangle_164603:template_55 | G1:1:533/child | provisional_for_review | 680 | pass |
| SS03-16 | G0:1:31 | G0:1:31 | G0:1:31 | G0:1:31 | provisional_for_review | 668 | pass |
| SS21-20 | G1:0:2 | G1:0:1970 | G1:0:1970 | G1:0:1970 | provisional_for_review | 667 | pass |

## Collision resolution

Parent candidates use source-qualified `origin_key` values for every join, ranking, diagnostic and gallery lookup. Raw candidate IDs remain source-local provenance. Exact cross-source geometry duplicates retain all source occurrences, including `line_template`, under one canonical parent when the relevant gates agree; legacy-only source metadata stays attached to each occurrence. A same-source geometry duplicate, metadata mismatch or W5-gate mismatch stops that view as ambiguous.

| view | source occurrences | canonical parents | raw-ID collisions | duplicate geometry groups |
| --- | ---: | ---: | ---: | ---: |
| GX0 | 768 | 768 | 0 | 0 |
| GX5 | 768 | 768 | 0 | 0 |
| Am2-150 | 768 | 767 | 0 | 1 |
| Am2-28019 | 768 | 764 | 1 | 4 |
| Am3-0 | 768 | 768 | 0 | 0 |
| SS03-17 | 768 | 722 | 1 | 46 |
| SS03-19 | 768 | 713 | 3 | 55 |
| SS03-16 | 768 | 683 | 5 | 85 |
| SS21-20 | 768 | 691 | 5 | 77 |

Exact-geometry deduplication can change pool counts and diagnostic ranks without changing the evidence for retained geometries. Per-view counts are recorded in `per_view.csv`; the `30:33` control remains available for the Am2-150 comparison.

## Legacy baseline source provenance

The legacy baseline reports the canonical parent used for joins, rankings, diagnostics and gallery lookups. For an exact-geometry merge, the occurrence column identifies the source record that supplied the winning legacy score. Both parent and occurrence identities remain available in the packet records.

| view | line parent | line occurrence | paint parent | paint occurrence |
| --- | --- | --- | --- | --- |
| GX0 | G0:22:4579 | G0:22:4579 | G0:22:4588 | G0:22:4588 |
| GX5 | G0:181:1029 | G0:181:1029 | G0:181:973 | G0:181:973 |
| Am2-150 | G0:30:30 | G0:30:30 | G0:30:33 | G0:30:33 |
| Am2-28019 | G1:15:168 | G1:15:168 | G1:15:168 | G1:15:168 |
| Am3-0 | G1:43:296 | G1:43:296 | G1:43:296 | G1:43:296 |
| SS03-17 | G0:1:83 | G1:1:16 | G1:1:71 | G1:1:71 |
| SS03-19 | G0:1:60 | G1:1:19 | G0:1:60 | G1:1:19 |
| SS03-16 | G0:1:60 | G1:1:20 | G0:1:31 | G1:1:13 |
| SS21-20 | G1:0:2 | G1:0:2 | G1:0:2 | G1:0:2 |

## Top candidates after each scoring change

This pool contains original and valid locally adjusted candidates. The first order rejects implausible camera geometry and uses the plain mean of each marking's paint evidence. The final order weights each marking by its visible span, with the same weighting applied to the geometry fallback when needed.

| view | camera-filtered top candidate | span-weighted top candidate | status |
| --- | --- | --- | --- |
| GX0 | line_template:rectangle_19114:template_134/child | line_template:rectangle_19114:template_134/child | provisional_for_review |
| GX5 | line_template:rectangle_20061:template_59 | line_template:rectangle_85795:template_72/child | provisional_for_review |
| Am2-150 | G1:30:9149/child | G1:30:137/child | provisional_for_review |
| Am2-28019 | line_template:rectangle_116543:template_43 | line_template:rectangle_210761:template_48 | provisional_for_review |
| Am3-0 | G1:43:299/child | G1:43:299/child | provisional_for_review |
| SS03-17 | G1:1:1308/child | G1:1:1308/child | provisional_for_review |
| SS03-19 | line_template:rectangle_164603:template_55 | G1:1:533/child | provisional_for_review |
| SS03-16 | G0:1:31 | G0:1:31 | provisional_for_review |
| SS21-20 | G1:0:1970 | G1:0:1970 | provisional_for_review |

## Known diagnostic controls

The positive and negative controls are directionally separated in the saved two-direction Q readouts, which is consistent with their prior rulings. This is a diagnostic result only: no threshold or automatic pass/fail was applied.

| view | origin | raw candidate | expected role | automatic pool | hard-valid | camera eligible | initial rank | camera-filtered rank | span-weighted rank | Q_geom | Q_paint10 | status |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Am2-150 | diagnostic:am2_window_00_frame_150:30:33 | 30:33 | positive_approved | yes | yes | yes | 46 | 34 | 27 | 0.459007 | 0.324535 | diagnostic-only; no gate |
| Am2-28019 | diagnostic:am2_window_01_frame_28019:184:4123 | 184:4123 | negative_rejected_false_paint | no | yes | yes | 1116 | 274 | 273 | 0.0251343 | 6.91502e-07 | diagnostic-only; no gate |
| SS03-19 | diagnostic:shuttleset_03_scene_0019:1:60 | 1:60 | positive_usable | yes | yes | yes | 25 | 25 | 35 | 0.914859 | 0.740931 | diagnostic-only; no gate |
| SS03-19 | diagnostic:shuttleset_03_scene_0019:165:6702 | 165:6702 | negative_rejected_hallucinated | no | yes | yes | 1206 | 1186 | 1186 | 0.131008 | 0.0912309 | diagnostic-only; no gate |

## Contrast-probe sensitivity

Each row reranks the original-plus-adjusted pool with the camera filter and visible-span weighting, using the saved raw ridge arrays. The table reports the top span-weighted candidate. The JSON retains the separate historical `r1` and `r2` orders for compatibility. Probe 10 remains the named pilot setting.

| view | probe | span-weighted top candidate | status | control target | control error |
| --- | ---: | --- | --- | --- | ---: |
| GX0 | 5 | line_template:rectangle_19114:template_134/child | provisional_for_review | approved_supplied_direction_control | 3146.83 |
| GX0 | 10 | line_template:rectangle_19114:template_134/child | provisional_for_review | approved_supplied_direction_control | 3146.83 |
| GX0 | 15 | G0:22:4580/child | provisional_for_review | approved_supplied_direction_control | 10.9798 |
| GX0 | 20 | G1:22:137/child | provisional_for_review | approved_supplied_direction_control | 9.70312 |
| GX5 | 5 | line_template:rectangle_19047:template_88 | provisional_for_review | approved_supplied_direction_control | 1266.28 |
| GX5 | 10 | line_template:rectangle_85795:template_72/child | provisional_for_review | approved_supplied_direction_control | 3.06901 |
| GX5 | 15 | line_template:rectangle_85795:template_72/child | provisional_for_review | approved_supplied_direction_control | 3.06901 |
| GX5 | 20 | line_template:rectangle_85795:template_72/child | provisional_for_review | approved_supplied_direction_control | 3.06901 |
| Am2-150 | 5 | G1:30:137/child | provisional_for_review | approved_supplied_direction_control | 7.53131 |
| Am2-150 | 10 | G1:30:137/child | provisional_for_review | approved_supplied_direction_control | 7.53131 |
| Am2-150 | 15 | G0:30:32/child | provisional_for_review | approved_supplied_direction_control | 20.7344 |
| Am2-150 | 20 | G1:30:21/child | provisional_for_review | approved_supplied_direction_control | 14.6109 |
| Am2-28019 | 5 | line_template:rectangle_210761:template_48 | provisional_for_review | approved_supplied_direction_control | 63586.6 |
| Am2-28019 | 10 | line_template:rectangle_210761:template_48 | provisional_for_review | approved_supplied_direction_control | 63586.6 |
| Am2-28019 | 15 | line_template:rectangle_216734:template_43 | provisional_for_review | approved_supplied_direction_control | 17663.7 |
| Am2-28019 | 20 | G1:15:3024 | provisional_for_review | approved_supplied_direction_control | 36.8172 |
| Am3-0 | 5 | G1:32:180/child | provisional_for_review | approved_supplied_direction_control | 21.061 |
| Am3-0 | 10 | G1:43:299/child | provisional_for_review | approved_supplied_direction_control | 18.6854 |
| Am3-0 | 15 | G1:42:1761/child | provisional_for_review | approved_supplied_direction_control | 20.1578 |
| Am3-0 | 20 | G1:42:1757/child | provisional_for_review | approved_supplied_direction_control | 19.954 |
| SS03-17 | 5 | G1:1:1308/child | provisional_for_review | approved_supplied_direction_control | 6.91297 |
| SS03-17 | 10 | G1:1:1308/child | provisional_for_review | approved_supplied_direction_control | 6.91297 |
| SS03-17 | 15 | G1:1:1775/child | provisional_for_review | approved_supplied_direction_control | 6.04282 |
| SS03-17 | 20 | G1:1:1775/child | provisional_for_review | approved_supplied_direction_control | 6.04282 |
| SS03-19 | 5 | G0:1:53806/child | provisional_for_review | approved_supplied_direction_control | 4.36483 |
| SS03-19 | 10 | G1:1:533/child | provisional_for_review | approved_supplied_direction_control | 2.77485 |
| SS03-19 | 15 | line_template:rectangle_164603:template_55 | provisional_for_review | approved_supplied_direction_control | 2.82384 |
| SS03-19 | 20 | line_template:rectangle_164603:template_55 | provisional_for_review | approved_supplied_direction_control | 2.82384 |
| SS03-16 | 5 | G0:1:5767 | provisional_for_review | approved_supplied_direction_control | 3.71196 |
| SS03-16 | 10 | G0:1:31 | provisional_for_review | approved_supplied_direction_control | 3.93599 |
| SS03-16 | 15 | G0:1:14/child | provisional_for_review | approved_supplied_direction_control | 5.22882 |
| SS03-16 | 20 | G0:1:5767/child | provisional_for_review | approved_supplied_direction_control | 2.60937 |
| SS21-20 | 5 | G1:0:1970 | provisional_for_review | approved_supplied_direction_control | 4.29401 |
| SS21-20 | 10 | G1:0:1970 | provisional_for_review | approved_supplied_direction_control | 4.29401 |
| SS21-20 | 15 | G1:0:1970 | provisional_for_review | approved_supplied_direction_control | 4.29401 |
| SS21-20 | 20 | G0:0:120/child | provisional_for_review | approved_supplied_direction_control | 2.78291 |

## Reference diagnostics after ranking lock

Reference metrics were joined after the automatic rankings and sensitivity orders were written.

| view | role | origin | supplied-control error | frozen-reference error | visible-landmark max error |
| --- | --- | --- | ---: | ---: | ---: |
| GX0 | ranked | line_template:rectangle_19114:template_134 | 3749.14 | 3751.99 | 3751.99 |
| GX0 | ranked | G0:22:4588 | 15.3241 | 19.0347 | 13.5919 |
| GX0 | ranked | G0:22:4579 | 33.0374 | 28.2083 | 21.4217 |
| GX0 | reference-near | G0:22:4588/child | 7.55582 | 14.2394 | 16.463 |
| GX0 | ranked | line_template:rectangle_19114:template_134/child | 3146.83 | 3149.65 | 3149.65 |
| GX5 | reference-near | line_template:rectangle_85795:template_72 | 0.000132077 | 5.05141 | 8.0538 |
| GX5 | ranked | G0:181:973 | 1051.35 | 1049.05 | 1049.05 |
| GX5 | ranked | line_template:rectangle_85795:template_72/child | 3.06901 | 7.63516 | 8.27634 |
| GX5 | ranked | G0:181:1029 | 1027.15 | 1024.85 | 1024.85 |
| Am2-150 | ranked | G0:30:148 | 17.3135 | 10.2184 | 12.681 |
| Am2-150 | ranked | G0:30:33 | 24.8809 | 16.2645 | 12.9493 |
| Am2-150 | reference-near | G1:30:137/child | 7.53131 | 17.7662 | 17.7662 |
| Am2-150 | ranked | G0:30:30 | 281.809 | 277.658 | 303.46 |
| Am2-28019 | ranked | G1:15:168 | 22.6419 | 22.6418 | 27.024 |
| Am2-28019 | ranked | line_template:rectangle_210761:template_48 | 63586.6 | 63586.6 | 63586.6 |
| Am2-28019 | reference-near | G1:15:0/child | 7.91533 | 7.91533 | 8.67062 |
| Am3-0 | reference-near | G0:43:22605 | 8.54181 | 8.54184 | 12.4256 |
| Am3-0 | ranked | G1:43:299 | 19.2789 | 19.2789 | 18.0296 |
| Am3-0 | ranked | G1:43:299/child | 18.6854 | 18.6854 | 15.1757 |
| Am3-0 | ranked | G1:43:296 | 20.2231 | 20.2231 | 28.5793 |
| SS03-17 | ranked | G1:1:5355 | 4.50246 | 4.50246 | unknown |
| SS03-17 | ranked | G1:1:71 | 8.35149 | 8.35149 | unknown |
| SS03-17 | ranked | G1:1:1308/child | 6.91297 | 6.91297 | unknown |
| SS03-17 | reference-near | G1:1:24646/child | 2.24395 | 2.24395 | unknown |
| SS03-17 | ranked | G0:1:83 | 10.5745 | 10.5745 | unknown |
| SS03-19 | reference-near | line_template:rectangle_164603:template_55/child | 1.49621 | 5.84019 | unknown |
| SS03-19 | ranked | line_template:rectangle_164603:template_55 | 2.82384 | 6.16456 | unknown |
| SS03-19 | ranked | G0:1:60 | 3.92497 | 9.5518 | unknown |
| SS03-19 | ranked | G1:1:533/child | 2.77485 | 6.99438 | unknown |
| SS03-16 | ranked | G0:1:31 | 3.93599 | 6.22356 | unknown |
| SS03-16 | reference-near | G0:1:5587/child | 2.14354 | 5.02402 | unknown |
| SS03-16 | ranked | G0:1:60 | 2.55821 | 6.44558 | unknown |
| SS21-20 | reference-near | G0:0:480/child | 1.49934 | 7.30378 | unknown |
| SS21-20 | ranked | G1:0:1970 | 4.29401 | 2.15606 | unknown |
| SS21-20 | ranked | G1:0:2 | 2.01746 | 5.45065 | unknown |

## Notes

The packet keeps historical player/camera subsets, raw junction continuation evidence and refit attempts visible. It makes no claim beyond this development corpus.

## Visual review

The final original-plus-adjusted selection is usable on seven of nine views. The new
source fixes GX5. It also introduces higher-ranked wrong courts on GX0 and Am2-28019.

| view | final selection | ruling | change from stage 5 |
| --- | --- | --- | --- |
| GX0 | `line_template:rectangle_19114:template_134/child` | wrong_court | regressed; previous usable candidate now ranks 4 |
| GX5 | `line_template:rectangle_85795:template_72/child` | usable | fixed the missing foreground-court proposal |
| Am2-150 | `G1:30:137/child` | usable | unchanged |
| Am2-28019 | `line_template:rectangle_210761:template_48` | wrong_court | regressed; previous usable candidate now ranks 2 |
| Am3-0 | `G1:43:299/child` | usable | unchanged |
| SS03-17 | `G1:1:1308/child` | usable | unchanged |
| SS03-19 | `G1:1:533/child` | usable | unchanged |
| SS03-16 | `G0:1:31` | usable | unchanged |
| SS21-20 | `G1:0:1970` | usable | unchanged |

Preflight and full-run source settings, ordering, counts, contamination checks and
proposal totals match on all nine views; elapsed times differ. The previous stage-5
selections retain their corners, homographies, gates and ranking evidence exactly. The
legacy selections also remain exact. Collision handling therefore still changes identity
and provenance only; it does not change a scientific conclusion.

An independent code and packet review confirmed that admission cannot read references or
player gates, the final scorer is unchanged, and the saved player diagnostic preserves
the final score order. The run host lacked the frozen-helper checksum ledger, so the
packaged integrity check was unavailable. A byte comparison against the local snapshots
confirmed all six imported files. The review also found an unused helper search path; it
resolved no import and was removed after the run without recomputing the packet.

## Decision

Directly pooling the line-template candidates under the current scorer is rejected. It
falls from eight usable final selections to seven. The actionable blocker is
**admission**. The wrong line-template winners are admitted with only 3 and 2 visible
markings per direction on GX0, and 3 and 6 on Am2-28019. Their support means are therefore
based on too little of the court. The correct new GX5 candidate has 6 and 6 visible
markings. Misranking is the symptom; source admission supplies the weak evidence that
causes it. Proposal and refit are not the remaining problem.

A bounded follow-up is warranted. Require a minimum number of visible markings in each
direction before the 256-candidate admission cap, then rerun so later candidates can
refill the pool. The saved-pool diagnostic in `line_visibility_sensitivity.json` keeps the
usable winner on all nine development views at thresholds 4, 5 and 6. Test thresholds 3
through 6, keep the final paint score and refit fixed, and check unused scenes before
adopting a rule.

The player-coverage filter in `player_coverage_sensitivity.json` also selects the usable
candidate on all nine saved pools. It was inspected after the visual rulings and couples
court detection to player tracking, so it remains a secondary comparison rather than the
next rule.
