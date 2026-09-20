# W5 holistic court-detector pilot

This is a development/fit packet on the frozen corpus. The B and C orders are provisional readouts, not an acceptance rule.

## Completed views

| view | A paint-first | C pilot | C R1 | C R2 | C status | children | determinism |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| GX0 | G0:22:4588 | G0:22:4580/child | G0:22:4580/child | G0:22:4580/child | provisional_for_review | 486 | pass |
| GX5 | G0:181:973 | G0:0:21005 | G0:181:948/child | G0:181:948/child | provisional_for_review | 410 | pass |
| Am2-150 | G0:30:33 | G1:30:9149/child | G1:30:9149/child | G1:30:137/child | provisional_for_review | 228 | pass |
| Am2-28019 | G1:15:168 | G1:0:35/child | G1:15:164/child | G1:15:164/child | provisional_for_review | 340 | pass |
| Am3-0 | G1:43:296 | G1:44:25 | G1:43:299/child | G1:43:299/child | provisional_for_review | 429 | pass |
| SS03-17 | G1:1:71 | G1:1:1308/child | G1:1:1308/child | G1:1:1308/child | provisional_for_review | 462 | pass |
| SS03-19 | G0:1:60 | G1:1:533/child | G1:1:533/child | G1:1:533/child | provisional_for_review | 450 | pass |
| SS03-16 | G0:1:31 | G0:1:31 | G0:1:31 | G0:1:31 | provisional_for_review | 427 | pass |
| SS21-20 | G1:0:2 | G1:0:1970 | G1:0:1970 | G1:0:1970 | provisional_for_review | 435 | pass |

## Collision resolution

Parent candidates use source-qualified `origin_key` values for every join, ranking, diagnostic and gallery lookup. Raw candidate IDs remain source-local provenance. Exact cross-source geometry duplicates retain all source occurrences under one canonical parent when their W5-relevant gates agree; legacy-only source metadata stays attached to each occurrence. A same-source geometry duplicate, metadata mismatch or W5-gate mismatch stops that view as ambiguous.

| view | source occurrences | canonical parents | raw-ID collisions | duplicate geometry groups |
| --- | ---: | ---: | ---: | ---: |
| GX0 | 512 | 512 | 0 | 0 |
| GX5 | 512 | 512 | 0 | 0 |
| Am2-150 | 512 | 511 | 0 | 1 |
| Am2-28019 | 512 | 508 | 1 | 4 |
| Am3-0 | 512 | 512 | 0 | 0 |
| SS03-17 | 512 | 466 | 1 | 46 |
| SS03-19 | 512 | 457 | 3 | 55 |
| SS03-16 | 512 | 427 | 5 | 85 |
| SS21-20 | 512 | 435 | 5 | 77 |

Exact-geometry deduplication can change pool counts and diagnostic ranks without changing the evidence for retained geometries. In this packet the Am2-150 merge leaves 511 canonical parents, 511 fit attempts and 228 valid children; the `30:33` control's Q values and R2 rank are unchanged from the earlier stage-3 packet.

## Arm-A occurrence provenance

Arm A reports the canonical parent used for joins, rankings, diagnostics and gallery lookups. For an exact-geometry merge, the occurrence column identifies the source record that supplied the winning legacy score. Both parent and occurrence identities remain available in the packet records.

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

## R1 and R2 rank-1 selections

The C pool contains parents and valid children. R1 is the camera-eligible plain-mean paint order. R2 is the camera-eligible span-weighted order, with the span-weighted geometry fallback when needed.

| view | R1 rank-1 | R2 rank-1 | status |
| --- | --- | --- | --- |
| GX0 | G0:22:4580/child | G0:22:4580/child | provisional_for_review |
| GX5 | G0:181:948/child | G0:181:948/child | provisional_for_review |
| Am2-150 | G1:30:9149/child | G1:30:137/child | provisional_for_review |
| Am2-28019 | G1:15:164/child | G1:15:164/child | provisional_for_review |
| Am3-0 | G1:43:299/child | G1:43:299/child | provisional_for_review |
| SS03-17 | G1:1:1308/child | G1:1:1308/child | provisional_for_review |
| SS03-19 | G1:1:533/child | G1:1:533/child | provisional_for_review |
| SS03-16 | G0:1:31 | G0:1:31 | provisional_for_review |
| SS21-20 | G1:0:1970 | G1:0:1970 | provisional_for_review |

## Known diagnostic controls

The positive and negative controls are directionally separated in the saved two-direction Q readouts, which is consistent with their prior rulings. This is a diagnostic result only: no threshold or automatic pass/fail was applied.

| view | origin | raw candidate | expected role | automatic pool | hard-valid | camera eligible | pilot rank | R1 rank | R2 rank | Q_geom | Q_paint10 | status |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Am2-150 | diagnostic:am2_window_00_frame_150:30:33 | 30:33 | positive_approved | yes | yes | yes | 43 | 31 | 24 | 0.459007 | 0.324535 | diagnostic-only; no gate |
| Am2-28019 | diagnostic:am2_window_01_frame_28019:184:4123 | 184:4123 | negative_rejected_false_paint | no | yes | yes | 827 | 22 | 22 | 0.0251343 | 6.91502e-07 | diagnostic-only; no gate |
| SS03-19 | diagnostic:shuttleset_03_scene_0019:1:60 | 1:60 | positive_usable | yes | yes | yes | 23 | 23 | 33 | 0.914859 | 0.740931 | diagnostic-only; no gate |
| SS03-19 | diagnostic:shuttleset_03_scene_0019:165:6702 | 165:6702 | negative_rejected_hallucinated | no | yes | yes | 903 | 885 | 885 | 0.131008 | 0.0912309 | diagnostic-only; no gate |

## Contrast-probe sensitivity

Each row reranks the C pool under the same R1 + R2 rules from the saved raw ridge arrays. The table reports the R2 provisional rank-1; the JSON retains the separate R1 and R2 orders. Probe 10 remains the named pilot setting.

| view | probe | R2 rank-1 origin | status | control target | control error |
| --- | ---: | --- | --- | --- | ---: |
| GX0 | 5 | G0:22:4579/child | provisional_for_review | approved_supplied_direction_control | 24.6564 |
| GX0 | 10 | G0:22:4580/child | provisional_for_review | approved_supplied_direction_control | 10.9798 |
| GX0 | 15 | G0:22:4580/child | provisional_for_review | approved_supplied_direction_control | 10.9798 |
| GX0 | 20 | G1:22:137/child | provisional_for_review | approved_supplied_direction_control | 9.70312 |
| GX5 | 5 | G0:181:948/child | provisional_for_review | approved_supplied_direction_control | 1060.33 |
| GX5 | 10 | G0:181:948/child | provisional_for_review | approved_supplied_direction_control | 1060.33 |
| GX5 | 15 | G1:31:1332 | provisional_for_review | approved_supplied_direction_control | 4064.92 |
| GX5 | 20 | G1:31:1338/child | provisional_for_review | approved_supplied_direction_control | 4460.49 |
| Am2-150 | 5 | G1:30:137/child | provisional_for_review | approved_supplied_direction_control | 7.53131 |
| Am2-150 | 10 | G1:30:137/child | provisional_for_review | approved_supplied_direction_control | 7.53131 |
| Am2-150 | 15 | G0:30:32/child | provisional_for_review | approved_supplied_direction_control | 20.7344 |
| Am2-150 | 20 | G1:30:21/child | provisional_for_review | approved_supplied_direction_control | 14.6109 |
| Am2-28019 | 5 | G1:15:164/child | provisional_for_review | approved_supplied_direction_control | 18.4088 |
| Am2-28019 | 10 | G1:15:164/child | provisional_for_review | approved_supplied_direction_control | 18.4088 |
| Am2-28019 | 15 | G1:15:4930 | provisional_for_review | approved_supplied_direction_control | 36.8172 |
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
| SS03-19 | 15 | G0:1:533/child | provisional_for_review | approved_supplied_direction_control | 2.6356 |
| SS03-19 | 20 | G0:1:7464/child | provisional_for_review | approved_supplied_direction_control | 3.39747 |
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
| GX0 | reference-near | G0:22:4588/child | 7.55582 | 14.2394 | 16.463 |
| GX0 | ranked | G0:22:4579 | 33.0374 | 28.2083 | 21.4217 |
| GX0 | ranked | G1:22:270 | 33.0374 | 28.2083 | 29.51 |
| GX0 | ranked | G0:22:4588 | 15.3241 | 19.0347 | 13.5919 |
| GX0 | ranked | G0:22:4580/child | 10.9798 | 12.9076 | 13.3603 |
| GX5 | ranked | G0:181:948/child | 1060.33 | 1058.03 | 1058.03 |
| GX5 | ranked | G0:181:996 | 1027.83 | 1025.52 | 1025.52 |
| GX5 | reference-near | G1:211:2312/child | 642.842 | 638.791 | 1066.33 |
| GX5 | ranked | G0:181:973 | 1051.35 | 1049.05 | 1049.05 |
| GX5 | ranked | G0:181:1029 | 1027.15 | 1024.85 | 1024.85 |
| Am2-150 | ranked | G0:30:30 | 281.809 | 277.658 | 303.46 |
| Am2-150 | ranked | G0:30:148 | 17.3135 | 10.2184 | 12.681 |
| Am2-150 | reference-near | G1:30:137/child | 7.53131 | 17.7662 | 17.7662 |
| Am2-150 | ranked | G0:30:33 | 24.8809 | 16.2645 | 12.9493 |
| Am2-28019 | ranked | G1:15:3024 | 36.8172 | 36.8172 | 34.4764 |
| Am2-28019 | ranked | G1:15:168 | 22.6419 | 22.6418 | 27.024 |
| Am2-28019 | ranked | G1:15:164/child | 18.4088 | 18.409 | 18.4089 |
| Am2-28019 | reference-near | G1:15:0/child | 7.91533 | 7.91533 | 8.67062 |
| Am3-0 | ranked | G1:43:296 | 20.2231 | 20.2231 | 28.5793 |
| Am3-0 | ranked | G1:43:299 | 19.2789 | 19.2789 | 18.0296 |
| Am3-0 | reference-near | G0:43:22605 | 8.54181 | 8.54184 | 12.4256 |
| Am3-0 | ranked | G1:43:299/child | 18.6854 | 18.6854 | 15.1757 |
| SS03-17 | ranked | G1:1:71 | 8.35149 | 8.35149 | unknown |
| SS03-17 | reference-near | G1:1:24646/child | 2.24395 | 2.24395 | unknown |
| SS03-17 | ranked | G0:1:83 | 10.5745 | 10.5745 | unknown |
| SS03-17 | ranked | G1:1:1308/child | 6.91297 | 6.91297 | unknown |
| SS03-17 | ranked | G1:1:5355 | 4.50246 | 4.50246 | unknown |
| SS03-19 | ranked | G0:1:74 | 4.10422 | 7.33574 | unknown |
| SS03-19 | ranked | G0:1:60 | 3.92497 | 9.5518 | unknown |
| SS03-19 | ranked | G1:1:533/child | 2.77485 | 6.99438 | unknown |
| SS03-19 | reference-near | G0:1:1278/child | 2.2068 | 6.38648 | unknown |
| SS03-16 | ranked | G0:1:60 | 2.55821 | 6.44558 | unknown |
| SS03-16 | ranked | G0:1:31 | 3.93599 | 6.22356 | unknown |
| SS03-16 | reference-near | G0:1:5587/child | 2.14354 | 5.02402 | unknown |
| SS21-20 | ranked | G1:0:1970 | 4.29401 | 2.15606 | unknown |
| SS21-20 | ranked | G1:0:2 | 2.01746 | 5.45065 | unknown |
| SS21-20 | reference-near | G0:0:480/child | 1.49934 | 7.30378 | unknown |

## Notes

The packet keeps historical player/camera subsets, raw junction continuation evidence and refit attempts visible. It makes no claim beyond this development corpus.
Visual review is still pending: `visual_rulings.json` has status `pending_review` and no rulings have been applied.
