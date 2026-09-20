# W5 holistic court-detector pilot

This is a development/fit packet on the frozen corpus. It compares the legacy paint-first
baseline, whole-court scoring of original candidates, and the same scoring over original
plus locally adjusted candidates. The JSON and CSV retain `A`, `B` and `C` as historical
field names.

## Completed views

| view | legacy paint-first | original + adjusted: initial | original + adjusted: camera-filtered | original + adjusted: span-weighted | final status | adjusted candidates | determinism |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| GX0 | 22:4588 | G0:22:4580/child | G0:22:4580/child | G0:22:4580/child | provisional_for_review | 486 | pass |
| Am2-150 | 30:33 | G1:30:9149/child | G1:30:9149/child | G1:30:137/child | provisional_for_review | 229 | pass |
| Am3-0 | 43:296 | G1:44:25 | G1:43:299/child | G1:43:299/child | provisional_for_review | 429 | pass |

## Stopped views

The following view stopped before scoring because the required candidate-ID collision assertion fired. No candidate IDs were remapped.

| view | reason |
| --- | --- |
| am2_window_01_frame_28019 | am2_window_01_frame_28019: G0/G1 candidate-ID collision: ['0:1338'] |
| shuttleset_03_scene_0019 | shuttleset_03_scene_0019: G0/G1 candidate-ID collision: ['1:3252', '1:533', '1:7480'] |

## Top candidates after each scoring change

This pool contains original and valid locally adjusted candidates. The first order rejects
implausible camera geometry and uses the plain mean of each marking's paint evidence. The
final order weights each marking by its visible span. It applies the same weighting to the
geometry fallback when needed.

| view | camera-filtered top candidate | span-weighted top candidate | status |
| --- | --- | --- | --- |
| GX0 | G0:22:4580/child | G0:22:4580/child | provisional_for_review |
| Am2-150 | G1:30:9149/child | G1:30:137/child | provisional_for_review |
| Am3-0 | G1:43:299/child | G1:43:299/child | provisional_for_review |

## Known diagnostic controls

The control readouts do not show a consistent positive-versus-negative separation in both two-direction Q measures. This is a diagnostic result only: no threshold or automatic pass/fail was applied.

| view | control | expected role | automatic pool | hard-valid | camera eligible | initial rank | camera-filtered rank | span-weighted rank | Q_geom | Q_paint10 | status |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Am2-150 | 30:33 | positive_approved | yes | yes | yes | 45 | 33 | 24 | 0.459007 | 0.324535 | diagnostic-only; no gate |

## Contrast-probe sensitivity

Each row reranks the original-plus-adjusted pool with the camera filter and visible-span
weighting, using the saved raw ridge arrays. Probe 10 remains the named pilot setting.

| view | probe | span-weighted top candidate | status | control target | control error |
| --- | ---: | --- | --- | --- | ---: |
| GX0 | 5 | G0:22:4579/child | provisional_for_review | approved_supplied_direction_control | 24.6564 |
| GX0 | 10 | G0:22:4580/child | provisional_for_review | approved_supplied_direction_control | 10.9798 |
| GX0 | 15 | G0:22:4580/child | provisional_for_review | approved_supplied_direction_control | 10.9798 |
| GX0 | 20 | G1:22:137/child | provisional_for_review | approved_supplied_direction_control | 9.70312 |
| Am2-150 | 5 | G1:30:137/child | provisional_for_review | approved_supplied_direction_control | 7.53131 |
| Am2-150 | 10 | G1:30:137/child | provisional_for_review | approved_supplied_direction_control | 7.53131 |
| Am2-150 | 15 | G0:30:32/child | provisional_for_review | approved_supplied_direction_control | 20.7344 |
| Am2-150 | 20 | G1:30:21/child | provisional_for_review | approved_supplied_direction_control | 14.6109 |
| Am3-0 | 5 | G1:32:180/child | provisional_for_review | approved_supplied_direction_control | 21.061 |
| Am3-0 | 10 | G1:43:299/child | provisional_for_review | approved_supplied_direction_control | 18.6854 |
| Am3-0 | 15 | G1:42:1761/child | provisional_for_review | approved_supplied_direction_control | 20.1578 |
| Am3-0 | 20 | G1:42:1757/child | provisional_for_review | approved_supplied_direction_control | 19.954 |

## Reference diagnostics after ranking lock

Reference metrics were joined after the automatic rankings and sensitivity orders were written.

| view | origin | supplied-control error | frozen-reference error | visible-landmark max error |
| --- | --- | ---: | ---: | ---: |
| GX0 | G0:22:4580/child | 10.9798 | 12.9076 | 13.3603 |
| GX0 | G0:22:4588 | 15.3241 | 19.0347 | 13.5919 |
| GX0 | G0:22:4579 | 33.0374 | 28.2083 | 21.4217 |
| GX0 | G1:22:270 | 33.0374 | 28.2083 | 29.51 |
| Am2-150 | G0:30:148 | 17.3135 | 10.2184 | 12.681 |
| Am2-150 | G1:30:137/child | 7.53131 | 17.7662 | 17.7662 |
| Am2-150 | G0:30:30 | 281.809 | 277.658 | 303.46 |
| Am2-150 | G0:30:33 | 24.8809 | 16.2645 | 12.9493 |
| Am3-0 | G1:43:299 | 19.2789 | 19.2789 | 18.0296 |
| Am3-0 | G1:43:296 | 20.2231 | 20.2231 | 28.5793 |
| Am3-0 | G1:43:299/child | 18.6854 | 18.6854 | 15.1757 |

## Notes

The packet keeps historical player/camera subsets, raw junction continuation evidence and refit attempts visible. It makes no claim beyond this development corpus.
