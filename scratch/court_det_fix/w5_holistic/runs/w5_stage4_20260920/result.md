# W5 holistic court-detector pilot

This is a development/fit packet on the frozen corpus. It compares the legacy paint-first
baseline, whole-court scoring of original candidates, and the same scoring over original
plus locally adjusted candidates. The JSON and CSV retain `A`, `B` and `C` as historical
field names.

## Completed views

| view | legacy paint-first | original + adjusted: initial | original + adjusted: camera-filtered | original + adjusted: span-weighted | final status | adjusted candidates | determinism |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| GX5 | 181:973 | G0:0:21005 | G0:181:948/child | G0:181:948/child | provisional_for_review | 410 | pass |

## Stopped views

The following view stopped before scoring because the required candidate-ID collision assertion fired. No candidate IDs were remapped.

| view | reason |
| --- | --- |
| shuttleset_03_scene_0017 | shuttleset_03_scene_0017: G0/G1 candidate-ID collision: ['1:80'] |
| shuttleset_03_scene_0016 | shuttleset_03_scene_0016: G0/G1 candidate-ID collision: ['1:14', '1:149', '1:167', '1:31', '1:913'] |
| shuttleset_21_scene_0020 | shuttleset_21_scene_0020: G0/G1 candidate-ID collision: ['0:133', '0:2', '0:208', '0:38', '0:71'] |

## Top candidates after each scoring change

This pool contains original and valid locally adjusted candidates. The first order rejects
implausible camera geometry and uses the plain mean of each marking's paint evidence. The
final order weights each marking by its visible span. It applies the same weighting to the
geometry fallback when needed.

| view | camera-filtered top candidate | span-weighted top candidate | status |
| --- | --- | --- | --- |
| GX5 | G0:181:948/child | G0:181:948/child | provisional_for_review |

## Known diagnostic controls

The control readouts do not show a consistent positive-versus-negative separation in both two-direction Q measures. This is a diagnostic result only: no threshold or automatic pass/fail was applied.

| view | control | expected role | automatic pool | hard-valid | camera eligible | initial rank | camera-filtered rank | span-weighted rank | Q_geom | Q_paint10 | status |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

## Contrast-probe sensitivity

Each row reranks the original-plus-adjusted pool with the camera filter and visible-span
weighting, using the saved raw ridge arrays. Probe 10 remains the named pilot setting.

| view | probe | span-weighted top candidate | status | control target | control error |
| --- | ---: | --- | --- | --- | ---: |
| GX5 | 5 | G0:181:948/child | provisional_for_review | approved_supplied_direction_control | 1060.33 |
| GX5 | 10 | G0:181:948/child | provisional_for_review | approved_supplied_direction_control | 1060.33 |
| GX5 | 15 | G1:31:1332 | provisional_for_review | approved_supplied_direction_control | 4064.92 |
| GX5 | 20 | G1:31:1338/child | provisional_for_review | approved_supplied_direction_control | 4460.49 |

## Reference diagnostics after ranking lock

Reference metrics were joined after the automatic rankings and sensitivity orders were written.

| view | origin | supplied-control error | frozen-reference error | visible-landmark max error |
| --- | --- | ---: | ---: | ---: |
| GX5 | G0:181:948/child | 1060.33 | 1058.03 | 1058.03 |
| GX5 | G0:181:973 | 1051.35 | 1049.05 | 1049.05 |
| GX5 | G0:181:1029 | 1027.15 | 1024.85 | 1024.85 |
| GX5 | G0:181:996 | 1027.83 | 1025.52 | 1025.52 |

## Notes

The packet keeps historical player/camera subsets, raw junction continuation evidence and refit attempts visible. It makes no claim beyond this development corpus.
