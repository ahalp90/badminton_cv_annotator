# W5 holistic court-detector pilot

This is a development/fit packet on the frozen corpus. It compares the legacy paint-first
baseline, whole-court scoring of original candidates, and the same scoring over original
plus locally adjusted candidates. The JSON and CSV retain `A`, `B` and `C` as historical
field names.

## Completed views

| view | legacy paint-first | original candidates | original + adjusted candidates | adjusted candidates | determinism |
| --- | --- | --- | --- | ---: | --- |
| GX0 | 22:4588 | G1:143:273 (provisional_for_review) | G0:22:4580/child (provisional_for_review) | 486 | pass |
| Am2-150 | 30:33 | G0:32:3188 (provisional_for_review) | G1:30:9149/child (provisional_for_review) | 229 | pass |
| Am2-28019 | 15:168 | G1:0:777 (provisional_for_review) | G1:0:35/child (provisional_for_review) | 341 | pass |
| Am3-0 | 43:296 | G1:44:25 (provisional_for_review) | G1:44:25 (provisional_for_review) | 429 | pass |
| SS03-19 | 1:19 | G0:1:74 (provisional_for_review) | G1:1:533/child (provisional_for_review) | 505 | pass |

## Known diagnostic controls

The positive and negative controls are directionally separated in the saved two-direction Q readouts, which is consistent with their prior rulings. This is a diagnostic result only: no threshold or automatic pass/fail was applied.

| view | control | expected role | automatic pool | hard-valid | Q_geom | Q_paint10 | status |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Am2-150 | 30:33 | positive_approved | yes | yes | 0.459007 | 0.324535 | diagnostic-only; no gate |
| Am2-28019 | 184:4123 | negative_rejected_false_paint | no | yes | 0.0251343 | 6.91502e-07 | diagnostic-only; no gate |
| SS03-19 | 1:60 | positive_usable | yes | yes | 0.914859 | 0.740931 | diagnostic-only; no gate |
| SS03-19 | 165:6702 | negative_rejected_hallucinated | no | yes | 0.131008 | 0.0912309 | diagnostic-only; no gate |

## Notes

The packet keeps historical player/camera subsets, raw junction continuation evidence and refit attempts visible. It makes no claim beyond this development corpus.
