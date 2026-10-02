# Temporal union evaluation

This table preserves native per-frame, common-union per-frame and shared-median choices for GX and Am3. Exact candidate IDs, scores and anchor homographies are in [temporal/results.json.gz](temporal/results.json.gz). Am3 full-frame previews are quality-95 lossy JPEGs; canonical GX review sheets and far-end crops remain linked raw evidence.

| Cohort | Access | Target | Criterion | Origin case | Arm | Candidate | Line score | Paint score | Preview |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| gx | native_per_frame | gxBQ_window_00_frame_0 | line | gxBQ_window_00_frame_0 | G1 | `22:137` | 0.3167559508266173 | 0.6363636363636364 |  |
| gx | native_per_frame | gxBQ_window_00_frame_0 | paint | gxBQ_window_00_frame_0 | G0 | `22:4588` | 0.2914252044933242 | 0.8636363636363636 |  |
| gx | native_per_frame | gxBQ_window_00_frame_5 | line | gxBQ_window_00_frame_5 | G0 | `181:1029` | 0.2733831196757967 | 0.2727272727272727 |  |
| gx | native_per_frame | gxBQ_window_00_frame_5 | paint | gxBQ_window_00_frame_5 | G0 | `181:973` | 0.2652880650341827 | 0.45454545454545453 |  |
| gx | native_per_frame | gxBQ_window_00_frame_689 | line | gxBQ_window_00_frame_689 | G0 | `211:1855` | 0.2937041142553413 | 0.5454545454545454 |  |
| gx | native_per_frame | gxBQ_window_00_frame_689 | paint | gxBQ_window_00_frame_689 | G1 | `123:169` | 0.2529348372957853 | 0.7272727272727273 |  |
| gx | native_per_frame | gxBQ_window_01_frame_5111 | line | gxBQ_window_01_frame_5111 | G0 | `136:496` | 0.3222341168999408 | 0.45454545454545453 |  |
| gx | native_per_frame | gxBQ_window_01_frame_5111 | paint | gxBQ_window_01_frame_5111 | G0 | `136:541` | 0.30684197228928867 | 0.6363636363636364 |  |
| gx | native_per_frame | gxBQ_window_02_frame_5766 | line | gxBQ_window_02_frame_5766 | G0 | `113:64686` | 0.2861457672800679 | 0.36363636363636365 |  |
| gx | native_per_frame | gxBQ_window_02_frame_5766 | paint | gxBQ_window_02_frame_5766 | G1 | `16:7` | 0.23748704020292108 | 0.6363636363636364 |  |
| gx | native_per_frame | gxBQ_window_03_frame_77876 | line | gxBQ_window_03_frame_77876 | G1 | `17:172` | 0.2622409511891932 | 0.6363636363636364 |  |
| gx | native_per_frame | gxBQ_window_03_frame_77876 | paint | gxBQ_window_03_frame_77876 | G1 | `17:172` | 0.2622409511891932 | 0.6363636363636364 |  |
| gx | native_per_frame | gxBQ_window_04_frame_86088 | line | gxBQ_window_04_frame_86088 | G0 | `226:17683` | 0.2684332826880585 | 0.5454545454545454 |  |
| gx | native_per_frame | gxBQ_window_04_frame_86088 | paint | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.2270314727166587 | 0.8636363636363636 |  |
| gx | common_union_per_frame | gxBQ_window_00_frame_0 | line | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.32454340751651933 | 0.7272727272727273 |  |
| gx | common_union_per_frame | gxBQ_window_00_frame_0 | paint | gxBQ_window_00_frame_0 | G0 | `22:4588` | 0.2914252044933242 | 0.8636363636363636 |  |
| gx | common_union_per_frame | gxBQ_window_00_frame_5 | line | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.33922379911831524 | 0.7272727272727273 |  |
| gx | common_union_per_frame | gxBQ_window_00_frame_5 | paint | gxBQ_window_00_frame_0 | G1 | `143:158` | 0.2853413297216945 | 0.8181818181818182 |  |
| gx | common_union_per_frame | gxBQ_window_00_frame_689 | line | gxBQ_window_00_frame_0 | G0 | `106:93784` | 0.31456952375565495 | 0.0 |  |
| gx | common_union_per_frame | gxBQ_window_00_frame_689 | paint | gxBQ_window_00_frame_0 | G0 | `22:4588` | 0.24531785286669341 | 0.8636363636363636 |  |
| gx | common_union_per_frame | gxBQ_window_01_frame_5111 | line | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.3605814979436284 | 0.7272727272727273 |  |
| gx | common_union_per_frame | gxBQ_window_01_frame_5111 | paint | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.3605814979436284 | 0.7272727272727273 |  |
| gx | common_union_per_frame | gxBQ_window_02_frame_5766 | line | gxBQ_window_00_frame_0 | G0 | `106:93818` | 0.3150306718917971 | 0.25 |  |
| gx | common_union_per_frame | gxBQ_window_02_frame_5766 | paint | gxBQ_window_00_frame_0 | G1 | `143:158` | 0.25928141856966336 | 0.8181818181818182 |  |
| gx | common_union_per_frame | gxBQ_window_03_frame_77876 | line | gxBQ_window_00_frame_0 | G0 | `106:93818` | 0.31840674964753807 | 0.2857142857142857 |  |
| gx | common_union_per_frame | gxBQ_window_03_frame_77876 | paint | gxBQ_window_00_frame_0 | G1 | `143:158` | 0.23539947931137184 | 0.8636363636363636 |  |
| gx | common_union_per_frame | gxBQ_window_04_frame_86088 | line | gxBQ_window_00_frame_0 | G0 | `106:93818` | 0.31019686064769075 | 0.2857142857142857 |  |
| gx | common_union_per_frame | gxBQ_window_04_frame_86088 | paint | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.2270314727166587 | 0.8636363636363636 |  |
| gx | target_eligible_per_frame | gxBQ_window_00_frame_0 | line | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.32454340751651933 | 0.7272727272727273 |  |
| gx | target_eligible_per_frame | gxBQ_window_00_frame_0 | paint | gxBQ_window_00_frame_0 | G0 | `22:4588` | 0.2914252044933242 | 0.8636363636363636 |  |
| gx | target_eligible_per_frame | gxBQ_window_00_frame_5 | line | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.33922379911831524 | 0.7272727272727273 |  |
| gx | target_eligible_per_frame | gxBQ_window_00_frame_5 | paint | gxBQ_window_00_frame_0 | G1 | `143:158` | 0.2853413297216945 | 0.8181818181818182 |  |
| gx | target_eligible_per_frame | gxBQ_window_00_frame_689 | line | gxBQ_window_00_frame_0 | G0 | `106:93784` | 0.31456952375565495 | 0.0 |  |
| gx | target_eligible_per_frame | gxBQ_window_00_frame_689 | paint | gxBQ_window_00_frame_0 | G0 | `22:4588` | 0.24531785286669341 | 0.8636363636363636 |  |
| gx | target_eligible_per_frame | gxBQ_window_01_frame_5111 | line | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.3605814979436284 | 0.7272727272727273 |  |
| gx | target_eligible_per_frame | gxBQ_window_01_frame_5111 | paint | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.3605814979436284 | 0.7272727272727273 |  |
| gx | target_eligible_per_frame | gxBQ_window_02_frame_5766 | line | gxBQ_window_00_frame_0 | G0 | `106:93818` | 0.3150306718917971 | 0.25 |  |
| gx | target_eligible_per_frame | gxBQ_window_02_frame_5766 | paint | gxBQ_window_00_frame_0 | G1 | `143:158` | 0.25928141856966336 | 0.8181818181818182 |  |
| gx | target_eligible_per_frame | gxBQ_window_03_frame_77876 | line | gxBQ_window_00_frame_0 | G0 | `113:107988` | 0.33909650500244287 | 0.375 |  |
| gx | target_eligible_per_frame | gxBQ_window_03_frame_77876 | paint | gxBQ_window_00_frame_0 | G1 | `143:158` | 0.23539947931137184 | 0.8636363636363636 |  |
| gx | target_eligible_per_frame | gxBQ_window_04_frame_86088 | line | gxBQ_window_00_frame_0 | G0 | `113:107988` | 0.3422340007250573 | 0.375 |  |
| gx | target_eligible_per_frame | gxBQ_window_04_frame_86088 | paint | gxBQ_window_04_frame_86088 | G1 | `16:44` | 0.2270314727166587 | 0.8636363636363636 |  |
| gx | common_union_temporal | shared | line | gxBQ_window_00_frame_0 | G0 | `106:93818` | 0.3040690962199702 |  |  |
| gx | common_union_temporal | shared | paint | gxBQ_window_00_frame_0 | G1 | `143:158` | 0.25928141856966336 | 0.8181818181818182 |  |
| am3 | native_per_frame | am3_window_00_frame_0 | line | am3_window_00_frame_0 | G1 | `43:153` | 0.3542950617374748 | 0.8181818181818182 | [temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_153__full.jpg](temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_153__full.jpg) |
| am3 | native_per_frame | am3_window_00_frame_0 | paint | am3_window_00_frame_0 | G1 | `43:159` | 0.3443092252358326 | 1.0 | [temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_159__full.jpg](temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_159__full.jpg) |
| am3 | native_per_frame | am3_window_01_frame_10514 | line | am3_window_01_frame_10514 | G1 | `30:0` | 0.3726418154651563 | 0.8636363636363636 | [temporal/am3_overlays/am3_window_01_frame_10514__am3_window_01_frame_10514__G1__30_0__full.jpg](temporal/am3_overlays/am3_window_01_frame_10514__am3_window_01_frame_10514__G1__30_0__full.jpg) |
| am3 | native_per_frame | am3_window_01_frame_10514 | paint | am3_window_01_frame_10514 | G1 | `30:325` | 0.35264510819159406 | 0.9545454545454546 | [temporal/am3_overlays/am3_window_01_frame_10514__am3_window_01_frame_10514__G1__30_325__full.jpg](temporal/am3_overlays/am3_window_01_frame_10514__am3_window_01_frame_10514__G1__30_325__full.jpg) |
| am3 | common_union_per_frame | am3_window_00_frame_0 | line | am3_window_00_frame_0 | G1 | `43:153` | 0.3542950617374748 | 0.8181818181818182 | [temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_153__full.jpg](temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_153__full.jpg) |
| am3 | common_union_per_frame | am3_window_00_frame_0 | paint | am3_window_00_frame_0 | G1 | `43:159` | 0.3443092252358326 | 1.0 | [temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_159__full.jpg](temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_159__full.jpg) |
| am3 | common_union_per_frame | am3_window_01_frame_10514 | line | am3_window_00_frame_0 | G1 | `43:156` | 0.3851214988185808 | 0.6818181818181818 | [temporal/am3_overlays/am3_window_01_frame_10514__am3_window_00_frame_0__G1__43_156__full.jpg](temporal/am3_overlays/am3_window_01_frame_10514__am3_window_00_frame_0__G1__43_156__full.jpg) |
| am3 | common_union_per_frame | am3_window_01_frame_10514 | paint | am3_window_00_frame_0 | G1 | `43:451` | 0.3722781730139674 | 1.0 | [temporal/am3_overlays/am3_window_01_frame_10514__am3_window_00_frame_0__G1__43_451__full.jpg](temporal/am3_overlays/am3_window_01_frame_10514__am3_window_00_frame_0__G1__43_451__full.jpg) |
| am3 | target_eligible_per_frame | am3_window_00_frame_0 | line | am3_window_00_frame_0 | G1 | `43:153` | 0.3542950617374748 | 0.8181818181818182 | [temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_153__full.jpg](temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_153__full.jpg) |
| am3 | target_eligible_per_frame | am3_window_00_frame_0 | paint | am3_window_00_frame_0 | G1 | `43:159` | 0.3443092252358326 | 1.0 | [temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_159__full.jpg](temporal/am3_overlays/am3_window_00_frame_0__am3_window_00_frame_0__G1__43_159__full.jpg) |
| am3 | target_eligible_per_frame | am3_window_01_frame_10514 | line | am3_window_00_frame_0 | G1 | `43:156` | 0.3851214988185808 | 0.6818181818181818 | [temporal/am3_overlays/am3_window_01_frame_10514__am3_window_00_frame_0__G1__43_156__full.jpg](temporal/am3_overlays/am3_window_01_frame_10514__am3_window_00_frame_0__G1__43_156__full.jpg) |
| am3 | target_eligible_per_frame | am3_window_01_frame_10514 | paint | am3_window_00_frame_0 | G1 | `43:451` | 0.3722781730139674 | 1.0 | [temporal/am3_overlays/am3_window_01_frame_10514__am3_window_00_frame_0__G1__43_451__full.jpg](temporal/am3_overlays/am3_window_01_frame_10514__am3_window_00_frame_0__G1__43_451__full.jpg) |
| am3 | common_union_temporal | shared | line | am3_window_00_frame_0 | G1 | `43:153` | 0.3691998938528789 |  |  |
| am3 | common_union_temporal | shared | paint | am3_window_00_frame_0 | G1 | `43:451` | 0.3554541277796459 | 1.0 |  |

References were not used before selection. The complete saved score matrices and candidate populations are published, not just the winners. See the raw source links below.

## GX full views and far-end detail

| Target | Six selected overlays | Raw and paint-overlay far-end crops |
| --- | --- | --- |
| gxBQ_window_00_frame_0 | [Full views](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_00_frame_0.jpg) | [Far end](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_00_frame_0__far_end.png) |
| gxBQ_window_00_frame_5 | [Full views](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_00_frame_5.jpg) | [Far end](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_00_frame_5__far_end.png) |
| gxBQ_window_00_frame_689 | [Full views](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_00_frame_689.jpg) | [Far end](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_00_frame_689__far_end.png) |
| gxBQ_window_01_frame_5111 | [Full views](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_01_frame_5111.jpg) | [Far end](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_01_frame_5111__far_end.png) |
| gxBQ_window_02_frame_5766 | [Full views](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_02_frame_5766.jpg) | [Far end](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_02_frame_5766__far_end.png) |
| gxBQ_window_03_frame_77876 | [Full views](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_03_frame_77876.jpg) | [Far end](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_03_frame_77876__far_end.png) |
| gxBQ_window_04_frame_86088 | [Full views](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_04_frame_86088.jpg) | [Far end](../pixel_temporal/evaluation_20260922/results/gx/review_sheets/gxBQ_window_04_frame_86088__far_end.png) |

## Complete numerical records

- gx: [selections](../pixel_temporal/evaluation_20260922/results/gx/selection.json), [all candidate geometry](../pixel_temporal/evaluation_20260922/results/gx/candidates.json.gz), [all per-frame scores](../pixel_temporal/evaluation_20260922/results/gx/scores/), [registration and run settings](../pixel_temporal/evaluation_20260922/results/gx/manifest.json)
- am3: [selections](../pixel_temporal/evaluation_20260922/results/am3/selection.json), [all candidate geometry](../pixel_temporal/evaluation_20260922/results/am3/candidates.json.gz), [all per-frame scores](../pixel_temporal/evaluation_20260922/results/am3/scores/), [registration and run settings](../pixel_temporal/evaluation_20260922/results/am3/manifest.json)
