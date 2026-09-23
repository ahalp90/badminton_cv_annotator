# Nine-case SVD matcher timing

SVD12 used **52.9% less summed matcher wall time** than full16 across nine
development cases: 7,329.534 s including SVD ranking versus 15,545.095 s.
The summed saving was 8,215.561 s, and the median per-case direct wall-time
ratio was 2.20×. These are independently timed arms on a shared host, so the
direct ratios include timing variation as well as removed work.

| Case | full16 wall (s) | SVD12 plus ranking wall (s) | Reduction |
|---|---:|---:|---:|
| GX0 (`gxBQ_window_00_frame_0`) | 4,825.544 | 1,799.536 | 62.7% |
| GX5 (`gxBQ_window_00_frame_5`) | 2,411.200 | 1,633.293 | 32.3% |
| Am2-150 (`am2_window_00_frame_150`) | 1,847.342 | 894.344 | 51.6% |
| Am2-28019 (`am2_window_01_frame_28019`) | 2,397.647 | 1,367.936 | 42.9% |
| Am3-0 (`am3_window_00_frame_0`) | 2,427.837 | 1,013.492 | 58.3% |
| SS03-17 (`shuttleset_03_scene_0017`) | 516.382 | 235.140 | 54.5% |
| SS03-19 (`shuttleset_03_scene_0019`) | 304.999 | 141.790 | 53.5% |
| SS03-16 (`shuttleset_03_scene_0016`) | 477.639 | 107.802 | 77.4% |
| SS21-20 (`shuttleset_21_scene_0020`) | 336.505 | 136.203 | 59.5% |

The full16 run spent 47.8% of its wall time on pairs retained by SVD12
(`full16_retained_work_fraction_wall` = 0.47835). For those same pairs, the
SVD12/full16 wall-time ratio was 0.986
(`shared_pair_wall_drift_ratio_svd12_over_full16`). This near-equal shared work
supports pruning as the main source of the aggregate saving, while individual
direct ratios remain sensitive to host timing variation. Historical-cache
comparisons passed for 2,160 pairs; shared-arm comparisons passed for 1,188
pairs, with zero mismatches.

The run used one repeat, six workers and one numerical thread per worker.
It measured the matcher stage only. Preparation, image loading, full court
scoring, refitting, scene processing and detector runtime are excluded.
Candidate retention and accuracy evidence is [separate](../evidence/webui_followup3_20260922/review_20260923/automatic_retention/README.md);
these timings make no end-to-end or accuracy claim. Detailed numbers and
comparisons are in [`results.json.gz`](results.json.gz) and
[`history_check.json.gz`](history_check.json.gz).
