# L1 result — fixed-budget admission probe

The broader admission rule finds a close court that score-only admission drops on Am2 frame 28019, pair 15. It also causes a severe regression on Am3 frame 0 under arm B, pair 43. The gain survives the existing geometry/player masks and the original per-pair shortlist cap, but the regression is not tolerable. This exact rule does not merit an all-pair matcher trial.

The metric is maximum corner distance in 960-by-540 working pixels, with the existing 180-degree relabelling. Arm A is the original `distinct[:512]`; arm B preserves the first 256 and admits 256 more rows by the fixed signed centre/log-span quantile cells. Both arms admit 512 rows per direction and form 262,144 combined courts. The control only measures the products; it never selects them.

| pair | combined | geometry/player | per-pair shortlist | result |
| --- | ---: | ---: | ---: | --- |
| Am3-R 43 | 46.0998 → 46.0998 | 46.0998 → 46.0998 | 48.5665 → 48.5665 | unchanged |
| Am3-B 43 | 4.2708 → 23.2484 | 4.2708 → 23.2484 | 4.2709 → 23.2484 | regressed |
| GX0-B 143 | 34.3268 → 34.3268 | 34.3268 → 34.3268 | 34.3268 → 34.3268 | unchanged |
| GX0-B 22 | 7.6621 → 7.6621 | 7.6621 → 7.6621 | 7.6621 → 7.6621 | unchanged |
| Am2-B 15 | 61.6207 → 7.5291 | 61.6207 → 7.5291 | 81.2626 → 7.5291 | improved |
| SS03-19-B 1 | 3.1340 → 3.1340 | 3.1340 → 3.1340 | 3.1340 → 3.1340 | unchanged |

The mask stage used the frozen `geometry()` and `zone_net.player_fractions()` functions. The cap used the frozen `finite_scores()` and `retain()` functions. No full-pool pixel score was calculated. The screen took 558.066 seconds with one process and reached 880.738 MiB peak RSS.

The compact [comparison table](comparison.csv) and [witnesses](witnesses.json) record counts, axis IDs, parameters and corners for every minimum. Overlays are available for [Am3-R](overlays/am3_window_00_frame_0_R_43.png), [Am3-B](overlays/am3_window_00_frame_0_B_43.png), [GX0 pair 143](overlays/gxBQ_window_00_frame_0_B_143.png), [GX0 pair 22](overlays/gxBQ_window_00_frame_0_B_22.png), [Am2](overlays/am2_window_01_frame_28019_B_15.png) and [SS03-19](overlays/shuttleset_03_scene_0019_B_1.png).
