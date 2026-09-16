# Lens distortion check on the GX footage

| case_id | group | n rows | median abs sagitta (norm) | max abs sagitta (norm) | max abs sagitta (px) | frac positive | spearman r |
|---|---|---|---|---|---|---|---|
| gxBQ_window_00_frame_0 | gx | 8 | 0.00119 | 0.00405 | 7.78 | 0.50 | 0.05 |
| gxBQ_window_00_frame_5 | gx | 8 | 0.00167 | 0.00475 | 9.13 | 0.25 | -0.38 |
| am2_window_00_frame_150 | amateur | 16 | 0.00256 | 0.00622 | 11.95 | 0.44 | -0.03 |
| am2_window_01_frame_28019 | amateur | 12 | 0.00132 | 0.00374 | 7.19 | 0.58 | -0.26 |
| am3_window_00_frame_0 | amateur | 13 | 0.00181 | 0.00642 | 12.33 | 0.46 | -0.21 |
| shuttleset_03_scene_0016 | broadcast | 18 | 0.00250 | 0.00679 | 6.52 | 0.39 | -0.04 |
| shuttleset_03_scene_0019 | broadcast | 17 | 0.00217 | 0.00819 | 7.86 | 0.41 | 0.04 |
| shuttleset_21_scene_0020 | broadcast | 14 | 0.00286 | 0.00736 | 7.07 | 0.21 | 0.04 |

Full machine-readable table: `table.csv`.

GX does not show a consistent outward bow that grows with radius: its two cases' fraction of rows bowing away from centre is 0.50 and 0.25 (near chance), and the Spearman correlation between absolute sagitta and distance from centre is weak and inconsistent in sign (0.05 and -0.38). The amateur views show the same pattern, with fractions of positive sagitta around 0.44 to 0.58 and correlations that are all weak and negative. The broadcast control does not sit below GX and amateur as a clean baseline; its median normalised absolute sagitta (0.0022 to 0.0029) is if anything a little higher than GX's (0.0012 to 0.0017), so the broadcast control bows about as much as, or more than, GX. The largest GX sagitta measured was 9.13 native pixels, out of a 1920-pixel-wide frame.

Gate: the broadcast control's median absolute sagitta is not small compared with GX's, so this test does not show evidence of lens distortion specific to the GX footage. The bow measured across all three groups looks like ordinary line-fragment fitting noise rather than a systematic barrel or pincushion effect. No fix is proposed here.
