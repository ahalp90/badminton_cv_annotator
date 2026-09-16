# Do later GX frames keep the precise directions

## Gate

Gate 1 (16 GX0 directions vs approved control): 6.022569
Gate 2 (candidates 1183, 122 vs approved control): 0.423931
Both match the required values to six decimals. Coordinates are correct.

## Per-frame table

| frame | seconds | min angle to 1183 (deg) | closest ID | min angle to 122 (deg) | closest ID | set_bound (working px) | note |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.177489 | 737 | 1.794981 | 5179 | 6.022569 | valid |
| 5 | 0.083 | 0.360670 | 624 | 2.359041 | 2661 | 8.584324 | valid |
| 689 | 11.505 | 0.028834 | 4447 | 1.671211 | 2661 | 3.061835 | indicative only, camera may have moved |
| 5111 | 85.347 | 0.170598 | 2382 | 0.780710 | 4830 | 6.091456 | indicative only, camera may have moved |
| 5766 | 96.285 | 0.267957 | 473 | 2.563435 | 304 | 5.387308 | indicative only, camera may have moved |
| 77876 | 1300.426 | 0.365064 | 6897 | 0.251494 | 7295 | 3.067713 | indicative only, camera may have moved |
| 86088 | 1437.555 | 0.199823 | 1537 | 1.420394 | 4480 | 2.588390 | indicative only, camera may have moved |

## Answers

Frames 5111 and 77876 have raw homogeneous-vector comparisons to both control directions within about one degree. Frames 689, 77876 and 86088 have incidence lower bounds relative to GX0's approved control under 3.5 working pixels. These are chart-dependent angle descriptions and lower bounds, not one-degree sufficiency or exclusion tests; attainable fits and generated courts were not measured. Camera equivalence for the later frames is unverified.
