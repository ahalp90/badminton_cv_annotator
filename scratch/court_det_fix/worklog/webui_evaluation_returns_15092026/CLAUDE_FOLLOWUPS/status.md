# Status

`R` is the remote experiment root.

Updated 2026-09-16. Eight checks done; no run on the compute host in flight. Read `HANDOVER.md` first.

## Why this set

The user asked which follow-ups would move toward a standalone court detector. The answer: the rank table tests a threshold-free identifiability gate; the GX0 rows check picks between "reject non-court lines" and "extract paint more precisely"; the GX direction check decides whether temporal pooling can help GX; the cap-loss check finds where close courts vanish inside the matcher; the distortion check tests whether the GX footage breaks the straight-line assumption. The per-view audit is held because the mat border persists in every frame and the dominance argument predicts its main result.

## Checks

| Check | Where | Agent | Output folder | State |
| --- | --- | --- | --- | --- |
| GX0 rows overlay | local | Sonnet | `gx0_rows/` | done; gate passed; none of the three rows is court paint |
| GX directions along the video | local | Sonnet | `gx_directions/` | done; gate passed; frames 0 and 5 lose the y-direction by about 2 degrees, frames 5111 and 77876 keep both within a degree |
| Rank table on exported winners | local | Sonnet | `identity_diagnostics/` | done; gate passed; rank 7 only on scene 19's false winner, sub-pixel separation only on Am2-28019's; every other unusable court has rank 8 and separation over 1.8 px |
| Lens distortion on GX, merged-row endpoints | local | Sonnet | `distortion/` | done; gate failed honestly: the method measures merge noise, not the lens; inconclusive |
| Lens distortion on GX, painted ridge | local | Sonnet | `distortion_ridge/` | done; gate failed on both controls; GX bows outward 2 to 5 px on all four edges but within method noise; distortion bounded at a few pixels, not the lever; no third attempt |
| Cap loss inside the matcher | the compute host | Fable | `cap_loss/` | done; both gates passed; the per-pair cap is not the loss on GX0 M or Am2-28019 B, and accounts for 4 px of 40 on Am3-0 R; the courts-before-the-gate check then cleared the two masks, and the axis-matching replay in `scratch/court_det_fix/line_identity/` placed the loss at the per-direction cap of 512 matchings |

## Held

The per-view audit. See `SCHEDULE.md`.

## Remote state

No job is running. The instrumented copies and the runs `cap_loss_20260915_110406` and `pregate_20260915_115949` remain under `R/cap_loss/`; all five jobs exited 0.

## Follow-on runs

| Check | Where | Agent | Output folder | State |
| --- | --- | --- | --- | --- |
| Person-box mask replay on GX0 | local | Sonnet | `person_mask_replay/` | done; gate passed; midpoint-in-box rule keeps the control y-direction, bound 6.02 to 1.22 px, best fit 6.80 to 1.50 px; gain arrives via reallocation; one view |
| Proposals before the player gate | the compute host | Fable | `pregate_loss/` | done; both gates passed; the nearest court before the masks is the nearest usable court on both case-arms (33.8 px GX0 M, 40.1 px Am3-0 R); nothing within 20 px among 22 and 17 million combined courts; the loss sits in the axis matching, with the per-axis cap of 512 still unseparated from enumeration |

## Continued elsewhere

The paint and person-box fragment filters, the axis-matching replay and the compute-host matcher run on four filter arms are in the tracked folder `scratch/court_det_fix/line_identity/` (report `results.md`, 2026-09-16). The per-view audit stays on hold; if resumed, spread the calibration frames across the video.
