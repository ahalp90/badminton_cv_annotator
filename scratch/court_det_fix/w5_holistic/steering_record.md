# W5 steering record

Change record for the holistic court-detector pilot. Each entry says what changed, which
observation motivated it, and what happened to the known controls. Numbers are rank-1
corner error against the approved supplied control, in native pixels, read only after the
automatic rankings were saved.

## Pilot outcome (run `w5_stage2_20260920`, initial provisional reduction)

Visual rulings are in `runs/w5_stage2_20260920/visual_rulings.json`.

| view | A | B | C | failure class |
|---|---|---|---|---|
| GX0 | usable | wrong_court | usable | B misranked, camera clue |
| Am2-150 | usable | needs_correction | needs_correction | misranked, shifted alias; paint spread clue |
| Am2-28019 | usable | wrong_court | wrong_court | misranked, camera clue |
| Am3-0 | usable | wrong_court | wrong_court | misranked, camera clue |
| SS03-19 | usable | usable | usable | none |

Every view had a usable court in the bounded parent population, so admission is not the
problem. All four bad rows are ranking failures. Best usable parent per view: GX0 22:4588
(15 px), Am2-150 30:137 (13 px), Am2-28019 15:0 (15 px), Am3-0 43:22605 (8.5 px), SS03-19
1:1278 (3 px).

What the false winners share:

- camera error 0.33 to 2.9, where every usable court sits at or below 0.064;
- large parts of the court outside the frame, so a few tiny spans (65 to 92 px) carry the
  paint readout while whole markings at zero support hide behind the per-marking mean.

Am2-150 is different. Its B winner has camera error 0.043 and is the right court family,
shifted sideways. Camera plausibility cannot separate it; the paint evidence must.

Refit: children improved every good parent on GX0, Am2-150, Am2-28019 and SS03-19. On
Am3-0 children damaged 14 of 49 parents within 25 px and improved 4. Refit is kept for
diagnosis and is not revised in this round.

## Audit rulings on seance 1

A contract audit of the seance-1 code found one bug and several hygiene gaps.

- R0, bug: `verifier.raw_junctions` passed the court-space arm direction to the image-space
  support and photometry tests. On the perspective views (all pilot views except SS03-19)
  the raw arm support, arm ridge contrast and contradiction records did not measure the
  arms. Junction evidence feeds no ranking, so the A/B/C selections stand. Fix: derive the
  direction from the projected samples, as `junction_observations.measure` does.
- Replay: `reconstruct_generation_entries` passed native corners to `select_pool`, where
  L2 divides by the native/working scale first. Replaying both ways gave identical 256-ID
  orders on the three replayed pilot views. Conform to L2 anyway.
- `sync.sh` pushed the frozen G0 records and baseline generation onto the host with
  `rsync --delete`. Frozen host inputs are compared, never overwritten.
- Arm A ranks the G0 union half by its S0 stripe scores and the G1 half by S1 scores,
  where L2's union cells rescored both under one observation set. Ruling: keep. The
  contract asks A to preserve each source's filtered-observation legacy replay, and the
  mix only touches the stripe tie-break behind `profile.score`.
- `evidence_sparse` fallbacks from contract section 13 were not implemented. Not triggered
  in the pilot; implement.
- `has_same_image_boxes` decides by case-name prefix. Derive it from the record.

## Rule revisions for seance 2

### R1: camera plausibility as eligibility

Change: a parent or child is eligible for the automatic ranking only when its saved
`camera_error` is present and at most 0.1. Ineligible candidates keep their full evidence
record and stay in the diagnostic lists. When no hard-valid candidate passes, the view
returns `no_plausible_camera` with no selection and the ungated order saved.

Motivation: the camera-clue failure class above. Sensitivity from the saved pilot evidence,
rank-1 control error under the pilot reduction gated at each limit, parents then
parents+children:

| limit | GX0 | Am2-150 | Am2-28019 | Am3-0 | SS03-19 |
|---|---|---|---|---|---|
| 0.03 | 33 / 11 | 25 / 18 | 37 / 18 | 24 / 19 | 4 / 3 |
| 0.05 | 33 / 11 | 17 / 18 | 37 / 18 | 24 / 19 | 4 / 3 |
| 0.1 | 33 / 11 | 17 / 39 | 37 / 18 | 24 / 19 | 4 / 3 |
| 0.2 | 33 / 11 | 17 / 39 | 37 / 18 | 24 / 19 | 4 / 3 |
| 0.3 | 33 / 11 | 17 / 39 | 1923 / 18 | 466 / 502 | 4 / 3 |
| 0.5 | 33 / 11 | 17 / 39 | 511636 / 6562 | 579 / 579 | 4 / 3 |
| none | 838 / 11 | 17 / 39 | 511636 / 6562 | 579 / 579 | 4 / 3 |

The winners are flat from 0.03 to 0.2 and collapse between 0.3 and 0.5. The historical
value 0.1 sits in the flat region and was not tuned on these views.

Tension with the W5 operating assumption: this turns an inherited cutoff back into a hard
eligibility test. It is kept hard, with an explicit status, because a soft penalty would
need a new scale parameter with no better provenance. The negative control SS03-19
165:6702 has camera error 0.069 and passes R1, so the paint reduction, not the gate, must
keep it below the positive control 1:60. Watch that row.

### R2: span-weighted directional means

Change: each direction's mean of the per-marking paint-backed readout `q_paint10` (and of
`q_geom` for the fallback) is weighted by that marking's projected visible span in pixels.
Markings with no readout or zero span drop out of their direction. `Q` stays the minimum
over the two directions and still needs both directions non-empty.

Motivation: the plain mean lets a 65 px stub count as much as a 900 px sideline, so a court
that is mostly out of frame can be carried by a few stubs. Am2-150 shows the same problem
inside the plausible-camera set: the shifted alias wins on stubs while the true court's
long markings are better explained.

Effect from the saved evidence, rank-1 control error, parents then parents+children:

| reduction | GX0 | Am2-150 | Am2-28019 | Am3-0 | SS03-19 |
|---|---|---|---|---|---|
| pilot rule | 838 / 11 | 17 / 39 | 511636 / 6562 | 579 / 579 | 4 / 3 |
| R1 only | 33 / 11 | 17 / 39 | 37 / 18 | 24 / 19 | 4 / 3 |
| R1 + R2 | 33 / 11 | 17 / 8 | 37 / 18 | 19 / 19 | 4 / 3 |
| R2 without R1 | 720 / 11 | 17 / 8 | 320 / 326 | 19 / 19 | 4 / 3 |

R2 alone does not replace R1: GX0 and Am2-28019 still pick implausible cameras. R1 alone
leaves the Am2-150 child ranking wrong. Together they answer both named failures. Children
are ranked with parents throughout; the C column is the operative one.

Expected rulings after R1 + R2, to be confirmed visually: GX0 usable, Am2-150 usable,
Am2-28019 usable or needs_correction (18 px native on a scale-2 view), Am3-0 usable or
needs_correction (19 px), SS03-19 usable.

### Sensitivity to run with R2

The paint-backed readout uses the historical contrast probe of 10. Seance 2 reranks under
R1 + R2 with the probe at 5, 10, 15 and 20 from the saved raw ridge arrays and reports the
rank-1 candidate per view. The probe stays at 10 unless that table shows a reversal.

## History

- 2026-09-20: pilot run `w5_stage2_20260920` ruled; R0 fix, R1 and R2 issued for seance 2.
