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

## Final close-out (run `w5_stage5_20260920`)

W5 closes with Arm C usable on eight of nine stress views. GX5 is the only failed view.
Its A, B and C winners all follow rear court or hall structure instead of the foreground
court. No visual call needs human escalation.

| view | A selection / ruling | B selection / ruling | C selection / ruling | C failure |
|---|---|---|---|---|
| GX0 | `G0:22:4588` / usable | `G1:22:270` / usable | `G0:22:4580/child` / usable | — |
| GX5 | `G0:181:973` / wrong_court | `G0:181:996` / wrong_court | `G0:181:948/child` / wrong_court | missing |
| Am2-150 | `G0:30:33` / usable | `G0:30:148` / usable | `G1:30:137/child` / usable | — |
| Am2-28019 | `G1:15:168` / usable | `G1:15:3024` / needs_correction | `G1:15:164/child` / usable | — |
| Am3-0 | `G1:43:296` / usable | `G1:43:299` / usable | `G1:43:299/child` / usable | — |
| SS03-17 | `G1:1:71` / usable | `G1:1:5355` / usable | `G1:1:1308/child` / usable | — |
| SS03-19 | `G0:1:60` / usable | `G0:1:74` / usable | `G1:1:533/child` / usable | — |
| SS03-16 | `G0:1:31` / usable | `G0:1:31` / usable | `G0:1:31` / usable | — |
| SS21-20 | `G1:0:2` / usable | `G1:0:1970` / usable | `G1:0:1970` / usable | — |

### Collision ruling

The collision work changed identity and provenance handling, not a scientific conclusion.
The final packet contains 4,608 source occurrences, 268 exact cross-source duplicate
groups, 4,340 canonical parents and 3,667 valid children. Every duplicate group contains
one G0 and one G1 occurrence with matching W5 gates.

GX0, Am3 and GX5 keep the same common-candidate evidence and selections as the earlier
packets. Am2-150 removes one exact duplicate parent and its child. Its shared-candidate
evidence and selected A/B/C geometries stay unchanged. Two diagnostic ranks move because
the duplicate entries above them disappeared. The collision repair also lets the five
previously stopped views complete. These changes do not reverse any W5 finding.

### Remaining blocker and next branch

The remaining blocker is **proposal**. GX5 has no usable foreground court in the bounded
population, including the saved material from before global retention. Ranking cannot
select geometry that was never proposed. Admission and refit are therefore downstream of
the failure. The earlier GX5 source audit in
[`automatic_axes_results.md`](../../../experiments/annotator/independent_court/recorded/player_guided/projective_patterns/automatic_axes_results.md)
found that the closest camera-eligible candidate was still 89.94 working pixels from the
control. It also found that the useful observed directions were lost during proposal.

Take **Branch P** next. Add the existing independent 2D line/template proposer as one
bounded source. Use cached lines and broad families, then feed its candidates through the
unchanged W5 evidence pass and R1+R2 judge. Run the same nine views as regression cases.
Only revisit line extraction if the cached fragments cannot represent the required paint.

Do not run another W5 recompute before that proposal change. Keep the unused-scene check
(D1) and guarded reuse/shadow integration (D2) for later; neither answers the current
proposal failure.

## History

- 2026-09-20: pilot run `w5_stage2_20260920` ruled; R0 fix, R1 and R2 issued for seance 2.
- 2026-09-20: final run `w5_stage5_20260920` ruled; collision work accepted; Branch P issued.
