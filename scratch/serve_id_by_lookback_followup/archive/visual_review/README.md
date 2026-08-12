# Accepted-contact diagnostic overlay sample

Twenty clips selected deterministically from the fixed 239 one-to-one rows. Selection uses only accepted-contact order, binary outgoing verdicts, three-way pre verdicts, and player attribution. The GT stroke frames were not used and are not included here.

## Viewing order

- `bottom_first_outgoing_false`: `01_bottom_first_outgoing_false_sset_01_set1_r8_f15809.mp4`, `02_bottom_first_outgoing_false_sset_15_set1_r2_f24369.mp4`, `03_bottom_first_outgoing_false_sset_21_set1_r5_f15721.mp4`, `04_bottom_first_outgoing_false_sset_01_set1_r12_f19587.mp4`, `05_bottom_first_outgoing_false_sset_15_set1_r4_f25826.mp4`
- `top_first_outgoing_false`: `06_top_first_outgoing_false_sset_01_set1_r1_f11567.mp4`, `07_top_first_outgoing_false_sset_15_set1_r3_f25134.mp4`, `08_top_first_outgoing_false_sset_21_set1_r2_f13568.mp4`, `09_top_first_outgoing_false_sset_01_set1_r2_f11894.mp4`, `10_top_first_outgoing_false_sset_21_set1_r3_f14553.mp4`
- `bottom_selected_pre_unavailable`: `11_bottom_selected_pre_unavailable_sset_01_set1_r14_f23073.mp4`, `12_bottom_selected_pre_unavailable_sset_15_set1_r1_f23665.mp4`, `13_bottom_selected_pre_unavailable_sset_21_set1_r6_f16509.mp4`, `14_bottom_selected_pre_unavailable_sset_01_set1_r16_f25015.mp4`, `15_bottom_selected_pre_unavailable_sset_15_set1_r12_f32865.mp4`
- `top_selected_pre_unavailable`: `16_top_selected_pre_unavailable_sset_01_set1_r3_f12358.mp4`, `17_top_selected_pre_unavailable_sset_15_set1_r6_f27587.mp4`, `18_top_selected_pre_unavailable_sset_21_set1_r9_f19024.mp4`, `19_top_selected_pre_unavailable_sset_01_set1_r6_f14977.mp4`, `20_top_selected_pre_unavailable_sset_15_set1_r9_f29818.mp4`

The deterministic selector cycles fixtures in `sset_01`, `sset_15`, `sset_21` order, then uses set/rally/contact order. It excludes any rally already chosen by an earlier stratum, so no clip shares a rally or focal contact.

## Legend

- Cyan thick tick and circle: focal accepted contact. Yellow thin ticks and circles: other accepted impulses in the clip.
- Magenta line and circle: recurrence-clean usable TrackNet trail and current point. The line never crosses an unusable frame.
- Dark grey X: raw TrackNet reported a point, but recurrence-clean rejected it. These points never join the magenta trail or enter a calculation.
- Blue-orange rectangle and cross: focal player's sticky picked bbox and ankle anchor, when available.
- Bottom strip: light grey means recurrence-clean usable. Dark grey means unusable. The cyan vertical line is the focal frame.
- Right panel: local pre/post runs use half-open source-frame bounds. It shows the inpaint-guard code/name and whether the current frame enters the recurrence-clean calculation. `pre=unavailable` means the pre-contact path failed a common eligibility check. It does not mean `not_incoming`.

The overlay deliberately does not apply the separate `producer_inpaint` sidecar. The fixed primary calculation uses the PR #82 `recurrence_clean` mask, including its inpaint guard, rather than the `producer_original` path.

Each 1920x1080 clip has approximately 60 base-30fps frames on either side of the focal contact, scaled to the fixture FPS and clipped to source-video bounds. The standalone clips use a 32 px reference HUD and the same libx264 fast/CRF 18 base encoding as the comparison panels.

## How the fixed checks work

A frame is recurrence-clean usable only when TrackNet marked it visible with finite, non-zero coordinates; the court is present; the focal player has a finite positive sticky bbox height and finite shuttle-to-player distance; and the recomputed recurrence guard is clear. The matching local path must then have at least five frames, reach the contact within the scaled two-frame gap, and have a largest-step ratio no greater than 4.0. A three-way pre result is `unavailable` when that shared path eligibility fails. `not_incoming` is a different result and is never used as a synonym for unavailable.

## Per-video guide

### `01_bottom_first_outgoing_false_sset_01_set1_r8_f15809.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `bottom_first_outgoing_false`.

The failed fixed check is: `no run`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `02_bottom_first_outgoing_false_sset_15_set1_r2_f24369.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `bottom_first_outgoing_false`.

The failed fixed check is: `local gap 14 over scaled 2`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `03_bottom_first_outgoing_false_sset_21_set1_r5_f15721.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `bottom_first_outgoing_false`.

The failed fixed check is: `largest-step ratio 28.10 over 4.0`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `04_bottom_first_outgoing_false_sset_01_set1_r12_f19587.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `bottom_first_outgoing_false`.

The failed fixed check is: `largest-step ratio 6.54 over 4.0`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `05_bottom_first_outgoing_false_sset_15_set1_r4_f25826.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `bottom_first_outgoing_false`.

The failed fixed check is: `local gap 13 over scaled 2`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `06_top_first_outgoing_false_sset_01_set1_r1_f11567.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `top_first_outgoing_false`.

The failed fixed check is: `local gap 17 over scaled 2`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `07_top_first_outgoing_false_sset_15_set1_r3_f25134.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `top_first_outgoing_false`.

The failed fixed check is: `largest-step ratio 6.83 over 4.0`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `08_top_first_outgoing_false_sset_21_set1_r2_f13568.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `top_first_outgoing_false`.

The failed fixed check is: `no run`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `09_top_first_outgoing_false_sset_01_set1_r2_f11894.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `top_first_outgoing_false`.

The failed fixed check is: `largest-step ratio 4.45 over 4.0`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `10_top_first_outgoing_false_sset_21_set1_r3_f14553.mp4`

It is the first accepted contact for the named player whose fixed binary outgoing check is false. It occupies its deterministic fixture-round-robin position within `top_first_outgoing_false`.

The failed fixed check is: `local gap 6 over scaled 2`.

Inspect whether a real outgoing shot was wrongly skipped, then compare the post-contact run and recurrence-clean strip with the stated failure.

### `11_bottom_selected_pre_unavailable_sset_01_set1_r14_f23073.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `bottom_selected_pre_unavailable`.

The failed fixed check is: `fewer than 5 frames (2)`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.
### `12_bottom_selected_pre_unavailable_sset_15_set1_r1_f23665.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `bottom_selected_pre_unavailable`.

The failed fixed check is: `no run`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.

### `13_bottom_selected_pre_unavailable_sset_21_set1_r6_f16509.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `bottom_selected_pre_unavailable`.

The failed fixed check is: `largest-step ratio 4.29 over 4.0`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.

### `14_bottom_selected_pre_unavailable_sset_01_set1_r16_f25015.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `bottom_selected_pre_unavailable`.

The failed fixed check is: `largest-step ratio 7.41 over 4.0`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.

### `15_bottom_selected_pre_unavailable_sset_15_set1_r12_f32865.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `bottom_selected_pre_unavailable`.

The failed fixed check is: `fewer than 5 frames (2)`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.

### `16_top_selected_pre_unavailable_sset_01_set1_r3_f12358.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `top_selected_pre_unavailable`.

The failed fixed check is: `largest-step ratio 4.25 over 4.0`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.

### `17_top_selected_pre_unavailable_sset_15_set1_r6_f27587.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `top_selected_pre_unavailable`.

The failed fixed check is: `fewer than 5 frames (4)`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.

### `18_top_selected_pre_unavailable_sset_21_set1_r9_f19024.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `top_selected_pre_unavailable`.

The failed fixed check is: `largest-step ratio 4.19 over 4.0`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.

### `19_top_selected_pre_unavailable_sset_01_set1_r6_f14977.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `top_selected_pre_unavailable`.

The failed fixed check is: `fewer than 5 frames (1)`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.

### `20_top_selected_pre_unavailable_sset_15_set1_r9_f29818.mp4`

It is the selected accepted contact for the named player whose fixed three-way pre verdict is unavailable. It occupies its deterministic fixture-round-robin position within `top_selected_pre_unavailable`.

The failed fixed check is: `largest-step ratio 73.54 over 4.0`.

Inspect whether incoming or not-incoming looks visually obvious despite the stated metric ineligibility, then compare the pre-contact run and recurrence-clean strip.
