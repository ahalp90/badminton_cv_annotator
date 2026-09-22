# Full-frame selection review

All 30 distinct final selections were inspected at full-frame scale on
22 September 2026. Cross-arm geometry verification confirms 30 geometries
among 81 selections. Opus 5 reviewed all 30 and agreed with these judgements.
It also read this preliminary review, so agreement is not blind replication.
These are development judgements, not held-out
accuracy or automatic acceptance decisions.

`Usable` means the projected court identifies the played court and follows
its visible markings well enough for further detector development. This is
a qualitative ruling, not a specified pixel tolerance. Cropped or obscured
markings cannot establish accuracy of the unseen portion. `Needs correction`
means visible misalignment; `wrong court` includes structures off the floor.

The image for each row is the corresponding run's
`gallery/<case>__<source-qualified winner with punctuation replaced by _>__full.png`.
The saved comparison supplies exact winner IDs. Floor (4,3) images were used
for common selections; the three differing (3,3) selections were also viewed.

| Case | Floors | Ruling | Visible reason |
| --- | --- | --- | --- |
| gxBQ_window_00_frame_0 | all | usable | Foreground outline, sidelines and visible service markings align; the wall proposal is not selected. |
| gxBQ_window_00_frame_5 | all | usable | The line-template child follows the foreground court, retaining the useful new source. |
| am2_window_00_frame_150 | all | usable | Foreground blue court boundaries and service markings align. |
| am2_window_01_frame_28019 | 3,3 | wrong court | Selects the neighbouring court at the right edge instead of the played court. |
| am2_window_01_frame_28019 | 4,3; 5,3 | usable | Recovers the played foreground court and its service markings. |
| am3_window_00_frame_0 | all | usable | Follows the played wooden-floor court; crossed markings from other sports remain distinct. |
| shuttleset_03_scene_0017 | all | usable | Outer boundary and service markings follow the broadcast court. |
| shuttleset_03_scene_0019 | all | usable | Outer boundary and service markings follow the broadcast court. |
| shuttleset_03_scene_0016 | all | usable | Outer boundary and service markings follow the broadcast court. |
| shuttleset_21_scene_0020 | all | usable | Outer boundary and service markings follow the broadcast court. |
| gxBQ_window_00_frame_689 | all | wrong court | Projects onto upper-left wall cladding, leaving the played court unmodelled. |
| gxBQ_window_01_frame_5111 | all | usable | Foreground court outline and visible markings align. |
| gxBQ_window_02_frame_5766 | all | usable | Foreground court outline and visible markings align. |
| gxBQ_window_03_frame_77876 | all | wrong court | Projects onto upper-left wall cladding. |
| gxBQ_window_04_frame_86088 | all | wrong court | Projects onto upper-left wall cladding. |
| yellow_short_frame_14 | 3,3 | wrong court | Projects a largely upright court onto the left wall and roof structure. |
| yellow_short_frame_14 | 4,3; 5,3 | wrong court | The replacement still projects onto wall and roof, not the yellow floor markings. |
| letterboxed_short_frame_45 | all | usable | Visible white badminton outline is consistent; the near boundary is outside the active picture. Green markings belong to an overlapping layout. |
| centre_short_frame_36 | all | usable | Follows the central played court, including the visible near baseline and centre marking. |
| am1_window_00_frame_54 | all | needs correction | Right sidelines and centre marking diverge from the yellow court; agreement on the left does not establish the whole court. |
| am3_window_01_frame_10514 | all | usable | Follows the same wooden-floor court through player occlusion. |
| am4_window_00_frame_0 | all | usable | Visible white outer court and service markings are consistent. Faded paint and floor seams limit assessment of some interior stretches. |
| am4_window_01_frame_13782 | 3,3 | wrong court | Shifted left, with a boundary across the played court and much of the projection off-frame. |
| am4_window_01_frame_13782 | 4,3; 5,3 | usable | Recovers the played white-lined court; faded interior stretches remain a visual limitation. |
| shuttleset_03_scene_0029 | all | usable | Broadcast court boundary and service markings align. Missing G1 inputs do not prevent viewing this W5 selection. |
| shuttleset_03_scene_0034 | all | usable | Broadcast court boundary and service markings align. |
| shuttleset_03_scene_0038 | all | usable | Broadcast court boundary and service markings align. |
| shuttleset_21_scene_0000 | all | usable | Broadcast court boundary and service markings align. |
| shuttleset_21_scene_0010 | all | needs correction | Projection is shifted down: the far boundary falls inside the background court and the near portion extends below frame. Composite contamination complicates the image, but the mismatch is visible. |
| shuttleset_21_scene_0039 | all | usable | Court outline and markings align despite foreground material in the composite. |

Count: (3,3) has 19/27 usable selections; (4,3) and (5,3)
each have 21/27. Both stricter floors recover the original nine views.
The six remaining failures show that a visibility floor alone does not
provide safe admission or selection on the expanded panel.

The full-frame references were additionally inspected for the letterboxed
view and Amateur-4 frame 0. They support the outer-court interpretation;
reference agreement was not used as a substitute for viewing the image.

## Existing player-support gate

Filtering the saved `(4,3)` ranking by the historical full-court gate changes
five winners. This gate requires valid geometry, camera error at most 0.1,
at least one foot within the expanded court in every sampled frame, and feet
in both court halves in at least half those frames. It uses temporal player
observations, not the spatial person masks under repair.

The replacement full-frame overlays were inspected. Their exact packet or
evaluation gallery paths are in `evaluation/ablation_winner_details.json.gz`.

| Case | Replacement | Ruling | Reason |
| --- | --- | --- | --- |
| GX689 | G1:123:71/child | usable | Follows the foreground court and its service markings. |
| GX77876 | G1:17:152 | usable | Foreground court recovered through substantial player occlusion. |
| GX86088 | G1:16:44/child | usable | Foreground court and visible service markings align. |
| yellow14 | G1:15:4392/child | wrong court | Still projects onto the wall and roof. Passing the player gate does not establish a floor plane. |
| SS21-10 | G0:30:18711/child | wrong court | The projection spans a shallow sideways strip, inconsistent with the visible broadcast court. Composite contamination remains a limitation. |

The other 22 winners are unchanged. This gives 24/27 qualitatively usable
selections, with yellow14, Am1-54 and SS21-10 still failing. No pool is empty.
This post-hoc development comparison supports retaining player evidence;
it does not validate the gate as a general acceptance rule.

Removing line-template proposals without this gate loses GX5: the replacement
`G0:181:948/child` projects onto the left wall. The corresponding Am1 replacement
`G1:0:52211/child` also projects onto the wall. Both were inspected full-frame.

## Can one proposal source be dropped?

With the same player-support gate and saved score ordering, restricting access
to G1 plus line-template proposals changes six of the 27 winners. Source
membership includes merged G0/G1 duplicates; the canonical ID alone does not
determine whether a candidate is accessible.

All six replacements were inspected full-frame. GX0, SS03-16, letterboxed45,
centre36 and SS21-0 remain qualitatively usable. SS21-10 remains wrong.
The other 21 winners are unchanged. Thus this restricted pool also gives
24/27 usable selections, with the same three failures. No pool is empty.
This supports testing a cheaper G1-plus-template design. It does not prove
equal precision, equivalent proposal coverage, or a measured runtime saving.

The converse restriction, G0 plus line-template proposals, empties GX77876
after the same gate. G0 alone also has no camera-eligible winner for Am2-28019.
G1 therefore provides necessary access on this panel under the tested rules.
