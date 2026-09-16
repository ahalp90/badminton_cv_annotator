# What the follow-up checks found

Written 2026-09-15 by the Claude Code session that ran them; the two follow-on checks were added 2026-09-16. Eight checks, seven with a replay gate that passed and one that failed its gate honestly twice. Every number below comes from the check folders named in brackets; each folder holds the script that produced it.

**Bottom line.** The detector's losses now have named locations, and none of them is where the direction experiment was looking. On GX0 the precise directions lose to a wall seam, a floor strip and a spectator's face that the merge admitted as court lines. Inside the matcher, the per-pair cap is innocent; on Amateur-3 a direction change of a fraction of a degree turns a 4 px pool into a 40 px pool before the cap. The two identifiability gates are free to apply but catch only the two false winners they were derived from; the sheared and mat-border courts pass both. Lens distortion on GX is bounded at a few pixels and is not a lever. Automatic direction selection along the GX video is unstable rather than uniformly bad, so pooling adjacent frames would inherit frame 0's fault while distant instants keep the good directions. The lever every check points at is deciding which fragments are court paint before any geometry is fitted. The two follow-on checks sharpen that: masking fragments inside person boxes on GX0 lets the unchanged selection keep the control's y-direction (bound 6.0 to 1.2 px), though through reallocation rather than through the removed face; and recording every court before the geometry and player masks shows those masks remove nothing near the control, so on Amateur-3 a direction pair that fits the control at 4.3 px yields no court under 46 px from the kept axis matchings. A further replay on 2026-09-16 (`../../line_identity/`) found the matching that reaches the fit is enumerated and passes every rule but sits about 11,500th in the axis-score order, so the per-direction cap of 512 removes it; with no cap the pair yields a 6.2 px court.

## The eight checks

**Which structures the three GX0 rows lie on** (`gx0_rows/`). Gate passed: rows 31 and 61 sit in candidate 737's support and row 51 in candidate 4104's. Row 31 is a set of vertical seams in the corrugated wall cladding above the floor. Row 51 is a tan floor strip beside the left sideline, not the painted line. Row 61 is the edge of a spectator's face in the bottom-right foreground, inside one of the person boxes the input pack already carries. The orchestrating session viewed the crops and agrees. This closes the question the direction experiment left open: the original pair won coverage-first allocation because it also covered these three structures, and no anchor or score change can repair that.

**Do later GX frames keep the precise directions** (`gx_directions/`). Gate passed: the bound over the baseline's 16 GX0 directions printed 6.022569 and the control pair 0.423931. Frames 0 and 5 keep the x-direction within 0.4 degrees but miss the y-direction by 1.8 and 2.4 degrees. Frames 5111 and 77876, at 85 seconds and 22 minutes, keep both within a degree. Three later frames fit the GX0 approved court under 3.5 working pixels, though the camera is unverified there. Selection along the video is unstable, and the frames next to 0 and 5 are the least likely to help.

**Rank table on the exported winners** (`identity_diagnostics/`). Gate passed: the WebUI's nine candidates reproduced exactly. Nineteen of twenty candidates have constraint rank 8; only the scene 19 false winner has rank 7. Only the Amateur-2 frame 28019 false winner has a sub-pixel separation between two markings. Every other unusable court, the sheared GX0 pair, both GX5 wall courts, the Amateur-2 mat-border line winner and the scene 17 paint overshoot, has rank 8 and separations from 1.8 to 11 px. The two gates are threshold-free and cost nothing, and they remove exactly two mechanisms and nothing else.

**Lens distortion on GX, merged-row endpoints** (`distortion/`). Gate failed: the broadcast control bowed as much as GX, so the method measured merge noise, not the lens. Kept as a record of what does not work.

**Lens distortion on GX, painted ridge** (`distortion_ridge/`). Gate failed on both controls for different reasons: the Amateur-3 ridge finder jumped between the paired sidelines and reflections, and one ShuttleSet edge bowed by two pixels. GX bowed outward on all four edges by 1.8 to 4.8 native pixels, the barrel signature, but with residuals as noisy as the controls. What survives is a bound: any distortion on GX moves the court lines by at most about five pixels over their spans, too small to explain the sheared winners or the direction loss. No third attempt.

**Where the close courts vanish inside the matcher** (`cap_loss/`). Both gates passed: the instrumented run reproduced every saved pair status, shortlist and winner byte for byte, and the nearest retained court matched the accounting table on all three case-arms. The per-pair cap discards closer courts than each pair's own retained set in most pairs, but only on Amateur-3 under R does any discarded court beat the pool's overall nearest, and there by 4 px of a 40 px gap. On GX0 under M and Amateur-2 frame 28019 under B the nearest proposed court is the retained one. The recorded proposals are already past the player gate, so the loss lies in the proposal step or that gate, and the two cannot yet be separated.

| Case-arm | Direction fit | Nearest proposed | Nearest retained | Cap's share of the gap |
| --- | ---: | ---: | ---: | ---: |
| GX0, M | 19.7 | 33.8 | 33.8 | none |
| Amateur-3, R | 4.3 | 40.1 | 44.2 | 4.1 px |
| Amateur-2 frame 28019, B | 3.0 | 15.3 | 15.3 | none |

Working pixels, maximum corner distance to the frozen control.

**Person-box mask replay on GX0** (`person_mask_replay/`). Gate passed: with no masking the replay reproduces the saved estimator exactly (retained IDs, working points, support masks). Dropping every fragment whose midpoint lies inside one of the frame's 18 person boxes removes 66 of 724 fragments, one of them within 6 px of a court edge. The merge and the coverage selection then run unchanged and one of the 16 selected directions coincides with the control's y-direction (0.000 degrees, against 1.795 at baseline). The incidence bound over the 16 directions falls from 6.02 to 1.22 working px and the best control fit from 6.80 to 1.50. The stricter rule, both endpoints inside a box, removes 55 fragments and changes nothing useful (bound unchanged, best fit 7.07). Both rules remove only row 61, the face; rows 31 and 51, the wall seam and the floor strip, sit outside every person box and survive. The agent that ran it inferred that the gain arrives through the greedy allocation choosing different leaders once the fragment mix changes rather than through the removed fragments' own directions (row 61's direction was never near the control axes); no single-fragment ablation tested that. It is one view. The frame carries 18 person boxes in the input pack.

**Courts before the player gate** (`pregate_loss/`). The matcher builds courts by matching observed line groups to court markings along each of the two selected directions (the axis matching), keeping the 512 best matchings per direction, combining them, and then applying a geometry mask and a player mask. This check recorded every combined court before the masks. Both gates passed: the instrumented run reproduced every saved pair status, shortlist and winner exactly, and the usable-court corners the cap-loss run recorded equal the combined courts at the usable indices in every matched pair. The masks are innocent. On both views the nearest court before the masks is the same court that survives both: 33.8 px on GX0 under the midpoint-anchor selection and 40.1 px on Amateur-3 under the precise-representative selection. Nothing within 20 px exists among 22.4 million (GX0) or 17.1 million (Amateur-3) combined courts. The direction pair behind the best direction fit is matched in both runs and its own nearest combined court is 33.8 px (GX0, pair 23, fit 19.7) and 46.1 px (Amateur-3, pair 43, fit 4.3). On those pairs the per-direction cap excluded 1,236 to 25,852 distinct matchings per direction, so the loss lies inside the axis matching, between enumeration and that cap; the axis-matching replay of 2026-09-16 places it at the cap.

| View, selection | Direction fit | Nearest combined | Nearest passing geometry | Nearest usable | Within 20 px anywhere |
| --- | ---: | ---: | ---: | ---: | ---: |
| GX0, midpoint anchor | 19.7 | 33.8 | 33.8 | 33.8 | 0 |
| Amateur-3, precise representative | 4.3 | 40.1 | 40.1 | 40.1 | 0 |

Working pixels, maximum corner distance to the frozen control.

**Continued in a tracked folder.** The axis-matching replay, the paint-profile study, the fragment-filter replay on all nine views and the compute-host matcher run on the filtered arms live in `scratch/court_det_fix/line_identity/` (`results.md` there is the report).

## What this changes

The earlier evaluation (`../CLAUDE_EVALUATION.md`) ended with six suggestions. Four are now answered and one is superseded.

- The rows check and the rank table both say the same thing: the wrong structures enter before geometry and survive every later stage. Line identity is the lever.
- The cap-loss check removes the per-pair cap from the suspect list; the pre-gate check removes the geometry and player masks. The matcher loss sits in the axis matching: which observed line groups get matched to which court markings along each direction, and the per-axis cap of 512 assignments.
- The GX direction check says the per-view route could work on GX only by choosing instants where selection is good, not by pooling neighbours. The held audit's calibration frames at 30, 60 and 90 are the wrong choice for GX if it is ever run.
- The distortion checks bound the effect and close the question for now.
- The identifiability gates go into the detector as free abstention flags, not as the fix.

## Next steps, in order of information per hour

Steps 1 and 2 of the earlier list ran on 2026-09-15 and are the two follow-on checks above.

1. **Replay the axis matching for the best-fit direction pairs locally.** Done 2026-09-16 in `scratch/court_det_fix/line_identity/` (tracked): the rule that loses the 4 px court on Amateur-3 is the per-direction cap of 512 kept matchings, ordered by the axis score; the near-fit matching survives every other rule.

2. **Extend the paint filter to floor height and colour, then replay direction selection on all nine views.** The wall seams sit above the playing floor and the tan strip is not white. The person-box mask is the first member of this filter family and already moves GX0. Replay is seconds per view; the measure is the incidence bound and best control fit per view against the E3 baseline, before any matcher run. Benefit: the first test of the line-identity lever across the population.

3. **Run the matcher on the views where step 2 moves the directions.** Roughly 20 to 40 minutes of compute-host time per case-arm. Benefit: the direction experiment showed that a closer control fit is not a closer generated court, so only this step says whether the filter reaches the winner.

4. **Keep the per-view audit on hold.** If it is resumed, choose calibration frames spread across the video rather than adjacent to the anchors, and keep Amateur-3 as the regression control.

## Limits

- One view explains the GX0 loss; the other views' merges have not been inspected the same way. The person-box result is also one view, and its gain came through reallocation.
- The direction-check bounds on frames after frame 5 assume a fixed camera that nobody has verified.
- "Combined" in the courts-before-the-gate check means the product of the 512 kept matchings per direction; the axis-matching replay in `line_identity/` separated the cap from the enumeration afterwards.
- The distortion methods are noisy; the five-pixel figure is a bound from a noisy measurement, not a calibration.
- All nine views are development data. Nothing here is a readiness claim.
