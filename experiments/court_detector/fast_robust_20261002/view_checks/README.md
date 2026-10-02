# View grouping and the player check

Keep the hash threshold at **0.30** for now. This investigation found a clear
failure in court selection after grouping, but no evidence that tightening the
hash threshold would fix it. The proposed player-scoring change needs a separate
evaluation; these results describe the existing detector. The follow-up asked
whether the failures came from grouping different camera views or from rejecting
accurate courts after grouping.

## Would a tighter hash threshold help?

The study used 988 scene middle frames from all 86 videos and compared 5,344
within-video pairs. It included four members spread through each dominant
group's timeline, difficult saved alignments, other group references and scenes
with no detected court. This is a deliberately selected diagnostic sample.
Its pair counts are not estimates of error rates across all video frames.

The detector's own hash and alignment functions were rerun on Carmack. All 540
sampled member-to-reference alignments reproduced the saved correlation and
corner shift exactly. Of 528 sampled comparisons against references created before the scene
joined its actual group, 521 exceeded the hash threshold. The other seven
failed alignment again.

For an additional comparison, each frame was reduced to the detector's 960×540
greyscale alignment image. Correlation was measured without warping either
image. Values at least 0.85 were labelled high similarity; values below 0.60 were
labelled low similarity. The 568 intermediate pairs remain unclear. These
cut-offs are exploratory proxies for camera view, **not manually verified view
labels**. Both hash distance and correlation depend on image appearance.

| Hash threshold | High similarity, of 1,841 | Unclear, of 568 | Low similarity, of 2,935 | Existing matches excluded, of 340 |
|---|---:|---:|---:|---:|
| 0.10 | 494 | 0 | 0 | 306 (90.0%) |
| 0.15 | 1,010 | 6 | 0 | 267 (78.5%) |
| 0.20 | 1,284 | 35 | 0 | 220 (64.7%) |
| 0.25 | 1,578 | 56 | 0 | 84 (24.7%) |
| 0.30 | 1,763 | 79 | 0 | 0 (0.0%) |
| 0.35 | 1,829 | 126 | 62 | 0 (0.0%) |

The first three count admitted pairs. The last column uses four time-spread members from each of 85 dominant groups.
The remaining video had no other dominant-group member. All 340 pairs already
passed the current hash and alignment checks, so retaining every pair at 0.30
is expected. Excluding a pair could split a group or assign the scene elsewhere;
these counts do not predict the final number of groups.

At 0.30, 1,842 pairs passed the hash: 1,763 had high similarity and 79 were
unclear. Four source images were inspected for this follow-up. The most suspect
admitted alignment was the same camera during a pre-match light show. One
reference frame also showed a plausible source of hash variation: the score
graphic was absent. This does not establish that every admitted pair has the
same camera view.

The evidence supports leaving 0.30 alone while fixing court selection. It does
not establish an optimal threshold or justify loosening it. No changed threshold
or masked hash was run through the extraction pipeline.

## Why ShuttleSet 30 rejected the accurate courts

The scene spans frames 26,011–26,145 at 30 fps. It has no usable labelled rally
frames. Its player check samples frames 26,033–26,123 every third frame: 31
samples over three seconds. All samples survived the scene-consistency check.

The correct court contained one player in all 31 samples and somebody in both
halves in none. The code requires at least one person in every sample and
somebody in each half in at least half the samples. The correct court therefore
failed during this ordinary break in play. The user inspected the whole scene
and confirmed that the other player was preparing at the sideline.

The scene's oversized court passed because it included line judges behind the
real far baseline. Their boxes passed the standing-person filter. The player
check accepts feet up to 15% of the court dimensions beyond its borders,
including 2.01 m beyond either baseline. It does not identify the two competitors.
An oversized court can therefore receive stronger apparent player support.

| Court checked in this scene | Samples with somebody inside | Samples with somebody in each half | Player check |
|---|---:|---:|---|
| Static reference court | 31/31 | 0/31 | Reject |
| Accurate group-reference court, aligned to this scene | 31/31 | 0/31 | Reject |
| Scene's own oversized court | 31/31 | 31/31 | Accept |
| Wrong court eventually shared across the group | 31/31 | 31/31 | Accept |

Replaying the player check with all 123 individual candidates agreed with the
saved first-rejection outcome at this scene: 120 failed here, including all 119
accurate candidates. The group-reference court used the exact saved alignment;
other individual candidates used identity transforms. Their saved alignment
shifts were below one pixel, but this remains an approximation rather than a
full replay of every candidate's checks.

The failure is the use of one member's player check to discard a court for the
whole group. A bonus for containing more people could preserve the same bias
towards oversized courts. These observations justify changing the sharing rule;
they do not determine a suitable player-score weight.

![ShuttleSet 30, frame 26,078: the original individual court fit](sset30_scene0094.png)

This is the scene's middle frame with its original individual court fit, before
sharing. The red dashes outline the predicted paint borders. Its near baseline
lies around y=2,063, below the 1,080-pixel image. The overlay uses the quality-95
review JPEG and was compressed with pngquant and oxipng. The
[4.5-second clip](../veto_scene.mp4) shows the full scene. The other 134 review
frames remain local scratch files.

## Saved measurements

- [Sample](sample.csv.gz): scene IDs, frame numbers and reasons for selection.
- [Hashes](hashes.csv.gz): packed image hashes for the 988 selected frames.
- [Pair table](pair_table.csv.gz): hash distances, saved and rerun alignments,
  greyscale correlations and original group membership.
- [Threshold counts](threshold_effects.csv.gz) and [hash bands](hash_band_table.csv.gz).
  Their inherited `same_view` and `different_view` column names mean the
  high- and low-similarity proxies above, not human labels.
- [Player replay](feet_summary.json.gz): frame indices, retained feet, courts
  and alignment values; [fractions](feet_court_fractions.csv.gz),
  [candidate outcomes](feet_candidates.csv.gz),
  [detections](feet_detections.csv.gz) and [court projections](feet_projections.csv.gz).
- [Sharing-fix pilot](sharing_pilot_group.json.gz): the 123-member ShuttleSet 30
  group rerun with `84bbba1e`, using the original individual fits. All source
  scores reproduced exactly. The [summary](sharing_pilot_summary.json.gz) and
  [comparison](sharing_pilot_comparison.csv.gz) give errors at 1280 × 720.

The saved pair table is sufficient to recalculate the threshold table: count
`hash_distance <= threshold` separately for `frame_correlation >= 0.85` and
`frame_correlation < 0.60`. For existing matches, select
`pair_kind == 'member_vs_own_reference'` and `later_reason` starting with
`time_quantile`, then count `hash_distance > threshold`.

The player fractions can be recalculated using
[`players.player_fractions`](../../../../src/court_detector/players.py) with the
`all_feet_px` array in the replay JSON. Convert each `courts_native_px` entry
to a court-metres-to-image homography with `cv2.getPerspectiveTransform`, using
`court_detector.geometry.CORNER_COURT_M` as the source points. Replace null feet
with NaN. The source detector
commit and extraction locations are in [reproduce.md](../reproduce.md).

## Should player presence become a score bonus?

Defer this change until after the submission. The current player rule affects
which candidates enter the search, as well as which fitted courts survive.
Replacing it with a bonus would change more than the final ranking. Candidates
rejected early have no saved scores, so the current extracts cannot show how the
proposed weighting would behave.

ShuttleSet 30 also gives a concrete warning: the correct court contains one
player, while an oversized court contains people in both halves. A larger
player bonus could favour the wrong court. Missing player support can mean a
break in play, a missed detection or incorrect geometry; the count alone does
not distinguish those cases.

First separate group selection from per-scene transfer checks. Any later player-weight experiment
should measure both corrected errors and newly damaged good fits, using separate
videos for choosing the weight and assessing it. Include non-rally scenes when
checking false detections. No player-weight experiment was run here.

## Would ranking by player tiers rescue scene 0094?

No. Repeating this scene's original search reproduced its saved choice and
corners exactly. The search found an accurate court, but the player check
removed it before selection. This was an individual-search failure as well as
a later sharing failure.

Consider three tiers: the current player rule first, one person in at least
half the samples second, and all other candidates third. Court score chooses
the winner within each tier. Applied to the saved candidates, this still
selects the oversized court:

| Candidate before final refit | Tier | Combined court score | Samples with somebody in each half |
|---|---:|---:|---:|
| Accurate line-template candidate | 2 | 0.844 | 0/31 |
| Oversized candidate selected originally | 1 | 0.589 | 31/31 |

These are selection-stage scores. The selected oversized court's final refit
raises its score to about 0.597. Among 1,091 camera-eligible candidates, 855
pass the current player rule. None of those is within 20 native-image pixels
of the accurate group-reference court at every corner. The best tier-2 court
is within 7.3 pixels before the final refit.

As a diagnostic, selecting without the player gate chooses the accurate
candidate. After its final refit, its worst corner is 9.9 pixels from the
dataset homography at native 1920 × 1080 resolution. This single-scene result
does not establish that removing the gate would improve other scenes.

Strict tiers preserve today's accepted choices when the candidate list and
final-refit rule stay fixed. They can add fallback courts when the first tier
is empty, but cannot repair this scene's choice. Changing candidate generation
or admitting fallback courts into sharing can also affect other scenes.
Giving court score priority over player tier would be a separate rule to
evaluate. No production player rule was changed.

The [search summary](scene0094_search.json.gz) records the exact replay and
diagnostic refits. The [candidate table](scene0094_candidates.csv.gz) contains
the measured player fractions, scores and corner errors. Its reference errors
use the group's original reference court; ground-truth errors use the dataset
homography. Both use native-image pixels.
