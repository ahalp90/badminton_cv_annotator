# Why a good court was lost during sharing

The main failure came after scenes had been grouped by camera view. A player
check in one scene rejected an accurate court for the entire group. Tightening
the earlier image-similarity check did not address that failure.

## The player check favoured an oversized court

In ShuttleSet 30 scene 0094, one player stood inside the real court while the
other prepared at the sideline. The detector required people in both halves
of a candidate court in at least half the sampled frames. The real court
therefore failed the check.

An oversized court passed because it also enclosed line judges behind the
far baseline. Their boxes passed the standing-person filter. The check allowed
feet up to 15% beyond the court dimensions and did not identify the two
competitors. Counting more people could therefore favour the wrong court.

The accurate court had somebody inside in all 31 samples, but somebody in each
half in none. The oversized court appeared to contain people in both halves
in all 31. Of 123 saved courts checked in this scene, 120 failed, including all
119 accurate ones. Checks for candidates other than the original group court
approximated their sub-pixel alignment shifts as zero.

![Original oversized court in ShuttleSet 30 scene 0094](sset30_scene0094.png)

The predicted near baseline lies below the image. The
[4.5-second source clip](../veto_scene.mp4) shows the break in play.

This led to the sharing fix: a court rejected by one receiving scene remains
available to the rest of the group. The [results report](../../report.md)
covers its effect across all 86 videos.

## A stricter image-similarity check was not the answer

The grouping investigation compared 5,344 pairs drawn from 988 scene frames.
At the existing hash-distance limit of 0.30, none of the admitted pairs fell
into the low-similarity category used for this check. Lowering the limit to
0.25 would exclude 84 of 340 existing matches; lowering it to 0.20 would
exclude 220. There was no evidence that either change would fix the court
selection failure.

The similarity categories came from greyscale image correlation, not manually
labelled camera views. They were a way to screen this selected sample, rather
than a measured grouping error rate. The threshold stayed at 0.30.

## Giving player support priority also kept the wrong court

A replay of scene 0094 found an accurate candidate with score 0.844. The
oversized candidate scored 0.589 but passed the player check, so a rule that
always preferred passing candidates still selected it.

Choosing by score instead recovered the accurate court, with a worst-corner
difference of 9.9 pixels from the official corners at native 1920 × 1080
resolution. That single-scene improvement motivated the
[eight-video selection trial](../../search_policy_trial/selection_results/README.md).
The larger trial found mixed individual changes and no final main-camera gain
after sharing.

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

The threshold table counts
`hash_distance <= threshold` separately for `frame_correlation >= 0.85` and
`frame_correlation < 0.60`. For existing matches, it selects
`pair_kind == 'member_vs_own_reference'` and `later_reason` starting with
`time_quantile`, then counts `hash_distance > threshold`.


[scene0094_search.json.gz](scene0094_search.json.gz) contains the single-scene
replay and adjusted fits. [scene0094_candidates.csv.gz](scene0094_candidates.csv.gz)
contains candidate scores, player counts and corner differences. These two
files use native-image pixels; the group comparison uses 1280 × 720 pixels.
