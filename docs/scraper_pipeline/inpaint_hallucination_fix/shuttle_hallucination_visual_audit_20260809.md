# Stride-8 shuttle hallucination visual audit

- **Issue:** [#31](https://github.com/ahalp90/badminton_cv_annotator/issues/31)
- **Provisional review:** 2026-08-09
- **Human verification:** 2026-08-10 by Curtis
- **Fixtures:** `sset_01`, `sset_15`, and `sset_21`

## Verdict

Curtis labelled all 18 deliberately high-risk samples as visual hallucinations
with high confidence. The blind human review changed one provisional result.
For `sset_01` frames `[41057, 41064)`, Curtis identified far-court marketing
text rather than a resting shuttle.

| Sidecar inpaint relation | Samples | Visual hallucinations | Real shuttle |
|---|---:|---:|---:|
| No overlap | 9 | 9 | 0 |
| Partial overlap | 3 | 3 | 0 |
| All frames, 1–2-frame run | 3 | 3 | 0 |
| All frames, greater-than-14-frame run | 3 | 3 | 0 |
| **Total** | **18** | **18** | **0** |

This is a purposive challenge set. It measures yield among selected high-risk
cases. It does not estimate population precision, recall, or prevalence.

The result supports two separate controls. Inpaint provenance is useful for
source-aware handling, while a motion or scene ranker is still needed for
base-TrackNet errors. Nine confirmed hallucinations have no sidecar overlap.

## Review records

The human checklist is the source of truth for the final visual labels. It has
one row per reviewed interval, including the exact half-open frame range,
decision, confidence, notes, reviewer, and review date.

- [Human visual review](analysis/human_visual_review_20260810.csv.gz)

The three fixture CSVs combine the analytical fields used to choose each sample,
the provisional first pass, and the final human review. Columns prefixed with
`provisional_` preserve the initial interpretation. Columns prefixed with
`human_` record Curtis's final decision. `provisional_human_agreement` identifies
the one corrected label.

- [`sset_01` labels](analysis/sset_01_visual_hallucination_audit.csv.gz)
- [`sset_15` labels](analysis/sset_15_visual_hallucination_audit.csv.gz)
- [`sset_21` labels](analysis/sset_21_visual_hallucination_audit.csv.gz)

`predicted_analysis_label=guard_uncaught_candidate` means the exploratory
RANSAC lens marked the frame while the recurrence guard code was zero. It is a
candidate label, not a claim that the position is false.

The human label set was `hallucination`, `real_shuttle`, or `uncertain`. Curtis
used `hallucination` with high confidence for all 18 rows. The human checklist
did not expose the provisional labels.

## Evidence checks and source limit

The raw tracks, frame masks, and chunk tables were checked before sampling.
Every track length matched `raw_manifest.json.gz`. Mask shapes matched their
tracks, all sampled ranges were in bounds, and uncaught chunk coverage equalled
the uncaught-mask frame total for each fixture.

| Fixture | Manifest video | Frames | FPS | Public source used for review |
|---|---|---:|---:|---|
| `sset_01` | `videos_288p/pilot_288p.mp4` | 154,393 | 25 | [O669aZhH0LI](https://www.youtube.com/watch?v=O669aZhH0LI) |
| `sset_15` | `videos_288p/vid15_288p.mp4` | 149,487 | 25 | [yu9oyMXRGHY](https://www.youtube.com/watch?v=yu9oyMXRGHY) |
| `sset_21` | `videos_288p/sset_21_288p.mp4` | 100,349 | 30 | [gloiZ_gTJaE](https://www.youtube.com/watch?v=gloiZ_gTJaE) |

The historical 288p files were not available locally or on Bourbaki. The
review used 360p video-only streams from the same public video IDs. The first
two downloads had the manifest frame count. The `sset_21` stream had one extra
final frame and was limited to the first 100,349 frames.

The public streams started at time zero and matched the manifest FPS and
duration. `validation_overlay --verify` passed for every rendered sample. The
disputed `sset_01` sample also aligned to the same fixed object for seven
frames. Curtis identified that object as marketing text. These checks strongly
support alignment, but there is no historical video hash to prove that the
decoded frame sequence is identical to the original 288p source.

Source CSV ranges use `stop_frame_exclusive`. Overlay segment ends use the
inclusive value `stop_frame_exclusive - 1`.

## Sampling method

Six samples were selected per fixture:

1. the longest sidecar-negative uncaught chunk;
2. the highest-radial-variance sidecar-negative uncaught chunk;
3. the longest zero-radial-variance sidecar-negative uncaught chunk;
4. the highest-radial-variance uncaught chunk with partial sidecar overlap;
5. a one- or two-frame inpaint run inside uncaught material, ranked by local
   coordinate jump; and
6. a 16-frame window inside a sidecar run longer than 14 frames, ranked first
   by uncaught-frame count.

This extends the earlier nine-sample proposal so that partial, short, and long
inpaint cases are all represented in every fixture.

The provisional labels were fixed before reading
[`tracknetv3_stride8_hallucination_webui_followup_20260801.md`](tracknetv3_stride8_hallucination_webui_followup_20260801.md).
Curtis then reviewed the 18 samples through a separate checklist that omitted
the provisional decisions.

## Per-sample result

| Fixture | Sample | Frames, half-open | Inpaint | Human label | Provisional comparison |
|---|---|---:|---|---|---|
| `sset_01` | `base_longest` | 47064:47082 | none | hallucination | agreed |
| `sset_01` | `base_high_variance` | 5860:5868 | none | hallucination | agreed |
| `sset_01` | `base_zero_variance` | 41057:41064 | none | hallucination | disagreed: provisional `real_shuttle` |
| `sset_01` | `inpaint_boundary` | 4492:4500 | partial, 3/8 | hallucination | agreed |
| `sset_01` | `inpaint_short_uncaught` | 142542:142544 | all, 2/2 | hallucination | agreed |
| `sset_01` | `inpaint_long_leak` | 71673:71689 | all, 16/16 | hallucination | agreed |
| `sset_15` | `base_longest` | 64280:64300 | none | hallucination | agreed |
| `sset_15` | `base_high_variance` | 148653:148657 | none | hallucination | agreed |
| `sset_15` | `base_zero_variance` | 6454:6456 | none | hallucination | agreed |
| `sset_15` | `inpaint_boundary` | 74255:74257 | partial, 1/2 | hallucination | agreed |
| `sset_15` | `inpaint_short_uncaught` | 136281:136283 | all, 2/2 | hallucination | agreed |
| `sset_15` | `inpaint_long_leak` | 42104:42120 | all, 16/16 | hallucination | agreed |
| `sset_21` | `base_longest` | 5228:5240 | none | hallucination | agreed |
| `sset_21` | `base_high_variance` | 29416:29419 | none | hallucination | agreed |
| `sset_21` | `base_zero_variance` | 5580:5584 | none | hallucination | agreed |
| `sset_21` | `inpaint_boundary` | 89436:89448 | partial, 5/12 | hallucination | agreed |
| `sset_21` | `inpaint_short_uncaught` | 5248:5250 | all, 2/2 | hallucination | agreed |
| `sset_21` | `inpaint_long_leak` | 99929:99945 | all, 16/16 | hallucination | agreed |

The human pass agreed with 17 provisional decisions and corrected one. The
final human result is 18 hallucinations and no real-shuttle controls.

## What the existing signals do

### Recurrence guard

All 18 samples have `guard_code=0`. The current recurrence guard does not catch
these intervals. This is expected because the review set came from the
guard-clean RANSAC candidate view.

### Sidecar provenance

The sidecar identifies nine inpaint-linked samples: three partial spans, three
short full-inpaint spans, and three long full-inpaint spans. All nine are
visual hallucinations. The sidecar does not establish correctness, and it has
no coverage for nine other confirmed hallucinations.

The long-run result supports a focused test of aligned producer-window support.
This audit did not reconstruct the producer lattice, so it does not establish
that every reviewed 16-frame interval was one fully unsupported aligned call.

### RANSAC motion candidate

The RANSAC lens produced a high-yield review set. It nominated all full chunks
and 12/16, 15/16, and 12/16 frames in the three long inpaint windows. Curtis
labelled every selected span as a hallucination. The lens is useful for ranking
spans, but this purposive set has no real-shuttle controls and does not support
automatic rejection.

### Fixed or stationary coordinate rule

Curtis labelled all three zero-variance cases as hallucinations. The provisional
pass misread fixed marketing text in `sset_01` as a resting shuttle. Three
purposively selected cases are too few to validate an automatic stationary
coordinate rule, especially without real-shuttle controls.

### Scene or court context

Graphics, close-ups, coin tosses, crowd shots, and between-rally resets account
for many confirmed failures. Scene context should improve ranking. It cannot be
the only control because the `sset_15` boundary sample and `sset_21` long sample
are false tracks during full-court play.

## Comparison with the parked follow-up

The independent labels support the follow-up's most relevant conclusions:

- source provenance, visual correctness, and consumer safety are different
  questions;
- local motion residuals are useful candidate generators, not correctness
  labels;
- span or producer-window observations are more defensible than treating each
  frame as an independent example; and
- a generic fitted detector should not replace a narrow source-aware rule.

The challenge set adds direct evidence that base-TrackNet errors remain
important. Nine of 18 confirmed visual failures are sidecar-negative. This
makes targeted heatmap-shape retention worth a bounded rerun if those spans
change downstream outcomes.

This review does not test Isolation Forest, split grade 3, aligned support
features, early versus late handling, or downstream event accuracy. The parked
document's claims about those items remain proposals or code-review findings,
not results of this visual audit.

## Small checks that would settle the remaining questions

1. Reconstruct aligned 16-frame producer blocks for the three long inpaint
   samples. Record selected count, non-inpaint pass-through count, and support
   distance on both sides.
2. Rerun the nine sidecar-negative false spans with peak heatmap probability,
   second-peak ratio, contour area, and heatmap mass. Add known real-shuttle
   controls before testing whether those fields separate valid cases.
3. Compare current late masking with early `(0,0)` replacement only for a
   strongest no-support block mask. Inspect changed rallies and landings.
4. Add known real stationary shuttles to any motion-ranker challenge set.
   Report top-k yield by video and source, without a population precision claim.
5. If the exact historic 288p sources become available, rerender these ranges
   and compare every visual label. This would remove the remaining source-hash
   uncertainty.

No production detector or rejection policy is changed by this audit.
