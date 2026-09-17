# L3 cached temporal opportunity

**Bottom line.** The expanded seven-frame pilot found discriminating cross-frame evidence in the
existing line scorer, but it does not support adopting temporal reuse. A stable sampled
view is available, while wrong automatic courts dominate the reference-blind selection.
The pilot asks whether different observations of the same automatic courts supply useful
different evidence, and whether one shared score can choose a court.

## Contract

GX is the cached single-video sample label. The run used `gxBQ_HwdgN4.mp4` at frames 0, 5, 689, 5111, 5766,
77876 and 86088. These span 0 to 1437.6 seconds. Every image is 1920x1080 and was
scored at 960x540. Frames 0 and 5 were the two origins. The user-authorised
representation extension used all seven complete packs. The panel contains 30 distinct
courts: 14 from origin 0 and 16 from origin 5, formed from each origin's top-eight line
and top-eight paint ranks. The frozen and local input-pack MD5 (Message-Digest 5) checksums match. No new
candidate detector or neural extraction ran. All 210 scores are valid, with no
common-subset exclusions.

The saved-origin diagonal replay agrees to below 6e-17. Each cross-origin score used the
target frame's image, fragments and feet. The full line and separately stored paint
vectors are in [score_matrix.json](score_matrix.json) and [score_matrix.csv](score_matrix.csv).

## Sampled-view check

One diagnostic used scale-invariant feature transform (SIFT) keypoints, a brute-force
descriptor matcher and random-sample consensus (RANSAC) homography fitting. It
registered each target to frame 0. The six fitted pairs have
0.874–0.981 inlier ratios and 11/12 or 12/12 covered grid cells. The identity-grid median
displacement is 0.21–5.10 native pixels. Held-out residual medians are 0.10–0.41 pixels;
large p90 values reflect moving people and overlaid captions. The [registration record](view_alignment.json)
and [representative alignment overlay](alignment_0_to_77876.png) show static court lines
and the building aligned, with moving content ghosted. This
supports the same pixel homography at these sampled instants. It is not a continuous guard.

## Selection result

The per-frame line winners were 22:4579 at frames 0 and 5111, 22:4580 at 5, 106:93784
at 689 and 86088, 106:93818 at 5766, and 181:950 at 77876. The shared median winner
was 106:93818, with line vector
`[0.301055, 0.303811, 0.305444, 0.302754, 0.318919, 0.256558, 0.250365]` and
paint vector `[0.25, 0.25, 0.25, 0.25, 0.25, 0.125, 0.125]`. Its post-lock annotated-corner
error median is 799.36 working pixels. The low-error frame-0 alternatives 22:4579,
22:4580 and the existing paint winner 22:4588 have medians of 14.07, 9.56 and 9.56.
The existing frame-5 line and paint winners 181:29836 and 10:1274 have medians of
529.13 and 528.30. These reference joins were made only after selection was locked.

The formal line dominance check leaves eight non-dominated courts, so line-score
aggregation does not isolate the good geometry. This is therefore classified as **discriminating
cross-frame evidence seen, with wrong alternatives still dominating this matrix**. It is
not a wiring-only result, a missing-view result, or evidence that the whole panel lacks
useful geometry. The result makes no acceptance or uninterrupted-reuse claim.

## W3 packet and next computation

See the compact W3 (practical-route review) [handoff](W3_PACKET.md). The one next computation is a
pre-registered rank-sum ablation using only the existing line and paint scores: rank each
cue within each frame, sum the two ranks, aggregate across all seven frames, and evaluate
the single shared winner after locking. If the low-error 22:* courts beat the 106/181
alternatives without per-frame hand choice, improve the existing discriminator. If not,
prioritise automatic candidate generation. A continuous reuse guard, new extraction and
the full audit remain deferred.

Files: [source_manifest.json](source_manifest.json), [view_alignment.json](view_alignment.json),
[run_l3_temporal.py](run_l3_temporal.py), and [alignment overlays](alignment_0_to_77876.png).
