# Paired reference comparison of the saved net settings

**Recommendation:** retain weight **0.04** and post-base overrun **4 working pixels** as the provisional setting used by the accepted selected-court gallery. The saved reference data show gains from a positive net weight, but do not establish a practically meaningful winner between 0.02 and 0.04. Weight 0.08 adds two measured degradations above 1 working pixel. Overrun 2, 4 and 8 select the same candidate on every saved frame.

This is a retrospective comparison of automatically selected candidates. It does not select the reference-best candidate. The separate stripe-polarity correction follows candidate selection and is not included in these setting scores. The user's assessment that the corrected gallery courts are usable is useful product feedback, but it is not a set of independent per-frame accuracy labels.

## Data and measure

The primary set has 71 unique frames: the seeded Am1 frame-54 pool replaces the original pool for that same image. The results file also reports the original 71-frame set and seeded Am1 separately. Development has 16 primary frames; withheld has 55. Here, withheld includes additional frames from development video sources and six frames from Centre/Am4, so frames are not independent source replicates.

The independent references cover 45 primary frames: 27 have visible painted-court landmarks and 18 broadcast frames have four reference corners. Each number below is the median or maximum Euclidean error among a frame's available reference points, in pixels on the saved 960-wide working image. The headline combines the two reference types; their separate results follow. The original reference packs and the same homography/corner conventions as `scan_saved.py::retrospective` were used. Ranking scores were never treated as quality labels. Two broadcast views marked `view_unverified` and 24 control frames without references are excluded from geometric error summaries.

| Net weight, overrun 4 px | Median frame error | Q75 | Q90 | Median frame maximum | Measured choices changed vs zero | Paired change >1 px: better / worse |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3.40 | 5.01 | 7.02 | 6.92 | 0 | — |
| 0.02 | 2.75 | 4.24 | 6.20 | 5.99 | 10/45 | 8 / 0 |
| **0.04** | **2.75** | **3.95** | **5.62** | **5.99** | **13/45** | **9 / 0** |
| 0.08 | 2.76 | 4.24 | 5.62 | 6.09 | 15/45 | 9 / 2 |

The median paired per-frame change versus zero is **0 px for every positive weight**, since most selected candidates stay the same. At the stricter 2 px threshold, all three positive weights improve four measured frames and worsen none. At 4 px, each improves one and worsens none. This threshold sensitivity makes the small weight differences less persuasive than the shared positive-weight effect.

The 0.02 and 0.04 arms choose different candidates on four primary frames; three have references. The 0.04 choices improve those three median errors by 0.30–1.56 px. The 0.04 and 0.08 arms differ on three frames; two have references. The 0.08 choices worsen those two median errors by 1.29 and 1.98 px. The other two inter-weight differences are unlabelled control frames. At weight 0.04, overrun 2, 4 and 8 have **identical selections on all 71 frames**; this dataset cannot distinguish those tolerances.

## Coverage and source sensitivity

For the 27 visible-landmark frames, the median frame error is 3.03 px at zero, 2.81 at 0.02, 2.81 at 0.04 and 3.01 at 0.08. For the 18 four-corner frames, the corresponding values are 3.99, 2.68, 2.68 and 2.68 px. Development's 13 measured frames have medians 4.02, 2.66, 2.66 and 2.75 px. The 32 measured withheld frames have medians 3.23, 2.79, 2.79 and 2.79 px. These split summaries remain descriptive because some frames share a video source.

Broadcast frames contribute 18/45 measured rows. Giving each of the ten referenced video sources equal weight, mean paired median-error changes versus zero are −0.73, −0.77 and −0.65 px for weights 0.02, 0.04 and 0.08. Excluding the seeded Am1 video, which has one large recovery, the nine-source means are −0.25, −0.29 and −0.16 px. A paired source-cluster bootstrap with 10,000 resamples and seed 20260923 gives a 95% percentile interval of **−0.093 to 0.000 px** for 0.04 minus 0.02, and **0.000 to 0.315 px** for 0.08 minus 0.04. These intervals are exploratory: ten video sources and only five measured inter-weight choice differences provide little independent evidence. Neither interval establishes a material advantage at the 2 px threshold.

The seeded Am1 pool alone improves from 11.29 px at zero to 1.20 px with any positive weight. In the original Am1 pool the same image's selected court has 130.68 px median error and remains unchanged across settings; the pool replacement is therefore material. The worst primary errors, Am1 frame 5352 at **402.18 px** and Yellow frame 14 at **361.31 px**, are also unchanged across settings. A 180° landmark relabelling does not resolve them. They remain in all summaries. Their size makes absolute mean error a poor discriminator here, while their paired contribution is zero.

The seven primary frames with no selected court abstain in every setting. Of eight labelled non-court controls, one is accepted in every setting. The net weight changes ranking among existing full-court candidates; it does not change acceptance here.

## Cases worth visual judgement

These are candidate differences where the references are absent or incomplete, or where median and worst-point errors disagree. They are review priorities, not inferred correctness labels.

| Frame | Reason |
|---|---|
| `sset_21_gloiZ_gTJaE_frame_00000001` | 0.04 changes the choice from 0.02; no reference. |
| `sset_21_gloiZ_gTJaE_frame_00081233` | 0.08 changes the choice from 0.04; no reference. |
| `shuttleset_21_scene_0039` | Positive weights change the choice from zero; reference view is unverified. |
| `gxBQ_window_00_frame_689` | 0.04 lowers median landmark error by 0.30 px but raises the maximum by 4.13 px. |
| `am2_window_00_frame_150` | 0.08 raises median landmark error by 1.98 px and maximum by 3.06 px versus 0.04. |

The reproducible calculations and per-frame results are in [paired_reference_analysis.py](paired_reference_analysis.py) and [paired_reference_results.json.gz](paired_reference_results.json.gz).
