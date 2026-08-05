# sset_01 replay and serve behaviour measurement

Generated: 2026-08-05T05:15:49.885858+00:00

Fixture profile: `une-189c5af-static-stride8` from source commit `189c5af58e45d23ae827dde516924194eb238e18`.

## Replay mask

The fresh current union differs from the pinned replacement mask on 0 frames. The raw union flags 91,521 of 142,237 scored frames. Duration filtering leaves 91,521 flagged frames, with precision 0.981 and recall 0.972. The e2e court-invalid union flags 91,521 scored frames.

Court absence contributes 91,521 flagged frames. Perspective shift contributes 0, and velocity drop contributes 0.

The e2e mask covers 518 of 37,011 GT-rally extent frames.

## Slow motion

The unchanged velocity signal uses a rally-speed median of 0.01388889 and threshold 0.00208333. It flags 0 frames.

## Replay duplicate margin

Supported for follow-up: 61 short replay intervals have a human-adjudicated immediately preceding live source. The long replay montage `[147049, 148312)` is excluded by human adjudication. 113 GT rallies provide different-rally negatives.

Retrieval margin unmeasured: The review establishes interval-level source relations but does not annotate exact live-source frame pairs. A retrieval margin remains follow-up work.

## Serve lookback

The current mask-policy candidate records 2 true positives, 7 false positives, and 61 false negatives across 63 target serve misses. Its precision is 0.222 and recall is 0.032.

The selected trigger frames contain 7 `live`, 0 `live-non-standard`, 0 `replay`, 2 `cutaway`, and 0 `other` labels. The evidence-only and current mask-policy counts are the same.

These results describe one labelled video. They do not authorise a production change.
