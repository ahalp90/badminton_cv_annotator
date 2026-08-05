# sset_01 non-play behaviour measurement

Generated: 2026-08-05T04:38:02.948125+00:00

Fixture profile: `une-189c5af-static-stride8` from source commit `189c5af58e45d23ae827dde516924194eb238e18`.

## Replay mask

The fresh current union differs from the pinned replacement mask on 0 frames. The raw union flags 91,521 of 142,237 scored frames. Duration filtering leaves 91,521 flagged frames, with precision 0.981 and recall 0.972. The e2e court-invalid union flags 91,521 scored frames.

Court absence contributes 91,521 flagged frames. Perspective shift contributes 0, and velocity drop contributes 0.

The e2e mask covers 518 of 37,011 GT-rally extent frames.

## Slow motion

The unchanged velocity signal uses a rally-speed median of 0.01388889 and threshold 0.00208333. It flags 0 frames.

## Replay duplicate margin

Killed: The reviewed timeline labels replay footage but does not independently pair any replay interval with its earlier live source interval. A retrieval margin needs that positive pair and a different-rally negative, so this study is killed without building a detector.

## Serve lookback

The current mask-policy candidate records 2 true positives, 7 false positives, and 61 false negatives across 63 target serve misses. Its precision is 0.222 and recall is 0.032.

The selected trigger frames contain 7 `live`, 0 `live-non-standard`, 0 `replay`, 2 `cutaway`, and 0 `other` labels. The evidence-only and current mask-policy counts are the same.

These results describe one labelled video. They do not authorise a production change.
