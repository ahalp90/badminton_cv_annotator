# Current findings

## Population and anchors

- 292 GT rallies exist end to end.
- 249 meet the current covered definition.
- 239 have a one-to-one predicted span and form the main set.
- At ±10, the earliest accepted contact is nearest to 119 serves, 19 first returns and 4 later strokes. Another 97 match no GT stroke. Five windows contain more than one GT stroke.
- Among the 97 unmatched anchors, later contacts recover the serve in 49 rallies and the first return in 36. Nine first match another later stroke, and three never match a GT stroke.

The ordinary contact detector often finds a useful contact sequence, but the earliest accepted candidate is not a reliable serve anchor.

## Motion evidence

- Only 24/239 primary rallies have a usable recurrence-mask path before the earliest contact.
- The uniquely labelled serve/return set contains 135 anchors: 118 serves and 17 first returns.
- Nineteen of those 135 have usable recurrence-mask paths.
- The fixed 0.05-BH rule correctly calls 9/17 first returns incoming and incorrectly calls 4/118 serves incoming.
- Adding the producer inpaint mask leaves 10/135 usable labelled paths. It correctly calls 7/17 returns and makes no false return calls, but loses useful paths.

Residual scatter, trend-to-jitter and path length do not separate the errors well enough to justify another threshold. The sample is small and the diagnostic groups overlap.

## Server attribution

On the 239 main rallies:

| Method | Correct |
|---|---:|
| Released alternating fit | 124 |
| Earliest-contact player | 152 |
| Direct fixed motion correction | 163 |
| Like-for-like prepend/refit | 159 |

The direct and prepend/refit methods use the same earliest-contact fallback and the same 15 motion triggers. Direct inference is correct in 13 of those 15 rallies; prepend/refit is correct in 9. The four losses are two later-contact overrides and two alternating-fit ties.

The earlier old-fit-fallback prepend/refit result is 127 correct with 217 known answers. Of the apparent 36-point gap from the direct result, 32 come from the different fallback and four from the triggered refit. It is not a fair headline comparison.

## Limits and next inference

The corrected analysis supports motion as a useful local correction. It does not support motion as a full server detector because paths are rarely usable. The most promising next work is better anchor selection, followed by better motion-path availability. The unchanged 0.05-BH rule then needs testing on new videos.
