# Reviewed failure clip

The user reviewed `53_0061.mp4` on 17 October 2026 and attributed the failure
to an unusual camera view at the rally midpoint used for the court fit.

The clip covers ShuttleSet22 video 53, rally `set2:25` (span 61). The saved
index records a `ValueError` during `pose_filling` in all four MonoTrack runs
using its own courts. That is the recorded failure stage; the unusual court-fit
view is the cause identified by the user's visual review.

This review does not change the predictions, scores or failure classifications.
It is benchmark interpretation evidence, outside the original-ShuttleSet
validation used to investigate player attribution.
