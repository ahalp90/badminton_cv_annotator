# Issue 90 plan

## Goal

Test the frozen PR #88 serve rule on unseen rallies and decide whether it is
safe to port into the annotator.

## Fixed contract

- Code base: `c0b6ee66ead471164ed11c7848d6023490abeed8`.
- Held-out videos: `sset_20` and `sset_22` only.
- Save predictions before loading or joining serve labels.
- Compare the unchanged PR #82 baseline with the unchanged PR #88 rule.
- Report server side, visible start, both correct, fixes, and regressions.

## Out

- Development videos `sset_01`, `sset_15`, and `sset_21`.
- Threshold changes, extra variants, classifiers, or manual prediction edits.
- Production integration.
- PR #91 or other unrelated pipeline cleanup.

## Batches and gates

1. Build raw shuttle, pose, court, and contact evidence for both videos.
   Gate: every stage covers the canonical frame count.
2. Produce label-blind PR #82 and PR #88 predictions.
   Gate: write the prediction file and checksum before labels are read.
3. Join ShuttleSet labels and create the comparison report.
   Gate: row counts and metric denominators reconcile.
4. Run focused tests and adversarially review the final diff.

Planned commit: `Evaluate the frozen serve rule on held-out rallies`.
