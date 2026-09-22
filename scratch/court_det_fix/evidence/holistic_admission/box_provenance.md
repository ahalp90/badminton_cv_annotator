# Which results survive the person-box correction

The box error affects spatial masks, not every experiment that used players.
W5 and the paint-only G0/G1 comparison remain valid. The frozen marking-
junction replay and broadcast junction replay have been repaired. The
five-case person-observation matcher comparison is now complete; see the
[matcher comparison](box_repair/evaluation_20260922/runs/person_observations_repair_20260922/matcher/comparison.md).

The three frozen packs contain 47 cases: 17 with boxes measured on the same
image, 10 with nearby-frame boxes, and 20 median-composite images with
source-frame boxes. Some historical consumers used all boxes as spatially
valid. A box from another image can hide real paint or leave a player unmasked;
the direction of the error is not predictable.

## Current interpretation, by experiment

| Evidence | What can be used now |
| --- | --- |
| W5 stages and line-template regression | No mismatched box reached produced W5 measurements. Their final comparisons remain valid |
| G0 baseline and G1 paint-observation generation | Box-independent generation; the complete 27-case crossed comparison is filed separately. Frozen-reference diagnostics do not establish accuracy |
| Direction agreement, cap-loss and axis replay | Their temporal-foot gates do not use bbox_px as a spatial mask. Their geometry findings remain valid |
| Frozen marking-junction replay | The corrected 20-case, 682-candidate packet retains all three scheme winners. Each scheme has 9/20 accurate picks under that replay's reporting rule. Acceptance was not evaluated |
| Broadcast junction-first ranking | Both repaired arms score 1/18, versus the invalid old 3/18. SS03-38, SS03-16 and SS21-29 change winners. The broad conclusion remains that junction-first ranking performs poorly here |
| Line-identity person arms | The old GX5 and four broadcast arms used invalid spatial masks. In the corrected five-case comparison, SS03-16, SS03-19 and SS21-20 retain close useful courts; SS03-17 has a close line winner but a poor paint winner; GX5 remains unrescued by mask correction |
| Marking-refit direct_occlusion and bidirectional_occlusion | Five of twenty original cases are affected. These masked-arm figures are not repaired by the separate junction replay. The unmasked results remain valid |
| Appearance-contradiction probe and renewed-pool junction studies | Distinct populations and reruns; the frozen 682-candidate repair does not automatically validate their historical aggregate totals |
| Saved cross-frame masked summaries and embedded paint-refit occlusion diagnostics | Keep them labelled unsafe. The published unmasked cross-frame table and headline paint winners remain valid; rerunning old diagnostics is conditional on decision value |

The corrected marking schemes are stripe_exclusive, contradictions_first and
complete_agreements_first. Their data is
[marking_junction_repair_v6.json.gz](box_repair/marking_junction_repair_v6.json.gz).
It replaces five player records using the
[six-image exact-detection packet](box_repair/exact_people_20260921.json.gz).
The repaired broadcast record is [result.json.gz](box_repair/result.json.gz).
These are development studies, with different candidate populations and
different denominators; do not combine their counts into detector accuracy.

## Historical qualification and follow-up

The corrected repair kept chosen directions fixed and changed the fragments
seen by axis matching. GX5 used same-image boxes; the four broadcast cases
used the supported composite mask. Corrected inputs remain in
[box_repair/person_observations/](box_repair/person_observations/), including
the v3 five-case manifest and case/estimator pairs. The completed result is
the [matcher comparison](box_repair/evaluation_20260922/runs/person_observations_repair_20260922/matcher/comparison.md).

The historical direction-changing person and `paint_person` variants still
have qualified totals. Valid same-image regressions do not support adopting
that global design, and the corrected masks do not rescue GX5.

## Provenance and recovery

The source audit was w5_holistic/box_provenance_impact.md. Its earlier repair
queue predates the completed marking-junction repair; the corrected packet
and the later campaign state establish that update. The original audit stays
recoverable in the consolidation backup.

The marking repair, exact detections and v3 person-observation inputs are
retained here byte-for-byte. The campaign originals and superseded repairs
are sealed in `.recovery/w5-preparation-20260922.tar.gz` at the investigation
root. Source manifests retain the original producing paths.
