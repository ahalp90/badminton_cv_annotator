# Findings and decisions

The branch can generate useful courts on difficult development views. It has
not yet produced a reliable, fast automatic selector. CourtKeyNet repair is
retired as an approach; removing CourtKeyNet remains a branch-completion task.

The nine-view stress panel and expanded 27-case corpus are development data,
not representative held-out evaluations. A nearby reference fit, a generated
candidate, an automatic winner and a visually usable court are different
outcomes. Keep those distinctions when interpreting every result below.

| ID | Finding and decision | Evidence |
| --- | --- | --- |
| D01 | The old chain passed 0/11 labelled amateur frames. Broadcast repair lessons survive, but CourtKeyNet is excluded from the replacement design. | [Retirement](evidence/retirement/README.md) |
| D02 | Better line evidence and useful candidate pools did not solve court identity. Preserve earlier inputs, proposal records and contrary examples; do not revive failed acceptance rules. | [Independent proposals](evidence/independent_proposals/README.md) |
| D03 | Midpoint and representative changes each caused regressions. Fixed-membership SVD is diagnostic, not a tested matcher improvement. Graph/SVD search remains conditional. | [Direction search](evidence/direction_search/README.md) |
| D04 | Score ordering and the 512-assignment cap discard court-compatible matches on traced pairs. A broader fixed-budget rule helped one case and badly regressed another. Screen all pairs before another expensive trial. | [Exact traces and stopped probe](evidence/direction_search/README.md#cap-and-duplicate-corrections) |
| D05 | G1 changes both proposal access and scoring observations. The four-case comparison does not assess the expanded 27-case corpus. Keep all unique populations and assess those effects separately. | [G0/G1](evidence/g0_g1/README.md) |
| D06 | Stage-5 W5 selected usable courts on 8/9 views. Adding line-template proposals recovered GX5 but introduced two wrong winners, giving 7/9. Test admission before the 256-candidate limit. | [W5](evidence/holistic_admission/README.md) |
| D07 | The person-box error affects particular spatial-mask arms. W5 and paint-only results survive. The five-case corrected person-observation matcher comparison remains unfinished. | [Provenance and repairs](evidence/holistic_admission/box_provenance.md) |
| D08 | The temporal rank sum is complete and selected a previously skewed court. Independent versus shared scoring on the same verified candidate union is still unmeasured. | [Pixel and temporal evidence](evidence/pixel_temporal/README.md) |

## What remains before a detector can ship

Interpret the current W5 result before choosing another detector change.
Locate remaining failures in proposal coverage, admission, ranking or cost.
Preserve the separate G0/G1 assessment and person-mask comparison rather than
treating W5 as their answer.

Representative labelled evaluation, ordinary-hardware latency with player
detections supplied, camera-change handling and annotator integration remain
separate requirements. Development search time on Carmack is not deployment
latency. Remove CourtKeyNet and accidental model/weight-loading dependencies
before the branch finishes.
