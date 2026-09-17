# L2 — candidate access versus fragment rescoring

This fixed-survivor replay uses the existing 960×540 working-pixel convention and the existing 180-degree corner relabelling. G0 is the saved baseline generation population; G1 is the saved `paint_observations` generation population. The union is ordered G0 first, then G1, with full origin keys preserved. S0 and S1 use the unchanged `stripe_observations` weighting and score functions.

## Winner changes

| case | G0,S0 line / paint | G0,S1 line / paint | G1,S0 line / paint | G1,S1 line / paint |
| --- | --- | --- | --- | --- |
| GX0 | G0:22:4579 (16.5px) / G0:22:4588 (7.7px) | G0:22:4594 (16.2px) / G0:22:4588 (7.7px) | G1:22:137 (10.1px) / G1:143:158 (21.1px) | G1:143:158 (21.1px) / G1:143:158 (21.1px) |
| Am2-150 | G0:30:30 (140.9px) / G0:30:33 (12.4px) | G0:30:32 (12.4px) / G0:30:33 (12.4px) | G1:30:0 (12.4px) / G1:32:45 (8.3px) | G1:30:0 (12.4px) / G1:32:45 (8.3px) |
| Am2-28019 | none / none | none / none | G1:15:164 (9.0px) / G1:15:164 (9.0px) | G1:15:168 (11.3px) / G1:15:168 (11.3px) |
| Am3-0 | G0:43:22603 (5.8px) / G0:43:22627 (8.5px) | G0:43:22605 (4.3px) / G0:43:22627 (8.5px) | G1:43:153 (11.8px) / G1:43:296 (10.1px) | G1:43:296 (10.1px) / G1:43:296 (10.1px) |

Candidate access changes the line winner on GX0, Am2-150, Am2-28019, Am3-0 and the paint winner on GX0, Am2-150, Am2-28019, Am3-0 when S0 is held fixed. Rescoring alone changes the line winner on GX0, Am2-150, Am3-0 and the paint winner on none when G0 is held fixed.
On the fixed union, S0→S1 changes the line winner on GX0, Am2-150, Am2-28019, Am3-0 and the paint winner on Am2-28019. This isolates ranker-input effects from access to G1 candidates, but it is still a development-population comparison rather than held-out evaluation.

Nearest available courts are recorded in `comparison.csv` for every cell. They are independent of the score choice and include both the full generation population and the camera-eligible subset. Controls measure the products only; they never select a winner.

## Diagonal checks and unresolved points

- GX0: saved G0/S0 and G1/S1 winner identities passed.
- Am2-150: saved G0/S0 and G1/S1 winner identities passed.
- Am2-28019: saved G0/S0 and G1/S1 winner identities passed.
- Am3-0: saved G0/S0 and G1/S1 winner identities passed.

The four pixel traces reproduce each saved aggregate profile and retain every passing centre shift, both-side contrasts, tested coordinates and availability mask. The overlays are diagnostic witnesses, not new visual approvals. The existing rulings remain: Am2-28019 `184:4123` and SS03-19 `165:6702` are false paint winners; Am2-150 `30:33` and SS03-19 `1:60` are the approved/usable contrary examples. The profile evidence shows what the scorer accepted, but does not by itself establish physical court-line ownership, especially for the far-Am2 alias.
