# Line identity: where the court detector loses the court, and whether removing non-paint fragments helps

Draft. Numbers are filled from `runs/` as each stage completes; see `evidence.md` for gates and `worklog.md` for the log.

## What the work tried to learn

The automatic court detector loses courts it could have found. The direction experiment showed that on several views the selected directions can support a court within a few working pixels of the frozen control, yet the matcher's pool holds nothing within 20 to 45 px. This work asks two questions. First: inside the matcher, which step loses the court? Second: if the loss comes from line groups that are not court paint, does removing fragments that are not white paint, or that lie inside a person box, before any geometry is fitted recover it?

## Terms

- Direction selection: the automatic choice of 16 vanishing directions from merged line groups (`vp_pruning.estimate`, coverage rule). The baseline selection (B) is the comparator throughout; the direction experiment's midpoint-anchor selection (M) and precise-representative selection (R) appear only where their records were reused.
- Direction fit: the least-squares court fitted to the control from one ordered pair of selected directions with both vanishing points fixed. It says what the directions could support; it is not a generated court.
- Axis matching: the matcher's step from one direction pair to courts. Along each direction it enumerates a scale and offset from every pair of direction-compatible observed line groups against every pair of court markings, keeps enumerations with at least three supported markings that the necessary player rule accepts, drops duplicates by marking assignment, and keeps the 512 best by score (the per-direction cap). Every kept horizontal matching combined with every kept vertical matching is a combined court.
- Masks and caps after the axis matching: the geometry mask, the player mask, the per-pair cap of 256 courts, and the global caps of the two rescoring stages.
- Control: the frozen reference court per view (six visually approved, three manual references). Distances are maximum corner distance in working pixels (960 by 540) with the 180-degree relabelling allowed.
