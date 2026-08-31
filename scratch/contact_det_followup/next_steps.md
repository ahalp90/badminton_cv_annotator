# Useful next work

Two questions remain:

- Can the annotator safely accept a small number of rallies from an unfamiliar broadcast style?
- What better evidence would help it recover missed first contacts?

## First priority: test selective auto-annotation across broadcast styles

The present keep-or-review model cannot find an almost-perfect subset. Its inputs may be too weak. The model has also not been tested across camera and broadcast styles.

The next experiment should hold out whole broadcast families. A family might share a tournament, camera layout, graphics package, frame rate, or another visible production convention. The groups should follow the available video metadata and content. File names alone are not enough.

The result should answer four plain questions:

- How many rallies did the annotator accept?
- How many accepted rallies were fully correct?
- Did the same threshold work for every held-out broadcast family?
- What caused the accepted mistakes?

The main plot should be precision against coverage. Show one line for each held-out family and one pooled line. Report the number of accepted rallies beside every high-precision point; a percentage based on three rallies is not useful evidence.

### A conversational brief for an agent

> Please find out whether the current annotator can safely auto-accept a small number of rallies from an unfamiliar broadcast style.
>
> Start with the rally-wide side vote enabled. Group the labelled videos by a real broadcast or camera convention. Hold out one whole group at a time. Train the accept-or-review model on the other groups.
>
> For each held-out group, show precision, coverage, and the raw number of accepted rallies. Pay special attention to the high-precision end of the curve. If no setting gets close to the target, tell us which errors still pass the filter. Explain what new evidence might separate them. Please do not tune a threshold on the group being scored.

A near-100% claim needs enough accepted rallies to support it. When the sample is small, report an uncertainty interval or give a clear count-based warning.

## Second priority: improve first-contact evidence upstream

The label-guided check found 300 complete A–D rally repairs after a start edit and the side vote. The safe model found only 24 in the pooled comparison. The stricter group-held-out estimate found seven. The V check found six. The search space contains useful answers. The chooser lacks the evidence to identify them.

The next first-contact study should focus on the inputs, not another threshold sweep. Useful questions include:

- Does the section begin after the true first contact?
- Did the candidate generator include the true first contact?
- When the candidate exists, which shuttle, pose, or scene evidence distinguishes it from the nearby false candidates?
- Are the failures concentrated in a few broadcast or camera conventions?

The study should separate those cases before training anything. A model cannot recover a contact that never enters its candidate list. Section-edge failures also need a different fix from candidate-ranking failures.

### A conversational brief for an agent

> Please trace the missed first contacts before building another chooser.
>
> For each failed rally, work out whether the labelled first contact falls outside the detected section, is absent from the candidate list, or is present but ranked badly. Summarise those three groups by video and broadcast style. Then inspect a small, representative sample from each group.
>
> If one group dominates, test the smallest change that addresses that cause. Keep the A–D and V boundary intact. Report complete-rally repairs and breaks rather than candidate accuracy alone.

## Work that can stay parked

The following ideas do not need another pass with the same saved inputs:

- close opposite-side duplicate removal, because no qualifying pairs were present
- the learned deletion model, because it broke more correct rallies than it repaired
- the current keep-or-review model, because precision remained low even at tiny coverage
- a global 0.85 cut-off for the ±5 measure, because it caused 126 breaks for 139 repairs

The 0.85 cut-off becomes worth one new test only if ±10 is chosen as the release tolerance. That test needs fresh raw scores on a held-out set.
