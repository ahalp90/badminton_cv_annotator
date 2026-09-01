# Useful next work

Three questions remain:

- Do the contacts that match only at ±10 still refer to the intended hits?
- Can the annotator safely accept a small number of rallies from an unfamiliar broadcast style?
- What better evidence would help it recover missed first contacts?

## First: check what ±10 adds

±10 frames is close enough for this project. Compare the ShuttleSet22 matches at ±5 and ±10. Inspect the contacts that match only at ±10, especially when two labelled hits are close together. Check the video and neighbouring labels to make sure each prediction still refers to the intended hit.

If those matches are sound, use ±10 as the main timing score and report ±5 alongside it. The annotator currently keeps proposed contacts with an HGB score of at least 0.90. On the 40 ShuttleSet development videos, lowering that cut-off to 0.85 repaired 167 contact lists and damaged 142 at ±10. Regenerate the raw ShuttleSet22 scores and test 0.85 once there.

That development result checks contact count and timing only. Most newly admitted frames do not have saved player-side guesses. The held-out test therefore needs to report fully correct rallies as well as contact timing.

## Second: test selective auto-annotation across broadcast styles

The present keep-or-review model cannot find an almost-perfect subset. Its inputs may be too weak. The model has also not been tested across camera and broadcast styles.

Hold out one whole broadcast family at a time. Build the families from the available video metadata and what appears on screen. A family might share a tournament, camera layout, graphics package, frame rate, or another visible production convention. Do not group videos from their file names alone.

Answer four plain questions:

- How many rallies did the annotator accept?
- How many accepted rallies were fully correct?
- Did the same threshold work for every held-out broadcast family?
- What caused the accepted mistakes?

Plot precision against coverage. Show one line for each held-out family and one pooled line. Report the number of accepted rallies beside every high-precision point. A percentage based on three rallies tells us very little.

### A conversational brief for an agent

> Please find out whether the current annotator can safely auto-accept a small number of rallies from an unfamiliar broadcast style.
>
> Start with the rally-wide side vote enabled. Group the labelled videos by a real broadcast or camera convention. Hold out one whole group at a time. Train the accept-or-review model on the other groups.
>
> For each held-out group, show precision, coverage, and the raw number of accepted rallies. Pay special attention to the high-precision end of the curve. If no setting gets close to the target, tell us which errors still pass the filter. Explain what new evidence might separate them. Please do not tune a threshold on the group being scored.

Always give the number of accepted rallies behind a near-100% result. When that number is small, include an uncertainty interval or say plainly that the result is fragile.

## Third: improve first-contact evidence upstream

On the 32 ShuttleSet training videos, at least one allowed start edit produced a complete correct rally in 300 sections after the side vote. The model and cut-off chosen from those same 32-video results repaired only 24 sections. When each group was excluded from that choice before it was scored, the selected models repaired seven. The fixed model repaired six sections on the eight ShuttleSet validation videos. The saved candidates contain useful answers. The chooser lacks the evidence to identify them.

Before training another first-contact model, find out where the useful information is being lost:

- Does the section begin after the true first contact?
- Did the candidate generator include the true first contact?
- When the candidate exists, which shuttle, pose, or scene evidence distinguishes it from the nearby false candidates?
- Are the failures concentrated in a few broadcast or camera conventions?

Separate those cases before training anything. A model cannot recover a contact that never enters its candidate list. Section-edge failures also need a different fix from candidate-ranking failures.

### A conversational brief for an agent

> Please trace the missed first contacts before building another chooser.
>
> For each failed rally, work out whether the labelled first contact falls outside the detected section, is absent from the candidate list, or is present but ranked badly. Summarise those three groups by video and broadcast style. Then inspect a small, representative sample from each group.
>
> If one group dominates, test the smallest change that addresses that cause. Keep the 32 ShuttleSet training videos separate from the eight ShuttleSet validation videos. Report complete-rally repairs and breaks rather than candidate accuracy alone.

## Work that can stay parked

The following ideas do not need another pass with the same saved inputs:

- close opposite-side duplicate removal, because no qualifying pairs were present
- the learned deletion model, because it broke more correct rallies than it repaired
- the current keep-or-review model, because precision remained low even at tiny coverage
- a global 0.85 cut-off for the ±5 measure, because it caused 126 breaks for 139 repairs

Do not rerun the 0.85 cut-off on the same saved inputs. If the ±10 check passes, regenerate the raw held-out scores and test 0.85 once.
