# A note for whoever picks this up

The new work is worth keeping. It helps explain where the detector loses useful courts and where it chooses badly among courts it already has. It does not yet show a new court that we know looks right in the image.

I would leave three things clear in the project notes.

**L1 helped one difficult example but hurt a good one.** The trial kept the same number of line assignments as before. A much closer Am2 court survived, but Am3 lost its close court. I would not extend this exact rule to a larger run. That does not rule out other ways of keeping a wider range of assignments.

**L2 found a useful scoring change.** On Am2 frame 150, using filtered line observations to score the original courts chose one at 12.44 px from the reference instead of 140.90 px. That court was already available; the change helped choose it. Changing both the proposed courts and their scoring was not better everywhere, so I would not make the whole combination the default.

**The suggested seven-frame calculation is already done.** Combining line and paint ranks chooses `22:4579` instead of the badly wrong `106:93818`. But `22:4579` was already judged skewed in an earlier image review. The result deserves to be kept without calling it a newly correct court.

This is the wording I would put beside that result:

> Combining line and paint ranks across seven GX frames picked an existing court instead of a wildly wrong alternative. The chosen court was previously judged skewed, so this is a better choice among the saved options, not a newly correct detection.

The C2 correction also stands: the actual duplicate kept is 29665, not 30886. The W2 image review gives a reason not to reject a court just because its markings share pixels or lack long connected runs; it does not show that every scoring approach has failed.

There is nothing further to launch for this handover. The rank calculation has been run, and no further experiments are needed on this standalone 1D path. The script is here for reproducibility only.

The [main report](W4_FINAL_REVIEW.md) has the numbers and limits. The [source notes](EVIDENCE_INDEX.md) point to the exact files at the reviewed commit. No new image judgement was made in W4, and no repository changes were made.
