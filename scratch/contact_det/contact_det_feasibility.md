# Contact detection feasibility

## Bottom line

The current pipeline does not find every contact. This remains true when every noisy raw proposal is kept.

At the useful 10-frame tolerance, the raw proposals find 83.8% of non-serve contacts and 66.1% of serves. At 15 frames they find 86.7% and 74.0%. A learned model that only searches around existing proposals therefore has a hard ceiling unless the proposal regions are widened or another way of making regions is added.

The simple ankle-height player rule gives almost the same result on these fixtures. Direct player-side attribution is already about 89% accurate on matched final contacts. The bad rally-level result comes mainly from fitting a strict Top/Bottom alternation after contacts have been missed.

For the learned contact scorer, start with scikit-learn histogram gradient boosting over the numeric evidence already available. Run a random forest on the same rows as a cheap control. If temporal learning is still needed, add a contact output to BST-X. Build X3D-S after that when the remaining failures show a real need for RGB.

## Contact coverage

The three fixtures contain 292 rallies and 3,128 labelled contacts: 292 serves and 2,836 later contacts.

| Proposal list | Allowed error | Serves found | Later contacts found | Unmatched proposals |
| --- | ---: | ---: | ---: | ---: |
| Raw and noisy | 5 | 144 / 292 (49.3%) | 2,206 / 2,836 (77.8%) | 3,820 / 6,170 |
| Raw and noisy | 10 | 193 / 292 (66.1%) | 2,377 / 2,836 (83.8%) | 3,600 / 6,170 |
| Raw and noisy | 15 | 216 / 292 (74.0%) | 2,460 / 2,836 (86.7%) | 3,494 / 6,170 |
| Current final | 5 | 132 / 292 (45.2%) | 2,093 / 2,836 (73.8%) | 1,481 / 3,706 |
| Current final | 10 | 178 / 292 (61.0%) | 2,303 / 2,836 (81.2%) | 1,225 / 3,706 |
| Current final | 15 | 208 / 292 (71.2%) | 2,355 / 2,836 (83.0%) | 1,143 / 3,706 |

The tolerance is scaled from a 30 fps base. Ten means eight frames in the 25 fps fixtures and ten in the 30 fps fixture.

The prediction save contains no ground-truth (GT) fields. It was made twice from the saved current `ad8da4f` artefacts, with identical bytes. The scorer verified the checksum before opening GT. An independent recalculation reproduced every count in the table.

PR88 does not change the conclusion. On its historical 239-rally subset, its accepted candidates find 82.7% of later contacts and 69.9% of serves at 10 frames. PR88 itself chooses the correct visible start in 132 of 239 rallies and gets both that start and the server side right in 117. Those are different tests and should not be combined.

## Player side

The current code does use geometry, but not blindly. It first picks the tracked player whose wrist is nearest the shuttle. It then uses the bottom of that player's box and the calibrated net band to call Top or Bottom. Finally, it fits one alternating Top/Bottom phase across the rally.

The ankle test keeps the nearest-wrist player choice and changes only the side rule:

```text
two visible players: smaller mean ankle y is Top
one visible player: ankle y above the net-band midpoint is Top, otherwise Bottom
```

At the 10-frame tolerance, the current rule gets 2,207 of 2,480 available matched contacts right. The ankle rule gets 2,208 of 2,481 right. Both round to 89.0%.

The rally-level final-hitter result is only about 49% for both rules. Server-side accuracy is 64.9% for the current rule and 64.2% for the ankle rule. Missed contacts alter the odd/even alternation, so a mostly correct direct side label can still produce a poor rally phase.

There is also little work for a scene-cut missing-player fallback in these fixtures. All 167 final candidates within 15 base-30 frames of a scene start already have a player-side answer. A lookback might help wrong answers near cuts, but it is not repairing missing attribution here.

## Learned scorer

Use the same search method for all three learned options:

1. Start from raw heuristic candidates.
2. Add cheap seeds for current misses, such as relaxed impulse peaks, wrist-motion peaks, shuttle gaps and suspected serves near scene starts.
3. Expand and merge the seeds into short search regions.
4. Score each possible centre frame with a tight temporal window.
5. Keep local score peaks above a held-out threshold.
6. When nearby peaks describe the same contact, keep the strongest score.

Measure region recall before model quality. A model cannot recover a contact outside every region.

The first model should be histogram gradient boosting because the repository already provides shuttle impulses, wrist distances, player positions, visibility, scene context and rule outcomes. It can show quickly whether those inputs are enough. A random forest is a useful control and costs almost nothing to add.

BST-X is the next neural option. It is about 1.84 million parameters in the current configuration and already accepts pose, shuttle and court position. It still needs consecutive contact-centred windows, visibility flags and a real contact output. Replacing the present 14-class clip head alone is not enough.

X3D-S adds RGB and may help when racket motion or broadcast context matters. It is also the largest data job: frame-exact decoding, court crops, crop validity and RGB alignment do not yet exist for this task. The standard model is still small at 3.79 million parameters, but cheap inference does not remove the data risk.

Use one contact score and a separate Top/Bottom score. A low contact score means no contact. A weak or missing player-side score means no side answer.

## Reports

- [Contact recall and player geometry](contact_recall_and_player_geometry.md) has the full measurement and PR88 details.
- [Binary contact detector options](binary_contact_detector_options.md) has the tree, BST-X and X3D-S comparison and the smallest useful experiment.
