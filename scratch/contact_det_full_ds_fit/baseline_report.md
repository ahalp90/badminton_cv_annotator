# Full-data contact baseline

## Bottom line

The best of the nine fixed runs is the reference HGB model using the original
motion values, balanced class weights and up to 24 negative examples per
positive example.

On the eight validation videos, it reaches 0.8924 precision, 0.8344 recall and
0.8625 F1 for contact timing within five frames at 30 frames per second. It
produces 99 fully correct video sections out of 609 accepted sections at the
main ten-frame limit. That is 16.3% of the accepted sections and 14.8% of all
668 labelled rallies.

The next contact experiment should target missed contacts, especially the first
contact of a rally. The result does not support removing extra contacts next.

## The nine fixed runs

Contact timing uses a five-frame limit after adjustment to 30 frames per
second. Player-side accuracy is among timing matches where the Top/Bottom rule
gave an answer. The last two columns use the agreed complete-rally check at ten
frames.

| Model run | Precision | Recall | F1 | Player side | Fully correct / accepted | Correct share |
|---|---:|---:|---:|---:|---:|---:|
| HGB, original motion, balanced, 12 negatives | 0.8889 | 0.8330 | 0.8601 | 0.9041 | 96 / 614 | 15.6% |
| HGB, motion adjusted to 30 fps, balanced, 12 negatives | 0.8891 | 0.8330 | 0.8601 | 0.8995 | 90 / 609 | 14.8% |
| RF, original motion, balanced, 12 negatives | 0.8844 | 0.8232 | 0.8527 | 0.9203 | 91 / 606 | 15.0% |
| RF, motion adjusted to 30 fps, balanced, 12 negatives | 0.8692 | 0.8329 | 0.8506 | 0.9151 | 85 / 612 | 13.9% |
| HGB, original motion, no class weights, 12 negatives | 0.8866 | 0.8374 | 0.8613 | 0.9005 | 97 / 606 | 16.0% |
| RF, original motion, no class weights, 12 negatives | 0.8801 | 0.8264 | 0.8524 | 0.9042 | 68 / 620 | 11.0% |
| HGB with 15 leaves, original motion, balanced, 12 negatives | 0.8829 | 0.8192 | 0.8498 | 0.8976 | 78 / 616 | 12.7% |
| HGB with learning rate 0.04, original motion, balanced, 12 negatives | 0.8891 | 0.8330 | 0.8601 | 0.8995 | 91 / 615 | 14.8% |
| HGB, original motion, balanced, 24 negatives | **0.8924** | 0.8344 | **0.8625** | 0.9028 | **99 / 609** | **16.3%** |

All nine runs chose six frames at 30 frames per second as the distance for
merging nearby predictions. The chosen HGB run uses a score cut-off of 0.9.

The HGB run without class weights has slightly higher recall, at 0.8374, and
comes close on complete rallies. The raw balanced RF run has the highest
player-side accuracy, at 0.9203. The chosen run still gives the best contact F1
and the most fully correct sections. Those are the two results the experiment
was set up to favour.

## What breaks the leading run

Of the 564 detected sections that line up with exactly one labelled rally, 99
are fully correct and 465 fail.

The 465 failures divide as follows at the ten-frame limit:

- 266 have one or more missing contacts and no extra contact;
- 42 have one or more extra contacts and no missing contact;
- 65 have both missing and extra contacts; and
- 92 have every contact time but at least one wrong player side.

The first three counts describe contact timing. Some of those sections also
have a player-side error.

The useful narrow case is already large: 94 sections are exactly one contact
short, have at least one predicted contact, and every contact that was found
has the right time and player side. Two more one-contact rallies have no
prediction at all. Including those empty sections would make the count 96, but
they are not otherwise-good rallies. Only ten sections have exactly one extra
contact while every labelled contact and player side is otherwise right.

The contact totals point to the same problem. Within five frames, the chosen
run finds 279 of 668 first contacts, or 41.8%. It finds 4,474 of 5,028 later
contacts, or 89.0%. At ten frames, those figures rise only to 45.5% and 89.2%.

Missing contacts therefore matter much more than extra contacts. Rally starts
are the clearest place to look first. A small test of start-specific contact
handling is better supported than a model that removes contacts.

## What the complete-rally number means

The existing scorer checks the detected rally sections made by the annotator.
It does not start with one row for each of the 668 labelled rallies.

There are 677 detected sections in these videos:

- 564 line up with exactly one labelled rally;
- six contain contacts from more than one labelled rally; those six sections
  cover 104 labelled rallies; and
- 107 do not line up with a labelled rally.

For the chosen run, the scorer accepts 609 sections. Of those, 557 line up with
one rally, six contain more than one rally, and 46 contain no labelled rally.
Only a section that lines up with one rally can be fully correct. This is why
the report says “accepted sections” even though the saved result uses the field
name `rallies_kept`.

This is the same complete-rally score used by the pilot, so it is valid for the
planned comparison. The chosen run is also best when only accepted sections
that match one rally are counted: 99 of 557, or 17.8%. The scoring boundary
therefore does not change the model choice.

It does show a separate limit in the fixed rally sections. The six sections
that combine several rallies contain 104 labelled rallies, which the current
score cannot check separately. Contact detection alone cannot make those
sections fully correct.

## Raising the confidence requirement

The chosen contact cut-off is already 0.9, so the whole-rally result does not
change as the minimum contact score rises from 0.0 to 0.9.

At 0.95, the system accepts 322 sections and gets 55 fully correct. Accuracy
rises from 16.3% to 17.1%, while the number accepted almost halves. The saved
contact score is therefore a poor way to reject bad whole rallies on its own.

## Decision from this result

Use `hgb_reference_raw_more_negatives` as the first contact baseline.

Do not run deletion-first cleanup now. Extra contacts are not the main failure.
The next small test should ask whether a rally-start rule or a limited way to
add one missed contact improves otherwise-correct rallies. Any trained second
model must use first-model predictions made without training on the same video.

The chosen model design is not yet the final fitted model. After later design
choices are fixed, predictions made without training on the same video will set
the final score cut-off and duplicate distance across all 40 videos. The model
will then be trained on all 40 and tested once on the non-overlapping
ShuttleSet22 videos.

## Saved evidence

The full checked result is `raw/validation_rally_result.json.gz`, with SHA-256
`e07ae3dfe2fa2b93714fa9f66c352b0d386355bf5b6d4eaca15a686cf0f0ac5b`.
The compact figures are in `baseline_summary.json`.
