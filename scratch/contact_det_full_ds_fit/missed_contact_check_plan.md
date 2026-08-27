# Plan for checking the missed contacts

## Aim

Find out whether the chosen HGB model already scores candidate frames near the
contacts it misses. Pay particular attention to the first contact of each
rally and to otherwise-correct sections that are one contact short.

This is a read-only check of the saved validation scores. It will not change a
prediction or train a model.

## Fixed input

Use `hgb_reference_raw_more_negatives`, the chosen run in
`baseline_summary.json`.

Check the saved score rows, kept predictions, complete-rally result, fixed
32/8 split and contact labels before using them. Check file sizes and SHA-256
hashes where the saved result provides them. Load the contact-label rows only
after the prediction files have passed those checks.

## Counts to make

Use five and ten frames after adjustment to 30 frames per second.

For each missed labelled contact, record one of these explanations. Check them
in this order so each contact is counted once:

1. a kept prediction is nearby, but the one-to-one timing match used it for a
   different labelled contact;
2. no kept prediction is nearby, but another nearby score reaches the cut-off
   and was removed near a kept prediction;
3. candidate frames are nearby, but every score is below the chosen cut-off;
   or
4. no saved candidate frame is nearby.

Keep the number of nearby candidate rows, kept rows and rows at or above the
cut-off in the saved detail. This makes the explanation checkable even when
more than one kind of row is nearby.

Report the four counts separately for first contacts and later contacts. Also
report the best nearby score and its frame distance in the saved detail.

Repeat the same check for the 94 single-rally sections that are exactly one
contact short, have at least one prediction, and have the right time and player
side for every prediction they do contain.

For those 94 sections, also record whether a nearby candidate frame is inside
the detected section or only outside it. A kept prediction outside the section
is a boundary problem, not a prediction that the within-section timing match
used for another contact.

## How the result will guide the next test

- If many missed first contacts have a nearby score, test one small change to
  contact selection near rally starts.
- If many have no nearby candidate frame, changing the model or cut-off cannot
  recover them from this search area.
- If the result is mixed, keep the next test limited to the clear part rather
  than adding a broad list of candidates.

## Outside this check

- no model training
- no new score cut-off or duplicate distance
- no added or removed contacts
- no new player-side answers
- no production-code change
- no ShuttleSet22 labels

The planned implementation commit is:

`Check where the baseline misses contacts`
