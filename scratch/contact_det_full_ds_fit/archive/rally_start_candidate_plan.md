# Plan for the rally-start candidate list

## Aim

Check whether a short list made from the chosen HGB scores contains enough of
the missed first contacts to justify a later contact-choice test.

This check will not add contacts to the baseline. It will not train a model or
choose a new score cut-off.

## Fixed input

Use `hgb_reference_raw_more_negatives` and its unchanged validation scores,
kept contacts and detected sections.

Start with the tracked `baseline_summary.json` and
`missed_contact_summary.json` files. Before parsing rows from any saved result,
check its file bytes against the hash recorded in those two files. The existing
prediction check must also verify the menu result, split, raw feature record
and contact-label file hash.

Build the candidate list twice and require byte-for-byte equal JSON. Use the
validation-video order, detected-section order and candidate order described
below. Sort JSON object keys and use the same compact spacing both times. Do
this before opening the saved missed-contact detail.

The validation labels may only be used after the candidate list is fixed.
They may measure coverage, but they may not change the list.

## Candidate list

For each detected section that contains a kept contact:

1. Use the earliest kept contact in the section as the fixed contact.
2. Use score rows from the same HGB search interval. Start at the first saved
   row in that interval. If the preceding detected section ends inside the
   same interval, start there instead.
3. End the search just before the fixed contact. This includes score rows that
   fall before the detected section starts.
4. Add the strongest earlier `contact_score` row that is farther than the chosen
   duplicate-removal distance from the fixed contact.
5. Add the next strongest row that is also farther than that distance from
   every row already in the list.

Break equal-score ties by taking the earlier frame first. Adjust the fixed
six-frame duplicate-removal distance for each video's frame rate in the same
way as the baseline.

The saved score file's `kept` field identifies the fixed contacts. Each list
therefore contains the fixed contact and at most two earlier score rows. A
section with no kept contact gets no list. Do not use the old filtered
heuristic contact or its failed hand-written choice rule. That signal was not
saved with the full-data scores.

## Fixed limits

The baseline has 615 detected sections with at least one kept contact. The
list must meet all four limits:

- at most three candidates for any detected section;
- at most 1,845 candidates in total, including the 615 fixed contacts;
- cover at least 50 of the 81 missed first contacts in the otherwise-correct
  one-short sections at the ten-frame timing limit; and
- add no more than 25 candidate frames for each newly covered first contact.

The last count uses the actual number of earlier candidates as its numerator.
The total is the sum of the section-list sizes. The same video-and-frame row
counts once for each section list that contains it because each use is a
separate choice for a later method.

Here, one-short means that the section is otherwise correct and is missing one
first contact. The list covers that contact when an earlier candidate from the
same section is within the frame-rate-adjusted ten-frame limit. Count the
section once even if more than one candidate is nearby. A candidate in another
section does not cover it.

The coverage limit keeps roughly the pilot's result: its HGB-only list added
590 candidates and covered 59 of 96 missed serves. The cost limit allows up to
1,230 added list entries divided by 50 covered first contacts, or 24.6 added
entries per covered contact. It is still stricter than the broad pilot list,
which added about 31.6 candidates for each recovered contact.

## Result to report

At five and ten frames after adjustment to 30 frames per second, report:

- how many missed first contacts have a candidate in the list;
- the same count for the 81 otherwise-correct one-short sections;
- how often the matching candidate is before the detected section;
- candidate counts for each section and video;
- total candidates, added candidates and added candidates per newly covered
  first contact; and
- score and frame-distance summaries for matching and non-matching candidates.

Also report which of the four fixed limits pass. A failed limit stops this
line of work. Passing the limits allows a separate plan for choosing one
candidate. It does not approve a hand-written rule or a trained model.

## Outside this check

- no contact is added, removed or moved
- no player-side answer is changed
- no model is trained
- no cut-off or duplicate distance is tuned
- no production code changes
- no ShuttleSet22 labels

The planned implementation commit is:

`Build the rally-start candidate list`
