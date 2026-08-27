# Rally-start candidate result

## Bottom line

The short rally-start list passes all four limits set before the result was
opened.

At the main ten-frame timing limit, it contains a useful earlier frame for 56
of the 81 otherwise-correct sections that are missing their first contact.
Thirty of those 56 contacts are covered only because the list reaches before
the detected section starts.

Keep the list for a separate plan about choosing one candidate. It is not a
new contact stream. No candidate has been added to the baseline, and this
result does not choose which earlier frame should be used.

## List size

The baseline has 677 detected sections. Of these, 615 contain a kept contact
and receive a list. The other 62 receive no list.

Every list has exactly three entries:

- the first kept contact in the section; and
- the two strongest earlier HGB score rows that are far enough apart.

This gives 1,845 entries in total. Of these, 1,230 are the earlier candidates
that a later method would need to accept or reject.

## The 81 target first contacts

At ten frames after adjustment to 30 frames per second:

- 56 of 81 missed first contacts are covered, or 69.1%;
- 41 have at least one matching candidate before the section starts;
- 30 are covered only by candidates before the section starts; and
- the list adds 21.96 earlier entries for each covered first contact.

At five frames, the list covers 45 of 81 contacts. Twenty-six of those are
covered only by a candidate before the section starts.

The matching candidates do not have a clean score advantage. At ten frames,
their median HGB score is 0.520. The other candidates in the same 81 lists have
a median score of 0.553. A simple rule that always takes the larger HGB score
is therefore poorly supported by this result.

## Other missed first contacts

The saved complete-rally result assigns 267 of the 364 missed first contacts at
ten frames to one detected section. The list covers 175 of those 267 contacts.
The remaining 97 cannot be checked with the same-section rule because the
complete-rally result does not assign them to one section.

At five frames, 290 of 389 missed first contacts have one detected section and
142 are covered by that section's list.

These wider counts are supporting evidence. The fixed pass decision uses the
81 otherwise-correct sections described above.

## Fixed limits

All four checks pass:

- no section has more than three candidates;
- the total is no more than 1,845 candidates;
- at least 50 of the 81 target first contacts are covered; and
- there are no more than 25 earlier entries per covered first contact.

The output was reproduced with identical compressed-file hashes. A separate
`jq` recount also found 81 targets, 56 covered contacts, 30 covered only before
the section, 1,845 total entries and 1,230 earlier entries.

## Decision

Keep this candidate-list construction for the next stage. Do not apply all
1,230 earlier entries as contacts and do not reuse the pilot's failed
hand-written choice rule.

Before training a method to choose one earlier frame, make first-model scores
for the training videos using models that did not train on those videos. This
is the next planning task. It also provides the out-of-fold scores needed to
set the final cut-off and duplicate-removal distance across all 40 videos.

## Saved evidence

The fixed list is `raw/rally_start_candidates.json.gz`, with SHA-256
`d60e7cb5eaa2e5bb01333f587662bf5ad96a546e22e6d89145f36efebdcf5f55`.

The checked result is `raw/rally_start_candidate_check.json.gz`, with SHA-256
`118774d5f5eb24aa819f1d6398c3d6c7b5b8cabf0027628e4a3d85d2320e1fd9`.
