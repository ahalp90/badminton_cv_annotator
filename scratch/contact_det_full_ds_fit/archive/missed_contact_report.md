# Where the baseline misses contacts

## Bottom line

The missed first contacts and missed later contacts have different causes.

Most missed first contacts already have a saved candidate frame nearby. The
chosen HGB model usually scores that frame below its 0.9 cut-off. Most missed
later contacts have no saved candidate frame nearby, so changing the cut-off
cannot recover them.

The next small test should build a short list of rally-start candidates without
using contact labels. It should include frames just before a detected section
starts. Do not lower the cut-off across every contact and do not try removing
extra contacts next.

## All missed contacts

At the main ten-frame limit, the chosen HGB run misses 364 of 668 first
contacts. The saved scores explain those misses as follows:

- 264 have nearby candidate scores below 0.9;
- 16 have a kept prediction nearby that the one-to-one match uses for another
  labelled contact;
- four have a nearby score at or above 0.9 that was removed near a kept
  prediction; and
- 80 have no saved candidate nearby.

This means 284 of the 364 misses, or 78.0%, still have a candidate frame within
the timing limit.

The run misses 542 later contacts at the same limit:

- 97 have nearby candidate scores below 0.9;
- 42 have a kept prediction nearby that matched another labelled contact;
- 24 have a nearby score at or above 0.9 that was removed; and
- 379 have no saved candidate nearby.

Nearly 70.0% of the later-contact misses therefore fall outside the saved
candidate frames. A different cut-off cannot fix that part.

The five-frame result says the same thing. A saved candidate is near 305 of 389
missed first contacts, but only 175 of 554 missed later contacts.

## The 94 sections that are one contact short

The narrow complete-rally case is clearer than the combined contact totals.
All 94 otherwise-correct one-short sections have a saved candidate near the
missing contact.

Eighty-one of the 94 are missing the first contact. Among those 81:

- 74 have only below-cut-off candidate scores;
- three have a score that reached the cut-off but was removed;
- two have a kept prediction inside the section that matched another contact;
- two have a kept prediction outside the section; and
- none lack a nearby candidate.

Only 42 have any nearby candidate inside the detected section. The other 39
have candidates only before the section starts. A start-specific test must
therefore allow an earlier frame instead of searching only inside the current
section.

The median best nearby score for these 81 missing first contacts is 0.497.
Thirty-nine reach 0.5, 25 reach 0.7, 18 reach 0.8 and seven reach 0.9. A single
lower cut-off will not recover every case. Applying it everywhere also cannot
address the 379 later-contact misses with no candidate nearby.

The remaining 13 sections are missing a later contact. All 13 have a candidate
inside the section. Eleven are below the cut-off and two have a kept prediction
that matched another contact. Their median best score is 0.860.

## What this result supports

The next check should copy the useful part of the pilot's rally-start idea:
make a small candidate list around each detected section start, including a
limited number of frames before the section. Fix that list without labels,
then measure how many missed first contacts it contains and how many extra
candidates it adds.

Set the size and coverage limits before reading the result. Stop if the list
becomes broad or repeats the pilot's poor trade-off.

Those numbers are not set yet. Do not build or score the list until a separate
plan records the maximum candidates per section, the maximum total candidates,
the minimum first-contact coverage and the maximum added candidates per newly
covered contact.

If a trained chooser follows, each training video's candidate scores must come
from a first contact model that did not train on that video. The final cut-off
and duplicate distance still wait for predictions made this way across all 40
videos.

## Saved evidence

The checked detail is `raw/missed_contact_check.json.gz`, with SHA-256
`dc54c9abf215f8f7975cc1e6e530e6a18d02d0adb7dbf7030f5dba10fd35c51e`.
The compact totals are in `missed_contact_summary.json`.
