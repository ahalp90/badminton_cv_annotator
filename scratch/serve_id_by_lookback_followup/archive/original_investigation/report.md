> ARCHIVED 2026-08-12: original first-iteration report. Current narrative: `../../report.md`.

# Outgoing motion is too strict for opener selection

Requiring credible outgoing motion does remove some false early impulse openers, but it rejects far too many real openers. At the primary +/-10-frame tolerance, the rule fixes 16 starts and damages 34. Another 127 rallies end with unavailable pre-contact evidence or no credible accepted contact.

The main failure occurs before the incoming check. Only 21 of 239 first accepted contacts have credible outgoing motion. The scan skips the first contact in 218 rallies and usually lands late: the median selected accepted rank is 3, with a maximum of 24.

The fixed rule is therefore not a useful opener replacement. The existing PR #82 incoming check cannot recover once the outgoing scan has moved past the serve and first return.

## Primary result

The search ran on all 239 one-to-one rallies without GT stroke frames or labels in its input. Each frozen search result was then scored against GT.

| +/-10 outcome | Rallies |
| --- | ---: |
| Fixed | 16 |
| Damaged | 34 |
| Unchanged correct | 18 |
| Unchanged wrong | 44 |
| Pre-contact trajectory unavailable | 100 |
| No credible accepted contact | 27 |

The 97 baseline-unmatched starts contain 15 fixes, 38 unchanged wrong results, 35 pre-contact unknowns, and 9 rallies with no credible contact.

The cost also falls heavily on the existing successes. Of the 119 starts already correct at +/-10, only 18 remain correct. The rule makes 34 classified results wrong and sends the other 67 to pre-contact unknown or no credible contact.

## What the two checks do

The binary outgoing search selects a contact in 212 rallies and finds none in 27. It rejects the first accepted contact in 218 rallies.

The PR #82 pre-contact check can classify 112 of the 212 selected contacts:

| Pre-contact result | Rallies | Correct category at +/-10 |
| --- | ---: | ---: |
| Incoming: imply an unshown serve | 94 | 28 |
| Not incoming: visible serve | 18 | 6 |
| Unavailable | 100 | n/a |

Most incoming calls are made after the sequential scan has already moved to a later rally contact. Sixty-six of the 94 implied-serve selections do not match GT contact 2 at +/-10.

## Tolerance checks

| Tolerance | Fixed | Damaged | Final category correct | Multiple GT matches at selected frame |
| --- | ---: | ---: | ---: | ---: |
| +/-5 | 16 | 23 | 32 | 0 |
| +/-10 | 16 | 34 | 34 | 12 |
| +/-30 | 15 | 44 | 39 | 167 |

The conclusion is stable at +/-5 and +/-10. The +/-30 result is highly ambiguous because 167 selected frames are close to more than one GT contact.

## Representative cases

| Rally | First accepted result | Selected result | Outcome |
| --- | --- | --- | --- |
| `sset_01/set2/r25` | Frame 70225 unmatched | Rank 2, frame 70244, incoming, exact GT contact 2 | Fixed by implying the serve |
| `sset_01/set1/r10` | Frame 17425 matches GT contact 1 | Rank 3, frame 17474, incoming, matches a later contact | Damaged after skipping the real serve |
| `sset_01/set2/r30` | Frame 75057 matches GT contact 1 | Rank 8, frame 75260, not incoming, matches a later contact | Damaged as a false visible serve |
| `sset_01/set1/r6` | Frame 14939 matches GT contact 1 | Rank 2, frame 14977, pre-contact evidence unavailable | Ends unknown after skipping the real serve |
| `sset_01/set1/r7` | Frame 15374 matches GT contact 1 | All four accepted contacts fail outgoing | Ends with no credible accepted contact |

## Method boundary

The fixed 239-rally population uses the existing GT-derived rally-to-span crosswalk. Search actions are still isolated from GT: the search receives only fixture geometry, trajectory arrays, scene and span bounds, and accepted contact frames. Frozen rows are joined to stroke frames only for scoring.

The outgoing predicate is deliberately binary. Missing and unusable post-contact evidence receive the same `false` result as measured absence of outgoing motion. Only the selected contact's PR #82 pre-check keeps an unavailable state.

Check mode rebuilt all three fixtures and directly matched every decompressed search field, scored field, and summary value across the 239 saved rows.

Independent read-only audits by `claude-opus-4-6-thinking` and `gemini-3.1-pro-high` both passed. Neither found GT leakage, scoring errors, denominator problems, hidden contact-chain logic, or a conclusion-changing omission.
