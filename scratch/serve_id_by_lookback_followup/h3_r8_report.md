# H3/R8 accepted-contact opener results

Neither opener search is good enough to replace the first accepted contact.
The relaxed outgoing-first search improves on the earlier H15/R4 experiment,
but it still produces a classified correct opener in only 43 of 239 rallies at
the primary +/-10 tolerance. The incoming-only predecessor search produces 26
correct results. Its admitted predecessor has unavailable evidence in 157
rallies.

The measured high-shot exception behaves differently from the ordinary timing
rule. It admits five long-gap predecessors, including clip 13, and all five are
the GT serve contact at +/-10. The ordinary 60-frame rule admits 196
predecessors. Only 3 produce a correct visible-serve result.

## Fixed experiment

Both searches use:

```text
recurrence halo              3 source frames per side
local trace window           30 base-30fps frames
minimum path                 5 usable frames
maximum local contact gap    2 base-30fps frames
largest_step_ratio           <= 8.0
incoming                     fitted_decrease_bh >= 0.05
outgoing                     fitted_decrease_bh <= -0.05
```

The incoming-only search uses the earliest accepted contact with positive
incoming evidence. It inspects only the nearest earlier accepted contact. The
predecessor is admitted within 60 base-30fps frames, or through the fixed
measured `high_shot_oob` exception with 12-base-30fps endpoint buffers.

The run saved 3,200 accepted-contact evidence rows before either search. It then
derived 239 GT-free result rows from the saved evidence. GT was joined after
the result fields were complete. A full rebuild matched every decompressed
evidence row, result field, score, and summary.

## What `unavailable` means in this run

`Unavailable` means the closest pre-contact run is absent or fails a common
eligibility check. It is not evidence that the shuttle was not incoming.

Across all 3,200 accepted contacts, 2,329 pre-contact paths are eligible and 871
are unavailable. The unavailable share is 27.2%, not nearly half of all
accepted contacts.

| Pre-contact path state | Contacts |
| --- | ---: |
| eligible | 2,329 |
| largest-step ratio above 8.0 | 321 |
| no usable run | 237 |
| local contact gap too large | 172 |
| one-frame run, so measurement unavailable | 74 |
| two to four frames | 56 |
| player or tracker context unavailable | 11 |

The three-way verdicts over all contacts are 1,963 incoming, 366 not incoming,
and 871 unavailable.

The earlier report's 100 `pre-contact unknown` rallies did not mean 100 impulses
failed the outgoing test. Those contacts had already passed outgoing. Their
selected contact lacked an eligible path before the contact.

## Search A: relaxed outgoing-first

The H3/R8 outgoing scan selects a contact in 234 rallies. Its terminal results
are:

| Category | Rallies |
| --- | ---: |
| first visible post-serve contact | 68 |
| visible serve | 23 |
| selected contact has unavailable pre evidence | 143 |
| no credible outgoing contact | 5 |

At +/-10, 43 results are classified correctly. The transition counts are 26
fixed, 13 damaged, 17 unchanged correct, 35 unchanged wrong, 143 unavailable,
and 5 without a credible outgoing contact.

The result improves on the completed H15/R4 outgoing experiment, which produced
34 correct results, 100 pre-contact unknowns, and 27 rallies without a credible
contact. H3/R8 selects an earlier frame in 129 rallies and recovers a selected
contact in 22 of the 27 earlier no-contact cases.

The extra outgoing sensitivity shifts the main failure rather than removing it.
The earlier selected contact has unavailable pre evidence in 143 rallies. Of
those, 90 have no usable pre run and 25 still exceed the 8.0 step-ratio limit.

The compact tolerance check does not change the conclusion:

| Tolerance | Correct | Fixed | Damaged | Pre unavailable | No credible contact |
| --- | ---: | ---: | ---: | ---: | ---: |
| +/-5 | 38 | 27 | 10 | 143 | 5 |
| +/-10 | 43 | 26 | 13 | 143 | 5 |
| +/-30 | 51 | 27 | 23 | 143 | 5 |

The +/-30 result has 63 selected frames with multiple GT contacts inside the
window, so it is only a coarse sensitivity check.

## Search B: earliest incoming plus predecessor

An incoming anchor exists in 234 rallies. Its accepted rank is 1 in 24 rallies,
2 in 107, 3 in 69, and 4 or later in 34. Five rallies have no incoming anchor;
all five include unavailable evidence.

The final categories are:

| Category | Rallies |
| --- | ---: |
| predecessor visible serve | 44 |
| anchor is first visible post-serve contact | 33 |
| predecessor evidence unavailable | 157 |
| no incoming anchor with unavailable evidence | 5 |

At +/-10, 26 results are correct. The transition counts are 23 fixed, 24
damaged, 3 unchanged correct, 27 unchanged wrong, 157 predecessor-unavailable,
and 5 without an incoming anchor.

The ordinary time-only predecessor rule is the weak part:

| Admission | Rallies | Classified as visible serve | Correct at +/-10 |
| --- | ---: | ---: | ---: |
| ordinary window | 196 | 39 | 3 |
| measured high-shot exception | 5 | 5 | 5 |
| no predecessor admitted | 38 | 33 first-post classifications | 18 |

The 157 ordinary predecessors with unavailable evidence remain unknown. Among
the 39 ordinary predecessors with measured no-incoming evidence, 36 are not the
GT serve at +/-10. This is direct evidence that proximity in time is not enough
to distinguish the serve from accepted contacts that do not match the GT serve.

The high-shot exception is useful in this fixed sample, but five cases are too
few for a broad production claim. The result supports retaining the exception
as a measured special case rather than widening the ordinary timing rule.

Clip 13 is recovered exactly as intended. Rally `sset_21:21:set1:6` has accepted
contacts 16509 and 16574. They bracket the measured high-shot state
`[16515, 16564)`. The endpoint distances are 6 and 10 frames. The predecessor
at 16509 is measured `not incoming` and matches GT contact 1 at +/-10.

## Player and occlusion check

The fixtures contradict the hypothesis that most first-contact outgoing
failures come from the bottom player. Under the earlier H15/R4 rule, 156 of 218
first-contact failures are Top and 62 are Bottom. The rule rejects almost both
groups, with failure rates of 92.9% and 87.3% respectively.

H3/R8 separates the groups more strongly:

| First accepted player | Outgoing false | Total first contacts | Failure rate |
| --- | ---: | ---: | ---: |
| Top | 103 | 168 | 61.3% |
| Bottom | 20 | 71 | 28.2% |

Top contacts also have the higher all-contact pre-unavailable rate: 592 of
1,755 Top contacts (33.7%), compared with 268 of 1,434 Bottom contacts (18.7%).
Eleven contacts have no player attribution and are unavailable.

The first-contact outgoing failures have several causes. Thirty-eight exceed
the two-frame local gap, 23 exceed ratio 8.0, 21 have no usable run, and 7 have
too little evidence to measure a five-frame path. Another 34 have an eligible
path whose fitted direction is not outgoing.

## Guard and step-ratio checks

The halo-15 reconstruction exactly matches production guard codes on every
fixture. Halo 3 clears 13,549 frames in `sset_01`, 15,161 in `sset_15`, and 8,429
in `sset_21` relative to halo 15. Recurrence cores and repeated attractor
positions remain rejected.

The local extractor uses one contiguous usable run. It does not calculate a
step across missing or rejected frames. Raising the limit from 4.0 to 8.0 makes
545 pre paths and 897 post paths fall inside the new ratio band. Another 321 pre
paths and 526 post paths still fail the final H3/R8 eligibility check because
their selected run exceeds 8.0.

## Conclusion

H3/R8 is a reasonable correction to the local trajectory measurement. It
recovers credible traces that H15/R4 rejected, and the high-shot exception
handles the intended long-gap cases.

The opener rules remain the larger problem. Outgoing-first still selects too
many contacts without usable evidence before them. Earliest-incoming plus an
ordinary time-only predecessor mostly lands on unavailable or wrong earlier
impulses. The measured high-shot exception is the only predecessor connection
that performs cleanly in this run.
