# Detailed results

Start with [../report.md](../report.md) for the explanation. This document keeps
the exact settings, counts and limits needed for implementation review.

## Scores

**Server attribution** is correct when the predicted court side matches the
annotated server side.

**Visible-start attribution** is correct when a visible-serve claim matches
annotated contact 1, or a first-return claim matches contact 2. The main timing
tolerance is ten base-30-fps frames.

| Rule | Server side | Visible start | Both |
| --- | ---: | ---: | ---: |
| PR #82 | 163/239 | 125/239 | 96/239 |
| Narrow high-shot correction | 164/239 | 127/239 | Not composed as a full rule |
| Preferred layered rule | 170/239 | 132/239 | 117/239 |
| Rank-1 fallback sensitivity | 171/239 | 131/239 | 117/239 |

PR #82 has 67 rallies with an incorrect visible-start attribution and the
correct server side.

## Populations

| Population | Rallies | Meaning |
| --- | ---: | --- |
| All annotated rallies | 292 | End-to-end context, including segmentation failures |
| Covered rallies | 249 | Includes merged predicted spans |
| One predicted span to one annotated rally | 239 | Fixed development population used here |

The 239-rally population comes from an annotated crosswalk. Prediction branches
do not read server labels or stroke labels, but the rule was assembled after
development labels were inspected.

## PR #82 reference

| Method | Correct server sides |
| --- | ---: |
| Released alternating fit | 124/239 |
| Player at earliest accepted contact | 152/239 |
| Direct incoming-motion correction | 163/239 |
| Prepend inferred server and refit | 159/239 |

PR #82 checks up to 30 base-30-fps frames before the earliest accepted contact.
At least five continuous shuttle observations are required. A robust fitted
distance decrease of at least 0.05 apparent player body heights means the
shuttle is moving towards that player.

Only 24 rallies had a usable pre-contact path under the original path-quality
check. Nineteen paths had a unique contact-1 or contact-2 timing label. The
fixed classifier got 11 of those 19 paths right.

## How path quality is checked

The direction rule first needs a continuous path that is close enough to the
contact and free from obvious tracking failures.

| Check | Value in the later pass | Plain meaning |
| --- | ---: | --- |
| Minimum observations | 5 | A direction call needs more than a few points |
| Local window | 30 base-30-fps frames | Use only motion near the contact |
| Maximum gap to contact | 2 base-30-fps frames | The observed run must reach the contact closely |
| Repeated-position pad | 3 source frames per side | Ignore a small area around a stale repeated tracker position |
| Largest-step ratio | At most 8.0 | Reject a one-frame leap over eight times the path's usual non-zero step |
| Incoming threshold | Fitted decrease at least 0.05 body heights | Shuttle moves towards the player |
| Outgoing threshold | Fitted decrease at most -0.05 body heights | Shuttle moves away from the player |

The path uses one continuous run. It never calculates a step across a missing
or rejected frame.

The original pass used a fifteen-frame repeated-position pad and a largest-step
ratio of 4.0. The later values were fixed from tracker behaviour and a reviewed
continuous example rather than swept against labels.

## Strict outgoing-contact search

The first follow-up selected the earliest accepted contact followed by credible
motion away from the player. It then classified the motion before that contact.

The search selected a contact in 212 rallies and found none in 27. It skipped
the first accepted contact in 218 rallies. The median selected rank was 3 and
the maximum was 24.

| Visible-start outcome | Rallies |
| --- | ---: |
| Fixed | 16 |
| Damaged | 34 |
| Stayed correct | 18 |
| Stayed wrong | 44 |
| Pre-contact path unavailable | 100 |
| No credible outgoing contact | 27 |

The final visible-start score was 34/239. The rule was rejected as an opener
replacement.

## Less brittle path check

The later pass saved path evidence for 3,200 accepted contacts.

### Before-contact path status

| Status | Contacts |
| --- | ---: |
| Usable | 2,329 |
| Largest-step ratio too large | 321 |
| No continuous usable run | 237 |
| Too far from contact | 172 |
| Measurement unavailable | 74 |
| Too few frames | 56 |
| Contact context unavailable | 11 |

The before-contact verdicts were 1,963 incoming, 366 not incoming and 871
unavailable.

### Outgoing-contact search

| Selected result | Rallies |
| --- | ---: |
| Return-like: incoming before selected contact | 68 |
| Serve-like: not incoming before selected contact | 23 |
| Before-contact path unavailable | 143 |
| No credible outgoing contact | 5 |

As a visible-start replacement, the search fixed 26 answers, damaged 13 and
left 148 without an answer. The final visible-start score was 43/239.

The 91 direct server answers were correct in 60 rallies.

## Predecessor and high-shot search

The separate predecessor search found the earliest contact with incoming motion
and inspected the accepted contact immediately before it.

The ordinary rule admitted a predecessor within 60 base-30-fps frames. It
admitted 196 predecessors. Thirty-nine had measured serve-like motion, but only
3 matched contact 1.

The long-gap exception required a measured high-shot state between the two
contacts. Each contact also had to be within 12 base-30-fps frames of its state
endpoint.

Five rallies used that exception. All five predecessor frames matched contact
1. Three kept the existing PR #82 frame. Two changed it, and both changes were
correct. Only one changed the server side, also correctly.

All five cases came from `sset_21`, video 21, set 1.

## Rejected expansions

| Candidate | Changes | Timing fixes | Timing damages | Other result |
| --- | ---: | ---: | ---: | --- |
| Broad wrist-proximity setup | 138 | 22 | 63 | 53 changed but stayed wrong |
| Continued same-player setup | 2 | 1 | 1 | No server change |
| Newly incoming at original contact | 9 | 4 | 2 | Server: 5 fixes, 4 damages |
| Later incoming anchor without predecessor | 6 | 4 | 1 | 1 stayed wrong |
| Ordinary predecessor with different players | 26 | 3 | 10 | 13 stayed wrong |
| Drop rank 1 using later evidence and alternation | 24 | 7 | 9 | 8 stayed wrong |

## Preferred layered server rule

The preferred rule uses the outgoing-selected inference when both sides of the
contact can be measured. It keeps PR #82 otherwise.

```text
if selected contact is return-like:
    choose the other player's side
elif selected contact is serve-like:
    choose the selected player's side
else:
    keep the PR #82 server answer
```

| Branch | Rallies | Correct server sides | Correct visible starts | Both |
| --- | ---: | ---: | ---: | ---: |
| Return-like; choose other side | 68 | 45 | 34 | 33 |
| Serve-like; choose selected side | 23 | 15 | 9 | 8 |
| PR #82 fallback | 148 | 110 | 89 | 76 |
| **Total** | **239** | **170** | **132** | **117** |

The rule changes 33 PR #82 server answers. It fixes 20 and damages 13. The
server rate rises from 68.20% to 71.13%. The exact paired two-sided p-value is
0.296.

The rule also makes 19 visible-start fixes and 12 visible-start damages.

## Rank-1 fallback sensitivity

The sensitivity keeps the same 91 direct branches. When they cannot answer, it
uses motion at the first accepted contact instead of preserving PR #82.

It gets 171 server sides, 131 visible starts and 117 joint answers right. It
changes 34 PR #82 server answers, with 21 fixes and 13 damages.

The preferred rule is retained because it preserves one more visible-start
answer and has a clearer fallback boundary.

## Curved-path proposal

The proposal keeps existing incoming calls. It reconsiders only a not-incoming
path with a well-supported interior turn. The supplied demonstration digitised
eight plotted errors and rescued two missed returns.

A later audit reports an exact 19-path timing improvement from 11 to 13, but
also reports server damage: 163 to 159 for PR #82 and 170 to 167 for a
fallback-only preferred-rule change.

The exact path inputs and audit helper are absent. These are audit claims rather
than reproduced results. The curved-path proposal is excluded from the server
rule.

## Historical terminology footnote

Archived records call the less brittle path check `H3/R8`. The first component
is the three-source-frame pad around repeated tracker positions. The second is
the 8.0 largest-step ratio. Live material uses the plain descriptions above.

## Limits

- Every rule was examined on the same development fixtures
- The bundle does not rerun TrackNet, pose, contact generation or segmentation
- The 239-rally population is not the full 292-rally end-to-end population
- The monocular two-dimensional distance trace does not recover shuttle height or depth
- An unseen evaluation is required before production use
