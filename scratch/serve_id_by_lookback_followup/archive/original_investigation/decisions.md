> ARCHIVED 2026-08-12: historical decisions. Current decision: `../../README.md`.

# Decisions

## Approved H3/R8 extension

`02_LAUNCH_H3_R8_DUAL_SEARCH.md` is the governing rule sheet for the next run.
The completed experiment below remains the baseline rather than being rewritten.

The extension uses a three-source-frame recurrence halo and
`largest_step_ratio <= 8.0` for both pre- and post-contact traces. It first saves
the three-way pre verdict and binary post verdict for every accepted contact.

It then runs two searches from that frozen evidence:

1. the existing sequential outgoing-first search; and
2. an incoming-only search anchored at the earliest accepted contact with
   positive incoming evidence.

The incoming-only search inspects only the immediately preceding accepted
contact. The predecessor is admitted by an inclusive 60-base-30fps gap, or by a
measured `high_shot_oob` state with each contact within an inclusive
12-base-30fps endpoint buffer. Admission is not proof of causation.

An eligible predecessor with `not incoming` is the visible-serve candidate. An
eligible predecessor with `unavailable` ends as insufficient trajectory. With
no eligible predecessor, the anchor is the first visible post-serve contact and
the search implies an unshown serve without inventing its frame.

The four-cap 10/30/60/75 comparison is out. Recursive chaining, cross-gap
continuity tests, and any outgoing gate in the incoming-only search are out.

The experiment scans accepted contacts in chronological order and stops at the first contact with credible outgoing motion. It then reuses the fixed PR #82 pre-contact check to classify that contact as a visible serve or the first visible post-serve contact.

## Sequential opener search

For accepted contacts `A0 < A1 < ... < An`:

```text
for each accepted contact from earliest to latest:
    credible outgoing motion -> select and stop
    otherwise                -> skip

no selected contact -> no credible accepted contact
```

The outgoing predicate is binary. Missing or unusable post-contact evidence is not credible outgoing motion and receives no separate search or reporting state.

## Classify the selected contact

Apply the existing PR #82 incoming-motion check to the selected contact:

```text
unavailable pre-contact evidence -> not enough shuttle trajectory to tell
incoming motion                 -> first visible post-serve contact; imply an unshown serve
no incoming motion              -> visible serve at the selected accepted frame
```

An implied serve has no invented frame. The result states only that the serve occurred before the selected visible contact.

The main search does not trace backwards through earlier contacts. A later contact never overrides an earlier `no outgoing` verdict. The experiment has no accepted-contact reconnection, contact-chain pass, cross-gap continuity test, or contact-gap threshold.

## Fixed motion measurements

```text
local window            30 base-30fps frames before or after the contact
minimum path            5 visible recurrence-clean frames
maximum local gap       2 base-30fps frames from the contact
gross-jump limit        largest_step_ratio <= 4.0 within the visible trace
incoming                fitted_decrease_bh >= 0.05
outgoing                fitted_decrease_bh <= -0.05
```

Each local path stays inside the contact's tracker scene. Unavailable evidence never becomes measured absence of motion.

## GT and reporting boundary

- Run the same GT-free search over all 239 one-to-one rallies
- Freeze each search result before joining GT by `(fixture, video_id, set_id, rally)`
- Use GT only to score the selected accepted contact and final category
- Report fixed, damaged, unchanged, pre-contact-unknown, and no-credible-contact outcomes
- Report accepted contacts skipped for non-credible outgoing motion, final accepted rank, visible serves, and implied serves
- Keep the 97 currently unmatched first impulses as a reporting slice, not the search population
- Keep ±10 primary and use ±5 and ±30 only as compact scoring checks

## Scope

- Keep production code and PR #82 files read-only
- Use accepted impulses only
- Keep raw and rejected impulses out
- Keep production serve-start logic, threshold sweeps, learned models, and dynamic programming out
- Keep all new code and evidence under `scratch/serve_id_by_lookback_followup/`
- Stop after the short audited report; do not propose production architecture
