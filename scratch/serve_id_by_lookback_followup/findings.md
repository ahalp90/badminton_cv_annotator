# Findings for the sequential accepted-contact search

## H3/R8 follow-up result

The completed findings below describe the original halo-15, ratio-4 experiment.
The separate H3/R8 result is reported in `h3_r8_report.md`.

The follow-up will use a three-frame recurrence halo and an 8.0 gross-step
ratio. It will save pre- and post-contact evidence for every accepted contact,
then run both the existing outgoing-first search and a new incoming-only search.

The incoming-only search begins at the earliest contact with positive incoming
evidence. It inspects only the nearest earlier accepted contact. The ordinary
admission window is 60 base-30fps frames. A measured high-shot state may admit a
longer gap when both contacts are within 12 base-30fps frames of the state's
respective endpoints. Neither rule proves that the earlier impulse caused the
shot.

The checked run saved 3,200 accepted-contact rows and 239 dual-search rows. The
relaxed outgoing-first search is correct in 43 rallies at +/-10. It has 143
selected contacts with unavailable pre evidence and 5 rallies with no credible
outgoing contact.

The incoming-only search is correct in 26 rallies. It has 157 admitted
predecessors with unavailable evidence. Of 39 ordinary-window predecessors
classified as a visible serve, only 3 match GT contact 1 at +/-10. All 5
predecessors admitted by the measured high-shot exception match contact 1.

The first-contact player hypothesis is also resolved. Under H3/R8, outgoing
fails for 103 of 168 Top first contacts and 20 of 71 Bottom first contacts.
The bulk of failures is Top, both by count and rate.

The experiment runs the same GT-free search over all 239 one-to-one rallies. It skips every accepted impulse without credible outgoing motion. It stops at the first contact with credible outgoing motion and uses the existing PR #82 incoming check to classify that contact.

The 97 first impulses unmatched at +/-10 base-30fps frames are an analysis slice. Ground truth (GT) does not decide whether the search runs.

## F1. Accepted contacts provide the only search candidates

PR #82 sorts each frozen span's accepted `filtered_by_rally` frames and checks them against the raw wrist, suppression, and exclusion fields. The search scans those accepted frames in chronological order.

Raw and rejected impulses remain out of scope.

## F2. Post-contact outgoing motion selects the opener candidate

If accepted impulse `Ai` lacks credible outgoing motion, the search skips it. The first `Ai` with credible outgoing motion is selected and the forward search stops.

A later contact never overrides an earlier `no outgoing` verdict. The search has no contact reconnection or broader contact-chain pass.

## F3. The forward search is binary

Missing or unusable post-contact evidence fails the credible-outgoing predicate. The forward search does not distinguish that case from measured absence of outgoing motion.

The selected contact's pre-contact check keeps its real three-way result: incoming, not incoming, or unavailable. Only pre-contact unavailability produces `not enough shuttle trajectory to tell`.

## F4. The existing incoming check classifies the selected contact

Incoming motion into the selected contact means it is the first visible post-serve contact. The search implies an unshown serve before that contact without inventing an exact serve frame.

Usable pre-contact evidence with no incoming motion means the selected accepted contact is the visible serve.

## F5. Exhausting the accepted contacts is a distinct result

If no accepted contact has credible outgoing motion, the search reports `no credible accepted contact`.

## F6. The existing PR #82 table fixes the baseline, not the new result

The checked PR #82 output contains 239 primary rallies. Their current first accepted impulses are:

| GT label at +/-10 | Rallies |
| --- | ---: |
| contact 1 | 119 |
| contact 2 | 19 |
| later | 4 |
| unmatched | 97 |

These labels score the result after the search has run. They never select a search branch.

## F7. PR #82 supplies the trajectory primitives

The existing work provides `closest_pre_contact_run`, `measure_incoming_motion`, `fit_robust_distance_trend`, recurrence-clean path checks, and the fixed 0.05-BH incoming rule.

The new code needs only the direct post-contact mirror and the sequential state machine. It does not need trace-end reasons, backwards origins, cross-gap tests, or a contact-gap distribution.

## F8. The primary outputs are transition counts over all 239 rallies

For every rally, record the current first-impulse GT label and the reconstructed outcome. Report:

- currently wrong starts fixed
- currently correct starts damaged
- results unchanged
- selected contacts with unavailable pre-contact evidence
- results ending with no credible accepted contact
- accepted contacts skipped for non-credible outgoing motion
- selected accepted rank
- visible serves and implied unshown serves
- final visible-contact GT ordinal

Repeat the relevant breakdown within the 97 currently unmatched starts. Keep +/-10 primary and use +/-5 and +/-30 only as compact checks.

## F9. The implementation remains scratch-only

No production code, rally segmentation, contact detector, raw-candidate promotion, learned model, dynamic programme, or threshold sweep belongs in this pass. The work should be one small analysis module, focused helper tests, checked compressed evidence, and a short report with roughly five examples.
