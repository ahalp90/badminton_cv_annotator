> ARCHIVED 2026-08-12: historical mechanism ledger. Current conclusion: `../../report.md`.

# Mechanisms under test

## Approved extension hypotheses

| Mechanism | Status | Supporting evidence | Risk | Next check |
| --- | --- | --- | --- | --- |
| A three-frame recurrence halo retains useful motion while keeping a small recurrence buffer | Verified measurement change | Halo 3 clears 37,139 old halo frames; halo 15 reconstructs production exactly | Opener accuracy remains limited by the search rule | Keep H3 as the local experimental mask, without changing production landing guards |
| Raising `largest_step_ratio` to 8.0 admits jumpy but coherent visible traces | Verified measurement change | 545 pre and 897 post paths lie in the 4-to-8 ratio band | 321 pre and 526 post paths still exceed 8 | Keep 8.0 for this experiment; do not tune further against GT |
| Earliest positive incoming evidence is a better rally anchor than earliest positive outgoing evidence | Rejected as a complete opener | It provides an anchor in 234 rallies | The final rule is correct in 26 rallies and leaves 162 terminal unknowns | Do not use the incoming-only rule as the opener |
| The nearest earlier accepted contact can identify the visible serve | Rejected for ordinary timing | 39 ordinary predecessors have measured no-incoming evidence | Only 3 of the 39 match GT serve at +/-10 | Treat ordinary temporal adjacency as insufficient evidence |
| Measured high-shot state can admit legitimate long serve-return gaps | Supported in a small fixed slice | All 5 admitted predecessors match GT serve, including clip 13 | Five cases are too few for a broad production claim | Retain as a narrow measured exception if predecessor work continues |

| Mechanism | Status | Supporting evidence | Contrary evidence or risk | Next check |
| --- | --- | --- | --- | --- |
| Early accepted impulses can be skipped unless they have credible outgoing motion | Measured, too broad | The rule fixes 15 of the 97 unmatched starts at +/-10 | It rejects the first accepted contact in 218 of 239 rallies | Do not carry this predicate forward as the opener gate |
| The first credible outgoing contact is the opener candidate | Measured, poor | A credible contact is found in 212 rallies | Median selected rank is 3 and the maximum is 24 | Treat late selection as the main failure mechanism |
| Incoming motion classifies the candidate as the first visible post-serve contact | Measured | PR #82 classifies 94 selected contacts as incoming | Only 28 of those selected frames match GT contact 2 at +/-10 | Do not treat the check as enough after the outgoing scan has drifted late |
| No incoming motion classifies the candidate as a visible serve | Measured | PR #82 classifies 18 selected contacts as not incoming | Only 6 of those selected frames match GT contact 1 at +/-10 | Report as a weak minority result |
| The same rule can improve bad starts without damaging correct starts | Rejected | It fixes 16 starts overall at +/-10 | It makes 34 classified starts wrong and leaves another 127 unknown or without a credible contact | Conclude that the fixed sequential rule is not viable |

## Additive correction mechanisms

| Mechanism | Status | Supporting evidence | Contrary evidence or risk | Decision |
| --- | --- | --- | --- | --- |
| Measured `high_shot_oob` state can correct a different visible serve | Supported in a narrow sample | Two interventions fix two starts and damage none | All five states come from one set | Keep as the only correction; require holdout validation |
| Later serve-setup pass after an early gate failure proves the later serve | Rejected | Finds 22 fixes in the old unmatched slice | 138 interventions cause 63 damages and 53 unchanged errors | No-op |
| Continued same-player wrist setup after the early impulse proves that impulse false | Rejected | One of two interventions fixes an unmatched start | The other intervention damages a correct start | No-op |
| Scene boundaries identify an exact later contact | Rejected as an exact-contact rule | Scene bounds improve PR #82 path measurement | A cut bounds visibility but does not prove a stroke frame | Retain only inside the baseline motion measurement |
| Direct local RGB stroke evidence could verify a candidate contact | Unmeasured next input | Source video can show racket and hitting-arm motion | Requires a new contact-verification experiment | Outside this pass |
