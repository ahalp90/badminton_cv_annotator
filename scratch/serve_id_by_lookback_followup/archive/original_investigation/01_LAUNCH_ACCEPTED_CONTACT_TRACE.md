> ARCHIVED 2026-08-12: historical launch brief. Current position: `../../HANDOVER.md`.

# Launch the accepted-contact opener investigation

Load `plan-and-execute`, `worklog`, `external-delegate`, `serena-pyrefly`, `style-py`, `write-clearly`, and `de-yuck`.

This is a fresh `gpt-5.6-sol` high-effort coordinator session. Ask only when a missing answer changes behaviour, numbers, scope, or safety.

Read at startup:

- `scratch/serve_id_by_lookback_followup/00_SHARED_CONTRACT.md`
- the Resume block in `scratch/serve_id_by_lookback_followup/worklog.md`
- `.github/AGENTS.md`
- `.codex/context.md`

Verify the actual Git branch, tip, and worktree before trusting the recorded baseline. Ensure Serena/Pyrefly is visible before semantic navigation. Report if it is not.

## Adjudicated intake

Verified facts:

- PR #82 stores each span's accepted contacts in chronological order
- The primary population is 239 one-to-one rallies across three fixtures
- Current first impulses score as 119 contact 1, 19 contact 2, 4 later, and 97 unmatched at ±10
- PR #82 supplies the recurrence-clean pre-contact path, robust distance trend, and fixed 0.05-BH incoming rule

Settled rulings:

- Run one GT-free search on every rally
- Keep each existing span as a coarse envelope
- Search accepted impulses only
- Move forward past a contact only when usable post-contact evidence says it lacks outgoing motion
- Stop the primary search with `not enough shuttle trajectory to tell` whenever evidence is unavailable
- A continue-past-unknown run is sensitivity-only
- For a credible contact with incoming motion, trace the incoming path backwards before inferring a missing serve
- Reinspect a connected earlier accepted origin using the same rules
- Infer an unshown serve only after a usable trace finds no accepted origin
- Do not invent an exact missing-serve frame
- Keep production serve-start, raw impulses, rejected impulses, production edits, and threshold sweeps out of scope

Leads requiring measurement:

- Some of the 97 unmatched first impulses may be junk that a post-contact outgoing check can skip
- Some later accepted contacts may connect backwards to an accepted serve
- Some usable incoming paths may have no accepted origin and support an implied unshown serve

Open question:

> What is the smallest fixed, GT-free continuity rule that connects an earlier accepted contact's outgoing path to the incoming path of a later accepted contact?

## Correct search state machine

For accepted contacts `A0 < A1 < ... < An`:

1. Start at `A0` and inspect post-contact motion.
2. Insufficient post-contact evidence returns unknown and stops the main search.
3. Usable post-contact evidence without outgoing motion marks the contact as junk and advances to the next accepted contact.
4. Outgoing motion makes the contact credible. Inspect its pre-contact motion.
5. Insufficient pre-contact evidence returns unknown.
6. Measured absence of incoming motion returns a visible serve at that accepted frame.
7. Incoming motion starts a backwards trace through earlier accepted contacts.
8. A connected earlier origin becomes the current contact and is inspected with the same before/after rules.
9. A usable trace reaching its observable boundary without an accepted origin returns an implied unshown serve before the current visible contact.
10. A trace broken by unavailable evidence returns unknown.

The sensitivity run may continue forward after Steps 2, 5, or 10. Label every such result separately.

## First task: corrected narrow code sweep

Read `findings.md` and `decisions.md`. Then inspect only:

- `scratch/serve_start_trajectory_exploration/trajectory_features.py`
- the smallest parts of `experiment_data.py` and `analyse_serve_trajectory.py` needed for accepted rows, path masks, scenes, and current measurements
- focused tests for those helpers

Use Serena/Pyrefly for symbols and references. Pair it with text search for dynamic or stored-field use.

Answer before implementation:

- Can the existing pre-contact helper be mirrored safely for post-contact runs?
- Which current path eligibility and contact-gap checks apply symmetrically?
- What data marks why a usable run ended?
- How can outgoing and incoming runs be compared in time and bbox-relative space without GT?
- Can the origin search revisit earlier accepted contacts without contradicting the forward junk verdict?

Use one Luna Max delegate at a time only when a question is mechanical. Verify every result locally. Stop any delegate that broadens into serve-start, raw-contact promotion, or production redesign.

## Planning gate

Update `evidence.md`, `mechanisms.md`, `runs.md`, and `worklog.md` after the sweep. Turn Q1 in `decisions.md` into concrete options with real symbols, shapes, and fixed numbers. Ask the user to resolve any behavioural choice before writing experiment code.

Then write the executable runbook, OUT-list, focused gate commands, reference checks, and exact commit messages. Commit messages are already authorised under the shared contract's writing limits.

## Later execution and audit

Implement only the approved scratch analysis and focused tests. Produce checked compressed row evidence and a short conclusion-first report. Apply the rule to all 239 rows before reading GT labels.

Report fixed, damaged, unchanged, and unknown transitions. Show the 97 unmatched rows as a separate slice. Include junk skips, backwards origins, implied serves, final visible-contact ordinals, and roughly five representative cases.

Apply the shared contract's writing and voicing gate to every user-facing document. Put the main ideas in a short top-level introduction, then reveal the technical evidence progressively. Do not give abandoned interpretations extra prominence unless they will matter later.

Before finalising, give the compact evidence and proposed conclusion to the declared Claude Opus and Gemini 3.1 auditors. Ask specifically about GT leakage, ordinal interpretation, threshold tuning, denominators, indirect truth use, and overclaiming. Resolve substantive findings.

Stop after the short report. Do not propose production architecture.
