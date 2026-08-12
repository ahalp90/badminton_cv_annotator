> ARCHIVED 2026-08-12: original scope. Current conclusion: `../../report.md`.

I want a **small, targeted follow-up experiment to PR #82**.

The hypothesis is:

> Some rallies are assigned to the correct rally span, but the accepted contact sequence starts with a spurious early contact. The real first contact then becomes contact 2 and the whole sequence is one step out of phase.
>
> Rather than trying to identify the rally start prospectively, can we start from a later, more credible contact and walk backwards through plausible preceding contacts until we reach the serve?

## Scope

Start from the existing PR #82 investigation on `investigation/serve-start-trajectory`.

Read only what you need from:

- `scratch/serve_start_trajectory_exploration/HANDOVER.md`
- `scratch/serve_start_trajectory_exploration/report.md`
- `scratch/serve_start_trajectory_exploration/trajectory_features.py`

Then inspect only the code needed to understand the accepted-contact rows already used by that investigation.

Do not broadly re-orient across the annotator.

Do not redesign rally segmentation.

Do not modify production code.

Do not invent a new contact detector.

Reuse PR #82's prepared inputs and measurement conventions wherever possible.

## Serena / Pyrefly

Use the `serena-pyrefly` MCP for code navigation and type-aware inspection.

Before relying on it, make sure the working folder for this investigation is visible to Serena. If you create a new scratch folder, add or expose it to Serena as needed rather than silently working outside its indexed workspace.

## Delegation

You may use up to **30 Luna Max mechanistic `@external-delegate` agents** for tightly scoped mechanical investigation.

Use them only where they reduce boring inspection work.

Keep the work **linear**.

That means:

- give an agent one narrow question;
- get its result back;
- integrate or verify it;
- then decide the next question.

Do not spawn a broad parallel swarm.

Do not ask agents to independently redesign the solution.

Do not let delegated work recursively expand the context.

Good delegation targets include:

- tracing where one field comes from;
- checking one subset of rallies;
- extracting counts;
- inspecting a fixed group of failure cases;
- comparing two small outputs;
- checking whether an existing helper can be reused;
- validating a table or plot.

The main thread owns the hypothesis, experimental design and conclusions.

## Independent audit

Before finalising the result, use:

- an **AGY Claude Opus auditor**;
- a **Gemini 3.1 auditor**.

Give them the compact evidence and the proposed conclusion.

Ask them to look specifically for:

- evaluation leakage;
- incorrect interpretation of contact ordinal;
- accidental threshold tuning;
- misleading denominators;
- cases where the claimed backwards reconstruction is really just using ground truth indirectly;
- conclusions stronger than the data supports.

Resolve substantive audit findings before writing the final report.

Do not turn the auditors into additional open-ended investigations.

## Data

Use **all three existing fixtures in full**.

There is no need to preserve a holdout among these three fixtures.

I have the rest of ShuttleSet available as a genuine later holdout.

So for this investigation, optimise for understanding the failure mode across all three fixtures rather than artificial train/test separation.

Still avoid evaluation leakage: ShuttleSet stroke times, ordinals and server labels may score a candidate rule, but they must not be inputs to that rule.

## Experiment

PR #82 found that 97/239 earliest accepted contacts are unmatched at ±10 frames, while later accepted contacts often line up with the serve or first return.

Take those existing accepted contact sequences and test a deliberately simple backwards view.

Given a credible contact at time `t`:

1. Look backwards for the nearest preceding accepted contact within a generous maximum gap, initially 2 seconds.
2. If there is one, treat it as the candidate previous logical contact.
3. If there isn't one, check whether the existing PR #82 pre-contact shuttle-trajectory machinery shows a clear incoming path into `t`.
   - If so, record that as evidence that a logical preceding contact is missing from the accepted sequence.
   - Do not try to hallucinate an exact frame or spatial position for that missing contact yet.
4. Repeat backwards only as far as the existing evidence naturally allows.

Here, "previous contact" means the previous **logical shuttle contact**, not the previous video frame.

The question is:

**Does looking backwards make it easier to identify and discard the spurious early "contact 1" cases that currently shift the sequence out of phase?**

Do not make it more sophisticated until we know the answer.

Reuse existing helpers such as:

- `closest_pre_contact_run`
- `measure_incoming_motion`
- `fit_robust_distance_trend`
- the existing fixed 0.05-BH incoming-motion rule

Do not design another trajectory classifier for this experiment.

## Keep the anchor question small

Do not turn this investigation into "find the perfect rally-end anchor".

Use the simplest credible later contact already present in the existing sequence that lets us test the backwards idea.

If anchor choice clearly dominates the result, document that as the next question and stop.

That can be a separate investigation.

## What to measure

Keep the evaluation narrow.

For the 239 one-to-one rallies, compare the current earliest accepted contact with the candidate sequence start implied by the backwards approach.

Report the GT contact ordinal matched by the current earliest accepted contact:

- contact 1;
- contact 2;
- later;
- unmatched.

Then report the same breakdown after the backwards reconstruction.

I especially want to know:

- how many current unmatched or incorrect starts become contact 1;
- how many currently correct contact-1 starts get damaged;
- how often the method needs to posit one missing preceding contact;
- whether the obvious one-contact phase-shift cases actually come back into phase;
- what prevents recovery when they do not.

Use PR #82's existing ±10 base-30fps alignment as the primary evaluation.

Use ±5 and ±30 only as sanity checks if they are already cheap to produce.

Server attribution is secondary.

Only report it if it falls out cheaply from the experiment. Do not turn this into another server-classification study.

## Gap window

Start with the suggested maximum predecessor gap of about **2 seconds**.

It is fine to inspect the actual gap distribution across all three fixtures.

If useful, compare a very small number of obvious windows to see whether the result is brittle.

Do not sweep many thresholds and select the winner.

## What not to build

I am testing whether a **dumber temporal reconstruction** is enough.

Do not introduce:

- dynamic programming;
- HMMs;
- learned classifiers;
- large scoring functions;
- threshold searches;
- a new rally-start heuristic stack;
- production changes.

If the simple rule fails, show why it fails.

Do not patch every failure with another condition.

## Files and testing

Keep this in `scratch`.

Create the smallest reproducible analysis needed.

Prefer a few small helpers over a large framework.

Add focused tests for any non-trivial new helper logic.

Do **not** run the full repository test suite. We are only changing a few scratch files.

Run only the relevant scratch/focused tests, plus lightweight checks such as Ruff or Pyrefly where useful.

## Useful examples

Inspect a small number of concrete rallies closely.

I would rather have five examples that clearly explain the mechanics than fifty vaguely summarised cases.

For each useful example, show something like:

```text
accepted sequence:
A0 -------- A1 ------ A2 ------ A3
spurious     serve     return    ...

current interpretation:
contact 1    contact 2  contact 3 ...

backwards interpretation:
discard A0
A1 = contact 1
A2 = contact 2
...
```

Use trajectory plots only where they actually help explain a decision.

## Deliverable

Give me a short report answering:

1. **Does backwards chaining materially improve identification of the true first contact?**
2. **Which specific failure mode does it fix?**
3. **How many correct starts does it damage?**
4. **What failure modes remain?**
5. **Is this strong enough to justify one more focused experiment?**

Include the key counts and roughly five representative examples.

Stop there.

Do not propose the production architecture yet.

## Writing

For all writeups, use `@write-clearly` and `@de-yuck`.

Write for a normal person who is cognitively overloaded.

Use progressive disclosure:

- conclusion first;
- then the few numbers that matter;
- then the explanation;
- put detailed mechanics and edge cases later.

Use simple words.

Use short sentences when short sentences will do.

Avoid project-management language.

Avoid phrases like:

- "workstream";
- "stakeholder";
- "roadmap";
- "learnings";
- "key takeaways";
- "action items";
- "moving forward";
- "operationalise";
- "leverage";
- "north star";
- "success criteria".

Use the concrete technical word instead.

The final report should have **no throat-clearing**.

Do not start with what you investigated, what files you read, or how the work was organised.

Start with the answer.
