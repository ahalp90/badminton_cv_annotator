# Readability feedback on the current serve-trajectory report

The report is much more technically complete than the earlier version. The main problem is no longer missing information.

The problem now is **cognitive load**.

A reader who is already tired or overloaded still has to hold too many populations, caveats, thresholds and method names in working memory before the main result becomes clear.

**This applies to the whole report, not just the opening summary.**

I am **not** asking for less technical detail. Keep the technical detail, the caveats, the per-video breakdowns, the diagnostic plots and the reproducibility information.

What I am asking for is **progressive disclosure**.

Ease the reader into each layer of detail:

- say what question a section is answering before introducing its machinery;
- give the plain-language result before the technical definition;
- introduce one denominator before adding another;
- explain why a distinction matters before asking the reader to remember its name;
- lead with the finding, then show the evidence, then add qualifications;
- keep the deeper technical material, but put it after the reader has a reason and a mental model for it.

The report should feel like a **ramp**, not an easy introduction followed by a sudden wall of dense technical material.

A reader who keeps going deeper should be able to build on what they already understand rather than repeatedly reconstructing the experiment from scratch.

Please revise for a reader who knows the project reasonably well but does **not** have spare attention.

## Put the result first

The report should open with a short **Bottom line** section of roughly 120–180 words.

That section should make these points almost impossible to miss:

- The released alternating server fit gets **124/239** one-to-one rallies right.
- Simply using the player at the earliest accepted contact gets **152/239** right.
- Usable pre-contact motion exists in only **24/239** rallies, so motion is a small correction signal rather than a general solution.
- Using that motion to flip the inferred server raises the result to **163/239**.
- Feeding the inferred missing serve back into the alternating fit does **not** preserve the gain: the inferred-player prepend reaches only **127/239**.
- The bigger upstream problem is contact selection. At the normal ±10 timing tolerance, **97/239** earliest contacts do not line up with a ShuttleSet stroke. In most of those rallies, a later accepted contact recovers the serve or first return.

That is the main story.

A reader should be able to stop after this short section and still have the right mental model.

The longer summary can follow, but it should support this story rather than making the reader discover it.

## Reduce the number of new concepts introduced in the opening

The current summary is accurate, but nearly every paragraph introduces another denominator or qualification:

292 → 249 → 239 → 135 → 19 → 24, plus ±5 / ±10 / ±30, two motion rules, two masks, trend/jitter diagnostics and several server methods.

That is too much to absorb in one pass.

In the opening:

- explain **292 / 249 / 239**;
- use **±10** for the main contact-alignment result;
- say briefly that ±5 and ±30 were checked as strict and sanity views, then leave their full numbers for the detailed section;
- do not introduce the 135-case unique-truth subset until the motion-classification section;
- do not introduce trend-to-jitter statistics in the top-level story;
- do not give rank-2/rank-3/rank-4 recovery counts in the summary;
- do not explain the entire 57 → 31 → 24 path-construction funnel in the summary.

The report already contains those details later. They do not all need to appear twice.

## Use sentences that explain one thing at a time

Some phrases are technically compact but expensive to parse.

For example:

> “135 unique ±10 serve/return anchors”

is harder to read than:

> “For 135 rallies, the first detected contact can be labelled confidently as either the serve or the first return.”

Likewise, prefer:

> “We have usable pre-contact motion in 24 of the 239 rallies.”

over:

> “Usable recurrence-mask evidence exists in 24/239 primary rallies.”

Keep internal names where they are needed for reproducibility, but introduce the human meaning first.

The reader should not have to translate terms such as:

- unique truth
- recurrence-mask evidence
- producer-mask version
- one-to-one population
- 0.05-BH direct rule

before understanding the sentence.

## Make the failed prepend experiment much more prominent

The current report contains an important negative result:

- released alternating fit: **124/239**
- earliest-contact player: **152/239**
- motion-corrected direct server inference: **163/239**
- prepend inferred server and rerun alternating fit: **127/239**

This is one of the most useful findings in the whole investigation.

The report should say plainly:

> The new information is useful when we use it directly. Feeding it back into the old alternating fit mostly loses the improvement.

Do not describe 127/239 merely as a small improvement over 124/239. The more important comparison is **163 → 127**.

This tells us something about how the existing alternating fit consumes the contact sequence.

## Separate “motion helps” from “motion is widely available”

The report does say that motion evidence is scarce, but the distinction should be visually and verbally stronger.

There are two separate findings:

1. **When usable incoming-motion evidence exists, it can correct some server calls.**
2. **Usable motion evidence exists in only 24/239 rallies.**

The 163/239 method is therefore not a trajectory classifier making 239 motion-based decisions. It is:

> use the earliest-contact player by default, then make a small number of motion-backed corrections.

Please say that directly whenever the 163/239 result is introduced.

## Keep the upstream contact problem visible

The 97 unmatched ±10 anchors are not a side issue.

They help explain why the whole experiment is difficult.

The useful simple result is:

- 97 earliest anchors do not match a GT stroke at ±10;
- 49 later recover the serve;
- 36 recover the first return without recovering the serve;
- only a small remainder behave differently.

That means many failures happen **before** the motion classifier.

The report should make this one of the major conclusions:

> Improving which accepted contact becomes the anchor may be more valuable than making the incoming-motion classifier more complicated.

## Simplify the plots in the main reading path

Apply the same overloaded-reader standard to the figures.

### Anchor alignment

The current anchor-alignment plot is broadly useful.

Keep it, but make ±10 visually or textually obvious as the practical baseline. ±5 and ±30 are supporting views.

### Unmatched-anchor follow-up

The main category breakdown is useful.

The accepted-contact rank breakdown is secondary detail. It can move to the deeper section or a table if combining both makes the figure harder to scan.

### Motion evidence and inpaint

This figure currently asks the reader to understand too much at once:

- path availability stages;
- two masks;
- the 239-rally population;
- the 135-rally truth subset;
- incoming/negative decisions;
- classification counts.

Please simplify it.

A reader should be able to answer two questions immediately:

1. How much usable motion evidence do we have?
2. What changes when producer-marked inpainted points are also removed?

If necessary, use two simpler figures or a small table rather than one dense figure.

### Trend and jitter

This is useful diagnostic material, but it is not part of the central story.

Keep it in the detailed analysis rather than making the reader process it on the way to the main conclusion.

### Server attribution

This should probably be the clearest plot in the report.

It should visibly compare at least:

- released alternating fit — 124/239
- earliest-contact player — 152/239
- motion-corrected direct inference — 163/239
- prepend inferred server + alternating refit — 127/239

The current main visual story should make **152 → 163** and **163 → 127** immediately obvious.

The prepend failure is too important to leave as a table-only detail.

## Suggested top-level structure

A simpler reading path would be:

### Bottom line

120–180 words. Give the main result, the failed prepend result, the evidence-availability limit and the upstream anchor problem.

### What are the 292, 249 and 239 rallies?

Explain the denominator issue once.

### Is the first accepted contact actually the serve?

Use the ±10 result first. Put ±5 and ±30 underneath as supporting checks.

### What happens when the first contact is wrong?

Show the 97 unmatched cases and later-contact recovery.

### Can incoming motion help?

First show how rarely usable evidence exists. Then show what it does when available.

### Does removing inpainted TrackNet points help?

One controlled comparison, in plain language.

### Does the inferred missing serve improve server identification?

Show the direct method and the prepend/refit method next to each other. Make the failed prepend result explicit.

### What should we do next?

Short answer: improve anchor/contact selection and evidence availability before adding a richer motion classifier.

Detailed diagnostics, per-video tables, trend/jitter analysis and exhaustive accounting can follow.

## Final standard

The current report is much better at proving that the analysis is careful.

The next revision should be better at **helping a tired person understand what happened all the way through the document**.

Do not solve this by making only the introduction easier and then leaving the rest of the report at the current density.

Every major section should have its own small ramp:

1. **What question are we answering?**
2. **What did we find, in ordinary language?**
3. **What numbers support that?**
4. **What technical details or caveats matter?**
5. **What does this change about the overall interpretation?**

The report can become more technical as it goes deeper. That is fine. What should not happen is a sudden jump from readable prose into paragraphs that require the reader to juggle several new populations, labels and exceptions at once.

On the first read, I should not be trying to remember every denominator or reconstruct the experiment.

I should come away with four clear ideas:

1. The earliest-contact player is already a much better server guess than the old alternating fit.
2. Motion evidence is rare, but it provides a useful small correction when available.
3. Prepending the inferred serve and rerunning the alternating fit largely throws that gain away.
4. A large part of the remaining problem is upstream: the earliest accepted contact is often not the right stroke.

Everything else should make those conclusions more trustworthy, not compete with them.

**Keep the detail. Change the order and pacing so the reader is eased into it.**
