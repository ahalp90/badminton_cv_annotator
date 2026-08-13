# Check the prepend/refit comparison before finalising the report

Please do a focused verification of the **server-attribution comparison only**.

There may be a mismatch in what the two headline methods do when incoming-motion evidence does **not** fire:

- the direct motion method appears to default to the **earliest-contact player**;
- the prepend/refit method appears to default to the **ordinary alternating fit** unless motion fires.

If that is true, then comparing their overall scores directly may mix together two effects:
1. the effect of the prepend/refit itself; and
2. the effect of using a different fallback on the many rallies where no motion trigger occurs.

## What to verify

Use the current local code and regenerated row-level outputs. Do not assume the observation above is correct.

Confirm:

1. What each method returns when motion **does not** fire.
2. What each method returns when motion **does** fire.
3. How many primary 239 rallies fall into each group.
4. Whether the published `163/239` versus `127/239` comparison is genuinely measuring the prepend/refit effect, or is partly explained by different fallback behaviour.

If the fallbacks differ, add an **apples-to-apples comparison** that keeps the same default server estimate and changes only what happens on motion-triggered rallies.

For example, if the direct method uses earliest-contact player as its default, compare:

- earliest-contact player + direct motion correction;
- earliest-contact player + prepend/refit correction on the same triggered rallies.

Please recompute the exact numbers locally rather than relying on numbers from this note.

## If a real difference remains after that

Only then inspect the small set of motion-triggered rallies where direct inference and prepend/refit disagree.

For those cases, determine whether the alternating fit is:
- overriding a correct local inference because of later contact votes;
- tying/abstaining;
- or behaving unexpectedly for some other reason.

Do not change the motion rule or contact detector as part of this check.

## Report impact

If the current comparison is misleading, fix the report and main server plot so they make the fallback behaviour explicit.

Keep the old-fit-based prepend result if it is still useful, but label it clearly rather than presenting it as a direct `163 → 127` degradation caused by refitting.

Please keep the write-up short. I only need:

- whether the comparison issue is real;
- the corrected like-for-like numbers, if needed;
- what happens on the triggered disagreements;
- the minimal report/plot changes required.

Do not optimise toward any expected result. The purpose of this pass is to make sure the comparison is logically fair and accurately described.
