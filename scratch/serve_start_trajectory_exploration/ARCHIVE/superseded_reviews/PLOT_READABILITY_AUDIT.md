# Plot readability review

Use this when reviewing the final plots.

The basic test is:

**Can someone who knows the project reasonably well understand what this plot is saying without having to reverse-engineer the analysis?**

Look at each plot before relying on the surrounding explanation.

Ask:

- What question does this plot answer?
- Is it obvious what each category means?
- Is it obvious how many rallies or contacts the result is based on?
- If the groups use different denominators, can I see that immediately?
- Are counts shown where percentages alone would hide something important?
- Can I tell the difference between “we had no usable evidence” and “the evidence said no”?
- If two results use different groups of rallies, is it clear why?
- Does the title describe the actual comparison in ordinary language?
- Does the plot make the main finding easy to see?
- Could the way it is drawn lead a reasonable reader to the wrong conclusion?

Use ordinary labels rather than implementation names.

The tolerance results should make **±5, ±10 and ±30 base-30fps frames** easy to compare. It should also be clear that ±10 is the usable baseline, ±5 is the stricter view, and ±30 is the sanity check.

For the inpaint comparison, the reader should be able to tell exactly what changes between the two versions. Don't expect them to know names such as `recurrence_clean` or `producer_original`.

Be particularly careful with 239, 249 and 292. They answer different questions. Don't put them next to one another in a way that suggests they are interchangeable measurements of the same thing.

Where motion evidence is involved, make the amount of **available evidence** visible. A plot should not make “no usable path” look like a negative classification.

If a plot needs a long paragraph just to explain its axes, categories or denominator, simplify it. A small table may be better.

As a final check, show each plot to a fresh reviewer and ask what they think it means before giving them the detailed explanation. If their reading differs materially from the intended one, change the plot.

The goal is simple:

**The plots should be easy to understand and difficult to misread.**