Please add a fresh red-team review of the **final outputs**, especially `report.md`.

The main question for that review is:

**Can someone who knows this project reasonably well, but has not followed this investigation closely, read the report once and understand what we found?**

That matters as much as getting the numbers right.

I want the report to start with an approximately **800-word summary** that tells the important story before going into the detailed analysis.

That summary should answer the questions I originally came in with, in normal language.

In particular, I want to understand:

- What do **292, 249, and 239 rallies** each represent?
- Why do we use different groups for different parts of the analysis?
- What exactly is the **earliest accepted contact**?
- Is that contact already required to look like a serve? If not, what actually allows it through?
- How often does that earliest contact line up with the ShuttleSet serve, the first return, a later stroke, or no plausible stroke at all?
- Show that at **±5, ±10, and ±30 base-30fps frames**.
- Treat **±10 as the main usable baseline**.
- Treat ±5 as the stricter, ideal timing view.
- Treat ±30 as the sanity check for whether we are even looking at the same physical stroke.
- Don't spend time on ±2. It is tighter than I consider useful here and is inside the region where manual GT timing error itself becomes relevant.

For the anchors that do **not** match at ±10, I especially want to know what happened next.

For example:

- Was the actual serve detected as the second or third accepted contact?
- Was the serve missing, but the first return detected?
- Or was the early anchor simply an ordinary contact candidate that does not correspond to a real stroke?

That distinction is central to the point of this investigation.

The report should make clear that the earliest anchor comes from the ordinary contact detector. It is **not a serve detector**. If it only had to satisfy things like shuttle impulse, wrist proximity and the normal contact filtering/suppression rules, say that plainly.

Please also keep these failure modes separate:

- rally segmentation failed;
- the chosen contact does not correspond to GT;
- there is no usable incoming-motion path;
- there is a usable path and it says “serve”;
- there is a usable path and it says “return”;
- the server-attribution step is right or wrong.

Those are different problems. I don't want them folded together into one accuracy number.

For merged rallies, use the **239 one-to-one cases** when the analysis really requires one predicted rally to correspond to one ShuttleSet rally.

But don't make the other numbers disappear.

Explain them simply. For example:

**We have 239 rallies where the prediction and the ShuttleSet rally match one-to-one. Those are the rallies we use when an analysis needs a single unambiguous rally match.**

Keep the **249 covered rallies** as a sensitivity check so we can see what happens under the current `COVERED` definition.

If you show a result over all **292 GT rallies**, make clear that it is an end-to-end result and therefore includes rally-segmentation failures as well.

For the motion analysis, tell me first **how often usable motion evidence exists**. Then tell me how well the rule works when that evidence exists. I don't want “no usable path” quietly treated as though the classifier looked at a good path and decided it was a serve.

For the TrackNet/InpaintNet comparison, make the comparison genuinely like-for-like. Hold the motion threshold fixed between the two versions. The only thing that should change in that comparison is whether producer-marked inpainted points are also excluded.

Please also break out the important results by `sset_01`, `sset_15`, and `sset_21` where that helps us see whether one video is driving the overall result.

## How I want the final report to read

Please write it like one technically careful person explaining the result to another technically literate person.

Use short, direct sentences.

Introduce a number, explain what it means, then explain why it matters.

Don't make the reader unpack several qualifications inside one sentence.

Avoid project-management and audit language in the report itself. I don't want the reader hearing about “gates”, “workstreams”, “conditioning subsets”, “evaluation regimes”, “artifacts”, “protocols”, or the machinery used to produce the investigation unless it is genuinely necessary to understand the result.

The checking process can be rigorous behind the scenes. The report should still sound human.

Please run the finished prose through my existing **`@write-clearly` and `@de-yuck`** skills. Use them to catch jargon, bureaucratic phrasing, unnecessary abstraction, long nested sentences and wording that makes the reader decode the process before learning the point.

## Fresh-reader check

Once the report is finished, give it to a fresh reviewer who has **not** been involved in the investigation.

For their first read, give them the report and the original questions, but not the plan, worklog, decisions file or implementation notes.

Ask them whether they can explain, from the report alone:

- what 292 / 249 / 239 mean;
- what the earliest accepted contact really is;
- what the ±5 / ±10 / ±30 results say;
- what is happening in the unmatched-anchor population;
- how often motion evidence is actually available;
- what the inpaint comparison changes;
- whether the server-identification idea helps;
- what the main caveats are;
- and what we should do next.

If they have to guess what a denominator means, work out why two numbers differ, inspect another file to understand a category, or misunderstand an ordinary contact candidate as a serve-specific detection, please fix the report itself.

Don't solve that by adding another explanatory side document.

Finally, independently check the headline numbers against the regenerated row-level outputs. I want the report to be both **easy to understand and hard to misinterpret**.

Please design the exact red-team procedure yourself around those goals. I don't want another large procedural framework added to the plan.

## Follow-up: Plots
The same fresh-reader standard applies to the plots. When you reach the final-output review, load `PLOT_READABILITY_AUDIT.md` and use it to red-team every plot that appears in or supports the report. Treat problems with confusing framing, labels, denominators or implied comparisons as report problems, not cosmetic issues.
