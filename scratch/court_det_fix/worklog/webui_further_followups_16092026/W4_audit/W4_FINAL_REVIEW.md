# What the new court-detector tests tell us

## Where I would leave the project

**I would keep developing this detector, but I would not make all these changes the default.** The tests help separate two problems: failing to propose a useful court, and choosing the wrong court from the ones already proposed.

There are useful results here. On one difficult Amateur-2 frame, changing which detected lines went into scoring picked a much closer court without generating any new courts. Combining two scores across seven GX frames also avoided a wildly wrong result. Neither result, however, shows that the new choice is correct in the image.

Two specific rules are not worth taking further as replacements. The new rule for keeping line assignments loses a close court on the already-good Amateur-3 example. The rule that picks one court by its median line score across frames picks a badly wrong court. That is a reason to drop those rules, not the whole detector or every approach that uses several frames.

**One claim needs to change:** the better seven-frame calculation chooses an existing GX court that an earlier image review rejected as skewed. It shows a better way to choose among these options. It does not show that the detector has found a correct court.

There is no new job to run for this handover. I already ran the one calculation the packet suggested. [Results](results/README.md)

## A few things to know before reading the numbers

A **candidate** is a proposed court. The detector may propose many courts, filter some out, keep a smaller set, and then choose the one with the best score. A useful court can disappear at any of those steps.

The errors below measure distance from a reference court. For each proposed court, the calculation measures all four corner distances and keeps the largest. It allows the same court to be labelled 180 degrees the other way round. All distances use the project's **960 × 540** working image, not the larger source image.

Smaller errors mean closer to that reference. They do not automatically mean a court is good enough. The project has no single pixel cutoff that establishes that. Also, the GX reference used in C2 differs from the annotations used in the seven-frame test; their errors should not be treated as the same measurement.

These are development examples, not a test of how well the detector works on new videos.

## Three findings that affect the decision

### 1. The new way to keep line assignments helps Am2 but hurts Am3 — L1

L1 asks whether the detector throws away useful line assignments too early. An assignment says which detected line represents which court marking. L1 compares the old rule, which keeps the highest-scoring assignments, with a new rule. The new rule keeps the first 256 by score, then fills another 256 slots with assignments spread across different positions and sizes.

**The number of choices stays the same.** Both rules keep 512 assignments in each direction. That gives 512 × 512 = 262,144 proposed courts for each tested pair of directions. Both then use the existing geometry and player checks, followed by the same limit of 256 courts per pair.

The clearest results are:

| Example | Closest court left with the old rule | Closest court left with the new rule |
|---|---:|---:|
| Am2, frame 28019, direction pair 15 | 81.2626 px | **7.5291 px** |
| Am3, frame 0, direction pair 43 | **4.2709 px** | 23.2484 px |

These numbers are from **after the filters and the 256-court limit**. Am2's closer court survives both. Am3 still loses its close court. These are actual proposed courts, not estimates of how good a fit might be possible. [L1 table and source](EVIDENCE_INDEX.md#l1)

The reference identifies the closest court left for evaluation. It does not pick the detector's winner. L1 did not test which court would win the final image scoring.

This was a test of six chosen direction pairs, not a run over every pair to find a final court. The Am2 and broadcast rows also lack the saved older results needed to check that the old rule exactly repeats its earlier run. They remain useful comparisons made by the same trial script, but I did not independently rerun them.

**I would leave this particular rule here.** It found something useful on Am2, but the Am3 loss is a good reason not to spend another run extending it to every pair. Other ways to keep a broader range of assignments remain possible.

### 2. Changing the scoring inputs can help without generating new courts — L2

L2 separates two changes that had previously been mixed together. One changes which courts reach the final choice. The other changes which detected line fragments are used to score those courts.

The script calls the original set of courts **G0** and the set made after filtering line observations **G1**. It calls scoring with the original observations **S0**, and scoring with the filtered observations **S1**. Within each set, S0 and S1 use the same courts, in the same order, with the same rules about which courts can be chosen and the same saved paint-profile scores. Only the score based on detected line fragments changes. [L2 code and table](EVIDENCE_INDEX.md#l2)

That makes this result useful:

| Example, using only the original courts | Old scoring inputs | Filtered scoring inputs |
|---|---:|---:|
| Am2, frame 150: error of the chosen line-score winner | 140.9044 px | **12.4405 px** |
| Am3, frame 0: error of the chosen line-score winner | 5.7619 px | **4.2709 px** |

**Those closer courts already existed.** The change helped the score choose them. It did not generate them. On both examples, the separate paint-score winner stayed the same.

Am2 frame 28019 shows the other problem. None of the original 256 courts passes the final checks, under either set of scoring inputs. The set generated from filtered observations includes courts that pass those checks. With that set, the old scoring inputs choose a court at 9.0005 px; the filtered inputs choose one at 11.3209 px. So the gain there comes from having different courts available, not from rescoring alone.

Adding more options can also make the choice worse. On Am3, combining G0 and G1 makes the old scorer pick a court at 11.8394 px rather than 5.7619 px, even though the closer original court remains available.

I would keep the result from rescoring the original courts as a useful lead. I would not say that changing both the proposed courts and their scoring makes the detector better overall.

The code preserves the comparison well enough to answer that question. Two limits still matter. Its checks against earlier runs confirm the same winning IDs, not every saved score. And its combined set has 512 entries with source IDs; some may describe the same physical court. A different ID alone is not evidence of a better court.

### 3. Combining scores across frames makes a better choice, but not a newly correct court — L3

L3 asks whether several views of the same court help choose a single result. It uses 30 proposed courts from GX frames 0 and 5, and scores them on seven saved frames. All 30 pass the stated checks on all seven frames. The method that chooses separately for each frame and the method that chooses one court across frames get **the same 30 options**.

The courts came from the automatic rankings, not from reference labels. For a court scored on another frame, the code uses that target frame's image, line fragments and player observations. It adds the reference-error measurements only after choosing the winners. [L3 code and data](EVIDENCE_INDEX.md#l3)

The first shared rule takes each court's median line score—the middle of its seven scores—and picks the highest. It chooses `106:93818`, a badly wrong court.

The packet then suggests a different calculation, using both the line score and the paint score. I ran that calculation in the original review and reran it for this rewrite. The results match.

#### How that calculation chooses a court

On each frame, rank all 30 courts by line score, with the best in first place. Rank them again by paint score. Add the two ranks for each court. Then average its seven totals. The lowest average wins.

Tied scores use the full court ID in ascending text order, as the packet specifies. No frames, candidates or weights were changed.

The winner is the existing frame-0 court **`22:4579`**. Its seven combined ranks are `4, 10, 23, 17, 25, 18, 22`. They sum to **119**, so its average is **17**. The next court, `22:4598`, averages **19**. [Full calculation](results/l3_rank_sum.json)

| Way of choosing | Median corner error over the seven frames |
|---|---:|
| Choose separately on each frame, using line score | 501.7482 px |
| Choose one court by median line score | 799.3567 px |
| Choose one court by combining line and paint ranks | **14.0689 px** |

For the first row, each frame uses its own winner. The other rows use one fixed court on every frame. The calculation uses the already-saved corner errors; it does not measure those corners again.

**This is a real improvement in the choice among these 30 entries. But `22:4579` was already the frame-0 line-score winner, and an earlier image review rejected it for shear: the court was skewed.** The calculation recovers that existing choice. It neither creates a better court nor changes the earlier judgement about its shape.

I would describe the result this way:

> Using line and paint ranks across seven GX frames picked the existing court `22:4579` instead of a wildly wrong alternative. That is useful, but the chosen court was previously judged skewed. This test has not shown a newly correct court.

That is the small correction the result needs. There is no reason to discard the calculation because the original idea of “success” was too generous.

#### What the seven frames do and do not tell us

The frames are 0, 5, 689, 5111, 5766, 77876 and 86088. They span roughly 24 minutes. The saved image-alignment results support a similar view at those sampled times. They do not show that the camera stayed unchanged throughout the gaps. I did not view the alignment images myself.

The code also checks that scoring on the original frame repeats saved numbers for one candidate from each origin. That is useful, but it is not a check of every original-frame score. Neither limit turns this into an invalid comparison; they limit what we can claim from it.

## The earlier explanation is now corrected — C2

C2 checks two claims that affected how the project understood its failures.

### Looking at one direction pair gave the wrong impression of the wider result

For GX0, filtering the observations makes the closest combination from direction pair 143 much better. But looking across all saved per-pair shortlists gives the opposite result.

I recalculated the four errors from the saved corner coordinates:

| Set being examined | Before filtering | After filtering |
|---|---:|---:|
| Combinations from direction pair 143 | 34.326758 px | **12.577456 px** |
| All saved per-pair shortlists, before the final overall limit | **7.662057 px** | 9.035950 px |

The first error falls by **21.749302 px**. The second rises by **1.373893 px**. The saved closest courts in the second row both come from pair 22, not pair 143. The baseline combination in the first row was not itself a saved shortlist entry. [C2 calculation](results/c2_check.json)

The checking code really does loop over every saved matched pair's shortlist. Its saved counts are 140 matched pairs in each version, with 30,001 courts before filtering and 19,763 after filtering. “All” here means those shortlists, not every possible court before any limit. I checked the code and the four saved examples; I did not rerun the large search. [C2 source](EVIDENCE_INDEX.md#c2)

This also explains why L2 gives a different filtered GX number, 10.096 px: L2 looks **after** the overall limit. C2's 9.036 px result is **before** that limit.

### The duplicate that was kept is 29665, not 30886

In the Am3 example, assignments 29686 and 29665 name the same detected lines: `(7, 197, 32, 3, 19)`. Assignment 29665 has the higher score: **0.784450**, compared with **0.657914**. The matcher sorts by score first, then keeps the first row for each assignment. So **29665** is the copy it keeps. Assignment 30886 describes a different set of lines.

“Kept as the duplicate's replacement” does not mean “reached the final 512.” The saved rank of 29665 is 1936, counting from zero, so it still misses that limit.

I checked the assignment IDs, scores and the code's ordering. I did not reconstruct Am3's reported corner errors or recalculate its ranks across the full set. Those errors vary one direction's assignment while holding the other fitted direction fixed; they are not errors for four final selected courts.

## The other corrections leave the right questions open

The recent wording no longer treats an unrun test as a failed idea. It leaves room to test other direction choices, several-frame scoring and lens distortion. [Corrected reports](EVIDENCE_INDEX.md#corrections)

In particular, a small **lower bound** on error does not show that the detector achieved that error. The earlier GX direction check measured bounds, not generated courts. The two distortion checks were inconclusive; they did not establish a reliable upper limit on distortion. Warnings about line spacing or ambiguous geometry also remain warnings, not tested rules for rejecting courts.

The later GX pilot does not complete the separate, still-unrun Amateur-2 temporal test. The failed older Yellow scoring rule does not settle that test either.

## What I can say about the images

**I did not inspect the court images in W4 or in this rewrite.** I cannot give the new Am3 or broadcast winners a fresh visual judgement. Earlier judgements remain earlier judgements, and a smaller corner error does not overwrite them.

The W2 image review reports an important counterexample. A wrong Am2 court got long, connected runs of bright pixels from the raised net. But a court already judged good also borrowed some support from a white sock and a crossing sideline. Some tests for its two markings used the same source pixels.

That is why I would not adopt a blanket rule that rejects a court whenever markings share pixels, or that demands long connected runs from every marking. The known good example defeats that simple fix. It does not show that every possible scoring change is futile. The detailed pixel return covers four intervals; I have not treated it as a new check of every line in every example. [W2 report and limits](EVIDENCE_INDEX.md#w2)

## What the tests support

The table includes conclusions we should avoid. They are not all claims made by the repository's authors.

| Conclusion | Does the evidence support it? | What I would say instead |
|---|---|---|
| C2 fixed the GX explanation and identified the right duplicate. | **Supported** | Pair 143 improves while the wider saved set worsens; 29665 is the duplicate kept. |
| The L1 rule is a better replacement. | **Contradicted** | It helps the tested Am2 pair but loses a close court on Am3. |
| L1 improves the detector's final chosen court. | **Not measured** | It tests what survives on six direction pairs, not the final winner. |
| L2 separates changes to available courts from changes to scoring. | **Supported** | Some gains come from scoring existing courts differently; others need different courts. |
| The L3 median-line result rules out useful several-frame scoring. | **Contradicted** | Combining line and paint ranks makes a much better choice on the same data. |
| The rank calculation found a newly correct court. | **Not measured** | It chose an existing court previously judged skewed. |
| W2 establishes a rule to reject all courts with shared or broken-up pixel support. | **Contradicted** | Its known good court would be a counterexample. |
| Untested direction, temporal and distortion ideas remain open. | **Supported** | These tests only settle the specific questions they measured. |

## Where this review ends

I would keep the useful scoring results, leave the failed L1 and median-line rules as recorded results, and correct the claim about the rank-sum winner. This handover does not ask for another reviewer, a broader test programme or a new remote run.

The colleague's note is in [HANDOVER.md](HANDOVER.md). The code and commands are in [HOW_TO_RUN.md](HOW_TO_RUN.md).

### Version and checks

This review concerns `fix/court-det` at **`b36402f1c994f2b000044589cdb6b001f190d0dd`**, “Preserve W2 pixel review evidence,” committed **17 September 2026, 06:23:50 UTC / 16:23:50 Melbourne time**. It was the latest commit when the original W4 review checked the branch. This rewrite uses the same version; it does not claim to have checked the branch again.

The original review read the relevant source and small result files through GitHub. The attempted archive read failed. The included numerical inputs were copied from the returned fields; they are real saved data, not made-up examples, but they are not byte-for-byte downloads of the original files.

For this rewrite I used the supplied pack and reran its Python calculations. The saved numbers and rankings match. I did not run the detector, use a remote host, regenerate the large court sets, inspect court images or change the repository. Missing large local files are an access limit, not a request to put them in Git.
