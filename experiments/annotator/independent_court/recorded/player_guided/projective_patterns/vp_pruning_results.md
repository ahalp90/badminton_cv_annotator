# Automatic directions recover the approved GX5 court

The new selector recovers the exact previously approved court in frame 5 of
the GX amateur video (GX5). That court still fails the existing floor-evidence
gate, so no GX court is emitted.
Four of nine scored cases are accepted by the detector, including three
ShuttleSet scenes. User inspection confirms usable GX5 and Amateur-2 frame 150
geometry, but finds substantial marking errors in several other panels,
including detector-accepted courts. Production is unchanged.

The larger aim is to recover court geometry across varied badminton videos
for automatic annotation. This experiment improves proposal generation;
the evidence checks still prevent automatic use of the recovered GX5 court.

## What was tested

The question was whether shared line directions could reduce the existing
rectangle search while preserving useful court proposals. Two population arms
used the same 47 cached views: seven GX frames, 20 other amateur frames and
20 ShuttleSet scenes from videos 03 and 21. Nine cases then ran through the
unchanged scorer and gates. These are development comparisons. ShuttleSet uses
existing median composite views, not raw single frames. Its two unverified
references were excluded from scoring. No new annotation was needed.

The estimator intersects observed lines and includes directions at infinity.
It retains up to 16 vanishing-point hypotheses using 1.5-degree line agreement.
The existing proposal families keep their 32-line caps and all 150 court
templates. Player checks, floor evidence, camera checks and acceptance remain
unchanged. Settings are shared across videos.

Ranking directions by supporting-line count failed on GX5: all 16 slots filled
with variants of the near-vertical direction. Useful alternatives existed but
were discarded. This arm fell back to the original random sampler on 27/47 cases.

The second arm prioritises directions that explain observations left uncovered
by earlier choices. It produces a nonempty pruned search on all 47 cases.
The search reaches its 16,384-rectangle budget on 29/47 cases. This budget is
four times the original 4,096 sample.

On GX5, pruning reduces 133,937 convex rectangles to 21,856. Selection and the
existing area gate retain 16,271. Both the approved GX5 seed and the known
Amateur-2 seed survive. Their saved corners reconstruct with zero difference
under the existing 1e-5 native-pixel check.

Both known seeds appear within the first 4,096 selections. Their recovery
therefore does not require the larger budget. A fixed random draw of 16,384
still misses GX5 and retains Amateur-2. These are seed-membership checks;
equal-budget downstream scoring has not been compared.

## What survives scoring

The table covers nine cases from the second, coverage-based arm. It reports
the largest Euclidean error across four reference corners,
scaled to 1280×720. “Closest generated” uses labels after generation, before
player, floor and camera gates. “Top retained” is emitted only when accepted;
a dash means no retained court. Pixel error alone does not establish usability.

| Case | Closest generated, px | Top retained, px | Detector decision |
| --- | ---: | ---: | --- |
| GX, frame 0 | 16.65 | — | Unsupported |
| GX, frame 5 | 3.37 | — | Unsupported |
| Amateur-2, frame 150 | 7.12 | 7.72 | Accepted |
| Amateur-2, frame 28019 | 8.75 | — | Unsupported |
| Amateur-3, frame 0 | 13.87 | 13.87 | Ambiguous |
| ShuttleSet 03, scene 17 | 7.94 | 8.85 | Ambiguous |
| ShuttleSet 03, scene 19 | 5.38 | 8.91 | Accepted |
| ShuttleSet 03, scene 16 | 8.08 | 9.26 | Accepted |
| ShuttleSet 21, scene 20 | 6.91 | 9.24 | Accepted |

Both previously accepted controls remain accepted: Amateur-2 frame 150 and
ShuttleSet 03 scene 19. The closest GX5 proposal has the earlier approved
geometry and passes player checks. The floor gate measures support from observed
line fragments. One line family has support 0.431 against
the existing 0.55 minimum, and two distinct supporting lines against the
required three. The floor gate rejects it before camera evaluation.

## User inspection and next diagnostic

The [panel-specific judgements](vp_pruning_visual_judgements.md)
record all the user's observations and uncertainties. GX5 looks good and
Amateur-2 frame 150 is perfectly usable. GX frame 0 is a gross regression:
its back boundary lands on a long-service line, and its right doubles sideline
lies between the real doubles and singles sidelines.

Several ShuttleSet panels place the near lines well but substantially inset
or progressively overshoot the far horizontals. Scene 19's closest generated
court is usable with noticeable defects; that approval does not apply to its
top retained court. Scene 20's top retained court is basically fine with a
significant far-end skew/corner-placement issue. Other detailed observations
do not establish blanket pass or fail judgements.

Minor paint-edge offsets were acceptable in several examples. The larger
marking errors remain despite small corner errors and detector acceptance.
Generation is resolved for the approved GX5 example, but that result does
not establish visually accurate recovery across cases.

The subsequent [marking diagnosis](marking_diagnosis_results.md) examines stripe
identity and drift. The later [spacing matcher](axis_matching_results.md) tests
line patterns with supplied directions. These follow-ups preserve the acceptance
gates and the visual judgements reported here.

Both 47-case population arms and all nine scoring runs completed, exit 0.
The tracer verifies identical direct and instrumented final outputs. Seven
synthetic tests, scoped lint, whole-project types and explicit script types pass.
Two independent technical reviews checked the selector, accounting and scoring
adapter. Recording defects were corrected; they did not change selected courts.

The [methods](methods.md) record settings and validation. Original measurements
were retained when fallback accounting was corrected.
Round-robin allocation favours rectangles shared by several direction pairs.
This and the missing equal-budget scoring comparison limit broader claims
about coverage, ranking and runtime.
