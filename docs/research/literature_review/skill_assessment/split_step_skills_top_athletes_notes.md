<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# Characteristics of split-step skills of the world's top athletes in badminton

**Verdict**: Skim, do not deep-read. The measurement recipe (foot position and stance width from homography, plus a reaction time anchored to opponent contact) is a useful template for a posture/positional dimension. The findings themselves are thin: three players, mostly null results, and a stats-reporting error.
**Status**: Reviewed by Claude 2026-07-12, not independently re-read by Curtis.

## Identity

- DOI: 10.1371/journal.pone.0316632
- Authors: Hidehiko Shishido, Takeshi Nishijima (meta said "unknown"; corrected from the paper)
- Venue / year: PLOS ONE, 2025
- Scope, one line: Measures where the feet land, how wide the stance is, and how fast the first step comes when a top men's singles player split-steps toward the forehand rear court, from broadcast match video.
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/split_step_skills_top_athletes.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/split_step_skills_top_athletes.pdf
- Note drafted: 2026-07-12

## What they built

A video-only measurement pipeline for the split-step, applied to BWF World Championships 2023 men's singles footage of the top four ranked players.

The method. They picked scenes where the player moves to the forehand rear court. They marked the moment the opponent hit the shuttle toward that corner as the trigger. For each such scene they manually marked the thenar (ball of the foot) of both feet on the frame, then used a homography to map image pixels to court metres. From that they read three things: stance width (Euclidean distance between the two feet, also expressed as a fraction of standing height), foot position on court (and the spread of those positions around each player's own median), and reaction time (frames from the opponent's contact until one foot leaves the ground, at 30 fps so 0.03 s per frame).

A second analysis counted where on court the opponent was hitting from in those scenes (a 3x3 grid, though only 6 cells ever fill), to see if shot placement explained differences in split-step behaviour.

Results, three players (ANTO dropped for too few scenes). Reaction time 0.24 to 0.25 s, near identical across players. Stance width about half of standing height (ratio 0.51 to 0.52). Position spread from each player's median 42 to 55 cm, with NARA tightest at 42.1 cm. NARA faced more shots into his backhand rear corner, which the authors read as giving him more time to reset to base.

## What holds up

The court calibration is properly checked. They estimated 38 known corner points and reported a mean error of 44.8 plus/minus 22.6 mm, range 5 to 92 mm, against a 40 mm line width. So the pixel-to-metre step is trustworthy for court-scale positions. That is the one piece of real validation in the paper.

The static geometry is reproducible in principle. Stance width and foot position come straight from marked points and a checked homography, so a second team could repeat them.

The reaction time figure is consistent internally (0.24 to 0.25 s across three players) and lines up with a prior study of trained juniors, which is at least a sanity check even if the authors misread what it means (see concerns).

## Methodology concerns

<!-- Index first, one line per entry: flag | "short verbatim quote for ctrl+F" | p.N -->

- methodological red-flag | "F (2, 145) = 3.06, p = 0.40" | p.9
- methodological yellow-flag | "ANTO was excluded from the analysis because of its low number of scenes" | p.9
- methodological yellow-flag | "The sample size was not determined in advance" | p.3
- methodological yellow-flag | "The static frame interval was 0.03 seconds" | p.7
- methodological yellow-flag | "denser at the base than those of the other two players" | p.13

**The same F-statistic is reported for three different tests.** p.9. Reaction time, stance width, and position all report `F (2, 145) = 3.06` but with p = 0.40, 0.86, and 0.04. One F value cannot give three different p values on the same degrees of freedom, and F = 3.06 at df (2,145) sits near p = 0.05, so none of the three printed p values is consistent with it. Something in the stats reporting is copied wrong. It does not sink the descriptive numbers, but it means the significance claims cannot be taken at face value.

**Three players carry the whole study.** p.9. Four were chosen, one (ANTO) dropped for having only 15 scenes, leaving three. With n = 3 the between-player tests are underpowered, and any "characteristic of top athletes" is really a description of three men.

**No skill-level contrast inside the study.** p.3, "top 10 world rankings". Every subject is elite, so the paper never shows that these numbers separate strong players from weak ones. The elite framing rests on comparison to an outside cited value, not on a measured gradient.

**Coarse timing.** p.7. Reaction time is counted in whole 30 fps frames, so 0.03 s resolution. A 0.25 s reaction is about 8 frames, and one frame is a 12% swing. Fine for a rough figure, rough for comparing players who sit within 0.01 s of each other.

**Single annotator, no reliability check.** The foot marking and the choice of trigger frame are manual, and no inter-rater or repeat-marking reliability is reported. Only the court homography is validated, not the human marking or the event timing.

## Hand-waves and gaps

<!-- Same index format as methodology concerns. -->

- academic padding | "This suggests the existence of an upper limit for the reaction time in badminton" | p.12
- methodological yellow-flag | "that of 16-year-old male badminton players with experience" | p.12
- academic padding | "denser at the base than those of the other two players" | p.13

**"Upper limit for reaction time" is a large claim from three players.** p.12. They observe elite reaction time equals a cited junior value and jump to a species-level ceiling. That is speculation, not a result.

**The junior comparison undercuts the headline.** p.12. If the world top three react no faster than trained 16-year-olds, then reaction time is not what separates elite from good, which weakens the whole "characteristic of top athletes" claim for that variable.

**Analysis 2 is a post-hoc story.** p.13. NARA's tighter positioning is explained by the opponent hitting more to his backhand rear, giving him time to reset. Plausible, but it is one correlation across three players, and NARA's 63 scenes come largely from a single final, so the pattern may be one match against one opponent rather than a stable trait.

## Relevance to the project (as of 2026-07-12)

Split-step is a genuine candidate NEW dimension for the evidence table, and it sits in one of the profile's named gaps (posture/positional states), so it is not already covered by the timing variables Ari has. Treat it as a candidate, not as evidenced, because this paper never shows the metrics separate skill levels.

What to measure, and how it maps to the pipeline:

- Stance width as a fraction of body height. RTMPose gives ankle and foot keypoints, CourtKeyNet homography puts them in court metres, height comes from the same keypoints. Extraction cost E (easy), no new module.
- Split-step position and its spread. The distance-from-own-median measure (Fig 9) is a within-player consistency score: tighter clustering could mark a better-positioned player. Same inputs as above, cost E.
- Split-step reaction time. Time from opponent contact to first foot-off. Needs two things the geometry does not: a reliable opponent-contact timestamp, and detection of "one foot leaves the ground" from ankle vertical motion. Foot-off is doable but noisy from RTMPose; the contact anchor is the blocker.

Extractability: **E for the static geometry (stance width, position spread), B-to-X for anything timed.** The timed metrics depend on an opponent-contact anchor, which is one of the pipeline's known-missing capabilities (serve/contact detection). TrackNetV3 shuttle tracks could yield a contact event from a trajectory direction change, but that module does not exist reliably yet. Without it, reaction time is X (blocked); with it, B (moderate).

Covered already: **no.** The evidence table covers rally-level timing (duration, shots per rally, frequency, work/rest). Split-step geometry and per-stroke reaction are posture/positional, a listed gap.

Relative-within-video form survives: **partly, yes.** The absolute bands here are world-elite and will not transfer to club (0.25 s reaction, 50%-height stance, 42 to 55 cm spread mean nothing at club level). But two of the three measures work as within-video relative scores. Position spread (consistency of where a player sets up) and stance-width ratio can be computed per player and ranked between the two players in one clip, which is exactly the relative comparison the project wants. The consistency measure is the most promising because it is a variability score, not an absolute band. Reaction time is the most likely to separate club levels if it can be extracted at all, since club players should differ by more than the 0.01 s that separated these elites, so the coarse 30 fps resolution hurts less at club level.

Bottom line for the table: log split-step position-consistency and stance-width ratio as easy, video-readable, relative candidates in the posture/positional gap. Log split-step reaction time as a higher-value but blocked candidate, gated on building opponent-contact detection. Mark all three as un-validated for level discrimination, because no paper yet shows they track skill level.

## Red-team pass (2026-07-12)

A second read of the canonical copy against this draft. Changes folded in:

- Confirmed the F-statistic red flag is real and not a transcription artefact of the conversion. The canonical md line 238, 242, 246 all show `F (2, 145) = 3.06` with p = 0.40, 0.86, 0.04. The render (page 9) is consistent with the md. This is a genuine source error, kept as the top concern.
- Softened nothing on the n=3 point; it is central.
- Checked the "6 cells not 9" observation. The paper says nine segmented areas but the Center column and Mid row are always zero (Fig 10), so effectively six cells carry data. This is a source wording quirk, not a conversion fault; noted in the canonical md figure block, not raised as a concern because it does not affect any result.

Genuine disagreements left standing:

- None. The one judgement call was the extractability grade for reaction time, and it is now settled at B-to-X. The grade splits on the contact anchor. It is X (blocked) while serve/contact detection is missing, and B (moderate) once a coarse contact estimate exists. A stricter reading would call it X flat, treating the missing detection as a hard blocker. That reading is rejected here, because a coarse contact estimate from TrackNetV3 direction changes may be good enough for a relative club comparison even if it is too rough for elite work. B-to-X is the final call.
