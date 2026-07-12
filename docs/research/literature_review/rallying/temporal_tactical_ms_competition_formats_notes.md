<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# Comparison of the temporal and technical-tactical characteristics in badminton mens singles under different competition formats

**Verdict**: Skim, do not rely on the small effects. A solid descriptive record of youth men's-singles timing and stroke shares under two scoring formats, but the timing bands duplicate what the evidence table already has from elite studies, and the many small "significant" results carry real false-positive risk. Useful mostly as a youth-elite reference point and a format contrast, not a level-grading source.
**Status**: Reviewed by Claude 2026-07-12, not independently re-read by Curtis.

## Identity

- DOI: 10.3389/fpsyg.2025.1634776
- Authors: unknown
- Venue / year: Frontiers in Psychology, 2025
- Authors (from paper): Zhao Y, Zhu A, Zhang S, Zhang Y (Taiyuan University of Technology)
- Scope, one line: Notational comparison of 40 youth men's-singles matches (20 individual 21-point, 20 team 11-point relay) at the 2024 BWF World Junior Championships, same athletes across both formats.
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/rallying/temporal_tactical_ms_competition_formats.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/rallying/temporal_tactical_ms_competition_formats.pdf
- Note drafted: 2026-07-12

## What they built

A hand-coded notational study. One trained analyst watched official BWF video frame by frame and logged every stroke in a custom C# WinForm tool (Figure 1), tagging technique, court position, direction, landing zone, and scoring outcome. The sample is 40 matches, 80 sets, 22,283 strokes. Twenty matches used the standard individual 21-point format and twenty used the new team 11-point relay format, drawn from the same pool of 17-to-18-year-old players so each athlete appears in both formats.

They compared the two formats across four groups of variables:
- Time: shots per rally, time per stroke, interval time, and the total time for shots, intervals, and whole game (Table 3); a three-phase rally-length split of 0-10 s, over 10 s, and over 20 s rest (Table 4); and a "rally rhythm coefficient" of strokes per rally over rally duration (Table 5).
- Technique: usage share of ten stroke types (Table 6), plus per-technique scoring rate (Table 8) and error rate (Table 9).
- Scoring: the nature of points won and lost, split into forced error, unforced error, active error, and direct point (Table 7).
- Space: hitting route by court side (Table 10) and landing point by zone (Table 11).

Every comparison used a Mann-Whitney U test with Cohen's d reported alongside.

## What holds up

The headline timing result is strong and unambiguous. Team relay matches roughly halve the whole-match clock: total stroke time drops 46.8 percent, total interval time 49.1 percent, and total game time 47.4 percent, each with p < 0.001 and very large effects (d = 2.408, 2.060, 2.073). Those effects are far too big to be an artefact of the analysis choices.

The paired finding also reads cleanly: rally-level intensity does not change. Shots per rally (10.05 vs 9.57), time per stroke (8.80 vs 8.56 s), and interval time (21.41 vs 20.42 s) all return p > 0.05 with tiny effects (d < 0.22). So the shorter format compresses the match by having fewer rallies overall, not by making each rally shorter or slower. That is a sensible, internally consistent story.

Coding reliability is acceptable: Cohen's kappa over 0.85 on all variables, checked by a re-code after 72 hours and a second analyst on three matches.

As descriptive numbers, the youth men's-singles values are usable: shots per rally near 10, time per stroke near 8.7 s, rest near 21 s, and a rally-length split where 0-10 s rallies make up about 69-71 percent of play.

## Methodology concerns

Index:
- red | "it was ensured that each athlete participated in both the individual and team formats of the event" | p.3
- yellow | "No Holm– Bonferroni or other multiple-comparison corrections were applied." | p.4
- yellow | "another 20 matches from the knockout and placement stages of the men's singles in the team event" | p.2
- yellow | "total sample size _N_ = 160" | p.4
- yellow | "the non-parametric Mann–Whitney U test was consistently applied to assess all inter-group differences" | p.4
- yellow | "block ( _p_ = 0.016, _z_ = −2.530, _d_ = 0.389), and set ( _p_ = 0.041" | p.5

**Paired data, unpaired test.** This is the sharpest problem. The design is explicitly paired: "it was ensured that each athlete participated in both the individual and team formats of the event" (p.3), and the paper sells this as its main strength for controlling skill. But it then compares the two formats with a Mann-Whitney U test, "inter-group differences were evaluated using individual Mann–Whitney U tests" (p.4), which is an independent-samples test. It assumes the two groups are unrelated, which is false here. The design-correct choice is a paired test such as Wilcoxon signed-rank. Using an unpaired test on paired data throws away the very pairing the study was built on and can bias the p-values in either direction. The large timing effects would survive this, but the many borderline technique and space results are exactly where it matters. Reporting parametric Cohen's d next to a rank-based test is a smaller version of the same mismatch.


**No multiple-comparison control across dozens of tests.** "No Holm– Bonferroni or other multiple-comparison corrections were applied." (p.4). Tables 3 through 11 run on the order of fifty separate tests at alpha 0.05. Many of the reported "significant" results sit right at the edge: block scoring rate p = 0.047, drop scoring rate p = 0.045, intercept usage p = 0.041, unforced error p = 0.03, several route and landing p-values between 0.008 and 0.026. With no correction, a handful of these are expected to be false positives. The authors defend this by framing each variable as its own research question, but that does not remove the risk when the discussion then reads them together as a pattern. Trust the big time effects; treat the many small technique and space results as suggestive only.

**Format is confounded with match stage.** The individual matches are knockout-stage only, but the team matches mix stages: "another 20 matches from the knockout and placement stages" (p.2), and elsewhere "the knockout and qualifying stages" (p.3, the two passages disagree on which stage). Qualifying or placement matches can differ in stakes and effort from knockout matches. So a difference the paper credits to scoring format could partly be a stage effect. This undercuts the claim that the same-athlete design gives a clean format contrast.

**Sample-size reporting contradicts itself.** The abstract and methods say "A total of 40 matches were analyzed, with 20 matches for each competition format" (p.2), but the statistics section says the data came from "40 individual matches (21-point system) and 40 team matches (11-point system)" with "total sample size _N_ = 160" (p.4). Eighty matches cannot also be forty. N = 160 is probably an athlete-or-set count rather than a match count, but the paper never reconciles the figures, so the true analysis unit and denominator are unclear.

**Test statistic is not actually consistent.** The methods insist "the non-parametric Mann–Whitney U test was consistently applied" (p.4), yet the rhythm coefficient is reported with a t statistic, "_t_ = 4.426" (p.5), and the route and landing prose also quote t values (for example "_t_ = −2.669", p.7). Mann-Whitney yields z, not t, and every other table reports negative z values while the rhythm z of 4.426 is positive and out of scale. Some comparisons look like they used a t-test after all, which contradicts the stated method and matters because the rhythm result is one the discussion leans on.

**Section 3.3 prose mislabels its own techniques.** The Table 6 write-up reports significant differences for "block ( _p_ = 0.016, _z_ = −2.530, _d_ = 0.389), and set ( _p_ = 0.041" (p.5). There is no "set" technique in the study, the p and z given for "block" actually match the Kill-and-brush row of Table 6, the "set" figures match the Intercept row, and the same sentence then lists "block" again among the non-significant techniques. The table numbers themselves are fine (verified against the render), but the prose that interprets them is garbled, so any claim about which techniques differ should be read off the table, not the text.

## Hand-waves and gaps

Index:
- yellow | "provides a rare quasi-experimental scenario to investigate the causal relationship between competition format and athletic performance" | p.2
- yellow | "the change in the scoring system does not increase the rhythm of the game but reduces it" | p.8

**Causal language on observational data.** The setup claims the design "provides a rare quasi-experimental scenario to investigate the causal relationship between competition format and athletic performance" (p.2). This is notational analysis of tournament video with no randomisation and, as noted above, a stage confound. It can describe differences between formats. It cannot establish that the format caused them.

**A trivial effect carries a headline conclusion.** The pace claim, "the change in the scoring system does not increase the rhythm of the game but reduces it" (p.8), rests on the rhythm coefficient dropping from 1.161 to 1.122. That difference is reported significant (p < 0.01) but with d = 0.18, below even the small-effect threshold the paper itself sets at 0.20. A gap that small is not a safe basis for a stated conclusion about how the format changes play, especially given the t-versus-z problem on that same number.

**Youth-only sample.** All players are 17 to 18 years old. The values are a youth-elite reference and should not be read straight across to adult elite or to amateur club players.

## Relevance to the project (as of 2026-07-12)

Short version: this refines nothing the evidence table is missing. The timing variables it reports are already covered by the elite-era studies in the table (Leong 2016, Hoffmann 2024, Laffaye 2015, Gawin 2015, Chiminazzo 2018, Le Mansec 2023). What is new here is a youth-elite band and a scoring-format contrast, neither of which is a club-level or skill-level contrast. For grading A/B/C/D club players it is background, not evidence.

**Timing numbers, for the record (individual 21-point vs team 11-point relay, mean ± SD):**
- Shots per rally: 10.05 ± 2.21 vs 9.57 ± 1.61 (p = 0.413, d = 0.184, no difference)
- Time per stroke: 8.80 ± 1.73 vs 8.56 ± 1.62 s (p = 0.554, d = 0.133, no difference)
- Interval time: 21.41 ± 4.48 vs 20.42 ± 4.59 s (p = 0.348, d = 0.211, no difference)
- Rally length 0-10 s: 69.09 vs 70.92 percent; over 10 s: 30.91 vs 29.08 percent
- Rally rhythm coefficient (strokes per rally / rally duration): 1.161 vs 1.122 (p < 0.01, d = 0.18)
- Total per-match (large differences): total stroke time 320.67 ± 79.50 vs 170.52 ± 50.30 s (d = 2.408); total interval time 789.31 ± 316.32 vs 401.77 ± 195.13 s (d = 2.060); total game time 1123.37 ± 428.98 vs 590.31 ± 214.23 s (d = 2.073)

**Technique and error contrasts (only the significant ones):** spinning net shot usage 0.08 vs 0.06 (p = 0.002); drop usage 0.03 vs 0.05 (p = 0.001); clear usage 0.02 vs 0.01 (p = 0.004); lift 0.20 vs 0.18 (p = 0.011); kill-and-brush scoring rate 0.50 vs 0.26 (p < 0.001, d = 0.789, the one large technique effect); intercept error rate 0.07 vs 0.02 (p = 0.006); forced error share 0.198 vs 0.270 (p = 0.004); unforced error share 0.300 vs 0.256 (p = 0.03). These separate two formats, not two skill levels, so they do not map onto club grading.

**Extractability against the current pipeline:**
- Cheap now or soon (B/E). Shots per rally and the 0-10 s / over-10 s rally-length split fall straight out of BST-X stroke counts plus rally segmentation once that lands. The rhythm coefficient is just strokes per rally over rally duration, so it is free once those two exist. Time per stroke and interval time need per-stroke and between-rally timestamps, which stroke detection already gives.
- Needs missing pieces. Everything that discriminated the formats here beyond raw timing needs capabilities the pipeline does not have: technique usage shares need reliable 10-class stroke labels mapped to this scheme (BST-X is 14-class, so a mapping step), and the scoring, error-nature, and per-technique scoring/error results all need rally-winner detection, serve detection, and forced-versus-unforced judgement. Those are the project's known gaps. So the interesting contrasts in this paper are exactly the ones we cannot currently extract.

**Covered already: yes.** The timing bands duplicate elite-era coverage in the table. Take from this paper only two things: a youth-elite descriptive band for the timing variables, and the confirmation that a shorter scoring format cuts total match time without changing per-rally intensity. Do not lift the small technique and space effects as level signals; they are format effects with false-positive risk and no skill-level interpretation.

## Red-team pass

A fresh-context agent read the paper and this draft and attacked it. Folded in:
- Added the paired-data-unpaired-test concern (the design pairs athletes across formats but uses an independent-samples Mann-Whitney). This is now the top methodology concern.
- Added the Section 3.3 prose-mislabelling concern (the "block"/"set" write-up is garbled; table numbers are fine).
- Corrected four page citations that were off by one or two against the markdown page markers (knockout-and-placement to p.2, qualifying-stages to p.3, quasi-experimental to p.2, rhythm-reduces-it to p.8).

Confirmed clean by the red-team: all means, SD, p, and d in the relevance section match Tables 3-9 exactly; the four surviving quotes are verbatim; the verdict follows the body; no overclaimed concerns.

Outstanding disagreements: none. Every red-team point was accepted.
