<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# Making the World's Fastest Racket Sport even Better: A Systematic Review of Artificial Intelligence-based Objective Player Performance Assessment in Badminton

**Verdict**: Non-peer-reviewed Research Square preprint. Read it once as a field map and a citation index for AI badminton player assessment, not as a graded evidence source. Its own numbers do not fully agree, so trust the reference list more than its claims.
**Status**: Reviewed by Claude 2026-07-12. Verdict stands as Claude's, not independently confirmed by Curtis.

## Identity

- DOI: 10.21203/rs.3.rs-9105158/v1
- Authors: Arthur, Chiong, et al.
- Venue / year: Research Square (preprint), 2026
- Scope, one line: PRISMA-guided systematic review of AI methods for objective badminton player assessment, 51 studies from 2018 to end of 2025 (a few early-access 2026 items), three databases (Web of Science Core Collection, Scopus, IEEE Xplore).
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/ai_systematic_review_player_assessment.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/ai_systematic_review_player_assessment.pdf
- Note drafted: 2026-07-12

## What they built

A narrative synthesis of 51 studies, guided by PRISMA, that answers three questions: which AI methods are used for badminton player assessment, what limits them, and how their outputs reach coaches and players.

The main deliverable is Table II, a nine-column catalogue of every included study (focus, player sample, data source, AI method, reported metrics, framework orientation, validation approach). This is the review's most useful artefact for the project. It reads as a ready-made shortlist of methods and datasets.

They sort the field into four method families and give each a share of the corpus (Figure 3): stroke analysis 47.1%, spatio-temporal/tactical 19.6%, movement-pattern 17.6%, multi-modal 15.7%. So roughly half the field is stroke recognition, and everything else is thinner.

They also map the assessable signals into four indicator types: biomechanical (joint angles, segment velocities), shot quality (accuracy, consistency, power), tactical (shot selection, court positioning), and movement-related (footwork efficiency, court coverage). The interpretability section tracks how few studies explain their outputs and points to recent vision-language and RAG-LLM systems (ChatMatch, Court-to-Conversation) as the emerging direction.

## What holds up

The scope and search are stated plainly: three databases, a keyword scheme in three themes (Table I), a PRISMA flow with counts (Figure 1), and explicit inclusion rules. Anyone could re-run the search.

The four-family taxonomy matches what the table actually contains, and the shares in Figure 3 add up to 51 studies (24 + 9 + 10 + 8). The headline finding is well supported by the corpus: the field is dominated by supervised stroke and pose recognition on elite video, with high reported accuracy but little shared-dataset or cross-session validation.

The stated gaps are credible and useful: elite and single-region samples, no standard metrics, weak links to skill-acquisition theory, and interpretability treated as a novelty. These match what a reader sees scanning Table II.

Table II itself, treated as a citation index, is reliable enough to mine. Cross-checking a sample of rows against the reference list, the method and dataset entries line up with the cited papers.

## Methodology concerns

<!-- Index -->
- [methodological yellow-flag] | study count disagrees with itself (34 vs 51) | "Thirty-four primary studies, summarised in Table II" | p.11
- [methodological yellow-flag] | PRISMA counts disagree (356 vs 359) | "356 records retrieved from the academic databases were processed to result in the 51 studies" | p.3
- [assumed knowledge, otherwise sound] | interpretability counts read as loose (4 vs 2) | "Only 4 of the 51 reviewed studies incorporated explicit interpretability mechanisms" | p.9
- [methodological yellow-flag] | citation mislabelling | "Wang et al. [29] integrated bounding-box projections from their YOLO-HGNet detector" | p.10
- [methodological red-flag] | same papers counted twice, and non-badminton studies, inside the 51 | Table II rows 20/36, 22/42, 48/49; rows 24 and 26 | p.4–7
- [assumed knowledge, otherwise sound] | accuracy figures pooled across incomparable tasks with no risk-of-bias step | "Accuracy (100% in small dataset)" | Table II row 33

Entries:

1. The discussion opens by calling it "Thirty-four primary studies, summarised in Table II" (p.11), but Table II lists 51 numbered rows and the abstract and PRISMA both say 51. The 34 figure is never explained. A reader cannot tell how many primary studies the synthesis actually rests on.

2. The body text says "356 records retrieved" (p.3) while the PRISMA diagram (Figure 1, p.15) starts from n = 359 identified. Small, but it is the kind of bookkeeping a peer reviewer would catch, and it is the sort of slip this preprint has not had checked.

3. The interpretability counts read loosely rather than as a flat contradiction. Section 3.3.1 says "Only 4 of the 51 reviewed studies incorporated explicit interpretability mechanisms" (p.9), while the discussion says "Only two studies, Wang [8] and Chen et al. [9], integrated explanation tools such as Grad-CAM or SHAP" (p.11). The 4 can be read as the two Grad-CAM/SHAP studies plus the two vision-language systems (ChatMatch, Court-to-Conversation), and the 2 as the Grad-CAM/SHAP subset. The review never states this, so the reader has to reconstruct it. Bookkeeping, not a false claim.

4. Citations are used loosely. The interpretability example on p.10 credits "Wang et al. [29]" with "their YOLO-HGNet detector", but in Table II YOLO-HGNet is study 12, reference [18] (Yang et al.). The same labels drift: p.10 credits "Wang et al. [9]" with introducing ShuttleSet, which is reference [22], while [9] is Chen et al. Treat every in-text citation in this review as needing a check against Table II before you rely on it.

5. The corpus of 51 is padded. Reference [30] and reference [46] are the identical paper (Peng and Zheng, Ain Shams Eng J 2025;16:103414), cited as separate rows 20 and 36. References [32] and [52] are the same paper with the author order reversed (Zheng/Chen and Chen/Zheng), cited as rows 22 and 42. Rows 48 [58] and 49 [59] describe the same VideoBadminton experiment (7,822 clips, 19 athletes, matching metrics). On top of that, row 24 [34] is squash player tracking and row 26 [36] is a multi-sport survey. So the headline "51 studies" and the Figure 3 shares rest on a corpus that double-counts several papers and folds in non-badminton work.

6. Reported accuracies are listed side by side with no risk-of-bias or quality weighting, including "Accuracy (100% in small dataset)" for a wrist-sensor prototype (row 33). The review notes weak validation in prose but still presents the numbers as if comparable. There is no meta-analysis and no per-study quality score.

## Hand-waves and gaps

<!-- Index -->
- [assumed knowledge, otherwise sound] | elite, single-region bias named but not quantified | "participant pools drawn largely from elite athletes in a single geographic region" | p.2
- [methodological yellow-flag] | validation weakness stated, then not carried into how findings are weighed | "only a small proportion of them employ shared validation datasets or evaluate repeatability across testing sessions" | p.2
- [out of depth] | no treatment of unsupervised or clustering methods for skill separation | absence across the whole review | n/a
- [methodological yellow-flag] | abstract claims quality criteria but no appraisal tool or reliability check is reported | "satisfied the established quality and eligibility criteria" | p.2

Entries:

1. The elite and single-region skew is stated in the abstract and repeated in the discussion, which is honest, but it is never quantified. The review does not report how many of the 51 are elite-only, so the reader has to reconstruct it from Table II (most rows read "Elite athletes").

2. The abstract admits shared-dataset and repeatability testing are rare, yet the synthesis still leans on headline accuracies to argue the field has "matured". The caveat and the conclusion sit in tension.

3. The review frames the whole field as supervised recognition and prediction. It never discusses unsupervised learning or clustering for separating skill levels without labels. K-means appears only as a sub-step in two pipelines (score profiling [6], court-line extraction [37]). For a project asking how to grade unlabelled players, this silence is itself the finding: the field it surveys does not offer a worked unlabelled-clustering method.

4. The abstract states the 51 studies "satisfied the established quality and eligibility criteria", but no named quality-appraisal tool is reported and no inter-rater reliability is given for the two-author screening, which the paper describes only as reaching "full agreement". So the quality bar the abstract advertises is not evidenced. There is also a caption slip worth knowing when citing Figure 2: its title reads "2015 to 2025" while the plotted axis runs 2018 to 2025, with no point at 2021.

## Relevance to the project (as of 2026-07-12)

Weight everything here as preprint-grade. The value is as a field map and a citation index, not as validated evidence.

**Project question 1 (video-assessable skill dimensions, 2023–2026).** The review names several dimensions the project can extract from its existing pipeline outputs, and it points at which ones have tested level or outcome links:

- Recovery positioning and six-corner footwork from pose keypoints (studies 7 [10], 9 [12], 13 [23]). These are video-cheap given RTMPose and court homography, and they sit in the profile's stated gap of posture and positional states.
- Positional heat maps and court coverage tied to player profiling and regrading (studies 3 [6] Sinadia, 11 [17] Asriani). Asriani [17] is cited as more sensitive than fixed-period review when reclassifying players, which is close to the project's grading goal.
- Shot influence and rally-outcome features (studies 19 [29], 32 [42], 1 [22] ShuttleSet) for within-rally value, though the review is clear these need heavy annotation and expert labels to scale.
- Grip force (study 41 [51]) and Quiet Eye gaze (study 17 [27]) show tested skill correlations (grip force r ≈ −0.6 to −0.7, beginner vs national athlete). Both need sensors or eye-tracking. They confirm a dimension matters, but the video pipeline cannot extract them.

**Project question 2 (clustering for unlabelled skill separation).** The review offers little. It does not surface a dedicated unsupervised method for separating skill levels without labels. That gap is the opportunity: this overview confirms the field grades players with supervised, labelled models, so unlabelled clustering of amateur play is close to open ground.

**Covered already vs the two prior sessions?** Partly. The reference list was already harvested last session, so most individual papers are not new. What is worth lifting is the review's own framing: the four-family taxonomy, the four indicator types, and the explicit naming of video-extractable, level-linked dimensions (recovery position, footwork patterns, positional heat maps) that match the profile's identified gaps. Timing variables, which the profile marks as well covered, are barely present here, so the review does not overlap much with Ari's existing evidence table.

**Most useful directions to follow (from Table II as a citation map):**
1. Asriani et al. [17] and Sinadia and Murwantara [6] for spatio-temporal and ML profiling aimed at player regrading, the closest existing work to level grading from cheap features.
2. Recovery-position and six-corner footwork models [10], [12], [23] for video-extractable, level-relevant posture and movement features.
3. ShuttleSet [22] and shot-influence [29] for rally-level tactical features, with eyes open to their annotation cost.

Note the review is a preprint, so verify any specific claim against the primary paper before it goes into the project's own writing.
