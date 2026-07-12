<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# MultiSenseBadminton: Wearable Sensor-Based Biomechanical Dataset for Evaluation of Badminton Performance

**Verdict**: Skim, do not full-read. A wearable-sensor dataset, not video, so it does not feed our pipeline; its one liftable idea is a multi-rater coach scoring scheme for skill ground truth.
**Status**: Reviewed by Claude 2026-07-12, not independently confirmed by Curtis

## Identity

- DOI: 10.1038/s41597-024-03144-z
- Authors: Minwoo Seong, Gwangbin Kim, Dohyeon Yeo, Yumin Kang, Heesan Yang, Joseph DelPreto, Wojciech Matusik, Daniela Rus, SeungJun Kim
- Venue / year: Scientific Data, 2024
- Scope, one line: An open multimodal wearable-sensor dataset of 25 players hitting forehand clears and backhand drives, with five annotation levels including a coach-scored skill level.
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/multisensebadminton_wearable_dataset.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/multisensebadminton_wearable_dataset.pdf
- Note drafted: 2026-07-12

## What they built

MultiSenseBadminton is an open dataset of 7,763 badminton swings from 25 players. Every player hit two strokes only: the forehand clear and the backhand drive. A machine launched every shuttlecock on a fixed trajectory, so the setting is controlled, not match play.

Each swing carries five synchronised wearable streams. Eye tracking (Pupil Invisible, 30 Hz gaze). Full-body motion (Perception Neuron Studio, 17 IMU trackers, joint angles and positions at 96 Hz). Arm muscle activity (gForcePro+ EMG, 8 channels, up to 1000 Hz). Leg muscle activity (Cognionics AIM EMG, 4 channels, 500 Hz). Foot pressure (Moticon insole, 16 pressure sensors plus a 6-axis IMU, 100 Hz). Three cameras add front, side, and whole views, and an eye camera adds first-person video and sound.

The data carries five annotation levels. Level 1 stroke type. Level 2 skill level. Level 3 shuttlecock landing position on a grid. Level 4 hitting point, front or back of the body. Level 5 stroke sound, good, maybe, or bad.

The skill-level scheme is the part worth reading. Three professional coaches watched each player's clears and drives and scored them from 1 to 7. The three scores were averaged, then binned into three classes: beginner, intermediate, expert. The dataset keeps each coach's raw score as well as the consolidated label.

They also ran a proof-of-concept classifier (ConvLSTM, LSTM, Transformer, against a majority-class baseline) over all five annotation types, to show the data supports machine learning.

## What holds up

The dataset is real, open, and documented. It is CC-BY on figshare, has IRB approval, and ships reader code on GitHub. The modality mix is genuinely wide for badminton, wider than any prior public set they list in Table 1.

The skill labels come from three qualified coaches, not self-report alone, and the paper reports inter-rater reliability for them. Agreement was moderate. Fleiss kappa across all three coaches was 0.64 for the clear and 0.75 for the drive, with pairwise Cohen kappa spread from 0.59 to 0.82 (Table 5). The paper's own line that values were "above 0.64" is a slight round-up, since two of the pairwise clear values (0.59, 0.63) sit just below it.

The annotation protocol is spelled out level by level, and the authors are honest about the dataset's limits: constrained environment, two strokes only, sensor noise and drift, and class imbalance.

## Methodology concerns

- methodological yellow-flag | "46.56 (8.65)" | p.18
- methodological yellow-flag | "we extracted a total of 7761 stroke instances from 18 participants" | p.16
- methodological yellow-flag | "the baseline accuracy occasionally surpasses that of the deep learning models" | p.18

Skill classification is only modest under subject-independent testing. Skill level is a 3-class target, so chance is about 33 percent. Under leave-three-out validation the best model reaches 46.56 percent accuracy and 48.35 percent balanced accuracy for the clear, and 46.98 percent for the drive, against a majority baseline near 34 percent (Table 7). That is well above the floor but far from usable. The paper's prose ("deep learning models outperformed the baseline") is true, but the win is a thin one and the absolute numbers are low.

The number that fragility rests on is the per-player label, not the instance count. Skill is one label per player, so all of a player's strokes share it. The headline "7761 stroke instances from 18 participants" is really about 18 independent skill labels, and leave-three-out then tests generalisation on only 5 or 6 experts (Table 5: clear 11/8/6 beginner/intermediate/expert; drive 11/9/5). That, not the raw swing count, is why the ~46 percent is shaky.

The learning pipeline also uses 18 of the 25 players. The reason is not stated at the pipeline step, but the missing-data section (p.14) documents missing eye gaze for several subjects, and 2D gaze is a required input feature, so the drop is plausibly gaze-driven rather than unexplained.

Class imbalance inflates the baseline, and for vertical landing and hitting point the majority-class baseline sometimes beats the deep models. The authors say so plainly, which is good, but it means several of the headline accuracy numbers describe the data skew, not model skill.

## Hand-waves and gaps

- out of depth | "rating their forehand clears and backhand drives on a scale ranging from 1 to 7" | p.9
- methodological yellow-flag | "it is advisable to recruit participants with intermediate or higher skill levels" | p.19

The paper never identifies which biomechanical quantities separate skill levels. Skill is only ever an end-to-end classification target. There is no per-feature or per-modality analysis, no "expert players show X in gaze or Y in foot pressure." So a reader cannot lift a list of measurable discriminators from this work, only the fact that the combined signal carries some weak skill information.

"Skill" here is judged on two beginner strokes hit off a machine. It is not full-game skill, and the authors admit future work should recruit intermediate-and-above players for real play. The label therefore measures early-stage stroke competence, not the broad club-grade skill our project cares about.

Sample is small and skewed: 25 players, only 5 female, training experience 0 to 22 years but heavily bunched at the low end (Table 3).

## Relevance to the project (as of 2026-07-12)

This is wearable-sensor data, not video, so the sensor streams do not plug into our pipeline (TrackNetV3, RTMPose, CourtKeyNet, BST-X, rally segmentation). None of the sensor signal is extractable through our tools: n/a, sensor not video.

One thing is worth crediting before moving on. The skill labels themselves were produced from video. Three coaches watched participant footage and scored each player, and the paper reports the reliability of that scoring. That is modest support for our core premise, that club-level skill on these strokes is judgeable from video, even though the paper's own models used sensors rather than video to predict it.

Its liftable value is narrow and sits in two places.

First, the labelling scheme is a usable template for building our own skill ground truth. Multiple qualified raters score each player on an ordinal scale, you average, then bin into levels, and you report inter-rater reliability. We could copy that method for A/B/C/D club grading. The specifics do not transfer: they used 3 levels not 4, a 1-to-7 scale, and bin thresholds (1, 3, 6) tuned to two beginner strokes. So borrow the process, not the numbers.

Second, the coach interviews (Table 2) name evaluation criteria that happen to be video-assessable: hitting the shuttle in front of the body (pose), landing location and trajectory (shuttle tracking), and ball speed (shuttle tracking). These are useful pointers toward what coaches watch. But they are expert opinion, not tested correlations with skill level, so they do not meet the evidence-table bar of "tested level or outcome correlation." They belong in the ideas column, not the evidence column.

On the second research question the paper is a near miss. It could have told us which biomechanical variables separate levels, which a pose pipeline might then approximate. It does not. It only shows weak end-to-end skill classification (about 46 percent, 3-class, subject-independent), with no variable-level attribution. So it does not help prioritise video dimensions.

Bottom line: worth one line in the sources list as the widest open badminton multimodal set, plus a saved note that its coach-scoring method is a template for our own skill ground truth. Not worth a full read, and it adds no tested skill discriminators to Ari's evidence table.

## Red-team log

A fresh reader attacked this note against the source. All five findings were folded in:

- Corrected the kappa claim. Table 5 has pairwise Cohen values of 0.59 and 0.63, so "above 0.64" was the paper's round-up, not the table.
- Credited that the skill labels were graded from video with reported reliability, which is modest support for the project premise, rather than dismissing the paper as wholly n/a.
- Softened "barely beats chance" to "modest," since 46 to 48 percent sits well above the 33 percent floor even if it is low.
- Fixed the "seven players dropped with no reason" line. The missing-data section documents missing eye gaze, and gaze is a required feature, so the 18-of-25 count is plausibly gaze-driven.
- Added the real reason the skill numbers are fragile: skill is one label per player, so the thousands of instances reduce to about 18 independent labels, tested on 5 or 6 experts.

No standing disagreements. The verdict (Skim) held under the attack; the video-graded-skill credit sharpens the relevance section without changing that the paper adds no extractable capability or tested discriminators.
