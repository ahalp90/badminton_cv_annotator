<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# StanceScorer: A Data Driven Approach to Score Badminton Player

**Verdict**: Skip the method, keep one idea. A 2020 wearable-IMU proof of concept that never produces an actual score. Its one transferable idea is reference-based scoring: measure a learner against a recording of a professional playing the same shot, with no graded skill labels needed. The sensor pipeline does not transfer to this project's video keypoints.
**Status**: Reviewed by Claude 2026-07-12, not independently re-read by Curtis.

## Identity

- DOI: 10.1109/PerComWorkshops48775.2020.9156220
- Authors: Indrajeet Ghosh, Sreenivasan Ramasamy Ramamurthy, Nirmalya Roy
- Venue / year: IEEE PerCom Workshops, 2020
- Scope, one line: Classify badminton strokes from wrist and palm IMUs, then score a learner's footwork stance by its distance from a professional's leg-IMU pattern for the same stroke.
- Canonical copy: restricted licence; held privately at 595-personal-notes/archive/restricted_papers/stancescorer_score_badminton_player.md
- Source PDF: restricted licence; held privately at 595-personal-notes/archive/restricted_papers/stancescorer_score_badminton_player.pdf
- Note drafted: 2026-07-12

## What they built

A three-stage pipeline on wearable sensor data, not video.

Stage one classifies the stroke. Four Shimmer IMUs sit on the dominant wrist, dominant palm, and both legs. Each carries accelerometers, a gyroscope, and a magnetometer, giving 48 features. A CNN reads the upper-limb streams and predicts which of 12 labelled strokes was played. It is compared against Decision Tree, Random Forest, SVM, and MLP baselines.

Stage two builds an "ideal stance" for each stroke from the professional player only. For each frame of a learner's leg data, it finds the k nearest frames in the professional's leg data for the same stroke label, then averages those k frames. The average is treated as the target stance.

Stage three scores by error. It computes MSE, RMSE, MAE, and MdAE between the learner's leg data and the professional's averaged target. The claim is that this error stands in for a stance score.

Data came from three players, one professional, one intermediate, one novice, 30 repetitions of each of 12 strokes.

## What holds up

- The stroke classifier works on its own data. CNN testing accuracy is 86.27% across all participants, rising to 93.66% on the intermediate-plus-novice pairing (Table IV). The CNN beats the shallow baselines by roughly 7% when all players are pooled.
- The reference-based framing is sound in principle. Scoring a learner by distance from a professional doing the same action needs no graded skill labels. That sidesteps the hardest data problem for this kind of work.
- They are honest about the main gaps. The limitations section names the missing actual score and the missing cross-user validation directly.
- Sampling was fast (512 Hz), so their argument that hand-labelled activity boundaries add only minute error is reasonable.

## Methodology concerns

<!-- Index first, one line per entry: flag | "short verbatim quote for ctrl+F" | p.N -->

- methodological red-flag | "average age: 27 years" | p.6 (n=3, one player per skill level)
- methodological red-flag | "the professional player did not move around to perform the Backhand overhead drop shot" | p.7 (reference confound)
- methodological yellow-flag | "this work does not address cross-user variation" | p.7
- methodological yellow-flag | "report the best results obtained in this paper" | p.7 (best-k reporting)
- methodological yellow-flag | "sliding windowing with 50% overlap" | p.6 (likely window leakage)
- methodological yellow-flag | "The label for the dataset was assigned by the authors of this paper" | p.6

**n=3, one player per level** (p.6): "For experimentation, we collected 30 iterations of each of the 12 strokes ... from all the 3 participants (3 males; average age: 27 years)." The entire scoring claim rests on one professional, one intermediate, and one novice. A single professional defines the target for every stroke, so his personal quirks become the definition of correct. Nothing here can show the method separates skill in general rather than fitting three specific people.

**The reference is confounded** (p.7): "the professional player did not move around to perform the Backhand overhead drop shot. A similar trend was noticed in the performance of the novice player. However, the intermediate player moved around a lot for backhand overhead drop shot." On this shot the novice scores better than the intermediate. The authors read this as a feature. It is closer to a warning. The novice wins by staying still like the professional, not by playing well. The error measures similarity to one person's habits, and those habits include not moving. That breaks the link between low error and high skill.

**No cross-user validation** (p.7): "In addition, this work does not address cross-user variation in playing the sport." Two people can play the same stroke well with different footwork. A pure distance-to-one-professional score punishes any valid variation. The authors flag this but do not test it.

**Best-k reporting** (p.7): "We tried different values of k ranging from 5 to 25 with an interval of 5 and report the best results obtained in this paper." k=25 was picked because it looked best on the same data being reported. With no held-out check this is tuning on the test set.

**Likely window leakage** (p.6): they used "sliding windowing with 50% overlap" and then a random "training and testing split of 80-20." They never say the split respects session or subject boundaries. With 50% overlap, adjacent windows share half their samples, so a random split can put near-copies of a training window into the test set. That inflates the reported classification accuracy. This is not stated as a fault in the paper, so treat it as plausible rather than proven, but the accuracy figures should be read with it in mind.

**Author-assigned labels** (p.6): "The label for the dataset was assigned by the authors of this paper." Labels came from watching the collection session. It is workable at this scale but adds a subjective step with no inter-rater check.

## Hand-waves and gaps

<!-- Same index format as methodology concerns. -->

- methodological red-flag | "this modeled error could be used for scoring the players" | p.4 (the core claim, never realised)
- assumed knowledge, otherwise sound | "an exact score was not computed in this paper" | p.8
- methodological yellow-flag | "the explanation of the stances in table I may look vague" | p.6

**The paper never scores anyone** (p.4, p.8): the contribution is stated as "We hypothesize that this modeled error could be used for scoring the players for their strokes and stances." Page 8 admits "an exact score was not computed in this paper." So a paper titled StanceScorer produces error histograms and a hypothesis, not a scorer. The whole scoring step is future work wearing a results section.

**Vague stance definitions** (p.6): "Although the explanation of the stances in table I may look vague, all the players played the strokes and stances as a player would normally play." Table I describes stances in phrases like "step sideways" and "heavy movements and jumps." Without a domain expert defining the target movement, the ground truth for a good stance is loose.

**Figure 4 does not match Table IV** (p.7): the classifier bar chart puts CNN near 95%, but Table IV reports 86.27% for all participants and the prose claims only a 7% gain over shallow learning. The chart never states which player combination it plots, so it may show a best-case pairing. The two numbers are not reconciled in the paper.

## Relevance to the project (as of 2026-07-12)

The project grades club-level players from unannotated video, using RTMPose keypoints and other CV outputs. This paper is worth reading once for its framing and skipping for its pipeline.

**Wrong input modality.** StanceScorer runs on four body-worn IMUs streaming accelerometer, gyroscope, and magnetometer data. The project has 2D pose keypoints from video. Nothing in the sensor pipeline transfers directly. The wrist and palm IMU features for stroke classification are already handled better in the project by BST-X on pose. So the classifier half is not useful here.

**The transferable idea is reference-based scoring.** The k-NN-plus-averaging step builds a target movement for a stroke from a reference player, then scores others by distance from it. That logic does port to pose data. You could take RTMPose keypoint trajectories for a stroke, align them, and score a player by distance from a stronger player's trajectory for the same stroke. This needs no A/B/C/D grade labels, which fits the unlabelled-video constraint. Extraction is Bridge level: the concept ports, but the code does not. You would need per-stroke pose alignment, a chosen reference, and handling for the modality swap.

**It scores a narrow thing.** This is a per-stroke stance (footwork) error, not a whole-player skill score. In the project it would be at most one candidate feature feeding a clustering or a dimension score, sitting under the "posture and positional states" gap in Ari's evidence table. It is not a finished grader.

**Not superseded for the idea, weak for evidence.** The reference-based framing still stands in 2026. The evidence behind it does not: n=3, one exemplar per level, a confounded reference, and no computed score. Do not cite it as validation that stance error tracks skill. Cite it, if at all, as an early instance of the reference-based framing, and lean on stronger 2023-2026 sources for the actual claim.

**Covered already?** No. The evidence table already covers timing variables well. This touches the posture and relative-comparison gap instead. But it does so on IMU data with no club-level validation, so it adds a framing to consider, not a validated dimension to adopt.

## Genuine disagreements with the draft

Red-team folded its fixes back in, so nothing stands unresolved. Two changes were made on the second read. A window-leakage concern was added, because the 50% overlap plus a random 80-20 split can leak near-copies into the test set and inflate accuracy. The "never produces a score" flag was raised from padding to a red flag, since a paper titled StanceScorer not computing a score is a core-claim gap, not filler.
