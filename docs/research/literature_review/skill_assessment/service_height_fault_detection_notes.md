<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# Automated Service Height Fault Detection Using Computer Vision and Machine Learning for Badminton Matches

**Verdict**: Skim, do not full-read for our project. It solves officiating with a purpose-built calibrated rig, which is a different problem from grading skill off arbitrary video, but its monocular-height caution and its detection evidence are worth lifting.
**Status**: Reviewed by Claude 2026-07-12, not independently re-read by Curtis.

## Identity

- DOI: 10.3390/s23249759
- Authors: Goh, Guo Liang; Goh, Guo Dong; Pan, Jing Wen; Teng, Phillis Soek Po; Kong, Pui Wah
- Venue / year: Sensors (MDPI), 2023
- Scope, one line: A two-camera calibrated rig plus a YOLOv5 detector that flags the "service too high" fault (shuttle struck above 1.150 m at contact), benchmarked against 200 Hz motion capture and eight human judges.
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/service_height_fault_detection.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/service_height_fault_detection.pdf
- Note drafted: 2026-07-12

## What they built

An automated umpire for one badminton rule: the serve must be struck with the whole
shuttle below 1.150 m. Two cameras sit at court side, one per side, about 1.9 m from the
sideline where a service judge would sit. Each camera is placed so its sensor centre is
exactly 1.150 m off the ground and its optical axis lies flat along the 1.150 m plane. A
gimbal handles roll and pitch during calibration.

That placement is the whole idea. Because the camera looks straight along the height
limit, the shuttle appearing above the image centre-line means it is above 1.150 m, and
below means legal. They do not measure absolute height. They read a binary above-or-below
straight off the pixels.

A YOLOv5 model detects shuttlecock, player, racket, and shoe at about 70 fps. The hitting
instant (frame N) is found by watching for a large horizontal jump in the shuttle position
between consecutive frames. At that frame the top of the shuttle bounding box gives the
height used for the ruling. Training used 25,235 images from six venues and 19 players,
with 1900 shuttle-only images added to balance the classes.

They benchmarked two ways. Against a Vicon 200 Hz motion capture system as ground truth,
and against eight human judges (three professional service judges, five untrained) using
the fixed height service tool. Only backhand low serves were tested, in a lab, with
servers deliberately aiming near the limit.

## What holds up

The calibration trick is clean and correct. Putting the optical axis on the 1.150 m plane
turns a hard 3D height problem into a direct 2D read. It sidesteps the known weakness of
monocular vision, which cannot recover absolute height from one arbitrary view.

The detector evidence is solid and reusable as a data point. YOLOv5 reached mAP@0.5 = 0.99
on the test set and 0.956 on 968 untrained cross-validation images, across six halls. That
is decent proof that shuttle, racket, player, and shoe detection at roughly 70 fps is
feasible on real venue footage.

The mocap benchmark is a genuine gold standard, and the reporting is candid. They state
plainly that even the system is only 58% accurate in the critical 1.150 to 1.155 m band,
and their limitations section lists the lab compromises honestly rather than burying them.

The range-of-confusion and midpoint-of-confusion metrics are a sensible way to score
judgement consistency, though they are author-invented and unvalidated, so treat them as a
useful lens rather than an established measure. The system's confusion ranges are narrow
(most within about 1140 to 1170 mm) where human ranges are wide and scattered (one human
overall span reaches roughly 980 to 1220 mm). The consistency gap is the strongest result
in the paper.

## Methodology concerns

Index:
- methodological yellow-flag | "our analysis was limited to the accuracy of service judgment calls for backhand low serves" | p.16
- methodological yellow-flag | "the exact threshold is not disclosed" | p.15
- methodological yellow-flag | "While a filtering algorithm to identify players has been developed, it is not discussed" | p.15
- methodological yellow-flag | "we evaluated the accuracy of the system based on detected services" | p.12
- assumed knowledge, otherwise sound | "achieving a 58% accuracy rate for detecting service heights between 1.150 and 1.155 m" | p.1

**Narrow test, broad claim.** The head-to-head against humans used only backhand low
serves, in a lab, with only two servers ("We engaged two badminton players to be the
servers", p.8) told to aim near the limit ("our analysis was limited to the accuracy of
service judgment calls for backhand low serves", p.16). Other serve types have different
shuttle kinematics, which the paper admits changes the error. The headline "beats humans by
3.5 times" rests on one serve type, two servers, and one setting.

**Accuracy scored only on detected serves.** The system's accuracy is computed on the
serves it managed to detect ("we evaluated the accuracy of the system based on detected
services", p.12), 204 for the system against 255 for humans. The shortfall is missed
detections, which are dropped from the tally rather than counted as errors. Excluding the
cases the system could not handle can flatter its numbers against the humans, who ruled on
every serve.

**Undisclosed threshold.** The hitting instant depends on a speed threshold that they
withhold for patent reasons ("the exact threshold is not disclosed", p.15). The result is
not reproducible from the paper alone. A reader cannot rebuild or test the core timing
step.

**Undisclosed filters.** Real-world use needs a filter to tell players from bystanders and
to reject white objects mistaken for shuttles. Both exist but are held back ("While a
filtering algorithm to identify players has been developed, it is not discussed", p.16). So
the system as described is not the system as it would run.

**Two different accuracies.** The 0.99 mAP is detection accuracy, meaning the model finds
the shuttle. The decision accuracy near the limit is far lower: 58% in the 1.150 to
1.155 m band ("achieving a 58% accuracy rate for detecting service heights between 1.150
and 1.155 m", p.1). Reading only the mAP would overstate how well the system actually rules
a close serve. The 58% is the real ceiling right at the line.

## Hand-waves and gaps

Index:
- methodological yellow-flag | "In the tournament settings, the system may exhibit better accuracy" | p.15
- assumed knowledge, otherwise sound | "the proposed method using monocular vision does not directly provide precise information of the height of the shuttlecock" | p.16
- methodological yellow-flag | "we did not evaluate the impact of distance on the accuracy of service judgment calls" | p.16

**Unfalsifiable optimism.** They argue the lab numbers understate real performance and that
tournaments would score higher ("In the tournament settings, the system may exhibit better
accuracy", p.15). The reasons given (lighting, markers on the shuttle) are plausible, but
no tournament accuracy is measured, so the claim cannot be checked.

**Leans on unpublished work.** The honest admission that monocular vision gives no direct
height ("the proposed method using monocular vision does not directly provide precise
information of the height of the shuttlecock", p.16) is immediately softened by pointing to
"initial unpublished research" that supposedly recovers height with calibration. That is
the crux for anyone without their rig, and it sits outside the paper.

**Distance untested.** They did not test how camera-to-server distance affects accuracy
("we did not evaluate the impact of distance on the accuracy of service judgment calls",
p.16), even though parallax is their own stated error source.

## Relevance to the project (as of 2026-07-12)

Bottom line: serve-height legality is a real objective rule, but it is not a clean skill
dimension for us, and our pipeline cannot produce it without a purpose-built rig this paper
depends on.

**Is serve legality an assessable dimension?** As a rule, yes, it is objective and
video-checkable, which is why it fits the "objectively assessable from video" test on
paper. In practice this paper assesses it only by cheating the geometry: a camera physically
placed on the 1.150 m plane so above-or-below is a direct pixel read. That is officiating
hardware, not video analysis of match footage.

**What our pipeline would need that it lacks.** Three things.
1. Serve and hitting-instant detection. Our pipeline has no serve detection (a known gap),
   and this paper's whole timing hinges on catching frame N at contact.
2. Absolute shuttle height at contact. CourtKeyNet gives a ground-plane homography, which
   recovers positions on the floor, not height above it. Height off the plane is not
   recoverable from one ground homography without extra constraints. The paper confirms
   monocular vision gives no direct height (p.16) and only sidesteps this with its
   calibrated placement, which we do not have.
3. Frame rate. They run at 70 fps and stress that temporal accuracy near contact drives the
   error. They also criticise a prior 25.8 fps method as too slow. Club and broadcast
   footage is usually 25 to 30 fps, right in the range they call inadequate.
   So extraction is X (infeasible from current outputs on arbitrary video). A softer proxy,
   serve-height consistency rather than legality, would be a heavy B (build) and still needs
   items 1 and 3.

**Does fault rate separate skill?** No, or at best unclear. The 1.150 m limit is a
compliance rule enforced by service judges at officiated matches. Club players (our A to D
targets) usually play with no service judge, so the fault is rarely called and rarely
constrains them. Serving high is a deliberate advantage (a flatter, harder-to-return
trajectory, per the paper's own framing), so a fault is a tactic that crossed a line, not a
lack of skill. Fault rate is a rules-discipline signal. A weak skill link might exist
through serve-height consistency (tight, repeatable serve height reads as control), but that
is untested here and is a different measurement from legality.

**Covered already?** No. Ari's evidence table covers timing variables and flags gaps in
club-level validation, posture, error profiles, and within-video comparison. Serve legality
is not in the table, and this paper does not fill any of those gaps. It is best logged as
out of scope for skill grading rather than as a new dimension to chase.

**Liftable regardless.** Two things transfer even though the task does not. First, the
monocular-height caution: any height-based feature we invent needs either a calibration
trick or a second view, because one ground homography will not give height. Second, the
detection evidence: YOLOv5 at 0.99 mAP over six venues is a useful reference point that
shuttle and racket detection on real venue footage is tractable.

## Red-team (folded 2026-07-12)

A fresh-context pass checked every quote and number against the canonical copy. All folded,
no standing disagreements. Changes made: corrected the player-filter quote page to p.15,
added the detected-serves selection-bias concern (p.12) and the two-server sample (p.8),
retagged the distance-gap from padding to yellow-flag, and softened the RoC/MoC praise to
note they are self-defined and unvalidated. All numbers verified correct (58/16%, mAP
0.99 and 0.956 on 968 images, 25,235 images, 70 fps, 200 Hz, 1.150 m, RoC 23.6/36.4/21.8 mm).
