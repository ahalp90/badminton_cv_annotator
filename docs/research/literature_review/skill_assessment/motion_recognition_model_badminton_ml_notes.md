<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team.
     Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# The analysis of motion recognition model for badminton player movements using machine learning

**Verdict**: Skip. This is a five-class stroke-type classifier dressed up with quantum-computing language, and it does less than the project pipeline already does. It classifies which of five stroke types a clip shows, not how well the stroke was played, so it touches skill assessment nowhere. The paper also carries several internal contradictions that undercut trust in the numbers.
**Status**: Reviewed by Claude, 2026-07-12. Not independently re-read by Curtis.

## Identity

- DOI: 10.1038/s41598-025-02771-9
- Authors: Xuanmin Zhu, Lizhi Liu, Jingshuo Huang, Genyan Chen, Xi Ling, Yanshuo Chen (meta said "unknown"; taken from the paper)
- Venue / year: Scientific Reports, 2025
- Scope, one line: Trains SVM, 3D CNN, and a "Quantum CNN" to label badminton broadcast clips as one of five stroke types (serve, forehand, backhand, forehand volley, backhand volley) from OpenPose joint features, and reports the QCNN as best.
- Canonical copy: restricted licence; held privately at 595-personal-notes/archive/restricted_papers/motion_recognition_model_badminton_ml.md
- Source PDF: restricted licence; held privately at 595-personal-notes/archive/restricted_papers/motion_recognition_model_badminton_ml.pdf
- Note drafted: 2026-07-12

## What they built

A stroke-type classifier over broadcast video. They pulled clips from four professional matches (2020 Tokyo Olympics, 2023 China Masters, 2023 Sudirman Cup, 2023 China Open), ran OpenPose on the frames to get 2D joint positions, and computed features such as joint angles, angular velocity, and acceleration. They then trained three model families to sort each clip into one of five stroke types.

The three families are a polynomial-kernel SVM, a 3D CNN, and a hybrid "Quantum Convolutional Neural Network" (QCNN) that the paper presents as the contribution. The QCNN is described as a quantum convolutional layer (two qubits, "quantum state encoding", a "Quantum ReLU") feeding a classical fully connected head.

They report per-stroke ACC, precision, recall, and F1 (Table 2), confusion matrices (Fig 5), a noise-robustness sweep at three Gaussian noise levels (Table 3), decision-tree feature importance (Table 4), a training/inference time table (Table 5), and a comparison against RF, KNN, DT, LSTM, and SACNN (Fig 7). Across all of these the QCNN comes out on top, with a headline overall accuracy of 0.965.

## What holds up

The feature-engineering recipe is ordinary but sound in outline. Pulling joint angles, angular velocity, and acceleration from pose keypoints, then Z-score normalising them, is standard and reproducible in principle.

The relative ranking of the classical models is unsurprising and internally sensible. The 3D CNN beats SVM on the harder volley classes (Table 2), which is what you would expect from a model that sees the temporal dimension. That part of the story is plausible.

That is close to the full list. The evaluation tables are laid out clearly, but as the concerns below explain, the numbers behind them do not reconcile with each other or with the stated method.

## Methodology concerns

<!-- Index first, one line per entry: flag | "short verbatim quote for ctrl+F" | p.N -->

- methodological red-flag | "identifying malicious code" | p.8
- methodological red-flag | "high-frame-rate cameras and inertial sensors to record posture information" | p.1
- methodological red-flag | "A total of 540 samples were collected" | p.9
- methodological red-flag | "Quantum ReLU" | p.10
- methodological yellow-flag | "the accuracy of QCNN reaches 0.965" | p.14
- methodological yellow-flag | "the F1-score of QCNN increased from 0.84 to 0.947" | p.14

**A malware-detection sentence is left in the method.** p.8. The QCNN fully connected layer is said to complete "the task of detecting and identifying malicious code." Badminton strokes are not malicious code. This is copied from a different paper's template and never edited out. It is small in itself, but it shows the text was assembled rather than written, which lowers trust in everything else described in that section.

**The data source contradicts itself.** p.1 versus p.8. The abstract says data was collected "using high-frame-rate cameras and inertial sensors to record posture information." The methods say the raw data came from "video recordings of broadcast and television programs" of four professional matches. Broadcast TV footage is not inertial-sensor capture, and you cannot mount sensors on Olympic players through a TV feed. One of these two accounts is false, and the paper never resolves which.

**The sample count does not add up.** p.9. The paper says "A total of 540 samples were collected" covering five stroke types, then says "Each stroke type was executed 30 times." Five types at 30 each is 150, not 540. The two figures cannot both be right, and the paper gives no breakdown that reconciles them. With the dataset this small and this unclear, the four-decimal accuracy figures are not credible.

**The quantum framing is decorative, not demonstrated.** p.10. Table 1 lists the QCNN as two qubits with a "Quantum ReLU" activation and "Quantum state encoding." "Quantum ReLU" is not a defined operation in quantum machine learning. No quantum framework, simulator, or hardware is named anywhere; the only stack reported is TensorFlow 2.1.3 on a GTX 3060. A genuine two-qubit circuit could not carry the classification load claimed here. The most likely reading is a classical network relabelled with quantum vocabulary, which makes the paper's sole novelty claim unsupported.

**The headline accuracy does not match the paper's own results.** p.14 versus p.11 and p.12. Fig 7 and the conclusion put QCNN overall accuracy at 0.965. The QCNN confusion matrix (Fig 5) has diagonal rates of 90, 88, 86, 85, and 88 percent, which average about 87 percent. The per-stroke ACC in Table 2 tops out at 0.905 and averages near 0.88. So two separate reports of the same model in the same paper sit around 0.87 to 0.88, and the third jumps to 0.965 with no explanation of where the gain comes from. The three figures cannot all describe the same test set.

**The cross-paper comparisons are not like-for-like.** p.14. The paper says QCNN's F1 "increased from 0.84 to 0.947" against an LSTM from a different study, and accuracy "improved from 0.909 to 0.945" against another. These are different datasets, different tasks, and different splits, so the comparison shows nothing about QCNN's merit. It is presented as if it were a controlled result.

## Hand-waves and gaps

<!-- Same index format as methodology concerns. -->

- methodological yellow-flag | "the relative position between the player's hand and the racket handle" | p.8
- methodological yellow-flag | "approved by Faculty of Education, Silpakorn University Ethics Committee" | p.17
- methodological yellow-flag | "The dataset can be accessed at" | p.9
- academic padding | "combining the theoretical frameworks of quantum mechanics and machine learning" | p.1

**Grip type cannot be read from this footage.** p.8. The paper lists grip type as a feature extracted "by analyzing the relative position between the player's hand and the racket handle." The video is 352 x 288 pixels of broadcast footage, and OpenPose does not track fingers or the racket. Resolving how a hand sits on a handle at that resolution is not possible. The feature is claimed but cannot have been measured as described.

**The ethics statement does not match the data.** p.17. The paper reports human-participant ethics approval and written informed consent. The data is broadcast footage of professional tournament matches. Broadcast footage of public professional events does not involve consenting study participants, so the ethics paragraph appears to belong to a different study or a different data source than the one actually used.

**The dataset link is not a dataset.** p.9. The "dataset can be accessed at" a generic gitcode datasets homepage, not a specific record, and the data-availability statement separately says data is available "on reasonable request." There is no reproducible dataset here.

**The confusion matrices look illustrative, not measured.** Fig 5, p.12. The SVM matrix is built almost entirely from multiples of five (70, 65, 60, 55, 10, 15, 20), and every row sums to exactly 100. Real per-class confusion counts on 150 to 540 samples would not land on round percentages like this. They read as hand-drawn examples rather than computed results.

## Relevance to the project (as of 2026-07-12)

This paper does not contribute a new assessable dimension or a liftable method. It is a stroke-type classifier, and the project already has a stronger one.

New dimension or method: none. The task is "which of five stroke types is this clip," which is stroke identity, not skill. Nothing here scores how well a stroke was executed, and nothing separates skill levels. Every subject is a professional, so the paper never shows its features track a skill gradient at all, let alone at club level.

Against the pipeline: redundant and weaker. BST-X already classifies 14 stroke classes; this paper does 5. RTMPose gives richer keypoints than the OpenPose 2D joints used here. The feature list (arm angle, twist angle, footwork, step length) is generic pose-derived engineering that the pipeline can already produce. There is no method to borrow that the project does not have in better form.

Against the evidence table (club A/B/C/D grading): contributes nothing. The evidence table needs dimensions with tested level or outcome correlations and cheap extraction. This paper offers no level correlation, no club data, no outcome link, and no within-video relative measure. Its feature-importance table (arm angle highest) is about which features help tell stroke types apart, not which features track player skill.

Extraction cost: X (nothing worth extracting). There is no metric here that maps to grading. The stroke-type output is already covered, and the "quantum" method is not trustworthy enough to reimplement even if it were novel.

Covered already: yes. Stroke classification is a solved part of the pipeline, done at higher class count and higher input quality.

Bottom line: log as a redundant classifier, not a source for the evidence table. No follow-up warranted.

## Red-team pass (2026-07-12)

A second read of the canonical copy against this draft. Checks and changes:

- Verified all ten ctrl+F quotes land in the canonical copy and sit on the pages cited.
- Reworked the headline-accuracy concern. The first draft argued that overall accuracy "cannot exceed the accuracy of every class," which is loose because per-class one-vs-rest accuracy has different semantics from overall accuracy. Replaced it with the airtight version: the QCNN confusion-matrix diagonal averages about 87 percent and Table 2 averages near 0.88, yet Fig 7 claims 0.965. Three reports of one model that do not agree is the real problem, and that framing does not depend on any accuracy-definition subtlety.
- Held the quantum-framing red flag but kept it hedged. "Quantum ReLU" being undefined and no quantum framework named is fact; "classical network relabelled" is stated as the most likely reading, not proven. That hedge is correct and stays.
- Held the confusion-matrix "illustrative" flag. Checked the arithmetic: a 15 percent test split of even 540 samples gives about 16 samples per class, whose percentages would fall on multiples of about 6.25, not the multiples of 5 seen in the SVM matrix. The flag is sound but it is a suspicion, not proof, so it stays in Hand-waves rather than the harder concerns list.

Genuine disagreements left standing:

- None material. The only judgement call is how hard to read the quantum framing. I treat it as decorative and unsupported. A reader who takes the two-qubit description at face value would call it a weak but real quantum method rather than a relabelled classical one. Either way the paper adds nothing the project needs, so the verdict does not move.
