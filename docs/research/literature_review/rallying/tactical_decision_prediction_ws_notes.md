<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# Prediction model and technical and tactical decision analysis of women's badminton singles

**Verdict**: Skim for the variable list, do not trust the model. It is a single-player case study of An Se-young that classifies already-finished rallies as won or lost. The 87.5 percent headline comes off a test set of about 16 rallies. Both tree models rank rally length as the top feature and final-shot landing zone second, and every feature is measured only after the rally ends, so this sorts finished points rather than predicting them. Useful as a menu of stroke and court-zone variables and as a reminder that rally length plus shot placement carry most of the signal.
**Status**: Reviewed by Claude 2026-07-12, not independently re-read by Curtis.

## Identity

- DOI: 10.1371/journal.pone.0312801
- Authors: unknown
- Venue / year: PLOS ONE, 2024
- Authors (from paper): Hanguang Yuan, Yaodong Wang, Kairan Yang, Yulu Bin (China University of Mining and Technology Beijing; Hunan Normal University; Beijing Sport University)
- Scope, one line: Case study of one elite player (An Se-young) over 10 of her 2023 matches against the world top five, classifying each scoring or losing rally with five off-the-shelf machine learning models and reading tactical patterns off descriptive shot shares.
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/rallying/tactical_decision_prediction_ws.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/rallying/tactical_decision_prediction_ws.pdf
- Note drafted: 2026-07-12

## What they built

The paper hand-codes rallies from 10 matches into a small table. Each rally becomes one row. The columns are: total number of strikes; the court zone and technique of the third-from-last, penultimate, and final shot; the serving zone; and the landing zone of the point-ending shot. Court zones use a 9-cell-per-side scheme (Fig 1, Table 1) and techniques use a 12-item code (Table 2). The label is whether An won the rally.

They feed these features to five standard classifiers: decision tree, random forest, XGBoost, support vector machine, and k-nearest neighbours. A 90/10 train/test split is used with light cross-validation. The support vector machine with an RBF kernel scores best at 87.5 percent (Table 4, Fig 4). Random forest and XGBoost feature importances (Fig 5) then drive a descriptive read of An's style: rally length and final landing spot matter most, and she wins mostly through long rallies finished from the backcourt rather than through outright smashes.

The rest of the results are descriptive shot shares: rally-length bands for scoring versus losing (Fig 6), technique mix across the last three shots (Figs 7 and 8), and court-zone mix across the last three shots (Figs 9 and 10). All figure numbers are transcribed into the canonical copy.

## What holds up

The variable design is sensible and matches our pipeline vocabulary. Coding each rally by the technique and court zone of its last three shots, plus total rally length, is a clean way to summarise a point. The stroke labels (high clear, drop, kill, net shot, net block, lift, drive, push, and so on) overlap heavily with a 14-class stroke classifier, and the 9-zone court scheme maps onto a homography grid.

The feature-importance ranking is the most portable finding. Across both tree models, number of strikes (0.20 to 0.23) and landing location (0.18 to 0.20) dominate, and the last-three-shot zone and technique features trail at 0.08 to 0.14 (Fig 5). Rally length separating scoring from losing is also consistent: 61 percent of scoring rallies run past 11 shots versus 53 percent of losing rallies, and short serve-receive rallies cost her points more often (17 percent of losing rallies versus 9 percent of scoring, Fig 6).

The descriptive shot-share tables are internally coherent and readable, and the paper is honest that it is a limited case study.

## Methodology concerns

<!-- Index first, one line per entry: flag | "short verbatim quote for ctrl+F" | p.N -->

- yellow | "the data set was randomly divided into training set and test set according to the ratio of 90% and 10%" | p.9
- yellow | "the scoring and losing situations as the \"output\" of the model" | p.9
- yellow | "the poor performance of random forest and XGBoost may be due to the low data dimension" | p.13
- yellow | "such as a small sample size, simplistic criteria for personal style classification, and the lack of consideration for differences in players' handedness" | p.7

**Tiny test set, headline accuracy is noisy.** "the data set was randomly divided into training set and test set according to the ratio of 90% and 10%" (p.9). The paper says it leans on cross-validation to cope with the small sample, but every reported accuracy is an exact multiple of 1/16 (56.25, 62.50, 68.75, 75.00, 87.50 percent), which points to a single held-out test set of only about 16 rallies rather than averaged folds. On 16 cases, one or two flips move the number by 6 points. The 87.5 percent headline and the gap between models cannot be read as stable.

**Every feature is retrospective, and one is outcome-correlated by construction.** The label is whether the rally was won: "the scoring and losing situations as the \"output\" of the model" (p.9). One input is "the scoring and losing location", the court zone of the point-ending shot (p.9), which is the second-ranked feature in Fig 5. It is a 9-cell zone code, not the win/loss label, so it does not leak the answer deterministically (Table 3 even codes a "Compulsory" error rally as a win). But that feature only exists once the rally has ended, so it cannot support the "prediction" framing and its high importance partly reflects that it is measured at the moment of the outcome.

**Feature list is inconsistent across the paper.** The text lists serving location as an input (p.9), but it appears in neither the correlation matrix (Fig 2, 7 inputs plus Score) nor the importance chart (Fig 5, 8 inputs). Landing location appears in Fig 5 but is absent from Fig 2. The three descriptions of the feature set do not agree, which undercuts both the correlation argument and the importance-driven tactical read.

**Feature importances come from weak models.** The style read leans on random forest and XGBoost importances (Fig 5), but the authors say "the poor performance of random forest and XGBoost may be due to the low data dimension" (p.13). Those two models scored only 62.5 percent. Importances from models that barely beat a coin are shaky ground for tactical claims.

## Hand-waves and gaps

<!-- Same index format as methodology concerns. -->

- padding | "such as a small sample size, simplistic criteria for personal style classification, and the lack of consideration for differences in players' handedness" | p.7
- yellow | "An's playstyle is sustained and unified; she does not seek continuous pressure" | p.1
- yellow | "the \"input\" and \"output\" multidimensional data of the dataset are downscaled to a two-dimensional space, and then clustered using K-means" | p.10

**"Prediction" is retrospective classification.** Every input feature is drawn from a completed rally: total strikes and the last three shots. Nothing is known before the point ends. The model cannot predict a shot or an outcome during live play, so the "prediction model" framing oversells what it does. It sorts finished rallies.

**Circular check of "learnable structure".** To argue the dataset has learnable features, section 4.1.2 runs PCA on "the \"input\" and \"output\" multidimensional data of the dataset ... and then clustered using K-means" (p.10) and reports that the win and loss clusters separate (Fig 3). Folding the win/loss label into the data before clustering builds in the separation it then claims as evidence. The exercise proves little.

**Style narrative outruns the evidence.** Claims like An "does not seek continuous pressure, but rather exploits and maximizes her aggression following any mistake" (p.1) are prose laid over descriptive shot shares. There is no test that ties these patterns to winning versus losing beyond the raw percentages, and no comparison player, so the "signature style" reading is one plausible story among several.

**Missing evaluation basics.** No confusion matrix, no precision or recall, no per-class breakdown, and no confidence intervals. The correlation matrix (Fig 2) shows every feature it includes correlating weakly with the score label (|r| <= 0.15), which sits awkwardly next to an 87.5 percent classifier and is never reconciled. Note the matrix omits landing and serving location, so it does not even cover the second-ranked feature.

## Relevance to the project (as of 2026-07-12)

Weak as a grading source, useful as a variable menu. This is elite women's singles, one player, no club-level data, so it does nothing for the A/B/C/D level-validation gap the evidence table still has open.

What is liftable is the framing of a rally as a short feature vector our pipeline can already build. Cheap to extract now (E): rally length in strikes (from rally segmentation), and the technique and court zone of the last three shots (BST-X stroke class plus CourtKeyNet homography). The technique-mix-by-shot-position and zone-mix-by-shot-position shares (Figs 7 to 10) are computable per player and could feed a within-video comparison.

What needs missing capabilities (B or X): the score label itself, which needs rally-winner detection (X, currently missing); serve detection for the serve-receive band in Fig 6 (X, missing); and the forced-versus-unforced error tag the paper mentions (X, judgement we cannot make yet). The single most predictive feature here, final-shot landing location, is cheap to measure but only meaningful once we know who won the rally.

New assessable dimension for the evidence table: rally-length share by outcome (fraction of a player's won versus lost rallies falling in short, medium, and long bands). It refines the timing-variable row from pure duration toward an outcome-linked distribution, but it is gated on rally-winner detection. Do not add it as a live dimension until that capability exists. The paper offers no tested level correlation, only a within-player scoring-versus-losing contrast on one elite athlete.

## Red-team disposition

A fresh agent re-read the paper and the draft. All quotes verified verbatim with correct pages. Points folded in: the verdict now names rally length as the top feature and landing zone second (matching Fig 5); the "outcome leakage" flag was softened to a retrospective, outcome-correlated feature since the landing zone is a court code and not the win/loss label (Table 3 codes a "Compulsory" error as a win); the |r| <= 0.15 line was qualified because Fig 2 omits landing and serving location; two new items were added, the inconsistent feature list across text/Fig 2/Fig 5 and the circular PCA-on-input-and-output check (Fig 3); and the tiny-test-set entry now notes the exact-1/16 accuracies contradict the stated cross-validation. No red-team point was rejected, so there are no standing disagreements.
