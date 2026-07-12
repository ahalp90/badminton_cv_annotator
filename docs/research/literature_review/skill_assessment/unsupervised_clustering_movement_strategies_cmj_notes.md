<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team.
     Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# Unsupervised Clustering Techniques Identify Movement Strategies in the Countermovement Jump

**Verdict**: Skim as a method template. The clustering recipe is clean and cheap to copy, but the domain is vertical jump, the k-selection is subjective, and the one honest validation leans on an external label the project does not have. Take the pipeline shape and the caution, not the biomechanics.
**Status**: Reviewed by Claude 2026-07-12, not independently re-read by Curtis.

## Identity

- DOI: 10.3389/fphys.2022.868002
- Authors: Bird, Mi, Koltun, Lovalekar, Martin, Fain, Bannister, Vera Cruz, Doyle, Nindl
- Venue / year: Frontiers in Physiology, 2022
- Scope, one line: Cluster 668 Marine officer candidates by 10 countermovement-jump force-plate and motion-capture variables, then check whether the clusters differ in later injury rates.
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/unsupervised_clustering_movement_strategies_cmj.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/unsupervised_clustering_movement_strategies_cmj.pdf
- Note drafted: 2026-07-12

## What they built

No model. This is a standard tabular clustering pipeline applied to jump biomechanics.

The recipe, in order (Figure 1 has the full flowchart):

1. Each subject performs three maximal countermovement jumps, measured by force plates and markerless motion capture; only the jump with the highest jump height is carried forward.
2. That jump is reduced to 10 scalar summary features: 4 kinetic (braking rate of force development, braking net impulse, propulsive net impulse, peak relative propulsive power) and 6 kinematic (braking phase duration, propulsive phase duration, max hip/knee/ankle flexion, dynamic valgus). These are per-jump aggregates, not waveforms.
3. Scale every feature to mean 0, standard deviation 1.
4. Screen for redundant features with a Spearman correlation matrix (Figure 2); drop one of any pair with |r| > 0.85. Nothing crossed the threshold, so all 10 stayed. Strongest pair was braking RFD vs braking phase at r = -0.84.
5. Cluster the 10 scaled features with plain k-means.
6. Pick k by eye: an elbow plot (Figure 3) plus three trial runs at k = 2, 3, 4 shown as 2-D PCA scatter plots (Figure 4), choosing the k with the least visual cluster overlap. They landed on k = 3.
7. Interpret the clusters after the fact using an external label they held back: proportion of each cluster that later got a musculoskeletal injury. Fisher's exact test compared those proportions; one-way ANOVA compared the 10 features across clusters.

Important detail: PCA is only used to draw the 2-D pictures. The clustering itself runs on the full 10 scaled features. There is no dimensionality reduction before k-means.

## What holds up

The pipeline is minimal and reproducible in scikit-learn in an afternoon: scale, correlation-prune, k-means, elbow, PCA plot. Nothing exotic.

The external-label idea is the transferable part. They clustered on features alone with no injury information, then asked whether a variable they never fed the model (injury) split across clusters. It did, cleanly and monotonically: low-risk cluster 13.8% injured, moderate 22.5%, high 30.5%, p < 0.001, relative risk 2.2 for high vs low. That is a real way to show clusters carry outside meaning rather than just partitioning noise.

The feature set is interpretable, so each cluster gets a plain-language movement description (efficient vs slow-and-deep). Interpretability survives because they never blended the features into opaque components before clustering.

## Methodology concerns

- methodological yellow-flag | "demonstrating that k-means clustering performed well at separating groups for each of the ten variables (p < 0.001)" | p.6
- methodological yellow-flag | "The elbow method is a subjective measure, evaluating for a" | p.6
- methodological yellow-flag | "the percent of female MOCs were primarily distributed to high-risk cluster (41%)" | p.6

The ANOVA step is circular. They cluster on 10 features, then run ANOVA showing the clusters differ on those same 10 features. That result is guaranteed by construction and proves nothing about cluster quality. The only non-circular check is the injury-proportion test, which uses a held-back label.

The cluster count is chosen subjectively. The elbow bent at both k = 2 and k = 3, and the tie was broken by eyeballing PCA overlap. No silhouette score, gap statistic, or stability check appears anywhere. A different viewer could defend k = 2.

The dominant cluster axis partly encodes sex, not movement quality alone. Female candidates piled into the high-risk cluster (41% of it, versus under 4% of each other cluster), while males spread across all three. So "movement strategy" here is entangled with body size and sex. The clusters recovered an anthropometry axis as much as a technique axis.

## Hand-waves and gaps

- academic padding | "The datasets presented in this article are not readily available" | p.12
- assumed knowledge, otherwise sound | "K-means clustering (MacQueen, 1967) is an unsupervised technique that initially randomly assigns cluster centroids" | p.5

Data is not shared, so the result cannot be reproduced or re-clustered.

k-means depends on its random start, but they report no seed sensitivity, no repeated runs, and no cluster-stability metric. With an already subjective k, this leaves the exact three-cluster solution unpinned.

## Relevance to the project (as of 2026-07-12)

This sits against research question 2: cluster unlabelled player performance to separate skill levels with no labels. The paper is a methodology exemplar even though the sport differs.

What transfers, step by step:
- Feature reduction to interpretable per-unit scalars: reuse. This is the model for turning our pose time-series, stroke sequences, and court positions into a fixed-length per-player feature vector. Their features are per-jump aggregates, so the pattern is aggregate-then-cluster, not cluster-on-waveforms.
- Scale to mean 0 / SD 1: reuse directly.
- Correlation prune at |r| > 0.85: reuse directly, cheap and sensible.
- Plain k-means on scaled scalars: reuse, trivial.
- k-selection by elbow plus PCA-overlap eyeballing: reuse the shape but strengthen it. Add silhouette and a stability check, because our data has no ground-truth k either.
- Validation by external label proportions: reuse the idea, but note the catch below.

Skill or style: this paper is a caution, not a proof. Its clusters tracked injury risk and, underneath that, sex and body size. It never claimed to rank skill. Read across to badminton, unsupervised clustering on pose and stroke features will most likely surface a blend of playing style, handedness, body type, and skill, with no guarantee that the biggest axis is skill. Movement strategy maps onto style variants at least as much as onto a quality ladder. Their own top split being sex is the warning.

The external-label trick is exactly our bottleneck. They could validate because they had a held-back label (injury). Our setting is unlabelled by design. To copy their validation we still need some outside signal per player: known A/B/C/D club level on a subset, match win/loss, or rally outcomes. Without any such anchor we can form clusters but cannot show they mean skill rather than style.

Extraction cost: low for the clustering itself, moderate-to-high for the feature engineering upstream. The back half (scale, prune, k-means, elbow, PCA) is a day in scikit-learn. The real work is deciding which scalar summaries of our time-series capture skill, and finding even a partial external label to interpret the clusters against. The clustering is the easy 20%.

Covered already by Ari's evidence table: no. The table covers timing variables and skill dimensions, not clustering methodology. This adds a method template, not a new dimension.

## Red-team residue

Settled 2026-07-12. The verdict stays skim. The back-half recipe is a day to rebuild and the jump biomechanics carry nothing across to badminton, so a deep read earns little. But this is the closest method exemplar in the pile for research question 2: it clusters an unlabelled population and shows the clusters tracking sex and body type rather than a skill ladder. So it keeps a raised citation role. It is the canonical illustration of the clustering caution (clusters find style and physique, not skill), and the clustering write-up should cite it as that even though it is not deep-read.
