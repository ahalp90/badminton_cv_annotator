<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# Clustering of variables methods and measurement models for soccer players' performances

**Verdict**: Skim for one liftable idea. This paper clusters VARIABLES, not players. It groups a set of correlated per-player metrics into a few dimensions with no labels, which is exactly the feature-reduction pre-step the project needs before it clusters players. CLV is the cheap, off-the-shelf part worth copying. Everything after (PLS-SEM, market-value validation) does not transfer.
**Status**: Reviewed by Claude 2026-07-12, not independently re-read by Curtis

## Identity

- DOI: 10.1007/s10479-023-05185-w
- Authors: unknown
- Venue / year: Annals of Operations Research, 2023
- Scope, one line: Group 29 FIFA19 player-performance attributes into dimensions with three variable-clustering methods, embed each grouping in a PLS-SEM higher-order model, and test which grouping best tracks EA's overall rating and players' market value and wage.
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/clustering_variables_soccer_player_performance.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/clustering_variables_soccer_player_performance.pdf
- Note drafted: 2026-07-12

## What they built

No new algorithm. They take a tabular matrix of 2662 outfield players by 29 attributes (top-5 European leagues, 2018-19 season, each attribute on a 0-100 scale) and run three off-the-shelf variable-clustering methods on it, then a downstream measurement model. The whole first half operates on the columns, the 29 attributes. Players are the rows, and they are never themselves clustered.

The three variable-clustering methods:

1. CLV, cluster of variables around latent variables (Vigneau and Qannari). Groups the attributes by maximising the squared covariance between each attribute and its cluster's first standardised principal component. Hierarchical step then a partitioning consolidation. R package ClustVarLV. Produced 6 clusters. This is unsupervised: no target variable.

2. PCovR, principal covariates regression (De Jong and Kiers). Reduces the 29 predictors to K components while at the same time regressing a criterion variable on those components. A parameter alpha in [0,1] trades dimension reduction against prediction of the criterion. Leave-one-out cross-validation picked alpha = 0.5 and 4 components. The criterion here was EA's own overall rating (OVE). So PCovR is supervised: it needs a target.

3. B-MBC, Bayesian model-based clustering. A Gaussian mixture over the attributes with a Dirichlet-process (stick-breaking) prior, so the number of clusters is estimated jointly with the other parameters rather than fixed. rjags, two chains, 55,000 MCMC iterations. Produced 4 clusters. It groups attributes that measure the same latent trait, with players entering only as replicate observations.

Similarity of each statistical grouping to the expert grouping, by Rand Index: CLV 0.810 (closest), B-MBC 0.665, PCovR 0.594.

The second half embeds each grouping (plus the expert grouping) into a formative-formative PLS-SEM with a higher-order component, the overall performance. Repeated-indicators approach. They dropped attribute DEF2 for a VIF above 10, then compared the four models on path coefficients, AIC/BIC/SRMR, and how well the resulting overall composite correlated with EA's overall rating (Table 5), market value and wage (Table 6), broken out by player role.

## What holds up

CLV is a clean, cheap, label-free way to collapse a set of correlated metrics into a handful of dimensions. It needs no target variable and no ground truth. That is precisely a feature-reduction pre-step, and it runs on a plain tabular matrix with an existing R package.

The comparison is honest about its own result. All three statistical groupings differed from the expert grouping, and the authors measured the gap with the Rand Index rather than asserting one grouping was better. They also quantified where the methods land relative to each other.

They used a real external anchor. The composites built from statistical clusters (B-MBC and CLV) correlated with players' market value and wage better than the expert grouping did. General value correlations: B-MBC 0.546, CLV 0.502, PCovR 0.473, expert (sofifa) 0.440. Wage correlations follow the same order. This is an outside signal the clusters were never fed, which is the right way to argue the groupings carry meaning.

They checked collinearity and acted on it (VIF, dropping DEF2), which is the correct discipline for formative indicators where redundant inputs are the main threat.

## Methodology concerns

- methodological yellow-flag | "the criterion variable is represented by the _overall_ (OVE) variable, that is, a weighted indicator of player’s overall performance also developed by experts based on the 29 attributes" | p.6
- methodological yellow-flag | "The model selection indices and the estimates obtained with a bootstrap procedure with 5000 resamplings were not particularly in line with each other" | p.17

The three methods are not doing the same job, yet they are ranked as if they were. PCovR is supervised: it reduces the attributes toward whatever predicts EA's overall rating. CLV and B-MBC are unsupervised. When the Table 5 contest is judged by correlation with EA's overall, the supervised method starts with an advantage that has nothing to do with clustering quality. The methods should be read as answering different questions rather than as three entries in one race. In fairness, this did not distort the paper's actual conclusion: on the decisive money criterion (Table 6) PCovR came third, behind both unsupervised methods, so the concern is largely academic here.

The cluster counts also differ (6 for CLV, 4 for PCovR and B-MBC), so the Rand Index gaps and the downstream model differences partly reflect granularity, not grouping quality alone. A 6-cluster solution can sit closer to a 6-dimension expert scheme for reasons of count rather than content.

The winner is a judgment call across criteria that disagree. AIC and BIC favoured PCovR. The bootstrap then flagged a PCovR coefficient that was negative, near zero, and significant, which undercut it. The authors settled on CLV. That is a defensible reading, but it rests on weighing several conflicting signals by hand rather than on one clear metric.

The external validation partly rewards fitting EA's own recipe. EA's overall rating was itself built from the same 29 attributes, and PCovR used that rating as its training target. So "concurrent validity" measured against EA overall is not a fully independent check for at least one of the methods.

## Hand-waves and gaps

- academic padding | "with the aim of delving into predictive validity in a future research stage" | p.15
- methodological yellow-flag | "Therefore, the model selection procedure tested 10 _α_ values (ranging from 0.5 to 1, by 0.05)" | p.6
- methodological yellow-flag | "The dataset analyzed for the current study belongs to the _FIFA 20 complete player dataset_" | p.18

PCovR's alpha search stopped at the edge of its own grid. They tested alpha from 0.5 to 1, and the optimum landed at the lower bound, 0.5. Values below 0.5 put even more weight on reduction, and those were never tried, so the reported optimum may just be the grid boundary rather than a true minimum.

Only concurrent validity is tested. Predictive validity, the part that would actually show the composites forecast future value, is deferred to future work. So the strongest claim they can support is correlation with same-year value and wage.

There is a small dataset-label mismatch. The analysis uses FIFA19 attributes, while the data-availability statement points to the FIFA20 complete player dataset. The Kaggle source spans several editions, so this is probably just loose wording, but it is not pinned down.

## Relevance to the project (as of 2026-07-12)

This sits against research question 1, the feature-reduction step, more than question 2. The run-2 framing is the right lens: the project has many candidate per-player metrics and wants to reduce or organise them before clustering players by skill without labels. This paper's "clustering of variables" is exactly that reduce-and-organise step. It groups correlated performance metrics into a few dimensions.

Read the split clearly, because the title is easy to misread. The paper clusters VARIABLES, the 29 attributes. It does NOT cluster players. Players are only rows or replicates. The roles in Tables 5 and 6 are given labels used for reporting, not clusters the method discovered. So this paper cannot tell us anything about whether clustering players would recover skill.

What transfers, method by method:
- CLV: liftable. Unsupervised, label-free grouping of correlated metrics into K dimensions. Run it on our per-player metric matrix as a pre-step, then cluster players on the reduced dimensions. Cheap: ClustVarLV in R, or reimplement as PCA within candidate variable groups. One caveat: CLV still makes us choose the number of clusters by eye from a delta-T plot, and picking that count with no ground truth is the hard part in an unlabelled setting. The reduction is also modest here, 29 attributes down to 6 dimensions, with each dimension still needing a human label after the fact.
- PCovR: usable only if we have a per-player target, for example known A/B/C/D level on a subset or match outcome. Supervised, so it fits our unlabelled design poorly, but it could exploit a partial label if we get one.
- B-MBC: heavier machinery (MCMC via rjags) for the same variable-grouping job CLV does more simply. Skip unless CLV proves unstable on our metrics.
- The PLS-SEM half and the market-value validation do not transfer. Badminton has no economic-value target, and the goal is skill level, not player price.

Separates skill or style: n/a. The paper never clusters players and never separates a skill axis from a style axis. Its dimensions describe the metric set, and its only external anchor is money. It offers no evidence either way on the skill-versus-style question that dogs player clustering.

Extraction cost: E, easy and cheap, for CLV as a variable-reduction pre-step. It runs on the tabular metric matrix the pipeline already produces, uses an off-the-shelf R package, and needs no video and no labels. Any real cost is upstream and ours regardless: deciding which candidate metrics to feed it.

Covered already: no. Ari's evidence table covers timing variables and skill dimensions, not feature reduction. The sibling note (unsupervised_clustering_movement_strategies_cmj) clusters observations (subjects). This note adds the other half: a way to group the variables first, before clustering players. Different step, not previously in the notes.

## Red-team pass

A fresh agent checked the note against the paper. Every number and quote verified, no factual errors, and the central variables-not-players claim held. Folded in: softened the PCovR "unfair race" flag (PCovR still lost on the money criterion, so it is academic here), dropped the K = 20 flag (the ceiling never binds, only 4 effective clusters), added the alpha-grid-boundary gap, the cluster-count confound in the Rand Index comparison, and the CLV manual-K and modest-reduction caveats. No standing disagreements.
