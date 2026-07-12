<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# Network and attribute-based clustering of tennis players and tournaments

**Verdict**: Skim, do not full-read. The headline method needs a match-participation network the badminton project cannot build, and the clusters it does produce split on playing style and court surface more than on skill. Useful mainly as a cautionary example.
**Status**: Reviewed by Claude 2026-07-12, not independently re-read by Curtis.

## Identity

- DOI: 10.1007/s00180-024-01493-2
- Authors: D'Urso, De Giovanni, Federico, Vitale (listed on the paper; acquire metadata said unknown)
- Venue / year: Computational Statistics, 2024
- Scope, one line: cluster 136 elite ATP players and 64 tournaments from one 2023 season, using both their performance attributes and the network of who entered which tournament.
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/network_attribute_clustering_tennis_players.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/skill_assessment/network_attribute_clustering_tennis_players.pdf
- Note drafted: 2026-07-12

## What they built

Two clustering pipelines run side by side on the same data, then compared.

Data has three parts. A player matrix X with 21 numeric performance attributes for 136 players. A tournament matrix Y with 18 attributes (numeric plus three qualitative: surface, indoor/outdoor, nation) for 64 tournaments. A binary participation matrix A, where an entry is 1 if a player entered a tournament draw. A is the only link between the two sides.

Pipeline one, the paper's own proposal: spatially-corrected fuzzy Partition Around Medoids (PAM). It clusters players on X and tournaments on Y separately. The trick is a "spatial" penalty term added to each objective. That term is built from A by cosine similarity: two players are "similar" if they entered the same tournaments, two tournaments if they drew the same players. The penalty nudges players who share tournaments toward the same cluster. Euclidean distance for players, Gower distance for tournaments (Gower handles the qualitative columns). Two tuning knobs per side: m for fuzziness, beta for how much the network penalty counts.

Cluster count is chosen by the fuzzy silhouette index over a grid of m and cluster count. Players land on 3 clusters (m1=1.15, silhouette 0.370); tournaments on 2 clusters (m2=1.05, silhouette 0.293). A unit is called a member if its top fuzzy membership clears 0.6 (players) or 0.7 (tournaments), otherwise it is a "fuzzy" unit. 6 players and 4 tournaments end up fuzzy.

Pipeline two, the comparison: a Degree-Corrected Stochastic Blockmodel (DCSBM) fit to A alone, ignoring all attributes, using the R package `greed`. It jointly finds communities on both sides of the bipartite network and picks the cluster count internally by optimising the Integrated Complete-data Likelihood. It returns 2 player clusters and 4 tournament clusters.

The clusters get labelled by reading the medoid attribute values (Figs 3-6) and by cross-tabulating the two pipelines (Tables 7, 8).

## What holds up

- The engineering is competent and the write-up is honest about the machinery. Gower distance for mixed data and fuzzy silhouette for cluster count are standard, sensible choices.
- The two-pipeline comparison is a fair way to show what the attributes add over the bare network. The contingency tables (Tables 7, 8) are real evidence, not decoration.
- The clusters are interpretable and the interpretations are consistent across both pipelines: one axis is serve-heavy versus return-heavy play, which lines up with hard court versus clay. That much is genuinely in the data (Figs 3-6).
- They are candid that the network penalty is fragile and that the two models answer different questions.

## Methodology concerns

- methodological yellow-flag | "so that they would be low enough not cause the partition to collapse into only one cluster" | p.1698
- methodological yellow-flag | "Cluster 2 mostly represents clay court specialists and/or lower level players" | p.1704
- methodological yellow-flag | "obtaining the optimal values of _m_ 1 = 1.15 , _C_ = 3 (see Table 3)" | p.1699
- methodological yellow-flag | "Here there is no external validation index for the clustering" | p.1699

The spatial penalty weights are set by hand to dodge a degenerate result, not tuned by any criterion. The authors admit the cosine matrices are dense, which means "if the weight given to the spatial term is too high ... the entire output partition collapses in one single cluster" (p.1696). So beta1 and beta2 were picked small enough to avoid collapse (p.1698). The network contribution therefore sits in a narrow safe band chosen for stability, not because that weight is right. The novelty of the paper is exactly this penalty, and it is the least principled part.

The winning cluster structure is weak in absolute terms. The best fuzzy silhouette values are 0.370 for players and 0.293 for tournaments (Tables 3, 4, on p.1699). Values that far below 1 point to loosely separated clusters, especially on the tournament side. These partitions are the best on the grid, not strong groupings.

The cluster labels conflate skill with style and surface. Cluster 2 is described as "clay court specialists and/or lower level players" (p.1704). That "and/or" is the whole problem for anyone hoping to read skill off these clusters: the method cannot tell a weak player from a clay specialist, because both show low serve numbers. See relevance section.

The DCSBM cluster count has no external check (p.1699). It is chosen by the same likelihood that fits the model, so there is no independent guard against over- or under-splitting on that side.

## Hand-waves and gaps

- academic padding | "sometimes confirming each other's outputs and sometimes highlighting something the other algorithm could not capture" | p.1710
- assumed knowledge, otherwise sound | "the _N_ = 136 players that played at least 10 matches on the ATP Tour" | p.1694
- methodological yellow-flag | "Code used in the analysis can be provided by the corresponding author on reasonable request" | p.1710
- assumed knowledge, otherwise sound | "a_{nh}\,a_{il}" | p.1695

The two models are never reconciled into one answer. The conclusion says they sometimes agree and sometimes do not (p.1710), with no summary agreement measure beyond the two contingency tables. The reader is left to decide which partition to trust.

Scope is narrow: one 2023 ATP season, 136 top male professionals who played at least 10 tour matches (p.1694), 64 tournaments. No amateurs, no women, no second season to test stability. Everyone in the sample is already an elite player, so the skill range is compressed before clustering even starts.

Neither code nor processed data ships. Both are "on reasonable request" (p.1710), so reuse means reimplementing from the description.

Two published equations carry small typos, preserved with [sic] in the canonical copy. The tournament cosine-similarity numerator (eq 2, p.1695) prints `a_{il}` where the column-cosine definition needs `a_{nl}`, and the tournament objective (eq 5) reuses the player membership `u_{s'e'}` where it should be `w_{s'e'}`. Neither changes the described method, but anyone reimplementing from the formulas should use the corrected forms.

## Relevance to the project (as of 2026-07-12)

Front-and-centre question: our project wants to cluster unlabelled amateur badminton players to separate skill levels, from per-player video features, with no tournament-results network. Does this paper help?

Short answer: the method as published does not transfer, and the paper is more useful as a warning than as a recipe.

The paper has two halves. Judge them separately.

The network half (DCSBM, and the cosine-similarity penalty that feeds the fuzzy PAM) is dead on arrival for us. Both are built entirely from the participation matrix A, meaning who entered which tournament. Our short scraped clips give per-player stats but no draw sheets, no cross-tournament identity linking, no who-played-whom graph. We cannot build A. Extraction cost for this half: X (not feasible without a results or participation network the project does not have and cannot scrape from clips).

The attribute half is the only reusable part, and only in a stripped-down form. Set the network penalty weight to zero and the spatial fuzzy PAM collapses to ordinary fuzzy PAM: standardise the per-player feature matrix, run fuzzy PAM, pick the cluster count by fuzzy silhouette, then read the medoids to label clusters. That recipe works on our data. But once the penalty is gone, nothing badminton-specific or novel remains; it is generic fuzzy clustering available from any clustering reference. Extraction cost for this half: E (easy, standard libraries), but the value is low because it is not really this paper's idea any more.

Does it separate skill or style? Mostly style and surface, only weakly skill, and it entangles the two. This is the load-bearing finding for us. In the DCSBM player split (Fig 6), the serve-versus-return style attributes pull the two clusters roughly 0.85 to 1.0 standard deviations apart (cluster B's serve stats sit about 0.6 to 0.75 SD below average, cluster A's about 0.2 to 0.3 above), while the overall-results attributes (win %, points won %) separate the clusters by only about 0.1 to 0.25 SD. So the dominant split is playing style, which tracks court surface, not who wins more. The PAM player clusters (Fig 3) tell the same story: cluster 3 is big servers, cluster 2 is weak servers, and the authors themselves label cluster 2 "clay court specialists and/or lower level players" (p.1704). They cannot pull those two apart. Cluster 1, the "default" group, quietly holds almost all the top players and sits near the global average on most stats, which means the strongest players are not isolated into their own cluster at all.

There is a structural reason the split lands on style. Of the 21 player attributes, the large majority describe serve and return mechanics (aces, first-serve-won, break points, several of them near-collinear), and only about four or five are overall-results measures. Standardised Euclidean distance weights every attribute equally, so the serve-return block dominates the distance by sheer count. A clustering built on this feature set is pushed toward style archetypes before the algorithm even runs. That is a feature-design lesson for us, not a tennis quirk.

Takeaway for the badminton clustering question: unsupervised clustering on raw performance stats is at real risk of finding style archetypes (attacking vs defensive, fast vs patient) rather than skill tiers, and of confounding the two when a style correlates with level. If we go the clustering route we need features chosen to track skill specifically, or a validation step that checks a found cluster against an outside skill signal, because silhouette-optimal clusters can be style clusters wearing a skill label. This paper is good evidence for that caution and thin as a method to copy.

Covered already versus the profile's evidence table: the timing and rally-density dimensions the table already covers are not what this paper measures, so no overlap there. It does not add a new liftable skill dimension. Its contribution to our thinking is the negative result above.

## Red-team pass (2026-07-12)

A fresh agent checked every quote against the canonical copy and attacked the claims. Folded in:

- Fixed a page error: the code-availability quote is on p.1710, not p.1709.
- Sharpened the Fig 6 reading. The earlier draft said style attributes "differ 0.6 to 0.75 SD between the two clusters"; those were one cluster's magnitudes, so the between-cluster gap is actually about 0.85 to 1.0 SD. Corrected, which strengthens the point.
- Dropped the Poisson-approximation flag. An average edge rate below 0.3 is the standard condition under which Poisson approximates Bernoulli, so the paper's claim was fair and did not need a test.
- Added the weak-silhouette concern (0.370 players, 0.293 tournaments).
- Added the feature-count mechanism to the relevance section: the attribute set is serve-return-dominated, so Euclidean distance is biased toward style before clustering starts.
- Added a hand-wave note on the two preserved equation typos (eqs 2 and 5).

All factual figures were confirmed correct (cluster counts, silhouettes, sample sizes, thresholds). No substantive disagreements remain between the draft and the red-team.
