<!-- Distillation + appraisal note. Claude drafted every section and ran the adversarial red-team. Status: reviewed by Claude on 2026-07-12 in a second consolidation pass; Curtis has not independently re-read it, so treat the verdict as Claude's, checked but not his personal sign-off. Flags for concerns and hand-waves:
       assumed knowledge, otherwise sound | academic padding |
       methodological yellow-flag | methodological red-flag | out of depth -->

# Badminton as a dynamic system: A new method for analyzing badminton matches based on perturbations

**Verdict**: Read for the idea, not the method. It names a real new dimension (who disrupts rally balance and how), but detection rests on an unpublished expert ruleset the pipeline cannot reproduce.
**Status**: Reviewed by Claude 2026-07-12, not independently re-read by Curtis.

## Identity

- DOI: 10.1080/02640414.2024.2323327
- Authors: Fabian Hammes; Daniel Link
- Venue / year: Journal of Sports Sciences, 2024
- Scope, one line: A manual annotation method that marks the decisive "perturbation" shot in a badminton rally and profiles players by where and how they create or concede instability.
- Canonical copy: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/rallying/badminton_dynamic_system_perturbations.md
- Source PDF: /srv/mergerfs/scratch_pool/Scratch_Data/Uni/cosc595/worktrees/research-literature-review/docs/research/literature_review/rallying/badminton_dynamic_system_perturbations.pdf
- Note drafted: 2026-07-12

## What they built

They treat one rally as a dynamic system with two attractor states, a point for Player A or a point for Player B. A rally starts balanced. The shot that tips it out of balance is the perturbation. They call the whole decisive sequence a "keyplay".

A keyplay has up to four actions:

- **Impulse**: the shot that creates the instability. This is the perturbation itself.
- **Follow-up**: the attacker's court coverage after the impulse.
- **Convert**: the attacker's next shot that tries to turn the advantage into a point.
- **Survival**: the defender's shot that tries to rebalance the rally.

A keyplay is **positive** when a good shot creates the advantage, or **negative** when a bad shot (a weak lift, a short clear) creates the instability. Direct winners and unforced errors from a balanced state also count as perturbations because they jump straight to an attractor.

Each impulse is annotated on a 3x3 court grid for shot zone and shot placement, plus shot type (Net, Lob, Defense, Drive, Smash, Drop, Clear, Other) and laterality (forehand or backhand). See Table 1 in the canonical copy.

Detection is done by a trained human rater applying an extensive ruleset. The ruleset keys on shot position, shot height and trajectory (rising, falling, roughly flat), and the time pressure on the player, and it also reads runs of shots (being forced to lift more than once in a row usually counts as instability). The full ruleset is not in the paper.

They ran three tests. The first two use six matches from the 2022 World Championships (women's and men's singles semis and finals); the third is a separate single match.

1. Rater agreement (two raters, 488 rallies).
2. Keyplay analysis against last-shot analysis (shot zone and shot type distributions, one observer's annotation).
3. A single-case player profile of the Tokyo 2021 Olympic women's final.

## What holds up

Impulse detection is repeatable at a moderate level. With a one-shot tolerance, two raters agreed on where impulses occurred at Jaccard J = .80. Classifying an impulse as positive or negative reached Cohen's kappa = .70. That is honest and reported with match-level ranges (J .73 to .91, kappa .53 to .81).

The core claim that a perturbation view differs from a last-shot view is backed by Figure 3. Direct errors cluster in the midcourt (56.4%), while negative impulses come mostly from the backcourt (40.0%). Direct winners are mostly smashes (45.5%), while positive impulses are mostly net shots (30.1%). The shot-type contrast is a strong effect (Cramer's V = .76 for errors versus negative impulses, V = .58 for winners versus positive impulses, both p < .01). This backs their point that the last shot of a rally is often just the visible end of a break that happened earlier. One caveat: these distributions come from a single observer's annotation, so they carry no inter-rater check of their own (the two-rater test covered only impulse detection, not this comparison).

The single case shows the method produces readable spatial profiles (Figure 4): Tai Tzu Ying's positive impulses concentrate in her right backcourt, her negative impulses in her left backcourt.

## Methodology concerns

- methodological red-flag | "it does not provide a precise definition of how to detect keyplays at an operational level" | p.7
- methodological yellow-flag | "available from the corresponding author, FH, upon reasonable request" | p.9
- methodological yellow-flag | "there is always room for a subjective assessment in keyplay detection" | p.8
- methodological yellow-flag | "the total number of impulses played" | p.6

The whole method turns on one judgement, and that judgement is not published. The authors admit "it does not provide a precise definition of how to detect keyplays at an operational level" (p.7) and say the ruleset is too long for the paper. The data availability statement puts the ruleset behind a request to the author, "available from the corresponding author, FH, upon reasonable request" (p.9). On-request data is normal practice on its own, but here it is the only route to the one component the paper leaves out, so it compounds the reproducibility problem rather than easing it. The paper describes an outcome that no one else can reproduce from the paper alone.

The detection is also inherently subjective. The authors say "there is always room for a subjective assessment in keyplay detection" (p.8), and the numbers agree: kappa = .70 for the positive/negative split is moderate, not strong. Two trained human badminton experts disagreed on impulse type in 69 of the shared cases, about one in seven. Any automated proxy would have to reproduce a label that even trained experts land only at moderate agreement.

The skill signal is unproven. The one player profile shows the more active player lost: "the total number of impulses played" was 83 for Tai Tzu Ying against 42 for Chen Yu Fei (p.6), yet Chen Yu Fei won the match. So raw perturbation count does not track winning, and the paper never tests whether any perturbation measure separates winners from losers or one level from another.

## Hand-waves and gaps

- methodological yellow-flag | "is the most important component for winning badminton" | p.3
- academic padding | "follow-up, convert and survival data have not been considered" | p.8

The paper asserts that creating perturbations and denying them "is the most important component for winning badminton" (p.3). This is stated as motivation, not tested anywhere in the results. It is the load-bearing assumption for using the method as a skill measure, and it is left as a claim.

Three of the four keyplay actions are defined but never used. The follow-up, convert and survival attributes are collected, then the analysis reports only the impulse, because "follow-up, convert and survival data have not been considered" (p.8). So the four-action model is mostly aspiration in this paper, and the parts that might describe how a player exploits an advantage are not shown.

The sample is elite only (World Championship and Olympic finalists) and small (six matches for the distributions, one for the profile). There is no club-level or amateur data.

## Relevance to the project (as of 2026-07-12)

**The dimension is real and new to our table; the method is not liftable.**

What the paper offers as a concept is a control-disruption dimension: who breaks the rally's balance, from where, and with what shot. Our evidence table already covers timing variables well (rally duration, shots per rally, shot frequency, work/rest density). This is different. It sits near the "error profiles" and "positional states" gaps Ari flagged, because a negative impulse is essentially an error precursor located on the court. So on the concept, covered_already = no.

Can we measure a perturbation from pipeline outputs? Partly, and not the part that matters. The annotation inputs map onto our pipeline reasonably well. Shot zone is the striker's court position at contact, which is RTMPose player keypoints projected through CourtKeyNet homography. Shot placement is where the shuttle goes, which is the TrackNetV3 track projected the same way. Shot type comes from BST-X (its 14 classes cover their 8), and shot height and trajectory come from the shuttle track. Time pressure could be proxied from inter-shot interval and player position. What we cannot get is the impulse threshold itself. That threshold is the unpublished expert ruleset, and even experts only reach kappa = .70 on it. Detecting a perturbation is not a measurement we can read off, it is a subjective label we would have to invent from scratch.

Does it need the detection the pipeline lacks? Helpfully, the core keyplay definition avoids rally-winner detection on purpose, since it keys on instability, not on who scores ("This definition does not focus on scoring"). So the impulse label itself does not need score or winner detection. But the "leads to a point" attribute and the whole winners-versus-errors comparison do need rally outcome, which our pipeline does not produce. Serve detection is not required.

Does it survive in relative within-video form? The shape fits us. Their player-profile use (analysis iv) is exactly a two-player, within-match comparison, which matches our need for relative within-video scoring. Their reliability and distribution studies instead pool keyplays across six matches, so the within-match framing is one of the method's uses, not the whole paper. Either way, the magnitude and the count both depend on the subjective detector, so the relative form inherits the same blocker.

Could perturbation frequency or magnitude separate skill levels? Unclear, leaning no on the evidence here. The only within-paper test of activity against outcome goes the wrong way (more impulses, lost match), and the data is elite only with no level contrast.

Extraction cost: **X** to lift their method, because the detection ruleset is unpublished and subjective. A crude automated proxy (for example, flag an impulse when a player is forced to lift a falling shuttle under time pressure, or when the shuttle-speed and court-position state crosses some heuristic) is a **B/X** research task of its own, not an extraction, and it would need its own validation before any level claim. If we want a momentum or control dimension, this paper is the concept and the vocabulary, not the recipe.

## Red-team disagreements

- Settled 2026-07-12. The red-team read the "within-match two-player comparison" framing as slight advocacy, since only the single-case profile is strictly within-match while the reliability and Figure 3 studies pool across matches. The softened wording stands: this is one of the method's uses, not its headline. The point that the concept applies cleanly to a single video is kept, because that is what matters for us. Nothing open.
