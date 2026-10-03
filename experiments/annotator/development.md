# How the auto-annotator developed

The auto-annotator grew from a set of motion and geometry rules into a system
that detects contacts, repairs rallies and ranks the results for review. The
central difficulty stayed the same throughout: finding most hits is much easier
than producing a rally with every hit, player assignment and clip boundary
correct.

The work below follows the problems that shaped the implementation, from the
July rule-based measurements through the learned models and court experiments
to the October refit. Results belong to their stated datasets and scoring rules;
they are not one continuous benchmark. [Evaluation](evaluation.md) explains the
changing populations and timing allowances.

## July–August: establish what the rules could recover

The first system combined shuttle movement, court geometry and player poses.
Rules identified likely live play, excluded replay-like footage, divided the
video into rough rallies and looked for sudden changes in shuttle velocity near
a player's wrist. These ingredients remain in the current system: they supply
context and candidate frames to the learned stages.

The [July measurement](runs/20260730-041328/report.md) compared eight combinations
of video, court source and shuttle sampling. Fixed reference courts helped
separate annotation problems from detection problems. Live court estimates
exposed the errors that an end-to-end system would actually encounter. The
three calibration videos also made it possible to repeat small heuristic
changes against the same inputs.

Two early experiments challenged plausible shortcuts:

- **Search further around the first and last hit.** A 90-frame buffer exposed
  19 additional correct candidate associations for each court mode, but many
  more candidates belonged to another hit or matched no labelled hit. The
  extra associations came from split-rally cases. None of the correct candidates
  in already covered rallies lay outside its predicted span. Wider matching
  windows therefore did not solve the underlying segmentation problem.
  [Boundary-search experiment](reports/boundary_search.md)
- **Change the distance used to assign a hit to a player.** Scaling
  wrist-to-shuttle distance by player height gave 80.40% accuracy on eligible
  labelled contacts in three videos. Raw pixels fell to 75.06%; projecting
  positions onto the court also regressed. All methods favoured the far player
  too strongly. A different distance formula did not repair that imbalance, and
  a court-plane projection is an imperfect model for an airborne shuttle or
  wrist. Body-height scaling survived the comparison.
  [Player-assignment experiment](reports/player_assignment.md)

These trials separated three questions: whether the right hit was ever
considered, whether it was selected, and whether the right player was assigned.
The learned stages address these questions separately.

## Late August: learn which candidate frames are hits

The contact-model pilot broadened the search before learning to score it. Its
search region covered **98.3% of labelled hits within ±10 frames while examining
31.9% of frames**, across three videos. A classifier could then reject weak
candidates without the initial rules having already discarded too many real hits.

Histogram gradient boosting, a classifier built from small decision trees,
improved timing F1 from the old rules' **72.6% to 87.4%**. Changing the rule that
merges nearby detections raised it to **88.8%**. That improvement mostly removed
extra events: 105 unmatched predictions disappeared at the cost of seven timing
matches.

The whole-rally result was much weaker: only **27 of 291 scorable predicted
sections** were completely correct. Most missed hits already had a candidate
nearby, and player assignment remained a major obstacle. More search alone
could not solve either problem. The [pilot report](../../scratch/contact_det/README.md)
records candidate coverage, scoring, event selection and error analysis
separately.

A larger study then compared nine random-forest and gradient-boosting
configurations on **40 original-ShuttleSet videos**, with model selection on
an eight-video validation group. It retained gradient boosting, raw per-frame
motion features, balanced class weights, a 0.9 contact cutoff and six-frame
merging of nearby detections. The model-choice process excluded ShuttleSet22
labels. [Full-dataset comparison](../../scratch/contact_det_full_ds_fit/baseline_report.md)

The larger study also tested an extra first-hit selector. Its best version was
correct on only 51.7% of added hits, below the experiment's 80% requirement.
Opening contacts clearly needed work, but that particular rescue rule was too
unreliable to retain.

## Late August: a good contact detector still made poor rallies

The first evaluation on **47 previously unseen ShuttleSet22 videos** reached
82.45% contact-timing F1 at ±5 frames. Player-side accuracy was 92.02% among
matched answers. Yet only **483 of 3,982 predicted sections** were fully correct.
Errors accumulated across the sequence, so strong contact-level scores did not
translate into trustworthy whole rallies.
[Cross-dataset test](../../scratch/contact_det_full_ds_fit/shuttleset22_test_report.md)

Badminton provides a useful constraint: successive hits alternate between
players. The next experiment chose the alternating court-half sequence that
best matched the evidence over the whole rally. Fully correct sections rose
from **483 to 901 at ±5 frames**, preserving every previously correct section.
At the separate ±10-frame allowance, the count rose from 524 to 995.

Other changes were less successful. A deletion model repaired 42 sections but
damaged 88 on 32 ShuttleSet development videos at ±5 frames. Lowering the contact
cutoff gave only a small net gain in timing-correct contact lists on the 40
development videos. The acceptance model tested on development data could not
reliably identify a high-precision subset. Rally-wide side assignment became
part of the system; those alternatives did not.
[Follow-up experiments](../../scratch/contact_det_followup/report.md)

## Early September: compare complete contact sequences

Missing serves, extra hits and missed later contacts called for decisions about
a rally as a whole. A locally plausible hit can still make the full sequence
worse. The next models therefore compared a small set of possible finished
sequences, including leaving the existing sequence unchanged.

The retained stages developed in three steps:

| Change | Fully correct rallies before → after | Why it mattered |
|---|---:|---|
| Score combinations of serve repair and extra-hit deletion | 995 → 1,435 | Edits could be judged in the context of the resulting rally |
| Allow one later missing hit, with a minimum score improvement of 0.05 | 1,435 → 1,597 | A bounded insertion repaired a common remaining error |
| Score the added hit independently and adjust clip boundaries | 1,597 → 1,763 | A plausible sequence also needed a supported added hit and usable clip bounds |

These results use the 47-video ShuttleSet22 comparison at ±10 frames, with
3,422 cleaned labelled rallies. The final boundary adjustment preserved which
contacts belonged to each rally while extending its clip. Applied on its own
to the 1,597-rally version, it recovered **135 rallies without losing any that
were already correct**. The earlier failed buffer search and this successful
boundary correction addressed different errors: searching for more hits around
a rough span versus enclosing an already selected contact sequence.

The repair limits were tested rather than assumed. Two later insertions added
too much ambiguity. A larger serve shortlist gained 19 correct rallies but lost
15, leaving little net benefit. Broad deletion remained damaging. A direct-answer
visual-language-model veto removed 11 wrong outputs but also discarded 39
correct ones. The retained system uses a small number of constrained repairs.

The [sequence experiment record](../../scratch/contact_det_closing_pass/experiment_lineage.md)
connects the alternatives and their saved outputs. The shorter
[final comparison](../../scratch/contact_det_closing_pass/followup_comparison.md)
separates the effects of insertion and boundary changes.

## September: rank review candidates separately

A strong contact score says little about whether another contact is missing.
Even a sequence model's preferred repair can be wrong. A separate confidence
model was developed to rank finished rallies, using evidence that includes
rejected candidates and gaps between selected hits.

The ranking experiments distinguished two kinds of success. A clip might contain
one whole rally while still giving it the wrong contact list. In the final
historical selection, 784 of 3,982 clips were retained; 740 could be judged from
the trusted labels. Of those 740, **728 (98.4%) contained one whole rally**, while
only **616 (83.2%) had an exact annotation**. The other 44 retained clips were
unjudgeable against those labels.

That difference matters for dataset construction. The review score can prioritise
promising material; it does not establish that the automatically generated labels
are safe to accept. The current model has no newly selected automatic-acceptance
cutoff. [Serve and confidence experiments](../../scratch/contact_det_closing_pass/serve_and_acceptance.md)

## September: improve the court evidence behind the models

Court geometry determines which people are treated as players and how their
positions relate to the shuttle. A bad court can therefore damage annotation
well beyond the corner coordinates themselves.

An early paired comparison held the fitted annotation models fixed while
repairing court handling and its surrounding processing. In problem video 53,
fully correct rallies rose from **7 to 35** at ±10 frames. Video 17 retained its
17 correct rallies, and neither video lost a previously correct rally. Separate
geometry controls checked repaired outlines against painted lines and reference
corners. The measured annotation benefit covers those two videos. [Court-repair comparison](reports/court_repair.md)

Further experiments explored fitting courts without relying on neural corner
predictions. They used detected lines, painted stripes, player movement, net
geometry and evidence across frames. Broadcast and amateur examples exposed a
recurring difficulty: a search could generate a good court while its selection
rule preferred a plausible wrong one. Background markings, neighbouring courts
and partly visible lines created competing explanations of the image.

Those prototypes made candidate quality and candidate selection separate
research questions. Their synthetic checks and visually promising examples
were insufficient evidence for replacing the whole production path.
[Court-fitting experiments](reports/court_fitting.md)

Frame-sampling work then compared three views around a scene midpoint,
independent fits, reuse of another frame's fit, and composites of visible
markings. This tested how much extra evidence could be recovered for the cost
of more detector work. The initial comparison left production sampling unchanged;
subsequent detector development adopted three-frame composites for fresh courts.
The [sampling report](reports/court_sampling.md) preserves the comparisons and
rerun details. The [court-detector design](../../docs/court_detector/design.md)
describes the resulting production system.

## October: preserve the implementation and refit on new courts

The refactored annotation code reproduced all 47 historical output streams
when supplied with the historical scores and models. Fresh fitting on old
court inputs reduced complete rallies from **1,763 to 1,734**. Component swaps
traced the difference to fitted-model behaviour in the evaluated path. Training
row order mattered even with the same examples and seed; library and regenerated
input changes also required separate checks.
[Refit investigation](reports/refit_regression.md)

The new-court comparison selected the base model over an optional rule that
rejects unreliable shuttle candidates without a nearby selected player. The
alternative gained three complete rallies on validation but lost four on
ShuttleSet22. Its review-queue advantage was inconsistent.

On the common 46-video comparison, the selected model recovered **1,744 complete
rallies**: ten more than the fresh old-court fit, but 19 fewer than the historical
model. It recovered more labelled contacts than either. This was a combined
change of courts, fitted models and training order, so the net result cannot
be attributed to court geometry alone. The [evaluation guide](evaluation.md)
puts the current model and historical baselines on the same footing.

## Where to continue

The experiments point to three useful areas for a future project:

- **Sequence errors and missing starts.** Contact search is already broad, while
  selection and bounded repair still leave usable evidence behind. Error analysis
  can distinguish a missing candidate from a candidate scored badly, or a repair
  that the current model cannot express.
- **Reliable review selection.** Ranking quality depends on queue size and the
  meaning of “correct”. A practical study could measure reviewer effort alongside
  exact annotation quality, including unjudgeable clips.
- **Transfer beyond the development footage.** The intended use includes new
  broadcasts and club recordings. Fresh labelled footage would test that goal
  more directly than further tuning against repeatedly inspected ShuttleSet22
  errors.

Court failures and camera cutaways remain relevant to all three. Better geometry
can restore useful input evidence; missing visual evidence may still prevent a
complete rally from being recovered. The existing experiments provide baselines
and rejected alternatives for a focused follow-up.
