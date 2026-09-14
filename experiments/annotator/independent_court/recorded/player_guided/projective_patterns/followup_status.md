# Progress through the court-detection follow-ups

The work follows the projective-pattern branch of the proposal-construction
plan. Candidate-retention and paint-ranking experiments were brought forward
when measured failures justified them. This is progress within the conditional
plan, rather than completion of five sequential implementation stages.

| Planned stage | Status after these experiments | What remains |
|---|---|---|
| 1. Proposal construction | Direction pruning, synthetic spacing tests, supplied-direction matching and automatic-direction matching are complete as experiments. Useful geometry is recoverable, but the automatic selector loses precise directions. | Redesign direction retention or refinement. General graph search remains optional and untested. |
| 2. Candidate retention and evidence | Camera-before-cap and removal of the global cap were compared on all nine views. Original gate outcomes were preserved. Global retention alone does not explain GX5. | Keep direction loss separate from per-pair loss. A partial-view evidence rule has not been validated. |
| 3. Ranking | Existing paint profiles were compared with line support on identical pools. They repair Amateur-2 ranking in the supplied-direction test but worsen GX0; larger automatic pools expose false winners. | Test ranking against those wrong-court/background examples. A general appearance rule is unresolved. |
| 4. Multiple frames | No new controlled temporal-generation or score-aggregation comparison. The existing ShuttleSet composites are input data, not this experiment. | Activate only after stable-view and complementary-frame evidence is established. |
| 5. Acceptance and replacement | Original gates remain in place. No replacement acceptance rule or production integration. | First obtain reliable automatic generation and selection, then measure coverage and wrong emissions on positive and negative cases. |

## How much closer is the detector?

The main advance is locating the bottleneck. The matcher produces usable courts
in all eight supplied-direction views that received visual feedback. The ninth
view's new candidates remain unjudged. Those results test matching capacity;
they do not establish automatic direction recovery.

The automatic gallery has now been inspected on all nine views. Six have a usable
winner under at least one ranking, chosen after inspection. GX0, GX5 and
Amateur-2 frame28019 have poor displayed winners and reference-selected diagnostics.
The [panel judgements](automatic_axes_visual_judgements.md) show why low corner
error and either ranking alone do not establish usability. This feedback does
not judge the earlier supplied-direction scene16 candidates.

Three difficult automatic cases now have a more precise diagnosis. Their full
image-derived direction banks contain useful directions that selection discards.
Feeding those observed directions through the unchanged matcher produces close
courts. GX5's selected line/paint court is 10.58 pixels from the manual reference;
Amateur-2 frame28019's paint winner is 7.27 pixels away. These are maximum corner
errors at 1280 × 720, from label-guided controls, not automatic success rates.

A subsequent GX0 control supplies discarded observed directions to the same
matcher. The user judged its line winner essentially perfect and its paint winner
perfect. Both are essentially ideal. This resolves the GX0 control visual check
and strengthens the direction-precision diagnosis. The directions were selected
using the approved court, so this is not an automatic improvement.

The next change can therefore target direction precision rather than replacing
the spacing matcher or repeating broad score sweeps. Ranking and acceptance
remain separate unresolved tasks. There is no defensible percentage-complete
estimate or established end-to-end improvement rate.

No new annotations are needed to expose the current failures. Two additional
hard holdout videos remain a final-readiness option, after development succeeds.
See the [automatic results](automatic_axes_results.md) for the decisive evidence.
