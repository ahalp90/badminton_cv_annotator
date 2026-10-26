# How the auto-annotator works

The annotator turns saved shuttle tracks, player poses and court geometry into
rallies, racket contacts and estimated outcomes. Start with
[`run_video()`](../../src/annotator/run_video.py): it shows the order of the
work. The [quickstart](quickstart.md) explains how to supply a video's files.

The main path has three parts:

1. Rules find playable stretches, rough rallies and possible contacts.
2. XGBoost scores contact frames, then chooses a contact sequence for each rally.
3. Rules assign court halves and estimate the server, winner, landing and hit height.

A separate model gives each rally a review score. It orders manual inspection;
it does not remove contacts or approve rallies. `heuristic_only=True` skips the
fitted models and uses the rule-based contacts for the outcome stage.

## Start with one shared video timeline

Every input uses the original video's frame numbers. The shuttle track has one
normalised `(x, y, visibility)` row per frame. Pose arrays contain detected player
boxes and keypoints. Court records identify usable scenes and provide the
image-to-court transform, called a homography.

Rally bounds are half-open: `[start, end)` includes `start` and excludes `end`.
The frame rate scales timing and movement thresholds. A frame is never renumbered
when a replay or tracking gap is excluded.

Within each usable court scene, the player picker follows one person on each
court half. It favours the same person across nearby frames; this is the
"sticky" player evidence in the code. Wrist-to-shuttle distances and player
motion are measured from the original track before masking.

## Find rough rallies and contacts

The exclusion mask marks footage such as replays and prolonged court absence.
When bounds are not supplied, a first pass finds rallies on the unmasked track
to estimate normal play speed for replay detection. Those preliminary bounds
are discarded; rally detection runs again after masking.

For rally detection, excluded coordinates are held at rest so they cannot create
movement. Full annotation also excludes frames without accepted scene geometry.

Shuttle-quality grades serve a different purpose. They mark unreliable shuttle
positions without necessarily excluding the entire frame. Those positions are
omitted from the normal-speed estimate and landing evidence. An optional contact
setting can also reject unreliable candidates without nearby player evidence;
that setting is off in the selected model.

Long rests divide shuttle activity into rough rallies. A sustained fast burst
confirms play. The selected opening rule includes the activity leading into the
burst. Tracking-gap rules prevent some high shots from being mistaken for rests.
Callers may supply rally bounds instead, as the supplied-boundary MonoTrack
benchmark does.

The contact heuristic looks for abrupt changes in shuttle velocity. It compares
each change with nearby motion, then checks wrist proximity and suppresses weaker
nearby candidates. In heuristic-only mode, these contacts feed player attribution
and outcome estimation directly. The learned path uses them as one source of
contact-search evidence.

See [Fixed heuristics](heuristics.md) for the individual rules and research controls.

## Score possible contact frames

The contact search includes impulse peaks, wrist-distance minima, visibility
changes, rough rally starts, scene starts and a serve look-back window.
[`contacts/features.py`](../../src/annotator/contacts/features.py) measures
85 values per searched frame: 17 shuttle/player signals at five time offsets.

[`contacts/model.py`](../../src/annotator/contacts/model.py) scores candidates
with the contact classifier. Candidates compete by probability and temporal
separation. The selected cutoff is 0.9. Lower-scoring candidates remain available
for the next stage, which can recognise a useful contact from the rally as a whole.

This produces both an initial contact stream and the scored candidates that could
repair it. Initial contacts outside every rough rally remain part of that stream.

### How contact training uses ground truth

The contact model learns from frames in these broad heuristic-defined search
regions, labelled against the human-annotated contacts. It is not restricted to
the heuristic's final hit predictions.

For example, at 30 fps, a heuristic seed at frame 95 creates a search region
extending about 15 frames either side. If ground truth says the contact was at
frame 100, frames **99–101 can become positive examples**, even though the
heuristic seed was five frames off.

Within those regions:

- Frames within **±1 frame** of a labelled contact are positive.
- Frames **2–4 frames away** are ignored.
- Frames **5+ frames away** are negative candidates. Nearby negatives are kept,
  while farther ones are sampled.

These distances scale with the video's frame rate. Class balancing gives the
positive and negative examples equal total weight. The narrow positive window
teaches the model to locate the contact precisely; the benchmark's ±10-frame
evaluation tolerance is a separate allowance.

Ground truth does not force a frame into the candidate set. A labelled contact
with no eligible candidate within ±1 frame contributes no positive example.
In the selected model's 32 training videos, **26,893 of 27,571 labelled contacts
(97.5%)** have at least one positive candidate frame, providing **80,651 positive
frame examples**. The remaining 678 contacts have no feature/search rows within
that radius. None were lost solely to the candidate-region filter once feature
rows were available.

The [saved coverage counts and measurement command](../../experiments/annotator/tree_simplification/results/boundary_retrain_v7/README.md#contact-training-coverage)
give the per-video breakdown. The training implementation is in
[`training/contact.py`](../../src/annotator/training/contact.py).

## Choose a contact sequence for each rally

The rally model compares alternative sequences, each containing a concrete set
of contact frames. The proposal builder shortlists up to two earlier serve frames
and six later-contact frames. It retains the original sequence and proposes serve
repairs, deletions and additions. The selected model allows up to three combined
changes and considers up to six contacts for deletion.

Some proposals come from the earlier serve/deletion rules. Those rules remain
because they contribute choices the current model was trained to score. The
retired sequence classifiers are no longer involved.

Each proposal describes the original and proposed contacts, their timing and
probabilities, raw court-half guesses, physical contact evidence and unused
candidates. The selector scores these rows and chooses the highest-scoring
proposal for each initial rally. Stable proposal order resolves equal scores.

The selected model has one rally selector. Experiments can fit more layers; each
extra layer receives the earlier layers' scores and choices as additional inputs.
Training and inference share the same proposal and feature builders.

An earlier serve can move a rally's start and absorb intervening contacts.
Contacts originally outside all rallies survive in the full stream even if an
individual proposal omits them. This is why contact membership is stored explicitly
instead of reconstructed from the final bounds.

## Finish boundaries and court halves

Boundaries and contacts are separate, interacting parts of the prediction.
The initial window helps determine which contacts belong to a rally. Candidate
revisions can change both the contacts and the window; the selector chooses
one revision for each rally.

The chosen window keeps its context: there is no maximum number of seconds
before the first contact or after the last. The final step can extend it to
provide 10 frames of context at 30 FPS, where neighbouring rallies, video edges
and contact membership allow. Supplied boundaries stay fixed.

The window still matters. Inside it, the selector's chosen contacts are the
rally's accepted contacts; rejected candidates stay rejected. Detector contacts
outside every chosen window remain in the full contact stream.

Training also allows unlimited outward context. It judges whether a proposal's
window covers one complete labelled rally, then checks its contacts and player
sides. A window that merges neighbouring rallies or cuts into the rally beyond
the timing allowance is still wrong.

When we assess discovery, we use the first and last accepted contacts as the
rally's endpoints. We allow up to 10 frames inward at 30 FPS and unlimited
outward extension, while still rejecting merged rallies. The boundary-only
confidence assessment adds an outward limit of 2 seconds before the first
labelled contact and 3 seconds after the last. It keeps the same inward
allowance and ignores errors in the interior contacts and player sides.

Dataset export is a separate step. It uses the first through last accepted
contact as the core rally interval, then adds exactly 2 seconds before and
3 seconds after, except at video edges. These clips may overlap. A detected
rally with no accepted contacts stops export so it can be reviewed. Export
does not change the discovery predictions used for evaluation.

Each contact initially has an individual court-half guess: `Top` for the far
half, `Bot` for the near half, or unknown. The finishing rule chooses the
alternating pattern that best matches those guesses. A tied vote keeps the
individual guesses. These are court halves, not player identities.

The review model uses evidence from the chosen proposal, selector scores and
final boundaries. It retains the original side guesses when measuring agreement;
using the enforced alternating sides would make that evidence misleading.

## Estimate outcomes and return the result

[`outcomes/video.py`](../../src/annotator/outcomes/video.py) derives server,
winner, landing and hit-height fields from the final contacts and geometry.
Insufficient evidence can leave an outcome unresolved while retaining the contacts.

`AnnotatorResult` holds the public result. The dataset builder saves all of its
fields. In particular:

- `filtered_by_rally` is the contact membership of each rally.
- `contact_events` is the full learned contact stream, including contacts outside rallies.
- `rally_confidence` holds review scores in rally order.
- Winner, landing and hit-height fields carry outcomes and explicit failures.

`RunCapture` optionally retains intermediate predictions, both exclusion masks
and requested landing diagnostics. It supports debugging, preparation and
experiments without changing the result contract.

The standalone command, dataset builder and MonoTrack adapter all call
`run_video()`. Their loading and persistence differ; the annotation decisions
come from the same function. See [Inputs and outputs](inputs_outputs.md) for
the saved contracts, [Code map](code_map.md) for source entry points, and
[How the trees work together](tree_stack.md) for model and training details.
