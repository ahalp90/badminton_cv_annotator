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

Boundary padding gives the first and last contacts some surrounding video. It
stops at neighbouring rallies and video limits. An extension is accepted only
when it preserves the rally's contact membership.

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
