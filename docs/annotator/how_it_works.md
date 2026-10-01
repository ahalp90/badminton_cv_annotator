# How the auto-annotator works

Heuristics find plausible rally spans and contact regions. Trees then select
contacts and repair their sequences. A separate ranking model helps decide
which finished rallies to review. This is a fixed chain of operations, with
three sequence-refinement stages; it does not run an unlimited self-improvement
loop.

## Evidence and initial rallies

`run_video` consumes frame-aligned shuttle tracks, player poses and court
geometry. The track stores normalised shuttle coordinates and visibility.
Court geometry defines the valid view and helps associate pose detections with
the two court halves. Replay and invalid-view masks exclude unsuitable frames.
Shuttle quality codes also reject selected unreliable observations.

Motion/rest rules produce initial rally spans and raw contact candidates.
Persistent player assignment reduces pose identity changes between frames.
Wrist proximity, shuttle impulses, visibility changes, rally starts and scene
starts define regions worth scoring. Search can extend before a rally's first
eligible interval to recover an earlier serve.

These heuristics remain part of the mixed model. Their detailed settings are
fixed preprocessing policy, carried with the fitted bundle. They are not an
independent set of knobs to adjust around an already-trained tree.

## The base contact tree

[`contacts/features.py`](../../src/annotator/contacts/features.py) describes each
searched frame using shuttle motion, wrist proximity, player ankle motion and
missing-data indicators. The tree uses 85 columns: 17 signals at five temporal
offsets, centred on the candidate frame. The offsets correspond to -10, -5, 0,
5 and 10 frames at 30 FPS and scale to the source frame rate.

Feature windows stop at search-interval boundaries. Unavailable observations
remain NaN, with visibility and validity signals telling the tree what is
missing. Motion features retain their raw per-frame units. Changing those units
or the feature order would change what the fitted tree sees.

The contact tree scores rows inside the candidate regions. By default, scores
must reach 0.9. Nearby candidates compete within their search interval: the
strongest survives within a distance of six frames at 30 FPS, scaled to the
source FPS. This fixed spacing also governs later candidate selection. Earlier
frames win equal-score ties. The surviving stream forms the
initial contacts within each heuristic rally.

A contact's score is the base tree's output. Later sequence repairs may retain
or introduce contacts with lower base scores when their sequence evidence is
better.

## Three sequence refinements

The pipeline builds one shared pool of alternatives. It can keep a sequence,
repair an earlier serve, delete a contact, or combine an allowed initial
edit with one later-contact insertion. The serve search is short and separated
from the existing first contact. Later candidates lie inside the sequence and
away from existing contacts.

The chooser stages run once, in order:

1. **Whole sequence:** choose among keeping the contacts and the permitted
   serve/deletion edits. An edit must strictly beat the keep option.
2. **Later contact:** consider the same edits with a possible missed later
   contact. Replace the previous choice only with a score advantage of at
   least 0.05.
3. **Scored insertion:** make the guarded choice again, adding a separate
   judgement of the inserted contact as evidence. The same 0.05 margin applies.

Later stages score alternatives from the shared pool; they do not accumulate
arbitrary contacts through repeated passes. Ties favour the simpler option.
The selected design allows at most one later-contact insertion per rally.

The serve shortlist contains up to two alternative earlier frames. Zero or one
alternative is valid: the first detected contact may already be the serve.
This count is separate from the number of hits in the rally; a serve that is not
returned can form a one-contact rally. The fresh retune includes these shorter
shortlists, which the old experiment preparation rejected.

## Rally bounds and player sides

After the final choices, rally bounds widen around the selected contacts with
ten-frame padding at 30 FPS, scaled to the source FPS. Bounds stay within the video
and neighbouring rallies. An extension is accepted only if it preserves exactly
which full-stream contacts belong to the rally. It cannot silently take a
contact from the gap or drop a chosen contact.

For each contact, raw side assignment picks the tracked player whose wrist is
nearest to the visible shuttle. It then compares that player's bounding-box
bottom with the net band. Feet above the band imply `Top`, below imply `Bot`,
and inside the band leave the side unknown. Missing usable evidence also leaves
it unknown. The bundle records one of two geometry choices:

- `video`: one net band for the whole video, preserving the earlier model's rule
- `scene`: the net band measured for the contact's scene, following camera/view
  changes through the video

The net band is the vertical image range around the net that separates near and
far player positions. This choice changes raw side evidence; it is not a change
of player identity or a separate rally detector.

For each finished rally, the pipeline picks the alternating sequence of court
halves that best agrees with those raw guesses. A tied vote preserves the raw
individual guesses, including unknown sides. Contacts outside all rallies also
keep their raw guesses. Downstream outcome estimates use the finished contacts
and sides.

## Landings and point winners

The next rally's server determines the winner when that player is known.
Otherwise, the annotator uses the observed landing and court lines. An out
landing behind or wide of the player attributed with the last hit leaves this
geometry-based winner undecided. It does not change the contact's player side.

If the final contact occurs before a normal court view returns, the landing
search can begin in the first accepted scene within the predicted rally. It
uses that scene's geometry and ends at its boundary or the rally end. A masked
returning view is not used. The contact retains its original time and player.
The partial flight cannot establish that the shuttle never crossed the net,
so it cannot produce a net-fault verdict.

If the shuttle data at the last predicted hit is marked unreliable, the landing
is left unavailable. Searching from the previous hit could measure the other
player's shot while retaining the last hitter's identity. Contacts and player
sides are kept, and the next-server winner rule can still apply.

## Review scores and settings

The review-ranking tree learns whether a whole predicted rally was correct.
It uses the chosen output, discarded contacts and evidence of missing contacts
in the gaps. Its score is returned as `rally_confidence`. Higher scores
rank rallies as more likely to be correct. This score neither edits the result
nor measures contact precision, and it is not a calibrated guarantee on a new
dataset.

The main maintenance choices have small Python config objects:

- `ContactModelConfig`: the contact score cut-off
- `ContactFitConfig`, `SequenceFitConfig`, `ConfidenceFitConfig`: tree fitting
  settings and seeds for a retune
- `TrainingSettings`: the workflow's fitting and preprocessing settings

`BaseAnnotatorConfig`, feature-region rules and sequence edit policy define the
inputs and alternatives the models were trained around. Keep them stable for a
straightforward retune. If a real failure requires changing them, refit and
check the complete model on held-out groups.
