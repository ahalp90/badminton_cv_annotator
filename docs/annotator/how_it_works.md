# Auto-annotator overview

The auto-annotator narrows a noisy video timeline into a small set of rallies and contact frames.

Rules first remove unusable frames and mark places where a hit is plausible. A contact tree scores those frames. Sequence trees then compare a small set of possible corrections, such as repairing an early serve or adding one missed later hit. Once the contact sequence is fixed, rule-based code assigns court halves, adjusts rally bounds and estimates outcomes.

This is a fixed chain that runs once per video. The three sequence stages each run once, in order; nothing loops until it converges.

![Auto-annotator data flow](figures/architecture.svg)

## What goes in

One video contributes four kinds of saved data:

- **Video metadata** — frame count, frame rate and coded size.
- **Shuttle track** — one `(x, y, visibility)` row per frame, plus inpaint information and shuttle-quality grades.
- **Pose detections** — player boxes and keypoints for each frame.
- **Court data** — accepted camera scenes, homographies, a court-present mask and the geometry needed to separate the near and far court halves.

Both the standalone command and the dataset builder load these files through the same dataset-builder functions. The annotation code therefore receives the same in-memory inputs in both cases.

The court detector supplies scene boundaries and court geometry. Its scene histograms are not inputs to the contact or sequence models.

## Stage 1: build the search area

`run_video()` first converts frame-count settings from their 30 FPS reference values to the video's actual frame rate.

It then builds court/player data, exclusion masks, rough rallies and the regions that the contact model will score.

### Court scenes and tracked players

Court data is divided into accepted camera scenes. Within each scene, the sticky player picker tries to keep one usable pose detection on each physical court half: far (`Top`) and near (`Bot`).

This produces frame-aligned information such as:

- the pose detection assigned to each court half;
- wrist-to-shuttle distance;
- ankle position and movement;
- whether enough player data is present on the frame.

Keeping the assignment across frames reduces the player-slot jumping that would occur if every frame were matched independently.

### Frames and shuttle positions excluded from evidence

Two separate safeguards handle evidence that can look physically convincing while being unsuitable for contact or landing decisions.

The **frame exclusion mask** removes whole stretches of video. In the dataset-builder path it marks sustained court absence and slow-motion-like shuttle movement, then filters out very short mask runs. Full annotation also excludes frames outside accepted court geometry. This keeps replay, slow motion and non-court footage from producing false rally or contact evidence.

The **shuttle guard** applies to the track rather than the whole frame. It looks for exact shuttle-coordinate patterns that recur many times, which can happen when tracking or inpainting falls into a fabricated loop. Grades `1`, `2` and `3` mean fabricated, suspicious-flat and degraded shuttle positions. The default settings treat all three as unreliable.

The grading matters in three places:

- unreliable frames are left out of the normal-speed estimate used by the slow-motion detector;
- the outcome rules do not use unreliable frames for a landing, and they leave the landing empty when the last hit of a rally sits on one;
- an optional model setting drops candidate contact frames that are unreliable and have no picked player nearby in time.

That optional setting is off by default. With it off, the contact tree still scores unreliable shuttle frames and can select one as a contact.

The two mechanisms are described in detail in [Fixed heuristics](heuristics.md).

### Rough rallies and contact search regions

Shuttle motion gives the initial rally spans. The current path finds stretches separated by long rest, requires a sustained fast burst to confirm real play, and back-fills the start to the beginning of that active region. Special handling keeps some difficult tracking gaps from immediately ending a high shot.

One important heuristic inside those spans is **shuttle impulse**: the size of the change between the incoming and outgoing shuttle velocity. A racket hit often creates a sudden change in speed, direction, or both. The raw rule compares each impulse with a rolling local impulse floor rather than with one fixed global value.

Within and around the rough spans, several signals mark frames as worth scoring:

- a strong impulse candidate that passed the heuristic contact chain;
- a weaker impulse at least 1.25 times the local floor;
- a local wrist-distance minimum;
- a visibility change;
- a rally start;
- a court-scene start;
- the serve look-back window.

These rules only decide **where to evaluate contact probability**. The fitted contact tree decides which of those frames survive.

The rules are still part of the fitted system. Their settings are saved in the model directory and were the inputs the trees trained on. They are not a separate set of knobs to adjust around an already-fitted tree.

## Stage 2: score possible contact frames

`contacts/features.py` builds one feature row for every searched frame.

The contact tree sees 85 values: 17 signals sampled at five time offsets. At 30 FPS the offsets are `-10, -5, 0, +5, +10` frames; the offsets scale with frame rate.

The 17 signals cover:

- shuttle x/y velocity, speed, impulse and impulse ratio;
- nearest, Top and Bot wrist gaps;
- x/y offset from the shuttle to the nearest wrist;
- Top and Bot ankle speed;
- shuttle visibility;
- pose validity for each court half;
- wrist validity for each court half.

Missing measurements remain `NaN`. Separate validity fields tell the model whether the player or wrist measurement was available. Feature windows stop at search-interval boundaries, so values are not borrowed from an unrelated region of the video.

Motion features keep their raw per-frame units. Changing those units, or the feature order, would change what the fitted tree sees.

The classifier is a `HistGradientBoostingClassifier`. It combines those 85 values into one model score for the positive contact class. Training positives are searched rows close to human-labelled contacts; nearby ambiguous rows are ignored and negative rows are sampled around them.

The model output is read through `predict_proba()`. The current initial-selection cutoff is `0.9`. That value is a classifier score threshold, not a separate physical confidence formula and not a guarantee of 90% real-world certainty.

Current initial selection rules are:

- contact score must reach `0.9`;
- contacts within six frames at 30 FPS compete with each other, inside one search interval;
- the higher score wins;
- equal scores keep the earlier frame.

The surviving frames form the initial contact sequence. Lower-scoring candidate rows are still available to the later sequence models, so sequence repair can recover one when the wider rally pattern supports it. [Tree model stack](tree_stack.md) explains that interaction and the evidence used by each later tree.

When the optional rule for guarded candidates is on, the dropped rows are removed before scoring. They then take no part in the initial selection or in later sequence repair.

## Stage 3: compare possible sequence repairs

A plausible contact stream can still contain one wrong or missing hit. The sequence stage handles a small, fixed set of corrections rather than searching arbitrary combinations.

For each rough rally, it creates alternatives that can:

- keep the current contacts;
- add or replace an earlier serve;
- delete one contact;
- combine one permitted early edit with one later-contact insertion.

The selected design allows at most one later inserted contact.

The serve shortlist holds up to two earlier frames. Zero or one is also valid, because the first detected contact may already be the serve. This count is separate from the number of hits in the rally: a serve that is not returned forms a one-contact rally. Later candidates lie inside the rally and away from the existing contacts.

Three chooser models examine the same set of alternatives:

1. **Whole sequence** — compares the original sequence with serve repairs and deletions. A change must beat the unchanged sequence.
2. **Later contact** — can switch to an alternative containing one added later hit, but only with a score margin of at least `0.05` over the previous choice.
3. **Scored insertion** — makes the same comparison while also considering a separate score for the proposed inserted contact. The same `0.05` margin applies.

Each stage starts from the previous stage's choice. The option set itself does not grow between stages. Ties go to the simpler option.

## Stage 4: set rally bounds and court halves

After contact selection, rally bounds can expand around the chosen contacts. The default padding is ten frames at 30 FPS, scaled for the video. Bounds stay inside the video and stop at the neighbouring rallies.

A boundary change is accepted only when it leaves contact membership unchanged. It cannot pull a gap contact into the rally or remove a selected contact.

Each contact also gets an initial court-half guess. The annotator finds the tracked player whose wrist is nearest the visible shuttle, then compares the bottom of that player's bounding box with the net band. The net band is the vertical image range around the net that separates far and near player positions. Feet above the band give `Top`, feet below give `Bot`, and feet inside the band leave the side unknown. Missing evidence also leaves it unknown.

The model directory stores one of two geometry modes:

- `video` — one net band is used for the whole video;
- `scene` — the net band comes from the contact's camera scene.

The final side pass chooses the alternating `Top`/`Bot` pattern that best matches those initial guesses. A tie leaves the individual guesses unchanged, including unknown sides. Contacts outside every rally also keep their initial guesses.

`Top` and `Bot` refer to physical court halves in the image. `Top` is the far half; `Bot` is the near half. They do not identify a particular player across a change of ends.

## Stage 5: estimate rally outcomes

Outcome code runs after final contacts and sides are known.

A known server in the next rally can identify the previous rally winner because the winner serves next. When that evidence is unavailable, landing geometry can sometimes provide an answer instead. A landing that is out behind or wide of the player credited with the last hit leaves the geometry-based winner undecided. It does not change that contact's court half.

The outcome code also estimates shuttle landing and hit height when the required shuttle and court data is available.

Two cases limit the landing search:

- **The last hit happens before a normal court view returns.** The search can start in the first accepted scene inside the predicted rally. It uses that scene's geometry and ends at the scene boundary or the rally end. A returning view that is masked is not used. The contact keeps its original time and court half. Because part of the flight was not seen, this partial flight cannot produce a net-fault verdict.
- **The shuttle data at the last hit is unreliable.** The landing is left empty. Searching from the previous hit could measure the other player's shot while keeping the last hitter's identity. Contacts and court halves are kept, and the next-server winner rule can still apply.

Outcome fields can remain empty when the evidence is insufficient. Missing data stays explicit rather than being replaced with a guess.

## Stage 6: score rallies for review

The confidence tree looks at the completed sequence and the nearby evidence for discarded or possibly missing contacts. It returns one score per rally.

A higher score means the rally looks more like the correctly assembled training examples. The score is used for review ordering. It does not edit the rally, it does not measure contact precision, and it is not automatically a calibrated probability on a new dataset.

## Rules versus fitted models

| Part | Main job | Learned? |
| --- | --- | --- |
| FPS scaling, masks, court scenes, tracked players | Prepare frame-aligned evidence and reject bad frames | No |
| Initial rally spans and search regions | Decide where contact scoring is worth running | No |
| Contact tree | Score possible contact frames | Yes |
| Serve / sequence / insertion choosers | Select one contact sequence from a limited set of alternatives | Yes |
| Boundary widening | Expand rally bounds without changing contact membership | No |
| Side alternation | Reconcile individual court-half guesses across a rally | No |
| Server / winner / landing / hit height | Derive outcome fields from final contacts and geometry | No |
| Rally confidence | Rank completed rallies for review | Yes |

A rule change can still require a new fit. For example, changing the contact search regions changes the frames and features that the contact and sequence models receive, even if no tree code changes. The [maintainer guide](maintaining.md) lists the common cases.

## Model directory contents

A model directory contains:

```text
models/annotator/
├── models.joblib
└── metadata.json
```

`models.joblib` is written with `joblib.dump()` and stores one serialised `AnnotatorModels` object. It contains:

- the contact classifier;
- the six models in `SequenceModels`;
- the rally confidence classifier;
- the contact prediction settings: the score cutoff and the optional rule for guarded candidates;
- rough-rally, mask and other preprocessing settings;
- the side-geometry mode.

`metadata.json` records the model schema, exact scikit-learn version, contact feature order and side-geometry mode. Loading stops if these do not match the current runtime or the serialised model object.

Changing contact feature order or scikit-learn version therefore requires a new fit. An old tree cannot be safely reused with a different feature order or scikit-learn runtime.

The main settings live in small Python config objects:

- `ContactModelConfig` — contact score cutoff and the optional rule for guarded candidates;
- `ContactFitConfig`, `SequenceFitConfig`, `ConfidenceFitConfig` — tree fitting settings and seeds;
- `TrainingSettings` — everything the fitting workflow uses, including the preprocessing settings;
- `BaseAnnotatorConfig` — rough-rally, mask and shuttle-guard settings.

## Result fields at a glance

The most commonly useful fields are:

- `spans` — final half-open rally bounds;
- `filtered_by_rally` — final contact frames grouped by rally;
- `contact_events` — the full final contact stream, with score and court half;
- `rally_confidence` — one review-ordering score per rally;
- `fitted_first_all`, `next_servers` and `striker_halves` — court-half and server information;
- `landings`, `verdict_rows`, `geometric_verdict_rows` — outcome data;
- `hit_height_by_frame` and `hit_height_failures` — hit-height results and explicit failures.

The exact saved files and all result fields are listed in [Inputs and outputs](inputs_outputs.md).

## Terms used in the code

| Term | Meaning here |
| --- | --- |
| **candidate region** | A time interval where at least one rule says a contact is plausible enough to score |
| **raw contact** | A heuristic contact candidate before the fitted contact tree selects final contact frames |
| **contact event** | A selected contact frame with its model score and court-half assignment |
| **sticky player** | The pose detection held for the `Top` or `Bot` court half across nearby frames |
| **scene court** | Court geometry associated with one accepted camera scene |
| **exclusion mask** | Boolean frame mask for frames removed from normal rally/contact evidence |
| **shuttle hallucination mask** | Boolean frame mask for shuttle positions whose guard grade the model settings reject |
| **option pool** | The limited keep/repair/delete/insert alternatives compared by the sequence models |
| **model directory** | `models.joblib` plus `metadata.json`, containing fitted models and their required settings; the code also calls it the model bundle |

## Main call chain

```text
python -m annotator
    ↓
annotator.cli.annotate_saved_video
    ↓
dataset_builder.vision loaders
    ↓
dataset_builder.vision.run_full_annotation_stage
    ↓
annotator.run_video.run_video
    ├─ rally / masks / courts  → rough rallies and frame evidence
    ├─ annotator.hybrid        → contacts, sequence repair, review score
    └─ outcomes                → winner, landing, hit height
    ↓
dataset_builder.vision.persist_annotation_run
```

The [code map](code_map.md) gives the file-by-file view.
