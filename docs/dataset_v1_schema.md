# Rally dataset v1 schema

| Item | Value |
| --- | --- |
| Schema | `rally-dataset/1.3` |
| Frozen on | 2026-09-04 |
| Status | Frozen |
| Owner | Issues [#18](https://github.com/ahalp90/badminton_cv_annotator/issues/18), [#138](https://github.com/ahalp90/badminton_cv_annotator/issues/138) |
| Builds on | [`rally_dataset_contract.md`](rally_dataset_contract.md) version 0.2 |

`src/dataset_builder/schema_v1.py` holds the frozen surface and is the single
source of truth. This document explains what it says and why.

## What the freeze means

Five things are frozen:

- the key of each table, meaning the columns that make a row unique;
- which columns each table has;
- the order those columns appear in;
- each column's type, and whether it can be null;
- each column's reliability class.

You can write code against those five and it will keep working. That is the
whole point of a freeze.

Two things are not frozen. The first is how much data an export holds, because
more videos can be processed at any time. The second is the raw primitive
layer, which is versioned by the pipeline run that produced it.

Changing anything on the frozen surface is a breaking change. It bumps the
schema string away from `rally-dataset/1.3` and it updates the freeze test in
`tests/test_dataset_builder_schema_v1.py`. That rule lets a reader tell a
changed schema from a bigger export.

`rally-dataset/1.0` froze on 2 September 2026. `rally-dataset/1.1` only adds
columns: `rallies.shots_per_rally`, the four position-derived `source_contacts`
columns, and the two `player_rallies` medians they roll up into. Issue #104 cut
those three features on 2 September because the annotator's predicted contacts
were too unreliable; the dataset now builds on human ShuttleSet contacts
instead, which removes that reason (see Kept features below). Nothing already
frozen in 1.0 changed name, type, nullability, or reliability class, so 1.0
code that ignores unknown columns still reads a 1.1 export correctly.
`rally-dataset/1.1` also adds `rallies.flaw_marked` (see Source contacts
below).

`rally-dataset/1.2` adds one table, `player_trends`, that fits a per-player
degradation trend over a rally-level or set-level feature (see Player
degradation trends below). Nothing already frozen in 1.1 changed, so 1.1
code that ignores unknown tables still reads a 1.2 export correctly.

`rally-dataset/1.3` adds one table, `commentary_rally_links`, that pairs a
commentary chunk to the rally it most likely discusses (see Commentary
tables below). It also adds `whisperx_aligned` as a
`timestamp_precision` value, for chunk timing replaced by forced-aligned word
boundaries under issue #136. Nothing already frozen in 1.2 changed, so 1.2
code that ignores unknown tables and values still reads a 1.3 export
correctly. `rally-dataset/1.1`, `1.2`, and `1.3` merged onto main in that
order.

## How to read reliability

Every column carries a reliability class. The class says where the value came
from. Origin is the only honest way to state how much a number can be trusted,
so it sits on the column itself instead of in a footnote.

| Class | Meaning |
| --- | --- |
| `observed` | Exact source identity, timing, or file metadata after validation. |
| `source_annotation` | Human ShuttleSet or ShuttleSet22 label carried verbatim. Not an annotator prediction. |
| `predicted` | Production annotator heuristic output. Measured weak on ShuttleSet: 66% of human rallies covered, strict contact F1 49%. |
| `derived` | Verified computation over frame-aligned primitives. Not validated against independent ground truth. |
| `curated` | Hand-maintained metadata table in the repository, checked against the ShuttleSet match tables at export. Not a prediction and not inferred from video. |
| `by_rally_origin` | `predicted` when `rally_origin` is `annotator`; `source_annotation` when `rally_origin` is `source_contacts`. |

Two measurements from the issue #104 benchmark set those classes. On
ShuttleSet, the production annotator covered 66.24% of the human-labelled
rallies. Covered means every human contact of that rally landed inside one
predicted span. Its strict contact F1 was 48.56%. F1 is a single score that
combines how many predicted contacts were real with how many real contacts
were found, and strict means the predicted frame had to be close to the human
one.

Both numbers are low. About a third of rallies are missed or split, and about
half of the contact decisions are wrong. So any value resting on a predicted
rally or a predicted contact is weak, and the schema labels it that way.

`by_rally_origin` exists because `rallies` holds two kinds of row. An
`annotator` row is a predicted span. A `source_contacts` row runs from the
first usable human contact of a ShuttleSet rally to one frame past the last.
The same column is therefore a guess on one row and a human label on another.
Filter on `rally_origin` before you trust a rally boundary.

A `source_contacts` rally can also carry `rallies.flaw_marked` (see Source
contacts below). On some of those rallies the contact frame number itself is
wrong, which corrupts every value built from it: `rallies.start_frame`,
`end_frame`, `duration_frames`, `duration_seconds`, `start_seconds`,
`end_seconds`, `clip_start_frame`, `clip_end_frame`; `player_rallies.posture_mad`,
`posture_frames_valid`, `posture_frames_linear`, `position_frames_valid`,
`position_frames_linear`, `recovery_distance_median`,
`movement_inefficiency_median`; and, on the flagged contact and its
neighbours, `source_contacts.recovery_distance`, `recovery_frames_valid`,
`movement_inefficiency_top`, `movement_inefficiency_bottom`. Filter on
`rallies.flaw_marked` before trusting any of these for frame-anchored work.

## Row ownership

A `player_rallies` row belongs to one court side within one rally.
`court_side` is `top` or `bottom`. `player_id` says which person was on that
side. It is a foreign key to the `players` table, which carries the person's
name and sex. The rally row repeats both ids as `top_player_id` and
`bottom_player_id`.

The sticky-player picker chooses one detection per side per frame and holds
that choice across the rally. It cannot recognise a person, so `player_id`
never comes from the video. It comes from the ShuttleSet match table: the
`downcourt` flag says who starts on the top court, sides swap for set 2, and
set 3 changes ends when a score first reaches 11. The Players section below
gives the rule, its cross-check, and the cases that leave `player_id` null.

Features that need identity across rallies, such as a player's degradation
across a match, group on `player_id`. `player_trends` is the one that does
this; see its formula below.

## Files

An export writes eleven things.

| Path | Holds |
| --- | --- |
| `rallies.csv.gz` | One row per rally. |
| `player_rallies.csv.gz` | One row per rally and court side. |
| `player_trends.csv.gz` | One row per player, scope, and trended feature. |
| `players.csv.gz` | One row per person the export references: name and sex. |
| `source_contacts.csv.gz` | Human ShuttleSet contact rows. |
| `primitive_artifacts.csv.gz` | A list of the raw frame-aligned files. |
| `transcript_segments.csv.gz` | Normalised commentary transcript segments. |
| `commentary_chunks.csv.gz` | Triaged commentary chunks, raw and cleaned. |
| `commentary_rally_links.csv.gz` | Chunk-to-rally links under the issue #138 pairing rule. |
| `player_signals/<video_id>/` | Four frame-aligned arrays per video, stored as `.npy.xz`. |
| `dataset_manifest.json.gz` | Provenance for the whole export. |

Two commands write this layout. `export-v1` reads a completed dataset-builder
run, so its `rallies` table holds `annotator` rows and, when the ShuttleSet
annotations are given, `source_contacts` rows too. `export-v1-shuttleset22`
reads the ShuttleSet22 primitives, which have no production run, so every
rally there is a `source_contacts` row. Both produce the same nine tables.

Three rules matter when reading the tables:

- Every table is a gzip-compressed CSV. An empty field means null and nothing
  else.
- A string column never holds an empty string. An empty field in a string
  column is therefore always a real null, never a blank value.
- `video_id` must be loaded as a string. `0012` and `12` are different videos,
  and reading them as numbers silently merges them. `read_table` in
  `schema_v1.py` applies the frozen types and does this for you.

## Running an export

Both commands take explicit paths and discover nothing. The ShuttleSet export
reads a completed dataset-builder run. Its `--run-dir` is the directory that
holds `rally_records.json.gz` and `run_manifest.json.gz`. On Carmack that is
the issue #103 run under `/scratch/cmarti/issue103_ad8da4f/`, run ID
`a5d37677def443469f6b83d8ee838e7b`.

```bash
PYTHONPATH=src uv run python -m dataset_builder export-v1 \
  --run-dir /scratch/cmarti/issue103_ad8da4f/<run directory> \
  --output-dir /scratch/<user>/dataset-v1/shuttleset \
  --fixed-sources configs/dataset_builder/shuttleset_sources_v1.toml \
  --ground-truth-root training/data/shuttleset/annotations \
  --commentary-root /scratch/<user>/<commentary preparation root> \
  --replay-mask-root /scratch/<user>/<replay mask root>
```

The ShuttleSet22 export reads the issue #106 and #120 primitives. Its
`--data-root` holds `extracted-simple/`, `annotations/`, and `sources/`. On
Carmack that is `/scratch/cmarti56/issue106-shuttleset22-data`.

```bash
PYTHONPATH=src uv run python -m dataset_builder export-v1-shuttleset22 \
  --data-root /scratch/cmarti56/issue106-shuttleset22-data \
  --output-dir /scratch/<user>/dataset-v1/shuttleset22 \
  --run-id issue106-ba24a95 \
  --commentary-root /scratch/<user>/<commentary preparation root> \
  --replay-mask-root /scratch/<user>/<replay mask root> \
  --inpainted-root /scratch/ahalperi/contact_det_full_ds_fit/shuttleset22-inpainted-extract
```

`--commentary-root` is optional. It expects the layout the issue #104
commentary benchmark used: `transcripts/<video_id>.json`,
`commentary/cleaned_chunks/<video_id>.json`,
`status/commentary_per_video_status.json`, and, when the issue #136 re-timed
sidecars exist, `commentary/retimed_chunks/<video_id>.json`. Without
`--commentary-root` all three commentary tables are written empty.

`--replay-mask-root` is also optional and only matters when
`--commentary-root` is given. It expects one
`<video_id>/definitive_exclusion_mask.npy.xz` per video. Without it, every
`commentary_rally_links.starts_on_masked_frame` value is null instead of a
real mask check; see [Commentary tables](#commentary-tables) below for why
that matters.

`--inpainted-root` is optional too, but matters: the ShuttleSet22 extract
under `--data-root` was run with InpaintNet off, so its shuttle track has a
visible shuttle on only 35-51% of frames per video, well under ShuttleSet's
77-93%. A later InpaintNet pass over the same videos fixed that, reaching
78-87% visible frames, and its output lives in a second root instead of
`extracted-simple/`. Passing `--inpainted-root` swaps that corrected track
and its guard codes into `primitive_artifacts` for every exported video.

The sticky player picker never reads a shuttle coordinate. It reads pose
boxes, scores, keypoints, detection counts, and court gate inputs; the
shuttle track only sets the length of its output arrays. So the flag changes
no kept feature value: measured on ShuttleSet22 matches 8 and 9, every
`player_rallies` column came out byte-identical between the plain and
corrected tracks, while track visibility moved from 35.2% to 78.2% on match
8 and 51.0% to 86.9% on match 9.

What the flag does change is provenance. The primitive bundle now names the
corrected track and carries its guard codes, so ShuttleSet22 gets guard
codes at all for the first time; without the flag it ships none. Serve
speed, the one unresolved feature that will read shuttle coordinates
directly, needs this wiring in place before it can land. Without the flag,
ShuttleSet22 and ShuttleSet sit on shuttle inputs of different quality, and
ShuttleSet22 ships no guard codes.

Any ShuttleSet22 export meant for delivery should pass `--inpainted-root`.
Omitting it is a provenance gap, not a numbers gap, but it is still one a
client copy should not carry silently: the exporter logs a warning naming
the flag when it is left off, and `dataset_manifest.json.gz` records
`inpainted_root: null` so the gap is checkable without reading logs. The
default stays off rather than pointing at a fixed path, because ShuttleSet
has no matching root and the flag is specific to ShuttleSet22.

Both commands accept a subset: `--video-id sset_01 --video-id sset_02` for
the run export, `--match-id 8 --match-id 9` for ShuttleSet22. A subset is
the right way to spot-check before a full run. On Carmack, two videos of
either corpus export in about 40 seconds on the CPU, so the full 40-video
and 47-video corpora each take about 15 minutes.

## Kept features and their formulas

### Posture variability

This is Ari's feature from issue #22. It measures how much a player scrunched
up or leaned over during a rally. A player under pressure changes shape a lot.
A comfortable player stays upright.

For each frame, take the average height of the two eye keypoints and the
average height of the two ankle keypoints, then measure the gap between them.
Divide that gap by the width between the two hip keypoints:

```text
posture_t = |mean(eye_y) - mean(ankle_y)| / hip_width
```

`hip_width` is the straight-line distance between the left and right hip
keypoints, in image pixels. Both parts of the ratio are measured in the same
picture, so the result has no units. A frame missing any of those keypoints
has no posture value.

Two choices in that formula are deliberate. Eye-to-ankle is used rather than
the height of the player's bounding box, because a raised racket arm stretches
the box and would report a posture change that never happened. Hip width is
the denominator because it is the most camera-invariant and body-invariant
measure Ari could find. It still shrinks when a player turns side-on, although
that turn usually does mean effort.

The rally-level value is the median absolute deviation, or MAD:

```text
posture_mad = median( |posture_t - median(posture_t)| )
```

Both medians run over the rally's frames that have a finite posture value.
Read it as three steps. Find the middle posture value for the rally. Measure
how far each frame sits from that middle value, ignoring sign. Take the middle
of those distances. Medians are used because a handful of bad frames cannot
drag a median around the way they drag an average.

`posture_mad` is `derived`. The formula is complete, it produced a value for
99.57% of player-rallies in the benchmark, and its median stayed between 1.022
and 1.029 when any single video was left out. Those results say the signal is
broadly available and stable. No independent posture ground truth exists to
check it against, so treat it as a repeatable measurement with no biomechanics
validation behind it. It is null when no frame in the rally had a finite value.

### Rally timestamps

Each `rallies` row carries its span in frames and in seconds, plus the frame
rate those two are related by.

- `start_frame` and `end_frame` are zero-based frame indices on the whole
  source video. The interval is half-open, so `start_frame` is inside the
  rally and `end_frame` is the first frame after it. `duration_frames` is
  `end_frame - start_frame`.
- `start_seconds`, `end_seconds`, and `duration_seconds` are those frame
  numbers divided by `fps`.
- `fps` is stored on every row. The corpus mixes 25 and 30 frame-per-second
  videos, so a single project-wide constant would be wrong for many rows.

Issue #22 asks for time to be normalised to base-30 frames. Do that conversion
yourself, using the row's own `fps`:

```text
base_30_frames = frames * 30 / fps
```

`duration_seconds` is the length of the span. The issue #22 rally duration is
measured from the final contact plus an offset, and issue #32 fixed the
offsets: 2 seconds before the first contact for the serve setup, and 3 seconds
after the last contact for the shuttle to land. `clip_start_frame` and
`clip_end_frame` store those bounds at the row's own `fps`, clamped to the
video, so `clip_end_frame - start_frame` is that rally duration on a
`source_contacts` row. On an `annotator` row `end_frame` is a predicted end
rather than a final contact, so the same columns are only a padded span.

### Interpolation and its provenance

Pose detection drops frames, so the player signals have gaps. Filling those
gaps helps, but only when a reader can see which frames were filled.

A court scene is a run of frames where one court homography holds and the
court is visible. Gaps are filled inside one court scene only, and only
between two observed frames. Nothing is filled before the first observation of
a scene or after its last one, because those fills would have no bounds.

Every frame carries an `interpolation_type` code in the matching array. `0`
means observed. `1` means filled by linear interpolation. `2` means backward
extrapolated. Code `2` is defined in the code and never emitted in v1, because
issue #22 does not define a safe policy for it.

`player_rallies` reports the counts per rally and side.
`posture_frames_valid` is how many frames had a value, and
`posture_frames_linear` is how many of those were filled.
`position_frames_valid` and `position_frames_linear` do the same for court
position. Use the ratio as a quality filter, because a rally whose signal is
mostly filled says less about the player than one that is mostly observed.

### Source contacts

`source_contacts` carries human ShuttleSet labels. The kept source fields are
the set number, the rally number within that set, the shot number within that
rally (`ball_round`), the stroke type (`contact_type`), and the contact frame
(`frame_num`). `flaw_marked` is kept too: it is ShuttleSet's own per-row flag,
unrelated to the curated `flaw_shot_records.csv` defect list (29 rows) the
classifier pipeline uses elsewhere. Every other ShuttleSet column is excluded
from v1 by decision, and adding one later is a schema change.

Four columns are added by the export rather than copied from the source.
`source_row` is the row's position in its set CSV, which keeps duplicate
source rows distinct. `contact_type_en` is the English name for the stroke
type, from the shared classifier taxonomy. ShuttleSet22 writes passive drop
with a homophone character, `過度切球` for the taxonomy's `過渡切球`; the
reader maps that one variant and leaves the verbatim label untouched. Any
other unmapped label gives a null. `player_id` is the hitter, resolved from
the source player letter through the match table; see Players below.
`rally_id` links the contact to its row in `rallies`, and is null when the
rally was unusable: an invalid frame, or contacts out of order. A
flaw-marked row does not null it; see below.

A flaw-marked row stays in its rally. On the 40 ShuttleSet videos, 1,314 of
33,486 contact rows carry the flag, across 1,154 of the 3,359 human rallies
(34.4%). 86.9% of the flagged rows (1,142) are the serve (`ball_round` 1).

Serve timing barely separates flagged from clean: the frame gap to the
rally's second shot has a median of 26 frames on a flagged serve against 21
on a clean one, and only 14.8% of flagged serves sit under 3 frames against
0.1% of clean ones. So roughly 150 of the 1,142 flagged serves carry a
visibly broken frame number; the other ~1,000 have ordinary timing, and for
those the flag's meaning is not known. The 172 flagged rows that are not the
serve are different: their median gap is 0 frames and 82.6% sit under 3, so
nearly all of those are a broken frame number too.

No upstream definition of this column exists anywhere in the repository. One
signal does separate flagged rallies from clean ones: 165 flaw-marked rallies
have no `lose_reason` recorded, against 6 clean. That fits a faulted or
replayed serve, and fits an annotator skipping the ending on a row already
flagged. It does not settle the meaning. Stroke type, `ball_round`, and the
hitter stay sound on every flagged row regardless of what the flag means.

For most flagged rows the flag's meaning is unknown. On the roughly 320 rows
with a demonstrably broken frame (about 150 serves plus the 172 others),
every frame-anchored value built from that row is wrong: `rallies.start_frame`,
`end_frame`, `duration_frames`, `duration_seconds`, `start_seconds`,
`end_seconds`, `clip_start_frame`, `clip_end_frame`; `player_rallies.posture_mad`,
`posture_frames_valid`, `posture_frames_linear`, `position_frames_valid`,
`position_frames_linear`, `recovery_distance_median`,
`movement_inefficiency_median`; and, on the flagged contact and its
neighbours, `source_contacts.recovery_distance`, `recovery_frames_valid`,
`movement_inefficiency_top`, `movement_inefficiency_bottom`.

Because of this, a flaw-marked row no longer drops its rally. Keeping it does
not depend on knowing what the flag means, and it was requested by the
feature's owner. The old whole-rally rule dropped 1,154 of 3,359 ShuttleSet
rallies and 542 of 3,964 ShuttleSet22 rallies, just for having one flagged
row. Those rallies are now kept: they get a `rally_id` and a span like any
other, first contact frame to one past the last. `rallies.flaw_marked` marks
them instead: true when any of the rally's contact rows carries the flag, so
a consumer doing frame-anchored work should filter on it. A rally is still
excluded only for an invalid frame or contacts out of order, exactly as
before.

The next three features are computed only for `source_contacts` rows,
because every one of them needs a real contact frame and, for recovery, the
hitter's identity. An `annotator` row is a predicted span with neither: it
has no per-shot contact rows at all, so there is nothing to count or window
around. Guessing from a predicted span would be exactly the kind of
plausible-looking, unverified number this dataset avoids (see "What is
absent and why" below).

### Shots per rally

`rallies.shots_per_rally` is the count of human contact rows in the rally:

```text
shots_per_rally = len(contact_frames)
```

Issue #104 measured this against the production annotator's predicted
contacts and found it exact on only 298 of 3,287 eligible rallies: the
predicted count either missed a real shot or invented one that never
happened. On a `source_contacts` row the count is exact by construction,
because it counts the same human contacts the rally's span is built from.
It is null on `annotator` rows, which have no contact rows to count.

### Away-from-centre recovery

For every contact, `source_contacts.recovery_distance` is the mean distance
the non-striking player kept from their own half-court centre, over the
+/- 5 base-30-frame window around that contact:

```text
recovery_distance = mean( |position_t - half_centre| ) for t in [contact - w, contact + w]
```

`half_centre` is `(0.5, 0.25)` for a player on the top half of the court and
`(0.5, 0.75)` for the bottom half, in the same normalised doubles-court
coordinates as `player_signals.court_position`, where `1.0` is the full
width or length of the doubles court. So `recovery_distance` is a normalised
doubles-court Euclidean distance, unitless. `w` is 5 base-30 frames,
converted to the row's own `fps`. The window clips to the rally: a contact
near the start or end of a rally gets a shorter window rather than reaching
into the previous or next rally. This is deliberate, not just a boundary
convenience: a frame before the serve shows the receiver set to receive, not
recovering, and a frame after the rally's last contact is dead time before
the next rally starts. Neither belongs in a recovery mean.

Recovery only means something once you know which player was not hitting
the shuttle. The hitter is `source_contacts.player_id`; the exporter matches
it against the rally's top and bottom player ids and treats whichever one is
not the hitter as the measured player. When the hitter is null or matches
neither player, `recovery_distance` is null rather than guessed.
`recovery_frames_valid` is how many frames of the window had a finite
position, the same kind of provenance count as
`player_rallies.posture_frames_valid`. It is zero, never null, both when the
window has no valid position and when no window could be built at all, so
`recovery_distance` is null exactly when `recovery_frames_valid` is zero.

Issue #104 cut this feature because the production annotator's predicted
contacts and predicted server were too weak to build a reliable window or
know which player was recovering. Human ShuttleSet contacts fix the contact
frame, and the hitter resolved through the match table fixes the recovering
player, so both weak inputs are gone.

`player_rallies.recovery_distance_median` is the median of one side's
`recovery_distance` values over the rally, i.e. the contacts where that side
was not striking. It ignores the contacts where `recovery_distance` is
null, and is itself null when the side has no non-null value in the rally.

### Movement inefficiency

For every contact except a rally's last, `source_contacts.movement_inefficiency_top`
and `movement_inefficiency_bottom` are how much extra distance each player
travelled between that contact and the next one, compared to a straight line
between their positions at the two contacts:

```text
movement_inefficiency = path_length - straight_line_displacement
```

measured over the closed interval from this contact's frame to the next
contact's frame in the same rally. Like recovery, this is a normalised
doubles-court Euclidean distance. A player who moves in a straight line
between the two contacts scores 0; a player who takes a longer route scores
higher. A side's value is null when any frame in the interval has no finite
position for that side, and both sides are null on a rally's last contact,
which has no next contact to define an interval.

Ari's spec line for this feature, "Deception (or skill at reading opponent):
`Player_t -> Player_t+1`", is read as naming the contact-to-contact interval
this measure is taken over, not as asking for a separate value. No distinct
deception column is computed.

Issue #104 cut this feature because the production annotator's predicted
contacts missed or added events, so an interval built between two predicted
contacts often did not match a real rally exchange. Human ShuttleSet
contacts fix each interval's start and end exactly.

`player_rallies.movement_inefficiency_median` is the median of one side's
interval values over the rally, ignoring nulls the same way the recovery
median does.

Issue #104's benchmark numbers for these two formulas, on the production
annotator's predicted contacts, were coverage of 38,155 of 40,962 recovery
windows and a leave-one-video-out median distance of 0.144 to 0.145, and
coverage of 74,056 of 74,914 movement intervals and a leave-one-video-out
median of 0.0595 to 0.0605. Those numbers describe the old predicted-contact
prototype, not this export: on human contacts the population and the exact
coverage and median differ, and this document does not restate them until
someone runs the real export and recomputes them the same way.

### Players

`players` has one row per person the export references: `player_id`,
`player_name`, and `sex`. The rows come from `configs/players.csv`, a
hand-maintained table of the 43 people named in the ShuttleSet and
ShuttleSet22 match tables. `sex` is `female` or `male` and means the BWF
singles draw the player competes in, women's or men's singles. It is not
inferred from a name or from video. Both players of a singles match are in
the same draw, so the exporter refuses a match whose two players disagree.
That check passes for all 102 matches across the two datasets. To add a
player, add a row to the CSV.

The column exists because posture variability divides by hip width, and hip
width differs between men and women. A reader who compares `posture_mad`
across players should group or normalise by `sex`.

Four columns carry the foreign key. `source_contacts.player_id` is the
hitter. ShuttleSet labels the match winner `A` and the loser `B`, and the
match table names both. `player_rallies.player_id` is the person on
`court_side` during the rally, and `rallies.top_player_id` and
`rallies.bottom_player_id` repeat the same two ids on the rally row, so a
rally links to its players without a join. The match table's `downcourt` flag says whether `A` starts on the
top (far) court. Sides swap for set 2, and set 3 changes ends after the rally
in which a score first reaches 11. This is the rule the classifier training
code already uses in `classifier_shared/player_mapping.py`.

The rule was checked against the hitters' pixel positions in the ShuttleSet
set CSVs on 3 September 2026. Player `A` is the match winner in all 44
matches, `downcourt` agrees with the first rally's positions in all 44, and
the rule agrees with the pixel-derived side in 98.9% of 3,568 rallies. The
disagreements are scattered single rallies, some inside sets 1 and 2 where a
change of ends cannot happen, so they are location-annotation noise rather
than a wrong rule. The rule is the source; the pixels are not used.

For a `source_contacts` rally the assignment is exact, because the rally's
set and score phase are known. For an `annotator` rally the export looks for
the side phase whose human-contact frame envelope overlaps the predicted
span. When exactly one phase overlaps, the row gets that phase's players.
When none overlaps, for example a span in the break between sets, or more
than one does, `player_id` is null. It is also null on every row of a video
that has no source annotations.

### Player degradation trends

Issue #138 asks for progression over set or rally: fit a trend line through a
feature's values and see whether a player gets better or worse as a match
wears on. `player_trends` is that trend, one row per player, scope, and
feature.

Only `source_contacts` rallies take part, because they are the only ones with
an exact `player_id`; an `annotator` row's identity is a guess and would make
the trend meaningless.

Two feature sources are trended, both read only from `source_contacts`
rallies:

- Every float-valued column of `player_rallies`: `posture_mad`,
  `recovery_distance_median`, and `movement_inefficiency_median` today.
  Found by column type rather than a hardcoded name, so a new float feature
  is trended automatically once it lands in that table.
- Named rally-level columns of `rallies`: `duration_seconds` and
  `shots_per_rally` today. A rally-level value is read once for each player
  named on the rally, `top_player_id` and `bottom_player_id`.

Serve speed proxy is not trended. Issue #104 left the feature itself
unresolved, so there is nothing yet to trend.

`scope` is `set` or `match`, reading issue #138's two named progressions:

- `scope=set` fits a feature's values within one ShuttleSet set, one point
  per rally the player played there, against that rally's `source_rally`
  number. A missing rally keeps its gap: the fit uses the real rally number,
  not a renumbered position, so rallies 1, 2, 5 are not read as three
  consecutive points. `scope_id` is the set number. This needs at least 3
  points, `MIN_TREND_POINTS_SET` in `degradation.py`; fewer points make a
  rally-by-rally line noise, not a trend. That floor is this project's
  choice, not something issue #138 sets.
- `scope=match` fits one point per set: the median of the feature over the
  player's rallies in that set, against the set number. `scope_id` is the
  fixed sentinel `0`, because no ShuttleSet set is ever numbered `0`. This
  needs at least 2 points, `MIN_TREND_POINTS_MATCH`. A ShuttleSet match has 2
  or 3 sets, and a line through 2 points is an exact fit, not a guess, so a
  2-set match still gets a trend.

Each video's manifest records how many fits were written and, per scope, how
many were skipped for too few points: `fits_written`,
`fits_skipped_insufficient_points_set`, and
`fits_skipped_insufficient_points_match`, under `trend_population`, alongside
the existing `source_population` counts.

The trend itself is ordinary least squares, `numpy.polyfit` degree 1:

```text
slope = polyfit(position, feature_value, degree=1)
```

`position` is the rally's `source_rally` number for `scope=set`, or the
set's `source_set` number for `scope=match`.

`slope_tanh = tanh(slope / temperature)` compresses the slope to `(-1, 1)` so
trends of differently scaled features read on one comparable scale. The
temperature is a single fixed constant, `DEGRADATION_TEMPERATURE = 2.0` in
`src/dataset_builder/degradation.py`. Issue #22 left this temperature
undefined. Issue #138 asked to sweep a range of values if that was cheap, and
otherwise pick a magic number like 2; the sweep was skipped, so 2 is that
named fallback. The raw `slope` is kept alongside `slope_tanh` so a reader
who wants different scaling can recover it exactly:
`slope = temperature * arctanh(slope_tanh)`. That is why the temperature is
stored on every row rather than left implicit in the code.

`player_trends` is `derived` on every column. It is a computation over
`player_rallies` and `rallies`, both themselves unvalidated against
independent ground truth, so the trend inherits that caveat rather than
adding a new one.

### The primitive bundle

`primitive_artifacts` lists files. It does not contain their contents. Each
row names one artifact for one video, with its location, path, md5, size,
reliability class, and a note. The note is the warning that must travel with
the file. TrackNet shuttle positions, for example, had a median error of 0.459
court units at human contact frames. The doubles court is 1.0 unit wide, so
that error is large and those positions must never be described as accurate.

The `player_signals/<video_id>/` arrays are the derived side of the bundle.
They hold the per-frame posture and court position for the top and bottom
players, plus the two interpolation-code arrays. Every array has one row per
decoded frame, so frame `i` lines up across all of them and with the frame
numbers in `rallies`.

### Commentary tables

`transcript_segments` holds normalised transcript segments.
`commentary_chunks` holds the triaged chunks with their raw text, cleaned
text, and cleaning diagnostics. Both are tied to the video and its timeline.
`commentary_rally_links` additionally links a chunk to the `source_contacts`
rallies it plausibly refers to. All three are auxiliary material for future
work, and they sit beside the annotation data rather than inside it.
Ari's spec also asked for a per-chunk sentiment label (positive/negative/
neutral) and a single-word concept descriptor; neither is in this table,
and both stay unresolved and gated on a labelled sample in
[`issue_104_follow_ups.md`](dataset_builder/issue_104_follow_ups.md).

`timestamp_precision` says how a chunk or segment's times were produced.
`caption` means automatic caption timing. `whisperx_coarse` means
segment-level WhisperX timing, not word-level. `whisperx_aligned` means the
issue #136 forced aligner replaced the coarse time with a matched word
boundary; only `commentary_chunks` rows can carry it, because only chunks
have a re-timed sidecar. A row keeps its coarse label when the aligner found
no match (`unmatched`) or its aligned start collided with another chunk's
(`collision`); its `start_seconds`/`end_seconds` are unchanged in that case.

#### The rally-link pairing rule

The original post-rally join paired only 2.24% of production rally spans,
with 12 of its 77 pairs claiming text that started inside a different rally,
so issue #104 cut rally association from v1. Issue #136 then re-timed 92.6%
of chunks to word boundaries, and issue #138 asked a different question: not
"does the first chunk after a rally belong to it", but "which rallies could
this chunk's start plausibly be commentary for". A chunk links to the rally
it starts inside (`relation` `inside`, `lag_seconds` `0.0`), plus every rally
that ended within `LAG_SECONDS` before it (`relation` `post_rally`,
`lag_seconds` the gap). A chunk matching more than one rally gets one row per
rally, every one marked `ambiguous`. A chunk starting exactly at a rally's
end matches neither test (the inside test excludes the end, the post-rally
test requires strictly after it), so it gets no link; the benchmark has the
same gap.

`LAG_SECONDS` is 10 seconds, chosen by sweeping the aligned times over 6,156
combined human rallies from ShuttleSet and ShuttleSet22:

| Lag window | Combined coverage | Ambiguous chunks |
| ---: | ---: | ---: |
| 8 s | 34.4% | 0 |
| 10 s | 36.1% | 0 |
| 15 s | 39.6% | 8 |
| 20 s | 42.5% | 97 |

Ten seconds is the widest window with zero ambiguous chunks, and coverage
keeps climbing past it with no natural knee, so a wider window is a
judgement call this schema does not make. The full sweep, from 2 to 20
seconds, is in
[the aligned rerun](dataset_builder/issue_104_shuttleset_benchmark.md#aligned-rerun-issue-136).

That zero-ambiguity count comes from the benchmark's `_multi_rally_counts`,
which skips any chunk starting on a masked frame and removes masked rallies
from the candidate set before counting ambiguity. This exporter keeps both:
a masked chunk still gets a link, and a masked rally still counts as a
candidate. So the swept figure describes the benchmark's filtered
population, not the population this table ships. The manifest's per-video
`commentary_ambiguous_links` and `commentary_multi_rally_chunks` counts (see
below) give the real ambiguity count for the shipped table.

`commentary_rally_links` covers only `source_contacts` rallies. Annotator
rallies are predicted spans, and the benchmark showed the two rally origins
cannot be safely mixed, so linking commentary to a predicted span was not
attempted.

#### Replay-masked chunks are linked, not dropped

A chunk that starts on a replay-masked frame usually describes the rally
that was just replayed, not the live rally at that timestamp. The benchmark
treated such a chunk as unpairable and dropped it, which is a conservative
choice, not a settled one: dropping it may throw away real signal. This
schema instead links the chunk normally and sets
`starts_on_masked_frame` so a consumer can filter either way.

The flag matters because it is common: of 2,892 ShuttleSet chunks, 1,890
start on a replay-masked frame under the benchmark's own mask handling. A
consumer who ignores the flag is treating almost two-thirds of ShuttleSet's
chunks as if they describe the rally their timestamp lands on, when many of
them more likely describe the replay of the rally before it.

That 1,890 figure is not this flag's count. The benchmark loaded each
video's `raw_replay_mask` and applied its own duration filter
(`filter_short_exclusion_runs`) on top; it never unions that with
`~court_present`. This exporter loads `definitive_exclusion_mask.npy.xz`,
production's array, which already has `filter_short_exclusion_runs` applied
and is unioned with `~court_present` (`src/annotator/run_video.py`,
`_finalize_exclusion_mask`). The two masks are built from different arrays,
so `starts_on_masked_frame` and the manifest's masked-start count on a real
export can differ from 1,890 of 2,892; that figure describes the benchmark's
population, not this flag's.

The flag needs `--replay-mask-root` at export time. Without it,
`starts_on_masked_frame` is null throughout and the manifest's
`commentary_masked_start_chunks` count is null for every video, saying
plainly that masking was never checked rather than implying zero masked
chunks.

Each video's manifest entry also carries `commentary_ambiguous_links`, the
count of `commentary_rally_links` rows for that video with `ambiguous` true,
and `commentary_multi_rally_chunks`, the count of distinct chunks behind
those rows. An ambiguous chunk contributes one row per rally it matches, so
`commentary_ambiguous_links` is always at least twice
`commentary_multi_rally_chunks` whenever either is non-zero. Unlike
`commentary_masked_start_chunks`, both are always computed on the shipped
population, with or without `--replay-mask-root`.

#### Accuracy is unmeasured

Coverage is fixed: the lag rule above always links a chunk to a specific set
of rallies. Whether a linked chunk actually talks about that rally is a
different question, and nobody has labelled a sample to check it. That is
why issue #104's disposition for rally association is `unresolved`, not
`keep`: the link ships because its coverage and construction are honestly
described, but a reviewer should treat every row as a plausible candidate,
not a verified label, until that sample exists.

## What is absent and why

No replacement heuristic was added for any absent feature. A plausible-looking
number with no evidence behind it is more dangerous than a missing column,
because a later reader cannot tell the difference.

| Group | What it means | Examples |
| --- | --- | --- |
| Unresolved | A definition or a data source is missing, so no value could be produced honestly. | Serve speed proxy, backward extrapolation, commentary sentiment, rally-to-commentary association. |
| Not measured | The trial never defined or benchmarked it. | Rest time, smash shuttle speed, stroke duration, split-step stance geometry, match duration. |
| Out of scope | Outside the trial, with no gate planned. | Net-game share, backhand proportion, forced-to-unforced error ratio, hit height, shot-selection deception. |

Rally-to-commentary association is the one unresolved feature that is not
absent. `commentary_rally_links` ships because its coverage is fixed and its
construction is fully described; the unresolved half is only accuracy,
covered above under [Accuracy is unmeasured](#accuracy-is-unmeasured). Every
other unresolved feature in the table has no columns in v1 at all.

The Feature dispositions table below lists every trial feature with its exact
reason. [`issue_104_follow_ups.md`](dataset_builder/issue_104_follow_ups.md)
holds the gate for each cut and unresolved feature: what has to change before
it can come back.

## Data dictionary

Generated from `src/dataset_builder/schema_v1.py`. Regenerate it with
`uv run python scripts/dataset_v1_dictionary.py` and paste the output between
the two markers. `tests/test_dataset_v1_schema_doc.py` fails when this block
and the module disagree.

<!-- dictionary:start -->

### rallies

File `rallies.csv.gz`. Key `(run_id, source_dataset, video_id, rally_origin, rally_id)`.

One row per rally. Annotator rows come from the production rally records. source_contacts rows come from usable human ShuttleSet contact rows. The two origins are never joined to each other here; the benchmark showed that join is unsafe.

| Column | Type | Nullable | Reliability | Description |
| --- | --- | --- | --- | --- |
| `run_id` | string | no | observed | Immutable dataset-builder run that produced the row. |
| `source_dataset` | string | no | observed | Dataset label that namespaces video identifiers, for example ShuttleSet. |
| `video_id` | string | no | observed | Exact string video identifier. Never coerce it to a number: 0012 and 12 differ. |
| `rally_origin` | string | no | observed | annotator: a predicted span from the production annotator. source_contacts: the half-open span from the first to one past the last usable human contact of one ShuttleSet rally. |
| `rally_id` | int64 | no | observed | Zero-based list position within one (run_id, source_dataset, video_id, rally_origin) group. Not stable across runs or origins. |
| `fps` | float64 | no | observed | Probed constant frame rate of the decoded source. Stored per row so time normalisation is explicit: base-30 frames = frames * 30 / fps. |
| `frame_count` | int64 | no | observed | Decoded frame total of the source video. Every frame-aligned array has this many rows. |
| `start_frame` | int64 | no | by_rally_origin | First frame of the rally, zero-based on the whole source video, included. |
| `end_frame` | int64 | no | by_rally_origin | One past the last frame of the rally. Intervals are half-open. |
| `duration_frames` | int64 | no | by_rally_origin | end_frame - start_frame at the source frame rate. |
| `start_seconds` | float64 | no | by_rally_origin | start_frame / fps on the source-video timeline. |
| `end_seconds` | float64 | no | by_rally_origin | end_frame / fps on the source-video timeline. |
| `duration_seconds` | float64 | no | by_rally_origin | duration_frames / fps. This is the span length, not the issue #22 rally duration from the final contact plus an offset. |
| `clip_start_frame` | int64 | no | by_rally_origin | start_frame minus 2 s of lead-in at the source frame rate, clamped at 0. Issue #32 context for the serve setup. |
| `clip_end_frame` | int64 | no | by_rally_origin | end_frame plus 3 s of tail at the source frame rate, clamped at frame_count; one past the last clip frame. On source_contacts rows the issue #22 rally duration from the final contact plus offset is clip_end_frame - start_frame. |
| `source_set` | int64 | yes | source_annotation | ShuttleSet set number for source_contacts rows. Null for annotator rows. |
| `source_rally` | int64 | yes | source_annotation | ShuttleSet rally number within its set for source_contacts rows. Null for annotator rows. |
| `top_player_id` | string | yes | derived | players.player_id of the person on the top court during this rally; same derivation and null cases as player_rallies.player_id. |
| `bottom_player_id` | string | yes | derived | players.player_id of the person on the bottom court during this rally; same derivation and null cases as player_rallies.player_id. |
| `shots_per_rally` | int64 | yes | derived | Count of human contact rows in this rally, exact by construction. Null on annotator rows, which have no contact rows. |
| `flaw_marked` | bool | no | source_annotation | True when any human contact row in this rally carries the ShuttleSet flaw flag. No upstream definition of this flag exists; for most flagged rows its meaning is unknown. On roughly 320 of the 40-video corpus's 1,314 flagged rows the contact frame number is demonstrably wrong (see docs/dataset_v1_schema.md, Source contacts), which corrupts every frame-anchored value derived from that rally. Stroke sequence, counts and hitters stay sound regardless. Filter on this column before trusting a frame-anchored value. False on annotator-origin rows. |

### player_rallies

File `player_rallies.csv.gz`. Key `(run_id, source_dataset, video_id, rally_origin, rally_id, court_side)`.

One row per rally and court side with the kept issue #22 features. Cut and unresolved features are absent by decision, not by omission.

| Column | Type | Nullable | Reliability | Description |
| --- | --- | --- | --- | --- |
| `run_id` | string | no | observed | Immutable dataset-builder run that produced the row. |
| `source_dataset` | string | no | observed | Dataset label that namespaces video identifiers, for example ShuttleSet. |
| `video_id` | string | no | observed | Exact string video identifier. Never coerce it to a number: 0012 and 12 differ. |
| `rally_origin` | string | no | observed | annotator: a predicted span from the production annotator. source_contacts: the half-open span from the first to one past the last usable human contact of one ShuttleSet rally. |
| `rally_id` | int64 | no | observed | Zero-based list position within one (run_id, source_dataset, video_id, rally_origin) group. Not stable across runs or origins. |
| `court_side` | string | no | derived | top or bottom: the court half the sticky-player picker assigned. A row belongs to a side within one rally, not to a person. |
| `player_id` | string | yes | derived | players.player_id of the person on court_side in this rally. source_contacts rows: exact, from the match table's downcourt flag, the set number, and the set-3 change of ends when a score first reaches 11. annotator rows: the one side phase whose human-contact frame envelope overlaps the span; null when no phase or more than one overlaps, or when the video has no source annotations. |
| `posture_frames_valid` | int64 | no | derived | Frames in the rally with a finite posture value after bounded linear interpolation. |
| `posture_frames_linear` | int64 | no | derived | Of posture_frames_valid, frames filled by linear interpolation between observed frames inside one court scene. |
| `posture_mad` | float64 | yes | derived | Posture variability: median absolute deviation over the rally of the per-frame posture \|mean eye y - mean ankle y\| / hip width. Unitless. Null when no frame has a finite value. A derived signal, not validated biomechanics. |
| `position_frames_valid` | int64 | no | derived | Frames with a finite court-normalised mean-ankle position after bounded linear interpolation. |
| `position_frames_linear` | int64 | no | derived | Of position_frames_valid, frames filled by linear interpolation. |
| `recovery_distance_median` | float64 | yes | derived | Median of this side's source_contacts.recovery_distance values over the rally, i.e. the contacts where this side was not striking. Unitless: normalised doubles-court Euclidean distance. Null when no contact in the rally has a recovery_distance for this side. |
| `movement_inefficiency_median` | float64 | yes | derived | Median of this side's source_contacts.movement_inefficiency_top or _bottom values over the rally. Unitless: normalised doubles-court Euclidean distance. Null when no interval in the rally has a value for this side. |

### player_trends

File `player_trends.csv.gz`. Key `(run_id, source_dataset, video_id, player_id, scope, scope_id, feature)`.

One row per player, scope, and trended feature: an ordinary least squares trend over that player's source_contacts rallies or sets, plus its tanh-normalised slope. Annotator rallies are excluded because their player identity is a guess, not a label.

| Column | Type | Nullable | Reliability | Description |
| --- | --- | --- | --- | --- |
| `run_id` | string | no | observed | Immutable dataset-builder run that produced the row. |
| `source_dataset` | string | no | observed | Dataset label that namespaces video identifiers, for example ShuttleSet. |
| `video_id` | string | no | observed | Exact string video identifier. Never coerce it to a number: 0012 and 12 differ. |
| `player_id` | string | no | derived | players.player_id of the person this trend is fit for. |
| `scope` | string | no | derived | set: trend across one player's rallies within one ShuttleSet set, ordered by source_rally. match: trend across that player's sets in the video, one point per set (the median of the feature over the player's rallies in that set), ordered by source_set. |
| `scope_id` | int64 | no | derived | ShuttleSet set number for scope=set. Fixed sentinel 0 for scope=match; no ShuttleSet set is ever numbered 0. |
| `feature` | string | no | derived | Feature this trend was fit over: a player_rallies float column (for example posture_mad) or a named rally-level column of rallies (duration_seconds, and shots_per_rally once that column exists). |
| `n_points` | int64 | no | derived | Values that fed the fit: rallies for scope=set (at least 3), sets for scope=match (at least 2). A fit with fewer points is not written. |
| `slope` | float64 | no | derived | Ordinary least squares slope of the feature value against its position: the rally's source_rally number for scope=set, or the set's source_set number for scope=match. |
| `slope_tanh` | float64 | no | derived | tanh(slope / temperature): the slope compressed to (-1, 1) so trends of differently scaled features are comparable. |
| `temperature` | float64 | no | derived | Tanh scaling constant used for slope_tanh, stored so the scaling reverses: slope = temperature * arctanh(slope_tanh). |

### players

File `players.csv.gz`. Key `(player_id)`.

One row per person referenced by this export. player_rallies.player_id and source_contacts.player_id are foreign keys to player_id.

| Column | Type | Nullable | Reliability | Description |
| --- | --- | --- | --- | --- |
| `player_id` | string | no | curated | Stable lowercase identifier shared by ShuttleSet and ShuttleSet22, from configs/players.csv. |
| `player_name` | string | no | curated | Display name, spelled as the ShuttleSet match tables spell it. |
| `sex` | string | no | curated | female or male: the BWF singles draw the player competes in. Both players of a singles match share the value; the exporter refuses a match where they differ. |

### source_contacts

File `source_contacts.csv.gz`. Key `(source_dataset, video_id, source_set, source_row)`.

Human ShuttleSet contact rows, restricted to the kept source fields: contact type, rally and shot numbers, set, and frame. All other ShuttleSet columns are excluded from v1. Rows are source-scoped and never annotator predictions. recovery_distance, recovery_frames_valid, movement_inefficiency_top, and movement_inefficiency_bottom are derived from this table's own contact order and the player-signal positions, not copied from ShuttleSet.

| Column | Type | Nullable | Reliability | Description |
| --- | --- | --- | --- | --- |
| `source_dataset` | string | no | observed | Dataset label that namespaces video identifiers, for example ShuttleSet. |
| `video_id` | string | no | observed | Exact string video identifier. Never coerce it to a number: 0012 and 12 differ. |
| `source_set` | int64 | no | source_annotation | Set number parsed from the ShuttleSet set CSV filename. |
| `source_row` | int64 | no | observed | Zero-based row position within that set CSV. Keeps duplicate source rows distinct. |
| `source_rally` | int64 | yes | source_annotation | ShuttleSet rally number within the set. |
| `ball_round` | int64 | yes | source_annotation | ShuttleSet shot number within the rally. |
| `player_id` | string | yes | source_annotation | players.player_id of the hitter: the source player letter resolved through the match table, where A is the match winner. Null when the letter is not A or B. |
| `frame_num` | int64 | yes | source_annotation | Human contact frame on the source-video timeline. Null when the source field is empty or not a number. |
| `contact_type` | string | yes | source_annotation | Verbatim ShuttleSet stroke-type label for the contact. |
| `contact_type_en` | string | yes | derived | English name for contact_type from the shared classifier taxonomy. Null when the label has no mapping. |
| `flaw_marked` | bool | no | source_annotation | True when the ShuttleSet flaw field is non-empty for this row. |
| `rally_id` | int64 | yes | derived | rally_id of the source_contacts row in rallies that this contact belongs to. Null when its rally was unusable: an invalid frame, or contacts out of order. A flaw-marked row does not null this; see rallies.flaw_marked. |
| `recovery_distance` | float64 | yes | derived | The measured player is this rally's other player, never the hitter: rallies.top_player_id when this row's player_id is bottom_player_id, and the reverse. recovery_distance is that player's mean distance from their own half-centre over the +/- 5 base-30-frame window around this contact, clipped to the rally. Unitless: normalised doubles-court Euclidean distance. Null when the hitter is not one of this rally's two players, or when no frame in the window has a finite position. |
| `recovery_frames_valid` | int64 | no | derived | How many frames of the recovery_distance window, for that same other player, had a finite position. Zero when the hitter could not be matched to a rally side, so no window could be built. Provenance, matching how the table records player_rallies.posture_frames_valid. |
| `movement_inefficiency_top` | float64 | yes | derived | Top player's path length minus straight-line displacement from this contact to the next contact in the rally. Unitless: normalised doubles-court Euclidean distance. Null on a rally's last contact, and null when a position in the interval is not finite. |
| `movement_inefficiency_bottom` | float64 | yes | derived | Bottom player's path length minus straight-line displacement from this contact to the next contact in the rally. Same units and null cases as movement_inefficiency_top. |

### primitive_artifacts

File `primitive_artifacts.csv.gz`. Key `(source_dataset, video_id, artifact)`.

Manifest of the raw primitive bundle: frame-aligned shuttle, pose, court, and mask artifacts from the run, plus the derived player-signal arrays. Files are referenced, not copied.

| Column | Type | Nullable | Reliability | Description |
| --- | --- | --- | --- | --- |
| `source_dataset` | string | no | observed | Dataset label that namespaces video identifiers, for example ShuttleSet. |
| `video_id` | string | no | observed | Exact string video identifier. Never coerce it to a number: 0012 and 12 differ. |
| `artifact` | string | no | observed | Canonical artifact name; see PRIMITIVE_ARTIFACT_NOTES. |
| `location` | string | no | observed | input_dir, export_dir, or inpainted_root: the root that relative_path is relative to. The dataset manifest records input_root and inpainted_root by name; export_dir is implicit, since the manifest file itself lives there. |
| `relative_path` | string | no | observed | POSIX path of the file under location. |
| `md5` | string | no | observed | MD5 of the stored file, matching the run manifest convention. |
| `size_bytes` | int64 | no | observed | Stored file size. |
| `reliability` | string | no | observed | Reliability class of the artifact's content. |
| `note` | string | no | observed | Reliability note for the artifact. |

### transcript_segments

File `transcript_segments.csv.gz`. Key `(source_dataset, video_id, segment_index)`.

Auxiliary component: normalised commentary transcript segments tied to the video, not to rallies. Rally association is cut from v1.

| Column | Type | Nullable | Reliability | Description |
| --- | --- | --- | --- | --- |
| `source_dataset` | string | no | observed | Dataset label that namespaces video identifiers, for example ShuttleSet. |
| `video_id` | string | no | observed | Exact string video identifier. Never coerce it to a number: 0012 and 12 differ. |
| `segment_index` | int64 | no | observed | Zero-based segment position in the normalised transcript. |
| `timestamp_precision` | string | no | observed | caption: automatic caption segment timing. whisperx_coarse: segment-level WhisperX timing, not word-level. whisperx_aligned: chunk timing replaced by forced-aligned word boundaries (issue #136); commentary_chunks only. None of the three were checked against rally boundaries; that check is the unmeasured part of commentary_rally_links. |
| `start_seconds` | float64 | no | observed | Segment start on the source-video timeline. |
| `end_seconds` | float64 | no | observed | Segment end on the source-video timeline. |
| `text` | string | no | observed | Normalised transcript text. May contain transcription errors. |

### commentary_chunks

File `commentary_chunks.csv.gz`. Key `(source_dataset, video_id, chunk_id)`.

Auxiliary component: relevance-triaged commentary chunks with raw and cleaned text, tied to the video. Sentiment, concept, and player link are unresolved and absent. commentary_rally_links carries the separate, unresolved rally association.

| Column | Type | Nullable | Reliability | Description |
| --- | --- | --- | --- | --- |
| `source_dataset` | string | no | observed | Dataset label that namespaces video identifiers, for example ShuttleSet. |
| `video_id` | string | no | observed | Exact string video identifier. Never coerce it to a number: 0012 and 12 differ. |
| `chunk_id` | string | no | observed | Cleaning-stage chunk identifier, unique within the video. |
| `timestamp_precision` | string | no | observed | caption: automatic caption segment timing. whisperx_coarse: segment-level WhisperX timing, not word-level. whisperx_aligned: chunk timing replaced by forced-aligned word boundaries (issue #136); commentary_chunks only. None of the three were checked against rally boundaries; that check is the unmeasured part of commentary_rally_links. |
| `start_seconds` | float64 | no | observed | Segment start on the source-video timeline. |
| `end_seconds` | float64 | no | observed | Segment end on the source-video timeline. |
| `text` | string | no | observed | Raw chunk text before cleaning. |
| `text_clean` | string | no | derived | Generated cleaned text. Not a human judgement of relevance or accuracy. |
| `bert_f1` | float64 | yes | derived | BERTScore F1 between text and text_clean. A cleaning diagnostic, not a truth probability. |
| `clean_pass` | bool | yes | derived | Whether the chunk passed the cleaning contract. |

### commentary_rally_links

File `commentary_rally_links.csv.gz`. Key `(run_id, source_dataset, video_id, chunk_id, rally_origin, rally_id)`.

One row per commentary chunk linked to one source_contacts rally under the issue #138 pairing rule. A chunk with more than one candidate rally gets one row per rally, all marked ambiguous. Coverage is measured; accuracy is not: nobody has labelled a sample to check that a linked chunk actually discusses its rally.

| Column | Type | Nullable | Reliability | Description |
| --- | --- | --- | --- | --- |
| `run_id` | string | no | observed | Immutable dataset-builder run that produced the row. |
| `source_dataset` | string | no | observed | Dataset label that namespaces video identifiers, for example ShuttleSet. |
| `video_id` | string | no | observed | Exact string video identifier. Never coerce it to a number: 0012 and 12 differ. |
| `chunk_id` | string | no | derived | commentary_chunks.chunk_id of the linked chunk. |
| `rally_origin` | string | no | derived | Always source_contacts: only ShuttleSet human-contact rallies are linked in v1. Kept as a column so the key joins directly to rallies. |
| `rally_id` | int64 | no | derived | rallies.rally_id of the linked rally. |
| `relation` | string | no | derived | inside: the chunk starts inside this rally's span. post_rally: this rally ended within LAG_SECONDS before the chunk started. |
| `lag_seconds` | float64 | no | derived | 0.0 for relation inside; otherwise the chunk's start_seconds minus this rally's end_seconds, the gap the pairing rule allowed. |
| `ambiguous` | bool | no | derived | True when this chunk links to more than one rally under the pairing rule; every one of its link rows carries the same value. |
| `starts_on_masked_frame` | bool | yes | derived | True when the chunk's start lands on a frame the optional replay mask marks excluded. Null when the export was not given a replay mask root; the export never drops a masked-start chunk on this basis. |

### Player-signal arrays

One directory per video under `player_signals/<video_id>/`. Every array has one row per decoded frame, so a frame index reads across all four.

| Array | File | Shape | Dtype | Reliability | Description |
| --- | --- | --- | --- | --- | --- |
| `posture` | `posture.npy.xz` | `(frame_count, 2)` | float64 | derived | Per-frame posture \|mean eye y - mean ankle y\| / hip width for the top and bottom sticky players, after bounded linear interpolation. NaN where unavailable. |
| `court_position` | `court_position.npy.xz` | `(frame_count, 2, 2)` | float64 | derived | Per-frame mean-ankle position of the top and bottom players projected into doubles-court unit coordinates, not clipped to [0, 1]. NaN where unavailable. |
| `posture_interpolation` | `posture_interpolation.npy.xz` | `(frame_count, 2)` | int8 | derived | interpolation_type per posture frame: 0 observed, 1 linear, 2 backward extrapolated (never emitted in v1). |
| `position_interpolation` | `position_interpolation.npy.xz` | `(frame_count, 2)` | int8 | derived | interpolation_type per court_position frame, same codes as posture_interpolation. |

### Primitive artifact notes

One note per artifact name that can appear in the `primitive_artifacts` table. The note is the reliability warning that travels with the file.

| Artifact | Reliability | Note |
| --- | --- | --- |
| `shuttle_track` | predicted | (frame_count, 3) TrackNet x, y normalised by resolution, and visibility. Median court error 0.459 units at human contacts. Do not describe as accurate. |
| `shuttle_guard_codes` | predicted | (frame_count,) inpaint hallucination guard grades. Mask rejected grades before using shuttle positions. |
| `shuttle_track_inpainted` | predicted | (frame_count, 3) TrackNet x, y normalised by resolution, and visibility, from a later InpaintNet pass over the ShuttleSet22 extract. The base ShuttleSet22 extract was run with InpaintNet off, so this replaces shuttle_track with a higher-visibility track. Do not describe as accurate. |
| `shuttle_guard_codes_inpainted` | predicted | (frame_count,) inpaint hallucination guard grades for shuttle_track_inpainted. Mask rejected grades before using shuttle positions. |
| `pose_kps` | predicted | (frame_count, slots, 17, 2) RTMLib keypoints per detection slot. Slots are not player identities. |
| `pose_bboxes` | predicted | (frame_count, slots, 4) detection boxes. NaN in inactive slots. |
| `pose_scores` | predicted | (frame_count, slots) detection scores. NaN in inactive slots. |
| `pose_kp_scores` | predicted | (frame_count, slots, 17) keypoint scores. |
| `pose_ndet` | predicted | (frame_count,) active detection count per frame. |
| `court_evidence` | predicted | Scene homography rows and gate inputs. Median corner error 4.34 px on ShuttleSet. |
| `court_keep_vote` | predicted | (frame_count,) CourtKeyNet keep vote mask. |
| `court_present` | predicted | (frame_count,) court-present mask that bounds every interpolation segment. |
| `raw_replay_mask` | predicted | (frame_count,) raw replay mask from the annotation stage. |
| `definitive_exclusion_mask` | predicted | (frame_count,) definitive exclusion mask from the annotation stage. |
| `posture` | derived | (frame_count, 2) float64. Per-frame posture \|mean eye y - mean ankle y\| / hip width for the top and bottom sticky players, after bounded linear interpolation. NaN where unavailable. |
| `court_position` | derived | (frame_count, 2, 2) float64. Per-frame mean-ankle position of the top and bottom players projected into doubles-court unit coordinates, not clipped to [0, 1]. NaN where unavailable. |
| `posture_interpolation` | derived | (frame_count, 2) int8. interpolation_type per posture frame: 0 observed, 1 linear, 2 backward extrapolated (never emitted in v1). |
| `position_interpolation` | derived | (frame_count, 2) int8. interpolation_type per court_position frame, same codes as posture_interpolation. |

### Feature dispositions

Every trial feature and where it ended up. Exported columns are named as `table.column`. A feature with no columns is absent from v1.

| Feature | Disposition | Columns | Reason |
| --- | --- | --- | --- |
| Rally timestamps: start, end, FPS, frame ranges | keep | `rallies.fps`, `rallies.start_frame`, `rallies.end_frame`, `rallies.duration_frames`, `rallies.start_seconds`, `rallies.end_seconds`, `rallies.duration_seconds` | Exact conversion over a complete population. Reliability follows rally_origin. |
| Posture variability (MAD) | keep | `player_rallies.posture_mad` | Formula complete, 99.57% coverage, stable leave-one-video-out medians. No independent posture ground truth, so labelled derived. |
| Linear-interpolation provenance | keep | `player_rallies.posture_frames_linear`, `player_rallies.position_frames_linear`, `player_signals.posture_interpolation`, `player_signals.position_interpolation` | Gaps are filled only between observations inside one court scene, and every filled frame is marked. |
| ShuttleSet source fields: contact type, round, set | keep | `source_contacts.source_set`, `source_contacts.source_rally`, `source_contacts.ball_round`, `source_contacts.contact_type` | Direct human-source fields, kept source-scoped and never presented as predictions. |
| Raw pose, court, and shuttle primitives | keep | `primitive_artifacts` | Kept as a separate referenced bundle with masks and reliability notes. |
| Commentary raw captions, normalised transcripts, cleaned text | keep | `transcript_segments`, `commentary_chunks` | Auxiliary component tied to the video with segment timestamps and a precision class. Not rally labels. |
| Rally duration from final contact plus offset | keep | `rallies.clip_start_frame`, `rallies.clip_end_frame` | Issue #32 fixed the offsets: 2 s before the first contact and 3 s after the last, clamped to the video. Exact on source_contacts rows; predicted spans on annotator rows. |
| Player identity and sex | keep | `players.player_id`, `players.sex`, `rallies.top_player_id`, `rallies.bottom_player_id`, `player_rallies.player_id`, `source_contacts.player_id` | Curated per-player table joined through the ShuttleSet match tables. Court sides map to people by the downcourt flag, the set number, and the set-3 change of ends. |
| Raw degradation slope | keep | `player_trends.slope`, `player_trends.n_points` | Issue #104 could not fit a trend without a retained feature set and stable player identity across rallies. Both now exist: player_rallies keeps float features and source_contacts rallies carry an exact player_id. |
| Tanh-normalised degradation | keep | `player_trends.slope_tanh`, `player_trends.temperature` | Issue #22 left the tanh scaling temperature undefined. Issue #138 asked to sweep it if that was cheap, and otherwise pick a magic number like 2. The sweep was skipped, so the feature's owner used that named fallback, 2.0; the raw slope is kept alongside it so the scaling reverses. |
| Shots per rally | keep | `rallies.shots_per_rally` | Issue #104 measured this against predicted contacts, exact on only 298 of 3,287 rallies. It is now the count of human ShuttleSet contact rows in the rally, exact by construction, so the weak input that cut it is gone. |
| Away-from-centre recovery | keep | `source_contacts.recovery_distance`, `source_contacts.recovery_frames_valid`, `player_rallies.recovery_distance_median` | Cut because predicted contact and server attribution were too weak for a player-specific window. Human ShuttleSet contacts fix the contact frame, and the hitter resolved against the match table's own side assignment fixes the non-striking player, so both weak inputs are gone. |
| Movement inefficiency | keep | `source_contacts.movement_inefficiency_top`, `source_contacts.movement_inefficiency_bottom`, `player_rallies.movement_inefficiency_median` | Cut because production intervals used predicted contacts that missed or added events. Human ShuttleSet contacts fix each interval's start and end exactly. |
| Serve speed proxy | unresolved | none | Return, static, and viewport endpoints are undefined and shuttle error is large. |
| Backward extrapolation | unresolved | none | No defined scene boundary, range, or provenance policy. |
| Rally-to-commentary association | unresolved | none | Issue #138's lag rule fixes coverage on aligned times with zero ambiguity at 10 s, but accuracy is unmeasured: nobody has labelled a sample to check the pairs are right. |
| Commentary sentiment, concept, and player link | unresolved | none | Supported schemas emit no semantic fields and no labelled population exists. |
| Out-of-position posture states | not_measured | none | The three states need pose-term definitions. |
| Rest time, work density, effective playing time | not_measured | none | Work density and cutaway handling need definitions. |
| Smash shuttle speed | not_measured | none | Needs smash classification, which production does not have. |
| Match duration | not_measured | none | Depends on complete-rally contacts. |
| Shot frequency within rally | not_measured | none | Depends on complete-rally contacts. |
| Aggression markers | not_measured | none | Depends on complete-rally contacts and shot classification. |
| Rally-length distribution by outcome and landing zone | not_measured | none | Depends on complete-rally contacts, outcome, and landing. |
| Stroke duration | not_measured | none | Needs a motion-onset definition. |
| Court coverage near the shuttle | not_measured | none | Needs a relative measure and event anchor. |
| Split-step stance geometry | not_measured | none | Needs a stance measure and event detector. |
| Net-game share, clear share, backhand proportion, forced-to-unforced error ratio, shot-outcome success by type, footwork-to-shot coupling, hit height, shot-selection deception | out_of_scope | none | Outside the trial. No gate planned. |

<!-- dictionary:end -->

## Provenance

Every export writes `dataset_manifest.json.gz` beside the tables. It records
the run id, the input root that `input_dir` paths are relative to, the code
version, the input manifest digest, the source annotation files with their
md5, the tables written with their md5 and row counts, one entry per video,
and the disposition registry.

It also records the curated player table as `players_table`, with its path,
md5, and size, and per video a `match_players` entry with the two player ids
and whether player A starts on the top court.

A ShuttleSet22 export has no dataset-builder run, so its code version and
manifest digests are null. In their place it records the sources TOML and,
per video, the court receipt with its code id and model md5. The run id is a
label the operator supplies for that artifact set.

That is enough to check whether two copies of the dataset are the same file
for file, and to trace any row back to the run and the source files that
produced it.
