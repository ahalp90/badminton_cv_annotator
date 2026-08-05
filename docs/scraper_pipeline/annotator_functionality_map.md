# Annotator functionality map

What the automatic annotator must do to one broadcast badminton video, written as
contracts so the outcome tests can be written against this map rather than against the
current code's shape. Contract text is the TARGET. Where a step already exists, its
current home is noted in brackets as "today: module.function" so claims can be checked;
where today's behaviour deliberately differs from the contract, the difference is
stated in place. The migration plan (the campaign's plan of record) rules where this
map and older docs differ.

Two scope notes up front:

- "Annotator" here means the automatic stroke-annotation chain that becomes
  src/annotator. It is unrelated to docs/annotator_unification_brief.md, which is about
  the human court-corner annotation GUI. Same word, different tool.
- The annotator consumes precomputed detection arrays and calibration. Producing them
  (video download, pose extraction, shuttle tracking, court fitting, commentary
  transcription) is acquisition work and stays in src/scraper.

## What the annotator produces, in one paragraph

Given one video and its detection arrays, the annotator finds every rally (start and
end frames), every racket-shuttle contact inside each rally, and per rally: the stroke
count, who hit the final stroke (striker), who served, whether each contact was struck
above or below net height, where the final shuttle landed, and who won the point.
"Every contact" here means the scoring contacts: detected candidates filtered down to
the credible ones (step 4 defines both sets). These are the annotation columns the
ground-truth scoring harness reads, in ShuttleSet's names: rally boundaries,
ball_round, hit-frame timing, player, server, hit_height, landing side,
getpoint_player. A separate per-span doubles screen says
whether a rally is doubles play (and therefore out of scope for the singles dataset).

## Inputs, per video

Each input is a contract; the concrete current source sits beside it.

1. The video's frame rate. fps is probed from the video file and fails loudly on
   variable-frame-rate input (today: fps_constants.probe_fps). A CLI whose inputs are
   arrays with no video to probe requires an explicit --fps. fps is never defaulted
   anywhere inside the chain.
2. Shuttle track: per frame, the shuttle's position and a visibility flag. Concretely a
   (frames, 3) float array of [x, y, visibility], x/y normalised to [0, 1]
   (TrackNetV3 output).
3. Player detections: per frame, up to N candidate people. Concretely four arrays:
   bboxes (frames, N, 4) in pixels, scores (frames, N), keypoints (frames, N, 17, 2)
   in pixels, and a detection count (frames,).
4. Court calibration: the homography (the image-to-court coordinate transform) and the
   court border lines for this video (today: a court_info dict with H and
   border_L/R/U/D), plus the resolution table keyed by video id (per-video frame
   width and height).
5. Court geometry in image space: the court geometry (x/y pixel ranges and a net
   band in image y). The net band is where a foot counts to neither court half.
6. Doubles evidence: a per-frame bool array, True where more than two people project
   inside the court (today: the vision lane's <video_id>_overcount.npy, produced on
   the compute cluster beside the pose output). Step 10 carries its limits and its
   planned successor.
7. Ground truth is never an input. GT enters only in the scoring harness, downstream
   of the annotator.

## Configuration

One base config carries the committed constants; a resolved config is built from it
exactly once per video at (base values, probed fps) and threads through every step as
an argument. Time-denominated constants live in the base-30 table (fps_constants), and
each table row declares how it scales: frame counts scale with fps, per-frame speeds
invert with fps, dimensionless values (fractions, ratios, multiples) never scale. That
declaration lives in the table itself (the ScalingKind of each row), so "which fields
scale, which way" has one home.

None semantics, stated once: injection arguments (below) use None to mean "compute
it". A strategy option field may use None with a documented meaning of its own (for
example "stillness gate off"). No config field anywhere uses None to silently
substitute a default value.

Slots: the two singles players are Top and Bottom, an integer enum whose members ARE
the values (TOP = 0, BOTTOM = 1) so it indexes the sticky result's arrays directly (today: sticky_anchor's
module ints SLOT_TOP = 0, SLOT_BOTTOM = 1). Output row order is pinned: Top is row 0,
Bottom is row 1. Selection order is the reverse and it matters: the picker assigns
Bottom first, so Bottom wins a shared-candidate or equidistant tie. The point-winner
layer's Half.TOP/Half.BOT name the same two players (the types module carries the one
Slot; the Half spelling survives only until its consumers migrate).

## The chain, step by step

Ordering has two hard dependencies. The dead-time mask (the per-frame "this frame is
replays or other non-play" mask) needs rally spans, because one of its signals
compares in-rally shuttle speed against the rest of the video. The serve gate needs
the sticky analysis (the player-picking pass defined in step 2). Sticky runs over
scene-gated tracker segments. The resolution is one tracker-segment sticky analysis,
one preliminary span pass for the mask, and one final segmentation pass:

1. preliminary segmentation, run unmasked and serve-ungated, finds bootstrap spans
   (a first cut of "when is play happening", used only to feed step 3)
2. one sticky analysis runs over the scene-gated tracker segments
3. the dead-time mask builds
4. final segmentation (masked, serve-gated) produces the rally spans and contacts
5. attribution, server fit, landing, winner, hit height, and the doubles screen
   (steps 5 to 10 below) read the final spans, the filtered contacts, and the one
   sticky result

### Step 1: preliminary segmentation (bootstrap spans)

Purpose: give the mask builder a first cut of play time before any mask or serve gate
exists. It is the same span-finding logic as step 4 run with no dead-time mask and no
serve gate (today: the span logic inside
annotator.rally_segmentation.segment_video).

Span-finding contract (shared with step 4), with the equality rules pinned:

- speed is the per-frame shuttle speed over the (possibly masked) track; a step
  needs the shuttle tracked in both of its frames, so a frame without one has speed
  NaN, and NaN is never fast
- a frame is FAST when its speed is strictly above start_speed
- a frame is AT REST when its rolling-median speed is strictly below rest_speed, OR
  when the visible fraction of its window is strictly below the visibility floor
  (mostly-untracked reads as rest)
- a LONG REST is a rest run at least end_rest_frames long; long rests separate active
  regions
- a QUALIFYING BURST is a fast run at least start_min_frames long
- inside each active region, the first qualifying burst that starts in the region
  opens the span; the span is half-open [burst start, region end). A region with no
  qualifying burst yields no span
- a video with no spans at all yields an empty annotation (zero rallies), reported as
  such rather than raised

### Step 2: one sticky analysis

Sticky is the player picker: per frame it picks the best Top and Bottom candidates
from the raw detections. Each slot carries an anchor blending a fixed half-court prior
with an exponential moving average of the slot's recent picks, so the pick is biased
toward where that player recently stood while the player stays detected. It stays in
bst_x, the project's stroke-classification model package (today:
bst_x preparing_data/heuristics/sticky_anchor.py, reached through a sys.path hack that
dies in this migration; the annotator imports it as a real package with public
names).

It runs ONCE per video, sequentially over the scene-gated tracker segments, resetting
its moving average at each tracker-segment boundary and on picker failure (reset details
below). The result is immutable and every consumer named in the table reads from it. Its fields,
coordinate systems, and owners:

| Field | What it is | Who consumes it |
|---|---|---|
| picks / positions | per frame, per slot: the picked player's position in normalised court coordinates, (frames, 2 slots, 2), rows (Top, Bottom) | serve gate (distance observation), striker attribution, contact evidence |
| joints | per frame, per slot: keypoints normalised by the pick's bbox diagonal and centred on the bbox | contact evidence (wrist positions) |
| ankle series | per frame, per slot: the picked player's ankle point in image fractions, NaN where absent | serve stillness gate |
| height series | per frame, per slot: the picked player's bbox height in pixels, NaN where absent | serve stillness gate (body-height normalisation) |
| standing_in_court_count | per frame: the standing head count, defined below | serve gate lane routing (median over the setup window), doubles screen (future input) |
| failed | per SLOT, per frame: True where that slot could not be picked | every consumer treats a failed slot's fields as no-evidence |

The counted population for standing_in_court_count: candidates that pass the score
filter (score strictly above the filter), project to a finite court position, stand
inside the court within count_margin, and are not sitting (persistent seated
officials never count). The count is over that whole population, so it can exceed two
even though only two players are ever picked.

Failure semantics, per slot (a deliberate tightening of today's whole-frame bundle):

- a frame where the observable population is EMPTY (no detections, the score filter
  passes nobody, or every projection is NaN) has count 0 and both slots failed
- a frame with a population but a failed pick for one slot keeps the other slot's
  fields and the count; only the failed slot is no-evidence, and its moving average
  resets to the half-court prior
- a whole-frame pick failure fails both slots and resets both averages
- frames outside every tracker segment carry no-evidence values for every field; the
  result covers tracker-segment frames only

Today's contract, for contrast: the count exists only inside a success-only return
(every failure path discards it, and even rally_segmentation discards it on success), an unpicked
slot resets immediately, and a partially-picked frame is flagged failed as a whole.
Returning the count on every population-bearing frame is the one additive API change
sticky needs; every existing bst_x call path keeps its current signatures and
behaviour.

### Step 3: the dead-time mask

Purpose: a per-frame bool mask, True where the frame is dead for annotation (replays,
crowd cuts, slow-motion), consumed by final segmentation so dead frames cannot open
or feed a rally.

ONE public builder produces it. A config strategy field chooses the producer:

- replay signals: the union of three independent signals, each degrading to all-False
  with a log line when its input is missing (a missing court mask must not veto a
  real perspective-shift replay). The signals, equality rules pinned: court-absence
  fires across any court-absent run at least court_absent_window long; perspective
  shift fires on segments whose corner displacement from the dominant view is
  strictly above threshold; velocity-drop (slow motion) fires on visible frames
  whose rolling-median speed is strictly below the slow-mo fraction of the in-rally
  median speed while at or above rest_speed (this is the signal that needs the
  bootstrap spans). (today: replay_mask.combine_mask)
- composition vote: broadcast-cut boundaries segment the timeline; each cut-to-cut
  segment is live when the fraction of its frames whose homography vote says "court
  view" is at or above the vote threshold (equality is deliberately live), dead
  otherwise. (today: composition_mask.build_composition_mask)
- union of the two

Hard failure states, kept: a composition result where NO segment clears the vote
raises (an all-dead mask is a bad vote input, not an answer); applying an all-True
mask to a track raises (nothing live to anchor to). The target builder also validates
its input lengths (vote array against frame count, cuts within range); today's
composition builder trusts them silently, and that trust does not carry over.

Whether the producer that loses the eventual comparison gets deleted is decided by
the post-migration sweep campaign, recorded there; until then both stay available
behind the one builder.

### Step 4: final segmentation (rally spans and contacts)

The span-finding contract from step 1, now with the dead-time mask applied and the
serve gate on. Output: the rally spans (rally_id is list position) and the RAW
contact candidates.

Contact detection contract (today: the impulse path inside
annotator.rally_segmentation): within each span, on the masked track,

- each junction (the boundary between adjacent frames) gets an impulse: the change
  in smoothed shuttle velocity across it. A span too short to smooth yields no
  candidates
- a junction is a candidate when all three straddling frames are visible AND
  impulse > contact_multiple x local_floor, where local_floor is the median impulse
  over visible junctions within the floor window, bounded below by a small epsilon
  (so a near-zero floor cannot make everything a contact); a junction with no
  visible neighbour gets a NaN floor and can never pass
- candidates deduplicate PER SPAN, largest-impulse-first within the dedup radius (at
  least radius frames apart); equal impulses keep the earlier frame (a pinned tie
  rule; exact ties occur in real data)
- the wrist gate then scores each surviving candidate: the wrist-to-shuttle gap in
  body-height units (the picked player's bbox height), read from the sticky cached
  distance at the candidate frame. A finite gap at or below the wrist threshold
  passes; a NaN gap fails closed
- gate survivors then pass ONE suppression pass over the whole video (not per span):
  candidates are accepted best-first (descending impulse, then ascending frame), and
  a candidate within the suppression radius of an already-accepted one is dropped;
  candidates in adjacent spans can suppress each other

Every raw candidate is a row (rally_id, contact_frame, proximity_ok, wrist_near).
proximity_ok and wrist_near are per-candidate verdicts: True (passed), None (that
gate never ran), and for wrist_near False means the candidate failed the wrist gate
OR was removed by suppression (both are "not a scoring contact"; the row survives for
audit either way).

Filtered-vs-raw ownership, stated plainly: segmentation emits the RAW set with
verdicts attached and filters nothing. The annotator's orchestration applies the one
filter (drop rows whose wrist_near is False; keep True and the unmeasured None) and
every downstream step (attribution, timing, hit height, scoring) consumes the
FILTERED set. Nothing downstream re-filters.

### The serve gate (inside step 4)

Purpose: a span may only open on a genuine serve. The CLAIMED SERVE FRAME is the
first frame of the qualifying burst under test (today's code passes exactly that
frame to its gate). Evidence is read over the setup window: the window_frames frames
ending at the claimed serve frame, INCLUSIVE (the Laws require both players
stationary from ready position until the serve is struck, so the contact frame
belongs to the window; today's lookback stops strictly before the burst, and the
inclusive window is a ruled change). The window length is its own base-30 row,
independently tunable. "Fails closed" throughout means: missing or invalid evidence
makes the gate say no.

Routing: the gate splits into three lanes on the UNROUNDED median of
standing_in_court_count over the window (an even window's median can land on halves;
0.0 / 0.5 / 1.0 / 1.5 / 2.0 are each an explicit test case):

| Median count | Lane | Requirements |
|---|---|---|
| >= 2 | standard | distance gate; stillness on BOTH players |
| >= 1 and < 2 | partial-player | distance gate; stillness on the visible player; single-slot coherence |
| < 1 | fail closed | none pass; explicit TODO |

The pieces:

- the distance gate reads the shuttle's distance to the nearest of the two sticky
  picks: the median of finite distances over the window must be at or below
  threshold; an all-NaN window fails closed
- the partial-player lane's name is honest about mixed evidence (median 1.5 is not
  proof of exactly one visible player). Single-slot coherence: ONE slot supplies the
  distance observation, the ankle and height rows, and the presence fraction for the
  whole window; evidence that switches slots fails closed
- the fail-closed lane exists because no detector exists yet for windows with no
  standing in-court evidence; it gets built post-migration, sized by the resweep.
  Note: the count is position-based, so a tight close-up whose bbox bottom projects
  into the court margin CAN read as 1 and route to the partial lane; the old
  "close-ups read zero" claim belonged to the size-banded counter and retires with
  it.

Stillness is series_drift on the ankle series: take the window's detected rows in
time order, split them in half (an odd count gives the extra row to the first half),
find the median x and median y of each half, and read the distance between the two
half-medians in body-height units (the mean of the height series over detected
rows). Fewer than two detected rows yields no drift (NaN), which fails closed. Per
required player: the drift ratio must be finite and at or below threshold, and that
player's detected fraction of the window must reach PLAYER_PRESENT_MIN_FRAC. An
unavailable required player fails closed (a deliberate change: today's gate can pass
on one player when the other's drift is NaN).

Undetected rows for series_drift: a row is undetected when it is non-finite OR both
coordinates are exactly zero (the paired-zero missing sentinel sticky, BST-X, and
TrackNet share). (0, y) and (x, 0) stay detected. The helper accepts sentinel-coded
point series, not arbitrary geometry where the origin is meaningful.

The historical court-scale distance and wide-shot gate was removed in W2.4. The
sticky-sourced three-lane gate above is the current serve gate.

### Step 5: striker attribution

Per filtered contact: attribute the stroke to Top or Bottom by which sticky-picked
player is nearer the shuttle at the contact frame. A contact yields no guess
(None) when the shuttle is invisible at that frame, no candidate exists, or the
nearer player's foot sits in the net band (ambiguous).

Per rally: strokes must alternate halves, so the per-contact guesses are tested
against the two possible alternating assignments and the better-scoring one gives
the rally's final-stroke half. When the two assignments score equal, the answer is
None: the rally then has no striker, no verdict row, and scores as an automatic miss
downstream. That is the contract: a tie is not guessed.

### Step 6: server and next-server

The rally's server is the fitted first-stroke half, derived from the final-stroke
half and the stroke count by parity. In badminton the point's winner serves the next
rally, so each rally gets a next-server prediction from the FOLLOWING rally's fitted
first stroke (if rally 4 opens with Bottom serving, rally 3's predicted winner is
Bottom); the last rally has none (None). Both inherit None wherever the alternation
fit tied.

### Step 7: landing

Per rally with a resolved striker: from the final contact, three per-frame series
over the following window feed the landing pick: a carry ratio (the shuttle's
distance to the nearest wrist, in body heights; LOW means hand-held), an ankle
ratio (shuttle nearer an ankle than any wrist reads as grounded), and the shuttle's
speed. Frames with no evidence are NaN. The pick finds descending runs of the
shuttle and takes each run's terminal frame; runs that read as carried (the player
holding the shuttle, not it landing) are dropped. When every terminal is carried,
the pick falls back to the LAST terminal anyway (the shipped keep-last-drop default;
nulling instead is an option, off by default). The pick is None only when the window
holds too few visible samples or no descending run at all. A landing carries: frame,
court-normalised position, court half, and three flags (at_border, masked,
net_ender).

### Step 8: winner

Per rally with a striker: the next-server evidence makes the call whenever it
exists. Verdict WON means the striker's half took the point; LOST means the other
half did; the winner call on a next-server row never consults the landing (the row's
margin diagnostics still read it). Only when no next serve is
attributable does the landing geometry make a best-guess call from the landing's
court-half membership (every landing position sits in one half, so a landing always
yields a call even where a confident verdict would abstain; the verdict is None only
when there is no landing at all, and that rally scores as a winner miss). The
homography's corner-error band annotates diagnostic margin flags on every row
(landed within the error band of a line or the net); it never changes the verdict.
The winner column is the verdict mapped to a half.

### Step 9: hit height

Per filtered contact: 1 when the shuttle at the contact frame is above the net-band
centre, 2 at or below (ShuttleSet's coding). An invisible shuttle at the contact
frame RAISES (a height needs a detected position, not a guess); the orchestration
catches that one error, records a failure row (rally_id, stroke index, frame,
reason), and continues. Failure rows are part of the output, not a crash and not
silence.

### Step 10: doubles screen

Per rally span: doubles when the doubles evidence holds on more than the configured
fraction of the span's frames. Empty evidence (an empty span, or a span with no
overlap) reads as not-doubles. Today the evidence is the vision lane's overcount
array; the sticky standing_in_court_count can only replace it after it proves out on
the fixtures (the count reads MORE-than-two; sticky's picked output is always
exactly two players and can never evidence doubles by itself). Until then the input
contract stays as is.

## Injection points

For testing and iteration, run_video accepts known quantities as explicit arguments.
None means compute; an injected value REPLACES its computed step entirely, never
overlays it. This replaces the old monkeypatch machinery.

| Injected input | Shape | What it bypasses | What still runs |
|---|---|---|---|
| rally spans | list of (start, end), half-open | BOTH segmentation passes' span-finding (the injected spans are the bootstrap spans and the final spans) | sticky (over the injected spans), the mask build, contact detection within the spans, everything downstream |
| contact frames | per rally_id, ascending frames | contact detection AND the wrist filter | attribution, server, landing, winner, hit height |
| dead-time mask | (frames,) bool, True = dead | the mask BUILD (no strategy is consulted) | final segmentation consumes the injected mask; the preliminary pass stays unmasked by design |

Injected contacts enter as the already-filtered set: the result bundle's raw and
filtered contact lists both hold exactly the injected rows, with proximity_ok and
wrist_near None (unmeasured). Injections compose (for example spans plus contacts);
each bypasses only its own step, and an injected downstream value never resurrects a
bypassed upstream computation.

## Outputs

The target result bundle, one per video:

- rally spans, ordered; rally_id is position
- raw contacts and filtered contacts (both kept; the filter is auditable)
- per rally: stroke count, striker half, server half, next-server half, the verdict
  row (verdict, source, margin diagnostics), the winner half derived from verdict
  plus striker, and the landing (or None)
- per filtered contact: hit height; plus the hit-height failure rows
- per rally: the doubles flag

Today's nearest shape is smaller (`scripts/archive/h_end_to_end.py`'s DetectedChain): it carries the
spans, contacts, per-rally halves, verdict rows, landings, and hit heights, while
the winner half is derived downstream by the scorer and the doubles flag lives in a
separate CLI. The target bundle folds both in.

The initial migration GT scoring harness reads this bundle and scores every
column against ShuttleSet labels. Its two primary metrics, boundary covered_fraction
and contact F1 at the standard matching tolerance (+/-5 base-30 frames, scaled per
fixture fps), carry the collapse floors: the harness FAILS a fixture when either
metric drops below its floor (half the recorded current value), and a None or
non-finite score fails outright. Everything else is display-only there.

## Failure behaviour, the rules in one place

- Gates fail closed: missing or all-NaN gate evidence means the gate says no. This
  holds for the serve distance gate, stillness, the lane coherence rule, and
  wrist_near on a NaN gap.
- Mask signals degrade open: a missing input to a union signal yields all-False plus
  a log line, never a veto and never a crash.
- Boundary contract violations raise: mismatched array lengths, an all-True mask at
  apply time, an all-dead composition vote, and a non-finite body-scale denominator
  are bugs or bad inputs, not conditions to absorb. (Today's composition builder
  skips the length checks; the target builder does not inherit that.) Option
  combinations the target API cannot express are rejected where the config types are
  built, replacing today's scattered runtime incompatibility raises.
- Per-item failures in long runs log and continue: a hit-height failure is a
  recorded failure row; a missing per-video file in a batch run is a logged skip.
  Neither crashes the batch.
- Absent evidence downstream is None, threaded honestly: a tied fit, a missing
  landing, an unresolvable verdict each carry None through to scoring as a miss,
  never a fabricated answer.

## Out of scope for this map

- bst_x model, pipeline, and training behaviour (sticky's API changes are additive
  only)
- acquisition: search, download, filtering, pose/track extraction, court fitting,
  commentary transcription (src/scraper)
- commentary cleaning and the rally-to-commentary join (today: commentary_cleaning,
  commentary_pairing): downstream consumers of the annotator's spans, not part of
  run_video
- the vision-lane overcount producer on HPC
- GT files, fixtures, and scoring internals (the harness owns them)
