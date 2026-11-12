# Historical joint annotator settings search

This plan describes the earlier three-model search. The maintained `hp_search`
command now ends after selector screening and validation repeats; it does not
export a production bundle. Use the [current training commands](README.md#reproduce-the-current-fit)
to fit the selected four-model method. Historical test comparisons require their
original checkout.

The original plan ran a 192-combination search on one 16 GB V100, targeting 8–14 hours including
five-seed checks of the finalists. The total run has a 24-hour limit. Use the
existing saved vision inputs and experiment functions, with persistent workers
and a bounded circular queue.

## Grid

Cross every value in this table. The current settings are included.

| Setting | Values |
|---|---|
| Body-height half-window, at 30 fps | 12, 36 |
| Contact maximum leaves | 31, 63 |
| Contact maximum boosting rounds | 180, 360 |
| Contact score cutoff | 0.75, 0.80, 0.85, 0.90 |
| Rally-selector maximum leaves | 15, 31, 63 |
| Rally-selector boosting rounds | 200, 400 |

This gives **2 × 2 × 2 × 4 × 3 × 2 = 192 combinations**. Keep the other
settings fixed: contact/selector learning rates 0.06/0.05, minimum child weight
5, L2 regularisation 1, selector span weight 372, three allowed contact edits,
six deletion candidates and the current training targets. Keep the existing
contact early-stopping rule. Fit the review model only for shortlisted runs.

The two body windows retain the earlier XGBoost comparison's F1 and whole-rally
leaders. Every existing held-group contact fit inspected reached its 180-round
cap, so 360 is a useful comparison. The selector range brackets the selected
31-leaf model and repeats its earlier 200/400-round comparison under different
contact inputs. Crossing all four cutoffs tests whether recovering more initial
contacts helps the selector produce complete rallies.

Keep end-rest at 90 and the other preprocessing settings at their current
defaults. The old 54-frame result came from the retired architecture; it does
not justify doubling this search. There are no automatic refinement rounds.

## Why another search

The previous XGBoost input sweep changed one setting at a time and held tree
settings and the 0.9 contact cutoff fixed. The older 48-combination Cartesian
search used the retired eight-model architecture. Neither tested this grid.

Since those searches, training has changed to direct proposal judgement and
scene-based player geometry, with wrist-distance tie-breaking for alternating
players. Several changes already have fitted baselines. This search tests their
joint settings and the contact cutoff; renames and readability changes alone
are not a reason to retrain.

See the [previous XGBoost comparison](../reports/tree_simplification.md),
[earlier Cartesian search](../tree_hp_tune/RESULTS.md) and
[current three-model results](results/three_models/README.md).

## Selection

1. Prepare current-code evidence and fit the current settings first. Compare
   the runner with the existing staged route on representative saved inputs.
   Time the lowest cutoff and largest proposed trees before sizing the full run.
2. Screen the grid at seed offset 0 on the fixed 32 training videos. Hold out
   each of the four groups for contact and selector fitting, following the
   earlier search: each fit leaves out eight videos, then scores those eight.
   Evaluate finished contact streams, including player
   alternation, boundaries and contacts outside discovered rallies.
3. Rank complete configurations by fully correct rallies at ±10 frames, then
   correct-player F1 at ±10, then fully correct rallies at ±5. Keep a second
   ranking with correct-player F1 first, whole rallies second and correct-player
   F1 at ±5 third. Prefer baseline on an exact tie, then fewer changed settings
   and a fixed configuration order. Pool counts across videos before computing
   rates. Also record contact precision and recall before and after refinement.
4. Freeze baseline plus the top two distinct configurations from each ranking:
   at most five configurations. Fit offsets 0, 100, 200, 300 and 400 and compare
   them on the eight validation videos. Reuse completed seed-0 work. Select by
   mean whole-rally correctness; retain the mean-F1 leader as a second result
   if it differs. Report paired seed differences and per-video results.
5. Freeze the validation choices and compare them with baseline on the familiar
   46-video test set. Keep contact training at 32 videos for both sides of this
   comparison. A later 40-video deployment fit is a separate step. Test results
   do not choose settings or seeds, and this run does not replace the default
   model automatically.

The broad screen uses the earlier stagewise held-group method: there is an
indirect dependency through the contact predictions used to train selectors.
Use those scores for screening, with validation reserved for final selection.
Five seeds measure variation in fitting on the same videos.

## Reuse and parallel work

The coarse screen needs only **two preparations, eight contact configurations
and 32 proposal families**. Each proposal family supplies all six selector
settings. There are 32 held-group contact fits and 768 held-group selector fits;
full-32 fits, review fitting and validation replay are deferred to the shortlist.

Use one persistent GPU process with two fitting threads, plus six persistent
CPU workers, each CPU worker using one numerical thread. Two concurrent fits
took 37.3 seconds on a measured training fold, versus 61.0 seconds for two
sequential fits. CPU workers prepare proposals and score completed fits while
the GPU fits the next models. Set models to CPU explicitly when passing them
to CPU workers. Include the largest proposal family in the initial memory and
timing check.

Use a **three-slot circular queue** for proposal families. Preparation claims a
free slot, the GPU worker consumes ready families, and the slot is released
after its selector fits and scoring finish. Preparation waits when all slots
are occupied. The coordinator reserves a slot before submitting preparation,
so CPU workers remain available for scoring. Queue small job records and paths.
Retain the active family's loaded arrays in the GPU worker and keep scoring
submissions bounded too.

- Prepare the two evidence sets once, using the existing preparation command.
  Reuse the saved vision outputs; no court, pose or shuttle-model inference.
- Load each preparation once per contact fitting batch. Preserve the existing
  sampling and fit-row orders when calling `fit_contact_model`.
- Score each video's contact candidates once per contact model and seed. Apply
  `remove_nearby_contacts` to those saved probabilities at each cutoff, using
  the existing FPS-scaled distance and float64 scores. Rebuild the rally context
  and proposals for each cutoff.
- Build and label each proposal family once. Keep its arrays loaded while
  trying all six selector settings. Reuse the held-group fits later for review
  training. Different seeds rebuild their contact-dependent proposals.
- Save settings, completed results, shortlisted models and one summary table.
  Resume from completed work using the existing file-based approach.

## Implementation

Check a run from the repository root:

```bash
python -m experiments.annotator.tree_simplification.hp_search_status --output RUN
```

The command shows the current stage, completed combinations and fits, elapsed
time and any failure. A detached launcher can write its PID to `RUN/process.pid`
and its exit status to `RUN/exit-code`; the status command uses these to identify
an exited process even if its last saved progress still says it was running.

Add a small runner under `experiments/annotator/tree_simplification/`, with a
worker helper only if that makes the runner easier to read. Production `src/`
changes are not needed for the current design.

Reuse `prepare.prepare_video`, `fit_contact_model`, `score_contact_features`,
`remove_nearby_contacts`, `build_rally_context`, the existing proposal and target
functions, `SelectorTrainer`, and the full-stream scorer. Follow the persistent
worker pattern in `annotator_batch_runs.py`. The historical search settings are
independent of the normal training command's defaults.

The broad screen must call the fitting/scoring functions directly:
`train.main` evaluates validation immediately. The current multi-seed driver
also launches GPU jobs according to CPU count; use the explicit GPU worker
limit above instead.

Check the runner's default outputs against the staged route, the four-cutoff
reuse against independent scoring, the held-group assignments, and queue
wraparound with more jobs than slots. Run relevant lint, type checks and tests
for the implementation. Keep checks focused on these changed behaviours.

## Time budget

Earlier V100 measurements gave 14–20 seconds per contact fit. A fresh timing
check on an existing 24-video training fold took 30.5 seconds for one 31-leaf,
400-round selector and 37.3 seconds for two concurrent fits. A 63-leaf fit took
30.9 seconds. Loading the pools took 36.3 seconds, which the runner avoids
repeating for every setting. The timing check used the pinned XGBoost 3.4.1
V100 build.

On two training videos, reducing the cutoff from 0.90 to 0.75 increased proposal
counts by 5–8%; construction stayed around 10–12 seconds per video and cutoff.
Those timings used current proposal code and existing fitted contact models.

| Work | Budget |
|---|---:|
| Preparation, baseline and timing checks | 1 hour |
| 192-combination screen, with overlapping CPU work | 4–6 hours |
| Five-seed shortlist and test comparison | 3–7 hours |
| Total | **8–14 hours; 24-hour limit** |

These estimates allow for larger trees and CPU scoring. Confirm them using the
current-code timing run before submitting the grid. Reserve seven hours for
the shortlist and test. If the coarse screen will not fit the remaining budget,
fix selector rounds at 400 before launch: that gives a complete **96-combination
grid** retaining every preprocessing, contact and cutoff choice. Do not choose
the reduction from partially completed scores.

At the deadline, preserve completed work and report any unfinished comparisons.
The time budget excludes implementing and testing the runner.
