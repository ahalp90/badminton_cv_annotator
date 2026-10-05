# Evaluation methods, evidence and commands

This file records how the closeout was produced: which saved inputs were treated as canonical, how the published counts were recomputed, and which checks depend on local footage or run artefacts. Commands are shown from the repository root unless noted otherwise.

**Contents**

[Investigation sequence](#investigation-sequence)  
[Canonical saved inputs](#canonical-saved-inputs)  
[Environment](#environment)  
[Result inventory](#result-inventory)  
[Pull the saved run](#pull-the-saved-run)  
[Rerun player state](#rerun-player-state)  
[Recount and context](#recount-and-context)  
[Summaries and court comparison](#summaries-and-court-comparison)  
[Figures and per-video viewer](#figures-and-per-video-viewer)  
[Footage checks](#footage-checks)  
[Detector audit checks](#detector-audit-checks)  
[Shuttle guard checks](#shuttle-guard-checks)  
[Validation already completed](#validation-already-completed)

## Investigation sequence

| Question | How it was checked |
|---|---|
| Which parts of the annotation are wrong? | Rescored the saved new-court output; counted missed and extra hits, wrong players and incomplete rallies; compared clips kept and discarded by selection. |
| What did better courts change? | Paired each labelled hit and rally with the old and new court decisions, then counted gains and losses against both old models. |
| Were court and player inputs available when contacts were missed? | Joined each labelled frame to the new scene records and to rerun player picks. |
| Are the earlier court failures fixed? | Redrew the two regression frames with the new outlines and counted matches in those scenes. |
| Do the labels agree with the footage? | Drew 24 missed hits at random and judged each from stills around the labelled frame. |
| Are bulk failures label errors? | Searched rallies for time-shifted labels, swapped sides and extra final hits; judged 28 cases blind from frame-exact stills. |
| What in the code causes the largest failures? | A read-only audit of the current code traced each failure to file and line; every finding was recounted from the saved run and its untested predictions scored on held-back data. |
| Does the shuttle guard mark the false final hits? | Read the saved guard code at every emitted hit, compared false final hits with real ones, and simulated dropping a flagged final hit. |
| How does original ShuttleSet compare? | Rescored the saved validation and held-out development outputs the same way. |

The selected model stayed fixed. No model was refitted for this closeout.

## Canonical saved inputs

| Input | Location |
|---|---|
| Selected run | Refit run directory `annotator-good-court-refit-20261003` (`$RUN` below) |
| Selected model bundle | `output/base/bundle` in that run; `data/annotator/sset_and_sset22_trained_20261003T041112Z` in the repository |
| Annotator code | `code/` in that run, commit `97b5e4de` |
| Court release | `court_sharing_patched`, commit `e0151791` |
| Labels | `output/shared/labels/{test,development}/*.csv`, cleaned |
| Old-court contexts and outputs | `scratch/annotator_wrapup_evaluation/results/` |
| Paired gained/lost rally IDs | `experiments/annotator/good_court_refit/evidence/saved-stream-analysis.json.gz` |

## Environment

The commands assume these locations:

```bash
export RUN=/path/to/annotator-good-court-refit-20261003
export WORK=/path/to/closeout/work/directory
export SOURCES=/path/to/shuttleset22/source/videos
export PREPARED=/path/to/shuttleset22/prepared/fixtures
export RUN_PYTHON=/path/to/the/run/environment/bin/python
```

`$WORK` sits beside the run. `$RUN_PYTHON` is the run's own environment: Python 3.12.13 and scikit-learn 1.9.1. Loading the model bundle needs that scikit-learn version. Steps that use `$RUN_PYTHON` ran beside the saved run; their outputs were then copied back into `raw/`.

Local scoring uses the project dependencies with `PYTHONPATH=.:src`. The figure and footage scripts also need pandas, NumPy, matplotlib and OpenCV; they were run with pandas 3.0 and matplotlib 3.11.

## Result inventory

| File | Contents |
|---|---|
| `results/proposals.csv.gz` | Every proposed clip with its judgement, errors and selection; test, validation and held-out development; ±10 and ±5 |
| `results/contacts.csv.gz` | Every labelled hit with match, player, offset and rally position |
| `results/predictions.csv.gz` | Every emitted hit with its match status |
| `results/rallies.csv.gz` | Every labelled rally with coverage and correctness |
| `results/contexts.csv.gz` | Court and player state at each ShuttleSet22 labelled frame |
| `results/scenes.csv.gz` | New scene records: status, reason and two-player vote |
| `results/summary.json.gz` | Headline, coverage, errors, inputs, timing, players, development and court comparison |
| `results/per_video.csv.gz` | Per-video counts |
| `results/court_change_groups.json.gz` | Labels and rallies grouped by old and new court decision |
| `results/miss_sample.csv.gz` | The 24 sampled missed hits |
| `results/miss_judgements.csv.gz` | Footage judgements for the sample |
| `results/label_check_cases.csv.gz` | The 28 failure-group cases |
| `results/label_check_judgements.csv.gz` | Footage judgements and label verdicts for those cases |
| `results/label_trace.csv.gz` | Official ShuttleSet22 rows behind every checked case |
| `results/tail_rallies.csv.gz` | The 262 rallies with an extra hit after the last label: extra counts, sole-error flag and official ending |
| `results/shuttle_guard_runs.json.gz` | Frame runs of each shuttle guard code, inpainted frames and hidden frames for the 47 ShuttleSet22 videos |

CSV reads need `dtype={"video": str}` because development IDs such as `sset_31` share the column with ShuttleSet22 numbers.

`raw/` is local and Git-ignored. It holds the pulled run, the player-state rerun and the sampled frames. The individual JSON files from individual VLM runs used while producing the two footage summaries were working files rather than retained evidence; the merged judgements are preserved in `results/miss_judgements.csv.gz` and `results/label_check_judgements.csv.gz`. The two summary scripts still expect those local JSON inputs, so those exact merge steps are not clean-checkout reruns.

## Pull the saved run

The closeout copied these run subfolders into `experiments/annotator/annotator_closeout_new_courts/raw/run/`, keeping their paths:

```text
output/base/eval
output/base/dev_heldout
output/shared/labels
output/shared/court
output/comparisons
courts
```

## Rerun player state

The saved streams do not keep per-frame player picks. The rerun rebuilds them with the selected bundle and checks that each rebuilt stream is identical to the saved one.

The player-state rerun was executed beside the saved run, then `$WORK/player_state/` was copied to `raw/player_state/`:

```bash
cd "$RUN"
PYTHONPATH=code:code/src "$RUN_PYTHON" "$WORK/recompute_player_state.py" \
  --run . --output "$WORK/player_state" --jobs 7
```

All 46 rebuilt streams matched the saved streams.

## Recount and context

```bash
PYTHONPATH=.:src python -m scratch.annotator_closeout_new_courts.scripts.evaluate_saved
python experiments/annotator/annotator_closeout_new_courts/scripts/collect_context.py
```

The recount takes about three minutes. It uses the library's contact matching and replicates its whole-rally rule at both tolerances. Correct sides use the raw stream event sides, as the run's own comparison does.

The context step joins each labelled ShuttleSet22 frame to its scene record and to the rerun player picks. A frame counts as court accepted when its scene passed both the court search and the two-player vote.

## Summaries and court comparison

```bash
PYTHONPATH=.:src python -m scratch.annotator_closeout_new_courts.scripts.summarise
python experiments/annotator/annotator_closeout_new_courts/scripts/summarise_court_change.py
```

The aggregate court baseline is the **fresh old-court refit**: current annotator fitting code and scikit-learn 1.9.1, but old court detector outputs. The refit using the new custom court detector was fitted in the same current environment. For the label-by-label court-decision grouping, old-court states come from the earlier historical evaluation's retained contexts; the fresh old-court refit used the same old court inputs, so those saved accept/reject states can be used for both old-output comparisons.

## Figures and per-video viewer

```bash
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_evaluation.py
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_readme_summary.py
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_closeout_story.py
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_readme_details.py
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_video_variation.py
python experiments/annotator/annotator_closeout_new_courts/scripts/plot_court_cases.py
python experiments/annotator/annotator_closeout_new_courts/scripts/build_video_view.py
```

`plot_readme_summary.py` draws contact metrics, selected clips and historical development. It checks the current counts against the saved tables when all required tables are present; otherwise it uses the published snapshot. The historical figure uses the earlier 3,422-rally stage totals. Those intermediate systems were not rerun with the new courts.

`plot_closeout_story.py` draws the combined court-decision comparison, missed contacts for the two court-input refits, rally gains/losses by court decision, and shuttle-guard rates. `plot_readme_details.py` draws selected-clip errors, cumulative timing accuracy, miss rates in court-accepted frames and input availability. Both use published counts embedded in the scripts.

`plot_video_variation.py` reads the saved `results/per_video.csv.gz` table and draws the video scatter plot and two contact-outcome panels. In a nested draft, pass `--results` with the existing results directory.

Run `plot_evaluation.py` before these four scripts because it also writes some of their filenames. The commands above use the final experiment location. To regenerate a nested draft, run its copies of the four plotting scripts; output paths follow the scripts' location. The [figure catalogue](figures/README.md) lists their outputs.

The court-case figures reuse the earlier evaluation's saved frames and old outlines. The new outlines come from the run's scene records.

## Footage checks

### Random miss sample

The sample and stills were produced with:

```bash
python experiments/annotator/annotator_closeout_new_courts/scripts/sample_misses.py
# beside the run, with extract_views.py and miss_sample.csv.gz copied to $WORK:
"$RUN_PYTHON" "$WORK/extract_views.py" --sample "$WORK/miss_sample.csv.gz" \
  --sources "$SOURCES" --output "$WORK/miss_frames"
# $WORK/miss_frames/ was then copied to raw/miss_frames/
```

The sample uses seed `20261004` and draws 24 of the 2,984 missed hits uniformly. Each case has a context sheet of nine stills at half-second steps over ±2 seconds, a burst of nine stills at three-frame steps around the label, and a full-resolution centre frame.

Three runs of a large vision-language model (VLM) judged eight cases each. This model interprets images and text. They saw the three images and the labelled side, and nothing about the model's output. Each recorded the view, whether all court corners were visible, how many players were visible, an estimated hit frame and a short note. Each case got one of four judgements: hit and player agree, timing label wrong, clear footage disagreement, or unclear.

The earlier OpenCV stills were checked against frame-exact decoding: 21 of 24 centre stills show the labelled frame and three show a neighbouring frame. A one-frame slip does not affect a ±10-frame judgement.

The merged output is retained as `results/miss_judgements.csv.gz`. Historically it was produced with:

```bash
python experiments/annotator/annotator_closeout_new_courts/scripts/summarise_judgements.py
```

That script expects the original local JSON files from individual VLM runs, which are not part of the retained evidence pack.

### Failure-group checks

The cases and stills were produced with:

```bash
python experiments/annotator/annotator_closeout_new_courts/scripts/sample_label_checks.py
# beside the run, with render_label_checks.py, label_check_cases.csv.gz and the miss-sample files in $WORK:
PYTHONPATH="$RUN/code/src" "$RUN_PYTHON" "$WORK/render_label_checks.py" \
  --cases "$WORK/label_check_cases.csv.gz" --sources "$SOURCES" --extracted "$PREPARED" \
  --output "$WORK/label_frames" --saved-sample "$WORK/miss_sample.csv.gz" --saved-stills "$WORK/miss_frames"
# $WORK/label_frames/ was then copied to raw/label_frames/
python experiments/annotator/annotator_closeout_new_courts/scripts/trace_label_rows.py
python experiments/annotator/annotator_closeout_new_courts/scripts/count_tail_endings.py
```

The sample uses seed `20261004` and has three groups:

- 16 of the 262 failed rallies whose labelled hits all match with the right player and whose first unmatched prediction comes 10–60 frames after the last label, at that prediction's frame;
- one wrongly assigned hit from each of the ten swapped-side rallies outside video 12;
- one missed hit from each of the two video 12 shifted rallies not already in the random sample.

The cases are shuffled and given neutral IDs.

The renderer takes frames from the validation overlay's span decoder, which selects them by timestamp. It draws the base shuttle track as a magenta box and captions each still with its offset from the query frame. Each case has the same three images as the miss sample. The `--saved-sample` option also runs the seek check on the earlier stills.

Three runs of a large vision-language model judged nine or ten cases each. They saw the query frame and its stills, but not whether that frame came from a label or a prediction, nor the label's side. Each recorded whether a racket met the shuttle within about ±10 frames, which player, the estimated frame, a confidence level and what happened to the shuttle next. `summarise_label_checks.py` turns those answers into label verdicts by fixed rules.

The example note in the brief happened to describe case L08's query frame as a contact into the net. The instance judging L08 noticed, still judged from the pixels and called no contact at low confidence. Removing L08 leaves the extra-hit result at 15 of 15.

`trace_label_rows.py` checks that cleaning kept every official row unchanged. It also records the official rows behind each case: the labelled hit, or the rally's final row and its recorded ending.

`count_tail_endings.py` rebuilds the same 262-rally pool and marks the 233 that fail on the extra hit alone: one clip covers the rally and holds exactly one extra. It joins each rally's official ending and prints the ending table.

The merged label-check output is retained as `results/label_check_judgements.csv.gz`. Historically it was produced with:

```bash
python experiments/annotator/annotator_closeout_new_courts/scripts/summarise_label_checks.py
```

As above, that merge script expects the original local JSON files from individual VLM runs.

## Detector audit checks

```bash
PYTHONPATH=src python experiments/annotator/annotator_closeout_new_courts/scripts/audit_checks.py
```

The script recomputes every count in the [detector audit](detector_audit.md). It reads the detector stages each rejected scene ran from the court release, compares each scene's 3-second foot window with the labelled rallies, measures vote-failed courts against the frame and the video's accepted courts, and follows each false final hit through the saved sequences and landings. It needs `raw/run/` and the official ShuttleSet22 rows.

### Corrections made during verification

The rechecks changed four statements from the initial audit:

- It put all 21 vote-failed scenes with ordinary courts at 42–50%. They range from 5% to 50%, with a median of 42%.
- Its oversized-court count, 339, included seven scenes flagged only for a small area. Four are one recurring wide view in video 38. The detector audit's table uses the stricter count, 286.
- It said nothing downstream can remove a trailing hit. The sequence chooser can; it chose not to in all 233.
- It predicted the false hit would cost the point winner. It does not: 80.1% against 74.2%.

## Shuttle guard checks

`scripts/pull_shuttle_guard_runs.py` was run on the machine holding the inpainted ShuttleSet22 extract, with the extract root as its only argument. It reads the inpainted tracks, guard codes and fill sidecars that the evaluated run read, and writes `results/shuttle_guard_runs.json.gz` to standard output.

```bash
python experiments/annotator/annotator_closeout_new_courts/scripts/count_guarded_final_hits.py
```

The script recomputes every count in the [shuttle guard analysis](shuttle_guard_addendum.md). It looks up the guard code at each emitted hit and compares false final hits with real ones. It then counts what dropping a flagged final hit would rescue and break.

## Validation already completed

- The recount reproduces all fifteen published ±10 totals for the selected model, including 1,744 complete rallies, 34,200 matches, 33,156 correct sides and the 591 / 117 / 17 queue.
- The replicated whole-rally rule agrees with the library's rule on every proposal at ±10.
- All 46 rebuilt player-state streams match the saved streams.
- Every per-video record in the viewer sums back to the headline contact count.
- Both rally-group comparisons cover all 3,327 rallies, and the label groups sum to 33,551 historical and 34,200 new matches.
- All 37,184 cleaned test labels are official ShuttleSet22 rows with the same frame and side.
- The pulled guard runs reproduce the [shuttle guard measurement](../../../scratch/shuttle_hallucinations/README.md)'s guard and inpainting counts for all 3,761 ShuttleSet22 rallies.
- Ruff passes on the closeout scripts.
