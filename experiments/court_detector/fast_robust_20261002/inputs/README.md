# Saved inputs for the court comparisons

These files let maintainers reproduce the published comparisons without rerunning
court detection. The [reproduction guide](../reproduce.md) gives the commands.

- `cohort.json.gz` and `videos/`: the original 86-video run, before court-sharing fixes.
- `player_tiebreak_results/`: the eight-video trials, split by their original worker
  names and stages. Each stage retains both methods' predictions, recorded choices,
  detector commit, completion status and elapsed time.
- `choices/*.jsonl.gz` within each stage: all recorded decisions and timings, plus
  `candidate_count` and the details of the selected candidates under
  `chosen_candidates`. The full candidate tables are distributed separately.

Stage A used `928ed398` and stage B used `cd776fb2`. The existing comparison
script checks the recorded commits and completed attempts before scoring results.
Source-video entries contain filenames; resolve them against your own video storage.
Official annotations live in `data/shuttleset/set/` and `data/shuttleset22/set/`.

## Optional bulk files

These locations are Git-ignored and distributed separately:

- `candidate_pools/<worker>/<stage>/*.jsonl.gz`: the original full scored candidate
  tables, useful for trying other selection rules without repeating detection.
- `../player_tiebreak_trial/rally_review/scene_clips/`: two complete failure-example
  scenes, useful for examining camera transitions around the sampled frames.

Neither is needed to reproduce the numerical comparisons. The candidate files are
unchanged copies of the trial records; compact committed choice records preserve
both methods' decisions. Source videos and model inputs are also separate data.
