# Test player support as a tie-breaker

The completed [selection comparison](selection_results/README.md) favours the
sharing fix as the leading repair. Score-first selection changes none of the
eight representative courts after sharing. The search without player rejection
is still running. The comparison includes 20 PNGs with explicit source names.

Run the same eight videos twice. Stage A changes final selection and refit
acceptance. Stage B also removes player-based rejection during search. Both
stages include the sharing fix. Each stage runs from its own committed checkout
and writes to its own directory. Original extracts remain untouched.

The trial picks the highest combined court score. On an exact tie, it prefers
candidates that pass the current player rule, then candidates containing a
person in at least half the samples, then the rest. Further ties keep the
existing ranking order. There is no score bonus or tolerance. Final refitting
retains geometry and camera checks without requiring player presence.

Stage A keeps existing search pruning. Stage B retains geometric checks and
search limits while admitting courts without player support. Full-court player
measurements can break ties after axes are combined. Early axis and template
ranking stays score-only where those measurements are not yet available.

## Videos and outputs

Carmack: ShuttleSet 30 and 36 (poor final courts), 11 and 21 (low-error final courts).
Bourbaki: ShuttleSet22 43 and 44 (poor), 27 and 51 (low-error). This includes some
individual-scene errors inside otherwise good videos. Each host runs four
videos concurrently, with three workers per video. The original runs took
1.3–3 hours per video; the broader search may take longer.

`trial_run_video.py` runs two choices from each scoring record. The checkout's
normal choice is saved in `videos/`; score-first selection is in `trial_videos/`.
Both use separate view pools. Candidate scores, player measurements and corners
are saved in `choices/`. A failed scene, group or missing arm fails the video.

**Use stage A's `videos/` as the fixed-sharing baseline.** Stage B's `videos/`
uses the changed search but the original final player veto, so it is an
auxiliary result rather than the original control. The three main comparisons
are A/videos, A/trial_videos and B/trial_videos.

The selection change is isolated in the experiment entry point. Production
selection defaults are unchanged until the trial is assessed. Stage B's search
changes are isolated by the separate Git commit and checkout. If adopted,
move the chosen selection rule into the detector and remove this temporary
entry-point substitution.

## Run and check progress

Reuse `run_courts.py` and `launch_trial.sh` from the original release extraction.
The runner records the checkout commit and refuses uncommitted tracked changes.
For each host, prepare `stage_a/` and `stage_b/` with these scripts, the host's
cohort as `cohort.json.gz`, and a detached `checkout/` at the appropriate commit.
Run each stage's `launch_trial.sh check` before starting.

```bash
# On each host, after preparation:
run_root=/scratch/ahalperi/court_det_fix/player_tiebreak_20261002
setsid nohup "$run_root/run_stages.sh" "$run_root" run \
  > "$run_root/chain.log" 2>&1 < /dev/null &
"$run_root/run_stages.sh" "$run_root" status
```

The chain waits for all four stage-A lanes before starting stage B. Any failed
lane stops the chain. `chain.exit` records completion or failure. The status
command shows the active stage, per-video progress and output counts. Resume
uses the same command and skips complete videos. Timings cover paired work;
they do not estimate the standalone cost of either choice rule.

Compare individual courts before sharing and final courts afterwards. Split
by rally overlap and main/other view. The static dataset homography only labels
the default view; alternate-view disagreement is not a measured error. Count
changed group assignments and new false-court risks outside rallies. Keep the
original cohort exclusions. Eight selected videos are a screening trial, not a
claim about accuracy across the entire corpus.
