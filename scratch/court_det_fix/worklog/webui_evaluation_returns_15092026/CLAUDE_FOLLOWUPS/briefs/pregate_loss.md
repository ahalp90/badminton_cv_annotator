# Brief: record proposals before the player gate

`R` is the remote experiment root.

This runs on the compute host. Read `~/.codex/remote_hpc.md` first and follow it: use the remote shell wrapper on the compute host and the remote copy wrapper, one remote session at a time, no foreground SSH held open for a long job, report connection failures rather than switching hosts. You are the only remote user. Never edit anything under `direction_agreement`, `automatic_axes_20260914` or `axis_matching_20260914`, locally or remotely.

## Why

The cap-loss run (`CLAUDE_FOLLOWUPS/cap_loss/`) showed the per-pair cap is not where close courts vanish: on Amateur-3 under R the best proposed court sits 40 px from the control while the direction fit is 4.3 px, and on GX0 under M the best is 33.8 px against a 19.7 px fit. Those proposals were recorded after the player gate. We need the courts before that gate to tell "the proposal step never produced a close court" from "the player gate rejected it".

## Where the gate sits

`propose_role` in `scratch/court_det_fix/worklog/checks/independent/player_guided/20260914/axis_matching/run_given.py` builds `transforms` from the horizontal and vertical axis matches, computes `zone.player_fractions(transforms, feet)`, forms a geometry mask `valid` and a player mask `usable`, and then builds `candidates` from the usable indices only. Its record already carries the counts `combined`, `geometry_valid` and `geometry_players`. The corners of every combined court exist inside that function before the masks are applied.

## Do

1. Reuse the remote folder `R/cap_loss/` and read its `run_automatic.py`, `run_matcher.py`, `common.py` and `run_remote.sh`, which are the instrumented copies from the earlier run; the local mirrors are in `CLAUDE_FOLLOWUPS/cap_loss/remote_src/`. Copy `axis_matching_20260914/run_given.py` into `cap_loss/` as well and make the copied `run_automatic.py` import `propose_role` from the copy; assert at start that it resolves to `cap_loss/run_given.py`.
2. In the copied `run_given.py` only, make `propose_role` also return, or stash on the returned record, a float32 array of every combined court's corners in working pixels, in the order of `transforms`, together with the boolean `valid` and `usable` masks. Change nothing about how candidates are built.
3. In the copied `run_automatic.py`, extend the existing side file so each pair also stores `pair_<id>_combined_corners`, `pair_<id>_valid` and `pair_<id>_usable`. Keep the arrays it already writes.
4. Launch two detached jobs with the copied `run_remote.sh` under a new run name `pregate_<UTC timestamp>`: GX0 with arm M and Amateur-3 frame 0 with arm R. Single-threaded, `nice -n 10`. Poll no more than once every five minutes with a Bash sleep between polls.
5. Gate 1, as before: each new generation record must equal the saved one for its case-arm in every pair status, shortlist ID list, shortlist corners, final entry IDs and both winner IDs. The earlier gate script `CLAUDE_FOLLOWUPS/cap_loss/gate_records.py` does this; reuse it. Gate 2: for every pair, the post-gate corners already recorded must equal the combined corners at the usable indices. Print both.
6. Pull the side files and records to `OUT/records/` with the remote copy wrapper.
7. Analyse locally with the frozen control from the E3 record, using the corner-error function already copied in `CLAUDE_FOLLOWUPS/cap_loss/analyse.py`. Per pair: nearest combined court, whether it passed the geometry mask, whether it passed the player mask, nearest usable court. Overall per case-arm: the nearest combined court anywhere, its pair, its two mask outcomes, and the player fractions the record stored for it if available.

## Write

`OUT/table.csv` with one row per pair per case-arm, and `OUT/note.md`: both gate outputs, then per case-arm the nearest combined court against the nearest usable court, then two or three sentences answering: does a court within a few pixels of the control exist before the gate, and if so which mask removes it. If no combined court is closer than the recorded post-gate nearest, say the proposal step itself never produces the court. No fix proposed.

`OUT` = `<repo>/scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/pregate_loss/`. Local python for analysis: `~/.venvs/badminton-cicd/bin/python`.
