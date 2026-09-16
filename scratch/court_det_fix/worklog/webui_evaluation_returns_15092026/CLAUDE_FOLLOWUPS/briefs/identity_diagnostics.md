# Brief: identity diagnostics on every exported winner

Read-only on every input. Write only under `OUT`.

## Why

An external audit found two mechanisms behind false paint winners. On ShuttleSet 03 scene 19 the false winner's visible markings leave court extent algebraically free (named-line constraint rank 7 instead of 8). On Amateur-2 frame 28019 the false winner projects the near baseline within half a working pixel of the near long-service line, so two markings share one ridge. The audit computed these for nine candidates. We want them for every exported winner, to see whether "rank 8 required" separates false winners from approved courts without any threshold.

## Inputs

All paths relative to the repository root `<repo>`.

- Winners: `experiments/annotator/independent_court/recorded/player_guided/projective_patterns/evaluation/ranking_records.json.gz`. Top level has `populations`, a list. Inspect the structure; each entry carries case, population, candidate ID, ranking role, corners and homography. There are 17 distinct automatic winners plus GX0 controls 1864 and 5144 and comparator 89.
- Visual rulings: `experiments/annotator/independent_court/recorded/player_guided/projective_patterns/automatic_axes_visual_judgements.md` and `automatic_axes_results.md`.
- Audit code: `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/_bundles/court_identifiability_extension_7299ff3/audit.py`. It has `reconstruct_h`, `clip_segments` (the 12-working-pixel availability rule), `constraint_rank`, `interval_records` and `line_separation`. Its `results/audit_results.json` holds the nine candidates' values. Copy `audit.py` into `OUT` and import from the copy; do not edit the geometric functions. Read its `main()` to see how it feeds candidates in.
- Guide: `experiments/annotator/independent_court/recorded/player_guided/projective_patterns/evaluation/README.md` for coordinate conventions and candidate identity (case plus population plus candidate ID).

## Gate, run first

Reproduce the nine candidates in `audit_results.json`: available interval counts, constraint ranks (scene 19 `165:6702` is 7, the others 8), and the projected separations for Amateur-2 `184:4123` (near baseline to near long-service, 0.378 to 0.452 working px) and `30:33` (far long-service to far baseline, 1.86 to 2.24 working px). Print your values beside theirs. If any differs beyond rounding, stop and report.

## Do

One row per candidate: case, population, candidate ID, which ranking chose it, visual ruling (quote the judgement file's words; "no ruling recorded" where none), available interval count, constraint rank, the minimum projected separation in working pixels between any two distinct markings within the image and which two, and the maximum corner error to the reference as the records give it.

Write `OUT/table.csv` and `OUT/note.md`: the gate output, the table, then answer only these: does every approved or usable court have rank 8; does every false winner have rank under 8 or a separation under one pixel; which candidates break either pattern. No thresholds proposed.

`OUT` = `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/identity_diagnostics/`. Use `~/.venvs/badminton-cicd/bin/python`.
