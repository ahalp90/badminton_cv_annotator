# Source map

The files beside this map are small source snapshots. Paths below are relative
to the repository root and retain the original line ranges used by the C2
check.

| Seed file | Original path | Relevant lines |
| --- | --- | --- |
| `projective_seed.py` | `scratch/court_det_fix/frozen_helpers_20260914/axis_matching/projective_seed.py` | `corner_errors` 14-18; `offsets` 70-96; `score_axes` 99-119; `necessary_players` 122-132; `match_axis` 135-187; `combine` 190-201 |
| `axis_replay.py` | `scratch/court_det_fix/line_identity/axis_replay.py` | `pair_record` 83-86; `sweep` 137-145; `nearest_over_product` 160-169; `replay_pair` 172-279 |
| `filter_replay.py` | `scratch/court_det_fix/line_identity/filter_replay.py` | `stage_one` 148-171; `axis_stage` 174-219; `write_inputs` 222-234 |
| `run_automatic.py` | `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/pregate_loss/remote_src/run_automatic.py` | `write_pool` 106-130; `generate` 133-207; filtered producer copy includes the pre-gate arrays |
| `run_given.py` | `scratch/court_det_fix/worklog/webui_evaluation_returns_15092026/CLAUDE_FOLLOWUPS/pregate_loss/remote_src/run_given.py` | `canonicalise` 76-83; `axis_diagnostic` 86-108; `pack_axis` 111-117; `propose_role` 136-171 |
| `run_population.py` | `scratch/court_det_fix/frozen_helpers_20260914/vp_pruning/run_population.py` | `prepare` 18-28 |
| `inspect_appearance.py` | `scratch/court_det_fix/frozen_helpers_20260914/axis_matching/inspect_appearance.py` | `profiles` 21-38; `inspect` 51-81 |
| `assignment.py` | `experiments/annotator/independent_court/assignment.py` | `Observations` 37-45; `prepare_observations` 67-112; support constants 24-34 |
| `detector.py` | `experiments/annotator/independent_court/detector.py` | `Settings` 35-59; `project` 95-103; line preparation and grouping 158-221 |
| `stripe_observations.py` | `experiments/annotator/independent_court/stripe_observations.py` | `fragment_weights` 40-46; `measure` 88-120; `score_model` 151-185 |

`run_automatic.py` and `run_given.py` are the instrumented producer copies
used for the pre-gate records. Their candidate construction and axis matcher
call are unchanged; the extra pre-gate arrays are the only C2-relevant
instrumentation.
