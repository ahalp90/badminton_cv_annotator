# What the court detector leaves out

The detector runs only the accepted D17 chain. This page lists what it drops
from the research scripts, and where each piece plugs back in. Places are named
by module and function, not line number, so they stay true as the code moves.

## Left out of the detector path

| Left out | What it fed | Where it plugs back in |
| --- | --- | --- |
| Pool stripe score, paint profile and the two research winner IDs | The research A ranking and saved records only. W5 duplicate merging, the C ranking, the net choice and the refit never read them | `legacy_evidence=True` (the default) on `automatic_generation.generate` and `run_automatic.evaluate_pool` |
| W5 research extras: B ranking, diagnostic controls, rank sensitivity, review candidates, saved arrays and case-record files | Research reports | `run_w5.process_case`, which wraps `run_w5.score_populations` |
| Replay of the parent's saved W5 fit before the stripe refit | A check that the refit starts from the same fit | `stripe_refit.refit_chosen(replay_check=...)`, on with the `self_checks` switch |
| The search's input audit and population files | The W5 stage, which read the populations back | `generation.ensure_populations`. The detector repeats the files' JSON round trip in memory (`detect.json_round_trip`) |
| Case registration and loader swaps for the packs and control views | Looking up a view's source record, provenance and frame by case ID | `wider_evaluation/run_cases.load_runtime`. The detector's caller passes these in `ViewInputs` |
| Frame MD5 check against the 20260922 manifest | A check that the frame is the frozen one | `run_views.py`, which keeps it for the 28 test views under `self_checks` |
| Per-function timers | Timing reports by function | `d17_timing/run_d17.py` (`clock.wrap`). The detector's `timing` switch gives seconds per stage only |

## Switches for each improvement (not built)

Each accepted improvement is a fixed constant today. An ablation switch for one
would be a `bool` on `Switches` and one branch in `CourtDetector.detect`. The
off path for each:

- **Seeded line templates.** Pass `seed_points=None` to
  `line_template_source.generate`. That gives the unseeded templates with the
  same visibility floor.
- **G1 paint filter.** Skip `search.paint_mask` and `search.filtered_source`,
  and pass an empty G1 population to `run_w5.score_populations`.
- **Net choice.** Call `net_choice.choose` with weight 0. That picks the top of
  the C ranking, which the self-check already exercises.
- **Stripe refit.** Skip `stripe_refit.refit_chosen` and return the chosen
  candidate's own corners from the case record.

Before an ablation run is trusted, check each off path against the earlier run
that used the same setting.
