# Court detector: current state

## Resume — 23 September 2026

**The accuracy experiments have a usable outcome. Compute cost blocks deployment.**
The next session should profile and integrate the accepted pieces, then make
substantial performance reductions. No optimisation or remote experiment is
running. The full experimental history belongs in
[DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md), and paths in
[FP_INDEX.md](FP_INDEX.md). [INDEX.md](INDEX.md) explains the document roles.

To start speed-up work, read the
[speed-up README](court_detector_optimisation_handover/README.md). The
23 September [shared contract](archive/20260925_optimisation_handover/handover_20260923/00_SHARED_CONTRACT.md)
and [launch prompt](archive/20260925_optimisation_handover/handover_20260923/01_LAUNCH_OPTIMISATION.md)
are archived.

## What to carry forward

- **Candidate sources:** retain G0, G1 and templates. G0 uses original fragments;
  G1 uses paint-filtered fragments. Each source preserves useful coverage
- **SVD12:** the fresh-generation default keeps twelve of sixteen direction
  groups. It saved 52.9% of measured matcher time, not total detector time
- **Am1 proposal seeds:** three automatic vanishing-point seeds expose a useful
  frame-54 court absent from the old pool
- **Bounded net selection:** keep weight 0.04 and overrun 4 working pixels.
  Lower-post support gives a limited bonus to paint score; missing support is
  neutral. Net and court paint colours are independent
- **Automatic stripe polarity:** carry forward the accepted correction of paint
  edge/centre labels, including possible dark markings. Keep old labels when
  polarity is unresolved

The user judges the displayed **net-selected, stripe-corrected courts very
usable**, including frequent tiny imperfections in amateur footage. This is an
overall gallery judgement, not independently recorded labels for every frame.

The [paired statistics](net_recovery/statistics/paired_reference_report.md) cover
45 reference-bearing frames. Net weights 0.02 and 0.04 tie at 2.75 working pixels
median per-frame reference-point error, versus 3.40 at zero and 2.76 at 0.08.
Weight 0.04 has a modestly better upper tail than 0.02. No statistically reliable
weight optimum is established; further small-parameter tuning is not a priority.

## What is not established

The accepted experiment uses saved pools: select a candidate with the bounded
net score, then apply stripe correction. **No single runner yet combines fresh
SVD12 G0/G1 generation, seeded templates, bounded selection and stripe refitting.**
`net_recovery/run_combined.py` reuses frozen G0/G1 and applies the older net rule;
it is not that baseline. The launch prompt names the components to assemble.

Fresh and historical paint scores differ on matching images/geometry. Compare
fresh trial arms with freshly measured controls. Do not attribute that drift to
an optimisation, or assume a matching candidate ID proves matching evidence.

Am1-5352 and Yellow14 remain large numerical failures in the old pools. All net
settings accept one of eight labelled non-court controls. Robust abstention and
failure frequency remain open. Scene consensus and the complete replacement
runtime are not integrated. Repeated agreement can still select the wrong court.

The floor/hue rejection experiments are closed without a useful adopted rule.
Do not reopen colour searches, manual paint selection or player-floor machinery.
The accepted stripe correction is a separate fitting result.

## Compute requirement and next work

The user would tolerate roughly **30 seconds for a five-minute video**, with
**90 seconds as an upper end**. Report startup/warmup separately; the user
suggested excluding it from that processing budget. The earlier 5%-of-duration
proposal was rejected. Target deployment hardware is unspecified.

For longer videos, expensive court search should roughly follow the number of
distinct compatible views. Current PySceneDetect ContentDetector cuts are followed
by per-cut court inference; perceptual hashes and alignment group recurring views
only afterwards. That grouping does not yet avoid repeated court searches.

The measured SVD12 G0-only trials take **4.3–33.6 minutes per frame**. Generation
uses **91.6%** of summed time; refitting/scoring alone takes **59–107 seconds**.
This establishes the need for substantial optimisation. These runs use cached
line inputs and omit G1/templates and complete video processing.

Start with saved stage timings and a bounded fast/slow profile. Establish one
fresh combined correctness baseline, then test measured reductions in generation
and fitting work. Preserve useful source coverage. Early reuse and scene-level
validation follow; they cannot make an unaffordable first search disappear.
Use paired statistics and only a few consequential or ambiguous visual checks.
Amateur references mix stripe centres and outer edges, so tiny drift is ambiguous.

## Working state and recovery

- Branch: `fix/court-det`; checkpoint commits and push are authorised. Query
  `git log -1` and `git status` for the final handover revision; do not reset
- Behaviour checkpoint: `82c8a4d`; statistical checkpoint: `94b242c`. The current
  close-out changes navigation/storage documentation, not detector behaviour
- Ranking tests, full frozen-pool replay, 20 stripe replays, scoped lint, whole
  project types and gallery checks passed. The statistical script, syntax, lint
  and record checks passed. All reported successful checks exit 0
- No new runtime test is required for this documentation/storage refresh. Its
  gates are content preservation, links, file disposition and fresh-reader entry
- Completed Carmack SVD jobs and transfers must not be relaunched. A staged older
  combined batch was never run and does not implement the accepted bounded rule
- Gallery/Serena services may remain at ports 8883/9121; check before restarting
- Raw datasets and full run outputs include local-only inputs. Read the
  [storage map](FP_INDEX.md#local-data-and-recovery) before a new checkout or remote run
- The [top-level refresh record](archive/20260923_top_level/README.md) preserves
  the previous prose and exact untracked-file inventory. Nothing was discarded

The accepted [selection gallery](net_recovery/bounded_gallery/index.html) and
[stripe-fitting gallery](net_recovery/polarity_gallery/index.html) remain available
for a named question. A new broad visual review is not the next task.
