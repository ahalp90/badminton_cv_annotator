> Historical record: the 23 September 2026 launch of the speed-up work, filed on
> 25 September 2026. The [speed-up README](../../../court_detector_optimisation_handover/README.md)
> owns current status; the [archive map](../../README.md) records the original
> paths. Content is unchanged.

# Launch: make the accepted court finder computationally affordable

Read [the shared contract](00_SHARED_CONTRACT.md) and mandatory repo context.
Use `external-delegate`, `plan-and-execute` for pipeline changes and the relevant
code-style skills. Do not restart colour/net discovery or launch a broad corpus
before measuring the expensive work on a small representative set.

## Verified intake

1. **Useful geometry exists.** Three automatic vanishing-point seeds expose an
   Am1-54 candidate that the net cue selects. The user accepts the recovery.
   It depends on the expanded pool; ranking cannot rescue the old pool.
2. **Keep G0, G1 and line-template sources.** G0 uses original line evidence;
   G1 uses paint-filtered evidence. Removing G0 loses a useful Am4 case; G1 and
   templates rescue cases including GX5. G0-only failures do not isolate SVD.
3. **Keep bounded net weight 0.04, overrun 4 working pixels.** A post earns support
   if any of its first six of 24 base-to-top samples matches a DeepLSD fragment,
   and no matched fragment extends more than the overrun below its projected
   base. Existing match tolerances are 4 px lateral, 8 degrees and 2 px extent.
   Reward is 0, 0.5 or 1 for zero, one or two supported posts. Add weight × reward
   to frozen paint score; retain gates and original rank on exact ties. Missing
   evidence is neutral. Tape and post tops supply no direct score.
4. **This is lower-post support, not post detection.** Only 4/26 supported posts
   on changed choices match the base sample. The overrun veto changes no winner;
   the bounded paint tradeoff protects the five known regression views. Court
   and net colours are independent. Inputs are cached DeepLSD lines, not Hough.
5. **Keep automatic stripe polarity.** It corrects edge/centre labels using court
   paint evidence, including possible dark paint. Retain old labels if unresolved.
   The accepted 20-case page selects with the net score first, then refits that
   candidate using stripe evidence. Net evidence does not constrain this refit.
   Do not silently move correction before ranking and claim the same experiment.
6. **Positive net weights improve reference agreement; fine tuning is unhelpful.**
   Of 71 unique frames, 45 have usable references (27 landmark, 18 four-corner).
   The seeded Am1 pool replaces its old pool. Median per-frame reference-point
   errors for weights 0/.02/.04/.08 are 3.40/2.75/2.75/2.76 working pixels.
   At .04, Q90 is 5.62 versus 6.20 at .02. Between positive weights, every measured
   median change is under 2 px. Ten-source bootstrap intervals include zero.
   No statistically reliable optimum is established. Overrun 2/4/8 selects
   identical courts. Stripe correction is outside this weight comparison.
7. **Known failures remain.** Am1-5352 and Yellow14 have large unchanged errors
   in the old pools. All settings accept one of eight labelled non-court controls.
   Seven of 71 frames have no gated selection. Two unverified views and 24
   controls lack usable reference geometry. Do not turn missing labels into wins.
8. **SVD12 is upstream search reduction.** It keeps 12 of 16 direction-support
   groups, not 12 lines. Fresh G0/G1 generation defaults to it; full16 remains.
   Nine-case matcher time fell 52.9%. That excludes the rest of the detector.
9. **Runtime is a blocker despite SVD12.** Six SVD12 G0-only baseline trials take
   4.3–33.6 minutes per frame. Generation is 91.6% of summed trial time;
   refitting/scoring alone takes 59–107 seconds per frame. These runs use six
   workers with one numerical thread each, cached line inputs, and omit imports,
   output writing, G1/templates and full video processing. Separate CPU and wall
   timings exist. The later 640-axis trial helps Am2 but increases summed runtime
   26.6%; it is a useful coverage witness, not an adopted global default.
10. **The combined fresh pipeline is untested.** The net experiment reuses older
    candidate pools. Fresh and historical paint measurements differ even on the
    same image/geometry; the cause remains unresolved. Compare fresh arms against
    freshly measured controls. Cache identities include geometry/evidence, not
   merely an origin-key string. The replay cache is repaired.
11. **The existing CourtKeyNet scene path groups views after court inference.**
    `src/annotator/composition_mask.py::detect_cuts` uses ContentDetector.
    `court_evidence.py::detect_scene_evidence` samples up to ten frames per raw
    cut interval before `court_views.py::matching_view_groups` compares perceptual
    hashes and alignment. Only initially valid court scenes can form groups;
    groups need at least three members. `court_evidence.py::_share_scene_corners`
    then shares geometry. This saves no first-pass detector calls. Distinct-view
    scaling needs earlier reuse; cut detection and decoding still scan the video.

## Next question and bounded plan

> Which work can be removed or accelerated while retaining the accepted court
> choices, and what does the combined detector cost per distinct camera view?

1. Record actual Git state and confirm a small set of source artefacts. Create
   the living state files in the shared contract before delegation. Do not rerun
   the completed SVD benchmark or all 71 views as a first action.
2. Use saved stage timings to pick a cheap broadcast case and a slow GX case.
   Trace and profile candidate generation before writing performance code.
   Separate startup, DeepLSD, directions, each proposal source, scoring, refits,
   net scoring, stripe correction and output cost in the eventual complete run.
3. Build a small fresh-baseline adapter; no integrated runner exists yet.
   `run_combined.main` reuses frozen G0/G1 and an older net rule. Generate fresh
   SVD12 G0/G1 plus `am1_recovery_trial.generate_seeded` templates before
   `run_w5.canonicalise_populations`, then score/refit into a complete W5 record.
   Build ranked/gated rows with `bounded_trial.post_features`; use `choose` at
   .04/4, then `refit_selected.refit_selection` for the chosen key/fresh record.
   Retain parent/child/fit-attempt provenance. Include Am1-54, GX0/GX5, Am2-28019
   as a depth witness, and regression/non-court controls. Preserve selection-then-
   correction order. Compare fresh arms with fresh controls, never historical scores.
4. Prioritise measured generation/refit waste. Leads include repeated equivalent
   work, array/batch implementation, caching within a view, and earlier cheap
   screening. These are leads, not diagnosed fixes. Smaller search caps and fewer
   candidate sources change coverage and require separate accuracy evidence.
5. Run one-factor trials with separate output directories and the same inputs,
   numerical-thread settings and hardware. Restore the baseline between trials.
   Observational profiling must not alter choices. For semantic optimisations,
   compare stable identities, geometry and ranking; timing alone is insufficient.
   For pruning changes, report coverage and automatic-selection losses explicitly.
6. Use statistics across the saved quality corpus after a candidate passes the
   bounded checks. Report medians, upper tails, material paired wins/losses,
   failure/abstention and source grouping. Retain contrary cases and unlabelled
   rows. Use a few visuals only where the numerical result cannot settle utility.
7. Reuse the existing scene contract and add earlier compatible-view reuse.
   Hash matches do not prove unchanged camera geometry. Validate first acquisition
   and cheap confirmation before adding robust scene consensus.
8. Stop a proposed optimisation after a matched run shows no useful saving or an
   unexplained coverage/quality loss. Keep its evidence; do not expand into a
   parameter sweep. Stop for input if an essential quality/runtime tradeoff exceeds
   the accepted imperfection level. Do not perfect every hard case before progress.

## Delegated questions and validation

- Sol high, code read/profile: `w5_holistic/automatic_generation.py`,
  `w5_holistic/line_template_source.py`, `w5_holistic/run_w5.py` and callers.
  Locate measured expensive loops and repeated work; propose one bounded change.
- Luna max, keyed records: `svd_search/run_20260923/cases/baseline/` and generation
  records. Extract stage timings/counts and pair a fast and slow case.
- Opus 5-5 xhigh, independent bounded audit: review the concrete proposed change
  and raw before/after cases for hidden coverage loss or unfair timing. No timeout.
- Assign implementation files explicitly to Sol high after the profile identifies
  a useful change. Keep writers separate. The coordinator approves conclusions.

Use scope-specific repo checks: scratch syntax/smoke; pipeline tests, Ruff and
Pyrefly via `~/.venvs/badminton-cicd/bin/`, plus AGENTS.md's applicable gates.
Record commands, exit codes, hardware, threads and warm/cold distinctions. No GPU
training or dependency replacement is justified. Commit coherent feature-branch
checkpoints; preserve the accepted saved baseline.

## Evidence to open only when needed

Paths below are relative to `scratch/court_det_fix/`, unless explicitly prefixed.
Stable result key: `(case_id, source_label, origin_key)`; Am1-54 has two pool labels.
Use `(case_id, arm)` for the SVD search trials. Match image hash and dimensions.

| Need | File or directory |
| --- | --- |
| Accepted scoring implementation/tests | `net_recovery/bounded_trial.py`, `net_recovery/test_bounded_trial.py` |
| Corrected-fit replay | `net_recovery/refit_selected.py`; `colour_consistency/edge_auto_trial.py` |
| Seeded proposals and mixed-score caveat | `colour_consistency/am1_recovery_trial.py`, `colour_consistency/AM1_RECOVERY.md`; `net_recovery/run_combined.py` |
| Statistics, script and all per-frame results | `net_recovery/statistics/paired_reference_report.md` and sibling `.py`/`.json.gz` |
| Fixed cohorts and original full scan | `net_recovery/bounded_split.json.gz`, `net_recovery/saved_net_scan.json.gz` |
| Net experiment record and prior user rulings | `net_recovery/WORKLOG.md` |
| SVD implementation and stage timings | `svd_runtime/README.md`, `svd_runtime/RESULTS.md`; `svd_search/WORKLOG.md`, `svd_search/run_20260923/cases/` |
| Full quality gates/reference conventions | `w5_holistic/verifier.py`; `net_recovery/scan_saved.py` |
| Existing scene codec/integration (repo-relative) | `src/dataset_builder/vision.py::build_detected_court_stage`, `persist_court_vision`; `src/dataset_builder/_court_codec.py::_validate_scene_records` |
| Prior measured/rejected ideas, only for a named issue | `DETECTOR_DECISIONS.md`, `FP_INDEX.md` |

Important local-only inputs under the repository root:
- `local_scratch/net_recovery/20260923/bounded/bounded_trial.json.gz` (about 17 MB),
  plus weight intervals and run/check logs.
- `local_scratch/net_recovery/20260923/selected_polarity/bounded_results.json.gz`
  and `bounded_requests.json.gz`: the 20 corrected fits and exact requests.
- `local_scratch/external_delegate/20260923-am1-recovery-trial/seeded_pool.json.gz`.
- `local_scratch/external_delegate/20260923-net-bounded-audit/result.md`: independently
  checked limitations. Full external outputs are leads; the worklog records rulings.
These inputs are not all in Git. Verify their presence before a new worktree or
remote checkout; copy/link explicitly. Never substitute a similarly named pool.

No compute experiment is active at handover. Completed remote SVD jobs and their
transfers must not be relaunched. A staged 11-case combined batch on Carmack was
never run; it predates the current bounded rule. Gallery/Serena services may
remain; their last known ports are 8883/9121. Check before starting duplicates.
The galleries in `net_recovery/{bounded_gallery,polarity_gallery}/` use tracked
images from `colour_consistency/gallery/`.

Persisted scene records retain cut intervals, samples, validity, active corners
and optional `view_group_index`. Reuse this contract; cuts are not distinct views.

Write one final optimisation report with per-stage/total timings, paired quality
results, accepted/rejected changes, unresolved failures and the next decision.
Record final Git/worktree state/checks and update pickup.
