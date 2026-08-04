# Orchestrator audit notes

Rulings per module: KEEP (hit-list), DETAIL (per-file detail only), KILL (with grounds).
Effort: S/M/L. Written incrementally; source of truth for the report.

## src/scraper/config.py (273 lines; verdicts 7C/1M, probe 2, incidental 1)

- KEEP stale-doc L5: docstring claims "stage 8/9 trajectory-rule constants" live here;
  module holds stages 1/2/3/10/11 + shared only. Fix: delete phrase. S.
- KEEP docs-orientation cluster (merge of verdict L145 B5, L154 s29, L157 streaming;
  probe L147 "candidate seats", L156 TrackNet): the stage-3 constants block leans on
  undefined codenames and unnamed consumers. The docstring's provenance key defines
  "spec sN" and D-numbers but not B5/s29/streaming. Fix: one-line definitions or plain
  names. S. (Cross-links to selection.py "B6" — package-wide codename habit.)
- KEEP (package entry, shared with download_scraped_videos.py) ordering/orientation:
  stage 4 exists only implicitly — downloader file breaks stageN naming (cross finding
  config L229), and its two constants (CONCURRENT_FRAGMENTS, DOWNLOAD_WORKERS) sit
  inside the shared rate-limit section labelled "stage 4". M (rename file or fix prose).
- DETAIL docs-prose: dense model-budget comment L148-152 (fold into codename cluster);
  semicolon-joined comment sentences L194, L229. S each, below hit-list threshold.
- KILL incidental L273 write_candidates blank-fill: documented deliberate schema
  stabilisation in docstring + header comment; rubric's explained-quiet-path exception.

## src/scraper/stage1_index.py (277 lines; verdicts 9C/2K, probe 3)

- KEEP stale-doc L101 (merged with probe L113): enrich_row docstring ":return: True
  when a metadata call actually hit YouTube" reads as success-only; timeout/error/
  parse-fail branches also return True. True actually means "request attempted, pace
  it". Probe misread it exactly that way — live comprehension trap. Fix: reword
  return line. S.
- MERGE into package codename entry: probe L212 (B5), probe/verdict L176 (OPEN).
  Note: config.py's module docstring DOES define "OPEN" and "spec sN"; stage modules
  don't point to it. Fix is a one-line pointer or plain wording per site. S.
- DETAIL style-house L72/L117 (stderr[:200]) + L81 (line[:120]): unnamed log-trunc
  limits, two different values for one concept. S, low pain.
- DETAIL overloaded-one-liner L127: or-fallback + conditional + str() in one line;
  split. S.
- DETAIL style-house L165: int(float(...)) lacks why (decimal duration strings). S.
- KILL repetition L125: two short parallel assignments; house rule against
  abstracting harmless syntax repetition.
- KILL data-structure L205 (TypedDict for rows): would create a second row-shape
  declaration competing with CANDIDATES_COLUMNS, the documented single source.
- KILL style-house L250 (int() around bool +=): idiomatic bool-as-int counting;
  no rubric rule; verifier already hedged.

## src/scraper/stage2_transcripts.py (355 lines; verdicts 4C, probe 2)

- KEEP (package entry) repetition L77-87 + L179-188: subprocess run/timeout/
  returncode/print-and-return-None pattern duplicated here twice, and twice more in
  stage1_index (L64-73, L109-118). Package-wide: one run_ytdlp(cmd, label, timeout)
  helper in config.py collapses ~6 sites. M. (Sibling of cross finding on the
  duplicated LLM retry loop in stage3/stage10.)
- DETAIL probe L64: WhisperX fallback triggers on ANY caption-pull failure (timeout,
  yt-dlp error), not only missing English track; module + whisperx_fallback
  docstrings narrate track-missing only. Add "or the caption pull fails". S.
- DETAIL data-structure L97/L250 (merged): segment dicts and {source, segments}
  return could carry TypedDicts; docstrings do state shapes, dicts are JSON-bound.
  S-M, below hit-list.
- KILL probe L312 (sidecar-per-video overstatement): module docstring's failure
  paragraph states log-and-skip; the probe missed present context.
- KILL overloaded-one-liner L122 (_vtt_seconds): cohesive h/m/s/ms arithmetic;
  house preference for raw arithmetic; splitting hurts.

## src/scraper/stage3_triage.py (300 lines; verdicts 4C/3K, probe 2, incidental 1)

- MERGE docs-prose L174 (D9/spec s4 bare shorthand) into package codename entry.
  Config.py docstring maps D-numbers to topics; stage modules cite them bare.
- DETAIL docs-prose L176: "enrichment" is stage-1 vocabulary used bare here; name
  the stage. S.
- DETAIL stale-doc L189: comment says "duration_min is never zero" but no such
  variable exists (expression is seconds/60.0); phantom name from a refactor.
  Greppable-name trap; trivial fix. S.
- DETAIL ordering L271/L275 (RESURRECTED from luna kill): _write_keep_back defined
  after its only caller, against the module's helpers-before-callers pattern.
  Luna killed on an indentation-whitespace technicality in the quote — over-strict
  kill; substance was real. Move the def up. S. (Note for funnel stats.)
- KILL docs-prose L50 ("yt-dlp id" jargon): the package's core tool name; common
  casual term within domain; defined by module context.
- KILL probe L187 (keep-rule compression): three legs are plainly documented at
  both docstring levels; probe error, not code trap.
- KILL probe L135 (SDK contract unverifiable): external-dependency knowledge is
  out of scope for module readability.
- KILL incidental L163 (broad except in retry): noqa + comment + re-raised as
  TriageError; rubric's explained-quiet-path exception. Justified.
- CROSS (keep, anchored here): circuit-breaker policy implemented 4 ways across
  stages 1/2/3/10 with different floors/shapes; unify or document divergence. M.

## src/scraper/download_scraped_videos.py (565 lines; verdicts 5C/1K/1M, probe 2;
   seeded: mega-module, mega-function _download_one 111, deep-nest L505)

- KEEP docs-orientation L1-4 (merges verdict L3 + L524): module docstring is 3 lines
  for a 565-line module, first sentence garbled garden-path grammar ("Scraper
  downloads request and verify audio"), no mention of manifest writing, worker pool,
  or CLI; main()'s "mass-failure status" never defined (exit 2 at >=50%). Rewrite
  docstring + one line on exit codes. S.
- KEEP comprehension-trap L48 (merged probe + MISREAD verdict): .f<digits> exclusion
  is the yt-dlp per-format intermediate convention (video.f137.mp4 pre-merge);
  nothing says so. Why-comment. S.
- KEEP mega-function _download_one L271-391 (seeded, subsumes verdict L369
  repetition): existing-file path + fresh download + audio verify in one 111-code-
  line body; split at the existing/fresh boundary; audio-probe failure handling then
  dedups naturally while keeping the unlink-vs-retain policy difference explicit. M.
- KEEP data-structure L244: task tuple (url, video_id, title, output_dir) unpacked
  positionally in 4 helpers; NamedTuple DownloadTask. S-M.
- DETAIL probe L212: [videos.<name>] header always json.dumps-quoted while other
  keys go through _toml_key; same output for dotted basenames — use _toml_key or
  one comment. S.
- DETAIL repetition L282/L350: identical multiple-completed-outputs raise block x2;
  tiny helper or fold into mega-function split. S.
- OK mega-module 565 (seeded): within flexible band; sections clear. The embedded
  ~120-line TOML writer is the bulk driver; cross finding (stage11 manifest
  re-validation -> shared reader in config) is the real fix. No separate entry.
- KILL deep-nesting L505 (seeded): with/for/try executor-collection idiom; early
  returns don't apply.
- CROSS confirmed by read (anchored here): _check_ytdlp L69 duplicates
  config.check_ytdlp; throttle flags L331-335 partially rebuild
  config.ytdlp_throttle_args (whose docstring assigns sleep-interval to exactly
  this path); VIDEO_EXTENSIONS L30 (x3 across package); `from . import config` +
  3 re-exported constants L26-28 vs package's from-import idiom; keep=='True'
  filter L440 (x3 across package). All S each, real.

## src/scraper/stage10_clean.py (467 lines; verdicts 9C/1K, probe 2; seeded
   deep-nest run_clean L210)

- KEEP comprehension-trap/structure L192-269 (merges verdict L248 one-liner, probe
  L248, data-structure L192, seeded deep-nest L210): run_clean's scoring
  bookkeeping — the `_score_pending` temp key set at L217, consumed-and-restored
  inside an OR with a side-effect pop at L248-249, re-read L260, stripped L264,
  threaded through `loaded_sidecars` positional 4-tuples across four sequential
  loops. Probe + verifier + my read all stumbled/confirmed. Fix: NamedTuple for
  loaded sidecars + explicit pending-set instead of temp dict keys. M. Top-tier
  entry for this module.
- KEEP docs-over-types L313/L344 (+ L75/L109 dict returns): fine-models passed as
  bare `tuple`, unpacked positionally across the load/refine boundary; clean result
  as bare `dict`. NamedTuple FineModels + typed clean-result shape. S-M.
- DETAIL probe L338: hallucination_silence_threshold=2.0 inline literal while its
  sibling signed-off settings sit as named constants L51-53; hoist. S.
- DETAIL repetition L196-199/L414-417: sidecar lookup+skip duplicated across the
  two passes. S.
- DETAIL style-house L374: multi-line comprehension with method call + nested
  iteration; house digest says unpack to explicit loop. S.
- AGREE KILL L58 (prompt constant "lacks docstring"): adjacent comment explains it.
- CROSS confirmed: call_clean_llm L109-131 near line-for-line copy of stage3's
  call_triage_llm (retry/backoff/print); _VIDEO_EXTS L47 (x3); keep=='True' L187 +
  L404 (x3). Shared helpers in config. S-M.

## src/scraper/stage11_pairing.py (365 lines; verdicts 4C/3K, probe 0, incidental 2)

- KEEP stale-doc L261: "_manifest_pairing_index" docstring says "rally ids"; it maps
  VIDEO ids (param video_ids, keys are video ids). One-word fix, real mislead at the
  manifest boundary. S.
- DETAIL style-house L97-103: one concept, two names in one signature —
  `_believed_replay_in_rally_interior` vs `duration_filtered_replay_mask`; align
  vocabulary (believed == duration-filtered is implicit). S.
- DETAIL docs-prose L136-139: 4-clause pairing-tie sentence; split. S.
- DETAIL docs-orientation L240: one-line docstring on the manifest validator;
  state the required shape. S.
- INCIDENTAL (report, silent-failure section): L222 missing chunk sidecar -> []
  with no log even for commentary-eligible videos (rallies quietly unpair);
  L231 missing replay mask -> None (pairing proceeds unmasked; main logs nothing).
  Both plausible S fixes (warn when eligible video lacks input). Luna incidentals,
  audit-confirmed.
- AGREE KILL L199/L207 (docs-over-types), L267 (repetition helper would encode no
  shared rule; branches differ meaningfully).
- PROBE clean: cold read fully held; coherence positive.
- CROSS confirmed: manifest re-validation here (L239-253) is a looser re-implement
  of download_scraped_videos' _validate_manifest; VIDEO_EXTENSIONS L44 (x3).

## src/annotator/batch_report.py (101 lines; verdicts 7C, probe 0, incidental 1)

- DETAIL misleading-name L22: _count_word pluralises, doesn't count; rename
  _pluralise. 2-line body adjacent to callers, so pain is low. S.
- DETAIL docs-prose L35/L36/L89 (grouped): domain terms (shuttle-track file,
  doubles filter, rally spans) used bare; one orienting sentence in the module
  docstring covers all three. S.
- KILL needless-abstraction L26 (_format_reason): 2-line named normalisation
  directly above caller; naming value >= hop cost.
- KILL repetition L40 (five sum() passes): explicit comprehension-sums beat one
  accumulator loop for a small report; house prefers explicit.
- KILL repetition L49 (video_word in both branches): one-line hoist, zero insight.
- KILL incidental L65 (reason=None -> blank): cosmetic report field, no data risk.
- PROBE clean.

## src/annotator/dead_mask.py (84 lines; verdicts 11C, probe 0)

- KEEP docs-orientation (merges 6 verdicts: L50/51/52/56/57/59): build_dead_mask is
  the public 9-param dispatcher with zero :param: docs and no shape annotations;
  which params each mode requires is discoverable only from the branches and the
  two imported builders. Fix: per-mode param table in docstring + shape notes
  ((n_frames,) bool masks, (n,3) track, vote threshold semantics). S-M.
- DETAIL style-house L74: keep_vote param renamed to votes after validation;
  align vocabulary. S.
- KILL docs-orientation L19 (_validate_composition_inputs docstring): private
  validator whose loud error messages are the contract; body plainly readable.
- KILL repetition L69/L80 (combine_mask call x2): branch-order dispatch clarity
  beats dedup; verifier's own note concedes ordering rationale.
- KILL overloaded-one-liner L40: idiomatic single-purpose numpy bounds guard;
  house named-mask rule targets reusable accumulation, not raise guards.
- PROBE clean (Haiku read a 3-mode mask dispatcher correctly — coherence positive).

## src/annotator/resolve.py (49 lines; verdicts 4C/1K, probe 3)

- KEEP docs-orientation L1-5 + L27 (merges 2 verdicts + probe L34/L36 context
  gaps): module + function docstrings say only "probing fps is the caller's
  business" (twice) and never say what resolve PRODUCES (fps-scaled constants +
  thresholds, direct base-30 timings converted to frame counts). The probe's two
  scale-semantics stumbles trace to this. Rewrite both docstrings, one line
  defining base-30 (values quoted at the 30 fps reference rate) or pointer to
  fps_constants. S. [PENDING: confirm where base-30 is canonically defined when
  auditing fps_constants.py.]
- KEEP (cross, anchored here): _OVERRIDABLE_BASE30_ROWS L14-23 hand-copies
  FpsConstants field names + one Stage8Thresholds field; build from
  FpsConstants._fields + explicit extras so new fields can't silently fail as
  "unknown". S. (Cross scan finding, audit-confirmed.)
- DETAIL docs-prose L3: negative phrasing ("never defaults it"); state positively.
- AGREE KILL L14 misleading-name "ROWS" (allowlist role clear from set difference).
- Probe L14 whitelist-rationale stumble folds into the two keepers above.

## src/annotator/calibration/schemas.py (203 lines; verdicts 2C/1K/2M, probe 0)

- DETAIL ordering L87-100: BEST_CONFIG_COMPARISON_COLUMNS splits the three
  contact schemas; move CONTACT_FRONTIER/STABILITY next to CONTACT_SWEEP. S.
- DETAIL docs-prose L3: drop "raw" before recall_5 (only precision axis is raw). S.
- KILL MISREAD L142 (VERDICT_ISSUED name): conventional CONSTANT=value style;
  _KEY-suffixed siblings disambiguate.
- KILL docs-prose L178: accurate concise sentence; positivity rewrite is churn.
- CROSS anchor: WINNER_JSON_TOLERANCES_BASE30 L143 is one of the three independent
  (1, 2, 5, 10) tolerance tuples (with sweep.py TOLERANCES + scoring defaults).
- PROBE clean.

## src/annotator/calibration/run_cli.py (134 lines; verdicts 8C/1M, probe 3)

- KEEP comprehension-trap L30 (merges verdict L30 MISREAD, probe L30/L81, verdict
  L75 params): the injected-seam contract is opaque — FixtureRunner =
  Callable[..., object] hides that a runner takes a Fixture (+ optional
  no_replay_mask kwarg) and returns the metrics object the flattener accepts; the
  L107 if/else exists precisely to avoid passing the kwarg to runners that lack
  it, and nothing says so. Fix: document the seam contract at the alias + one
  branch comment. S.
- DETAIL stale-doc L4: docstring says "adds only the two source roots" but the
  code sys.path-inserts exactly one (_BST_X); recount or reword. S. (Audit-found
  while checking the L18 claim.)
- DETAIL docs-prose L85-93: run_manifest docstring leaks internal kwarg syntax
  (``raw_exclusion_mask=``), "digest verification" without naming the artefact,
  and vague "let process failures raise"; tidy prose. S.
- KILL unclear-name L18 (_BST_X): canonical repo package name; rename is noise.
- KILL repetition L107: branch is load-bearing (kwarg-compat), fold comment into
  the seam-contract keeper.
- KILL needless-abstraction L61 (_validate_environment): named pre-loop phase.
- KILL docs-orientation L123 (main docstring): transparent wrapper, module
  docstring documents the command.
- KILL probe L62 (fixtures_root env var): cross-file; error message orients.

## src/annotator/calibration/selection.py (222 lines; verdicts 5C/2K, probe 5)

- KEEP misleading-name (merges probe L66 + L155): two selection-key names
  misdescribe their rule and the probe conflated exactly as the names invite —
  boundary_report_key_fewest_merges ranks by swallowed_rallies (excess rallies
  inside merged spans, not a merge count); contact_live_key_floored_f1 ranks by
  RAW +/-5 F1 (floors filter eligibility elsewhere, the F1 is never floored).
  Rename or correct in one docstring line each. S. High value: these name the
  winner-selection rules.
- KEEP docs-orientation L3-17 (merges verdict L3 B6 x2, docstring-bulk verdict,
  probe L9/L14; cross B6 finding): identify B6 as sweep.py once, break the
  16-field contract block into grouped bullets (aggregate vs strict-boundary
  fields), and add a one-line gloss for the phase vocabulary (boundary = rally
  span limits, contact = stroke contacts) covering verdicts L43/L124. S.
- DETAIL style-house L198-203 (RESURRECTED, luna indentation-kill #2): 6-line
  comprehension whose predicate calls two functions; house digest prefers an
  explicit loop. S.
- KILL magic-number L54 (the 2 in F1): the formula's own coefficient, F1 named
  in docstring.
- KILL probe L46 (rally undefined): core project domain term.
- AGREE KILL L30 constants comment claim.

## src/annotator/calibration/fixtures.py (389 lines; verdicts 9C/2K, probe 3)

- KEEP docs-prose/orientation L54-62 + L68 (merges 4 verdicts): Fixture docstring
  packs identity-grammar + court_present proxy + opaque tracker-lifecycle sentence
  ("producer choice re-approved at its activation commit") into two dense
  sentences; and the nested positional tuples (court_geo = (x_bounds, y_bounds,
  net_band), resolution = (width, height)) are documented nowhere. Rewrite
  docstring: split sentences, define stem-plus-role plainly, document tuple
  layouts, move/delete lifecycle sentence. S.
- REPO-WIDE codename cluster grows: "Wave 1" L247, "Wave 3 state packet" L293
  join B5/B6/s29/D-numbers/agy-F1 as undefined project-history codenames in code.
  One hit-list entry covers all sites.
- DETAIL probe L217: REF_COURT_M[0] positional access; nearby comment gives
  13.4 m but the index stays opaque — gloss `(length_m, width_m)` once. S.
- DETAIL probe L357: SHARED_FILES' consumer is not in this module and
  verify_fixture ignores it; name the consumer in a comment. [PENDING: confirm
  consumer in gt_scoring; if unconsumed, escalate.] S.
- DETAIL style-house L207: (3,3) shape note on inverse homography. S.
- DETAIL ordering L202/L238: _rounded_bounds defined after its caller. S.
- DETAIL data-structure (RESURRECTED from quote-gate kill #3): CalibrationGeometry
  and Fixture both store net_band twice — as a field AND as court_geo[2]. Drift
  invitation; pick one. S.
- KILL docs-orientation L141 (_read_source_frame): loud explicit validation body;
  docstring states role.
- KILL probe L194 (get_H contract): imported shared.court knowledge, error
  handling present.

## src/annotator/calibration/scoring.py (616 lines; verdicts 13C/6K/2M, probe 4;
   seeded mega-module 616, deep-nest strict_contact_rows L309)

- KEEP stale-doc L501 + L551 (RESURRECTED x2 from quote-gate kills #4-5; luna's
  own notes say "the underlying mismatch is visible"): both contact docstrings
  document a 3-tuple ``(rally_id, contact_frame, proximity_ok)`` while the hints
  declare a 4-tuple with two trailing bools; the 4th element is undocumented in
  the module. Document the real shape once (types.ContactCandidate is the likely
  source of truth). S.
- KEEP docs-over-types cluster (merges L163, L432 MISREAD, L452, L474, L525
  MISREAD, L534): six metric builders return bare `dict`/`dict[str, dict]` with
  stable keys; type the metric records. M.
- KEEP repetition L298 + seeded deep-nest L309: strict_contact_rows appends three
  7-key row dicts repeating the same prefix; sibling wide_edge_contact_rows
  already uses the `common` base + spread idiom — align, which also flattens the
  seeded nesting. S.
- KEEP style-house magic literals L347 (90 base-30 = 3 s wide-edge half-width)
  and L270 default ``(5, 10)`` strict tolerances: name both. S. (Guiding focus.)
- DETAIL docs-orientation L479 + probe L488: 30-line _raw_precision_curve
  docstring is valuable rationale but buries the contract; tighten and define
  the "real input" disjoint-span precondition plainly. S.
- DETAIL probe L131: COVERED also requires each stroke in exactly ONE span;
  enum comment doesn't cover the overlapping-span case. One comment line. S.
- DETAIL style-house L569: single-line double-for with .get; borderline house
  rule; explicit loop if touched. S.
- OK seeded mega-module 616: clean sections, cohesive scoring story; splitting
  forces file-hops. Carve-out applies.
- KILL probe L288 (docstring literally states the split/missed row behaviour),
  probe L260 (cross-file scaling rule, named in docstring), scanner type-hint
  hallucinations L83/L88/L138 (luna caught), L152/L185 shape notes (clear),
  L15 DEFAULT_TOLERANCES grounding (named constant suffices locally; the x3
  duplication is the cross entry).
- CROSS anchors confirmed: _prf L432-449 is the natural home for the safe-F1
  formula written x3 (selection.f1_raw_5, gt_scoring); DEFAULT_TOLERANCES L15
  one of three (1,2,5,10) tuples.

## src/annotator/calibration/sweep.py (673 lines; verdicts 15C/11K, probe 3;
   seeded mega-module 673, deep-nest L175 + L486)

- KEEP structure M — _row_for_result L330-381 (merges no-docstring + cohesion
  verdicts, L342 magic 0.005, L364 pieces comprehension, seeded L486 partially,
  probe L336): 52 lines assembling the whole report row with a 12-key dict
  literal crammed across 4 continuation lines, an unnamed 0.005 for
  min_contact_speed, and a double-nested span-pieces comprehension. NOTE the
  0.005 is a hand-mirrored config value in the report row — same trap class as
  the e2e manifest cross finding (CSV lies if the real constant moves). Fix:
  docstring, grouped assembly, named constant sourced from the real config,
  unpack pieces. M.
- KEEP naming/structure S — _settings L263-270 (merges unclear-name, killed-on-
  quote docstring claim, probe L263, L269 one-liner): it builds the sort-key
  tuple behind row["settings"]/selection.standard_tail with an undocumented
  presence/rank contract and a 3-decision append line. Rename _settings_sort_key,
  docstring, unpack. S.
- KEEP repetition S — L467 + L495 (merges both verdicts): CandidateSpec-from-row
  dict comprehension twice, and the same_winner_as_live one-liner does None-check
  + construction + two serialisations + comparison. _spec_from_row helper +
  named intermediates. S.
- DETAIL misleading-name L521 (_withheld prints stdout+stderr; rename). S.
- DETAIL docs-prose: L86 "pinned" rules, L258 "closed routing validation",
  L569 "closed boundary spec", L131 identity shape. All S one-liners.
- DETAIL repetition L154/L177: failure-print duplicated across sequential and
  pool paths. S.
- OK seeded mega-module 673: orchestrator with clear sections; the ~120-line
  winner-validation block is the only optional split.
- KILL probe L88 (delegation to selection is design, rule names orient);
  seeded deep-nest L175 (idiomatic pool loop); _validate_provenance docstring
  (loud errors carry contract — consistent with dead_mask/fixtures standard);
  luna's 8 short-helper docstring kills all AGREED.
- CROSS anchors: TOLERANCES L49; LABEL_SHIPPED L45 vs selection.GRID_LABEL;
  private import _OVERRIDABLE_BASE30_ROWS L39.

## src/annotator/calibration/gt_scoring.py (791 lines; verdicts a+b 23C/2K/5M,
   probe 2; seeded mega-module 791, mega-function score_video 119, deep-nest L576)

- KEEP L (top-tier) — score_video L527-645 (merges seeded mega-function +
  deep-nest, no-docstring verdict, seven-accumulator data-structure + repetition
  verdicts, L587 single-item loop, L631 mean_err one-liner, L632 27-positional
  RallyRow MISREAD): seven parallel [0,0,0,0] lists with an implicit
  primary-correct/primary-total/covered-correct/covered-total slot convention,
  each updated under different eligibility rules; one metric inexplicably wrapped
  in a single-element for loop; a 27-arg positional constructor across 6 lines
  with inline ternaries. Cohesive-algorithm carve-out does NOT excuse the
  liftable accumulator mechanics. Fix: docstring + small 4-slot accumulator
  record with add(ok, covered) + keyword RallyRow construction. L.
- KEEP M — record/loader docstring cluster (merges ~9 verdicts: GtWinner L275
  x2, Reconciliation L285, RallyRow L295, ColumnAgg L325, VideoScoring L332,
  load_fixture_arrays L380, load_gt_tables L390, reconcile_sets L460,
  flatten_metrics L648): zero docstrings on the module's five record types (27-
  and 20-field) and its loaders; ColumnAgg's primary=all vs secondary=covered
  distinction only discoverable by tracing. M.
- KEEP S — canonical_tolerance L372 (verdict + cross): hand-rolls the base-30
  half-up scaling that ScalingKind.FRAME_COUNT.scale owns; single-source
  violation on the scaling rule. Reuse + max(1,...) wrapper if needed. S.
- KEEP S — L435 LandingFilterOptions(7, 0.004, 5, 7, 0.75): five positional
  unexplained literals; keyword args + grounding comment. S. (Related: sweep's
  mirrored 0.005.)
- DETAIL naming: ab/A-B player labels undefined (L276/L281), sm_side L463,
  _norm_half L376 hides the any-non-Top->Bot coercion (comment the policy),
  _literal L748 rename. S each.
- DETAIL structure: flatten_metrics L666 6-pair update line + L672; split. S.
- DETAIL mega-module 791: REFERENCE_SCORES L33-250 is 28% of the module; an
  optional reference_scores.py data module drops it under 600. Optional.
- CODENAMES: "W2.9", "eee3e29", "Opus-checkpoint artefact" L31-32 join the
  repo-wide codename entry.
- CROSS confirmed at read: contact-gate recompute + reconciling assert L638
  (share score_contacts' count_gate instead); safe-F1 triplication L671;
  private import _read_homography_rows L22; ball_round vs count_gate naming.
- KILL probe L621/L608 (point_winner-owned conventions; cross-file), 'flipped'
  L491 (luna right), _literal abstraction kill (luna right).

## src/annotator/types.py (195 lines; verdicts 10C/2K/1M, probe 1)

- KEEP docs-orientation L185-195 (merges 5 verdicts): StickyResult's 8 bare
  np.ndarray fields carry one line of doc ("bbox_height is in pixels"); picks/
  distances_per_slot/wrist_dist_px/analysed semantics and shapes undiscoverable
  locally despite being the shared cache record for two stages. Field docs +
  shape notes. S-M.
- DETAIL: ReentryGuardVariant L75 (high-shot gap / re-entry buffer undefined —
  pointer to rally_segmentation), ServeStartConfig L93 ("sticky lane" jargon). S.
- CLUSTER (migration note): "until the threading stage rewires" L26 + config.py
  "constants is deliberately unwired until threading" L159-160 — future-work
  reference with no pointer; name/date the plan or trim. Joins codename entry.
- KILL Slot L104 MISREAD (first docstring line names sticky_anchor), module
  docstring L1 (covered by the cross stage-index entry), L159 "drops the pad"
  wording (return note resolves), COCO/ordering kills (agree).
- Functions compute_speed/true_runs/rolling_nanmedian are exemplary shape-
  annotated house style — positive baseline worth naming in report.

## src/annotator/config.py (173 lines; verdicts 10C, probe 3)

- FOLD into repo-wide codename entry: W2.9 (L125), "ruled 2026-07-07" (L106),
  "block-2 sweep pick" (L52), sset_01/comp_content27_v0p5 (L96-97), "three-arm
  remeasure" (L134), "decontaminated baseline" (L89), B5 (L86). The provenance-
  citation convention is deliberate and valuable; the fix is a glossary or
  first-use definitions, not deletion.
- DETAIL probe L147: __post_init__ invariant (reentry guard requires
  gap_state_demotion_bound) has no why-comment; one line. S.
- KILL L25 env-lookup one-liner (conventional idiom; luna hedged), L84
  "reprojected-corner displacement" (precise CV vocabulary, measurement
  defined; rubric keeps precise jargon).
- CROSS anchor: stage-number index gap (cross finding quotes L31 here).

## src/annotator/fps_constants.py (108 lines; verdicts 9C/1K/1M, probe 3)

- KEEP S — FpsConstants field-gloss cluster (merges 5 verdicts: end_rest_frames,
  min_descend_samples MISREAD samples-vs-frames, body_unit_half_window,
  impulse_floor_half_window_frames, blip_max_frames): the 24-field base-30 table
  is the package's tuning surface; ~6 fields use algorithm jargon with no gloss
  while others (court_absent_window twins) are exemplarily commented. One-line
  comments for the opaque fields. S.
- CROSS (major, audit-established): the half-up frame-count scaling rule exists
  in THREE places — types.ScalingKind.FRAME_COUNT.scale (types.py L40),
  _time here (L56), gt_scoring.canonical_tolerance (L372). types.py documents
  the first two as deliberately identical pending the "threading" migration;
  gt_scoring's copy is undocumented drift risk. Also speed() L68 duplicates
  ScalingKind.PER_FRAME_SPEED. Unify or at least point all copies at one
  declaration. S-M.
- DETAIL: _time L55 rename (_base30_to_frames) + one-line docstring; missing
  :param: tags on scale_for_fps/probe_fps one-liners (overrides_base30 contract
  unexplained; optional half-line on ffprobe rational strings); 1e-6 CFR
  epsilon unnamed L106. S each.
- KILL probe L3 (contract wording, not singleton claim), probe L101/L106
  (ffprobe external knowledge; error messages orient).
- RESOLVES resolve.py PENDING: base-30 is canonically defined in this module's
  docstring; resolve.py needs only a pointer.

## src/annotator/composition_mask.py (174 lines; verdicts 1C, probe 0)

- DETAIL docs-prose L105-107: one 3-clause sentence; split. S. Probe clean;
  cleanest module in the annotator package. (Judged on verified quote without a
  full re-read — prose-only finding.)

## src/annotator/doubles_flag.py (180 lines; verdicts 1C/1K, probe 0)

- DETAIL data-structure L104 (+ killed dup L95, no loss — same claim): the
  3-key verdict row dicts feeding the fixed CSV header could be a TypedDict;
  docstrings already state the keys. S. Probe clean.

## src/annotator/replay_mask.py (350 lines; verdicts 4C, probe 0)

- DETAIL docs-prose L177: multi-clause signal sentence; split. S.
- DETAIL docs-orientation L179: :param track: gives (t, 3) but not the
  [x, y, visibility] column meanings (compute_speed in types.py documents them;
  one clause here). S.
- DETAIL style-house L286: multi-line comprehension with method call in
  iterable+predicate; house rule says explicit loop. S.
- KILL needless-abstraction L293 (_cli_non_evidence): named CLI policy boundary,
  consistent with run_cli._validate_environment kill; luna hedged too.
- PROBE clean. CROSS anchors: _read_homography_rows privately imported by
  gt_scoring; filter_short_exclusion_runs imported by scraper stage11 (fine,
  public); replay CLI writes <id>_replay.npy — mask-filename mismatch checked
  at rally_segmentation.

## src/annotator/inpaint_guard.py (295 lines; verdicts 11C/3K/2M, probe 2)

- KEEP S — grade-code counting (merges L230 repetition + L121 range(4) x5
  magic): the module's own code_counts() helper is re-inlined at both build_mask
  exits, and range(4) stands in for the four grade codes five times. Call
  code_counts + name the code set (len(CODE_NAMES) or GRADE_CODES). Reuse-of-
  canonical-helper miss — core-priority class. S.
- KEEP S — halo construction L240-247 (merges L242 one-liner + probe L242):
  the concat/astype/diff edge idiom re-implements types.true_runs (the package's
  canonical run-finder); rewriting with true_runs(core) kills the one-liner and
  gives (start, end) pairs directly. Add why-comment for the window-1 halo
  width (frames sharing a window with an attractor frame). S.
- KEEP S — _validate_presence L187-202: four near-identical overlap sums; a
  _count_overlapping(groups, half) helper x4 also exposes that two of the
  written bounds (start+window>0, start<n_frames) are vacuous. S.
- DETAIL — private-helper docs cluster (merges L109 _empty_info, L125
  _candidate_attractors no-docstring + 5-tuple return + MISREAD, L206 _cover
  x2, L166 MISREAD, L109 data-structure info-dict): stable diagnostics dict
  built as literals in two places with dict[str, Any] annotation; docstrings +
  one construction site (or small record). S-M.
- DETAIL probe L55: (0, 0) no-detection sentinel never stated; one comment. S.
- DETAIL L150 moves: np.any(np.ptp(points, axis=0) > 0) per house vectorised
  style. S.
- DETAIL L40: 121-char signature; wrap. S.
- KILL agree: L127 line-length (117), L149 shape visible, L73 floor (docstring
  grounds it).

## src/annotator/experiment_records.py (358 lines; verdicts 12C/4K, probe 1,
   incidental 1)

- KEEP S-M — security-pipeline docs cluster (merges 7 verdicts: _regular_run_file
  L66, _candidate_files L175, _private_tokens L183, _planned_json_changes L204,
  _backup L243, _scanner_findings L259, clean_run scope probe L299): the module
  performs destructive deletion + sanitisation, yet its 30-60-line scanning/
  deletion helpers carry no docstrings; the deletion triggers (credentials via
  betterleaks, private-path regex, manifest tokens) are only discoverable by
  tracing. Boundary docstrings. S-M.
- KEEP S — incidental L221-223 (audit-verified): the sanitisation planner
  silently skips malformed configuration records; build_summary raises on the
  identical predicate. Mitigation exists (the rg pass still catches leaked
  paths, but by DELETING the file instead of sanitising it); the skip should
  log or raise. S.
- DETAIL: magic 8 configurations L94 (name or ground the count); "closed
  measurement manifests" L75 + count_ignored_npy L38 docstring wording. S.
- KILL: _number/_metric names (luna right), count_ignored_npy name (annotation
  clear), _write_json single-caller (2-line named intent, consistent standard),
  sanitise-comprehension repetition L217/L228 and validation-predicate
  repetition L79/L222 (differing raise-vs-skip policies ARE the point; helper
  would encode no shared rule — the incidental covers the policy question).
- KILL cross md5 finding (low-confidence cross item): one-line stdlib idiom;
  a shared helper adds a hop for nothing.

## src/annotator/court_evidence.py (684 lines; verdicts 13C/7K/1M, probe 3;
   seeded mega-module 684, deep-nest L222, mega-function
   build_detected_court_evidence 115)

- KEEP S-M — policy-constant cluster (merges L545/L599 0.5 scene-valid, L390
  +/-0.10 margin x3 merged, L194 min(10,...) sample cap; pairs with the e2e
  manifest cross finding): the three court-evidence policy values are inline
  literals here AND hand-mirrored as manifest metadata in e2e_court_annotator
  (PERSON_MARGIN/SCENE_THRESHOLD/COURT_SAMPLES). Name them at module scope,
  have e2e read them. Kills magic numbers + the manifest-drift trap at once.
- KEEP S — corner-order contract L343 (merges verdict x3-merged + probe; plus
  MISREAD L332 identity conversion + _scene_row docstring L326): the
  TL/TR/BR/BL -> upleft/upright/downLEFT=3/downRIGHT=2 swap looks like a bug
  and carries the sticky/replay column contract; the identity _as_ref_corners
  call invites a phantom conversion reading. Module-scope named mapping +
  comment + drop/name the identity step. S.
- KEEP S — "parent" vocabulary L43 (merged x3): the module's central concept
  (static ShuttleSet prior vs CourtKeyNet detector as court-evidence source)
  never defined; one docstring sentence. S.
- KEEP M — mega-function build_detected_court_evidence L567-684 (seeded):
  raw-evidence phase vs consensus phase already separated by the
  CourtConsensusError boundary; split there or extract the record loop. M.
- CROSS: ref_err_px=3.5 default x3 here + run_video + e2e REF_ERR_PX — one
  named constant. S.
- DETAIL: cryptic one-line docstrings L190 ("centred-bin"), L372, L404
  ("exactly-two majority to the sole court vector"); L506 quad-field
  conditional x5; L209 corner_floor rename + untyped detector seam (Protocol
  or comment); L311 _gate_resolution_table docstring. S each.
- KILL: seeded deep-nest L222 (sequential-read sampling loop, inherent);
  all 7 luna kills AGREED (incl. the sharp two-caller catch on
  _copy_court_info).

## src/annotator/point_winner.py (871 lines; verdicts 10C/4K/4M, probe 1,
   incidental 1; seeded mega-module 871, deep-nest L519)

- KEEP S — sticky/shipped vocabulary (merges L132, L320 MISREAD, L783 MISREAD
  "ported"): the module leans on "sticky tracker/pick" (its central input) and
  "shipped" without one defining line or pointer to sticky_anchor. One docstring
  sentence + drop the history verbs. S.
- MERGE into court_evidence corner-order entry: probe L668 — the same
  [0, 1, 3, 2] get_corner_camera reorder, again undocumented at the second
  site. Fix once, reference from both.
- MERGE into codename cluster: module docstring D5 / sset_01 / GT-anchored /
  SHIPPED (L1-7).
- DETAIL AUDIT-FOUND L409-412: the "Same machinery on the nearer ANKLE" comment
  is mis-indented one level OUT of the `for half` block it sits inside,
  visually splitting the loop body; sharper issue than the killed repetition
  claim it accompanied. Fix indent; optional _nearest_joint_ratio helper for
  the wrist/ankle pair. S.
- DETAIL prose group: noun-pile opening sentence L1; dense edge-case comments
  L229/L437 (valuable rationale, split sentences); docstring gaps L696 params,
  L723 "M=0", L637 TL/TR/BR/BL expansion, L366 base-30 pointer. S each.
- DETAIL seeded deep-nest L519: window-scan nesting; flatten with early
  continue only if touched. S.
- KILL incidental L408 (NaN->0.0): explained-quiet-path exception — the
  determinism comment sits two lines above the quoted line.
- KILL repetition L608/L741 (one-line y-bounds conditional x2; no shared rule),
  L650 one-liner (idiomatic metre-scaled hypot), luna's 4 kills AGREED.
- OK seeded mega-module 871: single-domain, cohesive; carve-out applies.

## src/annotator/run_video.py (649 lines; verdicts 10C/6K/2M, probe 2; seeded
   mega-module 649, mega-function run_video 419 CODE LINES, deep-nest L594)

- KEEP L (TOP of hit-list) — run_video() L199-649 (merges seeded mega-function +
  deep-nest, MISREAD L232 no-param-contract, style-house type gaps L22/L94/L187/
  L199, probe+MISREAD L327 homography_rows duck-typing, L568 shipped_winner):
  a 31-parameter entry point (5 untyped positionals, several untyped keywords)
  with an 8-line docstring covering three of them, followed by a 419-code-line
  staged body. This is a staged COMPOSITION (segmentation -> serve evidence ->
  sticky -> contacts -> attribution -> verdict -> landing -> hit height), not a
  cohesive algorithm — the carve-out does not apply. Fix: typed signature,
  grouped :param: contract, stage-extracted helpers, document the
  homography_rows accepted forms, rename shipped_winner. L. Audit-confirmed by
  signature read.
- KEEP S — module docstring L1: the pipeline heart carries a ONE-line docstring
  ("GT-free annotation-chain composition for one video"); 2-3 orienting
  sentences naming the stages and outputs. S.
- MERGE into codename cluster: sset_21/decontamination-commit history L97,
  "Brief H" + em-dash L190-194.
- DETAIL probe L28: rejected_grades meaning lives in inpaint_guard.CODE_NAMES;
  pointer comment. S.
- KILL L196 parity one-liner (docstring explains; within reason); agree all 6
  luna kills (capture-guard lifecycle rationale was a good one).

## src/annotator/e2e_court_annotator.py (1340 lines; verdicts a+b 24C/18K/3M,
   probe 3; seeded mega-module OVER ceiling)

- KEEP M — orchestration docstring cluster (merges 6 verdicts:
  _run_one_configuration 98 code lines L916, _setup L1136, _configuration_manifest
  L851, _write_scoring_outputs L809, _score_configurations L1201, _load_case
  L752): the runner's six orchestration stages have no docstrings. M.
- KEEP S-M — SHARED_FILES monkey-patch block L1145-1152 (merges probe L1148/
  L1152, MISREAD x2, [:3] magic): temporarily mutates gt_scoring's module
  global with a bare NOT-THREADSAFE note, an unexplained 3-pin slice, historical
  "existing helper" wording, and a phantom "no-GT path" comment for a case that
  does not exist here. Fix: name the collaborators + selection; better, replace
  the global mutation with a parameter on load_gt_tables. RESOLVES fixtures.py
  SHARED_FILES-consumer PENDING. S-M.
- KEEP S-M — positional-alignment traps (merges MISREAD L533 _scene_row
  dict(zip(cols, 27 positional values)), _horizon_row L560 + docs L522/L560,
  fixture.files magic indices L757 (silently skips [4]=kp_scores), CaseData
  empty-shape positionals L1177): explicit key/value construction + named
  indices via FixtureDigests order. S-M.
- KEEP S — role-order sort lambda L878-884 (+99 magic): 11-role list duplicated
  inside one lambda; module ROLE_ORDER constant + index map. S.
- MERGE: LANDING_OPTIONS positional literals L74 into gt_scoring L435 entry
  (two positional LandingFilterOptions construction sites); SCENE_ROW_COLUMNS
  retype L788, _configuration_values manifest literals L638, REF_ERR_PX,
  _scene_row name collision — all cross entries, audit-confirmed.
- KEEP M-L (optional) — seeded mega-module 1340: past the flexible ceiling with
  clear seams (pin/manifest layer, row adapters, configuration runner,
  setup/scoring/CLI). Split optional per minimal-implementation ethos; the
  docstring entries above are the cheap 80%.
- DETAIL: RunDriver Any-typed fields L209-212 (RESURRECTED x4 from trailing-
  comma quote kills; type them with gt_scoring's real types);
  _write_scene_evidence_partial 3-write overlap L1015; docs-prose L1299;
  planning-language module docstring (quote-gate killed; joins migration-note
  cluster); three one-line single-caller wrappers (L321/L440/L675) — inline if
  touched. S each.
- KILL: exception-handler dup L958 (stage-specific payloads; quote-gate kill,
  substance also weak), lambda-density L878 dup (covered by role-list entry),
  L1332 argv normalisation, _load_array/_write_annotations/_gt_rallies kills
  AGREED (good boundary rationale), main() docstring, _csv_value, (512,288)
  quote-kill (substance covered by cross DETECTOR_RESOLUTION entry),
  "BaseAnnotatorConfig()" label, _json_ready branches.

## src/annotator/rally_segmentation.py (1586 lines; verdicts a+b 23C/12K/1M,
   probe 3, incidental 1; seeded mega-module 1586 WAY over, deep-nest L845 +
   L1173, mega-function main 134)

- KEEP M — gap-state/serve-lane docstring cluster (merges 7 verdicts L402/L416/
  L436/L255/L670/L583/L296 + probe L743): the stage's hardest algorithmic
  helpers (_gap_is_high_shot_oob, _gap_passes_reentry_guard,
  _gap_state_rest_mask, serve_setup_still, _sticky_serve_setup_before,
  _find_rally_spans_quiet_start, build_serve_setup_inputs) have terse or no
  docstrings; the serve-lane TODO L743 invokes undefined historical semantics.
  Docstring pass. M.
- KEEP M — public-surface contracts (merges build_sticky_result L1127 10-param/
  3-line-doc, find_rally_spans L1235 valid-combination gaps, main L1435 MISREAD,
  tracker_segments L1071 untyped, probe L1270 assemble_contacts conditional
  gating undocumented): document parameter combinations + type
  tracker_segments. M.
- KEEP M-L — seeded mega-module 1586: strongest split candidate in the repo;
  clear seams at serve-setup lane (~L225-760), contact detection (~L900-1290),
  CLI/batch (~L1400-1586). Fold seeded deep-nests + main-134 here. M-L.
- KEEP cross S (verified at read): _load_replay_mask L1397-1409 is NAMED replay
  but loads `<id>_dead_mask.npy` and logs "no dead mask"; replay_mask.py's CLI
  writes `<id>_replay.npy` (which stage11 reads) — a replay-CLI output is
  silently invisible here (info-log only). One shared suffix constant + rename
  helper (_load_dead_mask). S.
- DETAIL: quiet_burst generator L592 (house rule, dup slice); rolling-mean
  scaffold dup L126; :param gaps L225/L623/L1412/L1427; chained ternary L1417;
  docs-prose L334 double negative, L1061 "base-30 nine", L1307 param history;
  segment_video 30-line docstring L1299 (verifier hedged; trim caveats).
  S each. NOTE: 5 more luna-b quote-gate kills (L1266, L1274, L1357, L1474,
  L1040) had plausible substance killed on indentation; L1274 triple ternary
  and L1266 comprehension would have been DETAILs anyway — no material loss.
- KILL: detect_contact_flags wrapper L1006 (documented test seam), probe L1162
  (sticky_anchor external contract), incidental L1535 (luna already resolved:
  commented intentional batch isolation), constants-scatter ordering kill
  AGREED, junction comment kill AGREED (good shape-grounding defence).
