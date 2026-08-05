# Readability review evidence report: `src/annotator` and `src/scraper`

Date: 2026-08-04. Report-only; no code was changed.

This is the detailed record of the review. Start with
[`readability_refactor_scoping_handoff.md`](readability_refactor_scoping_handoff.md)
for the current evaluation, priorities and scoping advice. The handoff also
records later corrections to proposed fixes in this report.

## Scope and coverage

- Target: 30 source modules (23 annotator incl. calibration, 7 scraper), 12,826
  lines. Excluded by ruling: `src/annotator/validation_overlay/` (helper tool),
  `__init__.py` files, and tests (deferred, not pre-authorised).
- Mechanical pass: 30 discovered, 30 parsed, 0 failures. 26 seeded findings
  (9 mega-module, 5 mega-function, 12 deep-nesting; 0 unused-private).
- Scan swarm: 129/129 jobs completed, 0 failed, 0 skipped (96 Haiku-low lens
  scans over 32 chunks, 30 cold-read probes — 28 Haiku low, 2 Sonnet low for the
  chunked giants — and 3 Sonnet cross-module scans grouped annotator-core /
  calibration / scraper).
- Verification + probe assessment: 33/33 codex `gpt-5.6-luna` effort-max calls
  (merged seat; three heavy modules split A/B), all verdicts parsed.
  ~2.66M codex tokens (~2.7% of the weekly quota).
- Pipeline deviations from the skill defaults, all user-ruled: no failure-
  handling lens; luna replaced both the Sonnet-verifier and Opus-assessor seats
  in one merged call per module.

## Audit levels

Every surviving finding passed luna's check (full module read at max effort +
verbatim-quote gate, demonstrably strict). The orchestrator's own source read
on top of that varied by module, and each hit-list entry carries a tag:

- **A1** — orchestrator read the full module and the finding site first-hand,
  on top of luna's verification. Strongest tier.
- **A2** — orchestrator read the finding site and surrounding region directly,
  but not the whole module.
- **A3** — claim rests on luna's ground-truth read and its verbatim in-file
  quote; the orchestrator did not open the source at that site. The A3 claims
  are deliberately low-interpretation ones (missing docstrings, parameter
  counts, quoted code), but they carry one fewer pair of eyes.

Orchestrator read depth by module — full read (A1 baseline): all 7 scraper
modules; batch_report, dead_mask, resolve, types, config, fps_constants,
inpaint_guard, court_evidence; calibration run_cli, selection, fixtures,
scoring, sweep, gt_scoring. Targeted read (A2 baseline): schemas (L60-190),
experiment_records (L196-336), point_winner (L380-455), run_video (L199-263),
rally_segmentation (L1390-1445). Quote-only (A3 baseline): composition_mask,
doubles_flag, replay_mask (prose-level findings only), e2e_court_annotator.
Per-file detail items inherit their module's baseline unless noted.

## Finding funnel

- Raised: 458 lens findings + 28 cross-module + 26 mechanical seeds.
- Luna verification (after duplicate merging): 275 CONFIRMED, 28
  MISREAD_BUT_TELLING (reclassified comprehension traps), 95 KILLED. These
  statuses account for all 398 verdict rows.
  Assessors added 58 probe-derived comprehension findings and 8 incidental
  silent-failure notes.
- Orchestrator audit: survivors consolidated into 35 hit-list entries and ~55
  per-file detail items; ~45 further findings killed at audit (probe-model
  errors, external-contract knowledge, house anti-abstraction rules). 6
  findings luna killed on verbatim-quote technicalities were resurrected on
  substance (see funnel note at the end). The orchestrator's own read depth
  varied by module — every entry carries an audit-level tag; see "Audit
  levels" below.

## Refactor hit-list

Ranked by onboarding pain against fix cost. Effort: S (< half hour), M (an
hour or two), L (a session).

### Tier 1 — the big two functions

| # | Site | Category | Problem | Fix sketch | Effort | Audit |
|---|------|----------|---------|-----------|--------|-------|
| 1 | `src/annotator/run_video.py:199` | mega-function | `run_video()`: 31 parameters (5 untyped positionals, several untyped keywords), an 8-line docstring covering three of them, then a 419-code-line staged body. It is a stage composition, not one algorithm — the cohesive-algorithm carve-out does not apply. | Typed signature; grouped `:param:` contract; extract per-stage helpers; document the `homography_rows` dual input forms; rename `shipped_winner`. | L | A2 (signature + docstring read first-hand; body size is mechanical fact) |
| 2 | `src/annotator/calibration/gt_scoring.py:527` | data-structure / mega-function | `score_video()` (119 code lines): seven parallel `[0, 0, 0, 0]` accumulator lists with an implicit slot convention, one metric arbitrarily wrapped in a single-item `for` loop, and a 27-argument positional `RallyRow` constructor with inline ternaries. | Small 4-slot accumulator record with `add(ok, covered)`; keyword `RallyRow` construction; docstring. | L | A1 |

### Tier 2 — cross-cutting single-source-of-truth repairs

| # | Site | Category | Problem | Fix sketch | Effort | Audit |
|---|------|----------|---------|-----------|--------|-------|
| 3 | repo-wide (anchor `src/scraper/config.py:145`) | docs-orientation | Project-history codenames appear in code comments with no definition anywhere a reader can find: B5/B6, W2.9, Wave 1/3, D-numbers, s29, sset_01, "Brief H", "block-2", "three-arm remeasure", the pending "threading" migration. The provenance-citation habit is good; the labels are opaque. | One short glossary (docs or a package docstring) plus first-use pointers; date/name the "threading" plan or trim those notes. | M | A1 (sites confirmed across full-read modules; e2e/rally sites A3) |
| 4 | `src/annotator/fps_constants.py:56` | repetition | The base-30 half-up scaling rule exists three times: `types.ScalingKind.FRAME_COUNT.scale`, `fps_constants._time`, and `gt_scoring.canonical_tolerance:372` (plus `speed()` duplicating `PER_FRAME_SPEED`). The first two are documented as deliberately identical pending migration; gt_scoring's copy is undocumented drift risk. | Point all copies at one declaration now; make gt_scoring call `ScalingKind.FRAME_COUNT.scale`. | S-M | A1 (all three formulas read first-hand) |
| 5 | `src/annotator/court_evidence.py:545` | style-house / repetition | The court-evidence policy values (0.5 scene-valid fraction, the ±0.10 person margin, `min(10, …)` sample cap) are inline literals here AND hand-mirrored as manifest metadata in `e2e_court_annotator._configuration_values:638` — the run manifest lies if either side moves. Sweep's `min_contact_speed: 0.005` at `sweep.py:342` is the same trap. | Name the constants at module scope; have e2e and sweep read them. | S-M | A1 court + sweep sides; A3 e2e mirror (Sonnet cross quote) |
| 6 | `src/annotator/calibration/sweep.py:49` | repetition | The `(1, 2, 5, 10)` tolerance tuple is defined independently three times (`sweep.TOLERANCES`, `scoring.DEFAULT_TOLERANCES`, `schemas.WINNER_JSON_TOLERANCES_BASE30`), and `load_boundary_winner` validates persisted documents against its local copy. | One definition; the others import it. | S | A1 (all three sites read) |
| 7 | `src/annotator/calibration/scoring.py:441` | repetition | The safe-F1 formula (0.0 when precision+recall is zero) is written out in `scoring._prf`, `selection.f1_raw_5`, and `gt_scoring.flatten_metrics:671`. | Extract `safe_f1(precision, recall)` in scoring.py; call it from the other two. | S | A1 (all three sites read) |
| 8 | `src/annotator/rally_segmentation.py:1397` | cohesion-coupling | `_load_replay_mask` is NAMED replay but loads `<id>_dead_mask.npy` and logs "no dead mask"; `replay_mask.py`'s CLI writes `<id>_replay.npy` (the name stage 11 reads). A replay-CLI output is silently invisible to this loader — the video just "runs unmasked". | One shared mask-suffix constant in config; rename the helper `_load_dead_mask`; document which producer feeds which consumer. | S | A2 (loader + stage11 reader read first-hand; replay-CLI writer name from cross quote) |
| 9 | `src/annotator/calibration/gt_scoring.py:638` | repetition | Re-derives the per-rally count gate that `score_contacts` already returns, then reconciles the two with a runtime assert; also the only module calling the metric `ball_round` (everywhere else: `count_gate`). | Read `count_gate` from `score_contacts`; drop the assert; align the name. | S | A1 (both ends read) |
| 10 | `src/scraper/config.py` (package) | repetition | Scraper dedup cluster: yt-dlp subprocess run/timeout/log pattern ×6 (stages 1, 2), LLM retry loop copied line-for-line (stage3 ↔ stage10), `VIDEO_EXTENSIONS` ×3, `keep == 'True'` filter ×3, `_check_ytdlp` duplicating `config.check_ytdlp`, `_download_one` partially rebuilding `config.ytdlp_throttle_args()` (whose docstring assigns those flags to exactly that path), stage11 re-validating sources.toml with a looser copy of the downloader's validator. | `run_ytdlp()`, `retry_llm_call()`, `VIDEO_EXTENSIONS`, `row_is_kept()` and one manifest reader, all in `config.py`. | M | A1 (every site read; all 7 scraper modules read in full) |
| 11 | `src/scraper/stage3_triage.py:249` | cohesion-coupling | The "block the run on mass failure" circuit-breaker is implemented four different ways across stages 1/2/3/10 (different floors, shapes, and one with no mid-batch check). | Either one config-level helper the stages call, or a docstring line per stage stating the divergence is deliberate. | M | A1 (all four implementations read) |
| 12 | `src/annotator/resolve.py:14` | repetition | `_OVERRIDABLE_BASE30_ROWS` hand-copies `FpsConstants` field names + one threshold field; a new fps-scaled field silently fails as "unknown override". | Build the frozenset from `FpsConstants._fields` + explicit extras. **Correction 2026-08-04:** `FpsConstants` is a frozen dataclass, not a NamedTuple, so `._fields` does not exist; use `dataclasses.fields(FpsConstants)`. Found by the sol-high audit of the refactor handoff; orchestrator-verified at fps_constants.py:22. | S | A1 (both files read) |

### Tier 3 — module-level structure and contracts

| # | Site | Category | Problem | Fix sketch | Effort | Audit |
|---|------|----------|---------|-----------|--------|-------|
| 13 | `src/annotator/rally_segmentation.py:1` | mega-module | 1,586 lines, well past the flexible ceiling, with clear seams: serve-setup lane (~225-760), contact detection (~900-1290), CLI/batch (~1400-1586). Also carries the seeded deep-nests and the 134-line `main`. **Correction 2026-08-04:** a luna-max seam map (scratch/swarm_review/handoff/rally_seg_seam_map.md) corrected the boundaries (serve lane 161-862 plus its facade at 1228-1256; contact block 915-1068 plus facades; CLI 1373-1586) and found the three-way split is NOT clean: shared smoothing helpers, the dual-use sticky-evidence block (1071-1225), and the `segment_video` facade (1283-1367) all cross the seams. Only the CLI extraction is a verified clean cut. | Split at the seams (three modules), or at minimum extract the CLI. | M-L | A2 (size/seams from mechanical top-level order; CLI region read) |
| 14 | `src/annotator/rally_segmentation.py:402` | docs-orientation | The stage's hardest helpers (`_gap_is_high_shot_oob`, `_gap_passes_reentry_guard`, `_gap_state_rest_mask`, `serve_setup_still`, `_sticky_serve_setup_before`, `_find_rally_spans_quiet_start`, `build_serve_setup_inputs`) have terse or no docstrings; the L743 TODO invokes undefined historical semantics. | Docstring pass over the gap-state and serve-lane helpers. | M | A3 (luna quotes; low-interpretation missing-docstring claims) |
| 15 | `src/annotator/rally_segmentation.py:1127` | docs-orientation | Public-surface contracts: `build_sticky_result` (10 domain-heavy params, 3-line doc), `find_rally_spans` (undocumented valid parameter combinations), `tracker_segments` (fully untyped), `main` (no docstring), `assemble_contacts` (conditional gating only in the code). | Parameter contracts + types. | M | A3 (quoted signatures) |
| 16 | `src/annotator/e2e_court_annotator.py:916` | docs-orientation | The runner's six orchestration stages (`_run_one_configuration` 98 code lines, `_setup`, `_configuration_manifest`, `_write_scoring_outputs`, `_score_configurations`, `_load_case`) have no docstrings. Module is 1,340 lines with clear seams if a split is ever wanted. | Stage docstrings now; split optional. | M | A3 (quoted signatures; sizes mechanical) |
| 17 | `src/annotator/e2e_court_annotator.py:1145` | comprehension-trap | Setup temporarily mutates `gt_scoring.SHARED_FILES` (a module global) with a bare "NB NOT THREADSAFE", an unexplained `[:3]` slice, and a comment about a "no-GT path" that does not exist in this module. | Name the collaborators and the 3-pin selection; better, pass the pin subset as a parameter to `load_gt_tables`. | S-M | A3 e2e side (probe + luna quotes agree); A1 gt_scoring side |
| 18 | `src/annotator/e2e_court_annotator.py:533` | data-structure | Positional-alignment traps: `_scene_row` builds its CSV row by zipping a column tuple against 27 positional values; `fixture.files` indexed by magic positions (silently skipping `[4]`); `CaseData` built from positional empty-shape arrays. | Explicit key/value construction; named indices from the `FixtureDigests` order. | S-M | A3 (quoted code; `[4]`=kp_scores cross-checked against fixtures.py read) |
| 19 | `src/scraper/stage10_clean.py:248` | comprehension-trap | `run_clean`'s scoring bookkeeping: the `_score_pending` temp key is set, consumed-and-restored inside an `or` with a side-effect `pop`, re-read, and stripped across four sequential loops, threaded through positional 4-tuples. Probe, verifier, and audit all stumbled here. | NamedTuple for loaded sidecars; explicit pending-set instead of temp dict keys. | M | A1 |
| 20 | `src/scraper/download_scraped_videos.py:271` | mega-function | `_download_one` (111 code lines) interleaves the existing-file path, fresh download, and audio verification; the audio-probe error handling then duplicates against `_verify_existing`. | Split at the existing/fresh boundary; dedup follows while keeping the unlink-vs-retain policy explicit. | M | A1 |
| 21 | `src/scraper/download_scraped_videos.py:1` | docs-orientation | A 565-line module with a 3-line docstring whose first sentence is garbled ("Scraper downloads request and verify audio"); `main`'s "mass-failure status" (exit 2 at ≥50%) undefined; the `.f<digits>` intermediate-file exclusion unexplained. | Rewrite docstring; one line on exit codes; why-comment on the yt-dlp intermediates. | S | A1 |
| 22 | `src/annotator/calibration/sweep.py:330` | structure | `_row_for_result` (52 lines, no docstring) assembles the whole report row with a 12-key dict literal crammed across 4 continuation lines and a double-nested span-pieces comprehension. | Docstring; grouped assembly; unpack the pieces comprehension. | M | A1 |
| 23 | `src/annotator/calibration/scoring.py:432` | docs-over-types | Six metric builders return bare `dict` / `dict[str, dict]` with stable keys (`score_boundaries`, `_prf`, `_tolerance_curve`, `_raw_precision_curve`, `_count_gate`, `score_contacts`). | TypedDicts for the metric records. | M | A1 |
| 24 | `src/annotator/calibration/gt_scoring.py:275` | docs-orientation | The module's five record types (incl. a 27-field `RallyRow` and 20-field `VideoScoring`) and its loaders have zero docstrings; `ColumnAgg`'s primary-vs-covered distinction is only discoverable by tracing. | Field-group docstrings on the records; one-liners on the loaders. | M | A1 |
| 25 | `src/annotator/experiment_records.py:259` | docs-orientation | The destructive clean/sanitise pipeline's 30-60-line helpers carry no docstrings; the three deletion triggers are discoverable only by tracing. Related: the sanitisation planner silently skips malformed configuration records (L221) where `build_summary` raises on the identical predicate — the rg pass still catches leaked paths, but by deleting the file instead of sanitising it. | Boundary docstrings; log or raise on the malformed-record skip. | S-M | A2 (scan/sanitise region L196-336 read first-hand, incl. the skip and its rg fallback) |

### Tier 4 — small, high-value fixes

| # | Site | Category | Problem | Fix sketch | Effort | Audit |
|---|------|----------|---------|-----------|--------|-------|
| 26 | `src/annotator/calibration/selection.py:60` | misleading-name | Two winner-selection key names misdescribe their rule, and the probe conflated exactly as invited: `boundary_report_key_fewest_merges` ranks by `swallowed_rallies` (excess rallies, not merge count); `contact_live_key_floored_f1` ranks by RAW F1 (floors filter elsewhere). | Rename or one corrective docstring line each. | S | A1 |
| 27 | `src/annotator/calibration/selection.py:3` | docs-orientation | Module docstring names "B6" five times as its counterpart without identifying it (it is sweep.py), and packs a 16-field contract into one block; "boundary"/"contact" phase vocabulary undefined. | Name sweep.py; group the field list; one-line phase gloss. | S | A1 |
| 28 | `src/annotator/calibration/scoring.py:501` | stale-doc | Both contact docstrings document a 3-tuple `(rally_id, contact_frame, proximity_ok)` while the hints declare a 4-tuple with two undocumented trailing bools. | Document the real shape once (types.ContactCandidate is the source of truth). | S | A1 (resurrected from a quote-gate kill after direct read) |
| 29 | `src/annotator/court_evidence.py:343` | comprehension-trap | The TL/TR/BR/BL → `downleft=3, downright=2` corner swap looks like a bug and carries the sticky/replay column contract; the same `[0, 1, 3, 2]` reorder recurs undocumented in `point_winner:668`; `_scene_row` also makes an identity `_as_ref_corners` call that reads as a phantom conversion. | Module-scope named mapping with a both-orders comment; reference it from point_winner; drop or name the identity step. | S | A1 court side; A3 point_winner:668 site (probe + luna quote) |
| 30 | `src/annotator/court_evidence.py:43` | docs-orientation | "Parent" — the module's central concept (static ShuttleSet prior vs CourtKeyNet detector) — is never defined; plus `build_detected_court_evidence` (115 code lines) has a natural split at the existing `CourtConsensusError` boundary. | One defining sentence; optional split. | S / M | A1 |
| 31 | `src/annotator/dead_mask.py:45` | docs-orientation | `build_dead_mask`, the public 9-parameter dispatcher, has zero `:param:` docs and no shape annotations; which params each mode needs is only in the branches. | Per-mode parameter table in the docstring + shape notes. | S-M | A1 |
| 32 | `src/annotator/types.py:185` | docs-orientation | `StickyResult`'s 8 bare `np.ndarray` fields carry one line of doc; picks/distances_per_slot/wrist_dist_px/analysed semantics and shapes are undiscoverable, and it is the shared cache record for two stages. | Field docs with shapes. | S-M | A1 |
| 33 | `src/annotator/inpaint_guard.py:230` | repetition | The module re-inlines its own `code_counts()` helper at both `build_mask` exits, `range(4)` stands in for the grade-code set five times, and the halo edge-detection re-implements `types.true_runs`. | Call `code_counts`; name the code set; rewrite halo via `true_runs(core)`. | S | A1 |
| 34 | `src/scraper/stage1_index.py:101` | stale-doc | `enrich_row`'s ":return: True when a metadata call actually hit YouTube" reads as success-only; timeout/error/parse-fail branches also return True (meaning: request attempted, pace it). The probe misread it exactly this way. | Reword the return line. | S | A1 |
| 35 | `src/annotator/calibration/gt_scoring.py:435` | style-house | `LandingFilterOptions(7, 0.004, 5, 7, 0.75)` — five positional unexplained literals, constructed the same way again at `e2e_court_annotator:74`. | Keyword arguments + a grounding comment at both sites. | S | A1 gt_scoring side; A3 e2e:74 site (luna quote) |

Also on the scraper side, small but real (both A1): `stage11_pairing.py:261`
stale-doc ("rally ids" for video ids, S) and `config.py:5` stale-doc (docstring
claims stage 8/9 trajectory constants that are not in the module, S).

## Per-file detail (below hit-list threshold)

- `scraper/config.py`: semicolon-joined comment sentences L194/L229; dense
  model-budget comment L148-152 (fold into the codename glossary).
- `scraper/stage1_index.py`: unnamed log-truncation limits (200/120) at
  L72/L81/L117; `int(float(...))` why-comment L165; split the 3-op fallback
  L127.
- `scraper/stage2_transcripts.py`: WhisperX docstrings narrate missing-track
  only, but any caption-pull failure triggers the fallback (L64); optional
  TypedDicts for segment/{source,segments} records.
- `scraper/stage3_triage.py`: "enrichment" is stage-1 vocabulary used bare
  L176; phantom variable name `duration_min` in comment L189; `_write_keep_back`
  defined after its only caller L275.
- `scraper/stage10_clean.py`: `hallucination_silence_threshold=2.0` inline
  while sibling settings are named constants L338; sidecar-lookup duplication
  across the two passes; multi-line comprehension L374; name the fine-models
  tuple (NamedTuple) and clean-result dict.
- `scraper/stage11_pairing.py`: believed-replay vs duration-filtered vocabulary
  in one signature L97; 4-clause pairing-tie sentence L138; manifest validator
  docstring L240. Silent-failure notes (audit-confirmed): missing chunk sidecar
  L222 and missing replay mask L231 both proceed quietly even for
  commentary-eligible videos — a warn line each would surface them.
- `annotator/batch_report.py`: `_count_word` pluralises rather than counts L22;
  one orienting sentence for the three bare domain terms.
- `annotator/dead_mask.py`: keep_vote→votes rename after validation L74.
- `annotator/resolve.py`: rewrite module+function docstrings to say what
  resolve PRODUCES; positive phrasing L3; base-30 pointer to fps_constants.
- `annotator/fps_constants.py`: gloss the ~6 jargon fields in the base-30
  table; rename `_time`; `:param:` tags on the two public functions; name the
  1e-6 CFR epsilon.
- `annotator/config.py`: why-comment on the reentry-guard/demotion-bound
  invariant L147; the rest folds into the codename glossary.
- `annotator/schemas.py` (calibration): move CONTACT_FRONTIER/STABILITY next to
  CONTACT_SWEEP L87-100; drop "raw" before recall_5 L3.
- `annotator/calibration/run_cli.py`: document the injected-seam contract at
  `FixtureRunner` L30 (incl. why the L107 kwarg branch exists); docstring says
  "two source roots" but code inserts one L4; tidy run_manifest prose.
- `annotator/calibration/fixtures.py`: rewrite the dense Fixture docstring +
  document tuple layouts L54-70; net_band stored twice (field + court_geo[2]);
  REF_COURT_M[0] gloss; SHARED_FILES consumer note; (3,3) shape note L207;
  `_rounded_bounds` after caller.
- `annotator/calibration/scoring.py`: tighten the 30-line
  `_raw_precision_curve` docstring; COVERED overlapping-span nuance comment;
  name the wide-edge 90 and (5, 10) literals (hit-list #35's sibling);
  strict_contact_rows row-building can adopt wide_edge's `common`+spread idiom
  (flattens the seeded deep-nest).
- `annotator/calibration/sweep.py`: rename `_settings` → `_settings_sort_key` +
  docstring + unpack its 3-decision append; `_spec_from_row` helper for the
  duplicated CandidateSpec construction and the `same_winner_as_live`
  monster line L495; `_withheld` rename; "pinned"/"closed" prose.
- `annotator/calibration/gt_scoring.py`: player-A/B vocabulary + `sm_side`
  rename + `_norm_half` coercion comment; split flatten_metrics' 6-pair update
  lines; optional `reference_scores.py` data module (drops the file under 600).
- `annotator/inpaint_guard.py`: `_validate_presence` four-sum helper (also
  exposes two vacuous bounds); private-helper docstrings + one info-dict
  construction site; (0,0) sentinel comment; `np.any(np.ptp(points, axis=0) > 0)`;
  121-char signature.
- `annotator/experiment_records.py`: magic 8 configurations L94; "closed
  measurement manifests" wording.
- `annotator/composition_mask.py` / `doubles_flag.py` / `replay_mask.py`:
  prose-level only (sentence splits, one TypedDict candidate, track column
  gloss, one comprehension). Cleanest modules in the target.
- `annotator/point_winner.py`: mis-indented "Same machinery on the nearer
  ANKLE" comment visually splits its loop L409-412 (audit-found); dense
  edge-case comments L229/L437 worth sentence-splitting; docstring gaps L696/
  L723/L637/L366; sticky/"shipped" vocabulary one-liner.
- `annotator/run_video.py`: rejected_grades pointer to
  `inpaint_guard.CODE_NAMES`; module docstring 2-3 sentences (hit-list #1
  covers the rest).
- `annotator/e2e_court_annotator.py`: type RunDriver's four `Any` fields;
  role-order sort lambda constant (+99 sentinel); `_write_scene_evidence_partial`
  overlap; three one-line single-caller wrappers.
- `annotator/rally_segmentation.py`: quiet_burst generator → explicit loop;
  rolling-mean scaffold dedup; chained ternary L1417; `:param:` gaps; "base-30
  nine" gloss; trim segment_video's caveat block.

## Comprehension traps (probe evidence)

Probes: 28 Haiku low + 2 Sonnet low (the chunked giants). Six modules produced
fully clean cold reads (batch_report, dead_mask, doubles_flag,
composition_mask, replay_mask, stage11_pairing) — modest positive evidence of
whole-module coherence. The traps that survived assessment, each anchored to
code that invited the misread:

- `stage1_index.enrich_row`: probe read "True = call hit YouTube" as
  success-only; the docstring says exactly that while the code returns True on
  timeout/error (hit-list #34).
- `selection`: probe ranked "fewest merges" by merge count and called the
  contact F1 "floored" — both conflations are the function names' fault
  (hit-list #26).
- `stage10_clean`: probe could not hold the `_score_pending` lifecycle; neither
  could a skim audit (hit-list #19).
- `e2e_court_annotator` (Sonnet probe): stumbled on the undefined "shared
  contract", the SHARED_FILES monkey-patch, and the phantom "no-GT path"
  comment (hit-list #17).
- `court_evidence` / `point_winner`: the `[0, 1, 3, 2]` corner reorder and the
  identity `_as_ref_corners` call (hit-list #29).
- `resolve` / `fps_constants` / `config`: base-30 semantics, "candidate seats",
  and the "unwired until threading" migration notes — resolved by the glossary
  entry (#3) plus resolve's docstring rewrite.
- Assessor-killed probe errors (not code faults): stage3's keep-rule
  compression, stage2's sidecar-per-video overstatement, scoring's
  midpoint/floor reading, fixtures' 1.5-multiplier doubt, replay_mask and
  stage11 minor wording. These are recorded as model misses, not findings.

## Incidental silent-failure notes (no dedicated lens ran)

Kept: stage11's two quiet missing-input paths (chunk sidecar, replay mask);
experiment_records' silent malformed-record skip at a sanitisation boundary;
the rally_segmentation ↔ replay_mask filename mismatch (hit-list #8).
Killed as documented-deliberate: config.write_candidates blank-fill,
stage3/stage10 broad retry catches (noqa + rationale), point_winner's NaN→0.0
determinism substitute (commented two lines above), rally_segmentation's
per-video batch isolation.

## Funnel behaviour notes

- Luna's merged verify+assess seat killed 95 of 398 verdict rows with stated
  grounds; audit agreed with all but 6. Roughly 25 of the kills were
  verbatim-quote technicalities (indentation, trailing commas, scanner
  ellipses) rather than substance; luna's kill notes preserved the substance,
  which is what made the 6 audit resurrections possible (stage3 helper order,
  selection comprehension, fixtures net_band duplication, scoring's two
   3-vs-4-tuple stale-docs, e2e RunDriver typing). If the skill re-runs with a
  codex verifier, softening the quote gate to "quote locates the site" would
  cut that churn.
- Haiku lens scans ran hot on docs categories (as designed); the naming and
  structure lenses produced most of the kills. The Sonnet cross-module scans
  were the highest-precision stage in the pipeline: 28 raised, ~24 confirmed at
  audit, including four of the hit-list's tier-2 entries.
- Positive baseline worth naming: `types.py`'s array primitives, the
  calibration validators' loud error contracts, and the scraper stages'
  docstring discipline (stage 1/2/3 module docstrings state failure behaviour
  plainly) are the house style working as intended.

## Deferred: associated tests

Per ruling, tests were not audited. Modules with hit-list entries map to these
test files for any follow-up pass: test_annotator_run_video,
test_annotator_scoring, test_annotator_measurement, test_calibration_scoring /
_selection / _schemas / _run_cli, test_sweep, test_annotator_fixtures,
test_scraper_stage1/2/3/10/11, test_scraper_download_videos, test_dead_mask,
test_inpaint_guard, test_point_winner, test_court_evidence,
test_annotator_serve_setup(_b2), test_sticky_anchor / _result,
test_tracker_segments, test_batch_report, test_annotator_experiment_records.
