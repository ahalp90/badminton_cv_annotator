# Readability refactor: scoping handoff

## Contents

- [Decision summary](#decision-summary)
- [Start here](#start-here)
- [Appendix A: detailed findings by area](#appendix-a-detailed-findings-by-area)
  - [Shared rules, records and misleading contracts](#shared-rules-records-and-misleading-contracts)
  - [Historical and pipeline naming](#historical-and-pipeline-naming)
  - [Scraper workflow](#scraper-workflow)
  - [Calibration and scoring](#calibration-and-scoring)
  - [Main video runner](#main-video-runner)
  - [Rally segmentation](#rally-segmentation)
  - [End-to-end court runner](#end-to-end-court-runner)
- [Appendix B: behavioural investigations and open questions](#appendix-b-behavioural-investigations-and-open-questions)
  - [Behaviour-sensitive findings](#behaviour-sensitive-findings)
  - [Open design questions](#open-design-questions)
- [Appendix C: dependencies and checks](#appendix-c-dependencies-and-checks)
  - [Dependencies to settle before batching](#dependencies-to-settle-before-batching)
  - [Focused comparison checks](#focused-comparison-checks)
- [Appendix D: evidence guide](#appendix-d-evidence-guide)
  - [Audit tags and known corrections](#audit-tags-and-known-corrections)
  - [Existing source maps](#existing-source-maps)
  - [Raw review material](#raw-review-material)
  - [Review boundary](#review-boundary)

## Decision summary

The review found concentrated readability problems rather than a repo that
needs rewriting from end to end. It examined 30 annotator and scraper modules,
then consolidated the evidence into 35 ranked findings and roughly 55 smaller
observations. Most friction sits in a few large orchestration paths, copied
rules, implicit data contracts, repeated scraper workflow and stale pipeline
naming. Other parts of the repo already show the desired direction through
clear array helpers, validators that fail loudly and scraper modules that
explain their failure behaviour. The report is a map for investigation, not a
verified implementation plan.

The strongest likely returns come from contained changes that clarify several
consumers at once. Copied scaling rules, tolerance values, metric formulas and
manifest settings can drift even where the copies currently agree. Shared
arrays and records often have stable but implicit field contracts. Some names
and docstrings state the wrong behaviour, while several scraper mechanisms are
nearly identical. Most of this is small or medium work that should reduce both
onboarding cost and future drift. The review's helper and record sketches are
candidate approaches; scoping still needs to find the simplest correct owner
and representation.

**Recommended starting point:** inventory the settled naming work and verify a
small group of shared-rule and contract candidates. Treat large orchestration
refactors and behaviour-sensitive findings as separate design or investigation
work.

Removing historical labels and numbered pipeline names is a project decision,
not an auditor preference. Live references such as B5, B6, W2.9 and invented
experiment titles must be replaced with the behaviour or rationale they are
meant to explain. Numbered scraper modules, tests, commands, errors and active
pipeline docs need descriptive names. The policy is settled, but scoping must
still find the full rename surface and choose replacements. Plan the cleanup
early and execute it as one coherent change. Archived worklogs may retain real
historical batch labels. Persisted fixture identifiers remain where renaming
would change a data contract.

The largest functions and modules create substantial friction, but size alone
does not make them the best first work. Extracting `run_video()` is a design
problem despite its useful internal stage boundaries. Its modes, early returns
and writes into an object supplied by the caller complicate helper ownership.
`score_video()` uses positional accumulators and records that deserve clearer
structure, but changing the whole function is still a session-sized job. A
broad split of `rally_segmentation.py` is less settled again. Its shared
smoothing, player data used by both sides and top-level `segment_video()` entry
point cross the proposed seams. These areas need focused design before code
moves, although smaller contract improvements may still be worthwhile.

Some findings touch behaviour and cannot be treated as obvious cleanup. Four
scraper paths use similar mass-failure protection with different thresholds,
but the review did not establish whether those differences are intentional.
Commentary pairing can continue without two inputs, and the experiment-record
sanitiser handles one malformed case differently from its summary path. A
duplicated count calculation also provides a reconciliation assertion. Dead
masks and replay masks are separate artefacts with separate consumers; similar
names are not grounds to unify them. Trace these cases end to end before calling
them defects or choosing new behaviour.

The report is anchored to commit `1afc86a` on `main`, dated 2026-08-04. Treat
its line numbers as navigation aids, find each candidate again by symbol and
read the current code. For accepted cross-file findings, check every named
producer, consumer and copy. Tests and several directories were outside the
review, so silence there is not evidence that no supporting work is needed.

Effort labels are rough comparisons: **small** means less than half an hour,
**medium** means an hour or two, and **large** means a focused session. The
original ranking mixed onboarding pain with expected fix cost, so it was never
an execution order. The table below instead compares confidence, likely first
moves, effort, risk and dependencies. Use it to choose candidates for
verification. Naming work is larger than the report estimated because it now
includes files and every reference to them.

The first scoping session should finish the naming inventory and verify a small
set of high-return candidates rather than design the whole refactor at once.
Classify each other candidate as **accepted**, **rejected** or **investigation
needed**. For accepted work, record the affected consumers, behaviour at risk
and a focused check that would catch a mistake. Then record dependencies before
choosing implementation batches. Finish with a defensible initial scope, an
explicit out-of-scope list, unresolved investigations and a dependency sketch.
Do not assume that any other finding or proposed fix will survive verification.

## Start here

Confidence describes the present evidence, not implementation readiness:

- **Established:** the existing review directly supports the diagnosis
- **Promising:** the diagnosis is supported, but the proposed response still
  needs validation
- **Investigation needed:** current behaviour or design intent is not yet
  established

| Area | Why it matters | Likely first move | Confidence | Rough effort or risk | Important dependency |
|---|---|---|---|---|---|
| Shared rules and contracts | Copied formulas, settings and implicit record shapes can drift across consumers | Verify the current copies and their consumers; choose one owner only where the semantics match | **Established** | Small to medium; low-to-moderate risk | Check every producer, consumer and copy; retain focused comparisons |
| Historical and pipeline naming | Stale labels and numbered modules misdescribe the live system and would contaminate new documentation | Inventory live labels and numbered names, choose descriptive replacements, and schedule one coordinated rename across source, tests, CLIs and active docs | **Established** | Medium to large; moderate rename risk | Plan first; execute before broad contract or documentation changes |
| Scraper workflow | Repeated subprocess, retry and validation code can drift, while similar failure guards may encode different policies | Separate exact duplication from deliberate policy differences; scope `_download_one()` with its existing/fresh boundary | **Promising** | Medium; moderate behaviour risk | Coordinate with module renames and establish failure intent first |
| Calibration and scoring | Positional records and undocumented metric shapes hide meaning; copied rules offer smaller wins nearby | Start with verified rule ownership and record contracts; design `score_video()` separately | **Promising** | Small to large; risk varies by candidate | Preserve persisted schemas and compare changed calculations on fixtures |
| Main video runner | One function carries most of the pipeline, so local changes require understanding many conditional handoffs | Use the existing map of its execution steps to design and review helper boundaries before extracting code | **Promising** | Large; high change surface | Preserve both early returns, contact modes and outputs supplied by the caller |
| Rally segmentation | The module is hard to navigate, but the proposed broad split crosses several shared pieces | Recheck the seam map; compare CLI extraction and contract work with a redesigned larger split | **Investigation needed** | Medium to large; high structural risk | Decide ownership of shared smoothing, derived player data and `segment_video` |
| End-to-end court runner | Six orchestration steps lack contracts, and setup relies on shared mutable and positional state | Verify the findings in current source and scope contracts before considering a split | **Promising** | Medium for contracts; larger if split | Trace the temporary change to `gt_scoring.SHARED_FILES` and its consumers |

The rest of this document is reference material, not a linear reading
assignment. Go to the matching area in Appendix A for the candidates being
scoped. Use Appendix B for behaviour questions, Appendix C after accepting work
and Appendix D when checking the review evidence.

## Appendix A: detailed findings by area

### Shared rules, records and misleading contracts

Frame-rate scaling, tolerance values, safe-F1 calculation and configuration
metadata are each copied between calibration or annotator modules. The copies
agreed where the review checked them, but no shared declaration keeps them in
step. That creates a realistic drift path in which scoring, validation and run
manifests describe different policies. Most of these repairs are small to
medium changes, and each can remove risk from several consumers at once.

Several shared arrays, tuples, dictionaries and records leave their field
meanings or shapes implicit. `StickyResult`, the dead-mask dispatcher, and the
calibration metric dictionaries are prominent examples. Callers have to
recover the contract from indexing and construction code, which makes correct
uses look arbitrary and real misalignments harder to spot. Most contract fixes
are small to medium docstring or typing changes.

A smaller group of names and docstrings states the wrong contract. Selection
keys misname the values they rank, contact docstrings describe a three-item
tuple where the type has four items, and `enrich_row()` describes `True` as
success even though failures after an attempted request also return `True`.
Review readers made the exact mistakes those descriptions invite. Correcting
them is small, high-return work because it removes misinformation rather than
adding explanation.

### Historical and pipeline naming

The live code still refers to old work packets and invented experiment titles
such as B5, B6, W2.9, Wave 1, Brief H, "block-2" and the "three-arm
remeasure". These labels do not explain the current rule they are meant to
justify, and some now point at a pipeline arrangement that no longer exists.
A glossary would preserve the indirection rather than fix it. Replace each live
reference with the relevant behaviour, rationale, date or commit, and delete
the reference when the history no longer helps the reader.

The scraper bakes obsolete ordering into the module names
`stage1_index.py`, `stage2_transcripts.py`, `stage3_triage.py`,
`stage10_clean.py` and `stage11_pairing.py`. The numbering also appears in
imports, test names, module commands, errors, comments and active pipeline
docs. Annotator docstrings and configuration comments describe current
behaviour as numbered stages, while `test_stage6_b4_config.py` preserves two
obsolete schemes at once. Rename the modules and consumers around the jobs
they perform, then replace numbered prose with the function, module or
artefact it actually means.

### Scraper workflow

The scraper repeats yt-dlp calls, LLM retries, extension lists, keep-row
filtering and manifest validation. A policy change made to one copy can leave
another path using an older timeout, retry or validation rule. Shared helpers
in `config.py` are one candidate response, but scoping must first separate
identical mechanisms from similar-looking policies.

`_download_one()` mixes the existing-file path, fresh downloads and audio
verification in 111 code lines. Some failure paths delete an artefact while
others retain it, so a generic extraction made too early could change recovery
behaviour. The existing/fresh boundary is a promising first seam because it
would expose those differences before any audio-check deduplication.

`run_clean()` in the commentary-cleaning module tracks pending work under a
temporary `_score_pending` dictionary key. It consumes that key through a
side-effecting `pop` inside a condition, then reconstructs the state across
later loops. Multiple review readers lost track of the lifecycle. A named
sidecar record and an explicit pending set are candidate representations that
need checking against the current control flow.

Four scraper paths implement mass-failure protection with different thresholds,
input shapes and mid-batch checks. The code does not state whether those
differences are deliberate. Trace the failure contract of each path before
choosing a parameterised helper, one standard policy, or clearer local
documentation.

### Calibration and scoring

`score_video()` in `src/annotator/calibration/gt_scoring.py` stores seven
metrics in parallel four-slot lists, then builds a 27-field `RallyRow`
positionally. The slot and field meanings are enforced by alignment rather
than names, so a harmless-looking insertion could move data into the wrong
field. A small accumulator record and keyword construction are plausible
responses, but the whole change remains a session-sized refactor.

Several calibration functions also return stable dictionaries or large record
types without documenting their fields. Callers can use those values only by
knowing the keys or tracing their construction, which makes the effective API
harder to discover than the type hints suggest. Typed records and short field
contracts are candidate improvements rather than settled designs.

### Main video runner

`run_video()` carries most of the annotation pipeline through 31 parameters
and a 419-code-line body. Its court-optional return, stop-after-segmentation
return and injected-versus-automatic contact branch make its handoffs
conditional. The caller also supplies a capture object that `run_video()`
writes into as it works. The execution map listed in Appendix D identifies
candidate helper boundaries, but the signatures and ownership still need
design. The nested landing-horizon loop is the heaviest coupling because it
reads many results from the surrounding rally pass and writes them into that
shared capture object.

### Rally segmentation

`rally_segmentation.py` is 1,586 lines and contains recognisable serve, contact
and CLI regions. Later tracing found that the serve and contact regions share
smoothing helpers and a block that builds player evidence used by both. The
top-level `segment_video()` entry point also composes both regions. A three-way
split therefore needs an explicit home for those shared pieces or it merely
trades file size for cross-file navigation. The CLI is the only clean
extraction established so far.

The module's difficult serve and gap helpers have terse or missing contracts.
Public functions also leave valid parameter combinations, types and conditional
gating implicit. Contract work may improve the module even if the broad split
does not survive design review.

### End-to-end court runner

`e2e_court_annotator.py` combines six undocumented orchestration functions in
a 1,340-line module. Its setup temporarily replaces the module-level
`gt_scoring.SHARED_FILES` tuple that tells the scoring loader which input files
to verify. Several records also depend on positional alignment. Those choices
hide both the step responsibilities and the data passed between them. Adding
function and record contracts is a contained response; splitting the module
needs separate seam analysis.

## Appendix B: behavioural investigations and open questions

### Behaviour-sensitive findings

| Site | What the review found | Question for scoping |
|---|---|---|
| Mask filenames | A helper in `rally_segmentation.py` is named for replay masks but loads `<id>_dead_mask.npy`. The CLI in `replay_mask.py` writes `<id>_replay.npy`, which the commentary-pairing module reads. | Confirm both producer-consumer chains. Keep dead masks and replay masks separate unless a deliberate behaviour change is approved. |
| Missing pairing inputs | Commentary pairing can continue without a transcript chunk sidecar or replay mask, even when the video is otherwise eligible for pairing. | Decide whether each absence is expected, deserves a warning, or should stop the item. |
| Experiment-record sanitising | The planner in `experiment_records.py` quietly skips one malformed configuration that `build_summary` rejects. Its later path-search fallback can then delete the file instead of sanitising it. | Decide whether the malformed record should be logged, rejected or handled explicitly. |
| Mirrored configuration values | Court-evidence and calibration-sweep policy values are copied into run metadata by hand. The manifest can therefore report a policy the run did not use. | Read metadata from executable declarations, or establish why an independent record is necessary. |
| Count-gate calculation | `gt_scoring` recalculates a value already returned by `score_contacts`, then reconciles the two with an assertion. | Decide whether the second calculation still protects a real boundary before removing it. |

### Open design questions

- **Scraper failure protection:** are the four policies intentionally
  different, or have the implementations drifted?
- **Rally segmentation:** would a broad split remain easier to navigate after
  placing the shared helpers, shared player evidence and `segment_video`?
- **End-to-end court runner:** do function and record contracts remove enough
  friction, or does the module still justify a structural split?
- **Court evidence:** does defining the central "parent" concept clarify
  whether the court geometry came from pre-recorded ShuttleSet data or
  CourtKeyNet detector output? If that definition is not enough, would splitting
  `build_detected_court_evidence()` where raw evidence collection ends and
  consensus calculation can fail make the flow clearer? The existing
  `CourtConsensusError` marks that boundary.
- **Tests:** should a wider test-readability review join this work, or should
  the scope remain limited to tests changed by the refactor?

## Appendix C: dependencies and checks

Use this appendix after accepting candidates. It turns the initial scope into
possible batches and names the focused checks needed to protect behaviour. It
does not add another priority order.

### Dependencies to settle before batching

- Scope the shared scraper download helpers and `_download_one()` together.
  Decide the existing-file versus fresh-download boundary before placing those
  helpers.
- Rename numbered scraper modules with their imports, tests, module commands,
  errors and active docs in one coherent change.
- Keep documentation and contract fixes with the module that owns the
  behaviour unless a cross-file contract gives them a better home.
- Design `run_video()` around both early returns, both contact modes and the
  outputs written into the caller's capture object. Appendix D links the map of
  these execution steps.
- Give shared smoothing, player evidence and `segment_video` an explicit owner
  before approving a broad `rally_segmentation.py` split.

### Focused comparison checks

Behaviour-sensitive readability changes need checks aimed at the value or
contract they could disturb:

- **Base-30 scaling (report entry 4):** compare the surviving declaration with
  every removed formula across the supported frame rates before deleting the
  copies.
- **Mask filenames (report entry 8):** trace the dead-mask and replay-mask chains
  separately, and confirm that each consumer still resolves its original
  artefact.
- **Count gate (report entry 9):** on one fixture, compare the value returned by
  `score_contacts` with the value currently recalculated by `gt_scoring` before
  removing the duplicate calculation.

Adapt these checks if scoping changes the implementation. Run the focused
comparisons and relevant tests for each batch. Before declaring the full
refactor complete, run the repository gates from the repo root:

- `~/.venvs/badminton-cicd/bin/ruff check .`
- `~/.venvs/badminton-cicd/bin/pyrefly check`
- `~/.venvs/badminton-cicd/bin/pytest`

The test run matters because runtime shape checks cover contracts the type
checker cannot prove. The repository also requires commit-plan approval before
committing; keep the worklog current while executing the agreed batches.

## Appendix D: evidence guide

Use this appendix while validating a finding. It explains the review's source
coverage, corrections, supporting maps and exclusions. None of these records
replaces a current-source check.

### Audit tags and known corrections

The [detailed evidence report](readability_review_evidence_report.md) gives each
entry a problem statement, fix sketch, effort estimate and audit tag. Fix
sketches are suggestions rather than verified designs. Audit tags describe how
much source was read, not whether a finding is correct.

- **A1:** the final auditor read the full module and finding site
- **A2:** the final auditor read the finding site and surrounding region
- **A3:** another verifier read the source and supplied an in-file quote, but
  the final auditor did not open that site

The `FpsConstants` finding (report entry 12) has a known correction. Its
original sketch names
`FpsConstants._fields`, but `FpsConstants` is a frozen dataclass. Use
`dataclasses.fields(FpsConstants)` when evaluating that approach.

The historical-label finding (report entry 3) also has a scope correction. Its
glossary sketch is superseded by the project decision to remove historical
labels and stale numbered pipeline names from live surfaces.

The report marks entries 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 29 and 35 as
cross-file findings. If one is accepted, inspect every named producer, consumer
and copy rather than checking only the first site.

### Existing source maps

Open these maps when scoping the corresponding large function or module:

- `scratch/swarm_review/handoff/run_video_stage_map.md` maps the execution
  steps, inputs, outputs, early returns and cross-step state in `run_video()`.
- `scratch/swarm_review/handoff/rally_seg_seam_map.md` maps candidate split
  seams, shared helpers and mask filename chains around
  `rally_segmentation.py`.

### Raw review material

The paths below are relative to `scratch/swarm_review/`:

- `packets/<module>__verdict.json` contains per-claim verdicts and source
  quotes. The module stem follows its source path, such as
  `annotator_run_video`; the heaviest modules split their verdicts into `_a`
  and `_b` files. Across all 33 files, the 398 rows comprise 275 `CONFIRMED`,
  95 `KILLED` and 28 `MISREAD_BUT_TELLING` verdicts.
- `archive/raw_packets/<module>__scans.json` and
  `archive/raw_packets/<module>__probe.json` contain the original scans and
  first-read responses.
- `packets/cross_findings.json` contains cross-module findings.
- `metrics.json` contains mechanical measurements.
- `audit_notes.md` records the final keep and reject decisions by module.
- `archive/worklog.md` records how the review was run. It is preserved as an
  audit trail, including its original filename references and incorrect
  408-row tally.
- `archive/review_pipeline/` contains the scripts used to build and validate
  the review packets. See `archive/README.md` for the old-to-new path map.

### Review boundary

The review assessed `src/annotator/` and `src/scraper/`. It excluded
`src/annotator/validation_overlay/`, `__init__.py` files, tests and the
untracked `frontend/` directory. These boundaries describe the available
evidence; they do not define the refactor scope. The review itself changed no
source code.
