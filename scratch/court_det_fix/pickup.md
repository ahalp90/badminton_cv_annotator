# Court-detector pickup

## Resume — 23 September 2026

**Am1 now has a promising automatic recovery.** Three extra vanishing-point
seeds expose useful proposals. A fixed positive net-support preference selects
one with 7.2 working pixels maximum visible-landmark disagreement, versus 45.0
for the expanded pool's paint-score winner. Saved GX5 and Letterboxed45 outputs
stay unchanged. The user accepts the Am1 improvement, including the remaining
inset to the inner edge of the right yellow stripe. This remains a small
development result.
See the [experiment report](colour_consistency/AM1_RECOVERY.md) and
[three-case gallery](colour_consistency/am1_gallery/index.html).

**The wider test blocks unrestricted net preference.** The fixed rule changes
13 of 71 saved choices. Visual inspection confirms clear regressions on
Letterboxed78 (rank 1 → 706) and unlabelled control frame 52563 (1 → 266).
The original courts are better. The Am1 recovery remains useful, but strong
net support cannot override arbitrarily weaker paint evidence.
See [the current worklog](net_recovery/WORKLOG.md) and
[15-case gallery](net_recovery/gallery/index.html).

Next: freshly measure baseline and seeded pools consistently before further
generation comparisons, then test a restrained net preference. The replay
cache is repaired and Am1's accepted net choice reproduces exactly. Fresh and
historical paint scores still differ on identical geometry, including on
Carmack; the cause is unresolved. The saved-pool scan is unaffected. Combined
GX5 retains its selected geometry but is not a clean seed-only ranking
comparison. All seven GX views are in the saved scan; their wider combined
runs are deferred. The projected-net gallery also exposes false fragment
matches and a focal-search-bound limitation. No replacement ranking rule or
production change is adopted. Keep the expanded pools for replay.

**The automatic colour trials are complete and do not justify a new rejection
rule.** The floor cue rejected 0/9 saved choices. The hue-only paint cue
rejected 0/12: one usable comparison, eleven inconclusive. Weak or neutral
colour cannot supply a dependable hue. The intermediate raw-colour comparison
rejected Am1 but still penalised paler examples of the same hue.
[The experiment record](colour_consistency/PLAN.md) gives the evidence.

**Carry automatic stripe polarity forward for fitting integration.** The user confirmed this choice because black markings are plausible and
finds its six displayed comparisons practically identical to the bright-paint
rule, with both improving on the original. It handles dark-stripe inversion in
synthetic checks and adds modest cost. Real dark-court robustness remains
unproven. Original labels remain when polarity is unresolved. The current
production fitter is unchanged. See the [final edge policy](edge_polarity/local_audit/ASSESSMENT.md).

Some amateur references mark paint centres and others mark outside edges.
The convention is unknown per case. Small signed reference drift cannot
establish improved or worsened alignment on those courts.

All proposed decisions must be automatic. The manual-floor diagnostic was
superseded. The floor and player-floor paths are closed: `sticky_anchor` pairs
depend on court presence. Do not add player-selection machinery or reopen
colour threshold searches to force a positive result.

**SVD search reduction is implemented and measured. Deeper search has a useful
visual result.** The remaining main task is the scene-level detector described
below, with unresolved false acceptance treated explicitly.

This is the only live handover. Read this file first; open linked evidence for
a named question. [INDEX.md](INDEX.md) gives the history,
[FP_INDEX.md](FP_INDEX.md) locates code/data, and
[DETECTOR_DECISIONS.md](DETECTOR_DECISIONS.md) records lasting rulings.

The target remains a CourtKeyNet-free **scene-level detector**: sample a few
frames, choose a robust court from compatible evidence, reuse proposals within
a verified stable camera view, and resample or abstain when support changes.
Sharing across scenes is a possible extension when camera/court change is
negligible. The sampling, agreement, change checks and complete annotator runtime
are not integrated. The experiments below operate on individual saved views.

## What was usefully tried

### 1. Candidate sources and fitting

The wider evaluation covers 47 court views and 24 controls. Full W5 combines
G0 proposals from original fragments, G1 proposals from paint-filtered fragments,
and line templates, then refits and ranks them. Removing G0 loses a tolerable
Am4-319 result; G1/templates rescue cases such as GX5. Keep all three sources.
Both source-selection arms accept one of eight labelled non-court controls;
automatic acceptance and fallback are still unresolved.

The paint-side investigation explained part of the inset. Resolving strongly
polarised centre fragments to paint edges was preferred by the user in the
four-case comparison, then examined on eight amateur sources. It is a partial
experimental improvement, not an integrated fitting rule. One insignificant
amateur regression was reported; Am1 remained very bad. Its false far baseline
follows the net's bottom white band. Fitting a wrongly identified object more
closely cannot resolve that identity error. See the
[centre-to-edge assessment](edge_polarity/local_audit/ASSESSMENT.md).

### 2. SVD efficiency and deeper matching

SVD12 ranks 16 direction-support groups and retains 12 before matching. It
preserves original directions and IDs. It is the default in **fresh experimental
W5 generation**, with full16 available; it is not the complete scene runtime.
Nine development cases retained all eight historically approved automatic
witnesses and every best reference-agreement candidate. Three score-winner
roles were lost, with mixed substitute quality: the user accepted that tradeoff.

The completed nine-case timing run used **52.9% less summed matcher wall time**
(7,329.534 versus 15,545.095 seconds). Historical/shared-pair checks passed.
This excludes image scoring, refitting and scene processing. It is one repeat
on a shared host, not an end-to-end deployment latency claim.
[Implementation and evidence](svd_runtime/README.md).

The follow-up has **18 completed outputs: six cases × three G0-only SVD arms**.
Baseline keeps 512 axes and 256 courts per pair/global; deeper uses 640 axes
with the same court caps; wider shortlist uses 512 axes and 512 court caps.
Fitting and ranking are held fixed. Deeper costs 26.6% more summed full-trial
runtime than baseline; wider shortlist costs 10.0% more. These timings include
preparation, generation and refitting, so their denominator differs from the
52.9% matcher saving. [Full results and review](svd_search/WORKLOG.md).

| Views | User's visual assessment |
| --- | --- |
| SS03-19, SS03-34 | Deeper detector selections are preferred and exactly reproduce saved W5 G1 geometry. All three arms' reference-best refits look perfect. Wider shortlist selects the same courts as baseline. |
| GX0 | All three SVD versions are near-perfect after refitting, better than saved W5's back-left overshoot. Initial detections overshoot there; refitting fixes it. |
| GX5 | SVD examples are hallucinations, although some use a genuine long-edge court line. Saved W5 is great. |
| Am1-54 | SVD examples are hallucinations. Saved W5 could be useful apart from its net-tape baseline mistake. |
| Am2-28019 | Only saved W5 and deeper are suitable. Deeper is acceptable before refitting and perfect afterwards. Its detector selection exactly matches saved W5. |

On Am2, deeper reduces selected reference disagreement from 323.6 to 18.4
native pixels; its best refit falls from 288.8 to 7.9. This is a real reason to
spend saved compute on deeper matching. No new depth default or adaptive policy
has been adopted. Keep G1/templates: this G0-only comparison cannot establish
that SVD screening caused every bad fit. Reference-best views use annotations
retrospectively and must never be mistaken for detector choices.

### 3. Colour evidence — diagnostic only

The WebUI probe separated Am1's neutral net tape from yellow paint; local replay
reproduces its tape-to-paint separation (local 64.33 versus returned 64.35 Lab
chroma units). All four reference
fragments belong to one service marking. GX has no supported far-baseline
fragments in that packet, so that particular veto is untestable. No genuine
supported far-baseline positive control validates the original veto.

The broader local diagnostic processed 71 views using preserved geometry.
It records raw paint, adjacent floor and paint-minus-floor colour separately;
uses perpendicular projected stripe widths and native-resolution sampling;
and excludes the target marking from its references. The audit and corrections
resolved floor-colour confounding, hidden within-marking variation and incorrect
perspective width sampling. The 20-Lab-unit ambiguity display flag is exploratory.

Useful limits emerged: saved Am1 has only one supported marking under this
sampling rule; a bad GX5 court can have very similar paint colours on its few
supported lines. Colour similarity alone therefore cannot certify geometry.
Greyscale footage supplies no independent chroma evidence. None of these
measurements has yet been used to rerank, reject or refit a court.

**The colour gallery is not a before/after experiment.** Saved W5 and saved
G1/templates apply the same earlier ranking to different source access. Of
64 cases with fits, 50 have identical geometry, 14 differ; seven further cases
have no fit. IDs and generated gallery coordinates were checked against the
saved records with no mapping error. The largely duplicate presentation was
unhelpful for judging colour's benefit. The separate renderer was replaced by
the actual shared SVD template, and the user confirmed its controls work. That
confirms the UI, not colour efficacy. [Measurements, corrections and limits](colour_consistency/PLAN.md).

## Completed automatic colour trials

The floor trial compared W5, baseline SVD and deeper SVD choices on Am2-28019,
GX0 and GX5. Reference floor colours came from the sides of observed internal
markings. All nine choices were retained, including the bad Am2 baseline.
At the stricter sensitivity setting two GX5 alternatives were rejected; this
threshold-dependent result does not justify adoption.

The observed-paint trial used native image samples along retained fragments
on twelve W5 choices. Distinct markings supplied references; the target marking
was excluded. Raw-colour comparison rejected none. Grouping references by hue
made Am1 separable, but comparing target colour strength still failed the
same-hue pale-paint check. The final hue-only rule made no rejection. Low
chroma and greyscale evidence remained inconclusive.

**Close both cues without integration.** Colour similarity cannot certify
geometry, and these trials did not establish a useful automatic contradiction.
Do not substitute annotated best candidates or add another threshold search.
The [worklog](colour_consistency/PLAN.md) preserves all arms and limitations.

## Am1 net lead, then integration

The [net assessment](NET_EVIDENCE_ASSESSMENT.md) now includes an actual selection
trial. The earlier no-go applied to the old candidate pool. Additional automatic
line-group seeds repair a specific omission; positive net support then selects
a much better visible fit. No post detector or colour selector was added.
The old scorer's missing-support penalty remains rejected.

The next decision depends on the new gallery and a bounded combined-change
regression check. Avoid another threshold search or broad proposal sweep.
The result does not resolve general false acceptance or scene-level agreement.

**User's intended order: finish colour, assess this final new lead, then build
a coherent, pragmatic deployable detector.** Bound the new lead to a useful
comparison and a go/no-go decision. Do not reopen the older idea catalogue as
another open-ended campaign. Integration and necessary validation continue;
perfecting every failure is not a prerequisite to making explicit fallback and
abstention choices.

## Remaining route to the detector

1. Establish useful selection/rejection evidence on named failures; retain
   necessary G0, G1 and template coverage and explicit fallback/abstention.
2. Test sparse scene sampling, robust agreement, stable-view reuse and camera
   change handling on short clips. A consistently selected wall remains wrong.
3. Integrate the scene interface, then remove CourtKeyNet/model-loading remnants.
4. Evaluate unseen cameras, failure frequency and ordinary-hardware latency.
   Neither development fit counts nor Carmack matcher timings establish these.

The current compute audit lists small optional optimisations. Greyscale reuse
already works in the experimental paths; do not rediscover or re-fix it.
The historical catalogue, including work overlapping the final packet, remains in
[decisions](DETECTOR_DECISIONS.md#older-ideas-worth-bringing-back).

## Local state and working agreements

- The colour trials, automatic-edge comparison and bounded Am1/net trial are
  complete; the user accepts the Am1 recovery with a small right-edge inset. Commits are authorised on
  `fix/court-det`. No production detector change has been adopted.
- Branch `fix/court-det`. This close-out checkpoint includes colour code and
  measurements, both galleries, compact search results, received packets and
  these status updates. Use `git log -1` and `git status` for the current revision.
- Both Carmack experiments and their transfers finished with exit 0. No compute
  job remains active. Search receipts and all 18 outputs
  are under `svd_search/`; do not relaunch the completed run.
- Current review: [automatic edge decisions](http://127.0.0.1:8881/), served from
  `colour_consistency/decision_gallery/`. Older galleries: [SVD](http://127.0.0.1:8879/),
  [historical colour diagnostic](http://127.0.0.1:8880/).
  If the local servers have ended, serve `svd_search/gallery/` and
  `colour_consistency/gallery/` respectively. Their index files and data persist.
- GPT-6 Sol high/default tier owns substantial coding, experiment runners and
  galleries. Use medium only for simple bounded tasks. GPT-6 Luna max/priority handles tightly bounded
  mechanics. Opus `claude-opus-5-5` high is authorised for bounded audits, with
  the coordinator checking material findings. Do not impose time limits on
  Opus. No silent GPT-5.6 substitution.
- Codex-agent, Anthropic and Carmack project sharing is authorised. Serena/Pyrefly
  may be reused at `http://127.0.0.1:9121/mcp` when reachable. Use up to six remote
  workers with one numerical thread each; read `~/.codex/remote_hpc.md` first and
  keep one remote connection at a time.
- Avoid provenance theatre: reuse successful checks and data, make one bounded
  audit count, and keep the user responsible for visual judgement. For galleries,
  delegate to Sol with explicit instructions to reuse
  `svd_search/gallery_template.html`; show changed decisions, abstention and
  unchanged good controls clearly.
- The user authorised this close-out commit and push on the feature branch.
  The twenty untracked frozen PNGs and large raw run artefacts remain local.
  The received WebUI packets are included unchanged, including their original
  JSON files and supplied images. Large older populations live under
  `worklog/remote_records_20260921/`; recovery archives remain sealed.
- SVD integration/runtime checks and colour scoped checks are recorded beside
  those experiments. The latest gallery builds, scoped lint, payload and JS
  checks passed. The user's browser check supersedes the blocked headless check.
  This handover update requires documentation/link checks, not another test run.
- The earlier close-out Opus 5-5 high audit covered the four top-level documents
  and selected supporting
  records. It found the story recoverable and consistent. The close-out clarified
  the colour measurement, historical overlap, packet age and current routing;
  the later frontier pre-evaluation is recorded in the assessment above.
- G1 scene 0029's original local input/estimator pair is incomplete; recover it
  only for an exact replay that needs it. Historical W2 caveats remain recorded.

The visual ideal is imperceptible misalignment, with outside paint edges
preferred. There is no agreed pixel cutoff or fallback frequency. The historical
24/27 W5 count means usable for development, not 24 clean fits. ShuttleSet's
static video homographies apply only to ordinary play views, not arbitrary
transitions or side-on shots. Preserve those limits in future comparisons.
