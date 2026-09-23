# Court-detector pickup

## Resume — 23 September 2026

**SVD search reduction is implemented and measured. Deeper search has a useful
visual result. Colour has only been measured on saved courts: no colour-based
selection, rejection or fitting improvement has been tested.** The next useful
experiment should change a decision on preserved candidates and show that
change clearly. Do not ask the user to review another unchanged-fit gallery.

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

## Recommended next experiment — not yet run

Start with **Am2's baseline court extending into the adjacent blue floor**.
Test a bounded floor-region contradiction cue on preserved candidates. Sample
court-side and outside strips, including stretches without detected paint.
Anchor appearance to independently reliable visible court regions. A colour
transition need not coincide with the painted badminton boundary; use paint
for exact placement. The current diagnostic flags ambiguity on four of baseline's
five supported markings and none of deeper's five, but that is only a lead.

Compare known bad fits with the retained good Am2/GX examples. Keep the candidate
pool fixed across comparator and colour arms; report rejection/abstention if
that pool lacks a good replacement. Adding deeper or template candidates is a
separate coverage change. Do not seed a selector with reference-best candidates
chosen using annotations. Am2's full saved W5 fit is already good: rejecting its
bad G0-only alternative would establish a useful cue, not an improvement to the
existing full W5 selection by itself.

Am1 supplies a second question: can colour measured along independently
supported **observed fragments** distinguish tape from real paint without
relying on the wrong fit's predicted stripe width? Keep that trial separate
from the Am2 floor cue. Use genuine-paint controls and retain ambiguous outcomes.

A useful deliverable is one small, real **before/after selection, rejection or
fit comparison**, plus unchanged good controls. If the cue changes no useful
decision, say so and stop it. Reuse the shared gallery template literally.
Do not launch another full search or a broad unchanged-image review for this.
This handover task does not itself launch either experiment.

## One final new lead, then integration

The user has supplied [the net-evidence frontier packet](webui_net_evidence_frontier_handover/README.md)
at the top level. Only its README and manifest were skimmed to identify it;
its new proposals are **unreviewed and untested locally**. It combines historical
net-image work and the Am1/GX audit with further net-assisted court-localisation
ideas. The packet describes the older `1353541` checkout, before the search and
colour results above. Its colour-probe script is identical to the earlier return.
Reconcile overlap with that replay and with the historical net-image replay in
[decisions](DETECTOR_DECISIONS.md#older-ideas-worth-bringing-back) before commissioning
work. Its `docs/03_FRONTIER_DIRECTIONS.md` is the route to the new
proposal space, not an instruction to run every listed idea.

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

- Branch `fix/court-det`. This close-out checkpoint includes colour code and
  measurements, both galleries, compact search results, received packets and
  these status updates. Use `git log -1` and `git status` for the current revision.
- Both Carmack experiments and their transfers finished with exit 0. No compute
  job or implementation worker remains active. Search receipts and all 18 outputs
  are under `svd_search/`; do not relaunch the completed run.
- Galleries: [SVD](http://127.0.0.1:8879/), [colour diagnostic](http://127.0.0.1:8880/).
  If the local servers have ended, serve `svd_search/gallery/` and
  `colour_consistency/gallery/` respectively. Their index files and data persist.
- GPT-6 Sol medium/default tier owns bounded coding and galleries; use high for
  a specific unresolved problem. GPT-6 Luna max/priority handles tightly bounded
  mechanics. Opus `claude-opus-5-5` high is authorised for bounded audits, with
  the coordinator checking material findings. No silent GPT-5.6 substitution.
- Codex-agent, Anthropic and Carmack project sharing is authorised. Serena/Pyrefly
  may be reused at `http://127.0.0.1:9121/mcp` when reachable. Use up to six remote
  workers with one numerical thread each; read `~/.codex/remote_hpc.md` first and
  keep one remote connection at a time.
- Avoid provenance theatre: reuse successful checks and data, make one bounded
  audit count, and keep the user responsible for visual judgement. For galleries,
  reuse `svd_search/gallery_template.html`; show changed decisions clearly.
- The user authorised this close-out commit and push on the feature branch.
  The twenty untracked frozen PNGs and large raw run artefacts remain local.
  The received WebUI packets are included unchanged, including their original
  JSON files and supplied images. Large older populations live under
  `worklog/remote_records_20260921/`; recovery archives remain sealed.
- SVD integration/runtime checks and colour scoped checks are recorded beside
  those experiments. The latest gallery builds, scoped lint, payload and JS
  checks passed. The user's browser check supersedes the blocked headless check.
  This handover update requires documentation/link checks, not another test run.
- Opus 5-5 high audited the four top-level documents and selected supporting
  records. It found the story recoverable and consistent. The close-out clarified
  the colour measurement, historical overlap, packet age and current routing;
  it did not evaluate the final packet's new proposals.
- G1 scene 0029's original local input/estimator pair is incomplete; recover it
  only for an exact replay that needs it. Historical W2 caveats remain recorded.

The visual ideal is imperceptible misalignment, with outside paint edges
preferred. There is no agreed pixel cutoff or fallback frequency. The historical
24/27 W5 count means usable for development, not 24 clean fits. ShuttleSet's
static video homographies apply only to ordinary play views, not arbitrary
transitions or side-on shots. Preserve those limits in future comparisons.
