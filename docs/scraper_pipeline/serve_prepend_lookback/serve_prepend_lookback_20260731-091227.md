# Serve-prepend lookback: current build orientation

> Update 2026-08-08: the three-video candidate follow-up is complete. Its
> [measurement and decision](serve_prepend_lookback_20260808_measurement.md)
> supersede this document for Issue 28. Keep this file for the earlier baseline
> and processing-order orientation.

Validated 2026-07-31 against commit `63f40938a62f6612ca9a63b61127d24442a80865`.
This is the authoritative current-code orientation for the deferred feature. It records an
exploratory measurement and the smallest next build. It does not implement serve prepend.

## TL;DR

The current measurement can run against the maintained processing chain. It ran
`run_video` on all 292 calibration rallies in `sset_01`, `sset_15`
and `sset_21`, using the canonical fixture inputs. It wrote one row per GT rally and
reloadable compressed evidence artefacts.

With the committed replay mask, 137 rallies have an unmatched GT serve while at least one later
GT stroke matches an accepted contact. The fixture counts are 64, 39 and 34. A clean visible-track
run appears in the one-second serve-centred window for 114 of those 137 cases. Seventeen missed
serve frames are on the believed replay mask. Only one missed case has an accepted contact within
the matching tolerance of the serve; 15 have a raw candidate that close.

These counts are leads for the next measurement, not a production-feature recommendation. The
`no_replay` run sets the per-frame `raw_exclusion_mask` vector to
`False` for every frame. This disables replay/cutaway masking at that input. Other
processing and downstream filters remain active. The run changes rally spans and contact counts
substantially, so it is a mask-sensitivity control rather than a cleaner baseline.

The current package is
`docs/scraper_pipeline/serve_prepend_lookback/`. Its script, evidence pack, README and
this orientation are the active materials. The archived design record is context only:
[docs/archive/serve_prepend_lookback.md](../../archive/serve_prepend_lookback.md).

## Contents

- [TL;DR](#tldr)
- [1. What is authoritative](#1-what-is-authoritative)
- [2. Current run and provenance](#2-current-run-and-provenance)
- [3. Current measurements](#3-current-measurements)
- [4. Current processing order](#4-current-processing-order)
- [5. Measurement outputs](#5-measurement-outputs)
- [6. Smallest next measurement and build](#6-smallest-next-measurement-and-build)
- [7. Decisions still owed](#7-decisions-still-owed)
- [8. Code map](#8-code-map)
- [9. Lean definition of done](#9-lean-definition-of-done)

## 1. What is authoritative

There is one active orientation for this feature: this document. The neighbouring files have
narrower jobs.

| Location | Purpose | Use it for |
| --- | --- | --- |
| `docs/scraper_pipeline/serve_prepend_lookback/` | Current feature package | Measurement script, compressed outputs, README and this orientation |
| `docs/archive/serve_prepend_lookback.md` | Archive-only design record | Historical framing and provenance; not current numbers or an active specification |

The archive may retain old figures and design alternatives. Do not use it to choose current
behaviour. The feature package contains the current evidence and the implementation orientation.

## 2. Current run and provenance

The run used
`src/annotator/calibration/fixtures.py::FIXTURES` and
`src/annotator/calibration/gt_scoring.py::build_run_video_inputs`. The local shell did
not set `ANNOTATOR_FIXTURES_ROOT`, so the command supplied the pinned fixture root:

~~~bash
ANNOTATOR_FIXTURES_ROOT=$PWD/local_scratch/autograder_architecture PYTHONPATH=src \
  ~/.venvs/badminton-cicd/bin/python -u \
  docs/scraper_pipeline/serve_prepend_lookback/measure_serve_prepend_lookback.py \
  --out docs/scraper_pipeline/serve_prepend_lookback/data/serve_prepend_lookback_20260731-040847 \
  --mask-mode both
~~~

The source arrays are the maintained, pinned calibration substrate in the repository's local
scratch area. They are not a newly extracted 2026-07-30 broadcast dataset. The processing chain
and matching code were current at the recorded SHA.

The three fixtures are:

| Fixture | Video id | FPS | GT rallies |
| --- | ---: | ---: | ---: |
| `sset_01` | 1 | 25 | 113 |
| `sset_15` | 15 | 25 | 104 |
| `sset_21` | 21 | 30 | 75 |

The script runs two modes:

- `committed` uses the fixture's supplied raw replay mask, duration-filtered by the
  current chain
- `no_replay` supplies a boolean vector with `raw_exclusion_mask = False` at
  every frame, disabling replay/cutaway masking at that input

The second mode is a sensitivity run. It changes segmentation before contact filtering, so its
result cannot be read as the isolated benefit of allowing contacts on replay frames. Other
processing and downstream filters remain active. `raw_exclusion_mask` names the
replay/cutaway mask input only; it is not a switch for every exclusion-like condition. In this run
`court_invalid_is_excluded=False`, so a missing court row is not automatically converted
into a replay exclusion.

## 3. Current measurements

### Target and evidence definitions

The target is a GT rally whose first stroke is unmatched after the existing span-overlap candidate
collection, while one or more later GT strokes match accepted contacts. This identifies a possible
repair case without counting a rally that has no usable contact evidence.

The ledger collects accepted contacts from every span overlapping the GT rally extent, including
split and missed span categories. It is therefore broader than the canonical
`score_video` covered-only matching metric. The category is retained so the next
measurement can separate ordinary covered rallies from boundary and whole-span failures.

A clean run is a sequence of consecutive frames with a visible shuttle track and inpaint code
`NO_FLAG`. The serve-centred window extends one second on either side of the GT serve.
The minimum clean-run length uses the existing five-frame base-30 tolerance scaled by FPS: four
frames at 25 FPS and five at 30 FPS.

The pre-contact lookback measures the same evidence before the nearest accepted contact assigned
to an overlapping span, using the current `serve_start_lookback_frames` value. If no
accepted contact exists, the script falls back to the GT serve frame and records that anchor source
as `gt_serve_fallback`. The current CSV label
`first_assigned_accepted_contact` is a misleading implementation label: the code chooses
the accepted contact nearest to the GT serve, not the earliest accepted contact in time. Treat that
label as a schema detail to tidy when the recorder is extended.

### Committed-mask baseline

| Fixture | Target serve misses | Clean serve window | Clean pre-contact lookback | Raw candidate within tolerance | Accepted candidate within tolerance | Serve on believed mask | Accepted contacts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sset_01` | 64 | 53 | 28 | 7 | 1 | 0 | 1,749 |
| `sset_15` | 39 | 36 | 12 | 6 | 0 | 0 | 1,233 |
| `sset_21` | 34 | 25 | 17 | 2 | 0 | 17 | 1,090 |
| **Pooled** | **137** | **114** | **57** | **15** | **1** | **17** | **4,072** |

The current chain preserves substantial visible-track evidence around the serve in the target
misses, but almost none of that evidence becomes an accepted contact at the serve frame. This is
evidence worth investigating, not a trigger recommendation. A lookback still needs a candidate
rule and a false-positive audit.

The raw and accepted near-serve columns search all contacts in the video. They indicate coarse
detector availability; they do not show that a contact belongs to the target rally. Before these
counts inform a feature decision, a follow-up recorder should emit explicitly span-scoped
candidate rows.

### Mask sensitivity

| Fixture | Mask mode | Target serve misses | Detected spans | Raw contacts | Accepted contacts | Covered / split / missed GT rallies |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sset_01` | committed | 64 | 113 | 3,096 | 1,749 | 110 / 2 / 1 |
| `sset_01` | no replay | 66 | 36 | 8,561 | 2,078 | 111 / 2 / 0 |
| `sset_15` | committed | 39 | 142 | 2,210 | 1,233 | 84 / 4 / 16 |
| `sset_15` | no replay | 40 | 94 | 7,179 | 1,688 | 99 / 4 / 1 |
| `sset_21` | committed | 34 | 98 | 2,246 | 1,090 | 54 / 20 / 1 |
| `sset_21` | no replay | 33 | 38 | 4,980 | 1,404 | 71 / 4 / 0 |

The no-replay run merges long stretches and changes the number of detected spans, so it is not a
feature ablation. It does show that replay masking is load-bearing for this decision. The next
measurement must compare the current trust rule and any candidate exemption against the same
current chain.

## 4. Current processing order

In normal mode, the chain builds sticky evidence before applying the definitive replay mask. The
believed mask then affects segmentation and removes any accepted contacts that survive on believed
frames.

~~~mermaid
flowchart LR
    raw[Track and raw mask] --> run[run_video]
    run --> sticky[Build sticky evidence]
    run --> belief[Filter short exclusion runs]
    belief --> segment[Segment and assemble contacts]
    sticky -. serve-start inputs built pre-mask .-> segment
    segment --> score[Score and inpaint filters]
    score --> drop[Drop contacts on believed mask]
    drop --> parity[Fit alternation and next server]
    parity --> output[Filtered contacts and outputs]

    classDef source fill:#c8dde8,stroke:#5a7a9a,color:#1a1a1a
    classDef auxiliary fill:#e0e0e0,stroke:#888888,color:#1a1a1a
    classDef bridge fill:#e8d5a3,stroke:#8a6a30,color:#1a1a1a
    classDef result fill:#9070a0,stroke:#6a4070,color:#ffffff
    class raw source
    class run,belief,segment auxiliary
    class sticky,score,drop bridge
    class parity,output result
~~~

Figure 1. The dotted edge represents the existing serve-start evidence convention. The current
measurement does not enable this path. The convention neither prepends a contact nor exempts serve
evidence from replay policy.

The following current rules constrain the build:

- `filter_short_exclusion_runs` is the shared belief helper. A raw run is believed only
  when it reaches the current FPS-scaled minimum, which is 13 frames at 25 FPS and 15 at 30 FPS
- `run_video` builds `bootstrap_spans` on the unmasked track when it needs to
  construct a mask, then calls `segment_video` again after the track is frozen by the
  believed mask
- an all-True mask fails because no live frame can anchor the frozen position. The
  `no_replay` mode is an explicit replacement with replay/cutaway masking disabled, not
  a missing-data sentinel
- `court_invalid_is_excluded` is separate and is false in this run. A missing court row
  is not automatically believed replay
- `court_optional=True` stops after segmentation and cannot answer the full-chain
  question
- the default `SpanOpen.BACK_FILL` can move a span start earlier, but it does not add a
  contact
- the current pairing rule allows boundary grace and rejects believed replay deeper in a rally
  interior. Span expansion and contact injection still need separate measurement

### Measurement lane

The recorder leaves the baseline output unchanged. It compares current contacts with GT and then
writes the evidence needed for the three build decisions.

~~~mermaid
flowchart LR
    inputs[Current fixture inputs] --> run[Current run]
    run --> observe[Raw and accepted contacts]
    gt[GT first strokes] --> match[Existing greedy matcher]
    observe --> match
    match --> verdict[Serve-miss and evidence ledger]
    verdict --> counter[Future in-memory counterfactual]
    counter --> parity[Existing parity functions]
    verdict --> semantics[Span or contact semantics]
    semantics --> pairing[Existing pairing predicate]
    parity --> evidence[Build evidence]
    pairing --> evidence

    classDef source fill:#c8dde8,stroke:#5a7a9a,color:#1a1a1a
    classDef bridge fill:#e8d5a3,stroke:#8a6a30,color:#1a1a1a
    classDef result fill:#9070a0,stroke:#6a4070,color:#ffffff
    class inputs,gt source
    class run,observe,match,verdict,counter,parity,semantics,pairing bridge
    class evidence result
~~~

Figure 2. The current script stops at the evidence ledger. It neither runs the future
counterfactual nor changes the production result.

## 5. Measurement outputs

The script is [measure_serve_prepend_lookback.py](measure_serve_prepend_lookback.py). It reuses
the existing fixture and scoring seams instead of copying the retired producer.

The run pack is
[data/serve_prepend_lookback_20260731-040847/](data/serve_prepend_lookback_20260731-040847/),
with its summary at
[summary.json.gz](data/serve_prepend_lookback_20260731-040847/summary.json.gz).

Each fixture and mask mode produces:

- one CSV file with one row for every GT rally
- one typed NumPy evidence file with one row for every GT rally

The CSV rows record GT and span identity; raw and accepted contact distances; the serve-centred
track and inpaint evidence; the pre-contact lookback evidence; the raw and believed mask state;
the lookback anchor source; court presence; and raw pose availability. The script does not select
a prominent person. The pose fields record availability only because the current code has no
general person-ranking helper.

When measuring a future sticky serve-setup gate, keep unavailable evidence, zero sticky count,
invalid body scale and measured gate rejection as separate reasons. A zero body scale currently
fails closed and should not be collapsed into a generic candidate miss.

The `*.npy.xz` files are standard NumPy `*.npy` streams wrapped in Python's
native XZ/LZMA compression at preset 9. Reload them with `lzma.open` and
`np.load`; they are not joblib or pickle files. The script reloads every array immediately
after writing it. It also reloads each gzip CSV to verify the header and row count, and each gzip
JSON file to verify its parsed value.

The output pack does not duplicate the large external fixture arrays. Its summary records the
fixture root, current Git SHA, mask modes, FPS-scaled constants and output names.

## 6. Smallest next measurement and build

The current run answers whether evidence can be collected. It does not settle feature design. The
smallest next step is a recording-only extension of the existing measurement script:

1. Keep the committed-mask run as the decision baseline. Use `no_replay` only to
   quantify mask sensitivity.
2. Add a bounded raw candidate rule. Reuse raw contact impulses, current pose arrays, inpaint
   codes and the existing FPS-scaled lookback constant.
3. Score every candidate, including covered rallies, junk, replay stretches and candidates with no
   clean track support. Separate track-bearing, no-track and believed-mask cases.
4. Run an in-memory counterfactual by copying the accepted contact map and adding one candidate
   frame through the existing `contacts={rally_id: [frame, ...]}` injection seam. The seam
   accepts frame numbers and re-attributes injected contacts inside `run_video`. It does
   not carry a caller-supplied `None` half guess.
5. Compare span expansion with contact injection only after measuring downstream results. Record
   prepend provenance before any production integration.

This injection arm is a downstream counterfactual, not a full candidate-detection rerun. Supplying
`contacts` bypasses `segment_video`'s contact-finding path, so the
`serve_start` path is not exercised. An injected frame on the definitive exclusion mask
is removed before filtered contacts are produced. A span-expansion counterfactual can use the same
seam with an adjusted `spans` list; do not combine that with
`serve_start`.

If an unassigned parity arm is still needed after attributed injection, use the existing
`fit_alternation` and `next_server_half` functions directly. Do not add a new
production seam for that measurement.

The current evidence supports a raw-evidence experiment, not a person-identity system. The pose
arrays support availability measurements, but the code has no general prominent-person selector.
Add one only if the measurement shows that a second consumer needs it.

Keep the work inside the existing chain. Do not add a second replay-belief helper, a parallel
parity implementation, a new GT matcher, a new pairing predicate or a general experiment
framework. Those additions would add maintenance without answering the three decisions.

## 7. Decisions still owed

### Evidence path

Can raw shuttle evidence, together with pose and bbox availability, trigger a candidate? The
current sticky serve-setup gate requires analysed sticky evidence and is unavailable in
`court_optional`. The measurement must distinguish a clean track from a fabricated or
degraded track before this decision is made.

### Prepend semantics

Should an accepted candidate expand the rally span, or inject a contact while leaving the span
unchanged? Expansion affects span consumers and pairing. Injection preserves the span but can place
a contact outside its interval. Both variants require the existing parity and pairing checks.

### Replay trust

Should a candidate on a believed replay frame remain subject to the existing no-contact rule? The
first pass should preserve that rule. Any narrow exemption requires live-versus-replay evidence and
a false-positive result. Because 17 current target misses occur on believed-mask frames, this is a
real coverage trade-off rather than a theoretical edge case.

## 8. Code map

The links below are relative to this document.

- [run_video.py](../../../src/annotator/run_video.py): `run_video`,
  `build_serve_options`, mask filtering, processing order, contact injection and
  point-winner calls
- [rally_segmentation.py](../../../src/annotator/rally_segmentation.py):
  `segment_video`, `assemble_contacts`, `detect_contact_flags`,
  `ServeStartOptions`, `_sticky_serve_setup_before` and `apply_replay_mask`
- [types.py](../../../src/annotator/types.py): `ServeStartConfig`,
  `ContactCandidate`, `StickyResult` and `true_runs`
- [replay_mask.py](../../../src/annotator/replay_mask.py):
  `filter_short_exclusion_runs`
- [point_winner.py](../../../src/annotator/point_winner.py): attribution, alternation and
  next-server machinery
- [fixtures.py](../../../src/annotator/calibration/fixtures.py): `FIXTURES`,
  external-root loading and digest checks
- [scoring.py](../../../src/annotator/calibration/scoring.py):
  `load_gt_rallies`, `greedy_match` and boundary classes
- [gt_scoring.py](../../../src/annotator/calibration/gt_scoring.py):
  `canonical_tolerance`, `build_run_video_inputs`,
  `score_video` and the existing `--no-replay-mask` precedent
- [commentary_pairing.py](../../../src/scraper/commentary_pairing.py):
  `_believed_replay_in_rally_interior` and the span-end pairing window

## 9. Lean definition of done

The exploratory measurement/build-planning batch is complete when:

- the current-chain script runs on all three canonical fixtures
- every mode writes one row per GT rally and the compressed readers reload successfully
- the committed-mask baseline is the only baseline used for feature decisions
- every candidate in the next measurement receives a verdict, including replay and junk cases
- the owner records the evidence-path, span/contact and replay-trust decisions
- the counterfactual reuses existing parity, pairing and contact-injection seams
- only then, if the numbers justify it, a narrowly scoped feature is integrated behind one clear
  configuration boundary

This exploratory pass did not run the repository's full lint, type or test gates. It ran the
measurement script, Ruff on the script, Python bytecode compilation and gzip/native-LZMA reload
checks.
