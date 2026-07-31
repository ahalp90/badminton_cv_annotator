# Serve-prepend lookback: current build orientation

Validated 2026-07-31 against commit `63f40938a62f6612ca9a63b61127d24442a80865`.
This note records the current evidence and build orientation for the deferred feature.

## TL;DR

The measurement can now be run against the current chain. The script ran the maintained
`run_video` chain on all 292 calibration rallies in `sset_01`, `sset_15` and `sset_21`, using the
canonical fixture inputs. It wrote a complete per-rally ledger and reloadable compressed artefacts
under the feature directory.

With the committed replay mask, 137 rallies have an unmatched GT serve and at least one later
GT stroke that matches an accepted contact. The fixture counts are 64, 39 and 34. Of those 137
cases, 114 contain a clean visible-track run within the one-second serve-centred window. Seventeen
missed serve frames fall on the believed replay mask. Only one missed case has an accepted contact
within the serve-matching tolerance; 15 have a raw candidate within that tolerance.

These results show why any prepend needs both a new raw-evidence decision and a counterfactual
check. On their own, they do not justify a production feature. In the `no_replay` run,
replay/cutaway masking is disabled (`raw_exclusion_mask` is `False` for every frame), but all other
processing and downstream filters remain active. Because this run changes rally spans and contact
counts substantially, it is a mask-sensitivity control, not a cleaner baseline.

The archived design record and this document serve different purposes. The archived record is at
[`docs/archive/serve_prepend_lookback.md`](../../archive/serve_prepend_lookback.md). This feature
directory contains the current measurement script, outputs and build orientation.

## Contents

- [TL;DR](#tldr)
- [1. Document locations](#1-document-locations)
- [2. Current run](#2-current-run)
- [3. Current measurements](#3-current-measurements)
- [4. Current processing order](#4-current-processing-order)
- [5. Measurement outputs](#5-measurement-outputs)
- [6. Build implications](#6-build-implications)
- [7. Decisions still owed](#7-decisions-still-owed)
- [8. Code map](#8-code-map)
- [9. Lean definition of done](#9-lean-definition-of-done)

## 1. Document locations

The two locations are related, but they are not parallel specifications.

| Location | Purpose | Use it for |
| --- | --- | --- |
| `docs/archive/serve_prepend_lookback.md` | Archived design record | Earlier problem framing, design constraints and deferred decisions |
| `docs/scraper_pipeline/serve_prepend_lookback/` | Current feature package | Measurement script, compressed run outputs and this current-code orientation |

The handover remains beside the non-standard-camera work because its original dependency was
the replay and cutaway lane. The feature directory is the better home for executable measurement
artefacts because it keeps the script, output pack and build notes together.

The feature directory [README](README.md) provides the short index. The raw external review
remains a scratch process artefact; its verified technical findings are incorporated here.

## 2. Current run

The run used `src/annotator/calibration/fixtures.py::FIXTURES` and
`src/annotator/calibration/gt_scoring.py::build_run_video_inputs`. Because the local shell did not
set `ANNOTATOR_FIXTURES_ROOT`, the command supplied the pinned fixture root explicitly:

```bash
ANNOTATOR_FIXTURES_ROOT=$PWD/local_scratch/autograder_architecture PYTHONPATH=src \
  ~/.venvs/badminton-cicd/bin/python -u \
  docs/scraper_pipeline/serve_prepend_lookback/measure_serve_prepend_lookback.py \
  --out docs/scraper_pipeline/serve_prepend_lookback/data/serve_prepend_lookback_20260731-040847 \
  --mask-mode both
```

The source arrays are the maintained, pinned calibration substrate available in the
repository's local scratch area. They are not a newly extracted 2026-07-30 broadcast dataset. The
processing chain and matching code are current at the SHA above.

The three fixtures are:

| Fixture | Video id | FPS | GT rallies |
| --- | ---: | ---: | ---: |
| `sset_01` | 1 | 25 | 113 |
| `sset_15` | 15 | 25 | 104 |
| `sset_21` | 21 | 30 | 75 |

The script runs two modes:

- `committed`: the fixture's supplied raw replay mask, duration-filtered by the current chain
- `no_replay`: the existing calibration precedent with replay/cutaway masking disabled; its
  `raw_exclusion_mask` is `False` for every frame

The second mode is a sensitivity run. It changes segmentation before contact filtering, so
its result cannot be interpreted as the isolated benefit of allowing contacts on replay frames.
All other processing and downstream filters remain active.

## 3. Current measurements

The target case is a GT rally whose first stroke remains unmatched after the existing
span-overlap candidate collection, while one or more later GT strokes match accepted contacts. In
this case, a prepend could repair the first contact without including rallies that have no usable
evidence at all.

The ledger collects accepted contacts from every span that overlaps the GT rally extent,
including spans in the split and missed categories. It is therefore broader than the canonical
covered-only matching metric in `score_video`. The category is retained so the next measurement
can distinguish ordinary covered rallies from boundary and whole-span failures.

A clean run is a sequence of consecutive frames with a visible shuttle track and inpaint code
`NO_FLAG`. The serve-centred window extends one second on either side of the GT serve. The minimum
clean-run length uses the existing five-frame base-30 tolerance scaled by FPS: four frames at
25 FPS and five at 30 FPS. The lookback run measures the same evidence before the first assigned
accepted contact, using the current `serve_start_lookback_frames` value.

### Committed-mask baseline

| Fixture | Target serve misses | Clean serve window | Clean pre-contact lookback | Raw candidate within tolerance | Accepted candidate within tolerance | Serve on believed mask | Accepted contacts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sset_01` | 64 | 53 | 28 | 7 | 1 | 0 | 1,749 |
| `sset_15` | 39 | 36 | 12 | 6 | 0 | 0 | 1,233 |
| `sset_21` | 34 | 25 | 17 | 2 | 0 | 17 | 1,090 |
| **Pooled** | **137** | **114** | **57** | **15** | **1** | **17** | **4,072** |

The current chain therefore preserves substantial visible-track evidence around the serve in
the target misses, but almost none of that evidence becomes an accepted contact at the serve
frame. This is evidence worth investigating, not a trigger recommendation. A lookback still needs
a candidate rule and a false-positive audit.

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

The no-replay run merges long stretches and changes the number of detected spans, so it is not
a feature ablation. It does show that replay masking is load-bearing for this decision: the
proposed feature must evaluate both the current trust rule and any candidate exemption against the
same current chain.

## 4. Current processing order

In normal mode, the chain builds sticky evidence before applying the definitive replay mask.
The believed mask then affects segmentation and removes any accepted contacts that survive on
believed frames.

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
    classDef boundary fill:#e8f0e8,stroke:#5a7a5a,color:#1a1a1a
    classDef bridge fill:#e8d5a3,stroke:#8a6a30,color:#1a1a1a
    classDef result fill:#9070a0,stroke:#6a4070,color:#ffffff
    class raw source
    class run,belief,segment boundary
    class sticky,score,drop bridge
    class parity,output result
~~~

Figure 1. The dotted edge represents the existing serve-start evidence convention. The current
measurement does not enable this path. The convention neither prepends a contact nor exempts serve
evidence from replay policy.

The following current rules constrain the build:

- `filter_short_exclusion_runs` is the shared belief helper. A raw run is believed only when it
  reaches the current FPS-scaled minimum, which is 13 frames at 25 FPS and 15 at 30 FPS
- `run_video` builds `bootstrap_spans` on the unmasked track when it needs to construct a mask,
  then calls `segment_video` again after the track is frozen by the believed mask
- all-True masks fail because no live frame can anchor the frozen position. The no-replay mode is
  a valid explicit replacement with replay/cutaway masking disabled, not a missing-data sentinel
- `court_invalid_is_excluded` is separate and is false in this run. A missing court row is not
  automatically believed replay
- `court_optional=True` stops after segmentation and cannot answer the full-chain question
- the default `SpanOpen.BACK_FILL` can move a span start earlier, but it does not add a contact
- the current pairing rule allows boundary grace and rejects believed replay deeper in a rally
  interior. Span expansion and contact injection still need separate measurement

### Measurement lane

The recorder leaves the baseline output unchanged. It compares current contacts with GT and
then writes the evidence needed for the three build decisions.

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

The script is [`measure_serve_prepend_lookback.py`](measure_serve_prepend_lookback.py). It
reuses the existing fixture and scoring seams instead of copying the retired producer.

The run pack is
[`data/serve_prepend_lookback_20260731-040847/`](data/serve_prepend_lookback_20260731-040847/),
with its summary at
[`summary.json.gz`](data/serve_prepend_lookback_20260731-040847/summary.json.gz).

Each fixture and mask mode produces:

- one `*_rallies.csv.gz` file with one row for every GT rally
- one `*_evidence.npy.xz` file containing a compact, typed NumPy evidence table

The CSV rows record GT and span identity; raw and accepted contact distances; serve-centred
track and inpaint evidence; pre-contact lookback evidence; raw and believed mask state; court
presence; and raw pose availability. The script does not select a prominent person. The pose
fields record availability only because the current code has no general person-ranking helper.

When measuring a future sticky serve-setup gate, keep unavailable evidence, zero sticky count,
invalid body scale and measured gate rejection as separate reasons. A zero body scale currently
fails closed and should not be collapsed into a generic candidate miss.

The `.npy.xz` files are standard NumPy `.npy` streams wrapped in Python's native `lzma` XZ
compression at preset 9. Reload them with `lzma.open` and `np.load`. The script reloads each array
immediately after writing it. It also reloads each gzip CSV to verify the header and row count, and
each gzip JSON file to verify the parsed value. The JSON and CSV files use the standard `gzip`
readers.

The output pack does not duplicate the large external fixture arrays. Its summary records the
fixture root, current Git SHA, mask modes, FPS-scaled constants and output names.

## 6. Build implications

The measurement establishes that the current data can be collected. It does not settle the
feature design.

The smallest next step is:

1. Keep the committed-mask run as the decision baseline. Use the no-replay run only to quantify
   mask sensitivity.
2. Add a recording-only candidate rule in the measurement lane. Reuse raw contact impulses,
   current pose arrays, current inpaint codes and the existing FPS-scaled lookback constant.
3. Score every candidate. Include already-covered rallies, junk, replay stretches and candidates
   with no clean track support. Separate track-bearing, no-track and believed-mask cases.
4. Run an in-memory counterfactual by copying the accepted contact map and adding one candidate
   frame through the existing `contacts={rally_id: [frame, ...]}` injection seam. That seam accepts
   frames, then re-attributes them inside `run_video`; it does not carry a caller-supplied `None`
   half guess. If an unassigned arm is still needed, compute it directly with the existing
   `fit_alternation` and `next_server_half` functions rather than adding a new production seam.
5. Choose span expansion or contact injection only after comparing the downstream result. Record
   prepend provenance in the measurement output before any production integration.

The current evidence supports a raw-evidence experiment, not a person-identity system. The
existing pose arrays support availability measurements, but the code has no general
prominent-person selector. Add one only if the measurement shows that a second consumer needs it.

Keep the implementation within the existing chain. Do not add a second replay-belief helper, a
parallel parity implementation, a new GT matcher, a new pairing predicate or a general experiment
framework for this feature. None of those additions would answer the three current decisions, and
each would add maintenance.

## 7. Decisions still owed

### Evidence path

Can raw shuttle evidence, together with pose and bbox availability, trigger a candidate? The
current sticky serve-setup gate requires analysed sticky evidence and is unavailable in
`court_optional`. Before this decision can be made, the measurement must distinguish a clean track
from a fabricated or degraded track.

### Prepend semantics

Should an accepted candidate expand the rally span, or inject a contact while leaving the span
unchanged? Expansion affects span consumers and pairing. Injection preserves the span but can place
a contact outside its interval. Both variants require the existing parity and pairing checks.

### Replay trust

Should a candidate on a believed replay frame remain subject to the existing no-contact rule?
The first pass should preserve that rule. Any narrow exemption requires live-versus-replay evidence
and a false-positive result. Because 17 current target misses occur on believed-mask frames, this is
a real coverage trade-off rather than a theoretical edge case.

## 8. Code map

- `src/annotator/run_video.py`: `run_video`, `build_serve_options`, mask filtering, scoring order,
  contact injection and point-winner calls
- `src/annotator/rally_segmentation.py`: `segment_video`, `assemble_contacts`,
  `detect_contact_flags`, `detect_contacts`, `ServeStartOptions`, `_sticky_serve_setup_before`
  and `apply_replay_mask`
- `src/annotator/types.py`: `ServeStartConfig`, `ContactCandidate`, `StickyResult` and `true_runs`
- `src/annotator/replay_mask.py`: `filter_short_exclusion_runs`
- `src/annotator/point_winner.py`: attribution, alternation and next-server machinery
- `src/annotator/calibration/fixtures.py`: `FIXTURES`, external-root loading and digest checks
- `src/annotator/calibration/scoring.py`: `load_gt_rallies`, `greedy_match` and boundary classes
- `src/annotator/calibration/gt_scoring.py`: `canonical_tolerance`,
  `build_run_video_inputs`, `score_video` and the existing `--no-replay-mask` precedent
- `src/scraper/stage11_pairing.py`: `_believed_replay_in_rally_interior` and the span-end pairing
  window

## 9. Lean definition of done

The measurement batch is complete when all of the following are true:

- the current-chain script runs on all three canonical fixtures
- every mode writes one row per GT rally and the compressed readers reload successfully
- the committed-mask baseline is the only baseline used for feature decisions
- every candidate in the next measurement receives a verdict, including replay and junk cases
- the owner records the evidence-path, span/contact and replay-trust decisions
- the counterfactual reuses existing parity, pairing and contact-injection seams
- only then, if the numbers justify it, a narrowly scoped feature is integrated behind one clear
  configuration boundary

This exploratory pass did not run the repository's full lint, type or test gates. It ran the
measurement script, Ruff on that script, Python bytecode compilation, and gzip/native-lzma reload
checks.
