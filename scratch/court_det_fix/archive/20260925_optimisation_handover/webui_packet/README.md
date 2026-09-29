> Historical record from the web-UI handover packet of 23 September 2026,
> filed on 25 September 2026. The
> [speed-up README](../../../court_detector_optimisation_handover/README.md)
> owns current status; the [archive map](../../README.md) records the original
> paths. Content is unchanged. The packet's `tools/`, `task_manifest.json`,
> `LOCAL_MODEL_PROMPT.md` and single-file copy are in git history at commit
> 92535b6e.

# Court detector optimisation handover

**Repository:** `ahalp90/badminton_cv_annotator`  
**Reviewed branch:** `fix/court-det`  
**Reviewed GitHub revision:** `2a00f77172a59374136e19246ede1d854aaaff01`  
**Static review date:** 2026-09-23

## Executive answer

Yes. The accepted detector contains several gross inefficiencies that can be
removed before changing its candidate sources, search caps, thresholds, ranking,
or accepted output semantics.

The largest high-confidence opportunities are:

1. **Apply an existing rejection gate before the expensive work it rejects.**
   `projective_seed.match_axis()` scores every axis interpretation before
   calculating `necessary_players()`, even though `player_compatible` is later
   required for eligibility. Calculate that mask first and do not score rejected
   hypotheses in the production path.
2. **Stop recomputing fixed linear algebra.** A direction-pair basis is inverted
   in `offsets()`, again while rectifying feet, and again in every 256-hypothesis
   `score_axes()` batch. Group lines, homogeneous endpoints, normalisation
   matrices, template-pair indexes, and the basis inverse should be prepared once
   at their natural lifetime.
3. **Do not invert every combined court.** `propose_role()` can combine as many as
   `512 × 512 = 262,144` axis pairs, after which
   `zone_net.player_fractions()` performs a generic batched 3×3 inverse for every
   court. Player coordinates can be obtained directly from the already rectified
   feet and the two axis scale/shift parameters.
4. **Do not project the same corners twice.** `canonicalise()` projects all
   transforms to decide the 180-degree relabelling; `geometry()` immediately
   projects the canonical transforms again. Return and reorder the first
   projection, including denominators, and validate it once.
5. **Batch the 256-candidate evidence stage.** `evaluate_pool()` calls stripe
   measurement and `gate_evidence()` one candidate at a time.
   `gate_evidence()` even rebuilds the same feet array for every candidate, while
   the underlying projection and score routines already accept arrays.
6. **Materialise records only after array-level retention.** The generator creates
   Python `Candidate` objects and large provenance dictionaries for every usable
   court, then keeps at most 256. Keep compact arrays through scoring and
   diversity selection; construct objects/JSON for survivors.
7. **Move compatible-view reuse before full search.** Current scene grouping is
   downstream of court inference, so it cannot avoid detector calls. A
   conservative validate-or-fallback path is required to reach the video-level
   target, but it should follow first-search acceleration rather than conceal it.

These are distinct from reducing G0/G1/template coverage or shrinking search
caps. The latter are semantic experiments and are deliberately outside the
first optimisation wave.

## Scope and confidence

This packet is a static source audit plus an execution plan. It uses the
repository's recorded timings, which show that fresh generation dominates and
that the later scoring/refit stage is itself tens of seconds per frame. It does
**not** pretend that a local profiler was run against the uncommitted development
corpus. The local model must verify the actual checkout, imported module paths,
working tree, hardware, and data inventory before modifying code.

The reviewed Git tree contains multiple frozen snapshots. Import precedence is
part of the program. Resolve `module.__file__` in the real local run before
editing; do not optimise a similarly named historical copy.

## Non-negotiable behaviour contract

Keep all of the following until a separately labelled semantic experiment proves
otherwise:

- G0, G1, and line-template candidate sources.
- SVD12 as the default direction-group screen, with full16 available.
- Existing candidate caps in the baseline arm.
- Existing gates, bounded net selection, source coverage, ranking, tie order,
  abstention behaviour, and selection-then-stripe-correction order.
- Float64 geometry unless a separate numerical-equivalence trial passes.
- Existing frozen development data and controls. **Do not request or create new
  annotations or a new held-out set.**
- Local uncommitted data, caches, and historical outputs. Never reset, clean, or
  overwrite them.

## Packet contents

| File | Purpose |
| --- | --- |
| `STATIC_AUDIT.md` | Call graph, complexity, concrete inefficiencies, and safety classification |
| `PATCH_QUEUE.md` | Ordered implementation backlog with invariants and stop conditions |
| `EXPERIMENT_PROTOCOL.md` | Reproducible local/HPC profiling and equivalence protocol |
| `SOURCE_MAP.md` | Actual source/import map and local-data checklist |
| `LOCAL_MODEL_PROMPT.md` | Ready-to-use prompt for the local frontier model |
| `task_manifest.json` | Machine-readable tasks, dependencies, and acceptance gates |
| `tools/bootstrap_session.sh` | Non-destructive session/bootstrap inventory |
| `tools/resolve_hotpath.py` | Print the modules actually imported by the W5 path |
| `tools/summarise_generation.py` | Quantify recorded matcher/combine work |
| `tools/compare_records.py` | Recursive JSON/JSON.GZ equivalence comparator |
| `court_detector_optimisation_handover.md` | Single-file concatenation of the packet |

## First local actions

From the repository root:

```bash
bash /path/to/packet/tools/bootstrap_session.sh
python /path/to/packet/tools/resolve_hotpath.py   --court-root scratch/court_det_fix
python /path/to/packet/tools/summarise_generation.py   scratch/court_det_fix/svd_search/run_20260923/generation/baseline/<case>.json.gz
```

Then capture one **fast broadcast** and one **slow GX** baseline with one worker,
one numerical thread, fresh output directories, `/usr/bin/time -v`, and stage
timings. Use the existing frozen cases; no annotation work is needed.

The first implementation should be **P1: lifetime-correct prepared geometry and
basis inverse reuse**, followed by **P2: player pruning before axis scoring**.
Both are bounded, measurable, and do not require changing search coverage.
