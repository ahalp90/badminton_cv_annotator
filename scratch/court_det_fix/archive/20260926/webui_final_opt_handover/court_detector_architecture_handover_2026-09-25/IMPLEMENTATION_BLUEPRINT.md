# Implementation blueprint

## 1. Target API

Replace the timing-script composition with one explicit detector entry point:

```python
@dataclass(frozen=True)
class DeploymentConfig:
    mode: Literal["exact", "cascade"] = "exact"
    workers: int = 1
    coarse_samples: int = 16
    proposal_k: int = 8192
    allow_previous_view: bool = True
    write_research_records: bool = False


def detect_court(
    frame: np.ndarray,
    source: SourceRecord,
    config: DeploymentConfig,
    previous: CourtState | None = None,
) -> DetectionResult:
    ...
```

`DetectionResult` should contain the corrected court/abstention plus counters needed to explain runtime and fallbacks. Research records should be an optional observer, not an intermediate transport.

## 2. Core data objects

```python
@dataclass(frozen=True)
class ViewFeatures:
    size: tuple[int, int]
    native_size: tuple[int, int]
    frame_working: np.ndarray
    grey_working: np.ndarray
    segments_g0: np.ndarray
    segments_g1: np.ndarray
    observations_g0: Observations
    observations_g1: Observations
    feet_working: np.ndarray
    directions: DirectionEstimate
    line_template_inputs: TemplateInputs


@dataclass(frozen=True)
class PairTask:
    source_name: Literal["G0", "G1"]
    pair_id: int
    pencil_ids: tuple[int, int]


@dataclass
class PairArrays:
    pair_id: int
    corners: np.ndarray
    homographies: np.ndarray
    axis_ids: np.ndarray
    axis_scores: np.ndarray
    coarse_scores: np.ndarray | None
    exact_scores: np.ndarray
    retained_positions: np.ndarray
    diagnostics: dict
```

Workers should exchange arrays and integer IDs. Do not create one Python `Candidate` plus provenance dict for every usable court.

## 3. Split `propose_role`

Current seam:

```text
run_given.propose_role
  basis_for
  match_axis x2
  combine all
  canonicalise all
  geometry all
  joint_player_fractions all
  finite_scores all usable
  Candidate objects all usable
```

Recommended seams:

```python
def propose_axes(...) -> AxisPairContext: ...
def combine_usable(context: AxisPairContext) -> UsableCourts: ...
def coarse_support(courts: UsableCourts, sample_count: int) -> np.ndarray: ...
def exact_support(courts: UsableCourts, positions: np.ndarray) -> np.ndarray: ...
def retain_pair_arrays(...) -> np.ndarray: ...
```

The exact mode passes every usable position to `exact_support`. The cascade mode passes a stable top-K subset.

## 4. Stable proposal selection

The committed measurement established that a subset produces the same greedy exact shortlist when every finally-retained court is included and the subset is passed in original proposal order.

```python
def proposal_positions(coarse: np.ndarray, k: int) -> np.ndarray:
    ranked = np.argsort(-coarse, kind="stable")[:k]
    return np.sort(ranked)  # Restore original proposal order before exact retain.
```

Do not feed coarse-rank order into exact retention; exact-score ties are broken by input order.

For the first implementation use K=8,192. Record:

- number of usable courts;
- K and whether it hit the full set;
- exact retained positions;
- weakest retained exact score;
- coarse ranks of retained courts in shadow mode;
- whether proposal and full shortlists match.

## 5. Optional exact response-map reuse

A local exact rewrite to test inside `finite_scores`:

```python
maps = detector._distance_maps((family0, family1), size)

# Only for pairs large enough to amortise the dense transform.
if len(homographies) >= response_map_threshold:
    response_maps = np.exp(
        -0.5 * np.square(maps / assignment.DISTANCE_SIGMA_PX)
    )
    scores = continuous_support_from_response_maps(...)
else:
    scores = continuous_support(...)
```

`continuous_support_from_response_maps` must keep projection, clipping, pixel conversion, visibility, and reduction order unchanged. Verify score bytes on saved pair inputs. The synthetic break-even was between roughly 1,024 and 4,096 courts, but the Carmack threshold must be measured.

## 6. Pair worker scheduler

```python
pair_tasks = build_pair_tasks(points, camera_bounds, all_16_groups=True)

with ProcessPoolExecutor(max_workers=config.workers) as pool:
    future_of = {
        pool.submit(run_pair_task, shared_handle, task, pair_config): task
        for task in pair_tasks
    }
    results_by_id = {}
    for future in as_completed(future_of):
        result = future.result()
        results_by_id[result.pair_id] = result

# Restore exact baseline order before global pooling/retention.
ordered = [results_by_id[task.pair_id] for task in pair_tasks if task.pair_id in results_by_id]
```

Required properties:

- one numerical thread in each worker;
- shared-memory or read-only memory-mapped feature arrays;
- no worker writes to shared dicts/files;
- deterministic pair IDs and merge order;
- exceptions include source/pair IDs;
- cancellation/fallback is coordinator-controlled.

A persistent process pool should serve multiple scenes to avoid Python/import startup per scene.

## 7. W5 scheduler

Refactor `process_case` into pure task functions:

```python
def measure_parent_task(shared: W5Shared, key: ParentKey) -> ParentResult: ...
def refit_parent_task(shared: W5Shared, parent: ParentPublic) -> RefitResult: ...
```

Coordinator flow:

1. canonicalise populations;
2. create unique homography keys in stable order;
3. schedule unique parent measurements;
4. expand duplicates from the unique result table;
5. build line maps once;
6. schedule refits for hard-valid parents;
7. restore parent order;
8. rank and choose serially.

This replaces the mutable measurement cache with explicit deduplication. It also makes deployment records optional.

## 8. Cascade fallback state machine

```text
START
  |
  v
coarse score all usable courts (16 samples)
  |
  v
exact top 4096 -> retain
  |
  +-- clearly unstable? --> exact top 8192 -> retain
  |                            |
  |                            +-- unstable? --> exact remainder/full
  |
  +-- exact certifier passes? --> accept
```

The first deployed cascade should skip the 4,096 stage and use 8,192 directly. Add the adaptive ladder only after call-level replay identifies a fallback rule with zero pair-shortlist misses on development and hold-out views.

Potential uncertainty signals:

- exact shortlist changed materially between budget levels;
- retained frontier has a small score margin;
- retained court lies close to the current K boundary in coarse rank;
- pair retains fewer than expected axis identities or hits caps;
- visibility is sparse/letterboxed;
- G0 and G1 disagree strongly;
- previous-view proposal conflicts with search;
- output is a non-court abstention boundary case.

## 9. Previous-view proposal

Store a minimal state:

```python
@dataclass(frozen=True)
class CourtState:
    homography_working: np.ndarray
    corrected_corners_native: np.ndarray
    camera_signature: np.ndarray
    source_frame_id: str
```

Validation/local search:

1. compare camera signature;
2. exact-measure previous homography on current view;
3. apply current stripe refit;
4. expand a small grid in `(sx, tx, sy, ty)` around the rectified map;
5. exact-score/refit the grid;
6. require current gates and temporal consistency;
7. otherwise run full search.

Never bypass current non-court gates merely because a prior scene had a court.

## 10. Deployment modes

| Mode | Purpose | Search semantics |
|---|---|---|
| `research` | reproduce existing records | current full files/diagnostics |
| `exact` | production oracle | full16, no proposal pruning, in memory, parallel |
| `cascade` | low-latency production | 16-sample proposal, exact head, fallback |
| `reuse` | repeated camera scenes | previous proposal first, then cascade/full fallback |

The `exact` mode is essential. It lets production canaries rerun the same scene and detect silent proposal drift without maintaining a separate code path.
