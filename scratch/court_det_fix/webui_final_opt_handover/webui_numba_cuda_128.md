Your four questions point to a fairly clear sequence: **consolidate the exact pipeline, test the 128 cap, try a bounded CPU-JIT pass, and then prototype GPU acceleration before undertaking graph search**. The GPU route preserves exhaustive coverage and has more measured headroom than another heuristic search layer.

## 1. Cutting proposed courts from 256 to 128

### There is encouraging evidence, but the location of the cap matters

The generator currently has two separate 256 limits:

1. **256 courts retained from each direction pair**: `keep_per_pair`.
2. **256 courts retained globally for each of G0 and G1**: `keep_global`.

Those are separate from the later merged W5 population, which includes G0, G1, and line-template candidates.

| Cap changed                                    | Accuracy evidence                                                                                                              | Likely speed effect                                                                      |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Per-pair 256 → 128                             | Strongly promising on the current sample: the ultimately accepted parent ranked no worse than **41st in its originating pair** | Small, because all combined courts are still generated and scored                        |
| Global G0/G1 256 → 128                         | Also promising: the accepted parent ranked no worse than **107th overall**                                                     | Meaningful, because fewer candidates enter pool evidence, W5 measurement, and refit      |
| Arbitrary cap of 128 after merging all sources | Not directly supported                                                                                                         | Potentially meaningful, but risks upsetting source coverage and line-template recoveries |

The committed analysis explicitly reports that the accepted parent ranked as low as 41st within its pair and 107th overall over the 20 court views. It also reports that a cap of 32 per pair already loses one accepted parent. Four accepted courts came from line templates rather than G0/G1.

So, on the measured data:

* **128 per pair would have retained every selected G0/G1 parent.**
* **128 globally would also have retained every selected G0/G1 parent.**
* There is only a 21-position margin between the observed worst global rank of 107 and a cap of 128.
* There has not been a committed end-to-end 128-cap run proving unchanged W5 selection, net choice, corrected geometry, and abstention.

The older six-view search-depth experiment offers indirect support: widening both caps from 256 to 512 cost 10% more wall time and did not show a comparable general quality gain, although it affected some already-failing examples. That suggests the useful population may have substantially saturated by 256, but it does not prove 128 is enough.

### Why the per-pair reduction is not a huge runtime win

Changing only `keep_per_pair` to 128 does **not** prevent:

* axis enumeration;
* axis scoring;
* Cartesian combination;
* canonicalisation;
* geometry checking;
* court support scoring.

The detector must still score all viable courts to know which 128 are best. It saves:

* some greedy-retention work;
* some candidate/provenance materialisation;
* a smaller pooled intermediate list;
* some memory.

The current complete run spends only 86 seconds in `retain` and around 175 seconds in pair-loop overhead, against 8,250 seconds in total. Consequently, I would expect **roughly 1–3%**, not a transformative reduction, from the per-pair cap alone.

### Why the global reduction could matter

Changing `keep_global` to 128 halves the maximum G0 and G1 populations passed downstream. W5 measures and attempts a refit for every merged parent, so this is where candidate count becomes expensive.

The generation search itself still runs, but the following shrink:

* legacy pool evidence;
* parent measurement;
* parent stripe evidence;
* refit attempts;
* child measurement;
* records and arrays in research mode.

Based on the present stage profile, my estimate is:

* **about 5–15% end-to-end saving** from a global cap of 128;
* an absolute optimistic ceiling below about 20%, because the 5,319-second search and the line-template population remain substantially intact.

That range is an inference, not a measured cap run.

### You can test much of this without repeating the expensive search

The best experiment is a replay from the saved 256-candidate populations:

| Arm       | Per-pair cap | Global G0/G1 cap |
| --------- | -----------: | ---------------: |
| Reference |          256 |              256 |
| A         |          128 |              256 |
| B         |          256 |              128 |
| C         |          128 |              128 |

For arm B, no proposal search should be necessary: truncate the saved retained G0/G1 populations to 128 and rerun W5, net choice, and stripe correction.

For arm A, the per-pair shortlist entries in the generation records should allow you to:

1. retain the first 128 from each pair;
2. reconstruct the global pool;
3. run the unchanged global diversity retention;
4. rerun W5.

The acceptance test should compare:

* selected parent and child;
* corrected court;
* full-court gate;
* bounded-net choice;
* abstention on all controls;
* candidate source;
* reference error where available;
* W5 candidate and refit counts;
* stage timings.

I would absolutely run this. I would not yet make 128 the default based solely on rank 107.

A sensible compromise, should plain 128 lose something, is **128 primary candidates plus a small diversity tail** from ranks 129–256—for example, 32 candidates selected for corner-space, source, direction-pair, or anchor diversity. That retains 160 rather than 256 while specifically protecting against the correct court being a structurally distinct but lower-scoring proposal.

---

## 2. Consolidating the committed detector into one end-to-end implementation first

**Yes. That is the sensible junction-point decision—but it should be an integration shell, not a monolithic rewrite.**

The accepted detector currently runs end to end only through `d17_timing/run_d17.py`. It depends on runtime function replacement, several snapshot module trees, repeated view preparation, population files written and read between stages, and a final refit that reads the case record and frame again. The wiring document exists precisely because no single normal detector entry point owns the whole chain.

Consolidation would provide three immediate benefits:

1. **Agentic context narrowing.** One explicit call graph, one set of active types, and one command become authoritative.
2. **Safer experimentation.** Caps, graph proposal, Numba, and CUDA become interchangeable backends or policies rather than edits spread across snapshots and monkey patches.
3. **A real exact speed-up.** The committed measurements attribute about 580 seconds, or 6.5% of the current baseline, to writing and rereading research populations, records, and arrays. A deployment-mode in-memory path can remove most of this while preserving the research path.

### The important distinction: orchestration integration, not algorithm integration

The first implementation should leave the mathematical functions intact and introduce something structurally like:

```python
@dataclass(frozen=True)
class PreparedView:
    source: SourceRecord
    frame: np.ndarray
    observations_g0: Observations
    observations_g1: Observations
    feet: np.ndarray
    size: tuple[int, int]
    directions: DirectionEstimate


@dataclass(frozen=True)
class CandidatePopulations:
    g0: tuple[CandidateRecord, ...]
    g1: tuple[CandidateRecord, ...]
    line_templates: tuple[CandidateRecord, ...]


class ProposalBackend(Protocol):
    def generate(
        self,
        view: PreparedView,
        settings: SearchSettings,
    ) -> CandidatePopulations: ...


class EvidenceBackend(Protocol):
    def measure_and_refit(
        self,
        view: PreparedView,
        populations: CandidatePopulations,
    ) -> W5Result: ...


class CourtDetector:
    def detect(
        self,
        source: SourceRecord,
        settings: DetectorSettings,
    ) -> DetectionResult: ...
```

The initial backends should simply call the existing committed code:

```text
NumpyReferenceProposalBackend
ExistingW5EvidenceBackend
ExistingNetChoice
ExistingPolarityCorrection
```

Later additions then fit cleanly:

```text
NumbaProposalBackend
CudaProposalBackend
GraphProposalPolicy
Cap128Policy
ReusePreviousCourtPolicy
```

### What the integration agent should not do

The first integration change should not:

* combine or regroup NumPy formulas;
* alter floating evaluation order;
* rename candidate identities;
* change stable tie order;
* merge G0 and G1 before their current boundary;
* replace source-specific scores with one common score;
* remove line templates;
* alter selection-then-polarity-correction order;
* remove full research records from research mode;
* “clean up” diagnostic behavior while moving it.

The first pass should be boring. Its success criterion is that the existing exact comparer reports unchanged saved results on all 28 views, excluding declared path and timing fields.

### Preserve stage boundaries intentionally

There should remain explicit seams after:

1. view preparation;
2. direction estimation;
3. G0 generation;
4. G1 generation;
5. line-template generation;
6. canonical population merge;
7. parent measurement;
8. child refit;
9. rank and net selection;
10. polarity correction.

That means you should not need to “separate it back out” later. The E2E implementation owns orchestration, while the individual stages remain replaceable.

### My sequencing recommendation

One cheap replay should precede the integration: test the 128 global cap against saved populations, because it can answer an important product decision without changing architecture.

Then:

1. create the exact E2E reference facade;
2. add the in-memory deployment mode;
3. put caps and proposal mechanisms behind settings/interfaces;
4. rerun the 128 experiment through the canonical entry point;
5. trial Numba;
6. trial CUDA;
7. undertake graph search only if the compute approaches still miss the budget.

That sequence reduces risk rather than merely rearranging code.

---

## 3. How much could Numba save?

I ran three synthetic microbenchmarks in this session, using Numba 0.65.1 and NumPy 2.3.5. The shapes were chosen to resemble the committed hot kernels; they were not run on Carmack and are not substitutes for real saved-input benchmarks.

### Results

| Kernel-shaped workload                                    |     NumPy | Numba, 1 thread |  Speed-up | Numba, 4 threads |  Speed-up |
| --------------------------------------------------------- | --------: | --------------: | --------: | ---------------: | --------: |
| Axis scoring: 32,768 hypotheses × 6 markings × 128 groups |  0.1875 s |        0.0471 s |  **4.0×** |         0.0211 s |  **8.9×** |
| Court support: 16,384 courts × 12 intervals × 64 samples  |  0.1793 s |        0.1382 s | **1.30×** |         0.0573 s | **3.13×** |
| Point-to-fragment distance: 768 × 650                     | 0.00426 s |       0.00372 s | **1.15×** |        0.00056 s |  **7.6×** |

Numerical comparison:

* axis matches and support counts were identical;
* axis-score maximum difference was \(2.8\times10^{-17}\);
* court-support maximum difference was \(3.4\times10^{-8}\);
* point-to-segment distances were identical.

### Why the result varies so much

`score_axes` is an excellent JIT target because Numba can:

* fuse the entire hypothesis loop;
* avoid allocating the large distance, nearest-group, response, and exclusivity temporaries;
* retain small arrays in registers/cache;
* emit a direct nested loop.

`continuous_support` is less favourable because much of its cost is:

* scattered distance-map reads;
* 768 samples per court;
* exponential evaluations;
* reductions already implemented efficiently inside NumPy.

The segment-distance primitive is likewise already one large vectorised NumPy expression. Numba helps more if the entire stripe-measurement routine is fused around it, rather than compiling that one primitive in isolation.

Numba’s strongest mode is fully compiled nopython code over NumPy arrays; it supports direct array access and many ufuncs, while `parallel=True` can parallelise supported loops on CPU. ([Numba Documentation][1])

### End-to-end estimate

The latest profile contains:

* `match_axis`: 1,384 seconds;
* `finite_scores`: 2,054 seconds;
* `stripe_measure`: 1,156 seconds;
* whole run: 8,250 seconds.

Not all of `match_axis` is the axis-scoring kernel, and not all stripe work will see the same benefit. A reasonable estimate is:

* **10–18% total saving** from a careful single-thread Numba pass over the two best kernels;
* **perhaps 20–25%** if stripe measurement and several geometry loops can be profitably fused too;
* not a credible 3× whole-pipeline improvement.

The multi-thread figures should not be counted as video-throughput savings when complete scenes already occupy all CPU cores. They are useful only when:

* fewer scenes than cores are active;
* one long scene remains at the tail;
* or single-scene latency is prioritised.

### How I would implement it

Start with just two functions:

```python
@numba.njit(
    explicit_signature,
    cache=True,
    fastmath=False,
    nogil=True,
)
def score_axes_kernel(...):
    ...
```

and a fused `continuous_support_kernel`.

Important constraints:

* use `fastmath=False`;
* use explicit dtypes and contiguous arrays;
* compile/warm the signatures at process startup;
* compare all retained IDs and support masks;
* CPU-recheck scores near a shortlist or hard-threshold frontier;
* keep the NumPy implementation as the reference backend.

The Carmack environment cited in the handover uses NumPy 2.5.3. Current Numba 0.67 officially supports NumPy 2.5, so this is no longer blocked by version compatibility. ([Numba Documentation][2])

Numba is the right first compiled experiment. A C++/pybind11 or Cython implementation may eventually edge it out, but it would have substantially more build and maintenance cost without first establishing that compiled loops move the end-to-end result.

---

## 4. Moving the detector to GPU

### This has the highest raw compute upside

Your instinct is directionally right. The major kernels contain enormous independent work:

* hundreds of thousands of axis hypotheses;
* tens or hundreds of thousands of combined courts;
* 12 × 64 support samples per court;
* many independent candidate measurements;
* multiple scenes available simultaneously.

That is a much better GPU workload than a single small image operation.

But “more than 100,000 threads” should be understood as **logical threads scheduled in waves**, not 100,000 cores executing simultaneously. Performance still depends on occupancy, memory access, warp divergence, reductions, and whether the arrays remain on the device. NVIDIA explicitly prioritises minimising host/device transfer, coalescing memory access, and avoiding divergent execution. ([NVIDIA Docs][3])

### What maps naturally to CUDA

#### Excellent targets

**Axis scoring**

One CUDA thread, or one small warp, can own one axis hypothesis:

```text
hypothesis
  × 5–6 canonical coordinates
  × up to 128 observed groups
```

The endpoint data are tiny and shared across all hypotheses, making them suitable for constant or shared memory.

**Combined-court support scoring**

One warp or block can own a court:

```text
court
  × 12 projected intervals
  × 64 distance-map samples
```

The template geometry is shared and the distance maps stay read-only on device.

**Fused court construction**

The following can become one kernel rather than several large intermediate arrays:

```text
axis-pair parameters
  -> homography
  -> canonical orientation
  -> corners
  -> geometry mask
  -> support score
```

That avoids writing and rereading every intermediate homography and corner array.

**Stripe/fragment distances**

These are large collections of independent point-to-segment calculations followed by reductions. They are also GPU-friendly, although the mapping is more complicated.

#### Less natural targets

* Python candidate dictionaries and provenance;
* stable greedy 2-pixel retention;
* source-identity merging;
* conditional rank logic;
* SciPy `least_squares`;
* small control-flow-heavy diagnostics;
* final net choice and polarity logic.

Those can remain on CPU initially.

### The memory problem is manageable if you do not materialise the sample tensor

For the maximum \(512\times512=262{,}144\) combined courts:

* float64 homographies: about **18 MiB**;
* float64 projected corners: about **16 MiB**;
* float64 scores: about **2 MiB**;
* two 960×540 float32 distance maps: about **4 MiB**.

Those are small for a modern GPU.

The dangerous intermediate is:

$$
262{,}144 \times 12 \times 64
$$

sample values:

* about **768 MiB** in float32;
* about **1.5 GiB** in float64.

The CUDA implementation should therefore reduce samples inside the kernel or process courts in chunks. It should never create the full court × interval × sample tensor.

### Simultaneous scenes make GPU more attractive

The right architecture is not one CUDA process for every scene. NVIDIA notes that multiple contexts on one GPU are time-sliced and incur context memory and switching overhead; one process using the primary context and streams is preferable. ([NVIDIA Docs][4])

So the GPU service should be:

```text
one GPU worker process
  ├── scene A stream / batch
  ├── scene B stream / batch
  ├── scene C stream / batch
  └── ...
```

It can batch hypotheses or courts from several scenes into the same launch. This directly uses your multi-scene deployment pattern rather than competing with it.

The data strategy should be:

1. copy each scene’s grouped observations and maps once;
2. keep them resident;
3. stream axis or court chunks;
4. retain scores and masks on device;
5. transfer only candidate indexes, compact geometry, and top/frontier candidates back.

Numba-CUDA documentation warns that passing host arrays directly can trigger implicit synchronous copies back to host; explicit device arrays and streams are necessary. ([NVIDIA GitHub][5])

CuPy’s memory and pinned-memory pools are useful for the same reason: they avoid repeated allocations and synchronisation. ([CuPy Documentation][6])

### How much could a GPU save end to end?

The present profile gives a useful Amdahl-law calculation.

| GPU-accelerated region                      | Current runtime share | If that region becomes 20× faster | Mean scene, from 295 s |
| ------------------------------------------- | --------------------: | --------------------------------: | ---------------------: |
| Axis matching + court support               |                   42% |                         1.66× E2E |                  177 s |
| Above + stripe measurement                  |                   56% |                         2.14× E2E |                  138 s |
| Above + geometry, combine, canonicalisation |                   67% |                         2.75× E2E |                  107 s |
| 75% of the pipeline                         |                   75% |                         3.48× E2E |                   85 s |

The runtime shares are calculated from the committed 8,250-second profile.

This shows two things:

1. **A minimal two-kernel GPU port will probably not meet the full target on its own.**
2. **A moderately broad GPU backend plausibly can meet the average 90-second target.**

The present worst scene is 666 seconds. Even accelerating 75% of it by 20× gives approximately 191 seconds under a uniform-share assumption. The slow GX scenes are more search-heavy than average, so they should benefit more than that simple model predicts, but a strict 90-second maximum will likely still require some combination of:

* 128-candidate downstream cap;
* deployment-mode I/O removal;
* camera/view reuse across scenes;
* coarse or graph proposal reduction;
* or moving more of W5 to the GPU.

These are modelled ceilings. I could not run a CUDA benchmark in this environment.

### Precision is the main architectural issue

The current reference pipeline uses float64 geometry. Consumer GPUs can have much weaker float64 throughput than float32, and GPU transcendental/reduction results will not generally reproduce CPU NumPy bits.

The safe hybrid is:

```text
GPU:
    enumerate and score all hypotheses/courts
    geometry and player masks
    approximate or nominally exact support
    compact top/frontier population

CPU reference:
    exact rescore of the retained/frontier population
    stable diversity retention
    W5 initially
    final ranking and correction
```

This is analogous to a non-neural region-proposal accelerator followed by the existing exact head.

It preserves exhaustive GPU coverage while containing numerical risk. It also gives you a clean fallback: when GPU ranks are too close at the cutoff, widen the CPU frontier.

### Technology choice

For a quick prototype:

* CuPy `RawKernel`, or
* Numba-CUDA/Numba-CUDA-MLIR.

Classic Numba-CUDA is now in maintenance mode, with new development directed toward Numba-CUDA-MLIR, so I would place the implementation behind a backend interface rather than coupling the detector directly to one compiler. ([NVIDIA GitHub][7])

The prototype should prove only two kernels first:

1. `score_axes_cuda`;
2. fused `combine_geometry_support_cuda`.

It should proceed only when, including transfer and synchronisation:

* each kernel is at least around 10× faster than its one-thread CPU reference on saved real inputs;
* the combined search is at least 2× faster;
* all selected candidates are recoverable;
* memory stays bounded with several simultaneous scenes;
* repeated runs are deterministic at the candidate-identity level;
* and the CPU frontier recheck restores the reference result.

## Recommended order from here

1. **Replay global cap 128 from saved populations now.**
2. **Build the thin exact E2E implementation around the committed stages.**
3. **Add an in-memory deployment mode and confirm its exact savings.**
4. **JIT `score_axes` with serial Numba as a low-risk CPU baseline.**
5. **Prototype the two CUDA kernels with one multi-scene GPU worker.**
6. **Use the resulting profile to decide whether graph proposal is still necessary.**

The key strategic point is that GPU acceleration can retain the current exhaustive search coverage. A graph search trades coverage for less work. Given the evidence currently available, I would test the exhaustive GPU route first.

[1]: https://numba.readthedocs.io/en/stable/reference/numpysupported.html?utm_source=chatgpt.com "Supported NumPy features — Numba 0+untagged.1117.g3190b91.dirty documentation"
[2]: https://numba.readthedocs.io/en/latest/release/0.67.0-notes.html?utm_source=chatgpt.com "Version 0.67.0 (11 August 2026) — Numba 0.69.0dev0+2461.g4b3e8fb9.dirty documentation"
[3]: https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-best-practices-guide/index.html?utm_source=chatgpt.com "CUDA C++ Best Practices Guide"
[4]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/?utm_source=chatgpt.com "CUDA Best Practices Guide — CUDA C++ Best Practices Guide 13.4 documentation"
[5]: https://nvidia.github.io/numba-cuda/user/kernels.html?utm_source=chatgpt.com "Writing CUDA Kernels — Numba CUDA"
[6]: https://docs.cupy.dev/en/stable/user_guide/memory.html?utm_source=chatgpt.com "Memory Management — CuPy 14.1.1 documentation"
[7]: https://nvidia.github.io/numba-cuda/?utm_source=chatgpt.com "Numba-CUDA — Numba CUDA"

