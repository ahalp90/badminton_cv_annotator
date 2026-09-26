# Assumptions and limits

1. **No direct Carmack rerun was available in this review.** The pack uses committed timing summaries, scripts, and source. Some full D17 inputs/results referenced by the handover are intentionally git-ignored or stored on Carmack.

2. **The 28-view set is small.** It is valuable for exact regression checking but cannot prove general precision/recall on arbitrary deployment video.

3. **Pair-shortlist identity is not the same as final accuracy.** The 16-sample/K=8,192 result is strong internal evidence, but the final detector still requires an end-to-end shadow comparison and wider hold-out video.

4. **Parallel latency is modelled, not measured.** The Amdahl tables assume 10% parallel overhead, 10 s fixed overhead, and selected parallel fractions. Memory bandwidth, process startup, shared-memory design, and NUMA placement can change the result materially.

5. **Synthetic benchmark hardware differs from Carmack.** This pack's synthetic scripts ran under Python 3.13.5, NumPy 2.3.5, Linux x86_64, with five visible logical CPUs. The repository's reference environment uses Python 3.12.13 and NumPy 2.5.3 on Carmack.

6. **Response-map reuse is only a local kernel result.** Its measured 1.15× gain at 16,384 synthetic courts does not include projection and clipping, so applying it to all of `finite_scores` would overstate the real benefit.

7. **The 90 s target needs a fixed hardware definition.** This pack treats the current one-thread-per-process measurements as the starting point and sizes 4–24 worker configurations. Cold-start and warm service targets should be specified separately.

8. **Fresh person detections can change precision.** The committed follow-up notes show a non-court control that gained a false court under fresh detections even though the seated-person rule was not the cause. Deployment acceptance must use the actual person/pose inputs, not only frozen feet.

9. **No new learned model is proposed.** The use of “RPN,” “ROI head,” and “backbone” describes engineering structure only.
